"""
Multi-agent trade reviewer.

Uses 3 reviewer agents with different stances (Pro, Neutral, Opposing)
plus a judge agent for final decision synthesis.
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import httpx
import pandas as pd
import structlog

from src.ai.market_research import fetch_market_research, format_research_for_prompt, set_cache_ttl
from src.ai.sentiment import fetch_fear_greed_index, get_trade_summary, FearGreedResult, TradeSummary
from src.ai.web_search import WEB_SEARCH_TOOL, handle_tool_calls, get_tools_for_model
from src.strategy.signal_scorer import SignalResult

logger = structlog.get_logger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class AgentReview:
    """Single agent's review with assigned stance."""

    stance: str  # "pro", "neutral", "opposing"
    model: str
    approved: bool
    confidence: float
    summary: str  # Short 1-sentence summary for notifications
    reasoning: str  # Longer reasoning for judge and logs
    sentiment: str


@dataclass
class MultiAgentReviewResult:
    """Result of multi-agent review process."""

    reviews: list[AgentReview]
    judge_decision: bool
    judge_confidence: float
    judge_reasoning: str
    judge_recommendation: str  # "wait", "accumulate", "reduce"
    final_veto_action: Optional[str]
    trade_context: dict = field(default_factory=dict)


# Legacy compatibility - keep ReviewResult for backwards compatibility
@dataclass
class ReviewResult:
    """Legacy result format for compatibility."""

    approved: bool
    confidence: float
    reasoning: str
    sentiment: str
    veto_action: Optional[str]
    trade_context: dict = field(default_factory=dict)


# System prompts for each stance
SYSTEM_PROMPT_PRO = """You are a Bitcoin trading analyst with a PRO stance on this trade.

Your role is to argue IN FAVOR of the proposed trade. Find and emphasize reasons why this trade SHOULD proceed.
Be persuasive but honest - acknowledge risks briefly while focusing on the opportunity.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Positive scores = bullish signals, negative = bearish
- Trade executes when |score| >= threshold

Trading timeframe context:
- Daytrading (short candles): Focus on momentum, quick reversals, tight risk management
- Swing trading (medium candles): Balance momentum with trend confirmation
- Position trading (long candles): Emphasize macro trends, fundamentals, wider stops acceptable

Focus on:
- Favorable indicator readings
- Positive market conditions
- Historical patterns supporting this trade
- Risk/reward opportunity appropriate for the timeframe

Respond with JSON only:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) with your key argument",
  "reasoning": "2-3 sentences with detailed analysis arguing FOR the trade"
}"""

SYSTEM_PROMPT_NEUTRAL = """You are a Bitcoin trading analyst with a NEUTRAL stance.

Your role is to provide balanced, unbiased analysis. Weigh both the opportunities and risks equally.
Present facts objectively without advocating for or against the trade.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Positive scores = bullish signals, negative = bearish
- Trade executes when |score| >= threshold

Trading timeframe context:
- Daytrading (short candles): Focus on momentum, quick reversals, tight risk management
- Swing trading (medium candles): Balance momentum with trend confirmation
- Position trading (long candles): Emphasize macro trends, fundamentals, wider stops acceptable

Analyze:
- Both positive and negative signals
- Current market sentiment vs technical signals
- Risk factors AND opportunities
- Overall signal quality relative to the trading timeframe

Respond with JSON only:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) with your key observation",
  "reasoning": "2-3 sentences with detailed balanced analysis"
}"""

SYSTEM_PROMPT_OPPOSING = """You are a Bitcoin trading analyst with an OPPOSING stance on this trade.

Your role is to argue AGAINST the proposed trade. Find and emphasize reasons why this trade should NOT proceed.
Be critical but honest - acknowledge potential upside briefly while focusing on risks.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Positive scores = bullish signals, negative = bearish
- Trade executes when |score| >= threshold

Trading timeframe context:
- Daytrading (short candles): Focus on momentum, quick reversals, tight risk management
- Swing trading (medium candles): Balance momentum with trend confirmation
- Position trading (long candles): Emphasize macro trends, fundamentals, wider stops acceptable

Focus on:
- Warning signs in indicators
- Market conditions that could hurt this trade
- Historical patterns suggesting caution
- Downside risks appropriate for the timeframe

Respond with JSON only:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) with your key concern",
  "reasoning": "2-3 sentences with detailed analysis arguing AGAINST the trade"
}"""

SYSTEM_PROMPT_JUDGE = """You are the final decision maker for a Bitcoin trading system.

You will receive three analyses from different agents:
1. A PRO stance (arguing for the trade)
2. A NEUTRAL stance (balanced view)
3. An OPPOSING stance (arguing against)

Your job is to:
1. Consider the strength of each argument
2. Weigh the confidence levels
3. Look for consensus or strong disagreement
4. Consider the trading timeframe (daytrading vs swing vs position trading)
5. Make the final decision and recommendation

Trading timeframe context:
- Daytrading: Requires quick decisions, momentum-focused, tight risk management
- Swing trading: Balance short-term signals with trend confirmation
- Position trading: Prioritize macro trends, can tolerate more volatility

Decision guidelines:
- If all three agree, follow the consensus
- If PRO and NEUTRAL approve with high confidence, likely approve
- If OPPOSING has very strong arguments (>0.8 confidence), consider rejecting
- For daytrading: Be more decisive, momentum matters
- For position trading: Be more patient, wait for stronger confirmation
- When in doubt, err on the side of caution

Respond with JSON only:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "recommendation": "wait"/"accumulate"/"reduce",
  "reasoning": "2-3 sentences explaining your decision synthesis"
}

Recommendation meanings:
- "wait": Hold position, wait for clearer signals
- "accumulate": Good opportunity to buy/add to position
- "reduce": Consider reducing exposure or taking profits"""

# Hold-specific system prompts for "interesting hold" reviews
# Key semantic: approved=true means "I AGREE with the bot's hold decision"
SYSTEM_PROMPT_PRO_HOLD = """You are a Bitcoin trading analyst reviewing a HOLD decision.

The bot decided NOT to trade because the signal score is below threshold.
Your role (PRO stance): Argue why the HOLD decision is CORRECT.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Trade executes when |score| >= threshold
- Current score is BELOW threshold, so bot is holding

Trading timeframe context:
- Daytrading (short candles): Quick setups, momentum-focused
- Swing trading (medium candles): Balance momentum with trend
- Position trading (long candles): Macro trends, patience

Focus on:
- Why the weak signal correctly indicates no action
- Why patience is appropriate here
- Risks avoided by not trading
- Market conditions favoring caution

Respond with JSON only:
{
  "approved": true if you AGREE with the hold (bot should stay passive), false if you think bot should act instead,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) arguing FOR the hold",
  "reasoning": "2-3 sentences explaining why holding is the right decision"
}"""

SYSTEM_PROMPT_NEUTRAL_HOLD = """You are a Bitcoin trading analyst reviewing a HOLD decision.

The bot decided NOT to trade because the signal score is below threshold.
Your role (NEUTRAL stance): Provide balanced analysis on whether the hold is correct.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Trade executes when |score| >= threshold
- Current score is BELOW threshold, so bot is holding

Trading timeframe context:
- Daytrading (short candles): Quick setups, momentum-focused
- Swing trading (medium candles): Balance momentum with trend
- Position trading (long candles): Macro trends, patience

Analyze:
- Both reasons to hold AND reasons to act
- Signal quality relative to threshold
- Current market conditions
- Whether the threshold seems appropriate

Respond with JSON only:
{
  "approved": true if you AGREE with the hold (bot should stay passive), false if you think bot should act instead,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) with your balanced view",
  "reasoning": "2-3 sentences with objective analysis of the hold decision"
}"""

SYSTEM_PROMPT_OPPOSING_HOLD = """You are a Bitcoin trading analyst reviewing a HOLD decision.

The bot decided NOT to trade because the signal score is below threshold.
Your role (OPPOSING stance): Argue why the HOLD decision is WRONG.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Trade executes when |score| >= threshold
- Current score is BELOW threshold, so bot is holding

Trading timeframe context:
- Daytrading (short candles): Quick setups, momentum-focused
- Swing trading (medium candles): Balance momentum with trend
- Position trading (long candles): Macro trends, patience

Focus on:
- Hidden opportunities being missed
- Why the threshold might be too conservative
- Reasons to act despite the weak signal
- What the bot might be overlooking

Respond with JSON only:
{
  "approved": true if you AGREE with the hold (bot should stay passive), false if you think bot should act instead,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "summary": "One short sentence (max 15 words) arguing AGAINST the hold",
  "reasoning": "2-3 sentences explaining why the bot SHOULD act despite the weak signal"
}"""

SYSTEM_PROMPT_JUDGE_HOLD = """You are the final decision maker for a Bitcoin trading system reviewing a HOLD decision.

The bot decided NOT to trade because the signal was below threshold.
You will receive three analyses from different agents:
1. A PRO stance (arguing the hold is correct)
2. A NEUTRAL stance (balanced view)
3. An OPPOSING stance (arguing the hold is wrong)

Your job is to:
1. Consider the strength of each argument
2. Weigh the confidence levels
3. Decide if the bot's HOLD decision is correct

Decision:
- approved=true → CONFIRM the hold is correct, bot should stay passive
- approved=false → OVERRIDE the hold, bot should consider acting

Guidelines:
- If all three agree the hold is correct, confirm it
- If PRO and NEUTRAL approve with high confidence, confirm the hold
- If OPPOSING has very strong arguments (>0.8 confidence), consider overriding
- When in doubt, the hold is usually correct (weak signals = wait)

Respond with JSON only:
{
  "approved": true to CONFIRM hold (stay passive), false to OVERRIDE hold (consider acting),
  "confidence": 0.0-1.0,
  "recommendation": "wait"/"accumulate"/"reduce",
  "reasoning": "2-3 sentences explaining your decision"
}

Recommendation meanings for holds:
- "wait": Correct to hold, wait for stronger signals
- "accumulate": Opportunity to buy even with weak signal
- "reduce": Consider reducing position even with weak signal"""

SYSTEM_PROMPT_HOLD = """You are a Bitcoin trading analyst. Explain why the trading bot is holding instead of trading.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Trade executes when |score| >= threshold
- Current score's magnitude is below threshold, so the bot holds

Trading timeframe context:
- Daytrading: Short candles, focus on momentum and quick setups
- Swing trading: Medium candles, balance momentum with trend
- Position trading: Long candles, focus on macro trends

Respond with JSON only:
{
  "sentiment": "bullish"/"bearish"/"neutral",
  "reasoning": "1-2 sentence explanation of the current market signals, considering the timeframe",
  "confidence": 0.0-1.0
}"""

# Market Analysis prompts (different stances for market outlook)
SYSTEM_PROMPT_MARKET_BULLISH = """You are a Bitcoin market analyst with a BULLISH outlook.

Your role is to identify and emphasize positive signals, upside potential, and reasons for optimism.
Be persuasive but honest - acknowledge risks briefly while focusing on opportunities.

You may have access to a web_search tool. Use it if you need additional market news or data.

Indicators explained:
- RSI: 0-100 scale. <30 = oversold (bullish), >70 = overbought (bearish)
- MACD histogram: Positive = bullish momentum, Negative = bearish momentum
- Bollinger %B: 0-1 scale. <0.2 = near lower band (oversold), >0.8 = near upper band (overbought)
- EMA gap: Price vs slow EMA. Negative = price below EMA (bearish trend)
- Fear & Greed: 0-100. <25 = Extreme Fear (contrarian bullish), >75 = Extreme Greed

Focus on:
- Oversold conditions as buying opportunities
- Positive momentum signals
- Contrarian opportunities in fear
- Technical support levels
- Recent positive news or developments

Respond with JSON only:
{
  "outlook": "bullish"/"bearish"/"neutral",
  "confidence": 0.0-1.0,
  "summary": "One short sentence (max 15 words) with your key bullish argument",
  "reasoning": "2-3 sentences with detailed bullish analysis"
}"""

SYSTEM_PROMPT_MARKET_NEUTRAL = """You are a Bitcoin market analyst with a NEUTRAL stance.

Your role is to provide balanced, unbiased analysis. Weigh both bullish and bearish factors equally.
Present facts objectively without advocating for any direction.

You may have access to a web_search tool. Use it if you need additional market news or data.

Indicators explained:
- RSI: 0-100 scale. <30 = oversold (bullish), >70 = overbought (bearish)
- MACD histogram: Positive = bullish momentum, Negative = bearish momentum
- Bollinger %B: 0-1 scale. <0.2 = near lower band (oversold), >0.8 = near upper band (overbought)
- EMA gap: Price vs slow EMA. Negative = price below EMA (bearish trend)
- Fear & Greed: 0-100. <25 = Extreme Fear, >75 = Extreme Greed

Analyze:
- Both positive and negative signals equally
- Current market sentiment vs technical signals
- Risk factors AND opportunities
- Overall market conditions
- Recent news impact on both sides

Respond with JSON only:
{
  "outlook": "bullish"/"bearish"/"neutral",
  "confidence": 0.0-1.0,
  "summary": "One short sentence (max 15 words) with your key observation",
  "reasoning": "2-3 sentences with detailed balanced analysis"
}"""

SYSTEM_PROMPT_MARKET_BEARISH = """You are a Bitcoin market analyst with a BEARISH outlook.

Your role is to identify and emphasize warning signs, downside risks, and reasons for caution.
Be critical but honest - acknowledge potential upside briefly while focusing on risks.

You may have access to a web_search tool. Use it if you need additional market news or data.

Indicators explained:
- RSI: 0-100 scale. <30 = oversold (bullish), >70 = overbought (bearish)
- MACD histogram: Positive = bullish momentum, Negative = bearish momentum
- Bollinger %B: 0-1 scale. <0.2 = near lower band (oversold), >0.8 = near upper band (overbought)
- EMA gap: Price vs slow EMA. Negative = price below EMA (bearish trend)
- Fear & Greed: 0-100. <25 = Extreme Fear, >75 = Extreme Greed (contrarian bearish)

Focus on:
- Overbought conditions as selling signals
- Negative momentum and trend weakness
- Resistance levels and potential reversals
- Risk factors in current conditions
- Recent negative news or developments

Respond with JSON only:
{
  "outlook": "bullish"/"bearish"/"neutral",
  "confidence": 0.0-1.0,
  "summary": "One short sentence (max 15 words) with your key concern",
  "reasoning": "2-3 sentences with detailed bearish analysis"
}"""

SYSTEM_PROMPT_MARKET_JUDGE = """You are the final decision maker synthesizing market analysis from three analysts.

You will receive three analyses from different perspectives:
1. A BULLISH stance (focusing on upside)
2. A NEUTRAL stance (balanced view)
3. A BEARISH stance (focusing on risks)

Your job is to:
1. Consider the strength of each argument
2. Weigh the confidence levels
3. Look for consensus or strong disagreement
4. Synthesize into a final market outlook

Decision guidelines:
- If all three agree on direction, follow the consensus
- If BULLISH and NEUTRAL are positive with high confidence, lean bullish
- If BEARISH has very strong arguments (>0.8 confidence), lean cautious
- When signals conflict, provide nuanced analysis

Respond with JSON only:
{
  "outlook": "bullish"/"bearish"/"neutral",
  "confidence": 0.0-1.0,
  "recommendation": "wait"/"accumulate"/"reduce",
  "reasoning": "2-3 sentences synthesizing the three perspectives"
}

Recommendation meanings:
- "wait": Hold current position, wait for clearer signals
- "accumulate": Good opportunity to buy/add to position
- "reduce": Consider reducing exposure or taking profits"""


class TradeReviewer:
    """
    Multi-agent trade reviewer.

    Uses 3 reviewers with different stances (randomly assigned)
    plus a judge for final decision synthesis.
    """

    def __init__(
        self,
        api_key: str,
        db,
        reviewer_models: list[str],
        judge_model: str,
        veto_reduce_threshold: float = 0.65,
        veto_skip_threshold: float = 0.80,
        position_reduction: float = 0.5,
        interesting_hold_margin: int = 15,
        review_all: bool = False,
        market_research_enabled: bool = True,
        ai_web_search_enabled: bool = True,
        market_research_cache_minutes: int = 15,
        candle_interval: str = "ONE_HOUR",
        signal_threshold: int = 60,
        max_tokens: int = 4000,
        api_timeout: int = 120,
    ):
        """
        Initialize multi-agent trade reviewer.

        Args:
            api_key: OpenRouter API key
            db: Database instance for trade history
            reviewer_models: List of 3 models for reviewers
            judge_model: Model for the judge
            veto_reduce_threshold: Judge confidence to reduce position (lower tier)
            veto_skip_threshold: Judge confidence to skip trade entirely (higher tier)
            position_reduction: Position size multiplier for "reduce" action
            interesting_hold_margin: Score margin from threshold for interesting holds
            review_all: Review ALL decisions (for debugging/testing)
            market_research_enabled: Fetch online research for market analysis
            ai_web_search_enabled: Allow AI models to search web during analysis
            market_research_cache_minutes: Cache duration for research data
            candle_interval: Candle timeframe for determining trading style
            max_tokens: Maximum tokens for AI API responses
            api_timeout: Timeout in seconds for API calls
        """
        self.api_key = api_key
        self.db = db
        self.reviewer_models = reviewer_models
        self.judge_model = judge_model
        self.veto_reduce_threshold = veto_reduce_threshold
        self.veto_skip_threshold = veto_skip_threshold
        self.position_reduction = position_reduction
        self.interesting_hold_margin = interesting_hold_margin
        self.review_all = review_all
        self.market_research_enabled = market_research_enabled
        self.ai_web_search_enabled = ai_web_search_enabled
        self.candle_interval = candle_interval
        self.signal_threshold = signal_threshold
        self.max_tokens = max_tokens
        self.api_timeout = api_timeout

        # Set cache TTL for market research
        set_cache_ttl(market_research_cache_minutes)

        # Circuit breaker
        self._consecutive_failures = 0
        self._max_failures = 5
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_reset_hours = 24

    def _determine_veto_action(self, approved: bool, confidence: float) -> Optional[str]:
        """
        Determine veto action based on judge decision and confidence tiers.

        Tiered veto system (v1.31.0):
        - approved=True: No veto action (trade proceeds)
        - approved=False, confidence < veto_reduce_threshold: No veto (info only)
        - approved=False, confidence >= veto_reduce_threshold: "reduce" position
        - approved=False, confidence >= veto_skip_threshold: "skip" trade entirely

        Args:
            approved: Whether the judge approved the trade
            confidence: Judge's confidence level (0.0 to 1.0)

        Returns:
            Veto action string ("skip", "reduce") or None if no veto
        """
        if approved:
            return None

        if confidence >= self.veto_skip_threshold:
            return "skip"  # High confidence disapproval: cancel trade
        elif confidence >= self.veto_reduce_threshold:
            return "reduce"  # Medium confidence disapproval: reduce position

        # Below reduce threshold: proceed with trade (info only)
        return None

    def _get_trading_style(self) -> tuple[str, str]:
        """
        Determine trading style based on candle interval.

        Returns:
            (style_name, style_description) tuple
        """
        short_intervals = ("ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE")
        medium_intervals = ("THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR")

        if self.candle_interval in short_intervals:
            return ("daytrading", "short-term daytrading (minutes to hours)")
        elif self.candle_interval in medium_intervals:
            return ("swing", "swing trading (hours to days)")
        else:
            return ("position", "position trading (days to weeks)")

    def _get_trading_mechanics_context(self, trading_style: str) -> str:
        """
        Get trading mechanics description based on trading style.

        This clarifies spot trading mechanics to prevent AI models from
        using futures/margin trading terminology (e.g., "long entry").

        Args:
            trading_style: One of "daytrading", "swing", or "position"

        Returns:
            Trading mechanics description for system prompts
        """
        base_mechanics = """Trading mechanics (spot trading, no leverage):
- BUY = spend fiat to acquire BTC (add to position)
- SELL = convert BTC holdings to fiat (reduce position)
- Position ranges from 0% (all fiat) to 100% (all BTC)"""

        style_focus = {
            "daytrading": """
Trading style: DAYTRADING
- Focus on quick entries/exits within hours
- Momentum and short-term reversals are key
- Tight risk management, small position adjustments""",
            "swing": """
Trading style: SWING TRADING
- Hold positions for hours to days
- Balance momentum with trend confirmation
- Moderate position sizing, allow for volatility""",
            "position": """
Trading style: POSITION TRADING (long-term)
- Hold positions for days to weeks
- Focus on macro trends and fundamentals
- Larger positions, wider tolerance for drawdowns""",
        }

        return base_mechanics + style_focus.get(trading_style, style_focus["swing"])

    def should_review(
        self, signal_result: SignalResult, threshold: int
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if this signal warrants AI review.

        Returns:
            (should_review, review_type) where review_type is "trade", "interesting_hold", or "hold"
        """
        if signal_result.action in ("buy", "sell"):
            return (True, "trade")

        distance_to_buy = abs(signal_result.score - threshold)
        distance_to_sell = abs(signal_result.score - (-threshold))

        if min(distance_to_buy, distance_to_sell) <= self.interesting_hold_margin:
            return (True, "interesting_hold")

        if self.review_all:
            return (True, "hold")

        return (False, None)

    def _check_circuit_breaker_reset(self) -> None:
        """Check if circuit breaker should auto-reset after timeout."""
        if self._last_failure_time and self._consecutive_failures >= self._max_failures:
            hours_since = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds() / 3600
            if hours_since >= self._circuit_breaker_reset_hours:
                self._consecutive_failures = 0
                self._last_failure_time = None
                logger.info("circuit_breaker_auto_reset", hours_since_failure=f"{hours_since:.1f}")

    async def review_trade(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        trading_pair: str,
        review_type: str = "trade",
        position_percent: float = 0.0,
        candles: Optional[pd.DataFrame] = None,
        quote_balance: Optional[Decimal] = None,
        base_balance: Optional[Decimal] = None,
        estimated_size: Optional[dict] = None,
        hide_balance_info: bool = False,
    ) -> MultiAgentReviewResult:
        """
        Multi-agent review process.

        Args:
            signal_result: Signal from SignalScorer
            current_price: Current market price
            trading_pair: Trading pair (e.g., "BTC-USD")
            review_type: "trade" or "interesting_hold"
            position_percent: Current position as percentage of portfolio
            candles: Optional DataFrame with OHLCV data for price action context
            quote_balance: Available quote currency balance for buying
            base_balance: Current base currency holdings for selling
            estimated_size: Estimated trade size dict with side, size_base, size_quote
            hide_balance_info: If True, hide balance info from reviewers (for Cramer Mode)

        Returns:
            MultiAgentReviewResult with all reviews and judge decision
        """
        self._check_circuit_breaker_reset()

        # Gather context
        fear_greed = await fetch_fear_greed_index()
        trade_summary = get_trade_summary(self.db, days=7)

        # When hide_balance_info is True (Cramer Mode), don't show balance to reviewers
        # This ensures judge evaluates signal quality, not execution feasibility
        ctx_position_percent = None if hide_balance_info else position_percent
        ctx_quote_balance = None if hide_balance_info else quote_balance
        ctx_base_balance = None if hide_balance_info else base_balance
        ctx_estimated_size = None if hide_balance_info else estimated_size

        context = self._build_context(
            signal_result, current_price, trading_pair, fear_greed, trade_summary, review_type,
            ctx_position_percent, candles, ctx_quote_balance, ctx_base_balance, ctx_estimated_size
        )

        # Multi-agent review for all decisions
        try:
            # Randomly assign stances to models
            stances = ["pro", "neutral", "opposing"]
            random.shuffle(stances)
            assignments = list(zip(self.reviewer_models, stances))

            logger.info(
                "multi_agent_review_starting",
                assignments=[(m.split("/")[-1], s) for m, s in assignments],
            )

            # Run all 3 reviewers in parallel
            reviews = await asyncio.gather(*[
                self._run_reviewer(model, stance, context)
                for model, stance in assignments
            ], return_exceptions=True)

            # Filter out exceptions
            valid_reviews = []
            for i, review in enumerate(reviews):
                if isinstance(review, Exception):
                    logger.error(
                        "reviewer_failed",
                        model=assignments[i][0],
                        stance=assignments[i][1],
                        error=str(review),
                    )
                    # Create fallback review
                    valid_reviews.append(AgentReview(
                        stance=assignments[i][1],
                        model=assignments[i][0],
                        approved=True,
                        confidence=0.0,
                        summary="Review unavailable",
                        reasoning=f"Review failed: {str(review)[:200]}",
                        sentiment="neutral",
                    ))
                else:
                    valid_reviews.append(review)

            # Run judge with all reviews
            judge_result = await self._run_judge(valid_reviews, context)

            # Determine veto action based on confidence tiers
            veto_action = self._determine_veto_action(
                approved=judge_result["approved"],
                confidence=judge_result["confidence"],
            )

            self._consecutive_failures = 0

            return MultiAgentReviewResult(
                reviews=valid_reviews,
                judge_decision=judge_result["approved"],
                judge_confidence=judge_result["confidence"],
                judge_reasoning=judge_result["reasoning"],
                judge_recommendation=judge_result["recommendation"],
                final_veto_action=veto_action,
                trade_context=context,
            )

        except Exception as e:
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)
            logger.error(
                "multi_agent_review_failed",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )

            # Circuit breaker: force position reduction on repeated failures
            if self._consecutive_failures >= self._max_failures:
                return MultiAgentReviewResult(
                    reviews=[],
                    judge_decision=True,
                    judge_confidence=0.0,
                    judge_reasoning=f"Review unavailable ({self._consecutive_failures} failures) - reducing position",
                    judge_recommendation="reduce",
                    final_veto_action="reduce",
                    trade_context=context,
                )
            else:
                return MultiAgentReviewResult(
                    reviews=[],
                    judge_decision=True,
                    judge_confidence=0.0,
                    judge_reasoning=f"Review failed: {str(e)[:200]}",
                    judge_recommendation="wait",
                    final_veto_action=None,
                    trade_context=context,
                )

    async def _review_hold(self, context: dict, review_type: str) -> MultiAgentReviewResult:
        """Simplified review for hold decisions (legacy, not currently used)."""
        try:
            # Use first reviewer model for holds
            prompt = self._build_hold_prompt(context)
            response = await self._call_api(self.reviewer_models[0], SYSTEM_PROMPT_HOLD, prompt)
            data = self._extract_json(response)

            reasoning = data.get("reasoning", "No analysis")
            return MultiAgentReviewResult(
                reviews=[AgentReview(
                    stance="neutral",
                    model=self.reviewer_models[0],
                    approved=True,
                    confidence=float(data.get("confidence", 0.5)),
                    summary=reasoning.split('.')[0] + '.' if '.' in reasoning else reasoning[:80],
                    reasoning=reasoning,
                    sentiment=data.get("sentiment", "neutral"),
                )],
                judge_decision=True,
                judge_confidence=float(data.get("confidence", 0.5)),
                judge_reasoning=data.get("reasoning", "No analysis"),
                judge_recommendation="wait",
                final_veto_action=None,
                trade_context=context,
            )
        except Exception as e:
            logger.error("hold_review_failed", error=str(e))
            return MultiAgentReviewResult(
                reviews=[],
                judge_decision=True,
                judge_confidence=0.0,
                judge_reasoning=f"Hold review failed: {str(e)[:100]}",
                judge_recommendation="wait",
                final_veto_action=None,
                trade_context=context,
            )

    async def _run_reviewer(
        self, model: str, stance: str, context: dict
    ) -> AgentReview:
        """Run single reviewer with assigned stance."""
        # Select prompts based on review type
        review_type = context.get("review_type", "trade")
        if review_type == "interesting_hold":
            system_prompts = {
                "pro": SYSTEM_PROMPT_PRO_HOLD,
                "neutral": SYSTEM_PROMPT_NEUTRAL_HOLD,
                "opposing": SYSTEM_PROMPT_OPPOSING_HOLD,
            }
        else:
            system_prompts = {
                "pro": SYSTEM_PROMPT_PRO,
                "neutral": SYSTEM_PROMPT_NEUTRAL,
                "opposing": SYSTEM_PROMPT_OPPOSING,
            }

        # Inject trading mechanics context based on trading style
        trading_style = context.get("trading_style", "swing")
        trading_mechanics = self._get_trading_mechanics_context(trading_style)
        system_prompt = f"{system_prompts[stance]}\n\n{trading_mechanics}"

        prompt = self._build_reviewer_prompt(context)
        response = await self._call_api(model, system_prompt, prompt)
        data = self._extract_json(response)

        # Parse response
        approved_raw = data.get("approved", True)
        if isinstance(approved_raw, str):
            approved = approved_raw.lower() == "true"
        else:
            approved = bool(approved_raw)

        summary = data.get("summary", "")
        reasoning = data.get("reasoning", "No reasoning provided")

        # Fallback: if no summary, use first sentence of reasoning
        if not summary and reasoning:
            summary = reasoning.split('.')[0] + '.' if '.' in reasoning else reasoning[:80]

        return AgentReview(
            stance=stance,
            model=model,
            approved=approved,
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            summary=summary,
            reasoning=reasoning,
            sentiment=data.get("sentiment", "neutral"),
        )

    async def _run_judge(
        self, reviews: list[AgentReview], context: dict
    ) -> dict:
        """Run judge to synthesize reviews and make final decision."""
        # Select judge prompt based on review type
        review_type = context.get("review_type", "trade")
        base_prompt = SYSTEM_PROMPT_JUDGE_HOLD if review_type == "interesting_hold" else SYSTEM_PROMPT_JUDGE

        # Inject trading mechanics context based on trading style
        trading_style = context.get("trading_style", "swing")
        trading_mechanics = self._get_trading_mechanics_context(trading_style)
        judge_prompt = f"{base_prompt}\n\n{trading_mechanics}"

        prompt = self._build_judge_prompt(reviews, context)
        response = await self._call_api(self.judge_model, judge_prompt, prompt)
        data = self._extract_json(response)

        approved_raw = data.get("approved", True)
        if isinstance(approved_raw, str):
            approved = approved_raw.lower() == "true"
        else:
            approved = bool(approved_raw)

        # Validate recommendation
        recommendation = data.get("recommendation", "wait")
        if recommendation not in ("wait", "accumulate", "reduce"):
            recommendation = "wait"

        return {
            "approved": approved,
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", "No reasoning provided"),
            "recommendation": recommendation,
        }

    def _get_timeframe_guidance(self, candle_interval: str) -> str:
        """
        Get timeframe-specific guidance for AI reviewers.

        This helps AI reviewers understand what's normal behavior for the given
        timeframe and adjust their analysis accordingly.

        Args:
            candle_interval: The candle interval string (e.g., "FIFTEEN_MINUTE")

        Returns:
            Guidance string appropriate for the timeframe
        """
        guidance = {
            "ONE_MINUTE": (
                "Timeframe Note: Ultra-short term (1-min candles). Momentum signals dominate, "
                "noise is high. Prioritize quick moves over trend confirmation. False signals are common."
            ),
            "FIVE_MINUTE": (
                "Timeframe Note: Very short term (5-min candles). Quick reversals expected. "
                "Prioritize momentum over trend. Watch for whipsaws."
            ),
            "FIFTEEN_MINUTE": (
                "Timeframe Note: Day trading (15-min candles). Balance momentum with short-term trend. "
                "Signals should confirm within 2-4 candles. 1-3% moves are normal."
            ),
            "THIRTY_MINUTE": (
                "Timeframe Note: Intraday swing (30-min candles). More patience required. "
                "Watch for trend changes at key levels. 2-4% daily moves are common."
            ),
            "ONE_HOUR": (
                "Timeframe Note: Swing trading (1-hour candles). Balance trend following with "
                "mean reversion. Confirmation may take 4-8 candles. Allow for intraday volatility."
            ),
            "TWO_HOUR": (
                "Timeframe Note: Swing trading (2-hour candles). Trend confirmation is important. "
                "Don't overreact to single candle movements. Multi-day holds are normal."
            ),
            "SIX_HOUR": (
                "Timeframe Note: Position trading (6-hour candles). Macro trend takes priority. "
                "Ignore short-term noise. Focus on major support/resistance levels."
            ),
            "ONE_DAY": (
                "Timeframe Note: Position trading (daily candles). Focus on fundamentals and "
                "macro trends. Ignore intraday noise entirely. Weeks-long holds are expected."
            ),
        }
        return guidance.get(candle_interval, "")

    def _summarize_recent_candles(self, candles: pd.DataFrame, n: int = 5) -> str:
        """
        Summarize recent price action for AI context.

        Provides a concise summary of the last N candles including:
        - Overall direction (bullish/bearish/mixed)
        - Number of green vs red candles
        - Total price range

        Args:
            candles: DataFrame with OHLCV data
            n: Number of recent candles to summarize (default: 5)

        Returns:
            Human-readable summary string
        """
        if candles is None or len(candles) < n:
            return ""

        recent = candles.tail(n)

        # Check for NaN values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        if recent[critical_cols].isna().any().any():
            logger.warning(
                "price_action_summary_nan_detected",
                nan_counts=recent[critical_cols].isna().sum().to_dict()
            )
            return ""

        # Count bullish (green) vs bearish (red) candles
        up_candles = (recent['close'] > recent['open']).sum()
        down_candles = n - up_candles

        # Calculate price range
        high = recent['high'].max()
        low = recent['low'].min()
        price_range = high - low
        mid_price = (high + low) / 2
        # Defensive check: mid_price > 0 and high >= low (handles corrupted data)
        range_pct = (price_range / mid_price * 100) if mid_price > 0 and high >= low else 0

        # Determine overall direction
        if up_candles >= 4:
            direction = "strongly bullish"
        elif up_candles >= 3:
            direction = "bullish"
        elif down_candles >= 4:
            direction = "strongly bearish"
        elif down_candles >= 3:
            direction = "bearish"
        else:
            direction = "mixed/choppy"

        # Check for trend momentum (closes trending)
        close_trend = recent['close'].diff().dropna()
        consecutive_up = (close_trend > 0).sum()
        consecutive_down = (close_trend < 0).sum()

        momentum = ""
        if consecutive_up >= 4:
            momentum = " with strong upward momentum"
        elif consecutive_down >= 4:
            momentum = " with strong downward momentum"

        return (
            f"Recent Price Action: {direction}{momentum} "
            f"({up_candles}/{n} green candles, {range_pct:.1f}% range)"
        )

    def _build_context(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        trading_pair: str,
        fear_greed: FearGreedResult,
        trade_summary: TradeSummary,
        review_type: str,
        position_percent: Optional[float] = 0.0,
        candles: Optional[pd.DataFrame] = None,
        quote_balance: Optional[Decimal] = None,
        base_balance: Optional[Decimal] = None,
        estimated_size: Optional[dict] = None,
    ) -> dict:
        """Build context dict for prompts and Telegram.

        When position_percent/quote_balance/base_balance are None (Cramer Mode),
        balance info is hidden from reviewers so they evaluate signal quality only.
        """
        trading_style, trading_style_desc = self._get_trading_style()

        # Generate price action summary if candles available
        price_action = self._summarize_recent_candles(candles) if candles is not None else ""

        # Calculate portfolio value if balances provided
        portfolio_value = None
        if quote_balance is not None and base_balance is not None:
            portfolio_value = float(quote_balance + base_balance * current_price)

        return {
            "review_type": review_type,
            "action": signal_result.action,
            "score": signal_result.score,
            "confidence": signal_result.confidence,
            "price": float(current_price),
            "trading_pair": trading_pair,
            "fear_greed": fear_greed.value,
            "fear_greed_class": fear_greed.classification,
            "win_rate": trade_summary.win_rate * 100,
            "total_trades": trade_summary.total_trades,
            "net_pnl": float(trade_summary.net_pnl),
            "breakdown": signal_result.breakdown,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candle_interval": self.candle_interval,
            "trading_style": trading_style,
            "trading_style_desc": trading_style_desc,
            "position_percent": position_percent,
            "threshold": self.signal_threshold,
            "price_action": price_action,
            "quote_balance": float(quote_balance) if quote_balance is not None else None,
            "base_balance": float(base_balance) if base_balance is not None else None,
            "portfolio_value": portfolio_value,
            "estimated_size": estimated_size,
        }

    def _build_reviewer_prompt(self, context: dict) -> str:
        """Build prompt for reviewer agents."""
        review_type = context.get("review_type", "trade")

        # Get timeframe-specific guidance
        timeframe_guidance = self._get_timeframe_guidance(context['candle_interval'])

        # Get price action summary if available
        price_action = context.get('price_action', '')
        price_action_line = f"\n{price_action}" if price_action else ""

        # Check for whale activity in breakdown
        breakdown = context['breakdown']
        whale_activity_line = ""
        if breakdown.get("_whale_activity"):
            whale_direction = breakdown.get("_whale_direction", "unknown").upper()
            whale_activity_line = f"\n⚠️ WHALE ACTIVITY ({whale_direction}): Volume {breakdown.get('_volume_ratio', 0)}x average"

        # HTF bias context - always show for full AI context
        # Use explicit None checks for null safety (avoid masking empty strings).
        # HTF values are expected to be: "bullish", "bearish", "neutral", or None.
        # Empty strings should NOT occur in production (would indicate a bug in get_trend()).
        # If empty strings appear, they are preserved for debugging (not masked as "unknown").
        # None values indicate missing/unavailable data and are replaced with "unknown".
        htf_trend = breakdown.get("_htf_trend") if breakdown.get("_htf_trend") is not None else "unknown"
        daily = breakdown.get("_htf_daily") if breakdown.get("_htf_daily") is not None else "unknown"
        four_h = breakdown.get("_htf_4h") if breakdown.get("_htf_4h") is not None else "unknown"

        htf_line = f"\n📊 HIGHER TIMEFRAME BIAS: {htf_trend.upper()} (Daily: {daily.upper()}, 4H: {four_h.upper()})"

        # Build portfolio section (hidden when balance info is None for Cramer Mode comparison)
        position_pct = context.get('position_percent')
        if position_pct is not None:
            portfolio_section = f"""
Portfolio:
- Current Position: {position_pct:.1f}% of portfolio
"""
        else:
            portfolio_section = ""  # Hide portfolio info for pure signal evaluation

        # Build common context sections
        common_context = f"""Price: ¤{context['price']:,.2f}
Signal Score: {context['score']:+d} (threshold: ±{context['threshold']})
Signal Breakdown: {json.dumps(context['breakdown'])}{whale_activity_line}{htf_line}

Trading Style: {context['trading_style_desc']}
Timeframe: {context['candle_interval']} candles
{timeframe_guidance}{price_action_line}
{portfolio_section}
Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Recent Performance (7 days):
- Win Rate: {context['win_rate']:.0f}%
- Net P&L: ¤{context['net_pnl']:+,.2f}
- Total Trades: {context['total_trades']}"""

        if review_type == "interesting_hold":
            return f"""Review this HOLD decision:

The bot decided NOT to trade because the signal score ({context['score']:+d}) is below the threshold (±{context['threshold']}).
The score is close to threshold, making this an "interesting hold" worth reviewing.

{common_context}

Should the bot stay passive (hold), or should it act despite the weak signal?
Analyze from your assigned perspective."""
        else:
            return f"""Review this trade:

Action: {context['action'].upper()}
{common_context}

Analyze this trade from your assigned perspective, considering the trading timeframe."""

    def _build_judge_prompt(self, reviews: list[AgentReview], context: dict) -> str:
        """Build prompt for judge with all reviews."""
        review_type = context.get("review_type", "trade")
        reviews_text = []

        for review in reviews:
            stance_label = {"pro": "PRO", "neutral": "NEUTRAL", "opposing": "OPPOSING"}[review.stance]
            model_short = review.model.split("/")[-1]

            # For holds: approved=true means "agree with hold", approved=false means "should act"
            if review_type == "interesting_hold":
                verdict = "HOLD CORRECT" if review.approved else "SHOULD ACT"
            else:
                verdict = "APPROVE" if review.approved else "REJECT"

            reviews_text.append(
                f"[{stance_label}] ({model_short}) - {verdict} ({review.confidence:.0%} confidence)\n"
                f"  Reasoning: {review.reasoning}"
            )

        # Check for whale activity
        breakdown = context.get('breakdown', {})
        whale_line = ""
        if breakdown.get("_whale_activity"):
            whale_direction = breakdown.get("_whale_direction", "unknown").upper()
            whale_line = f"\n⚠️ WHALE ACTIVITY ({whale_direction}): Volume {breakdown.get('_volume_ratio', 0)}x average"

        if review_type == "interesting_hold":
            return f"""Hold Decision Review at ¤{context['price']:,.2f}
Signal Score: {context['score']:+d} (below threshold ±{context['threshold']})
Trading Style: {context['trading_style_desc']}{whale_line}

The bot decided NOT to trade. Should this hold be confirmed or overridden?

Agent Reviews:
{chr(10).join(reviews_text)}

Based on these perspectives, decide: Is the hold correct (stay passive) or should the bot act?"""
        else:
            return f"""Trade Decision: {context['action'].upper()} at ¤{context['price']:,.2f}
Signal Score: {context['score']:+d}
Trading Style: {context['trading_style_desc']}{whale_line}

Agent Reviews:
{chr(10).join(reviews_text)}

Based on these three perspectives and the trading timeframe, make the final decision."""

    def _build_hold_prompt(self, context: dict) -> str:
        """Build prompt for hold analysis."""
        return f"""Analyze this hold decision:

Signal Score: {context['score']:+d} (need ≥+{context['threshold']} for buy or ≤-{context['threshold']} for sell)
Price: ¤{context['price']:,.2f}
Signal Breakdown: {json.dumps(context['breakdown'])}

Trading Style: {context['trading_style_desc']}
Timeframe: {context['candle_interval']} candles

Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Explain what the indicators are showing, considering the trading timeframe."""

    # ========== Market Analysis (Multi-Agent) ==========

    async def analyze_market(
        self,
        indicators: dict,
        current_price: Decimal,
        fear_greed: int,
        fear_greed_class: str,
        regime: str,
        volatility: str,
        price_change_1h: Optional[float] = None,
        price_change_24h: Optional[float] = None,
    ) -> MultiAgentReviewResult:
        """
        Multi-agent market analysis with online research.

        Uses 3 reviewers with bullish/neutral/bearish stances
        plus a judge for final synthesis.

        Args:
            indicators: Dict with rsi, macd_histogram, bb values, ema values
            current_price: Current BTC price
            fear_greed: Fear & Greed index value
            fear_greed_class: Fear & Greed classification
            regime: Current market regime
            volatility: Volatility level (low/normal/high/extreme)
            price_change_1h: Optional 1-hour price change %
            price_change_24h: Optional 24-hour price change %

        Returns:
            MultiAgentReviewResult with all reviews and judge decision
        """
        self._check_circuit_breaker_reset()

        # Fetch market research if enabled
        research_text = ""
        if self.market_research_enabled:
            try:
                research = await fetch_market_research()
                research_text = format_research_for_prompt(research)
                if research.errors:
                    logger.warning("market_research_partial", errors=research.errors)
                logger.info(
                    "market_research_fetched",
                    news_count=len(research.news),
                    news_titles=[n.title[:50] for n in research.news[:3]],
                    has_onchain=research.onchain is not None,
                    onchain_summary={
                        "hashrate_eh": research.onchain.hashrate_eh,
                        "mempool_mb": research.onchain.mempool_size_mb,
                        "avg_fee": research.onchain.avg_fee_sat_vb,
                    } if research.onchain else None,
                )
            except Exception as e:
                logger.error("market_research_failed", error=str(e))
                research_text = "(Research data unavailable)"

        # Build market context
        context = {
            "review_type": "market_analysis",
            "price": float(current_price),
            "fear_greed": fear_greed,
            "fear_greed_class": fear_greed_class,
            "regime": regime,
            "volatility": volatility,
            "price_change_1h": price_change_1h,
            "price_change_24h": price_change_24h,
            "indicators": indicators,
            "research": research_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Randomly assign stances to models
            stances = ["bullish", "neutral", "bearish"]
            random.shuffle(stances)
            assignments = list(zip(self.reviewer_models, stances))

            logger.info(
                "multi_agent_market_analysis_starting",
                assignments=[(m.split("/")[-1], s) for m, s in assignments],
            )

            # Run all 3 reviewers in parallel
            reviews = await asyncio.gather(*[
                self._run_market_reviewer(model, stance, context)
                for model, stance in assignments
            ], return_exceptions=True)

            # Filter out exceptions
            valid_reviews = []
            for i, review in enumerate(reviews):
                if isinstance(review, Exception):
                    logger.error(
                        "market_reviewer_failed",
                        model=assignments[i][0],
                        stance=assignments[i][1],
                        error=str(review),
                    )
                    # Create fallback review
                    valid_reviews.append(AgentReview(
                        stance=assignments[i][1],
                        model=assignments[i][0],
                        approved=True,  # Not used for market analysis
                        confidence=0.0,
                        summary="Analysis unavailable",
                        reasoning=f"Analysis failed: {str(review)[:200]}",
                        sentiment="neutral",
                    ))
                else:
                    valid_reviews.append(review)

            # Run judge with all reviews
            judge_result = await self._run_market_judge(valid_reviews, context)

            self._consecutive_failures = 0

            return MultiAgentReviewResult(
                reviews=valid_reviews,
                judge_decision=True,  # Not used for market analysis
                judge_confidence=judge_result["confidence"],
                judge_reasoning=judge_result["reasoning"],
                judge_recommendation=judge_result["recommendation"],
                final_veto_action=None,  # Not used for market analysis
                trade_context=context,
            )

        except Exception as e:
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)
            logger.error(
                "multi_agent_market_analysis_failed",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )

            return MultiAgentReviewResult(
                reviews=[],
                judge_decision=True,
                judge_confidence=0.0,
                judge_reasoning=f"Market analysis failed: {str(e)[:200]}",
                judge_recommendation="wait",
                final_veto_action=None,
                trade_context=context,
            )

    async def _run_market_reviewer(
        self, model: str, stance: str, context: dict
    ) -> AgentReview:
        """Run single market reviewer with assigned stance and optional web search."""
        system_prompts = {
            "bullish": SYSTEM_PROMPT_MARKET_BULLISH,
            "neutral": SYSTEM_PROMPT_MARKET_NEUTRAL,
            "bearish": SYSTEM_PROMPT_MARKET_BEARISH,
        }

        prompt = self._build_market_prompt(context)
        logger.debug("market_prompt_built", model=model, stance=stance, prompt_preview=prompt[:500])
        response = await self._call_api(
            model,
            system_prompts[stance],
            prompt,
            enable_tools=True,  # Enable web search for market analysis
        )
        data = self._extract_json(response)

        summary = data.get("summary", "")
        reasoning = data.get("reasoning", "No reasoning provided")

        # Fallback: if no summary, use first sentence of reasoning
        if not summary and reasoning:
            summary = reasoning.split('.')[0] + '.' if '.' in reasoning else reasoning[:80]

        # Map outlook to sentiment for consistency
        outlook = data.get("outlook", "neutral")

        return AgentReview(
            stance=stance,
            model=model,
            approved=True,  # Not used for market analysis
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            summary=summary,
            reasoning=reasoning,
            sentiment=outlook,  # Store outlook as sentiment
        )

    async def _run_market_judge(
        self, reviews: list[AgentReview], context: dict
    ) -> dict:
        """Run judge to synthesize market reviews."""
        prompt = self._build_market_judge_prompt(reviews, context)
        response = await self._call_api(self.judge_model, SYSTEM_PROMPT_MARKET_JUDGE, prompt)
        data = self._extract_json(response)

        # Validate recommendation
        recommendation = data.get("recommendation", "wait")
        if recommendation not in ("wait", "accumulate", "reduce"):
            recommendation = "wait"

        return {
            "outlook": data.get("outlook", "neutral"),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "reasoning": data.get("reasoning", "No reasoning provided"),
            "recommendation": recommendation,
        }

    def _build_market_prompt(self, context: dict) -> str:
        """Build prompt for market analysis reviewers with research data."""
        indicators = context.get("indicators", {})

        # Format indicator values
        rsi_str = f"{indicators.get('rsi', 'N/A')}"
        macd_str = f"{indicators.get('macd_histogram', 'N/A')}"
        bb_str = f"{indicators.get('bb_percent_b', 'N/A')}"
        ema_gap_str = f"{indicators.get('ema_gap', 'N/A')}"

        # Build price change info
        price_changes = []
        if context.get("price_change_1h") is not None:
            price_changes.append(f"1h: {context['price_change_1h']:+.2f}%")
        if context.get("price_change_24h") is not None:
            price_changes.append(f"24h: {context['price_change_24h']:+.2f}%")
        price_change_str = ", ".join(price_changes) if price_changes else "N/A"

        # Include research data if available
        research_section = ""
        research = context.get("research", "")
        if research and research != "(Research data unavailable)":
            research_section = f"""
=== ONLINE RESEARCH ===
{research}
"""

        # Web search hint
        tool_hint = ""
        if self.ai_web_search_enabled:
            tool_hint = "\nYou have access to a web_search tool if you need additional information."

        return f"""Analyze current Bitcoin market conditions:

=== MARKET DATA ===
Price: ¤{context['price']:,.2f}
Price Changes: {price_change_str}
Volatility: {context['volatility'].upper()}
Market Regime: {context['regime']}

Technical Indicators:
- RSI: {rsi_str}
- MACD Histogram: {macd_str}
- Bollinger %B: {bb_str}
- EMA Gap: {ema_gap_str}

Sentiment:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})
{research_section}
=== YOUR ANALYSIS ==={tool_hint}
Provide your market analysis from your assigned perspective."""

    def _build_market_judge_prompt(self, reviews: list[AgentReview], context: dict) -> str:
        """Build prompt for market judge with all reviews."""
        reviews_text = []
        for review in reviews:
            stance_label = review.stance.upper()
            model_short = review.model.split("/")[-1]
            outlook = review.sentiment.upper()
            reviews_text.append(
                f"[{stance_label}] ({model_short}) - Outlook: {outlook} ({review.confidence:.0%} confidence)\n"
                f"  Reasoning: {review.reasoning}"
            )

        return f"""Market Analysis at ¤{context['price']:,.2f}
Volatility: {context['volatility'].upper()}
Fear & Greed: {context['fear_greed']} ({context['fear_greed_class']})

Analyst Reviews:
{chr(10).join(reviews_text)}

Based on these three perspectives, provide the final market outlook."""

    async def _call_api(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        enable_tools: bool = False,
    ) -> str:
        """
        Call OpenRouter API with optional tool support.

        Args:
            model: Model identifier
            system_prompt: System message
            user_prompt: User message
            enable_tools: Whether to enable web search tool

        Returns:
            Model response content as string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request_body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        # Add tools if enabled for this model
        tools = None
        if enable_tools and self.ai_web_search_enabled:
            tools = get_tools_for_model(model, enabled=True)
            if tools:
                request_body["tools"] = tools
                request_body["tool_choice"] = "auto"  # Let model decide when to search

        logger.debug("api_request", model=model, tools_enabled=tools is not None)

        async with httpx.AsyncClient(timeout=float(self.api_timeout)) as client:
            # Initial request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Byte-Ventures/claude-trader",
                "X-Title": "Claude Trader",
            }

            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                return "{}"

            message = choices[0].get("message", {})

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls and tools:
                logger.info(
                    "tool_calls_received",
                    model=model,
                    num_calls=len(tool_calls),
                )

                # Execute tool calls
                tool_results = await handle_tool_calls(tool_calls)

                # Add assistant message with tool calls and tool results
                messages.append(message)
                messages.extend(tool_results)

                # Continue conversation with tool results
                request_body["messages"] = messages
                del request_body["tools"]  # Don't allow more tool calls
                if "tool_choice" in request_body:
                    del request_body["tool_choice"]

                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})

            # Return content
            content = message.get("content", "")
            if content:
                return content

            return "{}"

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        start = response.find('{')
        if start == -1:
            raise ValueError("No JSON found in response")

        depth = 0
        for i, char in enumerate(response[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(response[start:i+1])

        raise ValueError("Unbalanced JSON braces in response")
