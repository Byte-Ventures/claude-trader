"""
Multi-agent trade reviewer.

Uses 3 reviewer agents with different stances (Pro, Neutral, Opposing)
plus a judge agent for final decision synthesis.
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx
import structlog

from src.ai.sentiment import fetch_fear_greed_index, get_trade_summary, FearGreedResult, TradeSummary
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

Focus on:
- Favorable indicator readings
- Positive market conditions
- Historical patterns supporting this trade
- Risk/reward opportunity

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

Analyze:
- Both positive and negative signals
- Current market sentiment vs technical signals
- Risk factors AND opportunities
- Overall signal quality

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

Focus on:
- Warning signs in indicators
- Market conditions that could hurt this trade
- Historical patterns suggesting caution
- Downside risks and potential losses

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
4. Make the final decision and recommendation

Decision guidelines:
- If all three agree, follow the consensus
- If PRO and NEUTRAL approve with high confidence, likely approve
- If OPPOSING has very strong arguments (>0.8 confidence), consider rejecting
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

SYSTEM_PROMPT_HOLD = """You are a Bitcoin trading analyst. Explain why the trading bot is holding instead of trading.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Trade executes when |score| >= threshold
- Current score's magnitude is below threshold, so the bot holds

Respond with JSON only:
{
  "sentiment": "bullish"/"bearish"/"neutral",
  "reasoning": "1-2 sentence explanation of the current market signals",
  "confidence": 0.0-1.0
}"""


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
        veto_action: str = "info",
        veto_threshold: float = 0.8,
        position_reduction: float = 0.5,
        delay_minutes: int = 15,
        interesting_hold_margin: int = 15,
        review_all: bool = False,
    ):
        """
        Initialize multi-agent trade reviewer.

        Args:
            api_key: OpenRouter API key
            db: Database instance for trade history
            reviewer_models: List of 3 models for reviewers
            judge_model: Model for the judge
            veto_action: Action on veto - skip/reduce/delay/info
            veto_threshold: Confidence threshold to trigger veto
            position_reduction: Position size multiplier for "reduce" action
            delay_minutes: Minutes to delay for "delay" action
            interesting_hold_margin: Score margin from threshold for interesting holds
            review_all: Review ALL decisions (for debugging/testing)
        """
        self.api_key = api_key
        self.db = db
        self.reviewer_models = reviewer_models
        self.judge_model = judge_model
        self.veto_action = veto_action
        self.veto_threshold = veto_threshold
        self.position_reduction = position_reduction
        self.delay_minutes = delay_minutes
        self.interesting_hold_margin = interesting_hold_margin
        self.review_all = review_all

        # Circuit breaker
        self._consecutive_failures = 0
        self._max_failures = 5
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_reset_hours = 24

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
            hours_since = (datetime.now() - self._last_failure_time).total_seconds() / 3600
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
    ) -> MultiAgentReviewResult:
        """
        Multi-agent review process.

        Args:
            signal_result: Signal from SignalScorer
            current_price: Current market price
            trading_pair: Trading pair (e.g., "BTC-USD")
            review_type: "trade" or "interesting_hold"

        Returns:
            MultiAgentReviewResult with all reviews and judge decision
        """
        self._check_circuit_breaker_reset()

        # Gather context
        fear_greed = await fetch_fear_greed_index()
        trade_summary = get_trade_summary(self.db, days=7)
        context = self._build_context(
            signal_result, current_price, trading_pair, fear_greed, trade_summary, review_type
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

            # Determine veto action
            veto_action = None
            if not judge_result["approved"] and judge_result["confidence"] >= self.veto_threshold:
                veto_action = self.veto_action

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
            self._last_failure_time = datetime.now()
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
        system_prompts = {
            "pro": SYSTEM_PROMPT_PRO,
            "neutral": SYSTEM_PROMPT_NEUTRAL,
            "opposing": SYSTEM_PROMPT_OPPOSING,
        }

        prompt = self._build_reviewer_prompt(context)
        response = await self._call_api(model, system_prompts[stance], prompt)
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
        prompt = self._build_judge_prompt(reviews, context)
        response = await self._call_api(self.judge_model, SYSTEM_PROMPT_JUDGE, prompt)
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

    def _build_context(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        trading_pair: str,
        fear_greed: FearGreedResult,
        trade_summary: TradeSummary,
        review_type: str,
    ) -> dict:
        """Build context dict for prompts and Telegram."""
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
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _build_reviewer_prompt(self, context: dict) -> str:
        """Build prompt for reviewer agents."""
        return f"""Review this trade:

Action: {context['action'].upper()}
Price: ${context['price']:,.2f}
Signal Score: {context['score']:+d} (threshold: ±60)
Signal Breakdown: {json.dumps(context['breakdown'])}

Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Recent Performance (7 days):
- Win Rate: {context['win_rate']:.0f}%
- Net P&L: ${context['net_pnl']:+,.2f}
- Total Trades: {context['total_trades']}

Analyze this trade from your assigned perspective."""

    def _build_judge_prompt(self, reviews: list[AgentReview], context: dict) -> str:
        """Build prompt for judge with all reviews."""
        reviews_text = []
        for review in reviews:
            stance_label = {"pro": "PRO", "neutral": "NEUTRAL", "opposing": "OPPOSING"}[review.stance]
            model_short = review.model.split("/")[-1]
            verdict = "APPROVE" if review.approved else "REJECT"
            reviews_text.append(
                f"[{stance_label}] ({model_short}) - {verdict} ({review.confidence:.0%} confidence)\n"
                f"  Reasoning: {review.reasoning}"
            )

        return f"""Trade Decision: {context['action'].upper()} at ${context['price']:,.2f}
Signal Score: {context['score']:+d}

Agent Reviews:
{chr(10).join(reviews_text)}

Based on these three perspectives, make the final decision."""

    def _build_hold_prompt(self, context: dict) -> str:
        """Build prompt for hold analysis."""
        return f"""Analyze this hold decision:

Signal Score: {context['score']:+d} (need ≥+60 for buy or ≤-60 for sell)
Price: ${context['price']:,.2f}
Signal Breakdown: {json.dumps(context['breakdown'])}

Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Explain what the indicators are showing."""

    async def _call_api(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Call OpenRouter API."""
        request_body = {
            "model": model,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        logger.debug("api_request", model=model, prompt_preview=user_prompt[:100])

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/claude-trader",
                },
                json=request_body,
            )

            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"]

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
