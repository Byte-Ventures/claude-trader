"""
Claude AI trade reviewer.

Pre-trade validation that:
1. Analyzes every trade decision before execution
2. Can veto trades with configurable actions
3. Posts analysis to Telegram for transparency
4. Reviews "interesting holds" near threshold
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.ai.sentiment import fetch_fear_greed_index, get_trade_summary, FearGreedResult, TradeSummary
from src.strategy.signal_scorer import SignalResult

logger = structlog.get_logger(__name__)

# OpenRouter API endpoint (OpenAI-compatible)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ReviewResult:
    """Result of Claude's trade review."""

    approved: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    sentiment: str  # "bullish" / "bearish" / "neutral"
    veto_action: Optional[str]  # "skip" / "reduce" / "delay" / None
    trade_context: dict = field(default_factory=dict)  # Context for Telegram


# System prompt for trade review
SYSTEM_PROMPT_TRADE = """You are a Bitcoin trading risk analyst. Review proposed trades and assess if they should proceed.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Positive scores = bullish signals, negative scores = bearish signals
- Trade executes when |score| >= threshold (e.g., score >= 60 for buy, score <= -60 for sell)
- The breakdown shows each indicator's contribution to the total score

Consider:
1. Current market sentiment (Fear & Greed index provided)
2. Technical signal strength and indicator agreement
3. Recent trade performance (win rate, P&L)
4. Any concerning patterns

Respond with JSON only:
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "sentiment": "bullish"/"bearish"/"neutral",
  "reasoning": "1-2 sentence explanation"
}

Only disapprove (approved=false) with high confidence (>0.8) if you see clear danger signals like:
- Extreme greed (>75) during a buy signal
- Extreme fear (<25) during a sell signal
- Very poor recent performance (win rate < 30%)
- Technical signal contradicts sentiment strongly

Be conservative - when in doubt, approve the trade."""

SYSTEM_PROMPT_HOLD = """You are a Bitcoin trading analyst. Explain why the trading bot is holding instead of trading.

Signal scoring system:
- Score ranges from -100 (strong sell) to +100 (strong buy)
- Positive scores = bullish signals, negative scores = bearish signals
- Trade executes when |score| >= threshold (e.g., score >= 60 for buy, score <= -60 for sell)
- Current score's magnitude is below threshold, so the bot holds

The breakdown shows each indicator's contribution:
- RSI: Momentum (positive = oversold/buy, negative = overbought/sell)
- MACD: Trend momentum
- Bollinger: Mean reversion (positive = near lower band, negative = near upper band)
- EMA: Trend direction
- Volume: Confirmation boost/penalty
- Trend Filter: Counter-trend penalty

Respond with JSON only:
{
  "sentiment": "bullish"/"bearish"/"neutral",
  "reasoning": "1-2 sentence explanation of the current market signals",
  "confidence": 0.0-1.0
}"""


class TradeReviewer:
    """
    Claude AI trade reviewer.

    Reviews every trade decision and interesting holds,
    posting analysis to Telegram for transparency.
    """

    def __init__(
        self,
        api_key: str,
        db,
        veto_action: str = "info",
        veto_threshold: float = 0.8,
        position_reduction: float = 0.5,
        delay_minutes: int = 15,
        interesting_hold_margin: int = 15,
        model: str = "anthropic/claude-sonnet-4",
        review_all: bool = False,
    ):
        """
        Initialize trade reviewer.

        Args:
            api_key: OpenRouter API key
            db: Database instance for trade history
            veto_action: Action on veto - skip/reduce/delay/info
            veto_threshold: Confidence threshold to trigger veto (0.5-1.0)
            position_reduction: Position size multiplier for "reduce" action
            delay_minutes: Minutes to delay for "delay" action
            interesting_hold_margin: Score margin from threshold for interesting holds
            model: Model to use via OpenRouter (e.g., anthropic/claude-sonnet-4)
            review_all: Review ALL decisions (for debugging/testing)
        """
        self.api_key = api_key
        self.db = db
        self.veto_action = veto_action
        self.veto_threshold = veto_threshold
        self.position_reduction = position_reduction
        self.delay_minutes = delay_minutes
        self.interesting_hold_margin = interesting_hold_margin
        self.model = model
        self.review_all = review_all

        # Circuit breaker: track consecutive API failures
        self._consecutive_failures = 0
        self._max_failures = 5  # After 5 failures, force position reduction
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_reset_hours = 24  # Auto-reset after 24 hours

    def should_review(
        self, signal_result: SignalResult, threshold: int
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if this signal warrants AI review.

        Args:
            signal_result: Signal from SignalScorer
            threshold: Signal threshold for trades

        Returns:
            (should_review, review_type) where review_type is "trade", "interesting_hold", or "hold"
        """
        # Always review buy/sell signals
        if signal_result.action in ("buy", "sell"):
            return (True, "trade")

        # Check for "interesting hold" - score close to threshold
        distance_to_buy = abs(signal_result.score - threshold)
        distance_to_sell = abs(signal_result.score - (-threshold))

        if min(distance_to_buy, distance_to_sell) <= self.interesting_hold_margin:
            return (True, "interesting_hold")

        # Review all holds if debug mode enabled
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
                logger.info(
                    "claude_circuit_breaker_auto_reset",
                    hours_since_failure=f"{hours_since:.1f}",
                )

    async def review_trade(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        trading_pair: str,
        review_type: str = "trade",
    ) -> ReviewResult:
        """
        Review a trade decision or interesting hold.

        Args:
            signal_result: Signal from SignalScorer
            current_price: Current market price
            trading_pair: Trading pair (e.g., "BTC-USD")
            review_type: "trade" or "interesting_hold"

        Returns:
            ReviewResult with approval status and reasoning
        """
        # Check for circuit breaker auto-reset
        self._check_circuit_breaker_reset()

        # Gather context
        fear_greed = await fetch_fear_greed_index()
        trade_summary = get_trade_summary(self.db, days=7)

        # Build context for prompt and Telegram
        context = self._build_context(
            signal_result, current_price, trading_pair, fear_greed, trade_summary, review_type
        )

        # Build prompt
        prompt = self._build_prompt(context, review_type)

        # Call Claude with retry and circuit breaker
        try:
            response = await self._call_claude_with_retry(prompt, review_type)
            result = self._parse_response(response, context, review_type)
            # Reset failure counter on success
            self._consecutive_failures = 0
        except (RetryError, Exception) as e:
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now()  # Track for auto-reset
            logger.error(
                "claude_review_failed",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )

            # Circuit breaker: if too many failures, force position reduction
            if self._consecutive_failures >= self._max_failures:
                logger.warning(
                    "claude_circuit_breaker_triggered",
                    failures=self._consecutive_failures,
                    action="force_reduce",
                )
                result = ReviewResult(
                    approved=True,
                    confidence=0.0,
                    reasoning=f"Claude unavailable ({self._consecutive_failures} failures) - reducing position",
                    sentiment="neutral",
                    veto_action="reduce",  # Force position reduction
                    trade_context=context,
                )
            else:
                # Normal fail-open: approve trade
                result = ReviewResult(
                    approved=True,
                    confidence=0.0,
                    reasoning=f"Review failed: {str(e)[:200]}",  # Expanded from 50 to 200 chars
                    sentiment="neutral",
                    veto_action=None,
                    trade_context=context,
                )

        # For interesting holds, always approved (nothing to veto)
        if review_type == "interesting_hold":
            result.approved = True
            result.veto_action = None

        return result

    def _build_context(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        trading_pair: str,
        fear_greed: FearGreedResult,
        trade_summary: TradeSummary,
        review_type: str,
    ) -> dict:
        """Build context dict for prompt and Telegram."""
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

    def _build_prompt(self, context: dict, review_type: str) -> str:
        """Build the user prompt for Claude."""
        score = context['score']
        threshold = 60  # TODO: pass from settings

        if review_type == "trade":
            return f"""Review this trade:

Action: {context['action'].upper()}
Price: ${context['price']:,.2f}
Signal Score: {score:+d} (threshold: ±{threshold})
Signal Breakdown: {json.dumps(context['breakdown'])}

Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Recent Performance (7 days):
- Win Rate: {context['win_rate']:.0f}%
- Net P&L: ${context['net_pnl']:+,.2f}
- Total Trades: {context['total_trades']}

Should this trade proceed?"""
        else:
            # Hold analysis
            return f"""Analyze this hold decision:

Signal Score: {score:+d} (need ≥+{threshold} for buy or ≤-{threshold} for sell)
Price: ${context['price']:,.2f}
Signal Breakdown: {json.dumps(context['breakdown'])}

Market Context:
- Fear & Greed Index: {context['fear_greed']} ({context['fear_greed_class']})

Recent Performance (7 days):
- Win Rate: {context['win_rate']:.0f}%
- Net P&L: ${context['net_pnl']:+,.2f}

Explain what the indicators are showing."""

    async def _call_claude_with_retry(self, prompt: str, review_type: str) -> str:
        """Call Claude API with retry logic."""
        # tenacity doesn't support async decorators easily, so implement manually
        last_exception = None
        for attempt in range(3):
            try:
                return await self._call_claude(prompt, review_type)
            except httpx.HTTPStatusError as e:
                # Don't retry 4xx errors (client errors) except 429 (rate limit)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                last_exception = e
                wait_time = min(10, (2 ** attempt))  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(
                    "claude_api_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    wait_seconds=wait_time,
                )
                import asyncio
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_exception = e
                wait_time = min(10, (2 ** attempt))
                logger.warning(
                    "claude_api_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    wait_seconds=wait_time,
                )
                import asyncio
                await asyncio.sleep(wait_time)

        raise last_exception or Exception("Claude API call failed after retries")

    async def _call_claude(self, prompt: str, review_type: str) -> str:
        """Call OpenRouter API and get response."""
        system_prompt = SYSTEM_PROMPT_TRADE if review_type == "trade" else SYSTEM_PROMPT_HOLD

        request_body = {
            "model": self.model,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        logger.debug(
            "ai_request",
            model=self.model,
            review_type=review_type,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

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

            logger.debug(
                "ai_response",
                raw_response=data,
            )

            # Extract text from OpenAI-compatible response format
            choices = data.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"]

            return "{}"

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from response, handling potential nesting."""
        # Try direct parse first (cleanest case)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Find JSON boundaries with brace counting (handles nested JSON)
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

    def _parse_response(
        self, response: str, context: dict, review_type: str
    ) -> ReviewResult:
        """Parse Claude's JSON response into ReviewResult."""
        try:
            # Extract JSON with nested structure support
            data = self._extract_json(response)

            # Extract fields with defaults
            # Fix type coercion: string "false" evaluates to True with bool()
            approved_raw = data.get("approved", True)
            if isinstance(approved_raw, str):
                approved = approved_raw.lower() == "true"
            else:
                approved = bool(approved_raw)
            confidence = float(data.get("confidence", 0.5))
            sentiment = data.get("sentiment", "neutral")
            reasoning = data.get("reasoning", "No reasoning provided")

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            # Determine veto action if not approved
            veto_action = None
            if not approved and confidence >= self.veto_threshold:
                veto_action = self.veto_action

            logger.info(
                "claude_review_parsed",
                approved=approved,
                confidence=confidence,
                sentiment=sentiment,
                veto_action=veto_action,
            )

            return ReviewResult(
                approved=approved,
                confidence=confidence,
                reasoning=reasoning,
                sentiment=sentiment,
                veto_action=veto_action,
                trade_context=context,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("claude_response_parse_failed", error=str(e), response=response[:100])

            # Default to approved on parse failure
            return ReviewResult(
                approved=True,
                confidence=0.0,
                reasoning=f"Parse error: {response[:100]}",
                sentiment="neutral",
                veto_action=None,
                trade_context=context,
            )
