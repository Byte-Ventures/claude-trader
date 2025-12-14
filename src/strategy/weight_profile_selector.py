"""
AI-driven weight profile selection for signal scoring.

Analyzes market conditions using AI to select optimal indicator weights.
Profiles are designed for different market regimes:
- trending: Emphasizes MACD and EMA for directional moves
- ranging: Emphasizes RSI and Bollinger for mean reversion
- volatile: Balanced with extra Bollinger weight for volatility
- default: Balanced weights for uncertain conditions
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import structlog

from src.strategy.signal_scorer import SignalWeights

logger = structlog.get_logger(__name__)


# Predefined weight profiles optimized for different market conditions
WEIGHT_PROFILES: dict[str, SignalWeights] = {
    "trending": SignalWeights(rsi=10, macd=35, bollinger=10, ema=35, volume=10),
    "ranging": SignalWeights(rsi=35, macd=10, bollinger=35, ema=10, volume=10),
    "volatile": SignalWeights(rsi=20, macd=20, bollinger=30, ema=15, volume=15),
    "default": SignalWeights(rsi=25, macd=25, bollinger=20, ema=15, volume=15),
}


@dataclass
class ProfileSelection:
    """Result of AI weight profile selection."""

    profile_name: str
    weights: SignalWeights
    confidence: float  # 0.0 to 1.0
    reasoning: str
    selected_at: datetime
    market_context: dict


@dataclass
class ProfileSelectorConfig:
    """Configuration for weight profile selector."""

    enabled: bool = True
    cache_minutes: int = 15  # Match candle interval default
    fallback_profile: str = "default"
    model: str = "openai/gpt-5.2"  # Fast, capable model for profile selection


# System prompt for AI profile selection
SYSTEM_PROMPT = """You are a trading strategy optimizer. Select the best indicator weight profile for the given market conditions.

Available profiles:
1. "trending" - Emphasizes MACD (35%) and EMA (35%) for strong directional moves. RSI/BB reduced.
   Use when: Clear bullish/bearish trend, momentum building, breakouts

2. "ranging" - Emphasizes RSI (35%) and Bollinger (35%) for mean reversion. MACD/EMA reduced.
   Use when: Sideways/choppy markets, price oscillating between support/resistance

3. "volatile" - Balanced with extra Bollinger (30%) for volatility. All indicators moderate.
   Use when: High ATR, uncertain direction, potential reversals, major news

4. "default" - Balanced weights for uncertain conditions.
   Use when: No clear regime, mixed signals, transitional markets

Respond with JSON only:
{
  "profile": "trending|ranging|volatile|default",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation (1-2 sentences)"
}"""


class WeightProfileSelector:
    """
    AI-driven weight profile selector.

    Analyzes market conditions and selects optimal indicator weights.
    Results are cached to avoid excessive AI calls.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[ProfileSelectorConfig] = None,
    ):
        """
        Initialize weight profile selector.

        Args:
            api_key: OpenRouter API key for AI calls
            config: Optional configuration
        """
        self.api_key = api_key
        self.config = config or ProfileSelectorConfig()

        # Caching
        self._cached_selection: Optional[ProfileSelection] = None
        self._last_selection_time: Optional[datetime] = None

        # Circuit breaker for AI failures
        self._consecutive_failures = 0
        self._max_failures = 3

        logger.info(
            "weight_profile_selector_initialized",
            cache_minutes=self.config.cache_minutes,
            fallback_profile=self.config.fallback_profile,
            model=self.config.model,
        )

    def get_current_weights(self) -> SignalWeights:
        """Get current weights (from cache or default)."""
        if self._cached_selection:
            return self._cached_selection.weights
        return WEIGHT_PROFILES[self.config.fallback_profile]

    def get_current_profile(self) -> str:
        """Get current profile name."""
        if self._cached_selection:
            return self._cached_selection.profile_name
        return self.config.fallback_profile

    def should_update(self) -> bool:
        """Check if profile should be re-evaluated."""
        if not self.config.enabled:
            return False
        if self._last_selection_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_selection_time).total_seconds() / 60
        return elapsed >= self.config.cache_minutes

    async def select_profile(
        self,
        indicators: dict,
        volatility: str,
        trend: str,
        current_price: Decimal,
        fear_greed: Optional[int] = None,
    ) -> ProfileSelection:
        """
        Select optimal weight profile based on market conditions.

        Args:
            indicators: Dict with RSI, MACD, BB values
            volatility: Volatility level (low/normal/high/extreme)
            trend: Trend direction (bullish/neutral/bearish)
            current_price: Current market price
            fear_greed: Optional Fear & Greed index value

        Returns:
            ProfileSelection with chosen profile and reasoning
        """
        # Return cached if still valid
        if not self.should_update() and self._cached_selection:
            return self._cached_selection

        # Circuit breaker check
        if self._consecutive_failures >= self._max_failures:
            logger.warning(
                "profile_selector_circuit_open",
                failures=self._consecutive_failures,
            )
            return self._fallback_selection(indicators, volatility, trend)

        try:
            selection = await self._call_ai(
                indicators, volatility, trend, current_price, fear_greed
            )
            self._cached_selection = selection
            self._last_selection_time = datetime.now(timezone.utc)
            self._consecutive_failures = 0

            logger.info(
                "weight_profile_selected",
                profile=selection.profile_name,
                confidence=selection.confidence,
                reasoning=selection.reasoning[:100],
            )

            return selection

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(
                "profile_selection_failed",
                error=str(e),
                failures=self._consecutive_failures,
            )
            return self._fallback_selection(indicators, volatility, trend)

    def _fallback_selection(
        self,
        indicators: dict,
        volatility: str,
        trend: str,
    ) -> ProfileSelection:
        """Rule-based fallback when AI is unavailable."""
        # Simple heuristic fallback
        if volatility in ("high", "extreme"):
            profile = "volatile"
            reasoning = f"High volatility ({volatility}) detected, using volatile profile"
        elif trend in ("bullish", "bearish"):
            profile = "trending"
            reasoning = f"Strong {trend} trend detected, using trending profile"
        elif indicators.get("rsi") and (
            indicators["rsi"] < 35 or indicators["rsi"] > 65
        ):
            profile = "ranging"
            reasoning = "RSI at extreme, potential mean reversion - using ranging profile"
        else:
            profile = self.config.fallback_profile
            reasoning = "No clear market condition, using fallback profile"

        selection = ProfileSelection(
            profile_name=profile,
            weights=WEIGHT_PROFILES[profile],
            confidence=0.5,  # Lower confidence for fallback
            reasoning=reasoning,
            selected_at=datetime.now(timezone.utc),
            market_context={"fallback": True, "volatility": volatility, "trend": trend},
        )

        # Cache the fallback selection too
        self._cached_selection = selection
        self._last_selection_time = datetime.now(timezone.utc)

        logger.info(
            "weight_profile_fallback_used",
            profile=profile,
            reason=reasoning,
        )

        return selection

    async def _call_ai(
        self,
        indicators: dict,
        volatility: str,
        trend: str,
        current_price: Decimal,
        fear_greed: Optional[int],
    ) -> ProfileSelection:
        """Call AI model for profile selection."""
        import httpx
        import json

        prompt = self._build_prompt(
            indicators, volatility, trend, current_price, fear_greed
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "max_tokens": 200,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if not data.get("choices") or not isinstance(data["choices"], list):
                raise ValueError("Invalid API response: missing 'choices' array")
            if len(data["choices"]) == 0:
                raise ValueError("Invalid API response: empty 'choices' array")
            if not data["choices"][0].get("message"):
                raise ValueError("Invalid API response: missing 'message' in choice")
            if "content" not in data["choices"][0]["message"]:
                raise ValueError("Invalid API response: missing 'content' in message")

            content = data["choices"][0]["message"]["content"]
            result = self._parse_response(content)

            return ProfileSelection(
                profile_name=result["profile"],
                weights=WEIGHT_PROFILES[result["profile"]],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                selected_at=datetime.now(timezone.utc),
                market_context={
                    "volatility": volatility,
                    "trend": trend,
                    "rsi": indicators.get("rsi"),
                    "fear_greed": fear_greed,
                },
            )

    def _build_prompt(
        self,
        indicators: dict,
        volatility: str,
        trend: str,
        current_price: Decimal,
        fear_greed: Optional[int],
    ) -> str:
        """Build prompt for AI profile selection."""
        rsi = indicators.get("rsi", "N/A")
        macd = indicators.get("macd_histogram", "N/A")
        bb = indicators.get("bb_percent_b", "N/A")

        # Format values
        if isinstance(rsi, (int, float)):
            rsi = f"{rsi:.1f}"
        if isinstance(macd, (int, float)):
            macd = f"{macd:.2f}"
        if isinstance(bb, (int, float)):
            bb = f"{bb:.2f}"

        return f"""Select the optimal indicator weight profile for current market conditions.

Market Data:
- Price: ${float(current_price):,.2f}
- Volatility: {volatility.upper()}
- Trend: {trend}
- RSI: {rsi}
- MACD Histogram: {macd}
- Bollinger %B: {bb}
- Fear & Greed: {fear_greed if fear_greed else 'N/A'}

Choose the best profile for these conditions."""

    def _parse_response(self, content: str) -> dict:
        """Parse AI response to extract profile selection."""
        import json

        # Try direct JSON parse
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
            else:
                raise ValueError("No valid JSON in response")

        # Validate profile name
        profile = data.get("profile", "default").lower()
        if profile not in WEIGHT_PROFILES:
            logger.warning(
                "unknown_profile_in_response",
                profile=profile,
                defaulting_to="default",
            )
            profile = "default"

        return {
            "profile": profile,
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.7)))),
            "reasoning": data.get("reasoning", "No reasoning provided"),
        }

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker (e.g., after config reload)."""
        self._consecutive_failures = 0
        logger.info("profile_selector_circuit_breaker_reset")

    def invalidate_cache(self) -> None:
        """Invalidate the cached selection."""
        self._cached_selection = None
        self._last_selection_time = None
        logger.info("profile_selector_cache_invalidated")
