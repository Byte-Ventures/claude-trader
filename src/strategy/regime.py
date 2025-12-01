"""
Market regime detection and strategy adaptation.

Combines multiple signals to determine market conditions and adjust
trading parameters accordingly:
- Sentiment (Fear & Greed Index)
- Volatility (ATR-based)
- Trend (EMA-based)

Provides threshold adjustments and position multipliers based on regime.
"""

from dataclasses import dataclass
from typing import Optional

import structlog

from src.ai.sentiment import FearGreedResult, fetch_fear_greed_index

logger = structlog.get_logger(__name__)


@dataclass
class RegimeAdjustments:
    """Adjustments to apply based on current market regime."""

    threshold_adjustment: int  # Add to base threshold (negative = easier to trade)
    position_multiplier: float  # Multiply position size
    regime_name: str  # Human-readable regime label
    components: dict  # Breakdown for logging

    @property
    def effective_threshold(self) -> int:
        """Calculate effective threshold given a base."""
        # This is informational - actual calculation happens in signal_scorer
        return self.threshold_adjustment


@dataclass
class RegimeConfig:
    """Configuration for regime adaptation."""

    enabled: bool = True
    sentiment_enabled: bool = True
    volatility_enabled: bool = True
    trend_enabled: bool = True
    adjustment_scale: float = 1.0  # 0.0 = no adjustment, 1.0 = normal, 2.0 = aggressive


class MarketRegime:
    """
    Calculates market regime and provides trading adjustments.

    Combines sentiment, volatility, and trend signals to determine
    whether to be more aggressive (risk_on) or cautious (risk_off).
    """

    # Sentiment adjustments (Fear & Greed Index)
    SENTIMENT_ADJUSTMENTS = {
        "extreme_fear": {"threshold": -10, "position": 1.25},  # 0-25
        "fear": {"threshold": -5, "position": 1.1},  # 25-45
        "neutral": {"threshold": 0, "position": 1.0},  # 45-55
        "greed": {"threshold": 5, "position": 0.9},  # 55-75
        "extreme_greed": {"threshold": 15, "position": 0.75},  # 75-100
    }

    # Volatility adjustments (ATR-based levels)
    VOLATILITY_ADJUSTMENTS = {
        "low": {"threshold": -5, "position": 1.1},
        "normal": {"threshold": 0, "position": 1.0},
        "high": {"threshold": 5, "position": 0.85},
        "extreme": {"threshold": 10, "position": 0.6},
    }

    # Trend adjustments (applied relative to signal direction)
    TREND_ADJUSTMENTS = {
        "bullish": {"buy_threshold": -5, "sell_threshold": 5},
        "neutral": {"buy_threshold": 0, "sell_threshold": 0},
        "bearish": {"buy_threshold": 5, "sell_threshold": -5},
    }

    def __init__(self, config: Optional[RegimeConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or RegimeConfig()

    def _classify_sentiment(self, value: int) -> str:
        """Classify Fear & Greed value into category."""
        if value < 25:
            return "extreme_fear"
        elif value < 45:
            return "fear"
        elif value <= 55:
            return "neutral"
        elif value <= 75:
            return "greed"
        else:
            return "extreme_greed"

    def calculate(
        self,
        sentiment: Optional[FearGreedResult],
        volatility: str,
        trend: str,
        signal_action: str,
    ) -> RegimeAdjustments:
        """
        Calculate regime adjustments based on market conditions.

        Args:
            sentiment: Fear & Greed Index result (or None if unavailable)
            volatility: Volatility level ("low", "normal", "high", "extreme")
            trend: Trend direction ("bullish", "neutral", "bearish")
            signal_action: Current signal ("buy", "sell", "hold")

        Returns:
            RegimeAdjustments with threshold and position adjustments
        """
        if not self.config.enabled:
            return RegimeAdjustments(
                threshold_adjustment=0,
                position_multiplier=1.0,
                regime_name="disabled",
                components={},
            )

        components = {}
        threshold_adj = 0
        position_mult = 1.0
        scale = self.config.adjustment_scale

        # Sentiment component
        if self.config.sentiment_enabled and sentiment and sentiment.value is not None:
            sentiment_category = self._classify_sentiment(sentiment.value)
            adj = self.SENTIMENT_ADJUSTMENTS.get(
                sentiment_category, {"threshold": 0, "position": 1.0}
            )
            sentiment_threshold = int(adj["threshold"] * scale)
            sentiment_position = 1.0 + (adj["position"] - 1.0) * scale

            threshold_adj += sentiment_threshold
            position_mult *= sentiment_position

            components["sentiment"] = {
                "value": sentiment.value,
                "category": sentiment_category,
                "threshold_adj": sentiment_threshold,
                "position_mult": round(sentiment_position, 2),
            }

        # Volatility component
        if self.config.volatility_enabled:
            vol_adj = self.VOLATILITY_ADJUSTMENTS.get(
                volatility, {"threshold": 0, "position": 1.0}
            )
            vol_threshold = int(vol_adj["threshold"] * scale)
            vol_position = 1.0 + (vol_adj["position"] - 1.0) * scale

            threshold_adj += vol_threshold
            position_mult *= vol_position

            components["volatility"] = {
                "level": volatility,
                "threshold_adj": vol_threshold,
                "position_mult": round(vol_position, 2),
            }

        # Trend component (direction-dependent)
        if self.config.trend_enabled:
            trend_adj = self.TREND_ADJUSTMENTS.get(
                trend, {"buy_threshold": 0, "sell_threshold": 0}
            )
            if signal_action == "buy":
                trend_threshold = int(trend_adj["buy_threshold"] * scale)
            elif signal_action == "sell":
                trend_threshold = int(trend_adj["sell_threshold"] * scale)
            else:
                trend_threshold = 0

            threshold_adj += trend_threshold

            components["trend"] = {
                "direction": trend,
                "signal": signal_action,
                "threshold_adj": trend_threshold,
            }

        # Clamp values
        threshold_adj = max(-20, min(20, threshold_adj))
        position_mult = max(0.5, min(1.5, position_mult))

        # Determine regime name
        if threshold_adj <= -10:
            regime_name = "risk_on"
        elif threshold_adj >= 10:
            regime_name = "risk_off"
        elif threshold_adj >= 5:
            regime_name = "cautious"
        elif threshold_adj <= -5:
            regime_name = "opportunistic"
        else:
            regime_name = "neutral"

        return RegimeAdjustments(
            threshold_adjustment=threshold_adj,
            position_multiplier=round(position_mult, 3),
            regime_name=regime_name,
            components=components,
        )


# Cached sentiment for periodic refresh
_sentiment_cache: Optional[FearGreedResult] = None
_sentiment_last_fetch: Optional[float] = None

SENTIMENT_CACHE_SECONDS = 900  # 15 minutes


async def get_cached_sentiment() -> Optional[FearGreedResult]:
    """
    Get cached sentiment, fetching if stale.

    Returns cached sentiment or None if fetch fails.
    Caches for 15 minutes to avoid API rate limits.
    """
    import time

    global _sentiment_cache, _sentiment_last_fetch

    now = time.time()

    if _sentiment_last_fetch is None or (now - _sentiment_last_fetch) > SENTIMENT_CACHE_SECONDS:
        try:
            _sentiment_cache = await fetch_fear_greed_index()
            _sentiment_last_fetch = now
            logger.debug(
                "sentiment_fetched",
                value=_sentiment_cache.value if _sentiment_cache else None,
                classification=_sentiment_cache.classification if _sentiment_cache else None,
            )
        except Exception as e:
            logger.warning("sentiment_fetch_failed", error=str(e))
            # Keep stale cache if available
            if _sentiment_cache is None:
                return None

    return _sentiment_cache
