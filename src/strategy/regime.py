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

    # Trend-aware sentiment modifiers
    # Modifies how sentiment adjustments are applied based on trend context
    #
    # threshold_mult: Multiplier for sentiment threshold adjustment
    #   - 0.0 = ignore sentiment adjustment entirely
    #   - 1.0 = apply full sentiment adjustment
    #   - >1.0 = amplify sentiment adjustment
    #
    # position_mult: Additional position size multiplier (compounds with base)
    #
    # Format: (sentiment_category, trend, signal_action) -> modifiers
    SENTIMENT_TREND_MODIFIERS = {
        # ====================================================================
        # EXTREME FEAR
        # ====================================================================
        # BUY signals
        ("extreme_fear", "bullish", "buy"): {"threshold_mult": 1.2, "position_mult": 1.15},   # Contrarian opportunity
        ("extreme_fear", "bearish", "buy"): {"threshold_mult": 0.0, "position_mult": 0.7},    # Fear justified, don't catch knife
        ("extreme_fear", "neutral", "buy"): {"threshold_mult": 0.8, "position_mult": 1.0},    # Cautious opportunity
        # SELL signals
        ("extreme_fear", "bullish", "sell"): {"threshold_mult": 0.3, "position_mult": 0.8},   # Wall of worry, don't panic sell
        ("extreme_fear", "bearish", "sell"): {"threshold_mult": 0.7, "position_mult": 0.75},  # Capitulation zone, cautious
        ("extreme_fear", "neutral", "sell"): {"threshold_mult": 0.5, "position_mult": 0.85},  # Reduce panic selling
        # ====================================================================
        # FEAR (non-extreme)
        # ====================================================================
        # BUY signals
        ("fear", "bullish", "buy"): {"threshold_mult": 1.1, "position_mult": 1.05},   # Mild opportunity
        ("fear", "bearish", "buy"): {"threshold_mult": 0.3, "position_mult": 0.8},    # Fear justified
        ("fear", "neutral", "buy"): {"threshold_mult": 0.9, "position_mult": 1.0},    # Slight caution
        # SELL signals
        ("fear", "bullish", "sell"): {"threshold_mult": 0.5, "position_mult": 0.9},   # Mild wall of worry
        ("fear", "bearish", "sell"): {"threshold_mult": 0.85, "position_mult": 0.9},  # Cautious selling
        ("fear", "neutral", "sell"): {"threshold_mult": 0.7, "position_mult": 0.95},  # Reduce fear-driven selling
        # ====================================================================
        # GREED (non-extreme)
        # ====================================================================
        # BUY signals
        ("greed", "bullish", "buy"): {"threshold_mult": 0.7, "position_mult": 0.9},   # Momentum, reduced penalty
        ("greed", "bearish", "buy"): {"threshold_mult": 1.2, "position_mult": 0.8},   # Early denial, increase caution
        ("greed", "neutral", "buy"): {"threshold_mult": 1.0, "position_mult": 0.9},   # Standard greed caution
        # SELL signals
        ("greed", "bullish", "sell"): {"threshold_mult": 1.2, "position_mult": 1.1},  # Good time for profits
        ("greed", "bearish", "sell"): {"threshold_mult": 1.1, "position_mult": 1.0},  # Early denial, modest sell bias
        ("greed", "neutral", "sell"): {"threshold_mult": 1.0, "position_mult": 1.0},  # Standard
        # ====================================================================
        # EXTREME GREED
        # ====================================================================
        # BUY signals
        ("extreme_greed", "bullish", "buy"): {"threshold_mult": 0.5, "position_mult": 0.85},  # Momentum real, reduce penalty
        ("extreme_greed", "bearish", "buy"): {"threshold_mult": 1.5, "position_mult": 0.6},   # Denial phase, don't buy
        ("extreme_greed", "neutral", "buy"): {"threshold_mult": 1.0, "position_mult": 0.75},  # Full greed caution
        # SELL signals
        ("extreme_greed", "bullish", "sell"): {"threshold_mult": 1.5, "position_mult": 1.2},  # PRIME: sell into euphoria
        ("extreme_greed", "bearish", "sell"): {"threshold_mult": 1.3, "position_mult": 1.1},  # Denial, sell before herd
        ("extreme_greed", "neutral", "sell"): {"threshold_mult": 1.2, "position_mult": 1.0},  # Take profits
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

    def __init__(self, config: Optional[RegimeConfig] = None, custom_modifiers: Optional[dict] = None):
        """
        Initialize with optional configuration.

        Args:
            config: Regime adaptation configuration
            custom_modifiers: Custom sentiment-trend modifiers (overrides hardcoded defaults)
                Format: {"sentiment_trend_signal": {"threshold_mult": float, "position_mult": float}}
                Example: {"extreme_fear_bearish_buy": {"threshold_mult": 0.0, "position_mult": 0.7}}
        """
        self.config = config or RegimeConfig()
        self._load_modifiers(custom_modifiers)

    def _load_modifiers(self, custom_modifiers: Optional[dict] = None) -> None:
        """
        Load sentiment-trend modifiers from custom config or use defaults.

        Args:
            custom_modifiers: Custom modifiers dict with flattened keys
        """
        if custom_modifiers is None:
            # Use hardcoded defaults
            self.sentiment_trend_modifiers = self.SENTIMENT_TREND_MODIFIERS
            return

        # Convert flattened keys to tuple format used internally
        # Example: "extreme_fear_bearish_buy" -> ("extreme_fear", "bearish", "buy")
        converted_modifiers = {}
        for key, value in custom_modifiers.items():
            parts = key.rsplit("_", 1)  # Split from right: "extreme_fear_bearish" and "buy"
            if len(parts) != 2:
                logger.warning(
                    "invalid_sentiment_modifier_key",
                    key=key,
                    reason="Expected format: sentiment_trend_signal"
                )
                continue

            signal = parts[1]
            sentiment_trend = parts[0].rsplit("_", 1)  # Split again: "extreme_fear" and "bearish"

            if len(sentiment_trend) != 2:
                logger.warning(
                    "invalid_sentiment_modifier_key",
                    key=key,
                    reason="Expected format: sentiment_trend_signal"
                )
                continue

            sentiment = sentiment_trend[0]
            trend = sentiment_trend[1]

            converted_modifiers[(sentiment, trend, signal)] = value

        self.sentiment_trend_modifiers = converted_modifiers
        logger.info(
            "loaded_custom_sentiment_modifiers",
            count=len(converted_modifiers),
            keys=sorted(custom_modifiers.keys())[:5]  # Log first 5 keys as sample
        )

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

        # Sentiment component (trend-aware)
        if self.config.sentiment_enabled and sentiment and sentiment.value is not None:
            sentiment_category = self._classify_sentiment(sentiment.value)
            adj = self.SENTIMENT_ADJUSTMENTS.get(
                sentiment_category, {"threshold": 0, "position": 1.0}
            )

            # Base sentiment adjustments
            base_threshold = adj["threshold"]
            base_position = adj["position"]

            # Apply trend-aware modifiers
            # Key principle: fear in downtrend is justified (not opportunity),
            # greed in uptrend is momentum (not necessarily overbought)
            modifier_key = (sentiment_category, trend, signal_action)
            modifier = self.sentiment_trend_modifiers.get(
                modifier_key,
                {"threshold_mult": 1.0, "position_mult": 1.0}  # Default: no modification
            )

            # Apply modifiers to base adjustments
            # Example: extreme_greed (base_position=0.75) with amplifier (position_mult=1.2)
            # modified = 1.0 + (0.75 - 1.0) * 1.2 = 1.0 + (-0.25 * 1.2) = 0.7
            # This amplifies the 25% reduction to a 30% reduction
            modified_threshold = base_threshold * modifier["threshold_mult"]
            modified_position = 1.0 + (base_position - 1.0) * modifier["position_mult"]

            # Apply scale (use round() to avoid truncation bias with small values)
            sentiment_threshold = round(modified_threshold * scale)
            sentiment_position = 1.0 + (modified_position - 1.0) * scale

            threshold_adj += sentiment_threshold
            position_mult *= sentiment_position

            # Track whether trend modified the sentiment adjustment
            was_modified = modifier["threshold_mult"] != 1.0 or modifier["position_mult"] != 1.0

            components["sentiment"] = {
                "value": sentiment.value,
                "category": sentiment_category,
                "threshold_adj": sentiment_threshold,
                "position_mult": round(sentiment_position, 2),
                "trend_modified": was_modified,
                "original_threshold_adj": int(base_threshold * scale) if was_modified else None,
            }

        # Volatility component
        if self.config.volatility_enabled:
            vol_adj = self.VOLATILITY_ADJUSTMENTS.get(
                volatility, {"threshold": 0, "position": 1.0}
            )
            vol_threshold = round(vol_adj["threshold"] * scale)
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
                trend_threshold = round(trend_adj["buy_threshold"] * scale)
            elif signal_action == "sell":
                trend_threshold = round(trend_adj["sell_threshold"] * scale)
            else:
                trend_threshold = 0

            threshold_adj += trend_threshold

            components["trend"] = {
                "direction": trend,
                "signal": signal_action,
                "threshold_adj": trend_threshold,
            }

        # Clamp values
        unclamped_position = position_mult
        threshold_adj = max(-20, min(20, threshold_adj))
        position_mult = max(0.5, min(1.5, position_mult))

        # Log final position multiplier for observability (helps debug compounding effects)
        if position_mult != 1.0:
            logger.debug(
                "regime_position_multiplier",
                final=round(position_mult, 3),
                unclamped=round(unclamped_position, 3),
                clamped=unclamped_position != position_mult,
            )

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
