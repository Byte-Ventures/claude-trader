"""
Market data service for fetching candles and higher timeframe trends.

Handles:
- Higher timeframe (HTF) trend calculation with caching
- Daily and 4-hour trend analysis for multi-timeframe trading

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

import structlog

from src.api.exchange_protocol import ExchangeClient

if TYPE_CHECKING:
    from src.strategy.signal_scorer import SignalScorer

logger = structlog.get_logger(__name__)


@dataclass
class MarketConfig:
    """Configuration for market data service."""

    # Trading context
    trading_pair: str

    # Multi-timeframe configuration
    mtf_enabled: bool = False
    mtf_4h_enabled: bool = True
    mtf_daily_cache_minutes: int = 60
    mtf_4h_cache_minutes: int = 30
    mtf_daily_candle_limit: int = 50
    mtf_4h_candle_limit: int = 50


class MarketService:
    """
    Service for fetching and caching market data.

    Responsibilities:
    - Higher timeframe trend calculation (daily, 4-hour)
    - Trend caching with configurable TTL
    - Cache invalidation on settings change
    """

    def __init__(
        self,
        config: MarketConfig,
        exchange_client: ExchangeClient,
        signal_scorer: "SignalScorer",
    ):
        """
        Initialize market service.

        Args:
            config: Market configuration
            exchange_client: Exchange client for fetching candles
            signal_scorer: Signal scorer for trend calculation
        """
        self.config = config
        self.client = exchange_client
        self.signal_scorer = signal_scorer

        # HTF trend cache state
        self._daily_trend: str = "neutral"
        self._daily_last_fetch: Optional[datetime] = None
        self._4h_trend: str = "neutral"
        self._4h_last_fetch: Optional[datetime] = None

        # HTF cache performance metrics (accumulate over service lifetime)
        # Note: Counters grow unbounded, but Python ints have arbitrary precision
        # (no overflow). Safe for 24/7 operation.
        self._htf_cache_hits: int = 0
        self._htf_cache_misses: int = 0

    def get_htf_bias(self) -> tuple[str, Optional[str], Optional[str]]:
        """
        Get combined HTF bias from daily + 4-hour trends.

        Returns:
            Tuple of (combined_bias, daily_trend, 4h_trend)
            - Both bullish → "bullish"
            - Both bearish → "bearish"
            - Mixed/neutral → "neutral"
            - daily_trend is None when mtf_enabled=False
            - 4h_trend is None when mtf_4h_enabled=False
        """
        if not self.config.mtf_enabled:
            # When MTF is disabled, we don't fetch ANY HTF data, so both trends should be None
            return "neutral", None, None

        daily = self._get_timeframe_trend("ONE_DAY", self.config.mtf_daily_cache_minutes)

        # 4H is optional - when disabled, just use daily trend directly
        if not self.config.mtf_4h_enabled:
            # Daily-only mode: simpler, fewer API calls
            return daily, daily, None

        # MTF uses FOUR_HOUR (not SIX_HOUR) because:
        # - More frequent data points (6 candles/day vs 4 for 6H)
        # - More responsive to intraday trend shifts while still filtering hourly noise
        # - Provides good intermediate timeframe between daily and hourly trading
        # Note: SIX_HOUR remains a valid granularity for other uses, just not for MTF
        four_hour = self._get_timeframe_trend("FOUR_HOUR", self.config.mtf_4h_cache_minutes)

        # Combine: both must agree for strong bias
        if daily == "bullish" and four_hour == "bullish":
            combined = "bullish"
        elif daily == "bearish" and four_hour == "bearish":
            combined = "bearish"
        else:
            combined = "neutral"

        return combined, daily, four_hour

    def invalidate_cache(self) -> None:
        """
        Invalidate HTF trend cache when settings change.

        Called when MTF-related settings are updated at runtime to ensure
        the next iteration uses fresh data with the new parameters.
        """
        self._daily_last_fetch = None
        self._4h_last_fetch = None
        logger.info("htf_cache_invalidated")

    def get_cache_stats(self) -> tuple[int, int]:
        """
        Get cache performance statistics.

        Returns:
            Tuple of (cache_hits, cache_misses)
        """
        return self._htf_cache_hits, self._htf_cache_misses

    def update_config(self, config: MarketConfig) -> None:
        """
        Update the market configuration.

        Used for hot-reload of settings.

        Args:
            config: New market configuration
        """
        self.config = config
        logger.info("market_service_config_updated")

    def _get_timeframe_trend(self, granularity: str, cache_minutes: int) -> str:
        """
        Get trend for a specific timeframe with caching.

        Args:
            granularity: Candle granularity ("ONE_DAY" or "FOUR_HOUR")
            cache_minutes: Cache TTL in minutes

        Returns:
            Trend direction: "bullish", "bearish", or "neutral"
        """
        # Select appropriate cache based on granularity
        if granularity == "ONE_DAY":
            last_fetch = self._daily_last_fetch
            cached_trend = self._daily_trend
        elif granularity == "FOUR_HOUR":
            last_fetch = self._4h_last_fetch
            cached_trend = self._4h_trend
        else:
            raise ValueError(f"Unsupported granularity for HTF: {granularity}")

        now = datetime.now(timezone.utc)
        if last_fetch and (now - last_fetch) < timedelta(minutes=cache_minutes):
            self._htf_cache_hits += 1
            return cached_trend

        # Cache is stale or non-existent - count as cache miss
        # We count the miss once here, regardless of fetch outcome
        self._htf_cache_misses += 1

        # Select appropriate candle limit based on timeframe
        if granularity == "ONE_DAY":
            candle_limit = self.config.mtf_daily_candle_limit
        else:  # FOUR_HOUR
            candle_limit = self.config.mtf_4h_candle_limit

        try:
            candles = self.client.get_candles(
                self.config.trading_pair,
                granularity=granularity,
                limit=candle_limit,
            )

            # Validate candles before processing - need enough data for trend calculation
            # get_trend() only uses EMA crossover, so only ema_slow_period is required
            min_required = self.signal_scorer.ema_slow_period
            if candles is None or candles.empty or len(candles) < min_required:
                logger.warning(
                    "htf_insufficient_data",
                    timeframe=granularity,
                    candle_count=len(candles) if candles is not None and not candles.empty else 0,
                    required=min_required,
                )
                return cached_trend or "neutral"

            trend = self.signal_scorer.get_trend(candles)

            # Update cache
            if granularity == "ONE_DAY":
                self._daily_trend = trend
                self._daily_last_fetch = now
            elif granularity == "FOUR_HOUR":
                self._4h_trend = trend
                self._4h_last_fetch = now
            else:
                raise ValueError(f"Unsupported granularity for HTF: {granularity}")

            logger.info(
                "htf_trend_updated",
                timeframe=granularity,
                trend=trend,
                cache_hits=self._htf_cache_hits,
                cache_misses=self._htf_cache_misses,
            )
            return trend
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, NotImplementedError) as e:
            # Expected failures: network issues, API errors, data parsing issues,
            # or unsupported granularity (NotImplementedError).
            # Cache miss was already counted above when we decided to fetch
            # Fail-open: return cached trend or neutral, never block trading
            logger.warning("htf_fetch_failed", timeframe=granularity, error=str(e), error_type=type(e).__name__)
            return cached_trend or "neutral"
        except Exception as e:
            # Unexpected errors - log at error level but still fail-open
            # Cache miss was already counted above when we decided to fetch
            # Financial bot should never crash due to HTF analysis failure
            logger.error("htf_fetch_unexpected_error", timeframe=granularity, error=str(e), error_type=type(e).__name__, traceback=traceback.format_exc())
            return cached_trend or "neutral"
