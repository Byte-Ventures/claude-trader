"""
Multi-indicator confluence signal scorer.

Combines signals from multiple technical indicators to generate
a composite trading signal score from -100 (strong sell) to +100 (strong buy).

Trade execution requires score magnitude >= threshold (default: 60).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd
import structlog

from src.indicators.rsi import calculate_rsi, get_rsi_signal_graduated
from src.indicators.macd import calculate_macd, get_macd_signal_graduated, VALID_CANDLE_INTERVALS
from src.indicators.bollinger import calculate_bollinger_bands, get_bollinger_signal_graduated
from src.indicators.ema import calculate_ema_crossover, get_ema_signal_graduated, get_ema_trend
from src.indicators.atr import calculate_atr, get_volatility_level

logger = structlog.get_logger(__name__)


# Recommended signal thresholds by candle interval
# Shorter candles capture faster moves, so lower thresholds are appropriate
# Longer candles require higher conviction, so higher thresholds are better
_RECOMMENDED_THRESHOLDS = {
    "ONE_MINUTE": 50,       # Ultra-short term - catch fast moves
    "FIVE_MINUTE": 52,      # Very short term
    "FIFTEEN_MINUTE": 55,   # Day trading
    "THIRTY_MINUTE": 57,    # Intraday swing
    "ONE_HOUR": 58,         # Swing trading
    "TWO_HOUR": 60,         # Swing trading - balanced
    "SIX_HOUR": 62,         # Position trading
    "ONE_DAY": 65,          # Position trading - high conviction
}
_DEFAULT_THRESHOLD = 60


def get_recommended_threshold(candle_interval: Optional[str] = None) -> int:
    """
    Get recommended signal threshold for the given candle interval.

    Shorter candles have smaller price movements and faster reversals,
    so lower thresholds help catch more opportunities. Longer candles
    have more significant moves that warrant higher conviction thresholds.

    Args:
        candle_interval: Candle interval string (e.g., "FIFTEEN_MINUTE", "ONE_HOUR").
                        If None, returns the default threshold.

    Returns:
        Recommended threshold value (50-65 depending on interval).

    Example:
        >>> get_recommended_threshold("FIFTEEN_MINUTE")
        55
        >>> get_recommended_threshold("ONE_DAY")
        65
    """
    if candle_interval is None:
        return _DEFAULT_THRESHOLD
    if candle_interval not in VALID_CANDLE_INTERVALS:
        logger.warning(
            "invalid_candle_interval",
            interval=candle_interval,
            using="default",
            valid_intervals=list(VALID_CANDLE_INTERVALS)
        )
    return _RECOMMENDED_THRESHOLDS.get(candle_interval, _DEFAULT_THRESHOLD)


@dataclass
class SignalWeights:
    """Weights for each indicator in the composite score."""

    rsi: int = 25
    macd: int = 25
    bollinger: int = 20
    ema: int = 15
    volume: int = 15


@dataclass
class IndicatorValues:
    """Current values of all indicators."""

    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None
    volatility: str = "normal"


@dataclass
class SignalResult:
    """Result of signal calculation."""

    score: int  # -100 to +100
    action: str  # "buy", "sell", or "hold"
    indicators: IndicatorValues
    breakdown: dict[str, int]  # Score contribution by indicator
    confidence: float  # 0.0 to 1.0


class SignalScorer:
    """
    Confluence-based signal scoring system.

    Combines multiple indicators to generate a trading signal:
    - RSI: Momentum and overbought/oversold
    - MACD: Trend direction and momentum
    - Bollinger Bands: Volatility and mean reversion
    - EMA Crossover: Short-term trend
    - Volume: Confirmation of moves

    Each indicator contributes a weighted score. Trades execute when
    the total score exceeds the threshold.
    """

    def __init__(
        self,
        weights: Optional[SignalWeights] = None,
        threshold: int = 60,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
        atr_period: int = 14,
        momentum_rsi_threshold: float = 60.0,
        momentum_rsi_candles: int = 3,
        momentum_price_candles: int = 12,
        momentum_penalty_reduction: float = 0.5,
        candle_interval: Optional[str] = None,
        whale_volume_threshold: float = 3.0,
        whale_direction_threshold: float = 0.003,
        whale_boost_percent: float = 0.30,
        high_volume_boost_percent: float = 0.20,
        mtf_aligned_boost: int = 20,
        mtf_counter_penalty: int = 20,
    ):
        """
        Initialize signal scorer.

        Args:
            weights: Weights for each indicator
            threshold: Minimum score to trigger trade
            momentum_rsi_threshold: RSI must stay above this for momentum mode (default: 60)
            momentum_rsi_candles: Number of candles RSI must stay elevated (default: 3)
            momentum_price_candles: Number of candles to check for higher lows (default: 12)
            momentum_penalty_reduction: Factor to reduce overbought penalties (default: 0.5 = 50%)
            candle_interval: Candle interval for adaptive MACD scaling
                            (e.g., "FIFTEEN_MINUTE", "ONE_HOUR")
            *: Other indicator parameters

        Recommended thresholds by candle interval:
            - ONE_MINUTE to FIFTEEN_MINUTE (daytrading): 50-55 (catch faster moves)
            - THIRTY_MINUTE to TWO_HOUR (swing): 55-60 (balanced)
            - SIX_HOUR to ONE_DAY (position): 60-65 (higher conviction)
        """
        self.weights = weights or SignalWeights()
        self.threshold = threshold

        # Crash protection: track oversold buys
        self._oversold_buy_times: list[datetime] = []

        # Indicator parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.ema_fast = ema_fast
        self.ema_slow_period = ema_slow
        self.atr_period = atr_period
        self.candle_interval = candle_interval

        # Momentum mode parameters
        self.momentum_rsi_threshold = momentum_rsi_threshold
        self.momentum_rsi_candles = momentum_rsi_candles
        self.momentum_price_candles = momentum_price_candles
        self.momentum_penalty_reduction = momentum_penalty_reduction

        # Whale detection parameters
        self.whale_volume_threshold = whale_volume_threshold
        self.whale_direction_threshold = whale_direction_threshold
        self.whale_boost_percent = whale_boost_percent
        self.high_volume_boost_percent = high_volume_boost_percent

        # Multi-Timeframe confirmation parameters
        self.mtf_aligned_boost = mtf_aligned_boost
        self.mtf_counter_penalty = mtf_counter_penalty

    def update_settings(
        self,
        threshold: Optional[int] = None,
        whale_volume_threshold: Optional[float] = None,
        whale_direction_threshold: Optional[float] = None,
        whale_boost_percent: Optional[float] = None,
        high_volume_boost_percent: Optional[float] = None,
        rsi_period: Optional[int] = None,
        rsi_oversold: Optional[float] = None,
        rsi_overbought: Optional[float] = None,
        macd_fast: Optional[int] = None,
        macd_slow: Optional[int] = None,
        macd_signal: Optional[int] = None,
        bollinger_period: Optional[int] = None,
        bollinger_std: Optional[float] = None,
        ema_fast: Optional[int] = None,
        ema_slow: Optional[int] = None,
        atr_period: Optional[int] = None,
        momentum_rsi_threshold: Optional[float] = None,
        momentum_rsi_candles: Optional[int] = None,
        momentum_price_candles: Optional[int] = None,
        momentum_penalty_reduction: Optional[float] = None,
        candle_interval: Optional[str] = None,
        mtf_aligned_boost: Optional[int] = None,
        mtf_counter_penalty: Optional[int] = None,
    ) -> None:
        """
        Update scorer settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        """
        if threshold is not None:
            self.threshold = threshold
        if whale_volume_threshold is not None:
            self.whale_volume_threshold = whale_volume_threshold
        if whale_direction_threshold is not None:
            self.whale_direction_threshold = whale_direction_threshold
        if whale_boost_percent is not None:
            self.whale_boost_percent = whale_boost_percent
        if high_volume_boost_percent is not None:
            self.high_volume_boost_percent = high_volume_boost_percent
        if rsi_period is not None:
            self.rsi_period = rsi_period
        if rsi_oversold is not None:
            self.rsi_oversold = rsi_oversold
        if rsi_overbought is not None:
            self.rsi_overbought = rsi_overbought
        if macd_fast is not None:
            self.macd_fast = macd_fast
        if macd_slow is not None:
            self.macd_slow = macd_slow
        if macd_signal is not None:
            self.macd_signal_period = macd_signal
        if bollinger_period is not None:
            self.bollinger_period = bollinger_period
        if bollinger_std is not None:
            self.bollinger_std = bollinger_std
        if ema_fast is not None:
            self.ema_fast = ema_fast
        if ema_slow is not None:
            self.ema_slow_period = ema_slow
        if atr_period is not None:
            self.atr_period = atr_period
        if momentum_rsi_threshold is not None:
            self.momentum_rsi_threshold = momentum_rsi_threshold
        if momentum_rsi_candles is not None:
            self.momentum_rsi_candles = momentum_rsi_candles
        if momentum_price_candles is not None:
            self.momentum_price_candles = momentum_price_candles
        if momentum_penalty_reduction is not None:
            self.momentum_penalty_reduction = momentum_penalty_reduction
        if candle_interval is not None:
            self.candle_interval = candle_interval
        if mtf_aligned_boost is not None:
            self.mtf_aligned_boost = mtf_aligned_boost
        if mtf_counter_penalty is not None:
            self.mtf_counter_penalty = mtf_counter_penalty

        logger.info("signal_scorer_settings_updated")

    def update_weights(self, weights: SignalWeights) -> None:
        """
        Update indicator weights at runtime.

        Used by AI weight profile selector to adjust weights based on market conditions.

        Args:
            weights: New SignalWeights to apply
        """
        self.weights = weights
        logger.info(
            "signal_weights_updated",
            rsi=weights.rsi,
            macd=weights.macd,
            bollinger=weights.bollinger,
            ema=weights.ema,
            volume=weights.volume,
        )

    def is_momentum_mode(self, df: pd.DataFrame, rsi: pd.Series) -> tuple[bool, str]:
        """
        Detect if market is in sustained momentum mode.

        Momentum mode is active when RSI has been elevated for multiple candles
        and price structure shows higher lows (bullish continuation pattern).
        When active, overbought penalties are reduced to allow riding trends.

        Args:
            df: DataFrame with OHLCV data
            rsi: RSI series

        Returns:
            Tuple of (is_momentum_active, reason_string)
        """
        min_rsi_candles = self.momentum_rsi_candles
        min_price_candles = self.momentum_price_candles

        if len(rsi) < min_rsi_candles or len(df) < min_price_candles:
            return False, ""

        # Condition 1: RSI sustained above threshold for N candles
        # Use dropna() to handle any NaN values in the RSI series
        recent_rsi = rsi.tail(min_rsi_candles).dropna()
        if len(recent_rsi) < min_rsi_candles:
            # Not enough valid RSI values after dropping NaN
            return False, ""
        rsi_sustained = bool((recent_rsi > self.momentum_rsi_threshold).all())

        # Condition 2: Price making higher lows (bullish structure)
        # This checks if recent prices are staying above their local minimums,
        # indicating buyers are stepping in at progressively higher levels.
        close = df["close"].astype(float)
        recent_close = close.tail(min_price_candles)

        # Check for NaN in price data
        if recent_close.isna().any():
            return False, ""

        higher_lows = True

        # Iterate through recent candles (starting at index 3 to have a lookback window)
        # For each candle at position i, compare its price to the minimum of the
        # previous 3 candles. If current price <= that minimum, we're making lower lows.
        # Note: i is relative to recent_close (last N candles), not the full DataFrame.
        for i in range(3, len(recent_close)):
            # Get minimum of the 3 candles before position i
            lookback_start = max(0, i - 3)
            window_min = recent_close.iloc[lookback_start:i].min()
            # Current candle must be above the local minimum
            if recent_close.iloc[i] <= window_min:
                higher_lows = False
                break

        if rsi_sustained and higher_lows:
            return True, "sustained_rsi_higher_lows"
        elif rsi_sustained:
            return True, "sustained_rsi"
        return False, ""

    def record_oversold_buy(self) -> None:
        """Record an oversold buy for rate limiting during crashes."""
        now = datetime.now(timezone.utc)
        self._oversold_buy_times.append(now)
        # Clean old entries
        cutoff = now - timedelta(hours=24)
        self._oversold_buy_times = [t for t in self._oversold_buy_times if t > cutoff]
        logger.debug("oversold_buy_recorded", count=len(self._oversold_buy_times))

    def can_buy_oversold(self) -> bool:
        """Check if we can make another oversold buy (max 2 per 24h)."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        recent_buys = [t for t in self._oversold_buy_times if t > cutoff]
        return len(recent_buys) < 2

    def is_price_stabilized(self, close_prices: pd.Series, window_candles: int = 12) -> bool:
        """
        Check if price has stopped falling (stabilized).

        Args:
            close_prices: Series of closing prices
            window_candles: Number of candles to check (default 12 = ~2 hours with 10-min candles)

        Returns:
            True if price is stable/recovering (not making new lows)
        """
        if len(close_prices) < window_candles:
            return True  # Not enough data, allow trade

        recent = close_prices.tail(window_candles)
        current = recent.iloc[-1]
        min_in_window = recent.min()

        # Stabilized if current price > minimum in window (not making new lows)
        # Must be strictly greater - if current == min, we just made a new low
        return current > min_in_window

    def calculate_score(
        self,
        df: pd.DataFrame,
        current_price: Optional[Decimal] = None,
        htf_bias: Optional[str] = None,
        htf_daily: Optional[str] = None,
        htf_4h: Optional[str] = None,
    ) -> SignalResult:
        """
        Calculate composite signal score from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            current_price: Current price (uses latest close if not provided)
            htf_bias: Combined HTF bias ("bullish", "bearish", "neutral", or None)
            htf_daily: Daily timeframe trend (for AI context)
            htf_4h: 4-hour timeframe trend (for AI context)

        Returns:
            SignalResult with score, action, and breakdown
        """
        if df.empty or len(df) < max(self.ema_slow_period, self.bollinger_period, 26):
            return SignalResult(
                score=0,
                action="hold",
                indicators=IndicatorValues(),
                breakdown={},
                confidence=0.0,
            )

        # Get price series
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        price = float(current_price) if current_price else close.iloc[-1]

        # Calculate all indicators
        rsi = calculate_rsi(close, self.rsi_period)
        macd_result = calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal_period)
        bollinger = calculate_bollinger_bands(close, self.bollinger_period, self.bollinger_std)
        ema_result = calculate_ema_crossover(close, self.ema_fast, self.ema_slow_period)
        atr_result = calculate_atr(high, low, close, self.atr_period)

        # Store current indicator values
        indicators = IndicatorValues(
            rsi=rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            macd_line=macd_result.macd_line.iloc[-1] if not pd.isna(macd_result.macd_line.iloc[-1]) else None,
            macd_signal=macd_result.signal_line.iloc[-1] if not pd.isna(macd_result.signal_line.iloc[-1]) else None,
            macd_histogram=macd_result.histogram.iloc[-1] if not pd.isna(macd_result.histogram.iloc[-1]) else None,
            bb_upper=bollinger.upper_band.iloc[-1] if not pd.isna(bollinger.upper_band.iloc[-1]) else None,
            bb_middle=bollinger.middle_band.iloc[-1] if not pd.isna(bollinger.middle_band.iloc[-1]) else None,
            bb_lower=bollinger.lower_band.iloc[-1] if not pd.isna(bollinger.lower_band.iloc[-1]) else None,
            ema_fast=ema_result.ema_fast.iloc[-1] if not pd.isna(ema_result.ema_fast.iloc[-1]) else None,
            ema_slow=ema_result.ema_slow.iloc[-1] if not pd.isna(ema_result.ema_slow.iloc[-1]) else None,
            atr=atr_result.atr.iloc[-1] if not pd.isna(atr_result.atr.iloc[-1]) else None,
            volatility=get_volatility_level(atr_result),
        )

        # Calculate individual scores
        breakdown = {}
        total_score = 0

        # RSI component (graduated: returns -1.0 to +1.0)
        rsi_signal = get_rsi_signal_graduated(indicators.rsi, self.rsi_oversold, self.rsi_overbought)
        rsi_score = int(rsi_signal * self.weights.rsi)

        # MACD component (graduated: returns -1.0 to +1.0, adaptive to candle interval)
        macd_signal = get_macd_signal_graduated(macd_result, price, self.candle_interval)
        macd_score = int(macd_signal * self.weights.macd)

        # Bollinger Bands component (graduated: returns -1.0 to +1.0)
        bb_signal = get_bollinger_signal_graduated(price, bollinger)
        bb_score = int(bb_signal * self.weights.bollinger)

        # EMA component (graduated: returns -1.0 to +1.0)
        ema_signal = get_ema_signal_graduated(ema_result)
        ema_score = int(ema_signal * self.weights.ema)

        # Momentum mode: reduce overbought penalties during sustained uptrends
        momentum_active, momentum_reason = self.is_momentum_mode(df, rsi)
        if momentum_active:
            original_rsi = rsi_score
            original_bb = bb_score
            # Reduce overbought penalties by configured factor (only negative scores)
            # Use int() instead of // for symmetric rounding behavior
            reduction = self.momentum_penalty_reduction
            if rsi_score < 0:
                rsi_score = int(rsi_score * reduction)
            if bb_score < 0:
                bb_score = int(bb_score * reduction)
            logger.info(
                "momentum_mode_active",
                reason=momentum_reason,
                rsi_original=original_rsi,
                rsi_adjusted=rsi_score,
                bb_original=original_bb,
                bb_adjusted=bb_score,
            )
        breakdown["_momentum_active"] = 1 if momentum_active else 0

        breakdown["rsi"] = rsi_score
        total_score += rsi_score
        breakdown["macd"] = macd_score
        total_score += macd_score
        breakdown["bollinger"] = bb_score
        total_score += bb_score
        breakdown["ema"] = ema_score
        total_score += ema_score

        # Store raw indicator values for signal history
        breakdown["_rsi_value"] = indicators.rsi
        breakdown["_macd_histogram"] = indicators.macd_histogram
        # Bollinger band position: 0 = at lower band, 1 = at upper band
        if indicators.bb_upper and indicators.bb_lower and indicators.bb_upper != indicators.bb_lower:
            breakdown["_bb_position"] = (price - float(indicators.bb_lower)) / (float(indicators.bb_upper) - float(indicators.bb_lower))
        else:
            breakdown["_bb_position"] = None
        # EMA gap percent
        if indicators.ema_fast and indicators.ema_slow and indicators.ema_slow != 0:
            breakdown["_ema_gap_percent"] = ((float(indicators.ema_fast) - float(indicators.ema_slow)) / float(indicators.ema_slow)) * 100
        else:
            breakdown["_ema_gap_percent"] = None

        # Volume confirmation (boost on high volume, penalty on low volume)
        # Includes whale activity detection for extreme volume spikes
        #
        # NOTE: Whale direction (bullish/bearish) is informational only.
        # The volume boost amplifies the existing signal direction from indicators,
        # not the whale direction. This is intentional - whale activity increases
        # conviction in whatever direction the indicators suggest, rather than
        # overriding the technical analysis.
        if volume is not None and len(volume) >= 20:
            volume_sma = volume.rolling(window=20).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            if not pd.isna(volume_sma) and volume_sma > 0:
                volume_ratio = round(current_volume / volume_sma, 2)

                if volume_ratio > self.whale_volume_threshold:
                    # WHALE ACTIVITY: Extreme volume spike (configurable threshold, default 3x)
                    # Apply configurable boost (default 30%) - stronger signal than normal high volume
                    # Note: On neutral signals (total_score=0), boost is 0 but _whale_activity
                    # is still set True. This is intentional - whale activity on neutral signals
                    # is valuable information for AI reviewers even without directional bias.
                    volume_boost = int(abs(total_score) * self.whale_boost_percent)
                    if total_score > 0:
                        breakdown["volume"] = volume_boost
                        total_score += volume_boost
                    elif total_score < 0:
                        breakdown["volume"] = -volume_boost
                        total_score -= volume_boost
                    else:
                        breakdown["volume"] = 0
                    breakdown["_whale_activity"] = True
                    breakdown["_volume_ratio"] = volume_ratio

                    # Determine whale direction based on price movement during volume spike
                    if len(close) >= 2:
                        prev_price = close.iloc[-2]
                        current_price = close.iloc[-1]
                        if prev_price > 0 and not pd.isna(current_price) and current_price > 0:
                            price_change_pct = (current_price - prev_price) / prev_price
                            breakdown["_price_change_pct"] = round(price_change_pct, 6)
                            if price_change_pct > self.whale_direction_threshold:
                                breakdown["_whale_direction"] = "bullish"
                            elif price_change_pct < -self.whale_direction_threshold:
                                breakdown["_whale_direction"] = "bearish"
                            else:
                                breakdown["_whale_direction"] = "neutral"
                        else:
                            # Zero/negative prev price - can't calculate direction
                            breakdown["_whale_direction"] = "unknown"
                            breakdown["_price_change_pct"] = None
                    else:
                        breakdown["_whale_direction"] = "unknown"
                        breakdown["_price_change_pct"] = None
                elif volume_ratio > 1.5:
                    # High volume: boost signal by configurable percentage (default 20%)
                    volume_boost = int(abs(total_score) * self.high_volume_boost_percent)
                    if total_score > 0:
                        breakdown["volume"] = volume_boost
                        total_score += volume_boost
                    elif total_score < 0:
                        breakdown["volume"] = -volume_boost
                        total_score -= volume_boost
                    else:
                        breakdown["volume"] = 0
                    breakdown["_whale_activity"] = False
                    breakdown["_volume_ratio"] = volume_ratio
                elif volume_ratio < 0.7:
                    # Low volume: fixed 10-point penalty (consistent behavior)
                    if total_score > 0:
                        breakdown["volume"] = -10
                        total_score -= 10
                    elif total_score < 0:
                        breakdown["volume"] = 10
                        total_score += 10
                    else:
                        breakdown["volume"] = 0
                    breakdown["_whale_activity"] = False
                    breakdown["_volume_ratio"] = volume_ratio
                else:
                    breakdown["volume"] = 0
                    breakdown["_whale_activity"] = False
                    breakdown["_volume_ratio"] = volume_ratio
            else:
                # Invalid volume SMA (NaN or zero)
                breakdown["volume"] = 0
                breakdown["_whale_activity"] = False
                breakdown["_volume_ratio"] = None
        else:
            # Insufficient volume data (< 20 candles)
            breakdown["volume"] = 0
            breakdown["_whale_activity"] = False
            breakdown["_volume_ratio"] = None

        # Log whale activity detection
        if breakdown.get("_whale_activity"):
            logger.info(
                "whale_activity_detected",
                volume_ratio=breakdown["_volume_ratio"],
                volume_boost=breakdown["volume"],
                whale_direction=breakdown.get("_whale_direction", "unknown"),
                signal_direction="bullish" if total_score > 0 else "bearish" if total_score < 0 else "neutral",
            )

        # Store raw score before adjustments (for signal history)
        breakdown["_raw_score"] = total_score

        # Trend filter: penalize counter-trend trades (scaled by signal strength)
        # Skip penalty for extreme RSI (mean-reversion zones) with crash protection
        trend = get_ema_trend(ema_result)
        trend_adjustment = 0
        rsi_extreme = indicators.rsi is not None and (indicators.rsi < 25 or indicators.rsi > 75)

        # Crash protection checks for mean-reversion trades
        can_mean_revert = self.can_buy_oversold()
        price_stable = self.is_price_stabilized(close)

        if rsi_extreme and can_mean_revert and price_stable:
            # Allow mean-reversion trade - all conditions met
            logger.debug("trend_filter_skipped", reason="extreme_rsi_stable", rsi=indicators.rsi)
        elif rsi_extreme and not can_mean_revert:
            # Hit buy limit - apply trend filter anyway
            logger.debug("trend_filter_applied", reason="oversold_buy_limit_reached", rsi=indicators.rsi)
            if total_score > 0 and trend == "bearish":
                signal_confidence = abs(total_score) / 100
                trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
                total_score += trend_adjustment
        elif rsi_extreme and not price_stable:
            # Price still falling - apply trend filter
            logger.debug("trend_filter_applied", reason="price_still_falling", rsi=indicators.rsi)
            if total_score > 0 and trend == "bearish":
                signal_confidence = abs(total_score) / 100
                trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
                total_score += trend_adjustment
        elif total_score > 0 and trend == "bearish":
            # Scale penalty: stronger signals get less penalty (10-20 points)
            signal_confidence = abs(total_score) / 100
            trend_adjustment = -int(20 * (1 - signal_confidence * 0.5))
            total_score += trend_adjustment
        elif total_score < 0 and trend == "bullish":
            # Scale penalty: stronger signals get less penalty
            signal_confidence = abs(total_score) / 100
            trend_adjustment = int(20 * (1 - signal_confidence * 0.5))
            total_score += trend_adjustment
        breakdown["trend_filter"] = trend_adjustment

        # HTF (Higher Timeframe) bias modifier
        # Purpose: Reduce false signals by aligning trades with the macro trend
        # - Daily + 6-hour trends must agree for strong bias, otherwise neutral
        # - Expected impact: 30-50% reduction in false signals
        #
        # Application logic (asymmetric for sell signals):
        # - Bullish signal (+score): +boost if HTF bullish, -penalty if HTF bearish
        # - Bearish signal (-score): -boost if HTF bearish (more negative = stronger sell),
        #                            +penalty if HTF bullish (less negative = weaker sell)
        htf_adjustment = 0
        if htf_bias and htf_bias != "neutral":
            # Strong HTF bias (daily + 4H agree): apply full adjustment
            if total_score > 0:  # Bullish signal (potential buy)
                if htf_bias == "bullish":
                    htf_adjustment = self.mtf_aligned_boost  # +20: stronger buy signal
                elif htf_bias == "bearish":
                    htf_adjustment = -self.mtf_counter_penalty  # -20: weaker buy signal
            elif total_score < 0:  # Bearish signal (potential sell)
                if htf_bias == "bearish":
                    # Aligned: make more negative to strengthen sell signal
                    htf_adjustment = -self.mtf_aligned_boost  # e.g., -60 → -80
                elif htf_bias == "bullish":
                    # Counter-trend: make less negative to weaken sell signal
                    htf_adjustment = self.mtf_counter_penalty  # e.g., -60 → -40
        elif htf_daily and htf_daily != "neutral":
            # This branch only runs when MTF_4H_ENABLED=true AND daily/4H disagree.
            # When 4H is disabled, htf_bias == htf_daily, so the first branch handles it.
            #
            # Here: htf_bias is "neutral" (daily + 4H disagree), but daily has direction.
            # Apply half penalty when daily trend opposes signal - daily is more reliable.
            # Use round() to handle odd mtf_counter_penalty values correctly.
            half_penalty = round(self.mtf_counter_penalty / 2)
            if total_score > 0 and htf_daily == "bearish":
                # Buying into bearish daily trend - apply half penalty
                htf_adjustment = -half_penalty
            elif total_score < 0 and htf_daily == "bullish":
                # Selling into bullish daily trend - weaken sell signal
                htf_adjustment = half_penalty

        if htf_adjustment != 0:
            total_score += htf_adjustment
            logger.info(
                "htf_bias_applied",
                htf_bias=htf_bias or "neutral",
                htf_daily=htf_daily,
                htf_4h=htf_4h,
                signal_direction="bullish" if total_score > 0 else "bearish",
                adjustment=htf_adjustment,
                partial_penalty=htf_bias == "neutral" or htf_bias is None,
            )

        breakdown["htf_bias"] = htf_adjustment
        breakdown["_htf_trend"] = htf_bias if htf_bias is not None else "disabled"
        breakdown["_htf_daily"] = htf_daily if htf_daily is not None else "disabled"
        breakdown["_htf_4h"] = htf_4h if htf_4h is not None else "disabled"

        # Clamp score to -100 to +100
        total_score = max(-100, min(100, total_score))

        # Determine action
        if total_score >= self.threshold:
            action = "buy"
        elif total_score <= -self.threshold:
            action = "sell"
        else:
            action = "hold"

        # Calculate confidence with confluence factor
        # Combines magnitude with how many indicators agree
        if action != "hold":
            # Count agreeing indicators (non-zero contributions, excluding metadata keys)
            confluence_count = sum(
                1 for key, score in breakdown.items()
                if score != 0 and not key.startswith("_")
            )
            confluence_factor = confluence_count / 7  # 7 components: rsi, macd, bollinger, ema, volume, trend_filter, htf_bias

            # Combine magnitude and confluence (equally weighted)
            magnitude_confidence = abs(total_score) / 100
            confidence = (magnitude_confidence + confluence_factor) / 2
            confidence = min(1.0, confidence)
        else:
            confidence = 0.0

        result = SignalResult(
            score=total_score,
            action=action,
            indicators=indicators,
            breakdown=breakdown,
            confidence=confidence,
        )

        logger.debug(
            "signal_calculated",
            score=total_score,
            action=action,
            breakdown=breakdown,
            confidence=confidence,
        )

        return result

    def get_trend(self, df: pd.DataFrame) -> str:
        """
        Get current market trend.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            "bullish", "bearish", or "neutral"
        """
        if df.empty or len(df) < self.ema_slow_period:
            return "neutral"

        close = df["close"].astype(float)
        ema_result = calculate_ema_crossover(close, self.ema_fast, self.ema_slow_period)

        return get_ema_trend(ema_result)
