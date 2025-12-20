"""
Moving Average Convergence Divergence (MACD) indicator.

MACD shows the relationship between two exponential moving averages and is used
to identify momentum, trend direction, and potential reversal points.

Algorithm:
    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line

    This implementation uses standard EMA (alpha = 2/(period+1)) which is
    the conventional method for MACD calculation, as opposed to Wilder's
    smoothing used in RSI and ATR.

Signal Generation:
    The graduated signal function uses the histogram normalized by price:
    - Positive histogram = bullish signal (0 to +1.0)
    - Negative histogram = bearish signal (0 to -1.0)
    - MACD above/below signal line adds ±0.2 boost

    Normalization formula:
        signal = (histogram / price) * HISTOGRAM_SCALE_FACTOR

    The scale factor is ADAPTIVE to candle interval:
    - Shorter candles (1-15 min) have smaller price moves, need higher scale factors
    - Longer candles (6h-1d) have larger price moves, need lower scale factors

    This ensures MACD signals are appropriately weighted regardless of timeframe.

    Dead zone: Very small histogram values (< 0.1 normalized) return 0
    to avoid noise in the signal.

Parameters:
    - fast_period: 12 (standard)
    - slow_period: 26 (standard)
    - signal_period: 9 (standard)

Integration:
    Used by SignalScorer with 25% weight. Combined with RSI, Bollinger Bands,
    EMA crossover, and volume for confluence-based trading signals.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import structlog

from src.indicators.atr import ATRResult, calculate_atr_percent

logger = structlog.get_logger(__name__)

# Valid candle intervals (shared across adaptive functions)
VALID_CANDLE_INTERVALS = {
    "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
    "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR",
    "SIX_HOUR", "ONE_DAY"
}

# Adaptive scale factors by candle interval
# Shorter candles have smaller moves, need higher scale factors to produce signals
# Longer candles have larger moves, need lower scale factors
_HISTOGRAM_SCALE_FACTORS = {
    "ONE_MINUTE": 400,      # 0.25% of price = full signal
    "FIVE_MINUTE": 300,     # 0.33% of price = full signal
    "FIFTEEN_MINUTE": 200,  # 0.5% of price = full signal (default)
    "THIRTY_MINUTE": 175,   # 0.57% of price = full signal
    "ONE_HOUR": 150,        # 0.67% of price = full signal
    "TWO_HOUR": 125,        # 0.8% of price = full signal
    "SIX_HOUR": 100,        # 1.0% of price = full signal
    "ONE_DAY": 75,          # 1.33% of price = full signal
}
_DEFAULT_HISTOGRAM_SCALE_FACTOR = 200  # Fallback for unknown intervals

# Interval multipliers for dynamic scaling
# Shorter intervals need higher sensitivity (larger multipliers)
# Longer intervals need lower sensitivity (smaller multipliers)
#
# These multipliers are calibrated based on the observation that shorter timeframes
# produce noisier signals and need higher sensitivity to capture meaningful moves,
# while longer timeframes have clearer trends and need lower sensitivity to avoid
# overtrading. The values are proportionally scaled from the 15-minute baseline.
#
# Note: These values should be validated through backtesting across different assets
# and market conditions. Consider making them configurable parameters if optimization
# shows significant performance variations.
_INTERVAL_MULTIPLIERS = {
    "ONE_MINUTE": 2.0,      # Highest sensitivity for 1-minute candles
    "FIVE_MINUTE": 1.5,     # High sensitivity
    "FIFTEEN_MINUTE": 1.0,  # Baseline (default)
    "THIRTY_MINUTE": 0.875, # Slightly reduced
    "ONE_HOUR": 0.75,       # Reduced sensitivity
    "TWO_HOUR": 0.625,      # Lower sensitivity
    "SIX_HOUR": 0.5,        # Much lower sensitivity
    "ONE_DAY": 0.375,       # Lowest sensitivity for daily candles
}
_DEFAULT_INTERVAL_MULTIPLIER = 1.0  # Fallback for unknown intervals

_HISTOGRAM_DEAD_ZONE = 0.1  # Minimum normalized histogram for signal
_BASE_SIGNAL_CLAMP = 0.8  # Max base signal before relationship boost
_RELATIONSHIP_BOOST = 0.2  # Boost for MACD/signal line relationship
_MAX_SCALE_FACTOR = 1000  # Cap prevents oversensitivity in <0.05% ATR markets (stablecoins, extreme low liquidity)
_MIN_ATR_PERCENT = 0.1  # Minimum ATR (0.1% of price) to prevent extreme scaling in ultra-low volatility


def get_histogram_scale_factor(candle_interval: Optional[str] = None) -> float:
    """
    Get the appropriate histogram scale factor for the given candle interval.

    Shorter candles have smaller price movements, so they need higher scale
    factors to produce meaningful signals. Longer candles have larger movements
    and need lower scale factors.

    Args:
        candle_interval: Candle interval string (e.g., "FIFTEEN_MINUTE", "ONE_HOUR").
                        If None, returns the default scale factor.

    Returns:
        Scale factor for histogram normalization.
    """
    if candle_interval is None:
        return _DEFAULT_HISTOGRAM_SCALE_FACTOR
    if candle_interval not in VALID_CANDLE_INTERVALS:
        logger.warning(
            "invalid_candle_interval",
            interval=candle_interval,
            using="default",
            valid_intervals=list(VALID_CANDLE_INTERVALS)
        )
    return _HISTOGRAM_SCALE_FACTORS.get(candle_interval, _DEFAULT_HISTOGRAM_SCALE_FACTOR)


def get_dynamic_histogram_scale(
    atr_result: ATRResult,
    close: pd.Series,
    candle_interval: Optional[str] = None,
    interval_multipliers: Optional[dict[str, float]] = None,
) -> float:
    """
    Calculate dynamic histogram scale factor based on actual market volatility.

    This function makes the MACD indicator robust to different assets and
    volatility regimes by using ATR instead of hardcoded scale factors.

    The base scale is calculated so that a histogram equal to 2x ATR produces
    a full signal. This is then adjusted by an interval multiplier to account
    for the fact that shorter candles need higher sensitivity.

    Args:
        atr_result: ATR calculation result (used to measure current volatility)
        close: Series of closing prices (used to calculate ATR as % of price)
        candle_interval: Candle interval for baseline adjustment
                        (e.g., "FIFTEEN_MINUTE", "ONE_HOUR")
        interval_multipliers: Optional custom interval multipliers from config.
                             If None, uses hardcoded defaults (_INTERVAL_MULTIPLIERS)

    Returns:
        Dynamic scale factor for histogram normalization.
    """
    # Calculate ATR as percentage of price
    atr_percent_series = calculate_atr_percent(atr_result, close)

    # ATR needs at least 14 periods to be reliable (default ATR period)
    # Fallback to static scale factor if ATR has insufficient data or is unavailable
    if len(atr_percent_series) < 14:
        return get_histogram_scale_factor(candle_interval)

    # Enhanced validation: Check data quality across recent ATR values
    # ATR uses 14-period EMA, so validate last 14 values for reliability
    recent_atr = atr_percent_series.tail(14)
    if recent_atr.isna().any() or (recent_atr <= 0).any():
        logger.warning(
            "atr_data_quality_insufficient",
            nan_count=int(recent_atr.isna().sum()),
            zero_or_negative_count=int((recent_atr <= 0).sum()),
            using="static_scale_factor",
            candle_interval=candle_interval,
        )
        return get_histogram_scale_factor(candle_interval)

    atr_percent = recent_atr.iloc[-1]

    # Prevent extreme scaling in ultra-low volatility markets (stablecoins, low liquidity)
    # When ATR < 0.1%, fallback to static scaling to avoid oversensitivity
    if atr_percent < _MIN_ATR_PERCENT:
        logger.warning(
            "atr_too_low_for_dynamic_scaling",
            atr_percent=round(atr_percent, 4),
            min_threshold=_MIN_ATR_PERCENT,
            using="static_scale_factor",
            candle_interval=candle_interval,
        )
        return get_histogram_scale_factor(candle_interval)

    # Base calculation: full signal when histogram = 2x ATR
    # If ATR is 2% of price, then histogram of 4% produces full signal
    # So scale factor = 100 / (2 * atr_percent) = 50 / atr_percent
    base_scale = 50.0 / atr_percent

    # Apply interval adjustment (shorter intervals need higher sensitivity)
    # Use custom multipliers from config if provided, otherwise use defaults
    multipliers_to_use = interval_multipliers if interval_multipliers is not None else _INTERVAL_MULTIPLIERS
    interval_multiplier = multipliers_to_use.get(
        candle_interval or "FIFTEEN_MINUTE",
        _DEFAULT_INTERVAL_MULTIPLIER
    )

    dynamic_scale = base_scale * interval_multiplier

    # Cap the scale factor to prevent oversensitivity in extremely low volatility markets
    dynamic_scale = min(dynamic_scale, _MAX_SCALE_FACTOR)

    logger.debug(
        "dynamic_macd_scale",
        atr_percent=round(atr_percent, 3),
        base_scale=round(base_scale, 1),
        interval_multiplier=interval_multiplier,
        dynamic_scale=round(dynamic_scale, 1),
        candle_interval=candle_interval,
    )

    return dynamic_scale


@dataclass
class MACDResult:
    """MACD calculation result."""

    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """
    Calculate MACD indicator.

    Uses standard EMA calculation (alpha = 2/(period+1)) which is the
    conventional method for MACD.

    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        MACDResult with macd_line, signal_line, and histogram
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
    )


def get_macd_signal_graduated(
    macd_result: MACDResult,
    price: float,
    candle_interval: Optional[str] = None,
    atr_result: Optional[ATRResult] = None,
    close: Optional[pd.Series] = None,
    interval_multipliers: Optional[dict[str, float]] = None,
) -> float:
    """
    Get graduated trading signal from MACD (-1.0 to +1.0).

    Uses histogram normalized by price to determine signal strength:
    - Positive histogram = bullish (0 to +1.0)
    - Negative histogram = bearish (0 to -1.0)
    - MACD above signal line adds +0.2 boost
    - MACD below signal line adds -0.2 boost

    The scale factor is ADAPTIVE and can use two methods:

    1. DYNAMIC (preferred): Uses ATR-based volatility calculation
       - Robust to different assets and volatility regimes
       - Automatically adjusts to market conditions
       - Requires atr_result and close parameters

    2. STATIC (fallback): Uses hardcoded interval-based scale factors
       - Calibrated for typical BTC volatility patterns
       - Used when ATR data is not available

    A dead zone filters out noise when histogram is very small.

    Args:
        macd_result: MACD calculation result
        price: Current price (used for normalization)
        candle_interval: Optional candle interval for adaptive scaling
                        (e.g., "FIFTEEN_MINUTE", "ONE_HOUR")
        atr_result: Optional ATR result for dynamic volatility-based scaling
        close: Optional close price series (required if using atr_result)
        interval_multipliers: Optional custom interval multipliers from config.
                             If None, uses hardcoded defaults

    Returns:
        Float from -1.0 to +1.0
    """
    if len(macd_result.histogram) < 1 or price <= 0:
        return 0.0

    histogram = macd_result.histogram.iloc[-1]
    macd_line = macd_result.macd_line.iloc[-1]
    signal_line = macd_result.signal_line.iloc[-1]

    if pd.isna(histogram) or pd.isna(macd_line) or pd.isna(signal_line):
        return 0.0

    # Get scale factor: prefer dynamic (ATR-based) over static (interval-based)
    if atr_result is not None and close is not None:
        scale_factor = get_dynamic_histogram_scale(atr_result, close, candle_interval, interval_multipliers)
    else:
        scale_factor = get_histogram_scale_factor(candle_interval)

    # Normalize histogram by price using adaptive scale factor
    hist_normalized = (histogram / price) * scale_factor

    # Dead zone: very small histogram relative to price
    if abs(hist_normalized) < _HISTOGRAM_DEAD_ZONE:
        return 0.0

    # Base signal from histogram (clamped to ±0.8)
    base_signal = max(-_BASE_SIGNAL_CLAMP, min(_BASE_SIGNAL_CLAMP, hist_normalized))

    # Boost based on MACD/signal relationship (±0.2)
    if macd_line > signal_line:
        relationship_boost = _RELATIONSHIP_BOOST
    elif macd_line < signal_line:
        relationship_boost = -_RELATIONSHIP_BOOST
    else:
        relationship_boost = 0.0

    # Combine and clamp to -1.0 to +1.0
    total_signal = base_signal + relationship_boost
    return max(-1.0, min(1.0, total_signal))
