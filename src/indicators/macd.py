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

_HISTOGRAM_DEAD_ZONE = 0.1  # Minimum normalized histogram for signal
_BASE_SIGNAL_CLAMP = 0.8  # Max base signal before relationship boost
_RELATIONSHIP_BOOST = 0.2  # Boost for MACD/signal line relationship


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
    return _HISTOGRAM_SCALE_FACTORS.get(candle_interval, _DEFAULT_HISTOGRAM_SCALE_FACTOR)


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
) -> float:
    """
    Get graduated trading signal from MACD (-1.0 to +1.0).

    Uses histogram normalized by price to determine signal strength:
    - Positive histogram = bullish (0 to +1.0)
    - Negative histogram = bearish (0 to -1.0)
    - MACD above signal line adds +0.2 boost
    - MACD below signal line adds -0.2 boost

    The scale factor is ADAPTIVE to candle interval:
    - Shorter candles (1-15 min): Higher scale factors for smaller price moves
    - Longer candles (6h-1d): Lower scale factors for larger price moves

    A dead zone filters out noise when histogram is very small.

    Args:
        macd_result: MACD calculation result
        price: Current price (used for normalization)
        candle_interval: Optional candle interval for adaptive scaling
                        (e.g., "FIFTEEN_MINUTE", "ONE_HOUR")

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

    # Get adaptive scale factor based on candle interval
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
