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

    This scaling means that a histogram of 0.5% of price produces a
    full ±1.0 signal, appropriate for typical crypto volatility.

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
import pandas as pd
import numpy as np

# Signal generation constants
_HISTOGRAM_SCALE_FACTOR = 200  # Scales histogram to 0.5% of price = full signal
_HISTOGRAM_DEAD_ZONE = 0.1  # Minimum normalized histogram for signal
_BASE_SIGNAL_CLAMP = 0.8  # Max base signal before relationship boost
_RELATIONSHIP_BOOST = 0.2  # Boost for MACD/signal line relationship


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


def get_macd_signal_graduated(macd_result: MACDResult, price: float) -> float:
    """
    Get graduated trading signal from MACD (-1.0 to +1.0).

    Uses histogram normalized by price to determine signal strength:
    - Positive histogram = bullish (0 to +1.0)
    - Negative histogram = bearish (0 to -1.0)
    - MACD above signal line adds +0.2 boost
    - MACD below signal line adds -0.2 boost

    The histogram is scaled so that 0.5% of price produces a full ±1.0 signal.
    A dead zone filters out noise when histogram is very small.

    Args:
        macd_result: MACD calculation result
        price: Current price (used for normalization)

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

    # Normalize histogram by price (0.5% of price = 1.0 signal)
    hist_normalized = (histogram / price) * _HISTOGRAM_SCALE_FACTOR

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
