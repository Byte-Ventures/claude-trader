"""
Moving Average Convergence Divergence (MACD) indicator.

MACD shows the relationship between two exponential moving averages:
- MACD Line: Fast EMA - Slow EMA
- Signal Line: EMA of MACD Line
- Histogram: MACD Line - Signal Line

Trading signals:
- MACD crosses above signal: Buy
- MACD crosses below signal: Sell
- Histogram turning positive: Bullish momentum
- Histogram turning negative: Bearish momentum
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


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


def get_macd_signal(macd_result: MACDResult) -> int:
    """
    Get trading signal from MACD.

    Args:
        macd_result: MACD calculation result

    Returns:
        +1 for buy signal, -1 for sell signal, 0 for neutral
    """
    if len(macd_result.macd_line) < 2:
        return 0

    # Get current and previous values
    macd_current = macd_result.macd_line.iloc[-1]
    macd_prev = macd_result.macd_line.iloc[-2]
    signal_current = macd_result.signal_line.iloc[-1]
    signal_prev = macd_result.signal_line.iloc[-2]

    if pd.isna(macd_current) or pd.isna(signal_current):
        return 0

    # Check for crossover
    # Buy: MACD crosses above signal
    if macd_prev <= signal_prev and macd_current > signal_current:
        return 1

    # Sell: MACD crosses below signal
    if macd_prev >= signal_prev and macd_current < signal_current:
        return -1

    return 0


def get_macd_histogram_signal(macd_result: MACDResult) -> int:
    """
    Get trading signal from MACD histogram.

    Args:
        macd_result: MACD calculation result

    Returns:
        +1 for bullish, -1 for bearish, 0 for neutral
    """
    if len(macd_result.histogram) < 2:
        return 0

    hist_current = macd_result.histogram.iloc[-1]
    hist_prev = macd_result.histogram.iloc[-2]

    if pd.isna(hist_current) or pd.isna(hist_prev):
        return 0

    # Histogram crosses above zero
    if hist_prev <= 0 and hist_current > 0:
        return 1

    # Histogram crosses below zero
    if hist_prev >= 0 and hist_current < 0:
        return -1

    # Histogram increasing (bullish momentum)
    if hist_current > 0 and hist_current > hist_prev:
        return 1

    # Histogram decreasing (bearish momentum)
    if hist_current < 0 and hist_current < hist_prev:
        return -1

    return 0


def is_macd_converging(macd_result: MACDResult, lookback: int = 5) -> bool:
    """
    Check if MACD and signal lines are converging.

    Args:
        macd_result: MACD calculation result
        lookback: Number of periods to analyze

    Returns:
        True if lines are converging
    """
    if len(macd_result.histogram) < lookback:
        return False

    recent_hist = macd_result.histogram.tail(lookback).abs()

    # Lines are converging if histogram is getting smaller
    return recent_hist.iloc[-1] < recent_hist.iloc[0]
