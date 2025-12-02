"""
Exponential Moving Average (EMA) crossover indicator.

EMA gives more weight to recent prices, making it more responsive than SMA.
This implementation uses a dual EMA system for trend detection and signals.

Algorithm:
    EMA = price * alpha + EMA_prev * (1 - alpha)
    where alpha = 2 / (period + 1)

    This is the standard EMA calculation used in most trading platforms.
    Unlike Wilder's smoothing (used in RSI/ATR), standard EMA gives more
    weight to recent prices.

Signal Generation:
    The graduated signal function combines crossovers with gap momentum:

    1. Fresh crossover: +/- 1.0 (strongest signal)
       - Fast EMA just crossed above slow EMA: +1.0
       - Fast EMA just crossed below slow EMA: -1.0

    2. Gap-based signal: Uses percentage difference between fast and slow EMA
       - Gap > 0.3% and widening: +/- 0.4 to 0.8 (trend strengthening)
       - Gap > 0.3% but narrowing: +/- 0.2 to 0.4 (trend weakening)
       - Gap < 0.3%: 0.0 (dead zone - EMAs too close)

    The gap momentum factor reduces the signal when the trend is weakening
    (gap narrowing), which helps avoid late entries in exhausted trends.

Trend Classification:
    Based on percentage gap between fast and slow EMA:
    - > 1.0%: Bullish trend
    - < -1.0%: Bearish trend
    - -1.0% to 1.0%: Neutral (no strong trend)

Parameters:
    - fast_period: 9 (more responsive to price changes)
    - slow_period: 21 (smoother, represents medium-term trend)

Integration:
    Used by SignalScorer with 15% weight. Also used for trend filtering
    to penalize counter-trend trades.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

# Signal generation constants
_GAP_DEAD_ZONE_PERCENT = 0.3  # Minimum gap % for signal
_TREND_THRESHOLD_PERCENT = 1.0  # Gap % for bullish/bearish trend classification
_MAX_GAP_SIGNAL = 0.8  # Maximum signal from gap (before momentum adjustment)
_GAP_WEAKENING_FACTOR = 0.5  # Signal reduction when gap is narrowing


@dataclass
class EMAResult:
    """EMA calculation result."""

    ema_fast: pd.Series
    ema_slow: pd.Series
    crossover_up: pd.Series  # Boolean: fast crosses above slow
    crossover_down: pd.Series  # Boolean: fast crosses below slow


def calculate_ema(
    prices: pd.Series,
    period: int,
) -> pd.Series:
    """
    Calculate single EMA.

    Uses standard EMA formula: alpha = 2 / (period + 1)

    Args:
        prices: Series of closing prices
        period: EMA period

    Returns:
        EMA series
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_ema_crossover(
    prices: pd.Series,
    fast_period: int = 9,
    slow_period: int = 21,
) -> EMAResult:
    """
    Calculate EMA crossover system.

    Computes both EMAs and detects crossover points where the fast EMA
    crosses above or below the slow EMA.

    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 9)
        slow_period: Slow EMA period (default: 21)

    Returns:
        EMAResult with both EMAs and crossover signals
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    # Detect crossovers using numpy to avoid pandas dtype issues
    fast_above_slow = (ema_fast > ema_slow).to_numpy().astype(bool)
    fast_above_slow_prev = np.concatenate([[False], fast_above_slow[:-1]])

    # Crossover up: fast crosses above slow
    crossover_up = pd.Series(fast_above_slow & ~fast_above_slow_prev, index=ema_fast.index)

    # Crossover down: fast crosses below slow
    crossover_down = pd.Series(~fast_above_slow & fast_above_slow_prev, index=ema_fast.index)

    return EMAResult(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        crossover_up=crossover_up.fillna(False),
        crossover_down=crossover_down.fillna(False),
    )


def get_ema_signal_graduated(ema_result: EMAResult) -> float:
    """
    Get graduated trading signal from EMA position and momentum.

    Returns continuous signal based on:
    1. Fresh crossover: +/- 1.0 (strongest signal)
    2. Position + momentum: scaled by gap size and trend direction

    Zones:
    - Fresh crossover: +/- 1.0
    - Gap > 0.3% and widening: +/- 0.4 to 0.8 (scaled by gap)
    - Gap > 0.3% but narrowing: +/- 0.2 to 0.4 (reduced for weakening)
    - Gap < 0.3%: 0.0 (dead zone)

    The momentum factor (gap widening vs narrowing) helps distinguish
    between strengthening and weakening trends.

    Args:
        ema_result: EMA calculation result

    Returns:
        Float from -1.0 to +1.0
    """
    if len(ema_result.ema_fast) < 2:
        return 0.0

    fast = ema_result.ema_fast.iloc[-1]
    slow = ema_result.ema_slow.iloc[-1]

    if pd.isna(fast) or pd.isna(slow):
        return 0.0

    # Fresh crossover: strong signal
    if ema_result.crossover_up.iloc[-1]:
        return 1.0
    if ema_result.crossover_down.iloc[-1]:
        return -1.0

    # Calculate gap and momentum
    fast_prev = ema_result.ema_fast.iloc[-2]
    slow_prev = ema_result.ema_slow.iloc[-2]

    if pd.isna(fast_prev) or pd.isna(slow_prev) or slow == 0 or slow_prev == 0:
        return 0.0

    current_gap = (fast - slow) / slow * 100  # percentage gap
    prev_gap = (fast_prev - slow_prev) / slow_prev * 100
    gap_widening = abs(current_gap) > abs(prev_gap)

    # Dead zone: gap below threshold
    if abs(current_gap) < _GAP_DEAD_ZONE_PERCENT:
        return 0.0

    # Bullish: fast > slow
    if current_gap > 0:
        base = min(_MAX_GAP_SIGNAL, current_gap / 2)  # scale by gap size
        return base if gap_widening else base * _GAP_WEAKENING_FACTOR

    # Bearish: fast < slow
    if current_gap < 0:
        base = max(-_MAX_GAP_SIGNAL, current_gap / 2)
        return base if gap_widening else base * _GAP_WEAKENING_FACTOR

    return 0.0


def get_ema_trend(ema_result: EMAResult) -> str:
    """
    Determine current trend based on EMA positions.

    Uses the percentage gap between fast and slow EMA to classify trend:
    - > 1.0%: Bullish
    - < -1.0%: Bearish
    - Otherwise: Neutral

    Args:
        ema_result: EMA calculation result

    Returns:
        "bullish", "bearish", or "neutral"
    """
    if len(ema_result.ema_fast) == 0:
        return "neutral"

    fast = ema_result.ema_fast.iloc[-1]
    slow = ema_result.ema_slow.iloc[-1]

    if pd.isna(fast) or pd.isna(slow) or slow == 0:
        return "neutral"

    diff_percent = (fast - slow) / slow * 100

    if diff_percent > _TREND_THRESHOLD_PERCENT:
        return "bullish"
    elif diff_percent < -_TREND_THRESHOLD_PERCENT:
        return "bearish"

    return "neutral"


def get_ema_trend_from_values(ema_fast: float, ema_slow: float) -> str:
    """
    Determine current trend from raw EMA values.

    Convenience function when you already have the EMA values and don't
    need to work with the full EMAResult object.

    Args:
        ema_fast: Current fast EMA value
        ema_slow: Current slow EMA value

    Returns:
        "bullish", "bearish", or "neutral"
    """
    if ema_slow == 0:
        return "neutral"

    diff_percent = (ema_fast - ema_slow) / ema_slow * 100

    if diff_percent > _TREND_THRESHOLD_PERCENT:
        return "bullish"
    elif diff_percent < -_TREND_THRESHOLD_PERCENT:
        return "bearish"

    return "neutral"
