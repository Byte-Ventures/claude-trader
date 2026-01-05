"""
Average Directional Index (ADX) indicator.

ADX measures trend strength (not direction) on a scale of 0-100.
It helps determine whether the market is trending or ranging.

Algorithm:
    ADX is calculated from the Directional Movement System:
    1. True Range (TR) = max(high-low, |high-prev_close|, |low-prev_close|)
    2. +DM = high - prev_high if positive and > -(low - prev_low), else 0
    3. -DM = prev_low - low if positive and > (high - prev_high), else 0
    4. Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    5. +DI = (Smoothed +DM / Smoothed TR) * 100
    6. -DI = (Smoothed -DM / Smoothed TR) * 100
    7. DX = |+DI - -DI| / (+DI + -DI) * 100
    8. ADX = Wilder's smoothed DX

Interpretation:
    ADX < 20:  Weak/absent trend (ranging market, signals unreliable)
    ADX 20-25: Trend emerging (caution)
    ADX 25-50: Strong trend (signals reliable)
    ADX 50-75: Very strong trend (may be exhausting)
    ADX > 75:  Extremely strong trend (rare, potential reversal)

    Note: ADX measures trend STRENGTH, not direction.
    +DI > -DI indicates uptrend, -DI > +DI indicates downtrend.

Uses:
    - Signal filtering: Reduce position size or skip trades when ADX < 20
    - Confidence scaling: Higher ADX = more confidence in momentum signals
    - Trend confirmation: Use with RSI/MACD to validate signals
    - Breakout detection: ADX rising from low values signals new trend

Parameters:
    - period: 14 (Wilder's original recommendation)

Integration:
    Used by SignalScorer as a confidence multiplier. When ADX is low,
    the final signal score is reduced to reflect lower reliability.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# ADX threshold constants
ADX_WEAK_TREND = 20.0  # Below this = ranging/choppy
ADX_MODERATE_TREND = 25.0  # Above this = confirmed trend
ADX_STRONG_TREND = 40.0  # Above this = strong trend
ADX_EXTREME_TREND = 75.0  # Above this = potentially exhausting


@dataclass
class ADXResult:
    """ADX calculation result with directional indicators."""

    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series
    dx: pd.Series


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> ADXResult:
    """
    Calculate Average Directional Index using Wilder's Smoothed Moving Average.

    Uses Wilder's original smoothing method (alpha = 1/period) for all
    smoothed components to match the original algorithm from his 1978 book.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ADX calculation period (default: 14, Wilder's recommendation)

    Returns:
        ADXResult with ADX, +DI, -DI, and DX series
    """
    if len(high) < period * 2:
        # Not enough data for reliable ADX calculation
        empty = pd.Series([np.nan] * len(high), index=high.index)
        return ADXResult(adx=empty, plus_di=empty, minus_di=empty, dx=empty)

    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM: up move is positive and greater than down move
    plus_dm = pd.Series(0.0, index=high.index)
    plus_dm_mask = (up_move > down_move) & (up_move > 0)
    plus_dm = plus_dm.mask(plus_dm_mask, up_move)

    # -DM: down move is positive and greater than up move
    minus_dm = pd.Series(0.0, index=high.index)
    minus_dm_mask = (down_move > up_move) & (down_move > 0)
    minus_dm = minus_dm.mask(minus_dm_mask, down_move)

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period

    # Smooth True Range, +DM, -DM
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Calculate Directional Indicators
    # Avoid division by zero
    plus_di = pd.Series(0.0, index=high.index)
    minus_di = pd.Series(0.0, index=high.index)

    valid_tr = smoothed_tr > 0
    plus_di = plus_di.mask(valid_tr, (smoothed_plus_dm / smoothed_tr) * 100)
    minus_di = minus_di.mask(valid_tr, (smoothed_minus_dm / smoothed_tr) * 100)

    # Calculate DX (Directional Movement Index)
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)

    dx = pd.Series(0.0, index=high.index)
    valid_di = di_sum > 0
    dx = dx.mask(valid_di, (di_diff / di_sum) * 100)

    # Smooth DX to get ADX
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return ADXResult(adx=adx, plus_di=plus_di, minus_di=minus_di, dx=dx)


def get_adx_value(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Get the current ADX value from OHLCV DataFrame.

    Convenience function for signal scorer integration.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ADX calculation period

    Returns:
        Current ADX value or None if insufficient data
    """
    if len(df) < period * 2:
        return None

    result = calculate_adx(df["high"], df["low"], df["close"], period)
    adx_value = result.adx.iloc[-1]

    if pd.isna(adx_value):
        return None

    return float(adx_value)


def get_adx_confidence_multiplier(
    adx: Optional[float],
    weak_threshold: float = ADX_WEAK_TREND,
    strong_threshold: float = ADX_MODERATE_TREND,
) -> float:
    """
    Get signal confidence multiplier based on ADX trend strength.

    This multiplier is applied to the final signal score to reduce
    conviction during ranging markets where momentum signals are unreliable.

    Args:
        adx: Current ADX value (0-100 scale)
        weak_threshold: ADX below this = weak trend (default: 20)
        strong_threshold: ADX above this = confirmed trend (default: 25)

    Returns:
        Confidence multiplier (0.5 to 1.1):
        - ADX < weak_threshold: 0.5 (halve signal - choppy market)
        - ADX weak_threshold to strong_threshold: 0.75 (reduce signal)
        - ADX strong_threshold to 40: 1.0 (full signal)
        - ADX > 40: 1.1 (boost signal - strong trend)

    Examples:
        >>> get_adx_confidence_multiplier(15)  # Weak trend
        0.5
        >>> get_adx_confidence_multiplier(22)  # Emerging trend
        0.75
        >>> get_adx_confidence_multiplier(30)  # Confirmed trend
        1.0
        >>> get_adx_confidence_multiplier(50)  # Strong trend
        1.1
    """
    if adx is None:
        # No ADX data available, use neutral multiplier
        return 1.0

    if adx < weak_threshold:
        # Weak/absent trend - halve signal strength
        multiplier = 0.5
    elif adx < strong_threshold:
        # Trend emerging - reduce signal
        multiplier = 0.75
    elif adx <= ADX_STRONG_TREND:
        # Confirmed trend - full signal
        multiplier = 1.0
    else:
        # Strong trend - boost signal slightly
        multiplier = 1.1

    logger.debug(
        "adx_confidence_multiplier",
        adx=round(adx, 2),
        multiplier=multiplier,
        trend_strength=(
            "weak" if adx < weak_threshold
            else "emerging" if adx < strong_threshold
            else "strong" if adx > ADX_STRONG_TREND
            else "confirmed"
        ),
    )

    return multiplier


def get_trend_direction(plus_di: float, minus_di: float) -> str:
    """
    Determine trend direction from directional indicators.

    Args:
        plus_di: +DI value
        minus_di: -DI value

    Returns:
        "bullish" if +DI > -DI, "bearish" if -DI > +DI, "neutral" if equal
    """
    if plus_di > minus_di:
        return "bullish"
    elif minus_di > plus_di:
        return "bearish"
    else:
        return "neutral"


def classify_trend_strength(adx: Optional[float]) -> str:
    """
    Classify trend strength for display/logging.

    Args:
        adx: ADX value (0-100)

    Returns:
        Human-readable classification string
    """
    if adx is None:
        return "unknown"
    if adx < ADX_WEAK_TREND:
        return "weak"
    if adx < ADX_MODERATE_TREND:
        return "emerging"
    if adx < ADX_STRONG_TREND:
        return "moderate"
    if adx < ADX_EXTREME_TREND:
        return "strong"
    return "extreme"
