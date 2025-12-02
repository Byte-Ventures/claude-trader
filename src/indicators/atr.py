"""
Average True Range (ATR) indicator.

ATR measures market volatility by calculating the average of true ranges.
Higher ATR indicates higher volatility, lower ATR indicates lower volatility.

Algorithm:
    True Range (TR) = max(high-low, |high-prev_close|, |low-prev_close|)

    This implementation uses Wilder's Smoothed Moving Average (SMMA) for the
    ATR calculation, which is the standard method. Wilder's smoothing uses
    alpha = 1/period rather than standard EMA's alpha = 2/(period+1).

    Wilder's SMMA formula:
        SMMA(i) = ((SMMA(i-1) * (period - 1)) + current_TR) / period

    This is equivalent to EWM with alpha = 1/period, providing a smoother
    result than standard EMA and matching the original formula from Welles
    Wilder's 1978 book "New Concepts in Technical Trading Systems".

Uses:
    - Position sizing: Higher ATR = smaller position (volatility-adjusted)
    - Stop-loss placement: Stop at entry - (ATR * multiplier)
    - Take-profit placement: Target at entry + (ATR * multiplier)
    - Trailing stop distance: Trail by ATR * multiplier
    - Volatility filtering: Avoid trades during extreme volatility

Volatility Classification:
    Based on percentile ranking of current ATR vs recent history:
    - Low (<25th percentile): Quiet market, larger positions
    - Normal (25th-75th percentile): Standard conditions
    - High (75th-90th percentile): Elevated volatility, smaller positions
    - Extreme (>90th percentile): Crisis conditions, minimal exposure

Parameters:
    - period: 14 (Wilder's original recommendation)

Integration:
    Used by SignalScorer for volatility assessment. Also used by runner.py
    for ATR-based trailing stops and hard stops.
"""

from dataclasses import dataclass
from decimal import Decimal
import pandas as pd
import numpy as np


@dataclass
class ATRResult:
    """ATR calculation result."""

    atr: pd.Series
    true_range: pd.Series


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> ATRResult:
    """
    Calculate Average True Range using Wilder's Smoothed Moving Average.

    Uses Wilder's original smoothing method (alpha = 1/period) rather than
    standard EMA (alpha = 2/(period+1)) for more stable signals.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR calculation period (default: 14, Wilder's recommendation)

    Returns:
        ATRResult with ATR and true range series
    """
    # Previous close
    prev_close = close.shift(1)

    # True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use Wilder's smoothing: alpha = 1/period
    # This is equivalent to: SMMA(i) = ((SMMA(i-1) * (period-1)) + current) / period
    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()

    return ATRResult(atr=atr, true_range=true_range)


def get_atr_stop_loss(
    entry_price: Decimal,
    atr_value: float,
    multiplier: float = 1.5,
    side: str = "buy",
) -> Decimal:
    """
    Calculate stop-loss price based on ATR.

    Args:
        entry_price: Trade entry price
        atr_value: Current ATR value
        multiplier: ATR multiplier for distance (default: 1.5)
        side: Trade side ("buy" or "sell")

    Returns:
        Stop-loss price
    """
    atr_distance = Decimal(str(atr_value * multiplier))

    if side == "buy":
        return entry_price - atr_distance
    else:  # sell
        return entry_price + atr_distance


def get_atr_take_profit(
    entry_price: Decimal,
    atr_value: float,
    multiplier: float = 2.0,
    side: str = "buy",
) -> Decimal:
    """
    Calculate take-profit price based on ATR.

    Args:
        entry_price: Trade entry price
        atr_value: Current ATR value
        multiplier: ATR multiplier for distance (default: 2.0)
        side: Trade side ("buy" or "sell")

    Returns:
        Take-profit price
    """
    atr_distance = Decimal(str(atr_value * multiplier))

    if side == "buy":
        return entry_price + atr_distance
    else:  # sell
        return entry_price - atr_distance


def get_volatility_level(
    atr_result: ATRResult,
    lookback: int = 50,
) -> str:
    """
    Assess current volatility level based on ATR percentile.

    Compares current ATR to recent history to classify volatility:
    - low: ATR below 25th percentile (quiet market)
    - normal: ATR between 25th-75th percentile
    - high: ATR between 75th-90th percentile
    - extreme: ATR above 90th percentile (crisis conditions)

    Args:
        atr_result: ATR calculation result
        lookback: Number of periods for comparison (default: 50)

    Returns:
        "low", "normal", "high", or "extreme"
    """
    if len(atr_result.atr) < lookback:
        return "normal"

    recent_atr = atr_result.atr.tail(lookback)
    current_atr = atr_result.atr.iloc[-1]

    if pd.isna(current_atr):
        return "normal"

    # Calculate percentile
    percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100

    if percentile < 25:
        return "low"
    elif percentile < 75:
        return "normal"
    elif percentile < 90:
        return "high"
    else:
        return "extreme"


def get_position_size_multiplier(
    atr_result: ATRResult,
    lookback: int = 50,
) -> float:
    """
    Calculate position size multiplier based on volatility.

    Higher volatility = smaller position size to maintain consistent risk.

    Multipliers:
    - low volatility: 1.2x (larger position, quiet market)
    - normal volatility: 1.0x (standard position)
    - high volatility: 0.7x (reduced position)
    - extreme volatility: 0.4x (minimal position)

    Args:
        atr_result: ATR calculation result
        lookback: Number of periods for comparison (default: 50)

    Returns:
        Multiplier between 0.4 and 1.2
    """
    volatility = get_volatility_level(atr_result, lookback)

    multipliers = {
        "low": 1.2,      # Low volatility: larger position
        "normal": 1.0,   # Normal: standard position
        "high": 0.7,     # High volatility: reduced position
        "extreme": 0.4,  # Extreme: minimal position
    }

    return multipliers.get(volatility, 1.0)


def calculate_atr_percent(
    atr_result: ATRResult,
    close: pd.Series,
) -> pd.Series:
    """
    Calculate ATR as percentage of price.

    Useful for comparing volatility across different price levels and
    different assets.

    Args:
        atr_result: ATR calculation result
        close: Series of closing prices

    Returns:
        ATR as percentage of price (e.g., 2.5 means ATR is 2.5% of price)
    """
    return (atr_result.atr / close) * 100
