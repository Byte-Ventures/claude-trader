"""
Relative Strength Index (RSI) indicator.

RSI measures momentum by comparing the magnitude of recent gains to recent losses.
Values range from 0 to 100:
- < 30: Oversold (potential buy signal)
- > 70: Overbought (potential sell signal)

Algorithm:
    This implementation uses Wilder's Smoothed Moving Average (SMMA), which is
    the standard RSI calculation method. Wilder's smoothing uses alpha = 1/period
    rather than the standard EMA formula of alpha = 2/(period+1).

    Wilder's SMMA formula:
        SMMA(i) = ((SMMA(i-1) * (period - 1)) + current_value) / period

    This is equivalent to EWM with alpha = 1/period, which provides a smoother
    result than standard EMA and is the original formula from Welles Wilder's
    1978 book "New Concepts in Technical Trading Systems".

Signal Generation:
    The graduated signal function divides RSI into zones:
    - RSI <= oversold (35): Strong buy signal (+1.0)
    - RSI 35-45: Moderate buy signal (+0.3 to +0.7, linearly scaled)
    - RSI 45-55: Dead zone (0.0) - no signal to avoid noise
    - RSI 55-65: Moderate sell signal (-0.3 to -0.7, linearly scaled)
    - RSI >= overbought (65): Strong sell signal (-1.0)

Parameters:
    - period: 14 (Wilder's original recommendation, good for swing trading)
    - oversold: 35 (more aggressive than standard 30)
    - overbought: 65 (more aggressive than standard 70)

Integration:
    Used by SignalScorer with 25% weight. Combined with MACD, Bollinger Bands,
    EMA crossover, and volume for confluence-based trading signals.
"""

import pandas as pd
import numpy as np

# Dead zone thresholds for graduated signal
_RSI_DEAD_ZONE_LOW = 45
_RSI_DEAD_ZONE_HIGH = 55


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate RSI using Wilder's Smoothed Moving Average.

    Uses Wilder's original smoothing method (alpha = 1/period) rather than
    standard EMA (alpha = 2/(period+1)) for more stable signals.

    Args:
        prices: Series of closing prices
        period: RSI calculation period (default: 14, Wilder's recommendation)

    Returns:
        Series of RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta.where(delta < 0, 0.0))

    # Use Wilder's smoothing: alpha = 1/period
    # This is equivalent to: SMMA(i) = ((SMMA(i-1) * (period-1)) + current) / period
    alpha = 1.0 / period
    avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()

    # Calculate relative strength
    rs = avg_gains / avg_losses.replace(0, np.inf)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_rsi_signal(
    rsi_value: float,
    oversold: float = 35.0,
    overbought: float = 65.0,
) -> int:
    """
    Get trading signal from RSI value (binary version).

    Args:
        rsi_value: Current RSI value
        oversold: Oversold threshold (default: 35)
        overbought: Overbought threshold (default: 65)

    Returns:
        +1 for buy signal, -1 for sell signal, 0 for neutral
    """
    if pd.isna(rsi_value):
        return 0

    if rsi_value < oversold:
        return 1  # Buy signal
    elif rsi_value > overbought:
        return -1  # Sell signal

    return 0  # Neutral


def get_rsi_signal_graduated(
    rsi_value: float,
    oversold: float = 35.0,
    overbought: float = 65.0,
) -> float:
    """
    Get graduated trading signal from RSI value.

    Returns a continuous signal from -1.0 to +1.0 with dead zone around neutral.

    Zones:
    - RSI <= oversold (35): +1.0 (strong buy)
    - RSI 35-45: +0.3 to +0.7 (moderate buy, linearly scaled)
    - RSI 45-55: 0.0 (dead zone)
    - RSI 55-65: -0.3 to -0.7 (moderate sell, linearly scaled)
    - RSI >= overbought (65): -1.0 (strong sell)

    Args:
        rsi_value: Current RSI value
        oversold: Oversold threshold (default: 35)
        overbought: Overbought threshold (default: 65)

    Returns:
        Float from -1.0 to +1.0
    """
    if pd.isna(rsi_value):
        return 0.0

    # Dead zone: 45-55 returns 0
    if _RSI_DEAD_ZONE_LOW <= rsi_value <= _RSI_DEAD_ZONE_HIGH:
        return 0.0

    # Bullish zones (below dead zone)
    if rsi_value < _RSI_DEAD_ZONE_LOW:
        if rsi_value <= oversold:  # <= 35: strong buy
            return 1.0
        else:  # 35-45: scaled buy (0.3 to 0.7)
            return 0.3 + 0.4 * (_RSI_DEAD_ZONE_LOW - rsi_value) / (_RSI_DEAD_ZONE_LOW - oversold)

    # Bearish zones (above dead zone)
    if rsi_value > _RSI_DEAD_ZONE_HIGH:
        if rsi_value >= overbought:  # >= 65: strong sell
            return -1.0
        else:  # 55-65: scaled sell (-0.3 to -0.7)
            return -0.3 - 0.4 * (rsi_value - _RSI_DEAD_ZONE_HIGH) / (overbought - _RSI_DEAD_ZONE_HIGH)

    return 0.0
