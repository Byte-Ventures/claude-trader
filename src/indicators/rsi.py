"""
Relative Strength Index (RSI) indicator.

RSI measures momentum by comparing the magnitude of recent gains to recent losses.
Values range from 0 to 100:
- < 30: Oversold (potential buy signal)
- > 70: Overbought (potential sell signal)

For aggressive trading, we use tighter thresholds:
- < 35: Oversold
- > 65: Overbought
"""

import pandas as pd
import numpy as np


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        prices: Series of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta.where(delta < 0, 0.0))

    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()

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
    - RSI 35-45: +0.3 to +0.7 (moderate buy)
    - RSI 45-55: 0.0 (dead zone)
    - RSI 55-65: -0.3 to -0.7 (moderate sell)
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
    if 45 <= rsi_value <= 55:
        return 0.0

    # Bullish zones (below 45)
    if rsi_value < 45:
        if rsi_value <= oversold:  # <= 35: strong buy
            return 1.0
        else:  # 35-45: scaled buy (0.3 to 0.7)
            return 0.3 + 0.4 * (45 - rsi_value) / (45 - oversold)

    # Bearish zones (above 55)
    if rsi_value > 55:
        if rsi_value >= overbought:  # >= 65: strong sell
            return -1.0
        else:  # 55-65: scaled sell (-0.3 to -0.7)
            return -0.3 - 0.4 * (rsi_value - 55) / (overbought - 55)

    return 0.0


def is_rsi_divergence(
    prices: pd.Series,
    rsi: pd.Series,
    lookback: int = 10,
) -> tuple[bool, bool]:
    """
    Detect RSI divergence (price and RSI moving in opposite directions).

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high

    Args:
        prices: Series of closing prices
        rsi: Series of RSI values
        lookback: Number of periods to look back

    Returns:
        Tuple of (bullish_divergence, bearish_divergence)
    """
    if len(prices) < lookback or len(rsi) < lookback:
        return False, False

    recent_prices = prices.tail(lookback)
    recent_rsi = rsi.tail(lookback)

    # Find local minima and maxima
    price_min_idx = recent_prices.idxmin()
    price_max_idx = recent_prices.idxmax()
    rsi_min_idx = recent_rsi.idxmin()
    rsi_max_idx = recent_rsi.idxmax()

    current_price = prices.iloc[-1]
    current_rsi = rsi.iloc[-1]

    # Bullish divergence: lower price low, higher RSI low
    bullish = (
        current_price < recent_prices.min() and
        current_rsi > recent_rsi.min()
    )

    # Bearish divergence: higher price high, lower RSI high
    bearish = (
        current_price > recent_prices.max() and
        current_rsi < recent_rsi.max()
    )

    return bullish, bearish
