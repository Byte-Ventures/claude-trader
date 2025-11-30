"""
Bollinger Bands indicator.

Bollinger Bands consist of:
- Middle Band: Simple Moving Average (SMA)
- Upper Band: SMA + (std_dev * multiplier)
- Lower Band: SMA - (std_dev * multiplier)

Trading signals:
- Price touches lower band: Potential buy (oversold)
- Price touches upper band: Potential sell (overbought)
- Bandwidth squeeze: Volatility compression, breakout expected
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""

    upper_band: pd.Series
    middle_band: pd.Series
    lower_band: pd.Series
    bandwidth: pd.Series  # (upper - lower) / middle
    percent_b: pd.Series  # (price - lower) / (upper - lower)


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> BollingerResult:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        BollingerResult with all band values
    """
    # Middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Standard deviation
    rolling_std = prices.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    # Bandwidth: normalized band width
    bandwidth = (upper_band - lower_band) / middle_band

    # %B: where price is within the bands (0 = lower, 1 = upper)
    band_width = upper_band - lower_band
    percent_b = (prices - lower_band) / band_width.replace(0, np.inf)

    return BollingerResult(
        upper_band=upper_band,
        middle_band=middle_band,
        lower_band=lower_band,
        bandwidth=bandwidth,
        percent_b=percent_b,
    )


def get_bollinger_signal(
    price: float,
    bollinger: BollingerResult,
) -> int:
    """
    Get trading signal from Bollinger Bands.

    Args:
        price: Current price
        bollinger: Bollinger Bands result

    Returns:
        +1 for buy signal, -1 for sell signal, 0 for neutral
    """
    if len(bollinger.upper_band) == 0:
        return 0

    upper = bollinger.upper_band.iloc[-1]
    lower = bollinger.lower_band.iloc[-1]

    if pd.isna(upper) or pd.isna(lower):
        return 0

    # Price at or below lower band: oversold
    if price <= lower:
        return 1

    # Price at or above upper band: overbought
    if price >= upper:
        return -1

    return 0


def is_bollinger_squeeze(
    bollinger: BollingerResult,
    threshold_percentile: float = 20.0,
    lookback: int = 50,
) -> bool:
    """
    Detect Bollinger Band squeeze (low volatility).

    A squeeze often precedes a significant price movement.

    Args:
        bollinger: Bollinger Bands result
        threshold_percentile: Bandwidth percentile threshold
        lookback: Number of periods for percentile calculation

    Returns:
        True if bandwidth is in squeeze territory
    """
    if len(bollinger.bandwidth) < lookback:
        return False

    recent_bandwidth = bollinger.bandwidth.tail(lookback)
    current_bandwidth = bollinger.bandwidth.iloc[-1]

    if pd.isna(current_bandwidth):
        return False

    # Calculate percentile of current bandwidth
    percentile = (recent_bandwidth < current_bandwidth).sum() / len(recent_bandwidth) * 100

    return percentile < threshold_percentile


def get_percent_b_signal(bollinger: BollingerResult) -> int:
    """
    Get signal based on %B indicator.

    Args:
        bollinger: Bollinger Bands result

    Returns:
        +1 for buy, -1 for sell, 0 for neutral
    """
    if len(bollinger.percent_b) == 0:
        return 0

    percent_b = bollinger.percent_b.iloc[-1]

    if pd.isna(percent_b):
        return 0

    # %B < 0: Price below lower band (very oversold)
    if percent_b < 0:
        return 1

    # %B > 1: Price above upper band (very overbought)
    if percent_b > 1:
        return -1

    # %B < 0.2: Near lower band (oversold)
    if percent_b < 0.2:
        return 1

    # %B > 0.8: Near upper band (overbought)
    if percent_b > 0.8:
        return -1

    return 0
