"""
Bollinger Bands indicator.

Bollinger Bands are a volatility indicator that consists of three bands:
- Middle Band: Simple Moving Average (SMA) of price
- Upper Band: Middle Band + (standard deviation * multiplier)
- Lower Band: Middle Band - (standard deviation * multiplier)

Algorithm:
    Middle Band = SMA(price, period)
    Standard Deviation = STD(price, period)
    Upper Band = Middle + (StdDev * multiplier)
    Lower Band = Middle - (StdDev * multiplier)

    Derived metrics:
    - Bandwidth = (Upper - Lower) / Middle
      Measures band width relative to price, useful for detecting squeezes.

    - %B (Percent B) = (Price - Lower) / (Upper - Lower)
      Shows where price is within the bands:
      - %B = 0: Price at lower band
      - %B = 0.5: Price at middle band
      - %B = 1: Price at upper band
      - %B < 0 or > 1: Price outside bands

Signal Generation:
    The graduated signal function uses %B to generate continuous signals:
    - %B <= 0 (below lower band): +1.0 (strong buy - extreme oversold)
    - %B 0-0.35: +0.3 to +0.8 (moderate buy, linearly scaled)
    - %B 0.35-0.65: 0.0 (dead zone - near middle band)
    - %B 0.65-1.0: -0.3 to -0.8 (moderate sell, linearly scaled)
    - %B >= 1 (above upper band): -1.0 (strong sell - extreme overbought)

Parameters:
    - period: 20 (standard, ~1 month of trading days)
    - std_dev: 2.0 (standard, captures ~95% of price action)

Integration:
    Used by SignalScorer with 20% weight. Combined with RSI, MACD,
    EMA crossover, and volume for confluence-based trading signals.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

# Signal generation constants
_DEAD_ZONE_LOW = 0.35  # %B below this is bullish zone
_DEAD_ZONE_HIGH = 0.65  # %B above this is bearish zone
_MIN_SIGNAL_STRENGTH = 0.3  # Minimum signal in transition zones
_SIGNAL_RANGE = 0.5  # Signal range in transition zones (0.3 + 0.5 = 0.8 max)

# Epsilon for floating point comparison (band convergence detection)
_BANDWIDTH_EPSILON = 1e-10


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

    Uses Simple Moving Average (SMA) for the middle band, which is the
    standard Bollinger Bands calculation method.

    Args:
        prices: Series of closing prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        BollingerResult with all band values and derived metrics
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
    # Handle division by zero when bands converge (bandwidth = 0)
    band_width = upper_band - lower_band
    # When bandwidth is 0, set %B to 0.5 (middle) to avoid inf/nan
    percent_b = pd.Series(
        np.where(
            band_width.abs() < _BANDWIDTH_EPSILON,
            0.5,  # Default to middle when bands converge
            (prices - lower_band) / band_width
        ),
        index=prices.index
    )

    return BollingerResult(
        upper_band=upper_band,
        middle_band=middle_band,
        lower_band=lower_band,
        bandwidth=bandwidth,
        percent_b=percent_b,
    )


def get_bollinger_signal_graduated(
    price: float,
    bollinger: BollingerResult,
) -> float:
    """
    Get graduated trading signal based on %B position within bands.

    Returns continuous signal based on where price sits within the bands.
    Uses %B indicator which shows price position relative to the bands.

    Zones:
    - %B <= 0 (below lower band): +1.0 (strong buy)
    - %B 0-0.35: +0.3 to +0.8 (moderate buy, linearly scaled)
    - %B 0.35-0.65: 0.0 (dead zone - near middle band)
    - %B 0.65-1.0: -0.3 to -0.8 (moderate sell, linearly scaled)
    - %B >= 1 (above upper band): -1.0 (strong sell)

    Args:
        price: Current price (unused but kept for API consistency)
        bollinger: Bollinger Bands result

    Returns:
        Float from -1.0 to +1.0
    """
    if len(bollinger.percent_b) == 0:
        return 0.0

    pct_b = bollinger.percent_b.iloc[-1]
    if pd.isna(pct_b):
        return 0.0

    # Outside bands: strong signals
    if pct_b <= 0:
        return 1.0  # Below lower band: strong buy
    if pct_b >= 1:
        return -1.0  # Above upper band: strong sell

    # Dead zone: middle portion of band width
    if _DEAD_ZONE_LOW <= pct_b <= _DEAD_ZONE_HIGH:
        return 0.0

    # Lower zone (0 to dead zone low): bullish
    if pct_b < _DEAD_ZONE_LOW:
        # Scale from 0.3 at dead zone edge to 0.8 near lower band
        return _MIN_SIGNAL_STRENGTH + _SIGNAL_RANGE * (_DEAD_ZONE_LOW - pct_b) / _DEAD_ZONE_LOW

    # Upper zone (dead zone high to 1): bearish
    if pct_b > _DEAD_ZONE_HIGH:
        # Scale from -0.3 at dead zone edge to -0.8 near upper band
        return -_MIN_SIGNAL_STRENGTH - _SIGNAL_RANGE * (pct_b - _DEAD_ZONE_HIGH) / (1.0 - _DEAD_ZONE_HIGH)

    return 0.0
