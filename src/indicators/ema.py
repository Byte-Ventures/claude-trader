"""
Exponential Moving Average (EMA) indicator.

EMA gives more weight to recent prices, making it more responsive than SMA.

Trading signals:
- Fast EMA crosses above slow EMA: Buy (bullish crossover)
- Fast EMA crosses below slow EMA: Sell (bearish crossover)
- Price above both EMAs: Bullish trend
- Price below both EMAs: Bearish trend
"""

from dataclasses import dataclass
import pandas as pd


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
    import numpy as np
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


def get_ema_signal(ema_result: EMAResult) -> int:
    """
    Get trading signal from EMA crossover.

    Args:
        ema_result: EMA calculation result

    Returns:
        +1 for buy signal, -1 for sell signal, 0 for neutral
    """
    if len(ema_result.ema_fast) == 0:
        return 0

    # Check for crossover in most recent candle
    if ema_result.crossover_up.iloc[-1]:
        return 1

    if ema_result.crossover_down.iloc[-1]:
        return -1

    return 0


def get_ema_trend(ema_result: EMAResult) -> str:
    """
    Determine current trend based on EMA positions.

    Args:
        ema_result: EMA calculation result

    Returns:
        "bullish", "bearish", or "neutral"
    """
    if len(ema_result.ema_fast) == 0:
        return "neutral"

    fast = ema_result.ema_fast.iloc[-1]
    slow = ema_result.ema_slow.iloc[-1]

    if pd.isna(fast) or pd.isna(slow):
        return "neutral"

    # Strong bullish: fast significantly above slow
    diff_percent = (fast - slow) / slow * 100

    if diff_percent > 1.0:
        return "bullish"
    elif diff_percent < -1.0:
        return "bearish"

    return "neutral"


def get_price_vs_ema_signal(
    price: float,
    ema_result: EMAResult,
) -> int:
    """
    Get signal based on price position relative to EMAs.

    Args:
        price: Current price
        ema_result: EMA calculation result

    Returns:
        +1 if bullish, -1 if bearish, 0 if neutral
    """
    if len(ema_result.ema_fast) == 0:
        return 0

    fast = ema_result.ema_fast.iloc[-1]
    slow = ema_result.ema_slow.iloc[-1]

    if pd.isna(fast) or pd.isna(slow):
        return 0

    # Price above both EMAs: bullish
    if price > fast and price > slow:
        return 1

    # Price below both EMAs: bearish
    if price < fast and price < slow:
        return -1

    return 0
