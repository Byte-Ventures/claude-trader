"""
Volume Weighted Average Price (VWAP) indicator.

VWAP represents the average price a security has traded at throughout the period,
weighted by volume. It's widely used by institutional traders as a benchmark for
"fair value" and to minimize market impact when executing large orders.

Signal Generation:
    Price above VWAP → asset is trading above fair value (potential overextension)
    Price below VWAP → asset is trading below fair value (potential undervaluation)

    The graduated signal uses price deviation from VWAP as percentage:
    - Price > VWAP + threshold → bearish signal (0 to -1.0)
    - Price < VWAP - threshold → bullish signal (0 to +1.0)
    - Price near VWAP → neutral (dead zone)

    This is a mean-reversion signal: expect price to return toward VWAP.

Integration:
    VWAP data is fetched from Kraken's public API (regardless of trading exchange)
    and used as an additional signal component in SignalScorer.

    Note: The VWAP value updates per candle, but the signal is recalculated
    every iteration as the current price changes. This gives real-time signal
    updates even though the VWAP reference is candle-based.
"""

from decimal import Decimal
from typing import Optional, Union

import structlog

logger = structlog.get_logger(__name__)

# Default threshold for dead zone (percentage)
_DEFAULT_THRESHOLD_PERCENT = 0.5

# Signal scaling boundaries
_FULL_SIGNAL_DEVIATION_PERCENT = 1.0  # Deviation for full signal strength


def get_vwap_signal_graduated(
    price: Union[float, Decimal],
    vwap: Optional[Union[float, Decimal]],
    threshold_percent: float = _DEFAULT_THRESHOLD_PERCENT,
) -> float:
    """
    Get graduated trading signal based on price vs VWAP deviation.

    This is a mean-reversion signal:
    - Price above VWAP suggests overextension → sell signal
    - Price below VWAP suggests undervaluation → buy signal

    Signal strength scales linearly with deviation magnitude:
    - At threshold (e.g., 0.5%): signal = ±0.5
    - At 2x threshold (e.g., 1.0%): signal = ±1.0 (max)
    - Below threshold: signal = 0 (dead zone)

    Args:
        price: Current price (must be > 0)
        vwap: VWAP value from enrichment data (can be None if unavailable)
        threshold_percent: Minimum deviation % for non-zero signal (default: 0.5)

    Returns:
        Float from -1.0 to +1.0:
        - Positive = bullish (price below VWAP, expect mean reversion up)
        - Negative = bearish (price above VWAP, expect mean reversion down)
        - Zero = neutral (price near VWAP or VWAP unavailable)

    Examples:
        >>> get_vwap_signal_graduated(100.0, 99.0)  # Price 1% above VWAP
        -1.0  # Bearish - expect reversion down
        >>> get_vwap_signal_graduated(100.0, 101.0)  # Price 1% below VWAP
        1.0   # Bullish - expect reversion up
        >>> get_vwap_signal_graduated(100.0, 100.2)  # Price 0.2% below VWAP
        0.0   # Dead zone - within threshold
    """
    # Handle missing or invalid inputs
    if vwap is None:
        return 0.0

    # Convert to float for calculations
    price_f = float(price) if isinstance(price, Decimal) else price
    vwap_f = float(vwap) if isinstance(vwap, Decimal) else vwap

    if price_f <= 0 or vwap_f <= 0:
        return 0.0

    if threshold_percent <= 0:
        logger.warning(
            "invalid_vwap_threshold",
            threshold=threshold_percent,
            using=_DEFAULT_THRESHOLD_PERCENT,
        )
        threshold_percent = _DEFAULT_THRESHOLD_PERCENT

    # Calculate deviation as percentage
    # Positive deviation = price above VWAP
    # Negative deviation = price below VWAP
    deviation_percent = ((price_f - vwap_f) / vwap_f) * 100

    # Apply dead zone
    if abs(deviation_percent) < threshold_percent:
        return 0.0

    # Calculate signal strength
    # Scale from threshold to full signal deviation
    # At threshold: signal magnitude = 0.5
    # At full_signal_deviation: signal magnitude = 1.0
    full_deviation = _FULL_SIGNAL_DEVIATION_PERCENT

    # Linear interpolation from threshold to full signal
    if abs(deviation_percent) >= full_deviation:
        signal_magnitude = 1.0
    elif full_deviation <= threshold_percent:
        # Guard against division by zero when threshold >= full_deviation
        # (e.g., vwap_threshold_percent=1.0 with _FULL_SIGNAL_DEVIATION_PERCENT=1.0)
        # If we're past the threshold, just use full signal strength
        signal_magnitude = 1.0
    else:
        # Scale linearly: at threshold = 0.5, at full_deviation = 1.0
        # signal = 0.5 + 0.5 * (deviation - threshold) / (full - threshold)
        signal_magnitude = 0.5 + 0.5 * (
            (abs(deviation_percent) - threshold_percent)
            / (full_deviation - threshold_percent)
        )

    # Direction: price above VWAP = bearish (negative), below = bullish (positive)
    # This is mean-reversion logic: expect price to return toward VWAP
    if deviation_percent > 0:
        # Price above VWAP → bearish signal
        signal = -signal_magnitude
    else:
        # Price below VWAP → bullish signal
        signal = signal_magnitude

    logger.debug(
        "vwap_signal_calculated",
        price=round(price_f, 2),
        vwap=round(vwap_f, 2),
        deviation_pct=round(deviation_percent, 3),
        signal=round(signal, 3),
    )

    return signal


def calculate_price_vs_vwap_percent(
    price: Union[float, Decimal],
    vwap: Optional[Union[float, Decimal]],
) -> Optional[float]:
    """
    Calculate price deviation from VWAP as a percentage.

    Useful for logging, dashboards, and analysis.

    Args:
        price: Current price
        vwap: VWAP value (can be None)

    Returns:
        Deviation percentage (positive = above VWAP, negative = below)
        or None if VWAP is unavailable
    """
    if vwap is None:
        return None

    price_f = float(price) if isinstance(price, Decimal) else price
    vwap_f = float(vwap) if isinstance(vwap, Decimal) else vwap

    if vwap_f <= 0:
        return None

    return ((price_f - vwap_f) / vwap_f) * 100
