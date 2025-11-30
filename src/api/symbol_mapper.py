"""
Symbol mapper for translating between exchange formats.

Different exchanges use different symbol formats:
- Coinbase: BTC-USD, ETH-USD
- Kraken: XBT/USD (or XXBTZUSD for API), ETH/USD (or XETHZUSD)

This module provides bidirectional translation.
"""

from enum import Enum
from typing import Tuple


class Exchange(Enum):
    """Supported exchanges."""

    COINBASE = "coinbase"
    KRAKEN = "kraken"


# Kraken uses XBT instead of BTC
KRAKEN_SYMBOL_MAP = {
    "BTC": "XBT",
    "DOGE": "XDG",
}

KRAKEN_SYMBOL_REVERSE = {v: k for k, v in KRAKEN_SYMBOL_MAP.items()}

# Kraken API uses these prefixes for some assets
KRAKEN_ASSET_PREFIXES = {
    "XBT": "XXBT",
    "ETH": "XETH",
    "LTC": "XLTC",
    "XRP": "XXRP",
    "USD": "ZUSD",
    "EUR": "ZEUR",
    "GBP": "ZGBP",
    "SEK": "ZSEK",
}


def normalize_symbol(symbol: str, exchange: Exchange) -> str:
    """
    Convert exchange-specific symbol to normalized format (BTC-USD style).

    Args:
        symbol: Exchange-specific symbol (e.g., XBT/USD, XXBTZUSD)
        exchange: Source exchange

    Returns:
        Normalized symbol (e.g., BTC-USD)
    """
    if exchange == Exchange.COINBASE:
        # Coinbase already uses our normalized format
        return symbol.upper()

    if exchange == Exchange.KRAKEN:
        # Handle XXBTZUSD format
        if len(symbol) == 8 and symbol.startswith("X") and "Z" in symbol:
            base = symbol[1:4]
            quote = symbol[5:8]
        # Handle XBT/USD format
        elif "/" in symbol:
            base, quote = symbol.split("/")
        else:
            # Already in simple format
            if "-" in symbol:
                return symbol.upper()
            # Try to parse as 6-char format (e.g., XBTUSD)
            base = symbol[:3]
            quote = symbol[3:]

        # Convert Kraken symbols to standard
        base = KRAKEN_SYMBOL_REVERSE.get(base, base)
        quote = KRAKEN_SYMBOL_REVERSE.get(quote, quote)

        return f"{base}-{quote}".upper()

    return symbol.upper()


def to_exchange_symbol(normalized: str, exchange: Exchange) -> str:
    """
    Convert normalized symbol (BTC-USD) to exchange-specific format.

    Args:
        normalized: Normalized symbol (e.g., BTC-USD)
        exchange: Target exchange

    Returns:
        Exchange-specific symbol
    """
    if exchange == Exchange.COINBASE:
        # Coinbase uses our normalized format
        return normalized.upper()

    if exchange == Exchange.KRAKEN:
        # Parse normalized format
        parts = normalized.upper().split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid symbol format: {normalized}")

        base, quote = parts

        # Convert to Kraken symbols
        base = KRAKEN_SYMBOL_MAP.get(base, base)
        quote = KRAKEN_SYMBOL_MAP.get(quote, quote)

        # Kraken REST API uses the slash format for most endpoints
        return f"{base}/{quote}"

    return normalized.upper()


def to_kraken_asset(currency: str) -> str:
    """
    Convert currency code to Kraken asset code.

    Args:
        currency: Standard currency code (e.g., BTC, USD)

    Returns:
        Kraken asset code (e.g., XXBT, ZUSD)
    """
    # First convert to Kraken symbol
    kraken_symbol = KRAKEN_SYMBOL_MAP.get(currency.upper(), currency.upper())

    # Then add prefix if applicable
    return KRAKEN_ASSET_PREFIXES.get(kraken_symbol, kraken_symbol)


def from_kraken_asset(kraken_asset: str) -> str:
    """
    Convert Kraken asset code to standard currency code.

    Args:
        kraken_asset: Kraken asset code (e.g., XXBT, ZUSD)

    Returns:
        Standard currency code (e.g., BTC, USD)
    """
    # Remove common prefixes
    asset = kraken_asset.upper()

    # Try to find in prefix map
    for standard, prefixed in KRAKEN_ASSET_PREFIXES.items():
        if asset == prefixed:
            # Convert back from Kraken symbol
            return KRAKEN_SYMBOL_REVERSE.get(standard, standard)

    # Handle simple cases
    if asset.startswith("X") and len(asset) == 4:
        asset = asset[1:]
    elif asset.startswith("Z") and len(asset) == 4:
        asset = asset[1:]

    return KRAKEN_SYMBOL_REVERSE.get(asset, asset)


def parse_trading_pair(symbol: str) -> Tuple[str, str]:
    """
    Parse a trading pair into base and quote currencies.

    Args:
        symbol: Trading pair in any format (BTC-USD, XBT/USD, etc.)

    Returns:
        Tuple of (base, quote) in normalized form
    """
    # Try different separators
    for sep in ["-", "/", "_"]:
        if sep in symbol:
            parts = symbol.split(sep)
            if len(parts) == 2:
                base = KRAKEN_SYMBOL_REVERSE.get(parts[0].upper(), parts[0].upper())
                quote = KRAKEN_SYMBOL_REVERSE.get(parts[1].upper(), parts[1].upper())
                return base, quote

    # No separator - assume 6 char format
    if len(symbol) == 6:
        base = symbol[:3].upper()
        quote = symbol[3:].upper()
        base = KRAKEN_SYMBOL_REVERSE.get(base, base)
        quote = KRAKEN_SYMBOL_REVERSE.get(quote, quote)
        return base, quote

    raise ValueError(f"Cannot parse trading pair: {symbol}")


# Granularity mapping between our format and Kraken's
KRAKEN_GRANULARITY_MAP = {
    "ONE_MINUTE": 1,
    "FIVE_MINUTE": 5,
    "FIFTEEN_MINUTE": 15,
    "THIRTY_MINUTE": 30,
    "ONE_HOUR": 60,
    "TWO_HOUR": 120,
    "FOUR_HOUR": 240,
    "SIX_HOUR": 360,
    "ONE_DAY": 1440,
    "ONE_WEEK": 10080,
}

KRAKEN_GRANULARITY_REVERSE = {v: k for k, v in KRAKEN_GRANULARITY_MAP.items()}


def to_kraken_granularity(granularity: str) -> int:
    """Convert our granularity format to Kraken's (in minutes)."""
    return KRAKEN_GRANULARITY_MAP.get(granularity.upper(), 60)


def from_kraken_granularity(minutes: int) -> str:
    """Convert Kraken granularity to our format."""
    return KRAKEN_GRANULARITY_REVERSE.get(minutes, "ONE_HOUR")
