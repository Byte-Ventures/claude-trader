"""
Exchange factory for creating the appropriate exchange client.

Based on the EXCHANGE setting, creates either a Coinbase or Kraken client.
"""

import structlog

from config.settings import Exchange, Settings, get_settings
from src.api.coinbase_client import CoinbaseClient
from src.api.exchange_protocol import ExchangeClient
from src.api.kraken_client import KrakenClient

logger = structlog.get_logger(__name__)


def create_exchange_client(settings: Settings | None = None) -> ExchangeClient:
    """
    Create an exchange client based on settings.

    Args:
        settings: Optional settings instance. If not provided, uses global settings.

    Returns:
        Exchange client implementing ExchangeClient protocol

    Raises:
        ValueError: If required credentials are missing
    """
    if settings is None:
        settings = get_settings()

    if settings.exchange == Exchange.COINBASE:
        return _create_coinbase_client(settings)
    elif settings.exchange == Exchange.KRAKEN:
        return _create_kraken_client(settings)
    else:
        raise ValueError(f"Unsupported exchange: {settings.exchange}")


def _create_coinbase_client(settings: Settings) -> CoinbaseClient:
    """Create a Coinbase client from settings."""
    # Prefer key file if provided
    if settings.coinbase_key_file:
        logger.info("creating_coinbase_client", source="key_file")
        return CoinbaseClient(key_file=settings.coinbase_key_file)

    # Fall back to key/secret
    if settings.coinbase_api_key and settings.coinbase_api_secret:
        logger.info("creating_coinbase_client", source="api_key")
        return CoinbaseClient(
            api_key=settings.coinbase_api_key,
            api_secret=settings.coinbase_api_secret.get_secret_value(),
        )

    raise ValueError(
        "Coinbase credentials not configured. "
        "Set COINBASE_KEY_FILE or COINBASE_API_KEY + COINBASE_API_SECRET"
    )


def _create_kraken_client(settings: Settings) -> KrakenClient:
    """Create a Kraken client from settings."""
    if not settings.kraken_api_key or not settings.kraken_api_secret:
        raise ValueError(
            "Kraken credentials not configured. "
            "Set KRAKEN_API_KEY and KRAKEN_API_SECRET in your .env file"
        )

    logger.info("creating_kraken_client")
    return KrakenClient(
        api_key=settings.kraken_api_key,
        api_secret=settings.kraken_api_secret.get_secret_value(),
    )


def get_exchange_name(settings: Settings | None = None) -> str:
    """Get the name of the configured exchange."""
    if settings is None:
        settings = get_settings()
    return settings.exchange.value.title()
