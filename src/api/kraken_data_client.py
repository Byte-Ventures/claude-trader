"""
Kraken data enrichment client for VWAP and trade count data.

Provides supplementary market data from Kraken's public API regardless
of which exchange is used for actual trading.

Features:
- Public endpoints only (no authentication required)
- Rate limiting: ~1 request/second for public endpoints
- Caching with configurable TTL
- Fail-open design: errors return None, never block trading

Kraken OHLC Response Format:
    [time, open, high, low, close, vwap, volume, count]
    We extract: timestamp, vwap, count (indices 0, 5, 7)
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Tuple

import pandas as pd
import requests
import structlog

from src.api.symbol_mapper import to_exchange_symbol, to_kraken_granularity, Exchange

logger = structlog.get_logger(__name__)

BASE_URL = "https://api.kraken.com"


@dataclass
class EnrichmentCache:
    """Cached enrichment data with metadata."""

    data: pd.DataFrame  # DataFrame with timestamp, vwap, trade_count columns
    fetched_at: datetime
    symbol: str
    granularity: str


class KrakenDataClient:
    """
    Client for fetching enrichment data (VWAP, trade count) from Kraken.

    Works regardless of which exchange is used for trading.
    Uses public API only - no credentials required.

    Features:
    - Time-based caching with configurable TTL
    - Rate limiting (minimum interval between requests)
    - Fail-open design (returns None on errors, doesn't block trading)
    """

    # Minimum interval between requests (milliseconds)
    MIN_REQUEST_INTERVAL_MS = 1000

    def __init__(
        self,
        cache_ttl_seconds: int = 60,
        request_timeout: int = 10,
    ):
        """
        Initialize enrichment client.

        Args:
            cache_ttl_seconds: How long to cache enrichment data (default: 60s)
            request_timeout: HTTP request timeout in seconds (default: 10s)
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout
        self.session = requests.Session()

        # Rate limiting
        self._last_request_time: float = 0.0

        # Cache: keyed by (symbol, granularity)
        self._cache: dict[Tuple[str, str], EnrichmentCache] = {}

        # Metrics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._fetch_errors: int = 0

        logger.info("kraken_data_client_initialized", cache_ttl=cache_ttl_seconds)

    def get_enrichment_data(
        self,
        product_id: str,
        granularity: str,
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Get VWAP and trade count data from Kraken.

        Args:
            product_id: Trading pair in normalized format (e.g., BTC-USD)
            granularity: Candle interval (e.g., ONE_HOUR, FIFTEEN_MINUTE)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns [timestamp, vwap, trade_count] or None on failure.
            Timestamps are timezone-aware (UTC).
        """
        cache_key = (product_id, granularity)

        # Check cache first
        cached = self._cache.get(cache_key)
        if cached and self._is_cache_valid(cached):
            self._cache_hits += 1
            logger.debug("enrichment_cache_hit", symbol=product_id, granularity=granularity)
            return cached.data

        self._cache_misses += 1

        # Fetch fresh data
        try:
            data = self._fetch_ohlc_data(product_id, granularity, limit)
            if data is not None and not data.empty:
                self._cache[cache_key] = EnrichmentCache(
                    data=data,
                    fetched_at=datetime.now(timezone.utc),
                    symbol=product_id,
                    granularity=granularity,
                )
                logger.debug(
                    "enrichment_data_fetched",
                    symbol=product_id,
                    granularity=granularity,
                    candles=len(data),
                )
            return data
        except Exception as e:
            self._fetch_errors += 1
            logger.warning(
                "enrichment_fetch_failed",
                symbol=product_id,
                granularity=granularity,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _is_cache_valid(self, cached: EnrichmentCache) -> bool:
        """Check if cached data is still valid."""
        age = (datetime.now(timezone.utc) - cached.fetched_at).total_seconds()
        return age < self.cache_ttl_seconds

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed_ms = (time.time() - self._last_request_time) * 1000
        if elapsed_ms < self.MIN_REQUEST_INTERVAL_MS:
            sleep_time = (self.MIN_REQUEST_INTERVAL_MS - elapsed_ms) / 1000
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _fetch_ohlc_data(
        self,
        product_id: str,
        granularity: str,
        limit: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data from Kraken public API.

        Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
        We extract only: timestamp, vwap, trade_count
        """
        self._rate_limit()

        # Convert to Kraken format
        kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
        pair_code = kraken_pair.replace("/", "")
        interval = to_kraken_granularity(granularity)

        # Calculate 'since' to get approximately 'limit' candles
        seconds_per_candle = interval * 60
        since = int(time.time()) - (limit * seconds_per_candle)

        url = f"{BASE_URL}/0/public/OHLC"
        params = {
            "pair": pair_code,
            "interval": interval,
            "since": since,
        }

        response = self.session.get(url, params=params, timeout=self.request_timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            error_msg = ", ".join(data["error"])
            raise Exception(f"Kraken API error: {error_msg}")

        result = data.get("result", {})

        # Extract candle data (exclude 'last' key)
        candles = []
        for key, value in result.items():
            if key != "last" and isinstance(value, list):
                candles = value
                break

        if not candles:
            return None

        # Build DataFrame with enrichment data
        # Kraken format: [time, open, high, low, close, vwap, volume, count]
        rows = []
        for candle in candles[-limit:]:
            rows.append({
                "timestamp": datetime.fromtimestamp(candle[0], tz=timezone.utc),
                "vwap": Decimal(str(candle[5])) if candle[5] else None,
                "trade_count": int(candle[7]) if candle[7] else None,
            })

        return pd.DataFrame(rows)

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "fetch_errors": self._fetch_errors,
            "cached_pairs": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("enrichment_cache_cleared")
