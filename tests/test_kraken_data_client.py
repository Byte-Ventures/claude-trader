"""Tests for Kraken data enrichment client."""

import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.api.kraken_data_client import KrakenDataClient, EnrichmentCache


class TestKrakenDataClient:
    """Tests for KrakenDataClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = KrakenDataClient(cache_ttl_seconds=60, request_timeout=10)

    def test_initialization(self):
        """Test client initializes with correct defaults."""
        client = KrakenDataClient()
        assert client.cache_ttl_seconds == 60
        assert client.request_timeout == 10
        assert client._cache == {}
        assert client._cache_hits == 0
        assert client._cache_misses == 0

    def test_custom_initialization(self):
        """Test client initializes with custom parameters."""
        client = KrakenDataClient(cache_ttl_seconds=120, request_timeout=30)
        assert client.cache_ttl_seconds == 120
        assert client.request_timeout == 30

    @patch.object(KrakenDataClient, "_fetch_ohlc_data")
    def test_get_enrichment_data_fetches_on_cache_miss(self, mock_fetch):
        """Test that data is fetched when cache is empty."""
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "vwap": [Decimal("50000.0")],
            "trade_count": [100],
        })
        mock_fetch.return_value = mock_df

        result = self.client.get_enrichment_data("BTC-USD", "ONE_HOUR", limit=100)

        mock_fetch.assert_called_once_with("BTC-USD", "ONE_HOUR", 100)
        assert result is not None
        assert not result.empty
        assert self.client._cache_misses == 1
        assert self.client._cache_hits == 0

    @patch.object(KrakenDataClient, "_fetch_ohlc_data")
    def test_get_enrichment_data_uses_cache_on_hit(self, mock_fetch):
        """Test that cache is used when data is available."""
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "vwap": [Decimal("50000.0")],
            "trade_count": [100],
        })
        mock_fetch.return_value = mock_df

        # First call - cache miss
        self.client.get_enrichment_data("BTC-USD", "ONE_HOUR")
        assert self.client._cache_misses == 1

        # Second call - cache hit
        result = self.client.get_enrichment_data("BTC-USD", "ONE_HOUR")
        assert result is not None
        assert self.client._cache_hits == 1
        # fetch should only be called once
        assert mock_fetch.call_count == 1

    @patch.object(KrakenDataClient, "_fetch_ohlc_data")
    def test_cache_expires_after_ttl(self, mock_fetch):
        """Test that cache expires after TTL."""
        # Use a very short TTL
        client = KrakenDataClient(cache_ttl_seconds=1)

        mock_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "vwap": [Decimal("50000.0")],
            "trade_count": [100],
        })
        mock_fetch.return_value = mock_df

        # First call - cache miss
        client.get_enrichment_data("BTC-USD", "ONE_HOUR")
        assert mock_fetch.call_count == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call - should be a miss due to expired cache
        client.get_enrichment_data("BTC-USD", "ONE_HOUR")
        assert mock_fetch.call_count == 2

    @patch.object(KrakenDataClient, "_fetch_ohlc_data")
    def test_different_pairs_use_different_cache_entries(self, mock_fetch):
        """Test that different trading pairs have separate cache entries."""
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "vwap": [Decimal("50000.0")],
            "trade_count": [100],
        })
        mock_fetch.return_value = mock_df

        # Fetch BTC-USD
        self.client.get_enrichment_data("BTC-USD", "ONE_HOUR")
        # Fetch ETH-USD
        self.client.get_enrichment_data("ETH-USD", "ONE_HOUR")

        # Both should be cache misses
        assert mock_fetch.call_count == 2

    @patch.object(KrakenDataClient, "_fetch_ohlc_data")
    def test_fetch_error_returns_none(self, mock_fetch):
        """Test that fetch errors return None (fail-open)."""
        mock_fetch.side_effect = Exception("Network error")

        result = self.client.get_enrichment_data("BTC-USD", "ONE_HOUR")

        assert result is None
        assert self.client._fetch_errors == 1

    @patch("requests.Session.get")
    def test_rate_limiting(self, mock_get):
        """Test that rate limiting is enforced between requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "XXBTZUSD": [
                    [1704067200, "42000", "42100", "41900", "42050", "42025.5", "100", 50],
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Make first request
        start = time.time()
        self.client._fetch_ohlc_data("BTC-USD", "ONE_HOUR", 100)

        # Make second request immediately
        self.client._fetch_ohlc_data("BTC-USD", "ONE_HOUR", 100)
        elapsed = time.time() - start

        # Should have waited at least 1 second (MIN_REQUEST_INTERVAL_MS = 1000)
        assert elapsed >= 0.9  # Allow small tolerance

    @patch("requests.Session.get")
    def test_api_error_handling(self, mock_get):
        """Test that API errors are handled correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": ["EGeneral:Unknown pair"]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="Kraken API error"):
            self.client._fetch_ohlc_data("INVALID-PAIR", "ONE_HOUR", 100)

    def test_get_stats(self):
        """Test statistics retrieval."""
        self.client._cache_hits = 5
        self.client._cache_misses = 10
        self.client._fetch_errors = 2

        stats = self.client.get_stats()

        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 10
        assert stats["fetch_errors"] == 2
        assert stats["cached_pairs"] == 0

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.client._cache[("BTC-USD", "ONE_HOUR")] = EnrichmentCache(
            data=pd.DataFrame(),
            fetched_at=datetime.now(timezone.utc),
            symbol="BTC-USD",
            granularity="ONE_HOUR",
        )

        self.client.clear_cache()

        assert len(self.client._cache) == 0


class TestEnrichmentCache:
    """Tests for EnrichmentCache dataclass."""

    def test_cache_creation(self):
        """Test cache entry creation."""
        df = pd.DataFrame({"vwap": [100.0]})
        now = datetime.now(timezone.utc)

        cache = EnrichmentCache(
            data=df,
            fetched_at=now,
            symbol="BTC-USD",
            granularity="ONE_HOUR",
        )

        assert cache.symbol == "BTC-USD"
        assert cache.granularity == "ONE_HOUR"
        assert cache.fetched_at == now
        assert len(cache.data) == 1
