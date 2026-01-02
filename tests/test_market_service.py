"""
Unit tests for MarketService.

Tests cover:
- HTF trend calculation with caching
- Cache hit/miss tracking
- Cache invalidation
- Configuration updates
- Error handling and fail-open behavior

All tests use mocked components - NO real API calls.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock
from freezegun import freeze_time
import pandas as pd

from src.daemon.market_service import MarketService, MarketConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def market_config():
    """Default market configuration."""
    return MarketConfig(
        trading_pair="BTC-USD",
        mtf_enabled=True,
        mtf_4h_enabled=True,
        mtf_daily_cache_minutes=60,
        mtf_4h_cache_minutes=30,
        mtf_daily_candle_limit=50,
        mtf_4h_candle_limit=50,
    )


@pytest.fixture
def mock_exchange_client():
    """Mock exchange client."""
    client = Mock()
    # Return valid candle data by default
    candles = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="h"),
        "open": [100.0] * 50,
        "high": [105.0] * 50,
        "low": [95.0] * 50,
        "close": [102.0] * 50,
        "volume": [1000.0] * 50,
    })
    client.get_candles.return_value = candles
    return client


@pytest.fixture
def mock_signal_scorer():
    """Mock signal scorer with required attributes."""
    scorer = Mock()
    # Set required period attributes matching production defaults
    # Only ema_slow_period is used for HTF validation (get_trend only uses EMA)
    scorer.ema_slow_period = 21
    scorer.bollinger_period = 20
    scorer.macd_slow = 26
    # Default trend return
    scorer.get_trend.return_value = "neutral"
    return scorer


@pytest.fixture
def market_service(market_config, mock_exchange_client, mock_signal_scorer):
    """MarketService instance with mocked dependencies."""
    return MarketService(
        config=market_config,
        exchange_client=mock_exchange_client,
        signal_scorer=mock_signal_scorer,
    )


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization_default_state(market_service):
    """Test MarketService initializes with correct default state."""
    assert market_service._daily_trend == "neutral"
    assert market_service._4h_trend == "neutral"
    assert market_service._daily_last_fetch is None
    assert market_service._4h_last_fetch is None
    assert market_service._htf_cache_hits == 0
    assert market_service._htf_cache_misses == 0


def test_initialization_stores_config(market_config, mock_exchange_client, mock_signal_scorer):
    """Test MarketService stores provided config."""
    service = MarketService(market_config, mock_exchange_client, mock_signal_scorer)
    assert service.config == market_config
    assert service.config.trading_pair == "BTC-USD"
    assert service.config.mtf_enabled is True


# ============================================================================
# HTF Bias Tests - MTF Disabled
# ============================================================================

def test_get_htf_bias_mtf_disabled_returns_neutral(market_service):
    """Test get_htf_bias returns neutral with None trends when MTF is disabled."""
    market_service.config.mtf_enabled = False

    combined, daily, four_hour = market_service.get_htf_bias()

    assert combined == "neutral"
    assert daily is None
    assert four_hour is None


def test_get_htf_bias_mtf_disabled_no_api_calls(market_service, mock_exchange_client):
    """Test get_htf_bias makes no API calls when MTF is disabled."""
    market_service.config.mtf_enabled = False

    market_service.get_htf_bias()

    mock_exchange_client.get_candles.assert_not_called()


# ============================================================================
# HTF Bias Tests - MTF Enabled
# ============================================================================

def test_get_htf_bias_both_bullish(market_service, mock_signal_scorer):
    """Test combined bias is bullish when both timeframes are bullish."""
    mock_signal_scorer.get_trend.return_value = "bullish"

    combined, daily, four_hour = market_service.get_htf_bias()

    assert combined == "bullish"
    assert daily == "bullish"
    assert four_hour == "bullish"


def test_get_htf_bias_both_bearish(market_service, mock_signal_scorer):
    """Test combined bias is bearish when both timeframes are bearish."""
    mock_signal_scorer.get_trend.return_value = "bearish"

    combined, daily, four_hour = market_service.get_htf_bias()

    assert combined == "bearish"
    assert daily == "bearish"
    assert four_hour == "bearish"


def test_get_htf_bias_mixed_returns_neutral(market_service, mock_signal_scorer):
    """Test combined bias is neutral when timeframes disagree."""
    # Daily bullish, 4H bearish
    mock_signal_scorer.get_trend.side_effect = ["bullish", "bearish"]

    combined, daily, four_hour = market_service.get_htf_bias()

    assert combined == "neutral"
    assert daily == "bullish"
    assert four_hour == "bearish"


def test_get_htf_bias_4h_disabled_uses_daily_only(market_service, mock_signal_scorer):
    """Test when 4H is disabled, only daily trend is used."""
    market_service.config.mtf_4h_enabled = False
    mock_signal_scorer.get_trend.return_value = "bearish"

    combined, daily, four_hour = market_service.get_htf_bias()

    assert combined == "bearish"
    assert daily == "bearish"
    assert four_hour is None
    # Should only fetch daily candles
    assert mock_signal_scorer.get_trend.call_count == 1


# ============================================================================
# Cache Behavior Tests
# ============================================================================

def test_cache_miss_on_first_call(market_service, mock_exchange_client):
    """Test first call results in cache miss and API call."""
    with freeze_time("2024-01-01 12:00:00"):
        market_service.get_htf_bias()

        # Should fetch both daily and 4H candles
        assert mock_exchange_client.get_candles.call_count == 2
        hits, misses = market_service.get_cache_stats()
        assert hits == 0
        assert misses == 2  # One miss each for daily and 4H


def test_cache_hit_within_ttl(market_service, mock_exchange_client):
    """Test subsequent calls within TTL use cached values."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # First call - cache miss
        market_service.get_htf_bias()
        initial_call_count = mock_exchange_client.get_candles.call_count

        # Move forward 10 minutes (within both TTLs: daily=60min, 4h=30min)
        frozen_time.move_to("2024-01-01 12:10:00")
        market_service.get_htf_bias()

        # No additional API calls
        assert mock_exchange_client.get_candles.call_count == initial_call_count
        hits, misses = market_service.get_cache_stats()
        assert hits == 2  # Both daily and 4H from cache
        assert misses == 2  # Initial misses


def test_cache_miss_after_ttl_expires(market_service, mock_exchange_client):
    """Test cache expires after TTL and triggers new fetch."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # First call
        market_service.get_htf_bias()

        # Move past 4H cache TTL (30 min) but within daily TTL (60 min)
        frozen_time.move_to("2024-01-01 12:35:00")
        market_service.get_htf_bias()

        # Should fetch 4H again but not daily
        hits, misses = market_service.get_cache_stats()
        assert hits == 1  # Daily from cache
        assert misses == 3  # Initial 2 + 4H refresh


def test_cache_invalidation_clears_timestamps(market_service, mock_exchange_client):
    """Test cache invalidation clears last fetch timestamps."""
    with freeze_time("2024-01-01 12:00:00"):
        market_service.get_htf_bias()

        # Invalidate cache
        market_service.invalidate_cache()

        assert market_service._daily_last_fetch is None
        assert market_service._4h_last_fetch is None


def test_cache_invalidation_forces_refetch(market_service, mock_exchange_client):
    """Test cache invalidation forces new API calls on next request."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        market_service.get_htf_bias()
        initial_count = mock_exchange_client.get_candles.call_count

        # Move forward slightly and invalidate
        frozen_time.move_to("2024-01-01 12:05:00")
        market_service.invalidate_cache()
        market_service.get_htf_bias()

        # Should have made 2 more API calls
        assert mock_exchange_client.get_candles.call_count == initial_count + 2


# ============================================================================
# Cache Stats Tests
# ============================================================================

def test_get_cache_stats_returns_tuple(market_service):
    """Test get_cache_stats returns correct tuple format."""
    hits, misses = market_service.get_cache_stats()

    assert isinstance(hits, int)
    assert isinstance(misses, int)
    assert hits == 0
    assert misses == 0


def test_cache_stats_accumulate_correctly(market_service):
    """Test cache stats accumulate over multiple calls."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # First call - 2 misses
        market_service.get_htf_bias()

        # Second call within TTL - 2 hits
        frozen_time.move_to("2024-01-01 12:05:00")
        market_service.get_htf_bias()

        # Third call within TTL - 2 more hits
        frozen_time.move_to("2024-01-01 12:10:00")
        market_service.get_htf_bias()

        hits, misses = market_service.get_cache_stats()
        assert hits == 4
        assert misses == 2


# ============================================================================
# Configuration Update Tests
# ============================================================================

def test_update_config_replaces_config(market_service):
    """Test update_config replaces the configuration."""
    new_config = MarketConfig(
        trading_pair="ETH-USD",
        mtf_enabled=False,
        mtf_daily_cache_minutes=120,
    )

    market_service.update_config(new_config)

    assert market_service.config == new_config
    assert market_service.config.trading_pair == "ETH-USD"
    assert market_service.config.mtf_enabled is False


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_connection_error_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test ConnectionError returns cached trend (fail-open)."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # First successful call
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        # Simulate connection error on next fetch
        frozen_time.move_to("2024-01-01 13:05:00")  # Past both TTLs
        mock_exchange_client.get_candles.side_effect = ConnectionError("Network down")

        combined, daily, four_hour = market_service.get_htf_bias()

        # Should return cached values
        assert daily == "bullish"


def test_timeout_error_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test TimeoutError returns cached trend (fail-open)."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bearish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.side_effect = TimeoutError("Request timeout")

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bearish"


def test_empty_candles_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test empty candles DataFrame returns cached trend."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.return_value = pd.DataFrame()

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bullish"


def test_insufficient_candles_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test insufficient candles returns cached trend."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        # Return only 10 candles (need at least 26 for ema_slow_period)
        insufficient = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="h"),
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000.0] * 10,
        })
        mock_exchange_client.get_candles.return_value = insufficient

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bullish"


def test_none_candles_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test None candles returns cached trend."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.return_value = None

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bullish"


def test_value_error_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test ValueError in processing returns cached trend."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bearish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.side_effect = ValueError("Invalid data")

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bearish"


def test_not_implemented_error_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test NotImplementedError (unsupported granularity) returns cached trend."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.side_effect = NotImplementedError("Granularity not supported")

        combined, daily, four_hour = market_service.get_htf_bias()

        assert daily == "bullish"


def test_unexpected_error_returns_cached_trend(market_service, mock_exchange_client, mock_signal_scorer):
    """Test unexpected errors return cached trend (fail-open)."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        mock_signal_scorer.get_trend.return_value = "bullish"
        market_service.get_htf_bias()

        frozen_time.move_to("2024-01-01 13:05:00")
        mock_exchange_client.get_candles.side_effect = RuntimeError("Unexpected error")

        combined, daily, four_hour = market_service.get_htf_bias()

        # Should fail-open and return cached trend
        assert daily == "bullish"


def test_no_cached_trend_returns_neutral_on_error(market_service, mock_exchange_client):
    """Test error with no cached trend returns neutral."""
    mock_exchange_client.get_candles.side_effect = ConnectionError("Network down")

    combined, daily, four_hour = market_service.get_htf_bias()

    # Should return neutral when no cached value exists
    assert combined == "neutral"


# ============================================================================
# API Call Verification Tests
# ============================================================================

def test_daily_candles_fetched_with_correct_params(market_service, mock_exchange_client):
    """Test daily candles are fetched with correct parameters."""
    market_service.config.mtf_4h_enabled = False  # Only test daily

    market_service.get_htf_bias()

    mock_exchange_client.get_candles.assert_called_with(
        "BTC-USD",
        granularity="ONE_DAY",
        limit=50,
    )


def test_4h_candles_fetched_with_correct_params(market_service, mock_exchange_client, mock_signal_scorer):
    """Test 4H candles are fetched with FOUR_HOUR granularity."""
    # Need to check the second call (4H after daily)
    mock_signal_scorer.get_trend.return_value = "neutral"
    market_service.get_htf_bias()

    calls = mock_exchange_client.get_candles.call_args_list
    assert len(calls) == 2

    # Second call should be 4H - check kwargs explicitly
    _, kwargs = calls[1]
    assert kwargs.get("granularity") == "FOUR_HOUR", (
        f"Expected 4H candles with granularity='FOUR_HOUR', got: {kwargs}"
    )


# ============================================================================
# Unsupported Granularity Tests
# ============================================================================

def test_unsupported_granularity_raises_value_error(market_config, mock_exchange_client, mock_signal_scorer):
    """Test that unsupported granularity raises ValueError."""
    service = MarketService(market_config, mock_exchange_client, mock_signal_scorer)

    # Directly call _get_timeframe_trend with invalid granularity
    with pytest.raises(ValueError, match="Unsupported granularity"):
        service._get_timeframe_trend("INVALID_GRANULARITY", 60)
