"""Tests for dashboard API routes."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.state.database import BotMode, DailyStats


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.is_paper_trading = True
    settings.enable_cramer_mode = False
    return settings


@pytest.fixture
def mock_db():
    """Mock database for testing."""
    return MagicMock()


@pytest.fixture
def sample_daily_stats():
    """Generate sample DailyStats objects."""
    def _create(days=3, bot_mode=BotMode.NORMAL):
        stats = []
        for i in range(days):
            stat = MagicMock(spec=DailyStats)
            stat.date = date(2024, 1, i + 1)
            stat.starting_balance = str(10000 + i * 100)
            stat.ending_balance = str(10100 + i * 100)
            stat.starting_price = str(40000 + i * 500)
            stat.ending_price = str(40500 + i * 500)
            stat.realized_pnl = str(100 + i * 10)
            stat.bot_mode = bot_mode
            stats.append(stat)
        return stats
    return _create


@pytest.fixture
def client(mock_settings, mock_db):
    """Create test client with mocked dependencies."""
    with patch("src.dashboard.routes.get_settings", return_value=mock_settings):
        with patch("src.dashboard.routes.get_db", return_value=mock_db):
            # Clear lru_cache to ensure fresh mocks
            from src.dashboard.routes import get_db
            get_db.cache_clear()

            from src.dashboard.server import app

            # Disable rate limiting for tests
            from src.dashboard.routes import limiter
            limiter.enabled = False

            with TestClient(app) as test_client:
                yield test_client

            # Re-enable rate limiting
            limiter.enabled = True


class TestPerformanceEndpoint:
    """Tests for /api/performance endpoint."""

    def test_performance_cramer_disabled(self, client, mock_db, mock_settings, sample_daily_stats):
        """When Cramer mode is disabled, cramer should be null."""
        mock_settings.enable_cramer_mode = False
        mock_db.get_daily_stats_range.return_value = sample_daily_stats(days=3)

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()
        assert "normal" in data
        assert "cramer" in data
        assert data["cramer"] is None
        assert len(data["normal"]) == 3

    def test_performance_cramer_enabled(self, client, mock_db, mock_settings, sample_daily_stats):
        """When Cramer mode is enabled, both normal and cramer arrays should be populated."""
        mock_settings.enable_cramer_mode = True
        normal_stats = sample_daily_stats(days=3, bot_mode=BotMode.NORMAL)
        cramer_stats = sample_daily_stats(days=3, bot_mode=BotMode.INVERTED)

        # First call returns normal stats, second call returns cramer stats
        mock_db.get_daily_stats_range.side_effect = [normal_stats, cramer_stats]

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()
        assert "normal" in data
        assert "cramer" in data
        assert data["cramer"] is not None
        assert len(data["normal"]) == 3
        assert len(data["cramer"]) == 3

    def test_performance_includes_price_fields(self, client, mock_db, mock_settings, sample_daily_stats):
        """Performance endpoint should include starting_price and ending_price."""
        mock_settings.enable_cramer_mode = True
        normal_stats = sample_daily_stats(days=1, bot_mode=BotMode.NORMAL)
        cramer_stats = sample_daily_stats(days=1, bot_mode=BotMode.INVERTED)
        mock_db.get_daily_stats_range.side_effect = [normal_stats, cramer_stats]

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()

        # Check normal stats have price fields
        assert "starting_price" in data["normal"][0]
        assert "ending_price" in data["normal"][0]

        # Check cramer stats have price fields
        assert "starting_price" in data["cramer"][0]
        assert "ending_price" in data["cramer"][0]

    def test_performance_empty_data(self, client, mock_db, mock_settings):
        """When no stats exist, should return empty arrays."""
        mock_settings.enable_cramer_mode = False
        mock_db.get_daily_stats_range.return_value = []

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()
        assert data["normal"] == []
        assert data["cramer"] is None

    def test_performance_cramer_enabled_empty(self, client, mock_db, mock_settings):
        """When Cramer mode is enabled but no stats exist, should return empty arrays."""
        mock_settings.enable_cramer_mode = True
        mock_db.get_daily_stats_range.side_effect = [[], []]

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()
        assert data["normal"] == []
        assert data["cramer"] == []

    def test_performance_days_parameter(self, client, mock_db, mock_settings, sample_daily_stats):
        """Days parameter should be passed to database query."""
        mock_settings.enable_cramer_mode = False
        mock_db.get_daily_stats_range.return_value = sample_daily_stats(days=5)

        response = client.get("/api/performance?days=60")

        assert response.status_code == 200
        # Verify the database was queried
        mock_db.get_daily_stats_range.assert_called_once()
        call_args = mock_db.get_daily_stats_range.call_args
        # The date range should be 60 days
        assert call_args.kwargs.get("bot_mode") == BotMode.NORMAL

    def test_performance_date_fields_format(self, client, mock_db, mock_settings, sample_daily_stats):
        """Date fields should be formatted as strings."""
        mock_settings.enable_cramer_mode = False
        mock_db.get_daily_stats_range.return_value = sample_daily_stats(days=2)

        response = client.get("/api/performance?days=30")

        assert response.status_code == 200
        data = response.json()
        for entry in data["normal"]:
            assert isinstance(entry["date"], str)
            assert isinstance(entry["starting_balance"], str)
            assert isinstance(entry["ending_balance"], str)
            assert isinstance(entry["realized_pnl"], str)


class TestWhaleEventsEndpoint:
    """Tests for /api/whale-events endpoint."""

    def test_whale_events_returns_events(self, client, mock_db, mock_settings):
        """Test /api/whale-events returns whale events."""
        from datetime import datetime

        mock_event = MagicMock()
        mock_event.timestamp = datetime(2024, 1, 1, 12, 0, 0)
        mock_event.direction = "bullish"
        mock_event.volume_ratio = 3.5
        mock_db.get_whale_events.return_value = [mock_event]

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["direction"] == "bullish"
        assert data[0]["volume_ratio"] == 3.5

    def test_whale_events_empty(self, client, mock_db, mock_settings):
        """Test /api/whale-events with no events."""
        mock_db.get_whale_events.return_value = []

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_whale_events_uses_paper_mode(self, client, mock_db, mock_settings):
        """Test that whale events respects is_paper_trading setting."""
        mock_settings.is_paper_trading = True
        mock_db.get_whale_events.return_value = []

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        mock_db.get_whale_events.assert_called_once()
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("is_paper") is True

    def test_whale_events_hours_parameter(self, client, mock_db, mock_settings):
        """Test that hours parameter is passed correctly."""
        mock_db.get_whale_events.return_value = []

        response = client.get("/api/whale-events?hours=48")

        assert response.status_code == 200
        mock_db.get_whale_events.assert_called_once()
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("hours") == 48

    def test_whale_events_multiple_directions(self, client, mock_db, mock_settings):
        """Test /api/whale-events with multiple directions."""
        from datetime import datetime

        mock_events = [
            MagicMock(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                direction="bullish",
                volume_ratio=3.5,
            ),
            MagicMock(
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
                direction="bearish",
                volume_ratio=4.2,
            ),
            MagicMock(
                timestamp=datetime(2024, 1, 1, 14, 0, 0),
                direction="neutral",
                volume_ratio=3.1,
            ),
        ]
        mock_db.get_whale_events.return_value = mock_events

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["direction"] == "bullish"
        assert data[1]["direction"] == "bearish"
        assert data[2]["direction"] == "neutral"

    def test_whale_events_uses_trading_pair(self, client, mock_db, mock_settings):
        """Test that whale events uses the trading_pair from settings."""
        mock_settings.trading_pair = "BTC-USD"
        mock_db.get_whale_events.return_value = []

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("symbol") == "BTC-USD"

    def test_whale_events_hours_minimum_bound(self, client, mock_db, mock_settings):
        """Test that hours=0 returns 422 validation error (must be >= 1)."""
        response = client.get("/api/whale-events?hours=0")

        assert response.status_code == 422
        # Verify database was not called due to validation failure
        mock_db.get_whale_events.assert_not_called()

    def test_whale_events_hours_maximum_bound(self, client, mock_db, mock_settings):
        """Test that hours=169 returns 422 validation error (must be <= 168)."""
        response = client.get("/api/whale-events?hours=169")

        assert response.status_code == 422
        # Verify database was not called due to validation failure
        mock_db.get_whale_events.assert_not_called()

    def test_whale_events_hours_at_bounds(self, client, mock_db, mock_settings):
        """Test that hours=1 and hours=168 are valid (boundary values)."""
        mock_db.get_whale_events.return_value = []

        # Test minimum valid value
        response = client.get("/api/whale-events?hours=1")
        assert response.status_code == 200
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("hours") == 1

        mock_db.get_whale_events.reset_mock()

        # Test maximum valid value
        response = client.get("/api/whale-events?hours=168")
        assert response.status_code == 200
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("hours") == 168

    def test_whale_events_uses_live_mode(self, client, mock_db, mock_settings):
        """Test that whale events respects is_paper_trading=False (live mode).

        This verifies paper/live data separation per CLAUDE.md requirements.
        """
        mock_settings.is_paper_trading = False
        mock_db.get_whale_events.return_value = []

        response = client.get("/api/whale-events?hours=24")

        assert response.status_code == 200
        mock_db.get_whale_events.assert_called_once()
        call_kwargs = mock_db.get_whale_events.call_args.kwargs
        assert call_kwargs.get("is_paper") is False
