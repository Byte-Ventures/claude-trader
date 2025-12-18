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
