"""
Unit tests for ReportingService.

Tests cover:
- Daily/weekly/monthly performance reports
- Signal history cleanup
- Hourly market analysis
- Report scheduling logic
- Configuration updates
- Error handling

All tests use mocked components - NO real API calls or database operations.
"""

import asyncio
import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from freezegun import freeze_time
import pandas as pd

from src.daemon.reporting_service import ReportingService, ReportingConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def reporting_config():
    """Default reporting configuration."""
    return ReportingConfig(
        is_paper_trading=True,
        trading_pair="BTC-USD",
        signal_history_retention_days=90,
        hourly_analysis_enabled=True,
        candle_interval="ONE_HOUR",
        candle_limit=100,
        regime_sentiment_enabled=True,
    )


@pytest.fixture
def mock_notifier():
    """Mock Telegram notifier."""
    notifier = Mock()
    notifier.notify_periodic_report = Mock()
    notifier.notify_market_analysis = Mock()
    return notifier


@pytest.fixture
def mock_db():
    """Mock database."""
    db = Mock()
    db.get_daily_stats.return_value = None
    db.get_daily_stats_range.return_value = []
    db.cleanup_signal_history.return_value = 0
    return db


@pytest.fixture
def mock_exchange_client():
    """Mock exchange client."""
    client = Mock()
    client.get_current_price.return_value = Decimal("50000.00")
    candles = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="h"),
        "open": [50000.0] * 100,
        "high": [51000.0] * 100,
        "low": [49000.0] * 100,
        "close": [50500.0] * 100,
        "volume": [1000.0] * 100,
    })
    client.get_candles.return_value = candles
    return client


@pytest.fixture
def mock_signal_scorer():
    """Mock signal scorer."""
    scorer = Mock()
    scorer.calculate_score.return_value = Mock(
        score=50,
        indicators=Mock(
            rsi=50.0,
            macd_histogram=0.0,
            bb_upper=51000.0,
            bb_lower=49000.0,
            ema_slow=50000.0,
        )
    )
    return scorer


@pytest.fixture
def mock_trade_reviewer():
    """Mock AI trade reviewer."""
    reviewer = Mock()
    # Create async mock for analyze_market
    async_result = Mock()
    async_result.reviews = []
    async_result.judge_confidence = 0.8
    async_result.judge_recommendation = "hold"
    async_result.judge_reasoning = "Market is stable"

    async def mock_analyze(*args, **kwargs):
        return async_result

    reviewer.analyze_market = mock_analyze
    return reviewer


@pytest.fixture
def reporting_service(reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer):
    """ReportingService instance with mocked dependencies (no trade reviewer)."""
    service = ReportingService(
        config=reporting_config,
        notifier=mock_notifier,
        db=mock_db,
        exchange_client=mock_exchange_client,
        signal_scorer=mock_signal_scorer,
        trade_reviewer=None,
    )
    yield service
    service.close()


@pytest.fixture
def reporting_service_with_reviewer(reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer, mock_trade_reviewer):
    """ReportingService instance with trade reviewer for hourly analysis tests."""
    service = ReportingService(
        config=reporting_config,
        notifier=mock_notifier,
        db=mock_db,
        exchange_client=mock_exchange_client,
        signal_scorer=mock_signal_scorer,
        trade_reviewer=mock_trade_reviewer,
    )
    yield service
    service.close()


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization_default_state(reporting_service):
    """Test ReportingService initializes with correct default state."""
    assert reporting_service._last_daily_report is None
    assert reporting_service._last_weekly_report is None
    assert reporting_service._last_monthly_report is None
    assert reporting_service._last_cleanup_date is None
    assert reporting_service._last_hourly_analysis is None
    assert reporting_service._pending_post_volatility_analysis is False


def test_initialization_stores_config(reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer):
    """Test ReportingService stores provided config."""
    service = ReportingService(
        reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer
    )
    assert service.config == reporting_config
    assert service.config.trading_pair == "BTC-USD"
    assert service.config.is_paper_trading is True
    service.close()


def test_initialization_with_callbacks(reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer):
    """Test ReportingService stores sentiment callbacks."""
    success_cb = Mock()
    failure_cb = Mock()

    service = ReportingService(
        reporting_config, mock_notifier, mock_db, mock_exchange_client, mock_signal_scorer,
        on_sentiment_success=success_cb,
        on_sentiment_failure=failure_cb,
    )

    assert service._on_sentiment_success == success_cb
    assert service._on_sentiment_failure == failure_cb
    service.close()


def test_close_without_error(reporting_service):
    """Test close() completes without error."""
    reporting_service.close()
    # Should not raise


# ============================================================================
# Configuration Update Tests
# ============================================================================

def test_update_config_replaces_config(reporting_service):
    """Test update_config replaces the configuration."""
    new_config = ReportingConfig(
        is_paper_trading=False,
        trading_pair="ETH-USD",
        signal_history_retention_days=30,
    )

    reporting_service.update_config(new_config)

    assert reporting_service.config == new_config
    assert reporting_service.config.trading_pair == "ETH-USD"
    assert reporting_service.config.is_paper_trading is False


# ============================================================================
# Daily Report Tests
# ============================================================================

def test_check_daily_report_no_stats(reporting_service, mock_db, mock_notifier):
    """Test daily report is skipped when no stats available."""
    mock_db.get_daily_stats.return_value = None

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    mock_notifier.notify_periodic_report.assert_not_called()


def test_check_daily_report_with_valid_stats(reporting_service, mock_db, mock_notifier):
    """Test daily report is sent when valid stats exist."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "10500.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "51000.00"
    stats.total_trades = 5
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    mock_notifier.notify_periodic_report.assert_called_once()
    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["period"] == "Daily"
    assert call_kwargs["trades"] == 5


def test_check_daily_report_once_per_day(reporting_service, mock_db, mock_notifier):
    """Test daily report is only generated once per day."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "10500.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "51000.00"
    stats.total_trades = 5
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()
        reporting_service.check_daily_report()
        reporting_service.check_daily_report()

    # Should only be called once
    assert mock_notifier.notify_periodic_report.call_count == 1


def test_check_daily_report_queries_yesterday(reporting_service, mock_db):
    """Test daily report queries stats for yesterday."""
    with freeze_time("2024-01-15 10:00:00"):
        reporting_service.check_daily_report()

    mock_db.get_daily_stats.assert_called_once()
    call_args = mock_db.get_daily_stats.call_args
    assert call_args[0][0] == date(2024, 1, 14)  # Yesterday


def test_check_daily_report_handles_exception(reporting_service, mock_db, mock_notifier):
    """Test daily report handles exceptions gracefully."""
    stats = Mock()
    stats.starting_balance = "invalid"
    stats.ending_balance = "10500.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "51000.00"
    stats.total_trades = 5
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        # Should not raise
        reporting_service.check_daily_report()


# ============================================================================
# Weekly Report Tests
# ============================================================================

def test_check_weekly_report_not_monday(reporting_service, mock_db, mock_notifier):
    """Test weekly report is skipped on non-Monday days."""
    # Tuesday
    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_weekly_report()

    mock_db.get_daily_stats_range.assert_not_called()


def test_check_weekly_report_on_monday(reporting_service, mock_db, mock_notifier):
    """Test weekly report is generated on Monday."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "11000.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "52000.00"
    stats.total_trades = 10
    mock_db.get_daily_stats_range.return_value = [stats]

    # Monday
    with freeze_time("2024-01-08 00:00:00"):
        reporting_service.check_weekly_report()

    mock_db.get_daily_stats_range.assert_called_once()
    mock_notifier.notify_periodic_report.assert_called_once()
    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["period"] == "Weekly"


def test_check_weekly_report_once_per_week(reporting_service, mock_db, mock_notifier):
    """Test weekly report is only generated once per week."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "11000.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "52000.00"
    stats.total_trades = 10
    mock_db.get_daily_stats_range.return_value = [stats]

    with freeze_time("2024-01-08 00:00:00"):
        reporting_service.check_weekly_report()
        reporting_service.check_weekly_report()

    assert mock_notifier.notify_periodic_report.call_count == 1


def test_check_weekly_report_no_stats(reporting_service, mock_db, mock_notifier):
    """Test weekly report is skipped when no stats available."""
    mock_db.get_daily_stats_range.return_value = []

    with freeze_time("2024-01-08 00:00:00"):
        reporting_service.check_weekly_report()

    mock_notifier.notify_periodic_report.assert_not_called()


# ============================================================================
# Monthly Report Tests
# ============================================================================

def test_check_monthly_report_not_first_day(reporting_service, mock_db, mock_notifier):
    """Test monthly report is skipped on non-first days of month."""
    with freeze_time("2024-01-15 00:00:00"):
        reporting_service.check_monthly_report()

    mock_db.get_daily_stats_range.assert_not_called()


def test_check_monthly_report_on_first_day(reporting_service, mock_db, mock_notifier):
    """Test monthly report is generated on first of month."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "12000.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "55000.00"
    stats.total_trades = 50
    mock_db.get_daily_stats_range.return_value = [stats]

    with freeze_time("2024-02-01 00:00:00"):
        reporting_service.check_monthly_report()

    mock_db.get_daily_stats_range.assert_called_once()
    mock_notifier.notify_periodic_report.assert_called_once()
    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["period"] == "Monthly"


def test_check_monthly_report_once_per_month(reporting_service, mock_db, mock_notifier):
    """Test monthly report is only generated once per month."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "12000.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "55000.00"
    stats.total_trades = 50
    mock_db.get_daily_stats_range.return_value = [stats]

    with freeze_time("2024-02-01 00:00:00"):
        reporting_service.check_monthly_report()
        reporting_service.check_monthly_report()

    assert mock_notifier.notify_periodic_report.call_count == 1


# ============================================================================
# Signal History Cleanup Tests
# ============================================================================

def test_check_signal_history_cleanup_runs_once_per_day(reporting_service, mock_db):
    """Test cleanup runs once per day."""
    mock_db.cleanup_signal_history.return_value = 100

    with freeze_time("2024-01-15 10:00:00"):
        reporting_service.check_signal_history_cleanup()
        reporting_service.check_signal_history_cleanup()
        reporting_service.check_signal_history_cleanup()

    assert mock_db.cleanup_signal_history.call_count == 1


def test_check_signal_history_cleanup_uses_config(reporting_service, mock_db):
    """Test cleanup uses configured retention days."""
    mock_db.cleanup_signal_history.return_value = 50

    with freeze_time("2024-01-15 10:00:00"):
        reporting_service.check_signal_history_cleanup()

    mock_db.cleanup_signal_history.assert_called_with(
        retention_days=90,  # From config
        is_paper=True,      # From config
    )


def test_check_signal_history_cleanup_handles_exception(reporting_service, mock_db):
    """Test cleanup handles exceptions gracefully."""
    mock_db.cleanup_signal_history.side_effect = Exception("Database error")

    with freeze_time("2024-01-15 10:00:00"):
        # Should not raise
        reporting_service.check_signal_history_cleanup()


def test_check_signal_history_cleanup_runs_on_new_day(reporting_service, mock_db):
    """Test cleanup runs again on a new day."""
    mock_db.cleanup_signal_history.return_value = 0

    with freeze_time("2024-01-15 10:00:00") as frozen_time:
        reporting_service.check_signal_history_cleanup()

        frozen_time.move_to("2024-01-16 10:00:00")
        reporting_service.check_signal_history_cleanup()

    assert mock_db.cleanup_signal_history.call_count == 2


# ============================================================================
# Hourly Analysis Tests
# ============================================================================

def test_check_hourly_analysis_disabled(reporting_service):
    """Test hourly analysis is skipped when disabled."""
    reporting_service.config.hourly_analysis_enabled = False

    reporting_service.check_hourly_analysis("high", "trending")

    # No errors should occur


def test_check_hourly_analysis_no_trade_reviewer(reporting_service):
    """Test hourly analysis is skipped without trade reviewer."""
    assert reporting_service.trade_reviewer is None

    reporting_service.check_hourly_analysis("high", "trending")

    # No errors should occur


def test_check_hourly_analysis_low_volatility_skipped(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis is skipped during low volatility."""
    reporting_service_with_reviewer.check_hourly_analysis("low", "ranging")

    mock_notifier.notify_market_analysis.assert_not_called()


def test_check_hourly_analysis_normal_volatility_skipped(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis is skipped during normal volatility."""
    reporting_service_with_reviewer.check_hourly_analysis("normal", "ranging")

    mock_notifier.notify_market_analysis.assert_not_called()


def test_check_hourly_analysis_high_volatility_runs(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis runs during high volatility."""
    with freeze_time("2024-01-15 10:00:00"):
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

    mock_notifier.notify_market_analysis.assert_called_once()


def test_check_hourly_analysis_extreme_volatility_runs(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis runs during extreme volatility."""
    with freeze_time("2024-01-15 10:00:00"):
        reporting_service_with_reviewer.check_hourly_analysis("extreme", "crash")

    mock_notifier.notify_market_analysis.assert_called_once()


def test_check_hourly_analysis_respects_hourly_cooldown(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis respects 1-hour cooldown."""
    with freeze_time("2024-01-15 10:00:00") as frozen_time:
        # First call
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

        # 30 minutes later - should be skipped
        frozen_time.move_to("2024-01-15 10:30:00")
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

    assert mock_notifier.notify_market_analysis.call_count == 1


def test_check_hourly_analysis_runs_after_cooldown(reporting_service_with_reviewer, mock_notifier):
    """Test hourly analysis runs again after cooldown expires."""
    with freeze_time("2024-01-15 10:00:00") as frozen_time:
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

        # 65 minutes later - should run again
        frozen_time.move_to("2024-01-15 11:05:00")
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

    assert mock_notifier.notify_market_analysis.call_count == 2


def test_check_hourly_analysis_post_volatility_no_cooldown(reporting_service_with_reviewer, mock_notifier):
    """Test post-volatility analysis ignores cooldown."""
    with freeze_time("2024-01-15 10:00:00") as frozen_time:
        # First call during high volatility
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

        # Set pending post-volatility analysis
        frozen_time.move_to("2024-01-15 10:30:00")
        reporting_service_with_reviewer.set_pending_post_volatility_analysis(True)

        # Should run even within cooldown
        reporting_service_with_reviewer.check_hourly_analysis("normal", "ranging")

    assert mock_notifier.notify_market_analysis.call_count == 2


def test_set_pending_post_volatility_analysis(reporting_service):
    """Test setting pending post-volatility analysis flag."""
    assert reporting_service._pending_post_volatility_analysis is False

    reporting_service.set_pending_post_volatility_analysis(True)
    assert reporting_service._pending_post_volatility_analysis is True

    reporting_service.set_pending_post_volatility_analysis(False)
    assert reporting_service._pending_post_volatility_analysis is False


def test_check_hourly_analysis_clears_post_volatility_flag(reporting_service_with_reviewer):
    """Test post-volatility analysis clears the pending flag."""
    reporting_service_with_reviewer.set_pending_post_volatility_analysis(True)

    with freeze_time("2024-01-15 10:00:00"):
        reporting_service_with_reviewer.check_hourly_analysis("normal", "ranging")

    assert reporting_service_with_reviewer._pending_post_volatility_analysis is False


def test_check_hourly_analysis_handles_exception(reporting_service_with_reviewer, mock_exchange_client, mock_notifier):
    """Test hourly analysis handles exceptions gracefully."""
    mock_exchange_client.get_current_price.side_effect = Exception("API error")

    with freeze_time("2024-01-15 10:00:00"):
        # Should not raise
        reporting_service_with_reviewer.check_hourly_analysis("high", "trending")

    mock_notifier.notify_market_analysis.assert_not_called()


# ============================================================================
# Performance Calculation Tests
# ============================================================================

def test_daily_report_calculates_portfolio_return(reporting_service, mock_db, mock_notifier):
    """Test daily report correctly calculates portfolio return."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "10500.00"  # 5% gain
    stats.starting_price = "50000.00"
    stats.ending_price = "50000.00"
    stats.total_trades = 1
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["portfolio_return"] == pytest.approx(5.0, rel=0.01)


def test_daily_report_calculates_btc_return(reporting_service, mock_db, mock_notifier):
    """Test daily report correctly calculates BTC return (benchmark)."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "10000.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "55000.00"  # 10% BTC gain
    stats.total_trades = 0
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["btc_return"] == pytest.approx(10.0, rel=0.01)


def test_daily_report_calculates_alpha(reporting_service, mock_db, mock_notifier):
    """Test daily report correctly calculates alpha (outperformance)."""
    stats = Mock()
    stats.starting_balance = "10000.00"
    stats.ending_balance = "10800.00"  # 8% portfolio gain
    stats.starting_price = "50000.00"
    stats.ending_price = "52500.00"   # 5% BTC gain
    stats.total_trades = 5
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    # Alpha = portfolio_return - btc_return = 8% - 5% = 3%
    assert call_kwargs["alpha"] == pytest.approx(3.0, rel=0.01)


def test_daily_report_handles_zero_starting_balance(reporting_service, mock_db, mock_notifier):
    """Test daily report handles zero starting balance gracefully."""
    stats = Mock()
    stats.starting_balance = "0.00"
    stats.ending_balance = "10500.00"
    stats.starting_price = "50000.00"
    stats.ending_price = "51000.00"
    stats.total_trades = 1
    mock_db.get_daily_stats.return_value = stats

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_notifier.notify_periodic_report.call_args[1]
    assert call_kwargs["portfolio_return"] == 0.0


# ============================================================================
# Period Report Date Range Tests
# ============================================================================

def test_weekly_report_queries_correct_date_range(reporting_service, mock_db):
    """Test weekly report queries Monday to Sunday of previous week."""
    mock_db.get_daily_stats_range.return_value = []

    # Monday Jan 15, 2024
    with freeze_time("2024-01-15 00:00:00"):
        reporting_service.check_weekly_report()

    call_args = mock_db.get_daily_stats_range.call_args[0]
    # Previous week: Monday Jan 8 to Sunday Jan 14
    assert call_args[0] == date(2024, 1, 8)
    assert call_args[1] == date(2024, 1, 14)


def test_monthly_report_queries_correct_date_range(reporting_service, mock_db):
    """Test monthly report queries first to last day of previous month."""
    mock_db.get_daily_stats_range.return_value = []

    # Feb 1, 2024
    with freeze_time("2024-02-01 00:00:00"):
        reporting_service.check_monthly_report()

    call_args = mock_db.get_daily_stats_range.call_args[0]
    # Previous month: Jan 1 to Jan 31
    assert call_args[0] == date(2024, 1, 1)
    assert call_args[1] == date(2024, 1, 31)


# ============================================================================
# Async Timeout Tests
# ============================================================================

def test_run_async_with_timeout_returns_result(reporting_service):
    """Test _run_async_with_timeout returns coroutine result."""
    async def sample_coro():
        return "success"

    result = reporting_service._run_async_with_timeout(sample_coro())

    assert result == "success"


def test_run_async_with_timeout_returns_default_on_timeout(reporting_service):
    """Test _run_async_with_timeout returns default on timeout."""
    async def slow_coro():
        await asyncio.sleep(10)  # Will be cancelled
        return "never reached"

    result = reporting_service._run_async_with_timeout(
        slow_coro(),
        timeout=0.01,
        default="timed_out"
    )

    assert result == "timed_out"


# ============================================================================
# Paper vs Live Mode Tests
# ============================================================================

def test_daily_report_queries_paper_mode(reporting_service, mock_db):
    """Test daily report queries with correct is_paper flag."""
    reporting_service.config.is_paper_trading = True

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_db.get_daily_stats.call_args[1]
    assert call_kwargs["is_paper"] is True


def test_daily_report_queries_live_mode(reporting_service, mock_db):
    """Test daily report queries with correct is_paper flag for live mode."""
    reporting_service.config.is_paper_trading = False

    with freeze_time("2024-01-02 00:00:00"):
        reporting_service.check_daily_report()

    call_kwargs = mock_db.get_daily_stats.call_args[1]
    assert call_kwargs["is_paper"] is False


def test_cleanup_queries_paper_mode(reporting_service, mock_db):
    """Test cleanup uses correct is_paper flag."""
    reporting_service.config.is_paper_trading = True

    with freeze_time("2024-01-15 10:00:00"):
        reporting_service.check_signal_history_cleanup()

    call_kwargs = mock_db.cleanup_signal_history.call_args[1]
    assert call_kwargs["is_paper"] is True
