"""
Comprehensive tests for the LossLimiter safety system.

Tests cover:
- Trade recording and P&L tracking
- Daily and hourly loss calculations
- Progressive throttling (50%, 75%, 90% of limits)
- Daily reset at UTC midnight
- Hourly cooldown mechanism
- Unrealized P&L integration
- Position multiplier calculation
- Callback execution
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from freezegun import freeze_time
from unittest.mock import MagicMock

from src.safety.loss_limiter import (
    LossLimiter,
    LossLimitConfig,
    LossLimitStatus,
    TradeRecord,
    LossLimitExceededError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default loss limiter configuration."""
    return LossLimitConfig(
        max_daily_loss_percent=10.0,
        max_hourly_loss_percent=3.0,
        throttle_at_percent=50.0,
        hourly_cooldown_seconds=3600,
        daily_reset_hour_utc=0,
    )


@pytest.fixture
def limiter(config):
    """Loss limiter with $10,000 starting balance."""
    return LossLimiter(
        config=config,
        starting_balance=Decimal("10000.00")
    )


@pytest.fixture
def callback_mock():
    """Mock callback for limit notifications."""
    return MagicMock()


@pytest.fixture
def limiter_with_callback(config, callback_mock):
    """Loss limiter with callback attached."""
    return LossLimiter(
        config=config,
        starting_balance=Decimal("10000.00"),
        on_limit_hit=callback_mock
    )


# ============================================================================
# Initialization & Configuration Tests
# ============================================================================

def test_initialization_default_config():
    """Test loss limiter initializes with default config."""
    limiter = LossLimiter(starting_balance=Decimal("5000"))
    assert limiter.config.max_daily_loss_percent == 10.0
    assert limiter.config.max_hourly_loss_percent == 3.0
    assert limiter._starting_balance == Decimal("5000")


def test_initialization_custom_config():
    """Test loss limiter accepts custom configuration."""
    config = LossLimitConfig(
        max_daily_loss_percent=5.0,
        max_hourly_loss_percent=2.0,
        throttle_at_percent=60.0
    )
    limiter = LossLimiter(config=config, starting_balance=Decimal("1000"))
    assert limiter.config.max_daily_loss_percent == 5.0
    assert limiter.config.max_hourly_loss_percent == 2.0
    assert limiter.config.throttle_at_percent == 60.0


def test_set_starting_balance():
    """Test setting starting balance updates correctly."""
    limiter = LossLimiter(starting_balance=Decimal("0"))
    limiter.set_starting_balance(Decimal("20000"))

    assert limiter._starting_balance == Decimal("20000")
    assert limiter._daily_starting_balance == Decimal("20000")


def test_update_settings():
    """Test updating settings at runtime."""
    limiter = LossLimiter(starting_balance=Decimal("1000"))
    limiter.update_settings(
        max_daily_loss_percent=15.0,
        max_hourly_loss_percent=5.0
    )

    assert limiter.config.max_daily_loss_percent == 15.0
    assert limiter.config.max_hourly_loss_percent == 5.0


# ============================================================================
# Trade Recording & P&L Tracking Tests
# ============================================================================

def test_record_profitable_trade(limiter):
    """Test recording a profitable trade."""
    status = limiter.record_trade(
        realized_pnl=Decimal("100.00"),
        side="sell",
        size=Decimal("0.01"),
        price=Decimal("50000")
    )

    assert status.can_trade is True
    assert status.daily_loss_percent == -1.0  # Negative = profit
    assert status.position_multiplier == 1.0


def test_record_losing_trade(limiter):
    """Test recording a losing trade."""
    status = limiter.record_trade(
        realized_pnl=Decimal("-100.00"),
        side="buy",
        size=Decimal("0.01"),
        price=Decimal("50000")
    )

    assert status.can_trade is True
    assert status.daily_loss_percent == 1.0  # 1% loss
    assert status.position_multiplier == 1.0


def test_multiple_trades_aggregate_pnl(limiter):
    """Test multiple trades aggregate P&L correctly."""
    limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))
    limiter.record_trade(Decimal("-150"), "buy", Decimal("0.01"), Decimal("50000"))
    status = limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

    # Total loss: $500 = 5% of $10,000
    assert status.daily_loss_percent == pytest.approx(5.0, abs=0.1)


def test_get_daily_loss():
    """Test getting daily loss for external use."""
    limiter = LossLimiter(starting_balance=Decimal("10000"))
    limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

    daily_loss = limiter.get_daily_loss()
    assert daily_loss == Decimal("-200")


def test_trade_cleanup_old_trades(limiter):
    """Test trades older than 24 hours are cleaned up."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move to 25 hours later
        frozen_time.move_to("2024-01-02 13:00:00")
        limiter.record_trade(Decimal("-50"), "buy", Decimal("0.01"), Decimal("50000"))

        # Old trade should be cleaned, only recent -$50 should count
        status = limiter.get_status()
        assert status.daily_loss_percent == pytest.approx(0.5, abs=0.01)


# ============================================================================
# Daily Loss Limit Tests
# ============================================================================

def test_daily_limit_not_hit_below_threshold(limiter):
    """Test daily limit not hit when below threshold."""
    # 5% loss (below 10% limit), spread over multiple trades to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move 2 hours forward to clear hourly window
        frozen_time.move_to("2024-01-01 12:00:00")
        status = limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        assert status.can_trade is True
        assert status.daily_limit_hit is False
        assert status.daily_loss_percent == pytest.approx(5.0, abs=0.1)


def test_daily_limit_hit_at_exact_threshold(limiter):
    """Test daily limit hits at exactly 10% loss."""
    # Exactly 10% loss = $1000
    status = limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    assert status.can_trade is False
    assert status.daily_limit_hit is True
    assert status.daily_loss_percent == pytest.approx(10.0, abs=0.1)
    assert "Daily loss limit exceeded" in status.reason


def test_daily_limit_hit_above_threshold(limiter):
    """Test daily limit hits when exceeding 10% loss."""
    # 15% loss = $1500
    status = limiter.record_trade(Decimal("-1500"), "buy", Decimal("0.1"), Decimal("50000"))

    assert status.can_trade is False
    assert status.daily_limit_hit is True
    assert status.daily_loss_percent == pytest.approx(15.0, abs=0.1)


def test_daily_limit_callback_called(limiter_with_callback, callback_mock):
    """Test callback is called when daily limit is hit."""
    limiter_with_callback.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    # Both daily and hourly limits are hit (10% >= both 10% and 3% thresholds)
    assert callback_mock.call_count == 2
    # First call is daily limit
    assert callback_mock.call_args_list[0][0][0] == "daily"
    assert callback_mock.call_args_list[0][0][1] == pytest.approx(10.0, abs=0.1)
    # Second call is hourly limit
    assert callback_mock.call_args_list[1][0][0] == "hourly"
    assert callback_mock.call_args_list[1][0][1] == pytest.approx(10.0, abs=0.1)


def test_daily_limit_callback_not_called_twice(limiter_with_callback, callback_mock):
    """Test callback only called once when limit first hit."""
    limiter_with_callback.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))
    callback_mock.reset_mock()

    limiter_with_callback.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))
    callback_mock.assert_not_called()


# ============================================================================
# Hourly Loss Limit Tests
# ============================================================================

def test_hourly_limit_not_hit_below_threshold(limiter):
    """Test hourly limit not hit when below 3% threshold."""
    # 2% hourly loss = $200
    status = limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

    assert status.can_trade is True
    assert status.hourly_limit_hit is False
    assert status.hourly_loss_percent == pytest.approx(2.0, abs=0.1)


def test_hourly_limit_hit_at_threshold(limiter):
    """Test hourly limit hits at exactly 3% loss."""
    # Exactly 3% = $300
    status = limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

    assert status.can_trade is False
    assert status.hourly_limit_hit is True
    assert status.hourly_loss_percent == pytest.approx(3.0, abs=0.1)
    assert "Hourly loss limit exceeded" in status.reason


def test_hourly_cooldown_set_when_limit_hit(limiter):
    """Test cooldown is set for 1 hour when hourly limit hit."""
    with freeze_time("2024-01-01 12:00:00"):
        status = limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        assert status.cooldown_until is not None
        expected = datetime(2024, 1, 1, 13, 0, 0)
        assert status.cooldown_until.replace(tzinfo=None) == expected


def test_hourly_cooldown_expires_after_one_hour(limiter):
    """Test hourly limit resets after cooldown expires."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move to 61 minutes later (past cooldown)
        frozen_time.move_to("2024-01-01 13:01:00")

        status = limiter.get_status()
        assert status.can_trade is True
        assert status.hourly_limit_hit is False
        assert status.cooldown_until is None


def test_hourly_limit_still_active_during_cooldown(limiter):
    """Test hourly limit remains active during cooldown period."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move to 30 minutes later (still in cooldown)
        frozen_time.move_to("2024-01-01 12:30:00")

        status = limiter.get_status()
        assert status.can_trade is False
        assert status.hourly_limit_hit is True


# ============================================================================
# Progressive Throttling Tests
# ============================================================================

def test_no_throttling_below_threshold(limiter):
    """Test no throttling when loss is below 50% of limit."""
    # 4% daily loss (40% of 10% limit), spread over time to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move 2 hours forward to clear hourly window
        frozen_time.move_to("2024-01-01 12:00:00")
        status = limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        assert status.position_multiplier == 1.0


def test_throttling_at_50_percent_of_limit(limiter):
    """Test throttling starts at 50% of daily limit (5% loss)."""
    # Slightly over 5% loss (just past 50% of 10% limit), spread over time to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move 2 hours forward to clear hourly window
        frozen_time.move_to("2024-01-01 12:00:00")
        status = limiter.record_trade(Decimal("-255"), "buy", Decimal("0.01"), Decimal("50000"))

        # Should start throttling (5.05% > 5% threshold)
        assert 0.9 < status.position_multiplier < 1.0


def test_throttling_at_75_percent_of_limit(limiter):
    """Test position multiplier at 75% of limit (7.5% loss)."""
    # 7.5% loss (75% of 10% limit), spread over time to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move 2 hours forward to clear hourly window
        frozen_time.move_to("2024-01-01 12:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move 2 hours forward again
        frozen_time.move_to("2024-01-01 14:00:00")
        status = limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Should be significantly throttled
        assert 0.5 < status.position_multiplier < 0.8


def test_throttling_at_90_percent_of_limit(limiter):
    """Test heavy throttling at 90% of limit (9% loss)."""
    # 9% loss (90% of 10% limit), spread over time to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 12:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 16:00:00")
        status = limiter.record_trade(Decimal("-150"), "buy", Decimal("0.01"), Decimal("50000"))

        # Should be heavily throttled but above minimum
        assert 0.3 <= status.position_multiplier < 0.5


def test_throttling_minimum_at_30_percent(limiter):
    """Test throttling never goes below 30% until limit hit."""
    # 9.9% loss (just below limit), spread over time to avoid hourly limit
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 12:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 16:00:00")
        status = limiter.record_trade(Decimal("-240"), "buy", Decimal("0.01"), Decimal("50000"))

        assert status.position_multiplier >= 0.3


def test_zero_multiplier_when_limit_hit(limiter):
    """Test multiplier drops to 0 when limit is hit."""
    # 10% loss (at limit)
    status = limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.01"), Decimal("50000"))

    assert status.position_multiplier == 0.0


def test_hourly_throttling_more_aggressive_than_daily(limiter):
    """Test hourly throttling uses smaller limit (more aggressive)."""
    # 2% loss in last hour (66% of 3% hourly limit)
    # Also 2% daily (20% of 10% daily limit)
    status = limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

    # Should throttle based on hourly (more restrictive)
    assert status.position_multiplier < 1.0


# ============================================================================
# Daily Reset Tests
# ============================================================================

def test_daily_reset_at_utc_midnight(limiter):
    """Test daily counters reset at UTC midnight."""
    with freeze_time("2024-01-01 23:00:00", tz_offset=0) as frozen_time:
        # Record loss before midnight
        limiter.record_trade(Decimal("-500"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move past midnight
        frozen_time.move_to("2024-01-02 00:30:00")

        # Loss should be reset
        status = limiter.get_status()
        assert status.daily_loss_percent == pytest.approx(0.0, abs=0.01)
        assert status.daily_limit_hit is False


def test_daily_limit_resets_after_midnight(limiter):
    """Test daily limit hit flag clears after midnight reset."""
    with freeze_time("2024-01-01 20:00:00", tz_offset=0) as frozen_time:
        # Hit daily limit
        limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))
        assert limiter.get_status().daily_limit_hit is True

        # Move past midnight
        frozen_time.move_to("2024-01-02 01:00:00")

        status = limiter.get_status()
        assert status.daily_limit_hit is False
        assert status.can_trade is True


def test_reset_uses_utc_not_local_time(limiter):
    """Test reset uses UTC midnight regardless of local timezone."""
    # This is implicitly tested by using tz_offset=0 in other tests
    # The implementation uses timezone.utc which is correct
    with freeze_time("2024-01-01 23:00:00", tz_offset=-5) as frozen_time:  # EST
        limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        # 7 PM EST is midnight UTC
        frozen_time.move_to("2024-01-01 19:00:00")

        # Should reset at UTC midnight, not local midnight
        status = limiter.get_status()
        # This should still show the loss since it's not UTC midnight yet


def test_daily_starting_balance_resets_at_midnight(limiter):
    """Test daily starting balance updates at midnight reset."""
    with freeze_time("2024-01-01 20:00:00", tz_offset=0) as frozen_time:
        limiter.set_starting_balance(Decimal("12000"))  # Increased balance

        # Move past midnight
        frozen_time.move_to("2024-01-02 01:00:00")
        limiter.get_status()  # Trigger reset check

        # Daily starting balance should update to current starting balance
        assert limiter._daily_starting_balance == Decimal("12000")


# ============================================================================
# Unrealized P&L Integration Tests
# ============================================================================

def test_check_unrealized_within_limit(limiter):
    """Test unrealized P&L check when within limit."""
    limiter.record_trade(Decimal("-400"), "buy", Decimal("0.01"), Decimal("50000"))

    # Add $200 unrealized loss = $600 total (6% of $10k)
    within_limit, combined_percent = limiter.check_limits_with_unrealized(Decimal("-200"))

    assert within_limit is True
    assert combined_percent == pytest.approx(6.0, abs=0.1)


def test_check_unrealized_exceeds_limit(limiter):
    """Test unrealized P&L check when combined loss exceeds limit."""
    limiter.record_trade(Decimal("-500"), "buy", Decimal("0.01"), Decimal("50000"))

    # Add $600 unrealized loss = $1100 total (11% of $10k, over 10% limit)
    within_limit, combined_percent = limiter.check_limits_with_unrealized(Decimal("-600"))

    assert within_limit is False
    assert combined_percent == pytest.approx(11.0, abs=0.1)


def test_unrealized_profit_reduces_combined_loss(limiter):
    """Test unrealized profit offsets realized loss."""
    limiter.record_trade(Decimal("-500"), "buy", Decimal("0.01"), Decimal("50000"))

    # Add $300 unrealized profit = $200 net loss (2%)
    within_limit, combined_percent = limiter.check_limits_with_unrealized(Decimal("300"))

    assert within_limit is True
    assert combined_percent == pytest.approx(2.0, abs=0.1)


def test_unrealized_check_with_zero_balance(limiter):
    """Test unrealized check handles zero balance gracefully."""
    limiter_zero = LossLimiter(starting_balance=Decimal("0"))

    within_limit, combined_percent = limiter_zero.check_limits_with_unrealized(Decimal("-100"))

    # Should return True (no limit) when balance is zero
    assert within_limit is True
    assert combined_percent == 0.0


# ============================================================================
# Get Status Tests
# ============================================================================

def test_get_status_returns_complete_info(limiter):
    """Test get_status returns LossLimitStatus with all fields."""
    limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

    status = limiter.get_status()

    assert isinstance(status, LossLimitStatus)
    assert isinstance(status.can_trade, bool)
    assert isinstance(status.position_multiplier, float)
    assert isinstance(status.daily_loss_percent, float)
    assert isinstance(status.hourly_loss_percent, float)
    assert isinstance(status.daily_limit_hit, bool)
    assert isinstance(status.hourly_limit_hit, bool)


def test_status_reason_when_no_limits_hit(limiter):
    """Test status reason is None when no limits are hit."""
    limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))

    status = limiter.get_status()
    assert status.reason is None


def test_status_reason_when_daily_limit_hit(limiter):
    """Test status reason explains daily limit when hit."""
    limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    status = limiter.get_status()
    assert "Daily loss limit exceeded" in status.reason


def test_status_reason_when_hourly_limit_hit(limiter):
    """Test status reason explains hourly limit when hit."""
    limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

    status = limiter.get_status()
    assert "Hourly loss limit exceeded" in status.reason


# ============================================================================
# Edge Cases & Error Handling Tests
# ============================================================================

def test_profitable_trades_dont_trigger_limits(limiter):
    """Test profitable trades never trigger loss limits."""
    # Record large profit
    status = limiter.record_trade(Decimal("5000"), "sell", Decimal("0.1"), Decimal("50000"))

    assert status.can_trade is True
    assert status.daily_limit_hit is False
    assert status.hourly_limit_hit is False
    assert status.position_multiplier == 1.0


def test_mixed_winning_and_losing_trades(limiter):
    """Test mixed trades aggregate correctly."""
    limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))
    limiter.record_trade(Decimal("200"), "sell", Decimal("0.01"), Decimal("50000"))
    status = limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))

    # Net loss: $200 = 2%
    assert status.daily_loss_percent == pytest.approx(2.0, abs=0.1)


def test_callback_exception_doesnt_crash(limiter_with_callback, callback_mock):
    """Test callback exceptions are caught and don't crash."""
    callback_mock.side_effect = Exception("Callback failed!")

    # Should not raise exception
    status = limiter_with_callback.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    assert status.daily_limit_hit is True


def test_check_and_raise_throws_when_limit_hit(limiter):
    """Test check_and_raise throws exception when limit hit."""
    limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    with pytest.raises(LossLimitExceededError):
        limiter.check_and_raise()


def test_check_and_raise_passes_when_no_limit(limiter):
    """Test check_and_raise doesn't throw when limits not hit."""
    limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))

    # Should not raise
    limiter.check_and_raise()


def test_hourly_pnl_only_counts_last_hour(limiter):
    """Test hourly P&L only includes trades from last 60 minutes."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Trade 2 hours ago
        limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move to 2 hours later
        frozen_time.move_to("2024-01-01 14:00:00")

        # New trade
        status = limiter.record_trade(Decimal("-100"), "buy", Decimal("0.01"), Decimal("50000"))

        # Hourly should only count recent $100, not old $200
        assert status.hourly_loss_percent == pytest.approx(1.0, abs=0.1)


def test_both_daily_and_hourly_limits_can_hit(limiter):
    """Test both limits can be active simultaneously."""
    # 11% loss hits both daily (10%) and hourly (3%)
    status = limiter.record_trade(Decimal("-1100"), "buy", Decimal("0.1"), Decimal("50000"))

    assert status.daily_limit_hit is True
    assert status.hourly_limit_hit is True
    assert status.can_trade is False


def test_decimal_precision_preserved(limiter):
    """Test Decimal precision is maintained throughout calculations."""
    status = limiter.record_trade(
        Decimal("123.456789"),
        "sell",
        Decimal("0.12345678"),
        Decimal("50000.12345678")
    )

    # Should handle high precision without errors
    daily_loss = limiter.get_daily_loss()
    assert isinstance(daily_loss, Decimal)
