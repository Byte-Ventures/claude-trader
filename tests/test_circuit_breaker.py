"""
Comprehensive tests for the CircuitBreaker safety system.

Tests cover:
- State machine transitions (GREEN/YELLOW/RED/BLACK)
- Price movement detection (flash crashes, spikes, sustained drops)
- API and order failure tracking
- Cooldown and auto-recovery mechanisms
- Manual reset functionality
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime, timedelta
from freezegun import freeze_time
from unittest.mock import MagicMock, call

from src.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    BreakerLevel,
    BreakerStatus,
    CircuitBreakerOpenError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default circuit breaker configuration."""
    return CircuitBreakerConfig()


@pytest.fixture
def breaker(config):
    """Circuit breaker instance with default config."""
    return CircuitBreaker(config)


@pytest.fixture
def callback_mock():
    """Mock callback for trip notifications."""
    return MagicMock()


@pytest.fixture
def breaker_with_callback(config, callback_mock):
    """Circuit breaker with callback attached."""
    return CircuitBreaker(config, on_trip=callback_mock)


# ============================================================================
# Initialization & Configuration Tests
# ============================================================================

def test_initialization_default_config():
    """Test circuit breaker initializes with default config."""
    breaker = CircuitBreaker()
    assert breaker.level == BreakerLevel.GREEN
    assert breaker.can_trade is True
    assert breaker.position_multiplier == 1.0


def test_initialization_custom_config():
    """Test circuit breaker accepts custom configuration."""
    config = CircuitBreakerConfig(
        price_drop_yellow=3.0,
        price_drop_red=7.0,
        api_failures_yellow=3,
    )
    breaker = CircuitBreaker(config)
    assert breaker.config.price_drop_yellow == 3.0
    assert breaker.config.price_drop_red == 7.0
    assert breaker.config.api_failures_yellow == 3


def test_initialization_with_callback(callback_mock):
    """Test circuit breaker registers callback correctly."""
    breaker = CircuitBreaker(on_trip=callback_mock)
    breaker.trip(BreakerLevel.YELLOW, "test")
    callback_mock.assert_called_once_with(BreakerLevel.YELLOW, "test")


# ============================================================================
# State Machine Transition Tests
# ============================================================================

def test_green_to_yellow_price_drop(breaker):
    """Test GREEN → YELLOW transition on 5% price drop."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")
        breaker.check_price_movement(95.0)

        assert breaker.level == BreakerLevel.YELLOW
        assert breaker.can_trade is True
        assert breaker.position_multiplier == 0.5


def test_green_to_red_price_drop(breaker):
    """Test GREEN → RED transition on 10% price drop."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")
        breaker.check_price_movement(90.0)

        assert breaker.level == BreakerLevel.RED
        assert breaker.can_trade is False
        assert breaker.position_multiplier == 0.0


def test_green_to_black_order_failures(breaker):
    """Test GREEN → BLACK transition on 3 consecutive order failures."""
    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.GREEN

    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.YELLOW

    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.BLACK
    assert breaker.can_trade is False


def test_yellow_to_red_escalation(breaker):
    """Test YELLOW → RED escalation via API failures."""
    # Trigger YELLOW first (5 API failures)
    for _ in range(5):
        breaker.record_api_failure()
    assert breaker.level == BreakerLevel.YELLOW

    # Escalate to RED (5 more failures = 10 total)
    for _ in range(5):
        breaker.record_api_failure()
    assert breaker.level == BreakerLevel.RED


def test_no_downgrade_on_lower_severity(breaker):
    """Test that lower severity triggers don't downgrade state."""
    # Set to RED
    breaker.trip(BreakerLevel.RED, "test red")

    # Try to trigger YELLOW (should be ignored)
    breaker.trip(BreakerLevel.YELLOW, "test yellow")

    assert breaker.level == BreakerLevel.RED
    assert "test red" in breaker.status.reason


# ============================================================================
# Price Movement Detection Tests
# ============================================================================

def test_flash_crash_5_percent_triggers_yellow(breaker):
    """Test 5% flash crash triggers YELLOW."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        status = breaker.check_price_movement(95.0)

        assert status.level == BreakerLevel.YELLOW
        assert "dropped" in status.reason.lower()


def test_flash_crash_10_percent_triggers_red(breaker):
    """Test 10% flash crash triggers RED."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        status = breaker.check_price_movement(90.0)

        assert status.level == BreakerLevel.RED
        assert "dropped" in status.reason.lower()


def test_price_spike_8_percent_triggers_yellow(breaker):
    """Test 8% price spike triggers YELLOW."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        status = breaker.check_price_movement(108.0)

        assert status.level == BreakerLevel.YELLOW
        assert "spiked" in status.reason.lower()


def test_price_spike_15_percent_triggers_red(breaker):
    """Test 15% price spike triggers RED."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        status = breaker.check_price_movement(115.0)

        assert status.level == BreakerLevel.RED
        assert "spiked" in status.reason.lower()


def test_sustained_24h_crash_20_percent_triggers_red(breaker):
    """Test 20% sustained drop over 24 hours triggers RED.

    Note: This test validates the 24h window logic exists but may not trigger
    if the 1-hour window already triggered RED (10% drop threshold).
    The test confirms at minimum RED is triggered for the 20% drop.
    """
    with freeze_time("2024-01-01 00:00:00") as frozen_time:
        breaker.record_price(100.0)

        # Move to 23 hours later (still in 24h window, out of 1h window)
        frozen_time.move_to("2024-01-01 23:00:00")

        status = breaker.check_price_movement(80.0)

        # Should trigger RED (either from 24h or 1h window, both apply)
        assert status.level == BreakerLevel.RED


def test_price_change_at_exact_threshold(breaker):
    """Test boundary condition: price change exactly at threshold."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        # Exactly 5% drop (boundary for YELLOW)
        breaker.check_price_movement(95.0)

        # Should trigger (>= comparison)
        assert breaker.level == BreakerLevel.YELLOW


def test_price_history_cleanup(breaker):
    """Test price history is cleaned up after time window."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        breaker.record_price(101.0)
        breaker.record_price(102.0)

        # Move past 1-hour window
        frozen_time.move_to("2024-01-01 13:05:00")
        breaker.record_price(103.0)

        # Old entries should be cleaned
        assert len(breaker._price_history) < 4


def test_empty_price_history_no_crash(breaker):
    """Test that empty price history doesn't cause errors."""
    # Should not crash with empty history
    status = breaker.check_price_movement(100.0)
    assert status.level == BreakerLevel.GREEN


# ============================================================================
# Failure Counter Tests
# ============================================================================

def test_api_failures_5_triggers_yellow(breaker):
    """Test 5 consecutive API failures trigger YELLOW."""
    for i in range(4):
        breaker.record_api_failure()
        assert breaker.level == BreakerLevel.GREEN

    breaker.record_api_failure()
    assert breaker.level == BreakerLevel.YELLOW


def test_api_failures_10_triggers_red(breaker):
    """Test 10 consecutive API failures trigger RED."""
    for _ in range(10):
        breaker.record_api_failure()

    assert breaker.level == BreakerLevel.RED


def test_api_success_resets_counter(breaker):
    """Test API success resets failure counter."""
    # Accumulate some failures
    for _ in range(4):
        breaker.record_api_failure()

    # Success resets counter
    breaker.record_api_success()

    # Should need 5 more failures to trigger YELLOW
    for i in range(4):
        breaker.record_api_failure()
        assert breaker.level == BreakerLevel.GREEN


def test_order_failures_2_triggers_yellow(breaker):
    """Test 2 consecutive order failures trigger YELLOW."""
    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.GREEN

    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.YELLOW


def test_order_failures_3_triggers_black(breaker):
    """Test 3 consecutive order failures trigger BLACK."""
    for _ in range(3):
        breaker.record_order_failure()

    assert breaker.level == BreakerLevel.BLACK
    assert "manual reset required" in breaker.status.reason.lower()


def test_order_success_resets_counter(breaker):
    """Test order success resets failure counter."""
    breaker.record_order_failure()
    breaker.record_order_success()

    # Counter reset, should need 2 more for YELLOW
    breaker.record_order_failure()
    assert breaker.level == BreakerLevel.GREEN


# ============================================================================
# Cooldown & Auto-Recovery Tests
# ============================================================================

def test_yellow_cooldown_5min_recovery(breaker):
    """Test YELLOW state auto-recovers to GREEN after 5 min cooldown."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.trip(BreakerLevel.YELLOW, "test")
        assert breaker.level == BreakerLevel.YELLOW

        # Move to just before cooldown expiry
        frozen_time.move_to("2024-01-01 12:04:59")
        assert breaker.level == BreakerLevel.YELLOW

        # Move past cooldown
        frozen_time.move_to("2024-01-01 12:05:01")
        assert breaker.level == BreakerLevel.GREEN


def test_red_cooldown_4hour_recovery(breaker):
    """Test RED state auto-recovers to GREEN after 4 hour cooldown."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.trip(BreakerLevel.RED, "test")
        assert breaker.level == BreakerLevel.RED

        # Move to 3 hours 59 minutes (still in cooldown)
        frozen_time.move_to("2024-01-01 15:59:00")
        assert breaker.level == BreakerLevel.RED

        # Move past 4 hour cooldown
        frozen_time.move_to("2024-01-01 16:01:00")
        assert breaker.level == BreakerLevel.GREEN


def test_black_no_auto_recovery_by_default(breaker):
    """Test BLACK state does not auto-recover without config."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.trip(BreakerLevel.BLACK, "test")

        # Move forward 24 hours
        frozen_time.move_to("2024-01-02 12:00:00")

        # Should still be BLACK
        assert breaker.level == BreakerLevel.BLACK


def test_black_auto_recovery_when_configured():
    """Test BLACK state auto-recovers when black_recovery_hours is set."""
    config = CircuitBreakerConfig(black_recovery_hours=24)
    breaker = CircuitBreaker(config)

    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.trip(BreakerLevel.BLACK, "test")

        # Move to 24 hours + 1 second
        frozen_time.move_to("2024-01-02 12:00:01")

        # Should downgrade to RED
        assert breaker.level == BreakerLevel.RED

        # Move past RED cooldown (4 hours)
        frozen_time.move_to("2024-01-02 16:00:01")

        # Should now be GREEN
        assert breaker.level == BreakerLevel.GREEN


# ============================================================================
# Manual Reset Tests
# ============================================================================

def test_manual_reset_with_valid_code(breaker):
    """Test manual reset succeeds with valid confirmation code."""
    breaker.trip(BreakerLevel.BLACK, "test")

    result = breaker.manual_reset("RESET_CONFIRMED")

    assert result is True
    assert breaker.level == BreakerLevel.GREEN


def test_manual_reset_with_invalid_code(breaker):
    """Test manual reset fails with invalid confirmation code."""
    breaker.trip(BreakerLevel.BLACK, "test")

    result = breaker.manual_reset("WRONG_CODE")

    assert result is False
    assert breaker.level == BreakerLevel.BLACK


def test_manual_reset_clears_counters(breaker):
    """Test manual reset clears failure counters."""
    # Accumulate failures
    for _ in range(5):
        breaker.record_api_failure()
    for _ in range(2):
        breaker.record_order_failure()

    breaker.trip(BreakerLevel.BLACK, "test")
    breaker.manual_reset("RESET_CONFIRMED")

    # Counters should be reset
    # Should take 5 API failures again to trigger YELLOW
    for i in range(4):
        breaker.record_api_failure()
        assert breaker.level == BreakerLevel.GREEN


# ============================================================================
# Properties & Status Tests
# ============================================================================

def test_can_trade_property_green_yellow(breaker):
    """Test can_trade returns True for GREEN and YELLOW."""
    assert breaker.can_trade is True

    breaker.trip(BreakerLevel.YELLOW, "test")
    assert breaker.can_trade is True


def test_can_trade_property_red_black(breaker):
    """Test can_trade returns False for RED and BLACK."""
    breaker.trip(BreakerLevel.RED, "test")
    assert breaker.can_trade is False

    breaker.trip(BreakerLevel.BLACK, "test")
    assert breaker.can_trade is False


def test_position_multiplier_values(breaker):
    """Test position_multiplier returns correct values for each level."""
    assert breaker.position_multiplier == 1.0  # GREEN

    breaker.trip(BreakerLevel.YELLOW, "test")
    assert breaker.position_multiplier == 0.5  # YELLOW

    breaker.trip(BreakerLevel.RED, "test")
    assert breaker.position_multiplier == 0.0  # RED

    breaker.trip(BreakerLevel.BLACK, "test")
    assert breaker.position_multiplier == 0.0  # BLACK


def test_status_property_returns_complete_info(breaker):
    """Test status property returns BreakerStatus with all fields."""
    with freeze_time("2024-01-01 12:00:00"):
        breaker.trip(BreakerLevel.YELLOW, "test reason")

        status = breaker.status

        assert isinstance(status, BreakerStatus)
        assert status.level == BreakerLevel.YELLOW
        assert status.reason == "test reason"
        assert status.triggered_at is not None
        assert status.cooldown_until is not None
        assert status.can_trade is True


def test_check_and_raise_throws_when_red(breaker):
    """Test check_and_raise throws exception when RED or BLACK."""
    breaker.trip(BreakerLevel.RED, "test")

    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        breaker.check_and_raise()

    assert exc_info.value.level == BreakerLevel.RED
    assert "test" in str(exc_info.value)


def test_check_and_raise_passes_when_green_yellow(breaker):
    """Test check_and_raise doesn't throw when GREEN or YELLOW."""
    breaker.check_and_raise()  # GREEN - should pass

    breaker.trip(BreakerLevel.YELLOW, "test")
    breaker.check_and_raise()  # YELLOW - should pass


# ============================================================================
# Edge Cases & Error Handling Tests
# ============================================================================

def test_callback_exception_doesnt_crash(breaker_with_callback, callback_mock):
    """Test that callback exceptions are caught and don't crash the system."""
    callback_mock.side_effect = Exception("Callback failed!")

    # Should not raise exception
    breaker_with_callback.trip(BreakerLevel.YELLOW, "test")

    # Breaker should still be in YELLOW state
    assert breaker_with_callback.level == BreakerLevel.YELLOW


def test_multiple_simultaneous_triggers(breaker):
    """Test handling of multiple triggers in quick succession."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:30:00")

        # Trigger price drop (YELLOW)
        breaker.check_price_movement(95.0)

        # Immediately trigger API failures (would be RED)
        for _ in range(10):
            breaker.record_api_failure()

        # Should be at highest severity (RED)
        assert breaker.level == BreakerLevel.RED


def test_cooldown_check_on_property_access(breaker):
    """Test that cooldown expiry is checked on property access."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.trip(BreakerLevel.YELLOW, "test")

        # Move past cooldown
        frozen_time.move_to("2024-01-01 12:06:00")

        # Accessing level property should trigger cooldown check
        level = breaker.level
        assert level == BreakerLevel.GREEN


def test_price_history_boundary_conditions(breaker):
    """Test price history with exactly 2 entries (minimum for comparison)."""
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        breaker.record_price(100.0)
        frozen_time.move_to("2024-01-01 12:01:00")

        # Exactly 2 entries - should work
        status = breaker.check_price_movement(90.0)
        assert status.level == BreakerLevel.RED


def test_callback_receives_correct_parameters(breaker_with_callback, callback_mock):
    """Test callback is called with correct level and reason."""
    breaker_with_callback.trip(BreakerLevel.YELLOW, "test reason")

    callback_mock.assert_called_once_with(BreakerLevel.YELLOW, "test reason")

    # Reset mock
    callback_mock.reset_mock()

    breaker_with_callback.trip(BreakerLevel.RED, "another reason")
    callback_mock.assert_called_once_with(BreakerLevel.RED, "another reason")
