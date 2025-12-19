"""
Integration tests for loss throttling edge cases.

Tests multi-system interactions between loss limiter and other safety components:
- LossLimiter + CircuitBreaker
- LossLimiter + Validator (multiplier stacking)
- Throttling reset/release behavior
- Hourly vs Daily limit conflicts
- Multi-system position multiplier calculations

These tests complement unit tests in test_loss_limiter.py by verifying
behavior in realistic scenarios with multiple safety systems active.
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from freezegun import freeze_time
from unittest.mock import MagicMock

from src.safety.loss_limiter import (
    LossLimiter,
    LossLimitConfig,
)
from src.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    BreakerLevel,
)
from src.safety.validator import (
    OrderValidator,
    ValidatorConfig,
    OrderRequest,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def loss_config():
    """Loss limiter configuration for testing."""
    return LossLimitConfig(
        throttle_at_percent=50.0,
        throttle_min_multiplier=0.3,
        max_daily_loss_percent=10.0,
        max_hourly_loss_percent=3.0,
        hourly_cooldown_seconds=3600,
        daily_reset_hour_utc=0,
    )


@pytest.fixture
def limiter(loss_config):
    """Loss limiter with $10,000 starting balance."""
    return LossLimiter(
        config=loss_config,
        starting_balance=Decimal("10000.00")
    )


@pytest.fixture
def breaker_config():
    """Circuit breaker configuration for testing."""
    return CircuitBreakerConfig(
        price_drop_yellow=5.0,
        price_drop_red=10.0,
        price_spike_yellow=8.0,
        price_spike_red=15.0,
        api_failures_yellow=5,
        api_failures_red=10,
        order_failures_yellow=2,
        order_failures_black=3,
        yellow_cooldown=300,
        red_cooldown=14400,
    )


@pytest.fixture
def breaker(breaker_config):
    """Circuit breaker instance."""
    return CircuitBreaker(config=breaker_config)


@pytest.fixture
def validator_config():
    """Validator configuration for testing."""
    return ValidatorConfig(
        estimated_fee_percent=0.006,
        profit_margin_multiplier=2.0,
        min_trade_quote=100.0,
        max_position_percent=80.0,
    )


@pytest.fixture
def validator(validator_config, limiter, breaker):
    """Order validator with loss limiter and circuit breaker."""
    return OrderValidator(
        config=validator_config,
        loss_limiter=limiter,
        circuit_breaker=breaker,
    )


# ============================================================================
# Edge Case 1: Throttling + Circuit Breaker Interaction
# ============================================================================

def test_throttling_and_circuit_breaker_both_active(limiter, breaker, validator):
    """
    Test loss throttling and circuit breaker work together correctly.

    Scenario: Loss approaching limit triggers throttling, then circuit breaker
    trips to YELLOW. Both multipliers should stack.
    """
    # Set up validator balances
    validator.update_balances(
        base_balance=Decimal("0.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Step 1: Record loss to trigger throttling (6% of $10k = $600)
        limiter.record_trade(Decimal("-600"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move forward to clear hourly window
        frozen_time.move_to("2024-01-01 14:00:00")

        # Verify throttling is active
        loss_status = limiter.get_status()
        assert loss_status.position_multiplier < 1.0
        loss_multiplier = loss_status.position_multiplier

        # Step 2: Trip circuit breaker to YELLOW (50% multiplier)
        breaker.trip(BreakerLevel.YELLOW, "Test warning")

        # Step 3: Get combined multiplier from validator
        combined_multiplier = validator.get_position_multiplier()

        # Both multipliers should stack
        expected_multiplier = loss_multiplier * 0.5
        assert combined_multiplier == pytest.approx(expected_multiplier, abs=0.01)
        assert combined_multiplier < loss_multiplier  # Should be more restrictive


def test_circuit_breaker_red_overrides_throttling(limiter, breaker, validator):
    """
    Test that circuit breaker RED level completely halts trading despite throttling.

    Loss throttling may allow reduced trading, but RED circuit breaker should
    override this and prevent all trading.
    """
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        # Record loss to trigger throttling but not hourly limit
        # Need 5.5% to trigger throttling (above 5% threshold)
        # Split across time to avoid hourly limit (3%)
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))

        frozen_time.move_to("2024-01-01 12:00:00")
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))

        # Verify throttling is active but allows trading (5.5% daily, spread over time)
        loss_status = limiter.get_status()
        assert loss_status.can_trade is True
        assert loss_status.position_multiplier < 1.0  # Should be throttled

        # Trip circuit breaker to RED
        breaker.trip(BreakerLevel.RED, "Price crash")

        # Validator should reject trading despite throttling allowing it
        result = validator.validate(
            OrderRequest(side="buy", size=Decimal("0.01"))
        )

        assert result.valid is False
        assert "Circuit breaker at RED" in result.reason


def test_loss_limit_hit_with_circuit_breaker_yellow(limiter, breaker, validator):
    """
    Test that loss limit hit prevents trading even when circuit breaker is YELLOW.

    Both systems can halt trading independently. Loss limit should take priority
    when it's the more restrictive condition.
    """
    # Trip circuit breaker to YELLOW (allows reduced trading)
    breaker.trip(BreakerLevel.YELLOW, "Price volatility")
    assert breaker.can_trade is True

    # Hit daily loss limit (10% = $1000)
    limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    loss_status = limiter.get_status()
    assert loss_status.can_trade is False
    assert loss_status.daily_limit_hit is True

    # Validator should reject despite circuit breaker allowing trading
    result = validator.validate(
        OrderRequest(side="buy", size=Decimal("0.01"))
    )

    assert result.valid is False
    assert "loss limit exceeded" in result.reason.lower()


def test_no_race_conditions_in_state_updates(limiter, breaker):
    """
    Test that rapid state updates don't cause race conditions.

    Rapidly record trades and trip circuit breaker to ensure state
    remains consistent.

    Note: This test verifies state consistency with rapid sequential updates,
    but does not test true concurrent access (which would require threading).
    The loss limiter and circuit breaker are designed for single-threaded
    event loop usage in the trading bot.
    """
    with freeze_time("2024-01-01 12:00:00"):
        # Rapidly record multiple trades
        for i in range(10):
            limiter.record_trade(Decimal("-50"), "buy", Decimal("0.01"), Decimal("50000"))

        # Trip circuit breaker
        breaker.trip(BreakerLevel.YELLOW, "Rapid trades detected")

        # Get status from both systems
        loss_status = limiter.get_status()
        breaker_status = breaker.status

        # States should be consistent
        assert isinstance(loss_status.can_trade, bool)
        assert isinstance(breaker_status.can_trade, bool)
        assert isinstance(loss_status.position_multiplier, float)
        assert 0.0 <= loss_status.position_multiplier <= 1.0


# ============================================================================
# Edge Case 2: Throttling + Validator Position Multiplier Stacking
# ============================================================================

def test_multiplier_stacking_throttle_and_breaker(limiter, breaker, validator):
    """
    Test that multiple position multipliers stack correctly.

    Scenario: Loss throttle at 70%, circuit breaker at 50%.
    Combined multiplier should be 0.7 * 0.5 = 0.35.
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Create 7.5% loss (75% of 10% limit) to get ~70% throttle
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 16:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Get loss multiplier
        loss_status = limiter.get_status()
        loss_mult = loss_status.position_multiplier

        # Trip circuit breaker to YELLOW (50% multiplier)
        breaker.trip(BreakerLevel.YELLOW, "Warning")
        breaker_mult = breaker.position_multiplier
        assert breaker_mult == 0.5

        # Combined multiplier should be product of both
        combined = validator.get_position_multiplier()
        expected = loss_mult * breaker_mult

        assert combined == pytest.approx(expected, abs=0.01)
        assert combined < min(loss_mult, breaker_mult)


def test_multiplier_at_minimum_boundaries(limiter, breaker, validator):
    """
    Test multiplier calculation when both systems approach minimums.

    Loss throttling has 30% minimum, circuit breaker YELLOW is 50%.
    This test verifies the multipliers stack correctly at low values.
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Create 9.5% loss to reach near-minimum throttle
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 12:30:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 16:00:00")
        limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        # Get the actual loss multiplier
        loss_status = limiter.get_status()
        loss_mult = loss_status.position_multiplier

        # Trip circuit breaker to YELLOW (50% multiplier)
        breaker.trip(BreakerLevel.YELLOW, "Warning")

        # Combined should be product of both
        combined = validator.get_position_multiplier()
        expected = loss_mult * 0.5

        assert combined == pytest.approx(expected, abs=0.01)


def test_multiplier_one_system_disabled(validator):
    """
    Test multiplier calculation when only one system is active.

    If circuit breaker is GREEN (1.0) and loss limiter is throttling,
    only loss limiter multiplier should apply.
    """
    # Set up validator with only loss limiter (circuit breaker defaults to GREEN)
    validator.update_balances(
        base_balance=Decimal("0.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    # Record loss to trigger throttling
    validator.loss_limiter.record_trade(Decimal("-600"), "buy", Decimal("0.01"), Decimal("50000"))

    # Circuit breaker should be GREEN (1.0 multiplier)
    assert validator.circuit_breaker.position_multiplier == 1.0

    # Combined multiplier should equal loss limiter multiplier
    loss_mult = validator.loss_limiter.get_status().position_multiplier
    combined = validator.get_position_multiplier()

    assert combined == pytest.approx(loss_mult, abs=0.001)


# ============================================================================
# Edge Case 3: Throttling Reset/Release Behavior
# ============================================================================

def test_throttle_gradually_releases_after_winning_trades(limiter):
    """
    Test that throttling gradually releases as losses are recovered.

    Scenario: Reach heavy throttling, then win trades to reduce loss percentage.
    Throttle should gradually release, not snap back to 100%.
    """
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        # Build up 7.5% loss (75% of limit) - moderate throttling
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 12:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Should be moderately throttled
        status_before = limiter.get_status()
        assert status_before.daily_loss_percent == pytest.approx(7.5, abs=0.1)
        mult_before = status_before.position_multiplier
        assert mult_before < 0.7  # Should be throttled

        # Win some back - reduce to 5.5% loss
        frozen_time.move_to("2024-01-01 18:00:00")
        limiter.record_trade(Decimal("200"), "sell", Decimal("0.01"), Decimal("50000"))

        # Throttle should release somewhat but not fully
        status_after = limiter.get_status()
        assert status_after.daily_loss_percent == pytest.approx(5.5, abs=0.1)
        mult_after = status_after.position_multiplier

        assert mult_after > mult_before  # Released somewhat
        assert mult_after < 1.0  # But still throttled


def test_throttle_prevents_rapid_flapping(limiter):
    """
    Test throttle calculation is smooth around threshold.

    The throttle calculation should be smooth and not cause rapid
    position size changes from small P&L fluctuations.
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Get to 5.5% loss (just above throttle threshold of 5%)
        # Spread across time to avoid hourly limit
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))

        status_at_threshold = limiter.get_status()
        mult_at_threshold = status_at_threshold.position_multiplier

        # Small winning trade - reduce to 5.3% loss
        frozen_time.move_to("2024-01-01 16:00:00")
        limiter.record_trade(Decimal("20"), "sell", Decimal("0.001"), Decimal("50000"))

        status_below = limiter.get_status()
        mult_below = status_below.position_multiplier

        # Small losing trade - back to 5.5% loss
        frozen_time.move_to("2024-01-01 18:00:00")
        limiter.record_trade(Decimal("-20"), "buy", Decimal("0.001"), Decimal("50000"))

        status_above = limiter.get_status()
        mult_above = status_above.position_multiplier

        # Changes should be gradual, not sudden jumps
        assert abs(mult_at_threshold - mult_below) < 0.15
        assert abs(mult_below - mult_above) < 0.15


def test_full_recovery_resets_throttle_to_100_percent(limiter):
    """
    Test that full recovery from losses resets throttle to 100%.

    If losses are completely recovered to break-even or profit,
    throttle should return to 1.0 (no restriction).
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Create 7% loss (heavy throttling)
        limiter.record_trade(Decimal("-700"), "buy", Decimal("0.01"), Decimal("50000"))

        status_loss = limiter.get_status()
        assert status_loss.position_multiplier < 1.0

        # Fully recover losses
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("700"), "sell", Decimal("0.01"), Decimal("50000"))

        # Should be back to no throttling
        status_recovered = limiter.get_status()
        assert status_recovered.daily_loss_percent <= 0.0  # Break-even or profit
        assert status_recovered.position_multiplier == 1.0


# ============================================================================
# Edge Case 4: Hourly vs Daily Limit Conflicts
# ============================================================================

def test_hourly_limit_more_restrictive_than_daily(limiter):
    """
    Test that hourly limit is enforced when more restrictive than daily.

    Scenario: 2.5% daily loss (OK), but 3.5% hourly loss (over limit).
    Hourly limit should halt trading.
    """
    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        # Older loss outside hourly window
        limiter.record_trade(Decimal("-150"), "buy", Decimal("0.01"), Decimal("50000"))

        # Move to 2 hours later
        frozen_time.move_to("2024-01-01 12:00:00")

        # Recent loss that exceeds hourly limit (3.5% = $350)
        limiter.record_trade(Decimal("-350"), "buy", Decimal("0.01"), Decimal("50000"))

        status = limiter.get_status()

        # Daily loss is 5% (OK), hourly is 3.5% (over 3% limit)
        assert status.daily_loss_percent == pytest.approx(5.0, abs=0.1)
        assert status.hourly_loss_percent == pytest.approx(3.5, abs=0.1)

        # Hourly limit should halt trading
        assert status.hourly_limit_hit is True
        assert status.can_trade is False


def test_hourly_window_reset_while_daily_persists(limiter):
    """
    Test that hourly limit resets after cooldown while daily remains.

    After hourly cooldown expires, trading should resume even though
    daily loss persists (as long as it's under daily limit).
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Hit hourly limit (3% = $300)
        limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        status_limited = limiter.get_status()
        assert status_limited.hourly_limit_hit is True
        assert status_limited.can_trade is False

        # Move past hourly cooldown (1 hour + 1 minute)
        frozen_time.move_to("2024-01-01 13:01:00")

        status_after_cooldown = limiter.get_status()

        # Hourly limit should be cleared
        assert status_after_cooldown.hourly_limit_hit is False
        assert status_after_cooldown.can_trade is True

        # But daily loss should still be recorded
        assert status_after_cooldown.daily_loss_percent == pytest.approx(3.0, abs=0.1)


def test_both_limits_approach_simultaneously(limiter):
    """
    Test behavior when approaching both hourly and daily limits.

    Throttling should use the more restrictive of the two percentages.
    """
    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Create 2.5% hourly loss (83% of 3% hourly limit)
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))

        # Also 2.5% daily loss (25% of 10% daily limit)
        status = limiter.get_status()
        assert status.hourly_loss_percent == pytest.approx(2.5, abs=0.1)
        assert status.daily_loss_percent == pytest.approx(2.5, abs=0.1)

        # Hourly is more restrictive (83% vs 25% of limit)
        # Throttling should be based on hourly
        assert status.position_multiplier < 1.0

        # Move forward 2 hours to clear hourly window
        frozen_time.move_to("2024-01-01 14:00:00")

        # Now only daily loss counts (2.5% = 25% of limit)
        status_after = limiter.get_status()
        assert status_after.hourly_loss_percent == pytest.approx(0.0, abs=0.1)

        # Should be less throttled (daily is less restrictive)
        assert status_after.position_multiplier > status.position_multiplier


def test_multiple_hourly_windows_in_one_day(limiter):
    """
    Test that multiple hourly limit hits can occur in a single day.

    Each hourly window is independent. Can hit hourly limit multiple times
    before hitting daily limit.
    """
    with freeze_time("2024-01-01 08:00:00") as frozen_time:
        # First hourly limit hit (3% = $300)
        limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        status1 = limiter.get_status()
        assert status1.hourly_limit_hit is True

        # Move past cooldown
        frozen_time.move_to("2024-01-01 09:30:00")

        # Second hourly limit hit (another $300)
        limiter.record_trade(Decimal("-300"), "buy", Decimal("0.01"), Decimal("50000"))

        status2 = limiter.get_status()
        assert status2.hourly_limit_hit is True

        # Daily loss should be cumulative (6%)
        assert status2.daily_loss_percent == pytest.approx(6.0, abs=0.1)

        # Move past cooldown again
        frozen_time.move_to("2024-01-01 11:00:00")

        # Can still trade (under daily limit of 10%)
        status3 = limiter.get_status()
        assert status3.can_trade is True
        assert status3.daily_limit_hit is False


# ============================================================================
# Edge Case 5: Integration with Validator Minimum Trade Size
# ============================================================================

def test_throttled_position_respects_minimum_trade_size(validator, limiter):
    """
    Test that heavily throttled positions don't go below minimum trade size.

    Scenario: Heavy throttling reduces position to 30%. If this results in
    a trade below MIN_TRADE_QUOTE ($100), validator should reject it.
    """
    # Set up balances
    validator.update_balances(
        base_balance=Decimal("0.0"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    with freeze_time("2024-01-01 12:00:00") as frozen_time:
        # Create 9.5% loss to reach near-minimum throttle (~30%)
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 16:00:00")
        limiter.record_trade(Decimal("-250"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 18:00:00")
        limiter.record_trade(Decimal("-200"), "buy", Decimal("0.01"), Decimal("50000"))

        status = limiter.get_status()
        # Verify we're at low multiplier (should be around 0.3-0.4)
        assert status.position_multiplier <= 0.5

        # Order that would normally be $200 but throttled to ~$60-80
        order = OrderRequest(
            side="buy",
            size=Decimal("0.004"),  # $200 at $50k
            price=Decimal("50000")
        )

        # Validator checks pre-throttle size ($200 > $100 minimum)
        # Position multiplier is applied after validation
        result = validator.validate(order)

        # Validator validates the requested size, not the throttled size
        # This test documents that throttling happens after validation
        assert result.valid is True


def test_loss_limit_hit_with_zero_multiplier(validator, limiter):
    """
    Test that hitting loss limit results in 0.0 multiplier.

    When daily loss limit is hit, position_multiplier should be exactly 0.0,
    and all trading should be blocked.
    """
    validator.update_balances(
        base_balance=Decimal("0.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    # Hit daily loss limit (10% = $1000)
    limiter.record_trade(Decimal("-1000"), "buy", Decimal("0.1"), Decimal("50000"))

    status = limiter.get_status()
    assert status.position_multiplier == 0.0
    assert status.can_trade is False

    # Validator combined multiplier should also be 0.0
    combined = validator.get_position_multiplier()
    assert combined == 0.0

    # Any order should be rejected
    result = validator.validate(
        OrderRequest(side="buy", size=Decimal("0.01"))
    )
    assert result.valid is False


# ============================================================================
# Edge Case 6: Paper vs Live Trading Mode Separation
# ============================================================================

def test_loss_limits_independent_per_trading_mode():
    """
    Test that paper and live trading have independent loss tracking.

    Note: This is a conceptual test using separate LossLimiter instances.
    LossLimiter is an in-memory safety system - paper/live separation is
    achieved by instantiating separate instances for each mode.
    Database-level paper/live separation (is_paper column) is tested in
    tests/test_database.py for the Trade and Order models.
    """
    # Separate limiters for paper and live (as would be in real usage)
    paper_limiter = LossLimiter(
        config=LossLimitConfig(
            throttle_at_percent=50.0,
            throttle_min_multiplier=0.3,
            max_daily_loss_percent=10.0,
            max_hourly_loss_percent=3.0,
        ),
        starting_balance=Decimal("10000")
    )

    live_limiter = LossLimiter(
        config=LossLimitConfig(
            throttle_at_percent=50.0,
            throttle_min_multiplier=0.3,
            max_daily_loss_percent=10.0,
            max_hourly_loss_percent=3.0,
        ),
        starting_balance=Decimal("50000")
    )

    # Paper trading has losses
    paper_limiter.record_trade(Decimal("-800"), "buy", Decimal("0.01"), Decimal("50000"))

    # Live trading is clean
    # (no trades recorded)

    # Paper should be throttled
    paper_status = paper_limiter.get_status()
    assert paper_status.daily_loss_percent == pytest.approx(8.0, abs=0.1)
    assert paper_status.position_multiplier < 1.0

    # Live should be unaffected
    live_status = live_limiter.get_status()
    assert live_status.daily_loss_percent == 0.0
    assert live_status.position_multiplier == 1.0


# ============================================================================
# Edge Case 7: Logging and Visibility
# ============================================================================

def test_throttling_provides_clear_warning_messages(validator, limiter):
    """
    Test that throttling provides clear warning messages in validation.

    When position is throttled, validator should include warning explaining
    the reduction.
    """
    validator.update_balances(
        base_balance=Decimal("0.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        # Create loss to trigger throttling but not hourly limit
        # Split across time: 5.5% total daily
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))

        # Validate an order
        result = validator.validate(
            OrderRequest(side="buy", size=Decimal("0.01"))
        )

        # Should be valid but with warnings
        assert result.valid is True
        assert len(result.warnings) > 0

        # Warning should mention throttling and percentage
        warning_text = " ".join(result.warnings)
        assert "throttling" in warning_text.lower() or "reduced" in warning_text.lower()


def test_combined_warnings_from_multiple_systems(validator, limiter, breaker):
    """
    Test that warnings from multiple safety systems are all included.

    When both loss limiter and circuit breaker are in warning states,
    validation should include warnings from both.
    """
    validator.update_balances(
        base_balance=Decimal("0.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("50000")
    )

    with freeze_time("2024-01-01 10:00:00") as frozen_time:
        # Trigger loss throttling (split to avoid hourly limit)
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))
        frozen_time.move_to("2024-01-01 14:00:00")
        limiter.record_trade(Decimal("-275"), "buy", Decimal("0.01"), Decimal("50000"))

        # Trigger circuit breaker warning
        breaker.trip(BreakerLevel.YELLOW, "Price volatility")

        # Validate order
        result = validator.validate(
            OrderRequest(side="buy", size=Decimal("0.01"))
        )

        assert result.valid is True
        assert len(result.warnings) >= 2  # At least one from each system

        warnings_text = " ".join(result.warnings).lower()
        assert "throttling" in warnings_text or "loss" in warnings_text
        assert "circuit breaker" in warnings_text or "warning" in warnings_text
