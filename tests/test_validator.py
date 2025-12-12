"""
Comprehensive tests for the OrderValidator pre-trade validation system.

Tests cover:
- All 7 validation checks individually
- Kill switch check (highest priority)
- Circuit breaker check
- Loss limiter check
- Balance sufficiency check
- Position limits check
- Price sanity check (limit orders)
- Minimum trade size check
- Check prioritization and short-circuit behavior
- Warning vs rejection handling
- Position multiplier aggregation
- Integration with mocked safety dependencies
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from src.safety.validator import (
    OrderValidator,
    OrderRequest,
    ValidationResult,
    ValidatorConfig,
)
from src.safety.circuit_breaker import BreakerLevel, BreakerStatus
from src.safety.loss_limiter import LossLimitStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default validator configuration."""
    return ValidatorConfig(
        min_trade_quote=10.0,
        max_position_percent=80.0,
        price_sanity_percent=5.0,
    )


@pytest.fixture
def validator(config):
    """Validator without safety systems."""
    validator = OrderValidator(config=config)
    # Set balances: 0.1 BTC, $5000 USD, price $50,000
    validator.update_balances(
        base_balance=Decimal("0.1"),
        quote_balance=Decimal("5000"),
        current_price=Decimal("50000")
    )
    return validator


@pytest.fixture
def kill_switch_mock():
    """Mock kill switch."""
    mock = MagicMock()
    mock.is_active = False
    mock.reason = None
    return mock


@pytest.fixture
def circuit_breaker_mock():
    """Mock circuit breaker."""
    mock = MagicMock()
    mock.status = BreakerStatus(
        level=BreakerLevel.GREEN,
        can_trade=True,
        reason=None,
        triggered_at=None,
        cooldown_until=None
    )
    mock.position_multiplier = 1.0
    return mock


@pytest.fixture
def loss_limiter_mock():
    """Mock loss limiter."""
    mock = MagicMock()
    mock.get_status.return_value = LossLimitStatus(
        can_trade=True,
        position_multiplier=1.0,
        daily_loss_percent=0.0,
        hourly_loss_percent=0.0,
        daily_limit_hit=False,
        hourly_limit_hit=False,
        cooldown_until=None,
        reason=None
    )
    return mock


@pytest.fixture
def validator_with_safety(config, kill_switch_mock, circuit_breaker_mock, loss_limiter_mock):
    """Validator with all safety systems mocked."""
    validator = OrderValidator(
        config=config,
        kill_switch=kill_switch_mock,
        circuit_breaker=circuit_breaker_mock,
        loss_limiter=loss_limiter_mock
    )
    validator.update_balances(
        base_balance=Decimal("0.1"),
        quote_balance=Decimal("5000"),
        current_price=Decimal("50000")
    )
    return validator


# ============================================================================
# Initialization & Configuration Tests
# ============================================================================

def test_initialization_default_config():
    """Test validator initializes with default config."""
    validator = OrderValidator()
    assert validator.config.min_trade_quote == 100.0
    assert validator.config.max_position_percent == 80.0
    assert validator.config.price_sanity_percent == 5.0


def test_initialization_custom_config():
    """Test validator accepts custom configuration."""
    config = ValidatorConfig(
        min_trade_quote=20.0,
        max_position_percent=90.0,
        price_sanity_percent=10.0
    )
    validator = OrderValidator(config=config)
    assert validator.config.min_trade_quote == 20.0
    assert validator.config.max_position_percent == 90.0


def test_update_settings_at_runtime():
    """Test updating settings at runtime."""
    validator = OrderValidator()
    validator.update_settings(max_position_percent=75.0)

    assert validator.config.max_position_percent == 75.0


def test_update_balances():
    """Test updating balances for validation."""
    validator = OrderValidator()
    validator.update_balances(
        base_balance=Decimal("1.5"),
        quote_balance=Decimal("10000"),
        current_price=Decimal("60000")
    )

    assert validator._base_balance == Decimal("1.5")
    assert validator._quote_balance == Decimal("10000")
    assert validator._current_price == Decimal("60000")


# ============================================================================
# Check 1: Kill Switch Tests
# ============================================================================

def test_kill_switch_active_blocks_order(validator_with_safety, kill_switch_mock):
    """Test active kill switch blocks all orders."""
    kill_switch_mock.is_active = True
    kill_switch_mock.reason = "Emergency stop"

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is False
    assert "Kill switch active" in result.reason
    assert "Emergency stop" in result.reason


def test_kill_switch_inactive_allows_order(validator_with_safety):
    """Test inactive kill switch allows orders."""
    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    # May fail other checks, but kill switch should pass
    assert "Kill switch" not in (result.reason or "")


def test_no_kill_switch_allows_order(validator):
    """Test validator without kill switch allows orders."""
    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator.validate(order)

    # Should not fail on kill switch check
    assert result.valid is True or "Kill switch" not in result.reason


# ============================================================================
# Check 2: Circuit Breaker Tests
# ============================================================================

def test_circuit_breaker_red_blocks_order(validator_with_safety, circuit_breaker_mock):
    """Test RED circuit breaker blocks orders."""
    circuit_breaker_mock.status = BreakerStatus(
        level=BreakerLevel.RED,
        can_trade=False,
        reason="Price dropped 10%",
        triggered_at=None,
        cooldown_until=None
    )

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is False
    assert "Circuit breaker" in result.reason
    assert "RED" in result.reason


def test_circuit_breaker_black_blocks_order(validator_with_safety, circuit_breaker_mock):
    """Test BLACK circuit breaker blocks orders."""
    circuit_breaker_mock.status = BreakerStatus(
        level=BreakerLevel.BLACK,
        can_trade=False,
        reason="3 order failures",
        triggered_at=None,
        cooldown_until=None
    )

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is False
    assert "Circuit breaker" in result.reason


def test_circuit_breaker_yellow_warns_but_allows(validator_with_safety, circuit_breaker_mock):
    """Test YELLOW circuit breaker warns but allows trading."""
    circuit_breaker_mock.status = BreakerStatus(
        level=BreakerLevel.YELLOW,
        can_trade=True,
        reason="Price dropped 5%",
        triggered_at=None,
        cooldown_until=None
    )
    circuit_breaker_mock.position_multiplier = 0.5

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is True
    assert len(result.warnings) > 0
    assert any("Circuit breaker" in w for w in result.warnings)


def test_circuit_breaker_green_allows_order(validator_with_safety):
    """Test GREEN circuit breaker allows orders."""
    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    # Should pass circuit breaker check
    assert "Circuit breaker at GREEN" not in (result.reason or "")


# ============================================================================
# Check 3: Loss Limiter Tests
# ============================================================================

def test_loss_limit_exceeded_blocks_order(validator_with_safety, loss_limiter_mock):
    """Test exceeded loss limit blocks orders."""
    loss_limiter_mock.get_status.return_value = LossLimitStatus(
        can_trade=False,
        position_multiplier=0.0,
        daily_loss_percent=12.0,
        hourly_loss_percent=0.0,
        daily_limit_hit=True,
        hourly_limit_hit=False,
        cooldown_until=None,
        reason="Daily loss limit exceeded (12.0%)"
    )

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is False
    assert "Loss limit" in result.reason or "loss" in result.reason.lower()


def test_loss_throttling_warns_but_allows(validator_with_safety, loss_limiter_mock):
    """Test loss throttling produces warning."""
    loss_limiter_mock.get_status.return_value = LossLimitStatus(
        can_trade=True,
        position_multiplier=0.7,
        daily_loss_percent=6.0,
        hourly_loss_percent=0.0,
        daily_limit_hit=False,
        hourly_limit_hit=False,
        cooldown_until=None,
        reason=None
    )

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator_with_safety.validate(order)

    assert result.valid is True
    assert any("Loss throttling" in w or "70%" in w for w in result.warnings)


# ============================================================================
# Check 4: Balance Sufficiency Tests
# ============================================================================

def test_buy_order_insufficient_quote_balance(validator):
    """Test buy order fails with insufficient quote currency."""
    # Has $5000, trying to buy $6000 worth
    order = OrderRequest(side="buy", size=Decimal("0.12"))
    result = validator.validate(order)

    assert result.valid is False
    assert "Insufficient quote balance" in result.reason


def test_buy_order_sufficient_balance(validator):
    """Test buy order passes with sufficient balance."""
    # Has $5000, buying $1000 worth
    order = OrderRequest(side="buy", size=Decimal("0.02"))
    result = validator.validate(order)

    # Should pass balance check (may fail others)
    assert result.valid is True or "Insufficient" not in result.reason


def test_sell_order_insufficient_base_balance(validator):
    """Test sell order fails with insufficient base currency."""
    # Has 0.1 BTC, trying to sell 0.15 BTC
    order = OrderRequest(side="sell", size=Decimal("0.15"))
    result = validator.validate(order)

    assert result.valid is False
    assert "Insufficient base balance" in result.reason


def test_sell_order_sufficient_balance(validator):
    """Test sell order passes with sufficient balance."""
    # Has 0.1 BTC, selling 0.05 BTC
    order = OrderRequest(side="sell", size=Decimal("0.05"))
    result = validator.validate(order)

    # Should pass balance check
    assert result.valid is True or "Insufficient" not in result.reason


def test_balance_check_uses_limit_price_when_provided(validator):
    """Test balance check uses limit price for limit orders."""
    # Limit order at $40,000 (lower than market $50,000)
    # 0.1 BTC at $40,000 = $4,000 (have $5,000)
    order = OrderRequest(
        side="buy",
        size=Decimal("0.1"),
        price=Decimal("40000"),
        order_type="limit"
    )
    result = validator.validate(order)

    # Should pass balance check at limit price
    assert result.valid is True or "Insufficient" not in result.reason


# ============================================================================
# Check 5: Position Limits Tests
# ============================================================================

def test_position_limit_not_exceeded(validator):
    """Test order within position limits."""
    # Portfolio: 0.1 BTC * $50k + $5k = $10k total
    # Current position: $5k / $10k = 50%
    # Buying 0.02 BTC ($1k) would make it 60% (under 80% limit)
    order = OrderRequest(side="buy", size=Decimal("0.02"))
    result = validator.validate(order)

    assert result.valid is True
    assert not any("position limit" in w.lower() for w in result.warnings)


def test_position_limit_exceeded(validator):
    """Test order exceeding position limits."""
    # Buying 0.08 BTC ($4k) would make position 90% (over 80%)
    order = OrderRequest(side="buy", size=Decimal("0.08"))
    result = validator.validate(order)

    assert result.valid is False
    assert "position limit" in result.reason.lower()


def test_position_limit_warning_near_limit(validator):
    """Test warning when approaching position limit."""
    # Buying amount that gets to 75% (90% of 80% limit)
    # Current: 50%, limit: 80%, 90% of limit = 72%
    # Need 22% more = $2200 = 0.044 BTC
    order = OrderRequest(side="buy", size=Decimal("0.044"))
    result = validator.validate(order)

    assert result.valid is True
    assert any("nearing limit" in w.lower() for w in result.warnings)


def test_sell_order_doesnt_check_position_limit(validator):
    """Test sell orders don't check position limits."""
    # Sell orders reduce position, no limit needed
    order = OrderRequest(side="sell", size=Decimal("0.05"))
    result = validator.validate(order)

    # Should not mention position limits
    assert "position" not in (result.reason or "").lower()


def test_position_limit_with_zero_portfolio(validator):
    """Test position limit check with zero portfolio value."""
    validator.update_balances(
        base_balance=Decimal("0"),
        quote_balance=Decimal("0"),
        current_price=Decimal("50000")
    )

    order = OrderRequest(side="buy", size=Decimal("0.01"))
    result = validator.validate(order)

    # Should handle gracefully, fail on insufficient balance instead
    assert "Insufficient" in result.reason


# ============================================================================
# Check 6: Price Sanity Tests
# ============================================================================

def test_price_sanity_market_order_skipped(validator):
    """Test price sanity check skipped for market orders."""
    order = OrderRequest(side="buy", size=Decimal("0.01"), order_type="market")
    result = validator.validate(order)

    # Should not fail on price sanity
    assert result.valid is True


def test_price_sanity_limit_within_range(validator):
    """Test limit price within 5% of market price."""
    # Market: $50,000, limit: $51,000 (2% above)
    order = OrderRequest(
        side="buy",
        size=Decimal("0.01"),
        price=Decimal("51000"),
        order_type="limit"
    )
    result = validator.validate(order)

    assert result.valid is True


def test_price_sanity_limit_too_high(validator):
    """Test limit price too far above market."""
    # Market: $50,000, limit: $55,000 (10% above, over 5% sanity check)
    order = OrderRequest(
        side="buy",
        size=Decimal("0.01"),
        price=Decimal("55000"),
        order_type="limit"
    )
    result = validator.validate(order)

    assert result.valid is False
    assert "deviates" in result.reason.lower()


def test_price_sanity_limit_too_low(validator):
    """Test limit price too far below market."""
    # Market: $50,000, limit: $45,000 (10% below)
    order = OrderRequest(
        side="sell",
        size=Decimal("0.01"),
        price=Decimal("45000"),
        order_type="limit"
    )
    result = validator.validate(order)

    assert result.valid is False
    assert "deviates" in result.reason.lower()


def test_price_sanity_at_exact_boundary(validator):
    """Test limit price at exactly 5% deviation."""
    # Market: $50,000, limit: $52,500 (exactly 5% above)
    order = OrderRequest(
        side="buy",
        size=Decimal("0.01"),
        price=Decimal("52500"),
        order_type="limit"
    )
    result = validator.validate(order)

    # At boundary, should fail (> comparison)
    assert result.valid is False


def test_price_sanity_with_zero_market_price(validator):
    """Test price sanity check with zero market price."""
    validator.update_balances(
        base_balance=Decimal("0.1"),
        quote_balance=Decimal("5000"),
        current_price=Decimal("0")
    )

    order = OrderRequest(
        side="buy",
        size=Decimal("0.01"),
        price=Decimal("50000"),
        order_type="limit"
    )
    result = validator.validate(order)

    # Should skip sanity check when market price is zero
    assert result.valid is True or "deviates" not in result.reason.lower()


# ============================================================================
# Check 7: Minimum Size Tests
# ============================================================================

def test_minimum_size_below_threshold(validator):
    """Test order below minimum size fails."""
    # 0.0001 BTC * $50,000 = $5 (below $10 minimum)
    order = OrderRequest(side="buy", size=Decimal("0.0001"))
    result = validator.validate(order)

    assert result.valid is False
    assert "below minimum" in result.reason.lower()


def test_minimum_size_at_threshold(validator):
    """Test order at exactly minimum size."""
    # 0.0002 BTC * $50,000 = $10 (exactly minimum)
    order = OrderRequest(side="buy", size=Decimal("0.0002"))
    result = validator.validate(order)

    assert result.valid is True or "below minimum" not in result.reason.lower()


def test_minimum_size_above_threshold(validator):
    """Test order above minimum size passes."""
    # 0.001 BTC * $50,000 = $50 (well above minimum)
    order = OrderRequest(side="buy", size=Decimal("0.001"))
    result = validator.validate(order)

    assert result.valid is True


# ============================================================================
# Position Multiplier Aggregation Tests
# ============================================================================

def test_get_position_multiplier_no_safety_systems(validator):
    """Test multiplier is 1.0 without safety systems."""
    multiplier = validator.get_position_multiplier()
    assert multiplier == 1.0


def test_get_position_multiplier_circuit_breaker_only(validator_with_safety, circuit_breaker_mock):
    """Test multiplier from circuit breaker only."""
    circuit_breaker_mock.position_multiplier = 0.5

    multiplier = validator_with_safety.get_position_multiplier()
    assert multiplier == 0.5


def test_get_position_multiplier_loss_limiter_only(validator_with_safety, loss_limiter_mock):
    """Test multiplier from loss limiter only."""
    loss_limiter_mock.get_status.return_value.position_multiplier = 0.7

    multiplier = validator_with_safety.get_position_multiplier()
    assert multiplier == 0.7


def test_get_position_multiplier_combined(validator_with_safety, circuit_breaker_mock, loss_limiter_mock):
    """Test multipliers are multiplied together."""
    circuit_breaker_mock.position_multiplier = 0.5
    loss_limiter_mock.get_status.return_value.position_multiplier = 0.8

    multiplier = validator_with_safety.get_position_multiplier()
    assert multiplier == pytest.approx(0.4, abs=0.01)  # 0.5 * 0.8


# ============================================================================
# Validation Priority & Short-Circuit Tests
# ============================================================================

def test_kill_switch_checked_before_other_checks(validator_with_safety, kill_switch_mock):
    """Test kill switch is checked first, short-circuits other checks."""
    kill_switch_mock.is_active = True
    kill_switch_mock.reason = "Emergency"

    # Order that would fail other checks too
    order = OrderRequest(side="buy", size=Decimal("100"))  # Huge order
    result = validator_with_safety.validate(order)

    # Should fail on kill switch, not reach balance check
    assert result.valid is False
    assert "Kill switch" in result.reason
    assert "Insufficient" not in result.reason


def test_circuit_breaker_checked_before_balance(validator_with_safety, circuit_breaker_mock):
    """Test circuit breaker checked before balance."""
    circuit_breaker_mock.status = BreakerStatus(
        level=BreakerLevel.RED,
        can_trade=False,
        reason="Test",
        triggered_at=None,
        cooldown_until=None
    )

    order = OrderRequest(side="buy", size=Decimal("100"))
    result = validator_with_safety.validate(order)

    assert "Circuit breaker" in result.reason
    assert "Insufficient" not in result.reason


def test_all_checks_pass_for_valid_order(validator_with_safety):
    """Test valid order passes all checks."""
    order = OrderRequest(side="buy", size=Decimal("0.02"))
    result = validator_with_safety.validate(order)

    assert result.valid is True
    assert result.reason is None


# ============================================================================
# Edge Cases & Error Handling Tests
# ============================================================================

def test_validation_with_decimal_precision(validator):
    """Test validation handles high decimal precision."""
    order = OrderRequest(
        side="buy",
        size=Decimal("0.123456789"),
        price=Decimal("50000.123456789"),
        order_type="limit"
    )

    # Should handle without errors
    result = validator.validate(order)
    assert isinstance(result, ValidationResult)


def test_warnings_accumulate_across_checks(validator_with_safety, circuit_breaker_mock, loss_limiter_mock):
    """Test warnings from multiple checks accumulate."""
    circuit_breaker_mock.status = BreakerStatus(
        level=BreakerLevel.YELLOW,
        can_trade=True,
        reason="Warning 1",
        triggered_at=None,
        cooldown_until=None
    )
    loss_limiter_mock.get_status.return_value = LossLimitStatus(
        can_trade=True,
        position_multiplier=0.8,
        daily_loss_percent=6.0,
        hourly_loss_percent=0.0,
        daily_limit_hit=False,
        hourly_limit_hit=False,
        cooldown_until=None,
        reason=None
    )

    order = OrderRequest(side="buy", size=Decimal("0.02"))
    result = validator_with_safety.validate(order)

    # Should have warnings from both systems
    assert len(result.warnings) >= 2


def test_validation_result_dataclass_defaults():
    """Test ValidationResult has proper defaults."""
    result = ValidationResult(valid=True)
    assert result.valid is True
    assert result.reason is None
    assert result.warnings == []
