"""
Comprehensive tests for the TradeCooldown safety system.

Tests cover:
- Configuration defaults and custom values
- Time-based cooldown (buy/sell)
- Price-based cooldown (buy must drop, sell must rise)
- Combined time + price logic
- Cache initialization from database
- record_trade() updates state
- update_settings() hot-reload
- Edge cases (no previous trade, timezone handling)
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock

from src.safety.trade_cooldown import (
    TradeCooldown,
    TradeCooldownConfig,
    CooldownStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default trade cooldown configuration."""
    return TradeCooldownConfig()


@pytest.fixture
def cooldown(config):
    """Trade cooldown instance with default config, no database."""
    return TradeCooldown(config=config)


@pytest.fixture
def cooldown_no_time():
    """Trade cooldown with time cooldown disabled."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=0,
        sell_cooldown_minutes=0,
        buy_price_change_percent=1.0,
        sell_price_change_percent=1.0,
    )
    return TradeCooldown(config=config)


@pytest.fixture
def cooldown_no_price():
    """Trade cooldown with price cooldown disabled."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=15,
        sell_cooldown_minutes=15,
        buy_price_change_percent=0.0,
        sell_price_change_percent=0.0,
    )
    return TradeCooldown(config=config)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_default_config():
    """Test default configuration values."""
    config = TradeCooldownConfig()
    assert config.buy_cooldown_minutes == 15
    assert config.sell_cooldown_minutes == 0  # Disabled for safety
    assert config.buy_price_change_percent == 1.0
    assert config.sell_price_change_percent == 0.0


def test_custom_config():
    """Test custom configuration values."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=30,
        sell_cooldown_minutes=10,
        buy_price_change_percent=2.0,
        sell_price_change_percent=1.5,
    )
    assert config.buy_cooldown_minutes == 30
    assert config.sell_cooldown_minutes == 10
    assert config.buy_price_change_percent == 2.0
    assert config.sell_price_change_percent == 1.5


def test_initialization_default():
    """Test cooldown initializes with default config."""
    cooldown = TradeCooldown()
    assert cooldown.config.buy_cooldown_minutes == 15
    assert cooldown.config.sell_cooldown_minutes == 0
    assert cooldown._last_buy_time is None
    assert cooldown._last_buy_price is None
    assert cooldown._cache_initialized is False


# ============================================================================
# No Previous Trade Tests
# ============================================================================

def test_no_previous_buy_allows_trade(cooldown):
    """First buy should always be allowed."""
    can_execute, reason = cooldown.can_execute("buy", Decimal("100000"))
    assert can_execute is True
    assert reason is None


def test_no_previous_sell_allows_trade(cooldown):
    """First sell should always be allowed."""
    can_execute, reason = cooldown.can_execute("sell", Decimal("100000"))
    assert can_execute is True
    assert reason is None


# ============================================================================
# Time-Based Cooldown Tests
# ============================================================================

def test_buy_allowed_when_only_time_enabled(cooldown_no_price):
    """Buy allowed when only time cooldown enabled (price disabled = always passes with OR logic)."""
    # Record a buy
    cooldown_no_price.record_trade("buy", Decimal("100000"))

    # Immediately try another buy - allowed because price condition is disabled (passes)
    can_execute, reason = cooldown_no_price.can_execute("buy", Decimal("100000"))
    assert can_execute is True  # OR logic: disabled price condition = passes
    assert reason is None


def test_sell_allowed_when_only_time_enabled():
    """Sell allowed when only time cooldown enabled (price disabled = always passes with OR logic)."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=0,
        sell_cooldown_minutes=15,
        buy_price_change_percent=0.0,
        sell_price_change_percent=0.0,
    )
    cooldown = TradeCooldown(config=config)

    # Record a sell
    cooldown.record_trade("sell", Decimal("100000"))

    # Immediately try another sell - allowed because price condition is disabled (passes)
    can_execute, reason = cooldown.can_execute("sell", Decimal("100000"))
    assert can_execute is True  # OR logic: disabled price condition = passes
    assert reason is None


def test_buy_allowed_after_time_cooldown_expires(cooldown_no_price):
    """Buy should be allowed after time cooldown expires."""
    # Manually set last buy time to 20 minutes ago
    cooldown_no_price._last_buy_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    cooldown_no_price._last_buy_price = Decimal("100000")

    # Should be allowed (15 min cooldown < 20 min elapsed)
    can_execute, reason = cooldown_no_price.can_execute("buy", Decimal("100000"))
    assert can_execute is True
    assert reason is None


# ============================================================================
# Price-Based Cooldown Tests
# ============================================================================

def test_buy_allowed_when_only_price_enabled(cooldown_no_time):
    """Buy allowed when only price cooldown enabled (time disabled = always passes with OR logic)."""
    # Record a buy at 100000
    cooldown_no_time.record_trade("buy", Decimal("100000"))

    # Try to buy at 99500 (only 0.5% drop, need 1%) - allowed because time disabled (passes)
    can_execute, reason = cooldown_no_time.can_execute("buy", Decimal("99500"))
    assert can_execute is True  # OR logic: disabled time condition = passes
    assert reason is None


def test_buy_allowed_price_dropped_enough(cooldown_no_time):
    """Buy allowed if price dropped enough."""
    # Record a buy at 100000
    cooldown_no_time.record_trade("buy", Decimal("100000"))

    # Try to buy at 98500 (1.5% drop, need 1%)
    can_execute, reason = cooldown_no_time.can_execute("buy", Decimal("98500"))
    assert can_execute is True
    assert reason is None


def test_sell_allowed_when_only_price_enabled(cooldown_no_time):
    """Sell allowed when only price cooldown enabled (time disabled = always passes with OR logic)."""
    # Record a sell at 100000
    cooldown_no_time.record_trade("sell", Decimal("100000"))

    # Try to sell at 100500 (only 0.5% rise, need 1%) - allowed because time disabled (passes)
    can_execute, reason = cooldown_no_time.can_execute("sell", Decimal("100500"))
    assert can_execute is True  # OR logic: disabled time condition = passes
    assert reason is None


def test_sell_allowed_price_risen_enough(cooldown_no_time):
    """Sell allowed if price risen enough."""
    # Record a sell at 100000
    cooldown_no_time.record_trade("sell", Decimal("100000"))

    # Try to sell at 101500 (1.5% rise, need 1%)
    can_execute, reason = cooldown_no_time.can_execute("sell", Decimal("101500"))
    assert can_execute is True
    assert reason is None


# ============================================================================
# Combined Time + Price Tests
# ============================================================================

def test_buy_blocked_time_and_price_both_fail():
    """Buy blocked when both time and price conditions fail."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=15,
        sell_cooldown_minutes=0,
        buy_price_change_percent=1.0,
        sell_price_change_percent=0.0,
    )
    cooldown = TradeCooldown(config=config)

    # Record a buy
    cooldown.record_trade("buy", Decimal("100000"))

    # Try immediately at same price
    can_execute, reason = cooldown.can_execute("buy", Decimal("100000"))
    assert can_execute is False
    assert "wait" in reason
    assert "price drop" in reason


def test_buy_allowed_when_time_expires():
    """Buy allowed when time cooldown expires (OR logic - time condition passes)."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=15,
        sell_cooldown_minutes=0,
        buy_price_change_percent=1.0,
        sell_price_change_percent=0.0,
    )
    cooldown = TradeCooldown(config=config)

    # Set last buy to 20 minutes ago (time OK)
    cooldown._last_buy_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    cooldown._last_buy_price = Decimal("100000")

    # Try at same price - allowed because time passed (OR logic)
    can_execute, reason = cooldown.can_execute("buy", Decimal("100000"))
    assert can_execute is True
    assert reason is None


def test_buy_allowed_when_price_moves():
    """Buy allowed when price drops enough (OR logic - price condition passes)."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=15,
        sell_cooldown_minutes=0,
        buy_price_change_percent=1.0,
        sell_price_change_percent=0.0,
    )
    cooldown = TradeCooldown(config=config)

    # Record a buy just now (time NOT OK)
    cooldown.record_trade("buy", Decimal("100000"))

    # Try at 2% lower price - allowed because price dropped (OR logic)
    can_execute, reason = cooldown.can_execute("buy", Decimal("98000"))
    assert can_execute is True
    assert reason is None


def test_buy_allowed_both_conditions_pass():
    """Buy allowed when both time and price conditions pass."""
    config = TradeCooldownConfig(
        buy_cooldown_minutes=15,
        sell_cooldown_minutes=0,
        buy_price_change_percent=1.0,
        sell_price_change_percent=0.0,
    )
    cooldown = TradeCooldown(config=config)

    # Set last buy to 20 minutes ago
    cooldown._last_buy_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    cooldown._last_buy_price = Decimal("100000")

    # Try at lower price - both conditions pass
    can_execute, reason = cooldown.can_execute("buy", Decimal("98000"))
    assert can_execute is True
    assert reason is None


# ============================================================================
# record_trade() Tests
# ============================================================================

def test_record_buy_updates_state(cooldown):
    """record_trade updates buy state."""
    cooldown.record_trade("buy", Decimal("50000"))

    assert cooldown._last_buy_price == Decimal("50000")
    assert cooldown._last_buy_time is not None
    assert cooldown._last_sell_price is None


def test_record_sell_updates_state(cooldown):
    """record_trade updates sell state."""
    cooldown.record_trade("sell", Decimal("60000"))

    assert cooldown._last_sell_price == Decimal("60000")
    assert cooldown._last_sell_time is not None
    assert cooldown._last_buy_price is None


def test_record_trade_overwrites_previous(cooldown):
    """Recording a new trade overwrites the previous one."""
    cooldown.record_trade("buy", Decimal("50000"))
    first_time = cooldown._last_buy_time

    cooldown.record_trade("buy", Decimal("55000"))

    assert cooldown._last_buy_price == Decimal("55000")
    assert cooldown._last_buy_time >= first_time


# ============================================================================
# get_status() Tests
# ============================================================================

def test_get_status_no_trades(cooldown):
    """get_status returns correct values with no trades."""
    status = cooldown.get_status()

    assert status.can_buy is True
    assert status.can_sell is True
    assert status.last_buy_price is None
    assert status.last_sell_price is None
    assert status.buy_cooldown_remaining_seconds == 0
    assert status.sell_cooldown_remaining_seconds == 0


def test_get_status_with_recent_buy(cooldown_no_price):
    """get_status shows cooldown remaining after buy."""
    cooldown_no_price.record_trade("buy", Decimal("100000"))

    status = cooldown_no_price.get_status()

    assert status.can_buy is False  # Within cooldown
    assert status.last_buy_price == Decimal("100000")
    assert status.buy_cooldown_remaining_seconds > 0


# ============================================================================
# update_settings() Tests
# ============================================================================

def test_update_settings_partial():
    """update_settings only updates provided values."""
    cooldown = TradeCooldown()
    original_sell = cooldown.config.sell_cooldown_minutes

    cooldown.update_settings(buy_cooldown_minutes=30)

    assert cooldown.config.buy_cooldown_minutes == 30
    assert cooldown.config.sell_cooldown_minutes == original_sell


def test_update_settings_all():
    """update_settings can update all values."""
    cooldown = TradeCooldown()

    cooldown.update_settings(
        buy_cooldown_minutes=5,
        sell_cooldown_minutes=10,
        buy_price_change_percent=0.5,
        sell_price_change_percent=2.0,
    )

    assert cooldown.config.buy_cooldown_minutes == 5
    assert cooldown.config.sell_cooldown_minutes == 10
    assert cooldown.config.buy_price_change_percent == 0.5
    assert cooldown.config.sell_price_change_percent == 2.0


# ============================================================================
# invalidate_cache() Tests
# ============================================================================

def test_invalidate_cache_clears_state(cooldown):
    """invalidate_cache clears all cached trade data."""
    cooldown.record_trade("buy", Decimal("50000"))
    cooldown.record_trade("sell", Decimal("55000"))
    cooldown._cache_initialized = True

    cooldown.invalidate_cache()

    assert cooldown._cache_initialized is False
    assert cooldown._last_buy_time is None
    assert cooldown._last_buy_price is None
    assert cooldown._last_sell_time is None
    assert cooldown._last_sell_price is None


# ============================================================================
# Database Cache Tests
# ============================================================================

def test_cache_loads_from_database():
    """Cache loads last trades from database on first check."""
    mock_db = MagicMock()

    # Mock trade objects
    mock_buy = MagicMock()
    mock_buy.executed_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    mock_buy.price = "95000"

    mock_sell = MagicMock()
    mock_sell.executed_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    mock_sell.price = "96000"

    mock_db.get_last_trade_by_side.side_effect = lambda side, **kwargs: (
        mock_buy if side == "buy" else mock_sell
    )

    cooldown = TradeCooldown(db=mock_db)

    # Trigger cache load
    cooldown.can_execute("buy", Decimal("100000"))

    assert cooldown._cache_initialized is True
    assert cooldown._last_buy_price == Decimal("95000")
    assert cooldown._last_sell_price == Decimal("96000")
    mock_db.get_last_trade_by_side.assert_called()


def test_cache_handles_no_trades_in_db():
    """Cache handles case where no trades exist in database."""
    mock_db = MagicMock()
    mock_db.get_last_trade_by_side.return_value = None

    cooldown = TradeCooldown(db=mock_db)

    # Should not raise, should allow trade
    can_execute, reason = cooldown.can_execute("buy", Decimal("100000"))

    assert can_execute is True
    assert cooldown._cache_initialized is True


# ============================================================================
# Edge Cases
# ============================================================================

def test_unknown_side_allowed(cooldown):
    """Unknown trade side should be allowed."""
    can_execute, reason = cooldown.can_execute("unknown", Decimal("100000"))
    assert can_execute is True
    assert reason is None


def test_naive_datetime_handled():
    """Naive datetime from old trades should be handled."""
    cooldown = TradeCooldown()

    # Simulate naive datetime from database
    cooldown._last_buy_time = datetime.now() - timedelta(minutes=5)  # Naive
    cooldown._last_buy_price = Decimal("100000")

    # Should not raise
    can_execute, reason = cooldown.can_execute("buy", Decimal("99000"))

    # Should work (time check should handle naive datetime)
    assert isinstance(can_execute, bool)


def test_sell_cooldown_disabled_by_default(cooldown):
    """Sell cooldown should be disabled by default for safety."""
    cooldown.record_trade("sell", Decimal("100000"))

    # Should be allowed immediately (sell cooldown = 0)
    can_execute, reason = cooldown.can_execute("sell", Decimal("100000"))
    assert can_execute is True


def test_invalid_price_zero_blocked(cooldown):
    """Zero price should be blocked."""
    can_execute, reason = cooldown.can_execute("buy", Decimal("0"))
    assert can_execute is False
    assert reason == "invalid price"


def test_invalid_price_negative_blocked(cooldown):
    """Negative price should be blocked."""
    can_execute, reason = cooldown.can_execute("buy", Decimal("-100"))
    assert can_execute is False
    assert reason == "invalid price"


def test_invalid_price_none_blocked(cooldown):
    """None price should be blocked."""
    can_execute, reason = cooldown.can_execute("buy", None)
    assert can_execute is False
    assert reason == "invalid price"
