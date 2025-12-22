"""
Comprehensive unit tests for the PositionService class.

Tests cover:
- update_after_buy() with no existing position (new position)
- update_after_buy() with existing position (weighted average calculation)
- calculate_realized_pnl() with existing position
- calculate_realized_pnl() edge cases (no position, negative quantity guard)
- get_current_position() delegation to database
- get_portfolio_value() calculation
- update_config() method

Note: These are unit tests with mocked dependencies. Integration tests exist
in test_trading_daemon.py for end-to-end verification.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.daemon.position_service import PositionService, PositionConfig
from src.state.database import BotMode, Position


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default position service configuration."""
    return PositionConfig(
        trading_pair="BTC-USD",
        is_paper_trading=True,
        base_currency="BTC",
        quote_currency="USD",
    )


@pytest.fixture
def mock_db():
    """Mock database with configurable position returns."""
    db = MagicMock()
    db.get_current_position.return_value = None
    db.update_position.return_value = None
    return db


@pytest.fixture
def mock_exchange():
    """Mock exchange client with balance and price methods."""
    exchange = MagicMock()

    # Setup balance returns
    btc_balance = MagicMock()
    btc_balance.available = Decimal("1.5")

    usd_balance = MagicMock()
    usd_balance.available = Decimal("10000")

    def get_balance(currency):
        if currency == "BTC":
            return btc_balance
        elif currency == "USD":
            return usd_balance
        raise ValueError(f"Unknown currency: {currency}")

    exchange.get_balance = MagicMock(side_effect=get_balance)
    exchange.get_current_price.return_value = Decimal("50000")

    return exchange


@pytest.fixture
def position_service(config, mock_db, mock_exchange):
    """PositionService with mocked dependencies."""
    return PositionService(
        config=config,
        db=mock_db,
        exchange_client=mock_exchange,
    )


def create_mock_position(quantity: str, average_cost: str) -> MagicMock:
    """Helper to create a mock Position object."""
    position = MagicMock(spec=Position)
    position.get_quantity.return_value = Decimal(quantity)
    position.get_average_cost.return_value = Decimal(average_cost)
    return position


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization(config, mock_db, mock_exchange):
    """Test PositionService initializes correctly with dependencies."""
    service = PositionService(
        config=config,
        db=mock_db,
        exchange_client=mock_exchange,
    )

    assert service.config == config
    assert service.db == mock_db
    assert service.client == mock_exchange


def test_update_config(position_service):
    """Test update_config updates the configuration."""
    new_config = PositionConfig(
        trading_pair="ETH-USD",
        is_paper_trading=False,
        base_currency="ETH",
        quote_currency="USD",
    )

    position_service.update_config(new_config)

    assert position_service.config.trading_pair == "ETH-USD"
    assert position_service.config.is_paper_trading is False
    assert position_service.config.base_currency == "ETH"


# ============================================================================
# update_after_buy() Tests - No Existing Position
# ============================================================================

def test_update_after_buy_no_existing_position(position_service, mock_db):
    """Test buying with no existing position creates new position."""
    mock_db.get_current_position.return_value = None

    new_avg_cost = position_service.update_after_buy(
        size=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("10"),
        is_paper=True,
    )

    # Cost basis: (0.1 * 50000 + 10) / 0.1 = 50100
    assert new_avg_cost == Decimal("50100")

    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0.1"),
        average_cost=Decimal("50100"),
        is_paper=True,
        bot_mode=BotMode.NORMAL,
    )


def test_update_after_buy_no_existing_position_zero_fee(position_service, mock_db):
    """Test buying with no fee uses exact price as average cost."""
    mock_db.get_current_position.return_value = None

    new_avg_cost = position_service.update_after_buy(
        size=Decimal("0.5"),
        price=Decimal("48000"),
        fee=Decimal("0"),
        is_paper=True,
    )

    # Cost basis: (0.5 * 48000 + 0) / 0.5 = 48000
    assert new_avg_cost == Decimal("48000")


def test_update_after_buy_no_existing_position_inverted_mode(position_service, mock_db):
    """Test buying in inverted (Cramer) mode."""
    mock_db.get_current_position.return_value = None

    new_avg_cost = position_service.update_after_buy(
        size=Decimal("0.2"),
        price=Decimal("45000"),
        fee=Decimal("5"),
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )

    # Cost basis: (0.2 * 45000 + 5) / 0.2 = 45025
    assert new_avg_cost == Decimal("45025")

    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0.2"),
        average_cost=Decimal("45025"),
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )


# ============================================================================
# update_after_buy() Tests - With Existing Position (Weighted Average)
# ============================================================================

def test_update_after_buy_weighted_average_calculation(position_service, mock_db):
    """Test weighted average cost calculation with existing position."""
    # Existing position: 0.5 BTC @ $40,000
    existing = create_mock_position("0.5", "40000")
    mock_db.get_current_position.return_value = existing

    # Buy 0.5 BTC @ $50,000 with $20 fee
    new_avg_cost = position_service.update_after_buy(
        size=Decimal("0.5"),
        price=Decimal("50000"),
        fee=Decimal("20"),
        is_paper=True,
    )

    # Old value: 0.5 * 40000 = 20000
    # New value: 20000 + (0.5 * 50000) + 20 = 45020
    # New qty: 0.5 + 0.5 = 1.0
    # New avg: 45020 / 1.0 = 45020
    assert new_avg_cost == Decimal("45020")

    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("1.0"),
        average_cost=Decimal("45020"),
        is_paper=True,
        bot_mode=BotMode.NORMAL,
    )


def test_update_after_buy_dca_multiple_buys(position_service, mock_db):
    """Test DCA scenario with multiple incremental buys."""
    # First buy: 0.1 BTC @ $50,000
    mock_db.get_current_position.return_value = None

    avg1 = position_service.update_after_buy(
        size=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("0"),
        is_paper=True,
    )
    assert avg1 == Decimal("50000")

    # Second buy: 0.1 BTC @ $45,000 (price dropped)
    existing = create_mock_position("0.1", "50000")
    mock_db.get_current_position.return_value = existing

    avg2 = position_service.update_after_buy(
        size=Decimal("0.1"),
        price=Decimal("45000"),
        fee=Decimal("0"),
        is_paper=True,
    )

    # Old value: 0.1 * 50000 = 5000
    # New value: 5000 + (0.1 * 45000) + 0 = 9500
    # New qty: 0.2
    # New avg: 9500 / 0.2 = 47500
    assert avg2 == Decimal("47500")


def test_update_after_buy_small_add_to_large_position(position_service, mock_db):
    """Test adding small amount to large existing position."""
    # Existing: 10 BTC @ $45,000
    existing = create_mock_position("10", "45000")
    mock_db.get_current_position.return_value = existing

    # Buy 0.1 BTC @ $50,000 with $5 fee
    new_avg_cost = position_service.update_after_buy(
        size=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("5"),
        is_paper=True,
    )

    # Old value: 10 * 45000 = 450000
    # New value: 450000 + (0.1 * 50000) + 5 = 455005
    # New qty: 10.1
    # New avg: 455005 / 10.1 = 45050.0
    expected_avg = Decimal("455005") / Decimal("10.1")
    assert new_avg_cost == expected_avg


# ============================================================================
# calculate_realized_pnl() Tests - Normal Cases
# ============================================================================

def test_calculate_realized_pnl_profit(position_service, mock_db):
    """Test calculating profit on sell."""
    # Position: 1 BTC @ $40,000 avg cost
    existing = create_mock_position("1", "40000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("0.5"),
        sell_price=Decimal("50000"),
        fee=Decimal("10"),
        is_paper=True,
    )

    # PnL: (50000 - 40000) * 0.5 - 10 = 5000 - 10 = 4990
    assert pnl == Decimal("4990")

    # Should update position with reduced quantity
    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0.5"),  # 1 - 0.5
        average_cost=Decimal("40000"),  # Unchanged
        is_paper=True,
        bot_mode=BotMode.NORMAL,
    )


def test_calculate_realized_pnl_loss(position_service, mock_db):
    """Test calculating loss on sell."""
    # Position: 1 BTC @ $50,000 avg cost
    existing = create_mock_position("1", "50000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("0.5"),
        sell_price=Decimal("45000"),
        fee=Decimal("10"),
        is_paper=True,
    )

    # PnL: (45000 - 50000) * 0.5 - 10 = -2500 - 10 = -2510
    assert pnl == Decimal("-2510")


def test_calculate_realized_pnl_full_position_close(position_service, mock_db):
    """Test selling entire position."""
    # Position: 2 BTC @ $48,000 avg cost
    existing = create_mock_position("2", "48000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("2"),
        sell_price=Decimal("52000"),
        fee=Decimal("50"),
        is_paper=True,
    )

    # PnL: (52000 - 48000) * 2 - 50 = 8000 - 50 = 7950
    assert pnl == Decimal("7950")

    # Should set quantity to 0
    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0"),
        average_cost=Decimal("48000"),
        is_paper=True,
        bot_mode=BotMode.NORMAL,
    )


def test_calculate_realized_pnl_inverted_mode(position_service, mock_db):
    """Test realized PnL in inverted (Cramer) mode."""
    existing = create_mock_position("1", "45000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("1"),
        sell_price=Decimal("47000"),
        fee=Decimal("20"),
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )

    # PnL: (47000 - 45000) * 1 - 20 = 2000 - 20 = 1980
    assert pnl == Decimal("1980")

    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0"),
        average_cost=Decimal("45000"),
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )


# ============================================================================
# calculate_realized_pnl() Tests - Edge Cases
# ============================================================================

def test_calculate_realized_pnl_no_position(position_service, mock_db):
    """Test selling with no existing position returns negative fee."""
    mock_db.get_current_position.return_value = None

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("0.5"),
        sell_price=Decimal("50000"),
        fee=Decimal("10"),
        is_paper=True,
    )

    # No position = just deduct the fee
    assert pnl == Decimal("-10")

    # Should NOT call update_position when no position exists
    mock_db.update_position.assert_not_called()


def test_calculate_realized_pnl_zero_quantity_position(position_service, mock_db):
    """Test selling with zero quantity position returns negative fee."""
    existing = create_mock_position("0", "50000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("0.5"),
        sell_price=Decimal("55000"),
        fee=Decimal("15"),
        is_paper=True,
    )

    # Zero quantity = just deduct the fee
    assert pnl == Decimal("-15")
    mock_db.update_position.assert_not_called()


def test_calculate_realized_pnl_negative_quantity_guard(position_service, mock_db):
    """Test selling more than position size clamps to zero."""
    # Position: 0.5 BTC @ $50,000
    existing = create_mock_position("0.5", "50000")
    mock_db.get_current_position.return_value = existing

    # Try to sell 1 BTC (more than we have)
    pnl = position_service.calculate_realized_pnl(
        size=Decimal("1"),
        sell_price=Decimal("55000"),
        fee=Decimal("20"),
        is_paper=True,
    )

    # PnL still calculated on full size
    # (55000 - 50000) * 1 - 20 = 5000 - 20 = 4980
    assert pnl == Decimal("4980")

    # Position quantity should be clamped to 0, not negative
    mock_db.update_position.assert_called_once()
    call_args = mock_db.update_position.call_args
    assert call_args.kwargs["quantity"] == Decimal("0")


def test_calculate_realized_pnl_breakeven(position_service, mock_db):
    """Test selling at exactly break-even price."""
    existing = create_mock_position("1", "50000")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("1"),
        sell_price=Decimal("50000"),  # Same as avg cost
        fee=Decimal("0"),
        is_paper=True,
    )

    # Breakeven: (50000 - 50000) * 1 - 0 = 0
    assert pnl == Decimal("0")


def test_calculate_realized_pnl_high_precision_decimals(position_service, mock_db):
    """Test PnL calculation with high precision decimal values."""
    existing = create_mock_position("0.12345678", "48123.45678901")
    mock_db.get_current_position.return_value = existing

    pnl = position_service.calculate_realized_pnl(
        size=Decimal("0.12345678"),
        sell_price=Decimal("49000.12345678"),
        fee=Decimal("1.23456789"),
        is_paper=True,
    )

    # Verify it returns a Decimal (precision preserved)
    assert isinstance(pnl, Decimal)
    # Approximate check: (49000.12 - 48123.46) * 0.123 - 1.23 ≈ 107 - 1.23 ≈ 106
    assert pnl > Decimal("100")  # Rough sanity check


# ============================================================================
# get_current_position() Tests
# ============================================================================

def test_get_current_position_delegates_to_db(position_service, mock_db):
    """Test get_current_position delegates to database."""
    expected_position = create_mock_position("1", "50000")
    mock_db.get_current_position.return_value = expected_position

    result = position_service.get_current_position(is_paper=True)

    assert result == expected_position
    mock_db.get_current_position.assert_called_once_with(
        "BTC-USD",
        is_paper=True,
        bot_mode=BotMode.NORMAL,
    )


def test_get_current_position_uses_config_default(position_service, mock_db):
    """Test get_current_position uses config is_paper_trading when None."""
    position_service.config.is_paper_trading = True

    position_service.get_current_position(is_paper=None)

    mock_db.get_current_position.assert_called_once_with(
        "BTC-USD",
        is_paper=True,  # From config
        bot_mode=BotMode.NORMAL,
    )


def test_get_current_position_inverted_mode(position_service, mock_db):
    """Test get_current_position with inverted bot mode."""
    position_service.get_current_position(
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )

    mock_db.get_current_position.assert_called_once_with(
        "BTC-USD",
        is_paper=True,
        bot_mode=BotMode.INVERTED,
    )


def test_get_current_position_returns_none(position_service, mock_db):
    """Test get_current_position returns None when no position exists."""
    mock_db.get_current_position.return_value = None

    result = position_service.get_current_position(is_paper=True)

    assert result is None


# ============================================================================
# get_portfolio_value() Tests
# ============================================================================

def test_get_portfolio_value_with_provided_price(position_service, mock_exchange):
    """Test portfolio value calculation with provided price."""
    # BTC balance: 1.5, USD balance: 10000
    # Price provided: 50000
    # Portfolio: 10000 + 1.5 * 50000 = 10000 + 75000 = 85000

    value = position_service.get_portfolio_value(current_price=Decimal("50000"))

    assert value == Decimal("85000")
    # Should NOT call get_current_price when price is provided
    mock_exchange.get_current_price.assert_not_called()


def test_get_portfolio_value_fetches_price(position_service, mock_exchange):
    """Test portfolio value fetches price from exchange when not provided."""
    mock_exchange.get_current_price.return_value = Decimal("60000")

    # BTC balance: 1.5, USD balance: 10000
    # Portfolio: 10000 + 1.5 * 60000 = 10000 + 90000 = 100000

    value = position_service.get_portfolio_value(current_price=None)

    assert value == Decimal("100000")
    mock_exchange.get_current_price.assert_called_once_with("BTC-USD")


def test_get_portfolio_value_queries_correct_currencies(position_service, mock_exchange):
    """Test portfolio value queries correct base and quote currencies."""
    position_service.get_portfolio_value(current_price=Decimal("50000"))

    # Verify balance calls
    calls = mock_exchange.get_balance.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "BTC"  # Base currency
    assert calls[1][0][0] == "USD"  # Quote currency


def test_get_portfolio_value_zero_btc_balance(position_service, mock_exchange):
    """Test portfolio value with zero BTC balance."""
    btc_balance = MagicMock()
    btc_balance.available = Decimal("0")
    usd_balance = MagicMock()
    usd_balance.available = Decimal("5000")

    def get_balance(currency):
        if currency == "BTC":
            return btc_balance
        return usd_balance

    mock_exchange.get_balance = MagicMock(side_effect=get_balance)

    value = position_service.get_portfolio_value(current_price=Decimal("50000"))

    # Just the USD balance: 5000
    assert value == Decimal("5000")


def test_get_portfolio_value_zero_usd_balance(position_service, mock_exchange):
    """Test portfolio value with zero USD balance."""
    btc_balance = MagicMock()
    btc_balance.available = Decimal("2")
    usd_balance = MagicMock()
    usd_balance.available = Decimal("0")

    def get_balance(currency):
        if currency == "BTC":
            return btc_balance
        return usd_balance

    mock_exchange.get_balance = MagicMock(side_effect=get_balance)

    value = position_service.get_portfolio_value(current_price=Decimal("50000"))

    # Just BTC value: 2 * 50000 = 100000
    assert value == Decimal("100000")


# ============================================================================
# Paper vs Live Trading Mode Tests
# ============================================================================

def test_update_after_buy_live_trading(mock_db, mock_exchange):
    """Test buy updates work in live (non-paper) mode."""
    config = PositionConfig(
        trading_pair="BTC-USD",
        is_paper_trading=False,  # Live mode
        base_currency="BTC",
        quote_currency="USD",
    )
    service = PositionService(config, mock_db, mock_exchange)
    mock_db.get_current_position.return_value = None

    service.update_after_buy(
        size=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("10"),
        is_paper=False,  # Live mode
    )

    mock_db.update_position.assert_called_once_with(
        symbol="BTC-USD",
        quantity=Decimal("0.1"),
        average_cost=Decimal("50100"),
        is_paper=False,  # Verify live mode passed through
        bot_mode=BotMode.NORMAL,
    )


def test_calculate_pnl_live_trading(mock_db, mock_exchange):
    """Test PnL calculation works in live (non-paper) mode."""
    config = PositionConfig(
        trading_pair="BTC-USD",
        is_paper_trading=False,
        base_currency="BTC",
        quote_currency="USD",
    )
    service = PositionService(config, mock_db, mock_exchange)

    existing = create_mock_position("1", "50000")
    mock_db.get_current_position.return_value = existing

    service.calculate_realized_pnl(
        size=Decimal("0.5"),
        sell_price=Decimal("55000"),
        fee=Decimal("10"),
        is_paper=False,
    )

    mock_db.get_current_position.assert_called_once_with(
        "BTC-USD",
        is_paper=False,
        bot_mode=BotMode.NORMAL,
    )
