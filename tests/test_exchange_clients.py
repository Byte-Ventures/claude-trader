"""
Comprehensive tests for exchange clients and supporting modules.

Tests cover:
- PaperTradingClient (simulated trading)
- ExchangeFactory (client creation)
- SymbolMapper (symbol translation)
- Exchange protocol compliance
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from src.api.exchange_protocol import Balance, OrderResult, MarketData, ExchangeClient
from src.api.paper_client import PaperTradingClient, PaperTrade
from src.api.symbol_mapper import (
    Exchange,
    normalize_symbol,
    to_exchange_symbol,
    to_kraken_asset,
    from_kraken_asset,
    parse_trading_pair,
    to_kraken_granularity,
    from_kraken_granularity,
)


# ============================================================================
# Exchange Protocol Tests
# ============================================================================

def test_balance_total_property():
    """Test Balance.total property calculates correctly."""
    balance = Balance(
        currency="USD",
        available=Decimal("1000"),
        hold=Decimal("500")
    )

    assert balance.total == Decimal("1500")


def test_balance_zero_hold():
    """Test Balance with zero hold."""
    balance = Balance(
        currency="BTC",
        available=Decimal("1.5"),
        hold=Decimal("0")
    )

    assert balance.total == Decimal("1.5")


def test_order_result_success():
    """Test OrderResult for successful order."""
    result = OrderResult(
        order_id="order-123",
        side="buy",
        size=Decimal("0.1"),
        filled_price=Decimal("50000"),
        status="FILLED",
        fee=Decimal("25"),
        success=True
    )

    assert result.success is True
    assert result.error is None


def test_order_result_failure():
    """Test OrderResult for failed order."""
    result = OrderResult(
        order_id="",
        side="buy",
        size=Decimal("0"),
        filled_price=None,
        status="failed",
        fee=Decimal("0"),
        success=False,
        error="Insufficient balance"
    )

    assert result.success is False
    assert result.error == "Insufficient balance"


# ============================================================================
# PaperTradingClient Tests
# ============================================================================

@pytest.fixture
def mock_exchange_client():
    """Create a mock exchange client for testing."""
    mock = Mock(spec=ExchangeClient)
    mock.get_current_price.return_value = Decimal("50000")
    mock.get_market_data.return_value = MarketData(
        symbol="BTC-USD",
        price=Decimal("50000"),
        bid=Decimal("49990"),
        ask=Decimal("50010"),
        volume_24h=Decimal("1000"),
        timestamp=datetime.now()
    )
    mock.get_candles.return_value = pd.DataFrame({
        'timestamp': [datetime.now()],
        'open': [50000],
        'high': [50100],
        'low': [49900],
        'close': [50000],
        'volume': [100]
    })
    return mock


@pytest.fixture
def paper_client(mock_exchange_client):
    """Create PaperTradingClient with mock real client."""
    return PaperTradingClient(
        real_client=mock_exchange_client,
        initial_quote=10000.0,
        initial_base=0.0,
        trading_pair="BTC-USD"
    )


def test_paper_client_initialization(mock_exchange_client):
    """Test paper client initialization."""
    client = PaperTradingClient(
        real_client=mock_exchange_client,
        initial_quote=15000.0,
        initial_base=0.5,
        trading_pair="BTC-USD"
    )

    assert client._quote_balance == Decimal("15000")
    assert client._base_balance == Decimal("0.5")
    assert client._base_currency == "BTC"
    assert client._quote_currency == "USD"


def test_paper_client_parse_trading_pair(mock_exchange_client):
    """Test trading pair parsing."""
    client = PaperTradingClient(
        real_client=mock_exchange_client,
        trading_pair="ETH-EUR"
    )

    assert client._base_currency == "ETH"
    assert client._quote_currency == "EUR"


def test_get_balance_base(paper_client):
    """Test getting base currency balance."""
    balance = paper_client.get_balance("BTC")

    assert balance.currency == "BTC"
    assert balance.available == Decimal("0")
    assert balance.hold == Decimal("0")


def test_get_balance_quote(paper_client):
    """Test getting quote currency balance."""
    balance = paper_client.get_balance("USD")

    assert balance.currency == "USD"
    assert balance.available == Decimal("10000")
    assert balance.hold == Decimal("0")


def test_get_balance_unknown_currency(paper_client):
    """Test getting balance for unknown currency."""
    balance = paper_client.get_balance("EUR")

    assert balance.currency == "EUR"
    assert balance.available == Decimal("0")
    assert balance.hold == Decimal("0")


def test_get_current_price_delegates_to_real_client(paper_client, mock_exchange_client):
    """Test current price is fetched from real client."""
    price = paper_client.get_current_price("BTC-USD")

    assert price == Decimal("50000")
    mock_exchange_client.get_current_price.assert_called_once_with("BTC-USD")


def test_get_market_data_delegates_to_real_client(paper_client, mock_exchange_client):
    """Test market data is fetched from real client."""
    data = paper_client.get_market_data("BTC-USD")

    assert data.price == Decimal("50000")
    mock_exchange_client.get_market_data.assert_called_once_with("BTC-USD")


def test_get_candles_delegates_to_real_client(paper_client, mock_exchange_client):
    """Test candles are fetched from real client."""
    candles = paper_client.get_candles("BTC-USD", "ONE_HOUR", 100)

    assert len(candles) > 0
    mock_exchange_client.get_candles.assert_called_once_with("BTC-USD", "ONE_HOUR", 100)


def test_market_buy_success(paper_client):
    """Test successful market buy."""
    quote_size = Decimal("1000")
    result = paper_client.market_buy("BTC-USD", quote_size)

    # Verify order result
    assert result.success is True
    assert result.side == "buy"
    assert result.status == "FILLED"
    assert result.size > Decimal("0")  # Received some BTC

    # Verify balances updated
    assert paper_client._quote_balance < Decimal("10000")  # USD spent
    assert paper_client._base_balance > Decimal("0")  # BTC received


def test_market_buy_with_slippage(paper_client):
    """Test market buy applies slippage."""
    quote_size = Decimal("1000")
    result = paper_client.market_buy("BTC-USD", quote_size)

    # Fill price should be higher than market price due to slippage
    market_price = Decimal("50000")
    assert result.filled_price > market_price


def test_market_buy_with_fee(paper_client):
    """Test market buy applies fees."""
    quote_size = Decimal("1000")
    initial_balance = paper_client._quote_balance

    result = paper_client.market_buy("BTC-USD", quote_size)

    # Fee should be deducted
    assert result.fee == quote_size * paper_client.TAKER_FEE
    assert paper_client._total_fees == result.fee

    # Quote spent should equal quote_size
    assert initial_balance - paper_client._quote_balance == quote_size


def test_market_buy_insufficient_balance(paper_client):
    """Test market buy with insufficient balance."""
    quote_size = Decimal("20000")  # More than available
    result = paper_client.market_buy("BTC-USD", quote_size)

    # Verify failure
    assert result.success is False
    assert "Insufficient" in result.error
    assert result.size == Decimal("0")

    # Balances unchanged
    assert paper_client._quote_balance == Decimal("10000")
    assert paper_client._base_balance == Decimal("0")


def test_market_buy_price_fetch_failure(paper_client, mock_exchange_client):
    """Test market buy when price fetch fails."""
    mock_exchange_client.get_current_price.side_effect = Exception("API error")

    result = paper_client.market_buy("BTC-USD", Decimal("1000"))

    assert result.success is False
    assert "Failed to get market price" in result.error


def test_market_sell_success(paper_client):
    """Test successful market sell."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))

    initial_base = paper_client._base_balance
    initial_quote = paper_client._quote_balance

    # Now sell it
    base_size = initial_base / 2
    result = paper_client.market_sell("BTC-USD", base_size)

    # Verify order result
    assert result.success is True
    assert result.side == "sell"
    assert result.status == "FILLED"
    assert result.size == base_size

    # Verify balances updated
    assert paper_client._base_balance < initial_base  # BTC sold
    assert paper_client._quote_balance > initial_quote  # USD received


def test_market_sell_with_slippage(paper_client):
    """Test market sell applies slippage."""
    # Buy first
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    base_size = paper_client._base_balance

    result = paper_client.market_sell("BTC-USD", base_size)

    # Fill price should be lower than market price due to slippage
    market_price = Decimal("50000")
    assert result.filled_price < market_price


def test_market_sell_with_fee(paper_client):
    """Test market sell applies fees."""
    # Buy first
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    base_size = paper_client._base_balance

    result = paper_client.market_sell("BTC-USD", base_size)

    # Fee should be calculated on gross proceeds
    gross_quote = base_size * result.filled_price
    expected_fee = gross_quote * paper_client.TAKER_FEE

    assert result.fee == expected_fee


def test_market_sell_insufficient_balance(paper_client):
    """Test market sell with insufficient balance."""
    base_size = Decimal("1.0")  # Don't have any BTC
    result = paper_client.market_sell("BTC-USD", base_size)

    # Verify failure
    assert result.success is False
    assert "Insufficient" in result.error
    assert result.size == Decimal("0")


def test_market_sell_price_fetch_failure(paper_client, mock_exchange_client):
    """Test market sell when price fetch fails."""
    # Buy first
    mock_exchange_client.get_current_price.return_value = Decimal("50000")
    paper_client.market_buy("BTC-USD", Decimal("1000"))

    # Now make price fetch fail
    mock_exchange_client.get_current_price.side_effect = Exception("API error")

    result = paper_client.market_sell("BTC-USD", Decimal("0.01"))

    assert result.success is False
    assert "Failed to get market price" in result.error


def test_get_order_existing(paper_client):
    """Test getting existing paper order."""
    # Execute a trade
    result = paper_client.market_buy("BTC-USD", Decimal("1000"))
    order_id = result.order_id

    # Get order details
    order = paper_client.get_order(order_id)

    assert order["order_id"] == order_id
    assert order["status"] == "FILLED"
    assert order["side"] == "buy"


def test_get_order_nonexistent(paper_client):
    """Test getting non-existent order."""
    order = paper_client.get_order("fake-order-id")

    assert order == {}


def test_cancel_order_always_fails(paper_client):
    """Test that paper orders cannot be cancelled."""
    # Execute a trade
    result = paper_client.market_buy("BTC-USD", Decimal("1000"))

    # Try to cancel (should always fail)
    cancelled = paper_client.cancel_order(result.order_id)

    assert cancelled is False


def test_get_portfolio_value(paper_client, mock_exchange_client):
    """Test portfolio value calculation."""
    # Buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))

    # Set price to 50000
    mock_exchange_client.get_current_price.return_value = Decimal("50000")

    value = paper_client.get_portfolio_value("BTC-USD")

    # Should be remaining USD + (BTC * price)
    expected = paper_client._quote_balance + (paper_client._base_balance * Decimal("50000"))
    assert value == expected


def test_get_statistics(paper_client):
    """Test getting trading statistics."""
    # Execute some trades
    paper_client.market_buy("BTC-USD", Decimal("1000"))
    paper_client.market_buy("BTC-USD", Decimal("500"))

    stats = paper_client.get_statistics()

    assert stats["total_trades"] == 2
    assert Decimal(stats["total_fees"]) > Decimal("0")
    assert Decimal(stats["total_volume"]) == Decimal("1500")


def test_get_trade_history(paper_client):
    """Test getting trade history."""
    # Execute trades
    paper_client.market_buy("BTC-USD", Decimal("1000"))
    paper_client.market_buy("BTC-USD", Decimal("500"))

    history = paper_client.get_trade_history()

    assert len(history) == 2
    assert all(isinstance(trade, PaperTrade) for trade in history)


def test_reset(paper_client):
    """Test resetting paper trading account."""
    # Execute some trades
    paper_client.market_buy("BTC-USD", Decimal("1000"))
    paper_client.market_buy("BTC-USD", Decimal("500"))

    # Reset
    paper_client.reset(initial_quote=20000.0, initial_base=1.0)

    # Verify reset
    assert paper_client._quote_balance == Decimal("20000")
    assert paper_client._base_balance == Decimal("1.0")
    assert len(paper_client._trades) == 0
    assert paper_client._total_fees == Decimal("0")
    assert paper_client._total_volume == Decimal("0")


# ============================================================================
# Limit IOC Order Tests
# ============================================================================

def test_limit_buy_ioc_success(paper_client, mock_exchange_client):
    """Test successful limit buy IOC order."""
    initial_quote = paper_client._quote_balance  # 10000
    base_size = Decimal("0.1")
    limit_price = Decimal("50050")  # Above ask (50010), should fill

    result = paper_client.limit_buy_ioc("BTC-USD", base_size, limit_price)

    # Verify order result
    assert result.success is True
    assert result.side == "buy"
    assert result.status == "FILLED"
    assert result.size == base_size
    # IOC orders fill at market ask price (50010), not limit price
    assert result.filled_price == Decimal("50010")

    # Verify balances updated
    assert paper_client._base_balance == base_size
    assert paper_client._quote_balance < initial_quote


def test_limit_buy_ioc_cancelled_below_ask(paper_client, mock_exchange_client):
    """Test limit buy IOC cancelled when limit price below ask."""
    initial_quote = paper_client._quote_balance
    base_size = Decimal("0.1")
    limit_price = Decimal("49000")  # Below ask (50010), should NOT fill

    result = paper_client.limit_buy_ioc("BTC-USD", base_size, limit_price)

    # Verify order cancelled
    assert result.success is True  # Order submitted successfully
    assert result.status == "CANCELLED"
    assert result.size == Decimal("0")  # Nothing filled

    # Verify balances unchanged
    assert paper_client._quote_balance == initial_quote
    assert paper_client._base_balance == Decimal("0")


def test_limit_buy_ioc_insufficient_balance(paper_client):
    """Test limit buy IOC with insufficient balance."""
    base_size = Decimal("1000")  # Way more than we can afford
    limit_price = Decimal("50050")

    result = paper_client.limit_buy_ioc("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert "Insufficient" in result.error


def test_limit_buy_ioc_uses_taker_fee(paper_client, mock_exchange_client):
    """Test that limit IOC uses taker fee (not maker fee)."""
    base_size = Decimal("0.1")
    limit_price = Decimal("50050")

    result = paper_client.limit_buy_ioc("BTC-USD", base_size, limit_price)

    # IOC crossing spread = taker fee (0.6%)
    expected_fee = base_size * Decimal("50010") * paper_client.TAKER_FEE
    assert result.fee == expected_fee


def test_limit_sell_ioc_success(paper_client, mock_exchange_client):
    """Test successful limit sell IOC order."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    initial_base = paper_client._base_balance
    initial_quote = paper_client._quote_balance

    base_size = initial_base / 2
    limit_price = Decimal("49000")  # Below bid (49990), should fill

    result = paper_client.limit_sell_ioc("BTC-USD", base_size, limit_price)

    # Verify order result
    assert result.success is True
    assert result.side == "sell"
    assert result.status == "FILLED"
    assert result.size == base_size
    # IOC orders fill at market bid price (49990), not limit price
    assert result.filled_price == Decimal("49990")

    # Verify balances updated
    assert paper_client._base_balance < initial_base
    assert paper_client._quote_balance > initial_quote


def test_limit_sell_ioc_cancelled_above_bid(paper_client, mock_exchange_client):
    """Test limit sell IOC cancelled when limit price above bid."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    initial_base = paper_client._base_balance
    initial_quote = paper_client._quote_balance

    base_size = initial_base / 2
    limit_price = Decimal("51000")  # Above bid (49990), should NOT fill

    result = paper_client.limit_sell_ioc("BTC-USD", base_size, limit_price)

    # Verify order cancelled
    assert result.success is True  # Order submitted successfully
    assert result.status == "CANCELLED"
    assert result.size == Decimal("0")  # Nothing filled

    # Verify balances unchanged
    assert paper_client._quote_balance == initial_quote
    assert paper_client._base_balance == initial_base


def test_limit_sell_ioc_insufficient_balance(paper_client):
    """Test limit sell IOC with insufficient balance."""
    base_size = Decimal("1.0")  # Don't have any BTC
    limit_price = Decimal("49000")

    result = paper_client.limit_sell_ioc("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert "Insufficient" in result.error


def test_limit_sell_ioc_uses_taker_fee(paper_client, mock_exchange_client):
    """Test that limit sell IOC uses taker fee (not maker fee)."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    base_size = paper_client._base_balance

    limit_price = Decimal("49000")  # Below bid, should fill

    result = paper_client.limit_sell_ioc("BTC-USD", base_size, limit_price)

    # IOC crossing spread = taker fee (0.6%)
    expected_fee = base_size * Decimal("49990") * paper_client.TAKER_FEE
    assert result.fee == expected_fee


def test_limit_buy_ioc_market_data_failure(paper_client, mock_exchange_client):
    """Test limit buy IOC when market data fetch fails."""
    mock_exchange_client.get_market_data.side_effect = Exception("API error")

    result = paper_client.limit_buy_ioc("BTC-USD", Decimal("0.1"), Decimal("50050"))

    assert result.success is False
    assert "Failed to get market data" in result.error


def test_limit_sell_ioc_market_data_failure(paper_client, mock_exchange_client):
    """Test limit sell IOC when market data fetch fails."""
    # First buy some BTC
    mock_exchange_client.get_market_data.return_value = MarketData(
        symbol="BTC-USD",
        price=Decimal("50000"),
        bid=Decimal("49990"),
        ask=Decimal("50010"),
        volume_24h=Decimal("1000"),
        timestamp=datetime.now()
    )
    paper_client.market_buy("BTC-USD", Decimal("1000"))

    # Now make market data fail
    mock_exchange_client.get_market_data.side_effect = Exception("API error")

    result = paper_client.limit_sell_ioc("BTC-USD", Decimal("0.01"), Decimal("49000"))

    assert result.success is False
    assert "Failed to get market data" in result.error


# ============================================================================
# Limit Post-Only Order Tests
# ============================================================================

def test_limit_buy_post_only_success(paper_client, mock_exchange_client):
    """Test successful limit buy post-only order."""
    initial_quote = paper_client._quote_balance  # 10000
    base_size = Decimal("0.1")
    limit_price = Decimal("49000")  # Below ask (50010), should fill as maker

    result = paper_client.limit_buy_post_only("BTC-USD", base_size, limit_price)

    # Verify order result
    assert result.success is True
    assert result.side == "buy"
    assert result.status == "FILLED"
    assert result.size == base_size
    # Post-only fills at limit price (maker gets their price)
    assert result.filled_price == limit_price

    # Verify balances updated
    assert paper_client._base_balance == base_size
    assert paper_client._quote_balance < initial_quote


def test_limit_buy_post_only_rejects_if_would_cross_spread(paper_client, mock_exchange_client):
    """Test limit buy post-only rejected when limit >= ask."""
    initial_quote = paper_client._quote_balance
    initial_base = paper_client._base_balance
    base_size = Decimal("0.1")
    limit_price = Decimal("50010")  # At ask, would take liquidity

    result = paper_client.limit_buy_post_only("BTC-USD", base_size, limit_price)

    # Verify order rejected
    assert result.success is False
    assert result.status == "REJECTED"
    assert result.size == Decimal("0")
    assert "POST_ONLY" in result.error
    assert "take liquidity" in result.error

    # Verify balances unchanged
    assert paper_client._quote_balance == initial_quote
    assert paper_client._base_balance == initial_base


def test_limit_buy_post_only_rejects_above_ask(paper_client, mock_exchange_client):
    """Test limit buy post-only rejected when limit > ask."""
    initial_quote = paper_client._quote_balance
    base_size = Decimal("0.1")
    limit_price = Decimal("51000")  # Above ask, would definitely take

    result = paper_client.limit_buy_post_only("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert result.status == "REJECTED"
    assert "POST_ONLY" in result.error


def test_limit_buy_post_only_insufficient_balance(paper_client):
    """Test limit buy post-only with insufficient balance."""
    base_size = Decimal("1000")  # Way more than we can afford
    limit_price = Decimal("49000")

    result = paper_client.limit_buy_post_only("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert "Insufficient" in result.error


def test_limit_buy_post_only_uses_maker_fee(paper_client, mock_exchange_client):
    """Test that limit buy post-only uses maker fee (not taker fee)."""
    base_size = Decimal("0.1")
    limit_price = Decimal("49000")  # Below ask

    result = paper_client.limit_buy_post_only("BTC-USD", base_size, limit_price)

    # Post-only = maker fee (0.4%)
    expected_fee = base_size * limit_price * paper_client.MAKER_FEE
    assert result.fee == expected_fee
    # Confirm maker fee is less than taker fee
    taker_fee = base_size * limit_price * paper_client.TAKER_FEE
    assert result.fee < taker_fee


def test_limit_buy_post_only_market_data_failure(paper_client, mock_exchange_client):
    """Test limit buy post-only when market data fetch fails."""
    mock_exchange_client.get_market_data.side_effect = Exception("API error")

    result = paper_client.limit_buy_post_only("BTC-USD", Decimal("0.1"), Decimal("49000"))

    assert result.success is False
    assert "Failed to get market data" in result.error


def test_limit_sell_post_only_success(paper_client, mock_exchange_client):
    """Test successful limit sell post-only order."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    initial_base = paper_client._base_balance
    initial_quote = paper_client._quote_balance

    base_size = initial_base / 2
    limit_price = Decimal("51000")  # Above bid (49990), should fill as maker

    result = paper_client.limit_sell_post_only("BTC-USD", base_size, limit_price)

    # Verify order result
    assert result.success is True
    assert result.side == "sell"
    assert result.status == "FILLED"
    assert result.size == base_size
    # Post-only fills at limit price (maker gets their price)
    assert result.filled_price == limit_price

    # Verify balances updated
    assert paper_client._base_balance < initial_base
    assert paper_client._quote_balance > initial_quote


def test_limit_sell_post_only_rejects_if_would_cross_spread(paper_client, mock_exchange_client):
    """Test limit sell post-only rejected when limit <= bid."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    initial_base = paper_client._base_balance
    initial_quote = paper_client._quote_balance

    base_size = initial_base / 2
    limit_price = Decimal("49990")  # At bid, would take liquidity

    result = paper_client.limit_sell_post_only("BTC-USD", base_size, limit_price)

    # Verify order rejected
    assert result.success is False
    assert result.status == "REJECTED"
    assert result.size == Decimal("0")
    assert "POST_ONLY" in result.error
    assert "take liquidity" in result.error

    # Verify balances unchanged
    assert paper_client._quote_balance == initial_quote
    assert paper_client._base_balance == initial_base


def test_limit_sell_post_only_rejects_below_bid(paper_client, mock_exchange_client):
    """Test limit sell post-only rejected when limit < bid."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    base_size = paper_client._base_balance / 2
    limit_price = Decimal("48000")  # Below bid, would definitely take

    result = paper_client.limit_sell_post_only("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert result.status == "REJECTED"
    assert "POST_ONLY" in result.error


def test_limit_sell_post_only_insufficient_balance(paper_client):
    """Test limit sell post-only with insufficient balance."""
    base_size = Decimal("1.0")  # Don't have any BTC
    limit_price = Decimal("51000")

    result = paper_client.limit_sell_post_only("BTC-USD", base_size, limit_price)

    assert result.success is False
    assert "Insufficient" in result.error


def test_limit_sell_post_only_uses_maker_fee(paper_client, mock_exchange_client):
    """Test that limit sell post-only uses maker fee (not taker fee)."""
    # First buy some BTC
    paper_client.market_buy("BTC-USD", Decimal("5000"))
    base_size = paper_client._base_balance

    limit_price = Decimal("51000")  # Above bid

    result = paper_client.limit_sell_post_only("BTC-USD", base_size, limit_price)

    # Post-only = maker fee (0.4%)
    expected_fee = base_size * limit_price * paper_client.MAKER_FEE
    assert result.fee == expected_fee
    # Confirm maker fee is less than taker fee
    taker_fee = base_size * limit_price * paper_client.TAKER_FEE
    assert result.fee < taker_fee


def test_limit_sell_post_only_market_data_failure(paper_client, mock_exchange_client):
    """Test limit sell post-only when market data fetch fails."""
    # First buy some BTC
    mock_exchange_client.get_market_data.return_value = MarketData(
        symbol="BTC-USD",
        price=Decimal("50000"),
        bid=Decimal("49990"),
        ask=Decimal("50010"),
        volume_24h=Decimal("1000"),
        timestamp=datetime.now()
    )
    paper_client.market_buy("BTC-USD", Decimal("1000"))

    # Now make market data fail
    mock_exchange_client.get_market_data.side_effect = Exception("API error")

    result = paper_client.limit_sell_post_only("BTC-USD", Decimal("0.01"), Decimal("51000"))

    assert result.success is False
    assert "Failed to get market data" in result.error


def test_post_only_vs_ioc_fee_comparison(paper_client, mock_exchange_client):
    """Test that post-only orders have lower fees than IOC orders."""
    base_size = Decimal("0.1")

    # Post-only buy (maker)
    post_only_result = paper_client.limit_buy_post_only(
        "BTC-USD", base_size, Decimal("49000")
    )

    # Reset and do IOC buy (taker)
    paper_client.reset(initial_quote=10000.0)
    ioc_result = paper_client.limit_buy_ioc(
        "BTC-USD", base_size, Decimal("50050")  # Above ask to ensure fill
    )

    # Post-only should have lower fees
    assert post_only_result.fee < ioc_result.fee
    # Specifically: maker fee (0.4%) vs taker fee (0.6%)
    assert paper_client.MAKER_FEE < paper_client.TAKER_FEE


# ============================================================================
# SymbolMapper Tests
# ============================================================================

def test_normalize_symbol_coinbase():
    """Test normalizing Coinbase symbols (already normalized)."""
    assert normalize_symbol("BTC-USD", Exchange.COINBASE) == "BTC-USD"
    assert normalize_symbol("eth-usd", Exchange.COINBASE) == "ETH-USD"


def test_normalize_symbol_kraken_slash():
    """Test normalizing Kraken slash format."""
    assert normalize_symbol("XBT/USD", Exchange.KRAKEN) == "BTC-USD"
    assert normalize_symbol("ETH/EUR", Exchange.KRAKEN) == "ETH-EUR"


def test_normalize_symbol_kraken_api_format():
    """Test normalizing Kraken API format (XXBTZUSD)."""
    assert normalize_symbol("XXBTZUSD", Exchange.KRAKEN) == "BTC-USD"
    assert normalize_symbol("XETHZEUR", Exchange.KRAKEN) == "ETH-EUR"


def test_normalize_symbol_kraken_simple():
    """Test normalizing Kraken simple format."""
    assert normalize_symbol("XBTUSD", Exchange.KRAKEN) == "BTC-USD"
    assert normalize_symbol("ETHUSD", Exchange.KRAKEN) == "ETH-USD"


def test_to_exchange_symbol_coinbase():
    """Test converting to Coinbase format."""
    assert to_exchange_symbol("BTC-USD", Exchange.COINBASE) == "BTC-USD"
    assert to_exchange_symbol("eth-eur", Exchange.COINBASE) == "ETH-EUR"


def test_to_exchange_symbol_kraken():
    """Test converting to Kraken format."""
    assert to_exchange_symbol("BTC-USD", Exchange.KRAKEN) == "XBT/USD"
    assert to_exchange_symbol("ETH-EUR", Exchange.KRAKEN) == "ETH/EUR"


def test_to_exchange_symbol_kraken_invalid_format():
    """Test invalid format raises error."""
    with pytest.raises(ValueError, match="Invalid symbol format"):
        to_exchange_symbol("BTCUSD", Exchange.KRAKEN)


def test_to_kraken_asset():
    """Test converting to Kraken asset codes."""
    assert to_kraken_asset("BTC") == "XXBT"
    assert to_kraken_asset("USD") == "ZUSD"
    assert to_kraken_asset("ETH") == "XETH"
    assert to_kraken_asset("EUR") == "ZEUR"


def test_to_kraken_asset_no_prefix():
    """Test assets without special prefixes."""
    assert to_kraken_asset("DOGE") == "XDG"  # Has symbol map but no prefix
    assert to_kraken_asset("ADA") == "ADA"  # No special mapping


def test_from_kraken_asset():
    """Test converting from Kraken asset codes."""
    assert from_kraken_asset("XXBT") == "BTC"
    assert from_kraken_asset("ZUSD") == "USD"
    assert from_kraken_asset("XETH") == "ETH"
    assert from_kraken_asset("ZEUR") == "EUR"


def test_from_kraken_asset_x_prefix():
    """Test removing X prefix."""
    assert from_kraken_asset("XDGE") == "DGE"  # Unknown with X prefix


def test_from_kraken_asset_z_prefix():
    """Test removing Z prefix."""
    assert from_kraken_asset("ZGBP") == "GBP"


def test_parse_trading_pair_dash():
    """Test parsing pair with dash separator."""
    base, quote = parse_trading_pair("BTC-USD")
    assert base == "BTC"
    assert quote == "USD"


def test_parse_trading_pair_slash():
    """Test parsing pair with slash separator."""
    base, quote = parse_trading_pair("XBT/USD")
    assert base == "BTC"
    assert quote == "USD"


def test_parse_trading_pair_underscore():
    """Test parsing pair with underscore separator."""
    base, quote = parse_trading_pair("BTC_USD")
    assert base == "BTC"
    assert quote == "USD"


def test_parse_trading_pair_six_char():
    """Test parsing 6-char format."""
    base, quote = parse_trading_pair("XBTUSD")
    assert base == "BTC"
    assert quote == "USD"


def test_parse_trading_pair_invalid():
    """Test parsing invalid format."""
    with pytest.raises(ValueError, match="Cannot parse trading pair"):
        parse_trading_pair("INVALID")


def test_to_kraken_granularity():
    """Test converting granularity to Kraken format."""
    assert to_kraken_granularity("ONE_MINUTE") == 1
    assert to_kraken_granularity("FIVE_MINUTE") == 5
    assert to_kraken_granularity("ONE_HOUR") == 60
    assert to_kraken_granularity("ONE_DAY") == 1440


def test_to_kraken_granularity_unknown():
    """Test unknown granularity defaults to ONE_HOUR."""
    assert to_kraken_granularity("UNKNOWN") == 60


def test_from_kraken_granularity():
    """Test converting from Kraken granularity."""
    assert from_kraken_granularity(1) == "ONE_MINUTE"
    assert from_kraken_granularity(5) == "FIVE_MINUTE"
    assert from_kraken_granularity(60) == "ONE_HOUR"
    assert from_kraken_granularity(1440) == "ONE_DAY"


def test_from_kraken_granularity_unknown():
    """Test unknown minutes defaults to ONE_HOUR."""
    assert from_kraken_granularity(999) == "ONE_HOUR"


# ============================================================================
# Integration Tests
# ============================================================================

def test_paper_client_full_trade_cycle(mock_exchange_client):
    """Test complete buy-sell cycle."""
    client = PaperTradingClient(
        real_client=mock_exchange_client,
        initial_quote=10000.0,
        initial_base=0.0
    )

    initial_value = client.get_portfolio_value()

    # Buy BTC
    buy_result = client.market_buy("BTC-USD", Decimal("5000"))
    assert buy_result.success is True

    # Sell half
    sell_result = client.market_sell("BTC-USD", buy_result.size / 2)
    assert sell_result.success is True

    # Verify we have both currencies
    assert client._base_balance > Decimal("0")
    assert client._quote_balance > Decimal("0")

    # Portfolio value should be close to initial (minus fees and slippage)
    final_value = client.get_portfolio_value()
    loss = initial_value - final_value
    loss_percent = (loss / initial_value) * 100

    # Loss should be small (fees + slippage)
    assert loss_percent < 2  # Less than 2% loss


def test_paper_client_multiple_trades_statistics(mock_exchange_client):
    """Test statistics accumulation over multiple trades."""
    client = PaperTradingClient(
        real_client=mock_exchange_client,
        initial_quote=10000.0
    )

    # Execute multiple trades
    client.market_buy("BTC-USD", Decimal("1000"))
    client.market_buy("BTC-USD", Decimal("1000"))

    # Get base balance after buys
    base_after_buys = client._base_balance

    # Sell half
    client.market_sell("BTC-USD", base_after_buys / 2)

    stats = client.get_statistics()

    # Verify statistics
    assert int(stats["total_trades"]) == 3
    assert Decimal(stats["total_fees"]) > Decimal("0")
    # Volume should be sum of buys + sell (gross amounts)
    assert Decimal(stats["total_volume"]) > Decimal("2000")
