"""
CRITICAL TESTS for live exchange clients.

These tests verify the components that execute REAL trades with REAL money.
Bugs in these modules can cause direct financial loss.

Tests cover:
- Coinbase client (order execution, balance fetching, error handling)
- Kraken client (order execution, rate limiting, symbol mapping)
- Exchange factory (correct client creation)

All tests use mocked API responses - NO real API calls are made.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import pandas as pd

from src.api.coinbase_client import CoinbaseClient
from src.api.kraken_client import KrakenClient, KrakenRateLimitError
from src.api.exchange_protocol import Balance, MarketData, OrderResult


# ============================================================================
# Coinbase Client Tests - CRITICAL for Real Money Trades
# ============================================================================

@pytest.fixture
def coinbase_client():
    """Create Coinbase client with mocked authentication."""
    with patch('src.api.coinbase_client.build_jwt', return_value="mock_jwt_token"):
        return CoinbaseClient(api_key="test_key", api_secret="test_secret")


@pytest.fixture
def mock_coinbase_response():
    """Mock successful Coinbase API response."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True}
    return mock_resp


def test_coinbase_client_initialization_requires_credentials():
    """CRITICAL: Verify client cannot be created without credentials."""
    with pytest.raises(ValueError, match="Must provide api_key"):
        CoinbaseClient()


def test_coinbase_client_initialization_with_key_file():
    """Test client initialization from key file."""
    with patch('src.api.coinbase_client.load_key_file', return_value=("key", "secret")):
        with patch('src.api.coinbase_client.build_jwt', return_value="token"):
            client = CoinbaseClient(key_file=Path("/fake/path.json"))
            assert client.api_key == "key"
            assert client.api_secret == "secret"


def test_coinbase_get_balance_success(coinbase_client):
    """CRITICAL: Test balance fetching returns correct values."""
    mock_accounts = {
        "accounts": [
            {
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.1"}
            },
            {
                "currency": "USD",
                "available_balance": {"value": "10000"},
                "hold": {"value": "500"}
            }
        ]
    }

    with patch.object(coinbase_client, '_request', return_value=mock_accounts):
        balance = coinbase_client.get_balance("BTC")

        assert balance.currency == "BTC"
        assert balance.available == Decimal("1.5")
        assert balance.hold == Decimal("0.1")
        assert balance.total == Decimal("1.6")


def test_coinbase_get_balance_currency_not_found(coinbase_client):
    """Test balance for currency not in account returns zero."""
    mock_accounts = {"accounts": []}

    with patch.object(coinbase_client, '_request', return_value=mock_accounts):
        balance = coinbase_client.get_balance("ETH")

        assert balance.currency == "ETH"
        assert balance.available == Decimal("0")
        assert balance.hold == Decimal("0")


def test_coinbase_get_current_price(coinbase_client):
    """CRITICAL: Test price fetching returns correct decimal value."""
    mock_product = {"price": "50123.45"}

    with patch.object(coinbase_client, '_request', return_value=mock_product):
        price = coinbase_client.get_current_price("BTC-USD")

        assert price == Decimal("50123.45")
        assert isinstance(price, Decimal)


def test_coinbase_get_market_data(coinbase_client):
    """Test market data fetching."""
    # get_market_data calls ticker endpoint for bid/ask and get_product for volume
    mock_ticker = {
        "best_bid": "49990",
        "best_ask": "50010",
    }
    mock_product = {
        "price": "50000",
        "volume_24h": "12345.67"
    }

    with patch.object(coinbase_client, '_request', return_value=mock_ticker):
        with patch.object(coinbase_client, 'get_product', return_value=mock_product):
            data = coinbase_client.get_market_data("BTC-USD")

            assert data.symbol == "BTC-USD"
            # Price is mid of bid/ask: (49990 + 50010) / 2 = 50000
            assert data.price == Decimal("50000")
            assert data.bid == Decimal("49990")
            assert data.ask == Decimal("50010")
            assert data.volume_24h == Decimal("12345.67")


def test_coinbase_market_buy_success(coinbase_client):
    """CRITICAL: Test market buy executes correctly."""
    # Coinbase API returns order data in success_response or at root level
    mock_order_response = {
        "success_response": {
            "order_id": "order-123",
            "status": "PENDING",
        }
    }
    # Polling returns filled order status
    mock_poll_result = {
        "order_id": "order-123",
        "status": "FILLED",
        "filled_size": "0.02",
        "average_filled_price": "50000",
        "total_fees": "5.00"
    }

    with patch.object(coinbase_client, '_request', return_value=mock_order_response):
        with patch.object(coinbase_client, '_poll_order_fill', return_value=mock_poll_result):
            result = coinbase_client.market_buy("BTC-USD", Decimal("1000"))

            # Verify order was executed
            assert result.success is True
            assert result.order_id == "order-123"
            assert result.side == "buy"
            assert result.size == Decimal("0.02")
            assert result.filled_price == Decimal("50000")
            assert result.fee == Decimal("5.00")


def test_coinbase_market_buy_wrong_direction_prevented(coinbase_client):
    """CRITICAL: Verify buy order uses BUY side, not SELL."""
    with patch.object(coinbase_client, '_request') as mock_request:
        mock_request.return_value = {"success": True, "order_id": "test"}

        coinbase_client.market_buy("BTC-USD", Decimal("1000"))

        # Verify the API was called with correct side
        call_args = mock_request.call_args_list[0]
        order_data = call_args[1]['data']
        assert order_data['side'] == "BUY"  # NOT "SELL"


def test_coinbase_market_sell_success(coinbase_client):
    """CRITICAL: Test market sell executes correctly."""
    mock_order_response = {
        "success_response": {
            "order_id": "order-456",
            "status": "PENDING",
        }
    }
    # Polling returns filled order status
    mock_poll_result = {
        "order_id": "order-456",
        "status": "FILLED",
        "filled_size": "0.05",
        "average_filled_price": "51000",
        "total_fees": "12.75"
    }

    with patch.object(coinbase_client, '_request', return_value=mock_order_response):
        with patch.object(coinbase_client, '_poll_order_fill', return_value=mock_poll_result):
            result = coinbase_client.market_sell("BTC-USD", Decimal("0.05"))

            assert result.success is True
            assert result.side == "sell"
            assert result.size == Decimal("0.05")
            assert result.filled_price == Decimal("51000")


def test_coinbase_market_sell_wrong_direction_prevented(coinbase_client):
    """CRITICAL: Verify sell order uses SELL side, not BUY."""
    with patch.object(coinbase_client, '_request') as mock_request:
        mock_request.return_value = {"success": True, "order_id": "test"}

        coinbase_client.market_sell("BTC-USD", Decimal("0.1"))

        # Verify the API was called with correct side
        call_args = mock_request.call_args_list[0]
        order_data = call_args[1]['data']
        assert order_data['side'] == "SELL"  # NOT "BUY"


def test_coinbase_market_buy_api_error(coinbase_client):
    """CRITICAL: Test buy order handles API errors gracefully."""
    with patch.object(coinbase_client, '_request', side_effect=Exception("API Error")):
        result = coinbase_client.market_buy("BTC-USD", Decimal("1000"))

        assert result.success is False
        assert result.error is not None
        assert "API Error" in result.error or "error" in result.error.lower()


def test_coinbase_market_sell_api_error(coinbase_client):
    """CRITICAL: Test sell order handles API errors gracefully."""
    with patch.object(coinbase_client, '_request', side_effect=Exception("Insufficient funds")):
        result = coinbase_client.market_sell("BTC-USD", Decimal("0.1"))

        assert result.success is False
        assert result.error is not None


def test_coinbase_request_builds_correct_auth_header(coinbase_client):
    """Test authentication header is built correctly."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    with patch('src.api.coinbase_client.build_jwt', return_value="test_token") as mock_jwt:
        with patch.object(coinbase_client.session, 'request', return_value=mock_response):
            coinbase_client._request("GET", "/test/path")

            # Verify JWT was built with correct parameters
            mock_jwt.assert_called_once_with(
                coinbase_client.api_key,
                coinbase_client.api_secret,
                "GET",
                "/test/path"
            )


def test_coinbase_request_handles_non_200_status(coinbase_client):
    """Test error handling for non-200 HTTP status."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"

    with patch('src.api.coinbase_client.build_jwt', return_value="test_token"):
        with patch.object(coinbase_client.session, 'request', return_value=mock_response):
            with pytest.raises(Exception, match="API error 400"):
                coinbase_client._request("GET", "/test")


def test_coinbase_get_candles(coinbase_client):
    """Test fetching OHLCV data."""
    mock_candles = {
        "candles": [
            {
                "start": "1609459200",
                "open": "29000",
                "high": "29500",
                "low": "28500",
                "close": "29200",
                "volume": "100"
            },
            {
                "start": "1609462800",
                "open": "29200",
                "high": "29800",
                "low": "29000",
                "close": "29500",
                "volume": "150"
            }
        ]
    }

    with patch.object(coinbase_client, '_request', return_value=mock_candles):
        df = coinbase_client.get_candles("BTC-USD", "ONE_HOUR", 2)

        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df["open"].iloc[0] == Decimal("29000")
        assert df["close"].iloc[1] == Decimal("29500")


# ============================================================================
# Kraken Client Tests - CRITICAL for Real Money Trades
# ============================================================================

@pytest.fixture
def kraken_client():
    """Create Kraken client with mocked authentication."""
    with patch('src.api.kraken_client.validate_credentials'):
        return KrakenClient(api_key="test_key", api_secret="test_secret")


def test_kraken_client_initialization_validates_credentials():
    """CRITICAL: Verify client validates credentials on init."""
    with patch('src.api.kraken_client.validate_credentials', side_effect=ValueError("Invalid")):
        with pytest.raises(ValueError, match="Invalid"):
            KrakenClient(api_key="bad", api_secret="bad")


def test_kraken_symbol_mapping_btc_to_xbt(kraken_client):
    """CRITICAL: Verify BTC -> XBT symbol translation."""
    # Kraken uses XBT instead of BTC
    # Note: _public_request already extracts 'result' from response
    with patch.object(kraken_client, '_public_request') as mock_request:
        mock_request.return_value = {
            "XBTUSD": {"c": ["50000", "1"]}  # _public_request returns the 'result' part
        }

        price = kraken_client.get_current_price("BTC-USD")

        # Verify it called with XBTUSD (no slash for API call)
        # _public_request is called as: _public_request("Ticker", {"pair": "XBTUSD"})
        call_args = mock_request.call_args.args
        assert call_args[0] == "Ticker"
        assert call_args[1]['pair'] == "XBTUSD"


def test_kraken_get_balance(kraken_client):
    """CRITICAL: Test balance fetching with Kraken asset codes."""
    # _private_request already extracts 'result' from response
    mock_balance = {
        "XXBT": "1.5000000000",  # Kraken uses XXBT for BTC
        "ZUSD": "10000.0000"     # Kraken uses ZUSD for USD
    }

    with patch.object(kraken_client, '_private_request', return_value=mock_balance):
        balance = kraken_client.get_balance("BTC")

        assert balance.currency == "BTC"
        assert balance.available == Decimal("1.5")


def test_kraken_market_buy_uses_correct_symbol(kraken_client):
    """CRITICAL: Verify buy uses correct Kraken symbol format."""
    # _private_request already extracts 'result' from response
    mock_order = {
        "txid": ["order-123"]
    }

    mock_order_info = {
        "order-123": {
            "status": "closed",
            "vol_exec": "0.02",
            "price": "50000",
            "fee": "5.00",
            "descr": {"type": "buy"}
        }
    }

    with patch.object(kraken_client, 'get_current_price', return_value=Decimal("50000")):
        with patch.object(kraken_client, '_private_request') as mock_request:
            mock_request.side_effect = [mock_order, mock_order_info]

            kraken_client.market_buy("BTC-USD", Decimal("1000"))

            # First call should be AddOrder with correct symbol
            # _private_request is called as: _private_request("AddOrder", order_data)
            first_call = mock_request.call_args_list[0]
            assert first_call.args[0] == "AddOrder"  # endpoint name
            first_call_data = first_call.args[1]  # data is second positional arg
            assert first_call_data['pair'] == "XBTUSD"  # Kraken format (no slash)


def test_kraken_market_buy_uses_buy_type(kraken_client):
    """CRITICAL: Verify buy order type is 'buy', not 'sell'."""
    with patch.object(kraken_client, 'get_current_price', return_value=Decimal("50000")):
        with patch.object(kraken_client, '_private_request') as mock_request:
            mock_request.return_value = {"txid": ["test"]}

            kraken_client.market_buy("BTC-USD", Decimal("1000"))

            # market_buy makes multiple calls to _private_request, check the FIRST one (AddOrder)
            first_call_data = mock_request.call_args_list[0].args[1]  # data is second positional arg
            assert first_call_data['type'] == "buy"  # NOT "sell"


def test_kraken_market_sell_uses_sell_type(kraken_client):
    """CRITICAL: Verify sell order type is 'sell', not 'buy'."""
    with patch.object(kraken_client, '_private_request') as mock_request:
        mock_request.return_value = {"txid": ["test"]}

        kraken_client.market_sell("BTC-USD", Decimal("0.1"))

        # market_sell makes multiple calls to _private_request, check the FIRST one (AddOrder)
        first_call_data = mock_request.call_args_list[0].args[1]  # data is second positional arg
        assert first_call_data['type'] == "sell"  # NOT "buy"


def test_kraken_rate_limiting_waits_when_counter_full(kraken_client):
    """CRITICAL: Test rate limiting prevents exceeding API limits."""
    # Set counter near limit
    kraken_client._counter = 14.5
    kraken_client._last_counter_update = kraken_client._last_counter_update

    with patch('time.sleep') as mock_sleep:
        kraken_client._wait_for_counter_capacity(cost=2)

        # Should have slept to wait for counter decay
        mock_sleep.assert_called()


def test_kraken_rate_limiting_respects_backoff(kraken_client):
    """Test rate limit backoff period is respected."""
    import time
    kraken_client._rate_limit_backoff_until = time.time() + 10

    with patch('time.sleep') as mock_sleep:
        kraken_client._wait_for_counter_capacity()

        # Should sleep for backoff period
        assert mock_sleep.called
        sleep_time = mock_sleep.call_args[0][0]
        assert sleep_time > 0


def test_kraken_get_current_price(kraken_client):
    """Test price fetching."""
    # _public_request already extracts 'result' from response
    mock_ticker = {
        "XBTUSD": {
            "c": ["50123.45", "0.05"]  # [price, lot volume]
        }
    }

    with patch.object(kraken_client, '_public_request', return_value=mock_ticker):
        price = kraken_client.get_current_price("BTC-USD")

        assert price == Decimal("50123.45")


def test_kraken_market_buy_error_handling(kraken_client):
    """CRITICAL: Test buy handles errors gracefully."""
    with patch.object(kraken_client, '_private_request', side_effect=Exception("Insufficient funds")):
        result = kraken_client.market_buy("BTC-USD", Decimal("1000"))

        assert result.success is False
        assert result.error is not None


def test_kraken_market_sell_error_handling(kraken_client):
    """CRITICAL: Test sell handles errors gracefully."""
    with patch.object(kraken_client, '_private_request', side_effect=Exception("Insufficient balance")):
        result = kraken_client.market_sell("BTC-USD", Decimal("0.1"))

        assert result.success is False
        assert result.error is not None


# ============================================================================
# Exchange Factory Tests - CRITICAL for Client Selection
# ============================================================================

def test_exchange_factory_creates_coinbase_client_from_key_file():
    """Test factory creates Coinbase client with key file."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings, Exchange

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = Exchange.COINBASE
    mock_settings.coinbase_key_file = Path("/fake/key.json")

    with patch('src.api.exchange_factory.CoinbaseClient') as mock_coinbase:
        create_exchange_client(mock_settings)

        mock_coinbase.assert_called_once_with(key_file=Path("/fake/key.json"))


def test_exchange_factory_creates_coinbase_client_from_api_keys():
    """Test factory creates Coinbase client with API keys."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings, Exchange
    from pydantic import SecretStr

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = Exchange.COINBASE
    mock_settings.coinbase_key_file = None
    mock_settings.coinbase_api_key = SecretStr("test_key")
    mock_settings.coinbase_api_secret = SecretStr("test_secret")

    with patch('src.api.exchange_factory.CoinbaseClient') as mock_coinbase:
        create_exchange_client(mock_settings)

        mock_coinbase.assert_called_once()
        call_kwargs = mock_coinbase.call_args[1]
        assert call_kwargs['api_key'] == "test_key"
        assert call_kwargs['api_secret'] == "test_secret"


def test_exchange_factory_creates_kraken_client():
    """Test factory creates Kraken client."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings, Exchange
    from pydantic import SecretStr

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = Exchange.KRAKEN
    mock_settings.kraken_api_key = SecretStr("test_key")
    mock_settings.kraken_api_secret = SecretStr("test_secret")

    with patch('src.api.exchange_factory.KrakenClient') as mock_kraken:
        create_exchange_client(mock_settings)

        mock_kraken.assert_called_once()
        call_kwargs = mock_kraken.call_args[1]
        assert call_kwargs['api_key'] == "test_key"
        assert call_kwargs['api_secret'] == "test_secret"


def test_exchange_factory_raises_on_missing_coinbase_credentials():
    """CRITICAL: Verify factory fails without credentials."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings, Exchange

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = Exchange.COINBASE
    mock_settings.coinbase_key_file = None
    mock_settings.coinbase_api_key = None
    mock_settings.coinbase_api_secret = None

    with pytest.raises(ValueError, match="Coinbase credentials not configured"):
        create_exchange_client(mock_settings)


def test_exchange_factory_raises_on_missing_kraken_credentials():
    """CRITICAL: Verify factory fails without Kraken credentials."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings, Exchange

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = Exchange.KRAKEN
    mock_settings.kraken_api_key = None
    mock_settings.kraken_api_secret = None

    with pytest.raises(ValueError, match="Kraken credentials not configured"):
        create_exchange_client(mock_settings)


def test_exchange_factory_raises_on_unsupported_exchange():
    """Test factory raises error for unsupported exchange."""
    from src.api.exchange_factory import create_exchange_client
    from config.settings import Settings

    mock_settings = Mock(spec=Settings)
    mock_settings.exchange = "binance"  # Unsupported

    with pytest.raises(ValueError, match="Unsupported exchange"):
        create_exchange_client(mock_settings)


# ============================================================================
# Integration Tests - Order Flow
# ============================================================================

def test_coinbase_full_buy_order_flow(coinbase_client):
    """Test complete buy order flow from submission to confirmation."""
    mock_response = {
        "success_response": {
            "order_id": "order-123",
            "status": "PENDING",
        }
    }
    # Polling returns filled order status
    mock_poll_result = {
        "order_id": "order-123",
        "status": "FILLED",
        "filled_size": "0.02",
        "average_filled_price": "50000",
        "total_fees": "5.00"
    }

    with patch.object(coinbase_client, '_request', return_value=mock_response):
        with patch.object(coinbase_client, '_poll_order_fill', return_value=mock_poll_result):
            result = coinbase_client.market_buy("BTC-USD", Decimal("1000"))

            # Verify complete flow
            assert result.success is True
            assert result.order_id == "order-123"
            assert result.status == "FILLED"


def test_kraken_full_sell_order_flow(kraken_client):
    """Test complete sell order flow."""
    # _private_request already extracts 'result' from response
    mock_submit = {"txid": ["order-456"]}
    mock_details = {
        "order-456": {
            "status": "closed",
            "vol_exec": "0.05",
            "price": "51000",
            "fee": "12.75",
            "descr": {"type": "sell"}
        }
    }

    with patch.object(kraken_client, '_private_request') as mock_request:
        mock_request.side_effect = [mock_submit, mock_details]

        result = kraken_client.market_sell("BTC-USD", Decimal("0.05"))

        # Verify complete flow
        assert result.success is True
        assert result.order_id == "order-456"
        assert result.status == "closed"  # Kraken uses "closed" not "FILLED"
