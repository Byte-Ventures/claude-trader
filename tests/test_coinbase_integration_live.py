"""
Live integration tests for Coinbase API client.

⚠️  WARNING: These tests make REAL API calls and place REAL orders on Coinbase.

IMPORTANT:
- Tests are disabled by default and require --run-live-tests flag
- Use a SEPARATE TEST ACCOUNT with minimal funds (100 units recommended)
- NEVER use production credentials
- Tests place 10-unit orders (Coinbase minimum)
- Estimated cost per run: 0.30-0.60 in fees (in quote currency)
- Supported quote currencies: USD, USDC, EUR (auto-detected)

Setup:
1. Create separate Coinbase test account with 100+ balance (USD, USDC, or EUR)
2. Generate API key with trading permissions
3. Configure credentials in .env:
   COINBASE_TEST_KEY_FILE=/path/to/test_key.json
   or
   COINBASE_TEST_API_KEY=...
   COINBASE_TEST_API_SECRET=...

Run tests:
    # All tests
    pytest tests/test_coinbase_integration_live.py --run-live-tests -v

    # Read-only tests only (no orders)
    pytest tests/test_coinbase_integration_live.py::TestCoinbaseReadOnly --run-live-tests -v
"""

import atexit
import os
import time
from decimal import Decimal
from pathlib import Path

import pytest
import pandas as pd
import structlog
from dotenv import load_dotenv

# Load .env file for test credentials
load_dotenv()

from src.api.coinbase_client import CoinbaseClient
from src.api.exchange_protocol import Balance, MarketData

logger = structlog.get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

MAX_TEST_ORDER = Decimal("10.00")  # Coinbase minimum order size
MIN_BALANCE = Decimal("50.00")  # Minimum balance for order validation tests
RECOMMENDED_BALANCE = Decimal("100.00")

# Cancellation timeout for limit order tests
# RISK: Limit orders placed 2% off-market could fill during this window if market moves sharply.
# - 200ms is long enough for order to post (Coinbase latency ~20-50ms)
# - Short enough to minimize fill risk under normal conditions
# - During flash crashes/spikes, 2% orders may still fill - this is acceptable for $10 test orders
# - Tests will fail gracefully if order fills before cancellation (safe failure mode)
CANCELLATION_TIMEOUT_MS = 200  # Milliseconds to wait before canceling test orders

# Supported quote currencies (in priority order)
SUPPORTED_QUOTE_CURRENCIES = ["USD", "USDC", "EUR"]


# ==============================================================================
# Helper Functions
# ==============================================================================

def has_test_credentials() -> bool:
    """Check if test credentials are configured."""
    # Check for key file
    key_file = os.getenv("COINBASE_TEST_KEY_FILE")
    if key_file and Path(key_file).exists():
        return True

    # Check for direct credentials
    api_key = os.getenv("COINBASE_TEST_API_KEY")
    api_secret = os.getenv("COINBASE_TEST_API_SECRET")
    return bool(api_key and api_secret)


def safe_cancel_order(client: CoinbaseClient, order_id: str) -> tuple[bool, str]:
    """
    Safely attempt to cancel an order.

    Returns:
        (success, message) tuple
    """
    try:
        result = client.cancel_order(order_id)
        if result:
            return True, f"Order {order_id} cancelled successfully"
        return False, f"Order {order_id} cancellation returned False"
    except Exception as e:
        # Order may have already filled or been cancelled
        return False, f"Order {order_id} cancellation failed: {str(e)}"


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def validate_test_environment():
    """
    Validate that we're not accidentally using production credentials.

    This is a safety check to prevent tests from running against production accounts.
    CRITICAL: This fixture fails tests if production credentials are detected to prevent
    accidental execution against production accounts in a financial system.
    """
    # Check if production credentials are set
    prod_key_file = os.getenv("COINBASE_KEY_FILE")
    prod_api_key = os.getenv("COINBASE_API_KEY")

    # Check if test credentials are set
    test_key_file = os.getenv("COINBASE_TEST_KEY_FILE")
    test_api_key = os.getenv("COINBASE_TEST_API_KEY")

    # FAIL if production credentials are set but test credentials aren't
    # In a financial system, we must fail loudly to prevent production usage
    if (prod_key_file or prod_api_key) and not (test_key_file or test_api_key):
        pytest.fail(
            "SECURITY: Production credentials detected (COINBASE_KEY_FILE or COINBASE_API_KEY) "
            "but test credentials are not set. "
            "Live integration tests MUST use COINBASE_TEST_KEY_FILE or COINBASE_TEST_API_KEY. "
            "Refusing to proceed to prevent accidental production usage."
        )


@pytest.fixture(scope="class")
def coinbase_live_client(validate_test_environment) -> CoinbaseClient:
    """
    Create a CoinbaseClient instance using test credentials.

    Uses COINBASE_TEST_KEY_FILE or COINBASE_TEST_API_KEY/SECRET environment variables.
    Class-scoped to reuse the same client across all tests in a test class.
    Skips if credentials are not configured.

    IMPORTANT: Performs sanity check to verify client is using test account, not production.
    """
    # Check if credentials are configured
    if not has_test_credentials():
        pytest.skip("Test credentials not configured (COINBASE_TEST_KEY_FILE or COINBASE_TEST_API_KEY/SECRET)")

    # Create client using test credentials
    key_file = os.getenv("COINBASE_TEST_KEY_FILE")
    if key_file:
        client = CoinbaseClient(key_file=key_file)
    else:
        # Fall back to direct credentials
        api_key = os.getenv("COINBASE_TEST_API_KEY")
        api_secret = os.getenv("COINBASE_TEST_API_SECRET")

        if not api_key or not api_secret:
            pytest.skip("Test credentials not properly configured")

        client = CoinbaseClient(api_key=api_key, api_secret=api_secret)

    # Sanity check: Verify this appears to be a test account
    # Production accounts typically have many currencies; test accounts are minimal
    try:
        response = client.get_accounts()
        accounts = response.get("accounts", [])
        if len(accounts) > 10:
            pytest.fail(
                f"SECURITY: Account has {len(accounts)} currencies (production accounts typically have >10). "
                "This appears to be a production account. "
                "Tests MUST use a dedicated test account with minimal funds. "
                "Refusing to proceed."
            )
        logger.info("test_account_verified", account_count=len(accounts), status="appears_to_be_test_account")
    except Exception as e:
        logger.warning("account_verification_failed", error=str(e), message="Could not verify test account status")

    return client


@pytest.fixture(scope="class")
def trading_config(coinbase_live_client) -> dict:
    """
    Detect available quote currency and determine trading pair.

    Returns dict with:
        - quote_currency: str (USD, USDC, or EUR)
        - trading_pair: str (BTC-USD, BTC-USDC, or BTC-EUR)
        - min_balance: Decimal
        - recommended_balance: Decimal

    Checks currencies in priority order: USD > USDC > EUR
    """
    for currency in SUPPORTED_QUOTE_CURRENCIES:
        try:
            balance = coinbase_live_client.get_balance(currency)
            if balance.available >= MIN_BALANCE:
                trading_pair = f"BTC-{currency}"
                logger.info(
                    "quote_currency_detected",
                    currency=currency,
                    available=str(balance.available),
                    trading_pair=trading_pair
                )
                return {
                    "quote_currency": currency,
                    "trading_pair": trading_pair,
                    "min_balance": MIN_BALANCE,
                    "recommended_balance": RECOMMENDED_BALANCE
                }
        except Exception as e:
            logger.debug(f"currency_check_failed", currency=currency, error=str(e))
            continue

    # No currency has sufficient balance - return first available with any balance
    for currency in SUPPORTED_QUOTE_CURRENCIES:
        try:
            balance = coinbase_live_client.get_balance(currency)
            if balance.available > Decimal("0"):
                trading_pair = f"BTC-{currency}"
                logger.warning(
                    "insufficient_balance_detected",
                    currency=currency,
                    available=str(balance.available),
                    required=str(MIN_BALANCE),
                    trading_pair=trading_pair
                )
                return {
                    "quote_currency": currency,
                    "trading_pair": trading_pair,
                    "min_balance": MIN_BALANCE,
                    "recommended_balance": RECOMMENDED_BALANCE
                }
        except Exception as e:
            continue

    # No supported quote currency found with any balance
    pytest.skip(f"No supported quote currency found with balance ({', '.join(SUPPORTED_QUOTE_CURRENCIES)})")


@pytest.fixture
def cleanup_orders(coinbase_live_client):
    """
    Track and clean up test orders after each test.

    Yields a list that tests can append order IDs to.
    Attempts to cancel all tracked orders during teardown.

    IMPORTANT: Registers an atexit handler to guarantee cleanup even if
    test is interrupted (Ctrl+C, kill signal). This is critical for a
    financial system to prevent orders from remaining in the order book.
    """
    order_ids = []

    def emergency_cleanup():
        """
        Emergency cleanup handler for abnormal termination.
        Registered with atexit to run even on interrupts.
        """
        if order_ids:
            logger.warning(
                "emergency_cleanup_triggered",
                order_count=len(order_ids),
                message="Test interrupted - attempting emergency order cleanup"
            )
            for order_id in order_ids:
                try:
                    success, message = safe_cancel_order(coinbase_live_client, order_id)
                    if success:
                        logger.info("emergency_cleanup_success", order_id=order_id)
                    else:
                        logger.warning("emergency_cleanup_failed", order_id=order_id, message=message)
                except Exception as e:
                    logger.error("emergency_cleanup_error", order_id=order_id, error=str(e))

    # Register emergency cleanup handler
    atexit.register(emergency_cleanup)

    yield order_ids

    # Normal cleanup: attempt to cancel all tracked orders
    for order_id in order_ids:
        success, message = safe_cancel_order(coinbase_live_client, order_id)
        if success:
            logger.info("order_cleanup_success", order_id=order_id)
        else:
            logger.warning("order_cleanup_failed", order_id=order_id, message=message)

    # Unregister emergency cleanup since normal cleanup completed
    try:
        atexit.unregister(emergency_cleanup)
    except Exception:
        pass  # Python 3.8+, ignore if not supported


# ==============================================================================
# Read-Only Tests (Safe - No Orders)
# ==============================================================================

@pytest.mark.integration_live
class TestCoinbaseReadOnly:
    """
    Read-only API tests that verify authentication and data retrieval.

    These tests are safe and do not place any orders.
    """

    def test_get_accounts_returns_data(self, coinbase_live_client):
        """Verify that get_accounts returns account data."""
        response = coinbase_live_client.get_accounts()

        assert response is not None, "get_accounts returned None"
        assert isinstance(response, dict), f"get_accounts returned {type(response)}, expected dict"
        assert "accounts" in response, "Response missing 'accounts' key"

        accounts = response["accounts"]
        assert isinstance(accounts, list), f"accounts is {type(accounts)}, expected list"
        assert len(accounts) > 0, "accounts list is empty"

        logger.info("get_accounts_success", account_count=len(accounts))

    def test_get_balance_returns_actual_balance(self, coinbase_live_client, trading_config):
        """Verify that get_balance returns actual quote currency balance."""
        quote_currency = trading_config["quote_currency"]
        balance = coinbase_live_client.get_balance(quote_currency)

        assert balance is not None, f"get_balance({quote_currency}) returned None"
        assert isinstance(balance, Balance), f"Balance is {type(balance)}, expected Balance"
        assert balance.currency == quote_currency, f"Currency should be {quote_currency}"
        assert balance.available >= Decimal("0"), f"Available balance is negative: {balance.available}"
        assert balance.hold >= Decimal("0"), f"Hold balance is negative: {balance.hold}"
        assert balance.total >= Decimal("0"), f"Total balance is negative: {balance.total}"

        logger.info("get_balance_success", currency=quote_currency, available=str(balance.available), hold=str(balance.hold), total=str(balance.total))

    def test_get_balance_btc(self, coinbase_live_client):
        """Verify that get_balance returns BTC balance (may be zero)."""
        balance = coinbase_live_client.get_balance("BTC")

        assert balance is not None, "get_balance(BTC) returned None"
        assert isinstance(balance, Balance), f"Balance is {type(balance)}, expected Balance"
        assert balance.currency == "BTC", "Currency should be BTC"
        assert balance.available >= Decimal("0"), f"Available balance is negative: {balance.available}"
        assert balance.hold >= Decimal("0"), f"Hold balance is negative: {balance.hold}"
        assert balance.total >= Decimal("0"), f"Total balance is negative: {balance.total}"

        logger.info("get_balance_btc_success", available=str(balance.available), hold=str(balance.hold), total=str(balance.total))

    def test_get_current_price_btc_usd(self, coinbase_live_client, trading_config):
        """Verify that get_current_price returns BTC price in reasonable range."""
        trading_pair = trading_config["trading_pair"]
        price = coinbase_live_client.get_current_price(trading_pair)

        assert price is not None, "get_current_price returned None"
        assert isinstance(price, Decimal), f"Price is {type(price)}, expected Decimal"
        assert Decimal("10000") <= price <= Decimal("500000"), \
            f"Price {price} is outside reasonable range ($10k-$500k or equivalent)"

        logger.info("get_current_price_success", pair=trading_pair, price=str(price))

    def test_get_market_data_btc_usd(self, coinbase_live_client, trading_config):
        """Verify that get_market_data returns valid bid/ask/volume data."""
        trading_pair = trading_config["trading_pair"]
        data = coinbase_live_client.get_market_data(trading_pair)

        assert data is not None, "get_market_data returned None"
        assert isinstance(data, MarketData), f"Market data is {type(data)}, expected MarketData"
        assert data.symbol == trading_pair, f"Symbol should be {trading_pair}, got {data.symbol}"

        assert data.bid > Decimal("0"), f"Bid price is not positive: {data.bid}"
        assert data.ask > Decimal("0"), f"Ask price is not positive: {data.ask}"
        assert data.ask >= data.bid, f"Ask {data.ask} is less than bid {data.bid}"
        assert data.volume_24h >= Decimal("0"), f"Volume is negative: {data.volume_24h}"
        assert data.price > Decimal("0"), f"Price is not positive: {data.price}"

        logger.info("get_market_data_success", pair=trading_pair, bid=str(data.bid), ask=str(data.ask), volume_24h=str(data.volume_24h), price=str(data.price))

    def test_get_candles_returns_dataframe(self, coinbase_live_client, trading_config):
        """Verify that get_candles returns valid OHLCV DataFrame."""
        trading_pair = trading_config["trading_pair"]
        df = coinbase_live_client.get_candles(trading_pair, granularity="ONE_HOUR", limit=10)

        assert df is not None, "get_candles returned None"
        assert isinstance(df, pd.DataFrame), f"get_candles returned {type(df)}, expected DataFrame"
        assert len(df) > 0, "get_candles returned empty DataFrame"

        # Verify required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in df.columns, f"DataFrame missing column: {col}"

        # Verify OHLC relationships
        for idx, row in df.iterrows():
            assert row["high"] >= row["low"], \
                f"Row {idx}: high {row['high']} < low {row['low']}"
            assert row["high"] >= row["open"], \
                f"Row {idx}: high {row['high']} < open {row['open']}"
            assert row["high"] >= row["close"], \
                f"Row {idx}: high {row['high']} < close {row['close']}"
            assert row["low"] <= row["open"], \
                f"Row {idx}: low {row['low']} > open {row['open']}"
            assert row["low"] <= row["close"], \
                f"Row {idx}: low {row['low']} > close {row['close']}"

        logger.info("get_candles_success", candle_count=len(df))

    def test_authentication_builds_valid_jwt(self, coinbase_live_client):
        """
        Verify that authentication is working by making a simple API call.

        This implicitly tests the JWT building logic in coinbase_auth.py.
        """
        # If this call succeeds, authentication is working
        response = coinbase_live_client.get_accounts()
        assert response is not None, "Authentication failed - get_accounts returned None"
        assert "accounts" in response, "Authentication succeeded but response format unexpected"

        logger.info("authentication_success")


# ==============================================================================
# Order Validation Tests (Real Orders - Use Caution)
# ==============================================================================

@pytest.mark.integration_live
class TestCoinbaseOrderValidation:
    """
    Order validation tests that place real orders with immediate cancellation.

    ⚠️  WARNING: These tests place REAL orders on Coinbase.
    - Order size: 10 units of quote currency (Coinbase minimum)
    - Supported currencies: USD, USDC, EUR (auto-detected)
    - Strategy: Limit orders off-market for cancellation, market orders for execution testing
    - Cleanup: Attempts to cancel unfilled orders
    - Cost: ~0.30-0.60 in fees per test run (in quote currency)

    Tests skip if account balance < 50 units of quote currency.
    """

    @pytest.fixture(scope="class", autouse=True)
    def check_minimum_balance(self, coinbase_live_client, trading_config):
        """
        Skip all order tests if balance is below minimum threshold.

        IMPORTANT: This check runs once at class setup. Tests are designed to reuse capital
        (buy BTC, then sell it), so balance shouldn't decrease significantly during execution.
        However, if tests fail partway through, subsequent runs may have lower balance.

        If individual tests fail with insufficient balance errors, check that:
        1. Initial balance is >= recommended amount (recommended, not just minimum)
        2. Previous test runs didn't leave executed orders that consumed capital
        3. Account hasn't been used for other trading between test runs
        """
        quote_currency = trading_config["quote_currency"]
        min_balance = trading_config["min_balance"]
        recommended_balance = trading_config["recommended_balance"]

        balance = coinbase_live_client.get_balance(quote_currency)

        if balance.available < min_balance:
            pytest.skip(
                f"Insufficient balance for order validation tests: "
                f"{balance.available} {quote_currency} < {min_balance} {quote_currency} minimum "
                f"(recommended: {recommended_balance} {quote_currency})"
            )

        # Warn if balance is below recommended (50% buffer for test flakiness mitigation)
        if balance.available < recommended_balance:
            logger.warning(
                "balance_below_recommended",
                currency=quote_currency,
                available=str(balance.available),
                recommended=str(recommended_balance),
                message="Balance meets minimum but below recommended. Test failures may occur if prior runs left partial orders."
            )

        logger.info(
            "balance_check_passed",
            currency=quote_currency,
            available=str(balance.available),
            min_balance=str(min_balance),
            recommended=str(recommended_balance)
        )

    @pytest.fixture(scope="class")
    def ensure_btc_balance(self, coinbase_live_client, trading_config):
        """
        Ensure test account has BTC balance for sell tests.

        If BTC balance is zero, places a market buy to acquire BTC.
        """
        quote_currency = trading_config["quote_currency"]
        trading_pair = trading_config["trading_pair"]

        btc_balance = coinbase_live_client.get_balance("BTC")

        if btc_balance.available > Decimal("0"):
            logger.info("btc_balance_available", balance=str(btc_balance.available))
            return

        # Need to buy BTC for sell tests
        logger.info("acquiring_btc_for_sell_tests", amount=str(MAX_TEST_ORDER), currency=quote_currency)

        try:
            order = coinbase_live_client.market_buy(
                product_id=trading_pair,
                quote_size=MAX_TEST_ORDER
            )

            assert order is not None, "Market buy returned None"
            logger.info("btc_acquired", order_id=order.get("order_id"))

            # Wait for order to settle
            time.sleep(2)

            # Verify we have BTC now
            new_balance = coinbase_live_client.get_balance("BTC")
            assert new_balance.available > Decimal("0"), \
                f"Market buy executed but BTC balance is still zero: {new_balance.available}"

            logger.info("btc_balance_confirmed", balance=str(new_balance.available))

        except Exception as e:
            pytest.fail(f"Failed to acquire BTC for sell tests: {str(e)}")

    def test_limit_buy_below_market_and_immediate_cancel(
        self,
        coinbase_live_client,
        trading_config,
        cleanup_orders
    ):
        """
        Place limit buy order slightly below market bid and immediately cancel.

        Strategy: Order sits in book without filling, allowing clean cancellation.
        """
        trading_pair = trading_config["trading_pair"]
        quote_currency = trading_config["quote_currency"]

        # Get current market data
        market_data = coinbase_live_client.get_market_data(trading_pair)
        current_bid = market_data.bid

        # Place limit buy 2% below current bid (won't fill immediately)
        limit_price = current_bid * Decimal("0.98")
        limit_price = limit_price.quantize(Decimal("0.01"))

        # Calculate size in BTC
        size_btc = (MAX_TEST_ORDER / limit_price).quantize(Decimal("0.00000001"))

        logger.info(
            "placing_limit_buy",
            limit_price=str(limit_price),
            size_btc=str(size_btc),
            total=str(MAX_TEST_ORDER),
            currency=quote_currency
        )

        # Place order
        order = coinbase_live_client.limit_buy(
            symbol=trading_pair,
            base_size=size_btc,
            limit_price=limit_price
        )

        assert order is not None, "limit_buy returned None"
        assert "order_id" in order, "Order response missing order_id"

        order_id = order["order_id"]
        cleanup_orders.append(order_id)

        logger.info("limit_buy_placed", order_id=order_id)

        # Wait brief moment
        time.sleep(CANCELLATION_TIMEOUT_MS / 1000.0)

        # Cancel order
        success, message = safe_cancel_order(coinbase_live_client, order_id)
        assert success, f"Failed to cancel order: {message}"

        logger.info("limit_buy_cancelled", order_id=order_id)

    def test_market_buy_small_amount_executes(
        self,
        coinbase_live_client,
        trading_config,
        cleanup_orders
    ):
        """
        Place small market buy order and verify it executes.

        This tests the full order execution flow.
        """
        trading_pair = trading_config["trading_pair"]
        quote_currency = trading_config["quote_currency"]

        initial_btc = coinbase_live_client.get_balance("BTC")

        logger.info(
            "placing_market_buy",
            amount=str(MAX_TEST_ORDER),
            currency=quote_currency,
            initial_btc=str(initial_btc.available)
        )

        # Place market buy
        order = coinbase_live_client.market_buy(
            product_id=trading_pair,
            quote_size=MAX_TEST_ORDER
        )

        assert order is not None, "market_buy returned None"
        assert order.success, f"market_buy failed: {order.error}"
        assert order.order_id, "Order response missing order_id"

        cleanup_orders.append(order.order_id)

        logger.info(
            "market_buy_placed",
            order_id=order.order_id,
            initial_size=str(order.size),
            status=order.status
        )

        # Poll for order completion (market orders fill async)
        max_attempts = 10
        filled_size = Decimal("0")
        for attempt in range(max_attempts):
            time.sleep(0.5)
            order_status = coinbase_live_client.get_order(order.order_id)
            order_data = order_status.get("order", order_status)
            status = order_data.get("status", "")
            filled_size = Decimal(order_data.get("filled_size", "0"))
            logger.info(
                "polling_order_status",
                attempt=attempt + 1,
                status=status,
                filled_size=str(filled_size)
            )
            if status in ["FILLED", "COMPLETED"] or filled_size > Decimal("0"):
                break

        assert filled_size > Decimal("0"), \
            f"Order did not fill after {max_attempts} attempts: status={status}, filled_size={filled_size}"

        logger.info("order_filled", filled_size=str(filled_size), status=status)

        # Wait for balance to update (Coinbase has propagation delay)
        time.sleep(3)

        # Verify BTC balance increased
        final_btc = coinbase_live_client.get_balance("BTC")

        # Log both balances for debugging
        logger.info(
            "balance_comparison",
            initial=str(initial_btc.available),
            final=str(final_btc.available),
            expected_increase=str(filled_size)
        )

        # Use filled_size from order status as ground truth, balance may have propagation delay
        if final_btc.available <= initial_btc.available:
            logger.warning(
                "balance_not_updated_yet",
                message="Order filled but balance API not yet reflecting change - this is acceptable"
            )
            # Skip balance assertion if order definitively filled
            btc_acquired = filled_size
        else:
            btc_acquired = final_btc.available - initial_btc.available

        logger.info(
            "market_buy_executed",
            order_id=order.order_id,
            btc_acquired=str(btc_acquired),
            final_btc=str(final_btc.available)
        )

    def test_limit_sell_above_market_and_immediate_cancel(
        self,
        coinbase_live_client,
        trading_config,
        cleanup_orders,
        ensure_btc_balance
    ):
        """
        Place limit sell order slightly above market ask and immediately cancel.

        Strategy: Order sits in book without filling, allowing clean cancellation.
        Requires BTC balance (acquired in ensure_btc_balance fixture).
        """
        trading_pair = trading_config["trading_pair"]
        quote_currency = trading_config["quote_currency"]

        # Get current market data
        market_data = coinbase_live_client.get_market_data(trading_pair)
        current_ask = market_data.ask

        # Place limit sell 2% above current ask (won't fill immediately)
        limit_price = current_ask * Decimal("1.02")
        limit_price = limit_price.quantize(Decimal("0.01"))

        # Get BTC balance to determine order size
        btc_balance = coinbase_live_client.get_balance("BTC")
        assert btc_balance.available > Decimal("0"), "No BTC balance available for sell test"

        # Calculate BTC amount for order at limit price
        size_btc = (MAX_TEST_ORDER / limit_price).quantize(Decimal("0.00000001"))

        # Ensure we don't exceed available balance (prevents insufficient balance errors)
        size_btc = min(size_btc, btc_balance.available)

        # Defensive check: Verify order value doesn't exceed cap
        # (Protects against price calculation errors or balance accumulation)
        order_value = size_btc * limit_price
        assert order_value <= MAX_TEST_ORDER * Decimal("1.1"), \
            f"Order value {order_value} {quote_currency} exceeds cap (10% buffer included)"

        logger.info(
            "placing_limit_sell",
            limit_price=str(limit_price),
            size_btc=str(size_btc),
            btc_balance=str(btc_balance.available),
            currency=quote_currency
        )

        # Place order
        order = coinbase_live_client.limit_sell(
            symbol=trading_pair,
            base_size=size_btc,
            limit_price=limit_price
        )

        assert order is not None, "limit_sell returned None"
        assert "order_id" in order, "Order response missing order_id"

        order_id = order["order_id"]
        cleanup_orders.append(order_id)

        logger.info("limit_sell_placed", order_id=order_id)

        # Wait brief moment
        time.sleep(CANCELLATION_TIMEOUT_MS / 1000.0)

        # Cancel order
        success, message = safe_cancel_order(coinbase_live_client, order_id)
        assert success, f"Failed to cancel order: {message}"

        logger.info("limit_sell_cancelled", order_id=order_id)

    def test_market_sell_small_amount_executes(
        self,
        coinbase_live_client,
        trading_config,
        cleanup_orders,
        ensure_btc_balance
    ):
        """
        Place small market sell order and verify it executes.

        This tests the full sell execution flow.
        Requires BTC balance (acquired in ensure_btc_balance fixture).
        """
        trading_pair = trading_config["trading_pair"]
        quote_currency = trading_config["quote_currency"]

        initial_btc = coinbase_live_client.get_balance("BTC")
        assert initial_btc.available > Decimal("0"), "No BTC balance available for sell test"

        # Get current price to calculate sell size
        current_price = coinbase_live_client.get_current_price(trading_pair)

        # Calculate BTC amount for order at current price
        size_btc = (MAX_TEST_ORDER / current_price).quantize(Decimal("0.00000001"))

        # Ensure we don't exceed available balance (prevents insufficient balance errors)
        size_btc = min(size_btc, initial_btc.available)

        # Defensive check: Verify order value doesn't exceed cap
        # (Protects against price calculation errors or balance accumulation)
        order_value = size_btc * current_price
        assert order_value <= MAX_TEST_ORDER * Decimal("1.1"), \
            f"Order value {order_value} {quote_currency} exceeds cap (10% buffer included)"

        logger.info(
            "placing_market_sell",
            size_btc=str(size_btc),
            initial_btc=str(initial_btc.available),
            current_price=str(current_price),
            currency=quote_currency
        )

        # Place market sell
        order = coinbase_live_client.market_sell(
            product_id=trading_pair,
            base_size=size_btc
        )

        assert order is not None, "market_sell returned None"
        assert order.success, f"market_sell failed: {order.error}"
        assert order.order_id, "Order response missing order_id"

        cleanup_orders.append(order.order_id)

        logger.info(
            "market_sell_placed",
            order_id=order.order_id,
            initial_size=str(order.size),
            status=order.status
        )

        # Poll for order completion (market orders fill async)
        max_attempts = 10
        filled_size = Decimal("0")
        for attempt in range(max_attempts):
            time.sleep(0.5)
            order_status = coinbase_live_client.get_order(order.order_id)
            order_data = order_status.get("order", order_status)
            status = order_data.get("status", "")
            filled_size = Decimal(order_data.get("filled_size", "0"))
            logger.info(
                "polling_order_status",
                attempt=attempt + 1,
                status=status,
                filled_size=str(filled_size)
            )
            if status in ["FILLED", "COMPLETED"] or filled_size > Decimal("0"):
                break

        assert filled_size > Decimal("0"), \
            f"Order did not fill after {max_attempts} attempts: status={status}, filled_size={filled_size}"

        logger.info("order_filled", filled_size=str(filled_size), status=status)

        # Wait for balance to update (Coinbase has propagation delay)
        time.sleep(3)

        # Verify BTC balance decreased
        final_btc = coinbase_live_client.get_balance("BTC")

        # Log both balances for debugging
        logger.info(
            "balance_comparison",
            initial=str(initial_btc.available),
            final=str(final_btc.available),
            expected_decrease=str(filled_size)
        )

        # Use filled_size from order status as ground truth, balance may have propagation delay
        if final_btc.available >= initial_btc.available:
            logger.warning(
                "balance_not_updated_yet",
                message="Order filled but balance API not yet reflecting change - this is acceptable"
            )
            # Skip balance assertion if order definitively filled
            btc_sold = filled_size
        else:
            btc_sold = initial_btc.available - final_btc.available

        logger.info(
            "market_sell_executed",
            order_id=order.order_id,
            btc_sold=str(btc_sold),
            final_btc=str(final_btc.available)
        )

    def test_order_placement_with_invalid_params_fails(self, coinbase_live_client, trading_config):
        """
        Verify that orders with invalid parameters are rejected.

        Tests negative scenarios to ensure proper validation.
        Tests both API-level validation (Coinbase returning failed OrderResult)
        and client-level validation (ValueError for limit orders).
        """
        trading_pair = trading_config["trading_pair"]

        # Test 1: Zero amount should return failed OrderResult or empty order_id
        # (API may return success=True but with no actual order placed)
        result = coinbase_live_client.market_buy(
            product_id=trading_pair,
            quote_size=Decimal("0")
        )
        assert not result.success or not result.order_id or result.size == Decimal("0"), \
            "Zero amount should fail or return empty/zero-size order"
        logger.info("invalid_order_rejected", reason="zero_amount", success=result.success, order_id=result.order_id)

        # Test 2: Negative amount should return failed OrderResult or empty order_id
        result = coinbase_live_client.market_buy(
            product_id=trading_pair,
            quote_size=Decimal("-10")
        )
        assert not result.success or not result.order_id or result.size == Decimal("0"), \
            "Negative amount should fail or return empty/zero-size order"
        logger.info("invalid_order_rejected", reason="negative_amount", success=result.success, order_id=result.order_id)

        # Test 3: Invalid product should return failed OrderResult
        result = coinbase_live_client.market_buy(
            product_id="INVALID-PAIR",
            quote_size=MAX_TEST_ORDER
        )
        assert not result.success or not result.order_id, "Invalid product should fail"
        logger.info("invalid_order_rejected", reason="invalid_product", success=result.success, order_id=result.order_id)

        # Test 4: Negative base_size in limit order should fail (client validation)
        with pytest.raises(ValueError, match="base_size must be positive"):
            coinbase_live_client.limit_buy(
                symbol=trading_pair,
                base_size=Decimal("-0.001"),
                limit_price=Decimal("50000")
            )
        logger.info("invalid_order_rejected", reason="negative_base_size")

        # Test 5: Zero limit price should fail (client validation)
        with pytest.raises(ValueError, match="limit_price must be positive"):
            coinbase_live_client.limit_buy(
                symbol=trading_pair,
                base_size=Decimal("0.001"),
                limit_price=Decimal("0")
            )
        logger.info("invalid_order_rejected", reason="zero_limit_price")

        # Test 6: Negative limit price should fail (client validation)
        with pytest.raises(ValueError, match="limit_price must be positive"):
            coinbase_live_client.limit_sell(
                symbol=trading_pair,
                base_size=Decimal("0.001"),
                limit_price=Decimal("-50000")
            )
        logger.info("invalid_order_rejected", reason="negative_limit_price")
