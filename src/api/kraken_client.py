"""
Kraken API client implementation.

Implements the ExchangeClient protocol for Kraken exchange.
Uses HMAC-SHA512 authentication for private endpoints.

Rate Limiting:
- Kraken uses a counter-based system for private endpoints
- Counter starts at 0, max is 15-20 depending on account tier
- Each API call adds 1-2 points to counter
- Counter decays by 1 every 3 seconds
- We implement conservative rate limiting to avoid hitting limits
"""

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from threading import Lock
from typing import Optional

import pandas as pd
import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.api.exchange_protocol import Balance, MarketData, OrderResult
from src.api.kraken_auth import get_auth_headers, get_nonce, validate_credentials
from src.api.symbol_mapper import (
    from_kraken_asset,
    to_exchange_symbol,
    to_kraken_asset,
    to_kraken_granularity,
    Exchange,
)

logger = structlog.get_logger(__name__)

BASE_URL = "https://api.kraken.com"


def log_retry(retry_state) -> None:
    """Log retry attempts for debugging."""
    logger.warning(
        "kraken_api_retry",
        attempt=retry_state.attempt_number,
        wait=f"{retry_state.next_action.sleep:.1f}s",
        error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
    )


class KrakenRateLimitError(Exception):
    """Raised when Kraken rate limit is hit."""
    pass


class KrakenClient:
    """
    Kraken API client implementing the ExchangeClient protocol.

    Features:
    - HMAC-SHA512 authentication for private endpoints
    - Automatic symbol translation (BTC-USD -> XBT/USD)
    - Automatic retry with exponential backoff
    - Counter-based rate limiting for private endpoints
    - Automatic backoff on rate limit errors
    """

    RETRY_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        KrakenRateLimitError,
    )

    # Kraken rate limit constants
    MAX_COUNTER = 15  # Conservative limit (actual is 15-20)
    COUNTER_DECAY_RATE = 0.33  # Points per second (1 every 3 seconds)
    PRIVATE_CALL_COST = 1  # Cost per private API call

    def __init__(
        self,
        api_key: str,
        api_secret: str,
    ):
        """
        Initialize Kraken client.

        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret (base64-encoded)
        """
        validate_credentials(api_key, api_secret)

        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

        # Public endpoint rate limiting (simple interval)
        self._last_public_request = 0.0
        self._public_interval = 0.5  # 500ms between public requests

        # Private endpoint rate limiting (counter-based)
        self._counter = 0.0
        self._last_counter_update = time.time()
        self._rate_limit_backoff_until = 0.0

        # Fee rate caching (1 hour TTL)
        self._fee_rate_cache: Optional[Decimal] = None
        self._fee_rate_cache_time: Optional[datetime] = None
        self._fee_rate_cache_ttl: int = 3600  # 1 hour in seconds
        self._fee_rate_cache_lock = Lock()  # Thread safety for cache operations

        logger.info("kraken_client_initialized")

    def _update_counter(self) -> None:
        """Update the rate limit counter based on decay."""
        now = time.time()
        elapsed = now - self._last_counter_update
        decay = elapsed * self.COUNTER_DECAY_RATE
        self._counter = max(0, self._counter - decay)
        self._last_counter_update = now

    def _wait_for_counter_capacity(self, cost: int = 1) -> None:
        """Wait until there's capacity in the rate limit counter."""
        self._update_counter()

        # Check if we're in backoff period from a rate limit error
        now = time.time()
        if now < self._rate_limit_backoff_until:
            wait_time = self._rate_limit_backoff_until - now
            logger.debug("kraken_rate_limit_backoff", wait_seconds=f"{wait_time:.1f}")
            time.sleep(wait_time)
            self._counter = 0  # Reset counter after backoff

        # Wait if counter would exceed limit
        if self._counter + cost > self.MAX_COUNTER:
            wait_needed = (self._counter + cost - self.MAX_COUNTER) / self.COUNTER_DECAY_RATE
            logger.debug("kraken_counter_wait", counter=f"{self._counter:.1f}", wait_seconds=f"{wait_needed:.1f}")
            time.sleep(wait_needed)
            self._update_counter()

        self._counter += cost

    def _rate_limit_public(self) -> None:
        """Rate limit for public endpoints (simple interval)."""
        elapsed = time.time() - self._last_public_request
        if elapsed < self._public_interval:
            time.sleep(self._public_interval - elapsed)
        self._last_public_request = time.time()

    def _handle_rate_limit_error(self) -> None:
        """Handle a rate limit error by setting backoff."""
        self._rate_limit_backoff_until = time.time() + 30  # 30 second backoff
        logger.warning("kraken_rate_limit_hit", backoff_seconds=30)

    def _public_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Make a public (unauthenticated) request.

        Args:
            endpoint: API endpoint (e.g., Ticker, OHLC)
            params: Query parameters

        Returns:
            Response data
        """
        self._rate_limit_public()

        url = f"{BASE_URL}/0/public/{endpoint}"

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            error_msg = ", ".join(data["error"])
            # Check for rate limit error
            if "EAPI:Rate limit exceeded" in error_msg or "EGeneral:Too many requests" in error_msg:
                self._handle_rate_limit_error()
                raise KrakenRateLimitError(error_msg)
            logger.error("kraken_api_error", endpoint=endpoint, error=error_msg)
            raise Exception(f"Kraken API error: {error_msg}")

        return data.get("result", {})

    def _private_request(
        self,
        endpoint: str,
        data: Optional[dict] = None,
        cost: int = 1,
    ) -> dict:
        """
        Make a private (authenticated) request.

        Args:
            endpoint: API endpoint (e.g., Balance, AddOrder)
            data: POST data
            cost: Rate limit counter cost (default 1, some endpoints cost 2)

        Returns:
            Response data
        """
        # Wait for rate limit capacity before making request
        self._wait_for_counter_capacity(cost)

        url_path = f"/0/private/{endpoint}"
        url = f"{BASE_URL}{url_path}"

        if data is None:
            data = {}

        nonce = get_nonce()
        data["nonce"] = nonce

        headers = get_auth_headers(
            self.api_key,
            self.api_secret,
            url_path,
            nonce,
            data,
        )
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        response = self.session.post(url, data=data, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()

        if result.get("error"):
            error_msg = ", ".join(result["error"])
            # Check for rate limit error
            if "EAPI:Rate limit exceeded" in error_msg or "EGeneral:Too many requests" in error_msg:
                self._handle_rate_limit_error()
                raise KrakenRateLimitError(error_msg)
            logger.error("kraken_api_error", endpoint=endpoint, error=error_msg)
            raise Exception(f"Kraken API error: {error_msg}")

        return result.get("result", {})

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def get_balance(self, currency: str = "BTC") -> Balance:
        """
        Get account balance for a currency.

        Args:
            currency: Currency code (BTC, USD, etc.)

        Returns:
            Balance information
        """
        result = self._private_request("Balance")

        # Convert currency to Kraken asset code
        kraken_asset = to_kraken_asset(currency)

        # Try different possible asset codes
        possible_keys = [
            kraken_asset,
            currency.upper(),
            f"X{currency.upper()}",
            f"Z{currency.upper()}",
        ]

        for key in possible_keys:
            if key in result:
                return Balance(
                    currency=currency,
                    available=Decimal(result[key]),
                    hold=Decimal("0"),  # Kraken doesn't provide hold separately in Balance
                )

        # Return zero balance if not found
        logger.warning("balance_not_found", currency=currency, available_keys=list(result.keys()))
        return Balance(currency=currency, available=Decimal("0"), hold=Decimal("0"))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def get_current_price(self, product_id: str = "BTC-USD") -> Decimal:
        """
        Get current market price.

        Args:
            product_id: Trading pair in normalized format (e.g., BTC-USD)

        Returns:
            Current price
        """
        kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
        # Remove slash for API call
        pair_code = kraken_pair.replace("/", "")

        result = self._public_request("Ticker", {"pair": pair_code})

        # Get first result (there should only be one)
        for pair_data in result.values():
            # 'c' is last trade closed [price, lot volume]
            return Decimal(pair_data["c"][0])

        raise Exception(f"No ticker data for {product_id}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def get_market_data(self, product_id: str = "BTC-USD") -> MarketData:
        """
        Get current market data including bid/ask.

        Args:
            product_id: Trading pair in normalized format

        Returns:
            MarketData with current prices
        """
        kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
        pair_code = kraken_pair.replace("/", "")

        result = self._public_request("Ticker", {"pair": pair_code})

        for pair_data in result.values():
            return MarketData(
                symbol=product_id,
                price=Decimal(pair_data["c"][0]),  # Last trade price
                bid=Decimal(pair_data["b"][0]),  # Best bid
                ask=Decimal(pair_data["a"][0]),  # Best ask
                volume_24h=Decimal(pair_data["v"][1]),  # 24h volume
                timestamp=datetime.now(timezone.utc),
            )

        raise Exception(f"No market data for {product_id}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def get_candles(
        self,
        product_id: str = "BTC-USD",
        granularity: str = "ONE_HOUR",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV candle data.

        Args:
            product_id: Trading pair in normalized format
            granularity: Candle size (ONE_MINUTE, ONE_HOUR, ONE_DAY, etc.)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
        pair_code = kraken_pair.replace("/", "")

        interval = to_kraken_granularity(granularity)

        # Calculate 'since' timestamp to get approximately 'limit' candles
        seconds_per_candle = interval * 60
        since = int(time.time()) - (limit * seconds_per_candle)

        result = self._public_request("OHLC", {
            "pair": pair_code,
            "interval": interval,
            "since": since,
        })

        # Get candle data (exclude 'last' key which is the last ID)
        candles = []
        for key, value in result.items():
            if key != "last" and isinstance(value, list):
                candles = value
                break

        # Convert to DataFrame
        # Kraken format: [time, open, high, low, close, vwap, volume, count]
        data = []
        for candle in candles[-limit:]:  # Take last 'limit' candles
            data.append({
                "timestamp": datetime.fromtimestamp(candle[0], tz=timezone.utc),
                "open": Decimal(candle[1]),
                "high": Decimal(candle[2]),
                "low": Decimal(candle[3]),
                "close": Decimal(candle[4]),
                "volume": Decimal(candle[6]),
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def market_buy(
        self,
        product_id: str,
        quote_size: Decimal,
    ) -> OrderResult:
        """
        Execute a market buy order.

        Note: Kraken market orders require base size, so we calculate
        approximate BTC amount from USD quote_size.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            quote_size: Amount to spend in quote currency (USD)

        Returns:
            OrderResult with execution details
        """
        try:
            kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
            pair_code = kraken_pair.replace("/", "")

            # Get current price to calculate base size
            current_price = self.get_current_price(product_id)

            # Calculate approximate base size (add small buffer for price movement)
            base_size = quote_size / (current_price * Decimal("1.005"))

            # Round to 8 decimal places (standard for BTC)
            base_size = base_size.quantize(Decimal("0.00000001"))

            order_data = {
                "pair": pair_code,
                "type": "buy",
                "ordertype": "market",
                "volume": str(base_size),
            }

            result = self._private_request("AddOrder", order_data)

            order_ids = result.get("txid", [])
            order_id = order_ids[0] if order_ids else ""

            logger.info("kraken_market_buy", order_id=order_id, quote_size=str(quote_size))

            # Get order details to find fill price
            if order_id:
                order_info = self._get_order_info(order_id)
                filled_price = Decimal(order_info.get("price", "0")) if order_info.get("price") else None
                filled_size = Decimal(order_info.get("vol_exec", "0"))
                fee = Decimal(order_info.get("fee", "0"))
                status = order_info.get("status", "pending")
            else:
                filled_price = current_price
                filled_size = base_size
                fee = Decimal("0")
                status = "submitted"

            return OrderResult(
                order_id=order_id,
                side="buy",
                size=filled_size,
                filled_price=filled_price,
                status=status,
                fee=fee,
                success=True,
            )

        except Exception as e:
            logger.error("kraken_market_buy_failed", error=str(e), product_id=product_id, quote_size=str(quote_size))
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=str(e),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        before_sleep=log_retry,
    )
    def market_sell(
        self,
        product_id: str,
        base_size: Decimal,
    ) -> OrderResult:
        """
        Execute a market sell order.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount to sell in base currency (BTC)

        Returns:
            OrderResult with execution details
        """
        try:
            kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
            pair_code = kraken_pair.replace("/", "")

            order_data = {
                "pair": pair_code,
                "type": "sell",
                "ordertype": "market",
                "volume": str(base_size),
            }

            result = self._private_request("AddOrder", order_data)

            order_ids = result.get("txid", [])
            order_id = order_ids[0] if order_ids else ""

            logger.info("kraken_market_sell", order_id=order_id, base_size=str(base_size))

            # Get order details
            if order_id:
                order_info = self._get_order_info(order_id)
                filled_price = Decimal(order_info.get("price", "0")) if order_info.get("price") else None
                filled_size = Decimal(order_info.get("vol_exec", "0"))
                fee = Decimal(order_info.get("fee", "0"))
                status = order_info.get("status", "pending")
            else:
                filled_price = self.get_current_price(product_id)
                filled_size = base_size
                fee = Decimal("0")
                status = "submitted"

            return OrderResult(
                order_id=order_id,
                side="sell",
                size=filled_size,
                filled_price=filled_price,
                status=status,
                fee=fee,
                success=True,
            )

        except Exception as e:
            logger.error("kraken_market_sell_failed", error=str(e), product_id=product_id, base_size=str(base_size))
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=str(e),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def limit_buy_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Execute a limit buy order with IOC time-in-force.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount to buy in base currency (BTC)
            limit_price: Maximum price willing to pay

        Returns:
            OrderResult with execution details
        """
        try:
            kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
            pair_code = kraken_pair.replace("/", "")

            order_data = {
                "pair": pair_code,
                "type": "buy",
                "ordertype": "limit",
                "price": str(limit_price),
                "volume": str(base_size),
                "timeinforce": "IOC",  # Immediate-or-cancel
            }

            result = self._private_request("AddOrder", order_data)

            order_ids = result.get("txid", [])
            order_id = order_ids[0] if order_ids else ""

            logger.info(
                "kraken_limit_buy_ioc",
                order_id=order_id,
                base_size=str(base_size),
                limit_price=str(limit_price),
            )

            # Get order details
            if order_id:
                order_info = self._get_order_info(order_id)
                filled_price = Decimal(order_info.get("price", "0")) if order_info.get("price") else None
                filled_size = Decimal(order_info.get("vol_exec", "0"))
                fee = Decimal(order_info.get("fee", "0"))
                status = order_info.get("status", "pending")
            else:
                filled_price = limit_price
                filled_size = Decimal("0")
                fee = Decimal("0")
                status = "submitted"

            return OrderResult(
                order_id=order_id,
                side="buy",
                size=filled_size,
                filled_price=filled_price,
                status=status,
                fee=fee,
                success=True,
            )

        except Exception as e:
            logger.error(
                "kraken_limit_buy_ioc_failed",
                error=str(e),
                product_id=product_id,
                base_size=str(base_size),
                limit_price=str(limit_price),
            )
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=str(e),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def limit_sell_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Execute a limit sell order with IOC time-in-force.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount to sell in base currency (BTC)
            limit_price: Minimum price willing to accept

        Returns:
            OrderResult with execution details
        """
        try:
            kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)
            pair_code = kraken_pair.replace("/", "")

            order_data = {
                "pair": pair_code,
                "type": "sell",
                "ordertype": "limit",
                "price": str(limit_price),
                "volume": str(base_size),
                "timeinforce": "IOC",  # Immediate-or-cancel
            }

            result = self._private_request("AddOrder", order_data)

            order_ids = result.get("txid", [])
            order_id = order_ids[0] if order_ids else ""

            logger.info(
                "kraken_limit_sell_ioc",
                order_id=order_id,
                base_size=str(base_size),
                limit_price=str(limit_price),
            )

            # Get order details
            if order_id:
                order_info = self._get_order_info(order_id)
                filled_price = Decimal(order_info.get("price", "0")) if order_info.get("price") else None
                filled_size = Decimal(order_info.get("vol_exec", "0"))
                fee = Decimal(order_info.get("fee", "0"))
                status = order_info.get("status", "pending")
            else:
                filled_price = limit_price
                filled_size = Decimal("0")
                fee = Decimal("0")
                status = "submitted"

            return OrderResult(
                order_id=order_id,
                side="sell",
                size=filled_size,
                filled_price=filled_price,
                status=status,
                fee=fee,
                success=True,
            )

        except Exception as e:
            logger.error(
                "kraken_limit_sell_ioc_failed",
                error=str(e),
                product_id=product_id,
                base_size=str(base_size),
                limit_price=str(limit_price),
            )
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=str(e),
            )

    def _get_order_info(self, order_id: str) -> dict:
        """Get order information from Kraken."""
        try:
            result = self._private_request("QueryOrders", {"txid": order_id})
            return result.get(order_id, {})
        except Exception as e:
            logger.warning("get_order_info_failed", order_id=order_id, error=str(e))
            return {}

    def get_order(self, order_id: str) -> dict:
        """Get order details by ID."""
        result = self._private_request("QueryOrders", {"txid": order_id})
        return result.get(order_id, {})

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self._private_request("CancelOrder", {"txid": order_id})
            return True
        except Exception as e:
            logger.error("cancel_order_failed", error=str(e), order_id=order_id)
            return False

    def get_trading_fee_rate(self, product_id: str) -> Decimal:
        """
        Get current trading fee rate for a product.

        Kraken fees are volume-tiered:
        - Taker: 0.26% (default <$50K/30d), down to 0.10% (>$10M/30d)
        - Maker: 0.16% (default <$50K/30d), down to 0.00% (>$10M/30d)

        Since the bot primarily uses IOC limit orders (which typically execute as taker),
        we use the taker fee rate for conservative profit margin validation.

        Fee rates are cached for 1 hour to reduce API calls.

        Args:
            product_id: Trading pair in normalized format (e.g., BTC-USD)

        Returns:
            Current taker fee rate as decimal (e.g., Decimal("0.0026") for 0.26%)
        """
        # Check cache first (thread-safe)
        with self._fee_rate_cache_lock:
            if (self._fee_rate_cache is not None
                and self._fee_rate_cache_time is not None
                and (datetime.now(timezone.utc) - self._fee_rate_cache_time).total_seconds() < self._fee_rate_cache_ttl):
                return self._fee_rate_cache

        try:
            # Convert to Kraken pair format (e.g., BTC-USD -> XBTUSD)
            kraken_pair = to_exchange_symbol(product_id, Exchange.KRAKEN)

            # Get trade volume to determine fee tier
            response = self._private_request("TradeVolume", {"pair": kraken_pair})

            # Extract fee information from response
            if "fees" in response:
                if not isinstance(response["fees"], dict):
                    logger.warning(
                        "fees_invalid_type",
                        product_id=product_id,
                        fees_type=type(response["fees"]).__name__,
                        message="fees is not a dict, using default",
                    )
                elif kraken_pair in response["fees"]:
                    fee_info = response["fees"][kraken_pair]
                    if not isinstance(fee_info, dict):
                        logger.warning(
                            "fee_info_invalid_type",
                            product_id=product_id,
                            kraken_pair=kraken_pair,
                            fee_info_type=type(fee_info).__name__,
                            message="fee_info is not a dict, using default",
                        )
                    elif "fee" in fee_info:
                        try:
                            # IMPORTANT: Kraken returns fees as PERCENTAGE (0.26 = 0.26%)
                            # This is different from Coinbase which returns as DECIMAL (0.006 = 0.6%)
                            # We must convert percentage to decimal: 0.26% -> 0.0026
                            # Do not change this conversion logic without updating tests
                            fee_percent = Decimal(str(fee_info["fee"]))
                            # Sanity check: fee_percent is percentage (0.001 = 0.001%, 5.0 = 5.0%)
                            if Decimal("0.001") <= fee_percent <= Decimal("5.0"):
                                fee_rate = fee_percent / Decimal("100")
                                # Cache successful result (thread-safe)
                                with self._fee_rate_cache_lock:
                                    self._fee_rate_cache = fee_rate
                                    self._fee_rate_cache_time = datetime.now(timezone.utc)
                                return fee_rate
                            else:
                                logger.warning(
                                    "fee_percent_out_of_range",
                                    product_id=product_id,
                                    kraken_pair=kraken_pair,
                                    fee_percent=str(fee_percent),
                                    message="Fee percentage outside reasonable range (0.001-5.0), using default",
                                )
                        except (ValueError, TypeError, ArithmeticError) as e:
                            logger.warning(
                                "fee_conversion_failed",
                                product_id=product_id,
                                kraken_pair=kraken_pair,
                                fee_value=fee_info.get("fee"),
                                error=str(e),
                                message="Failed to convert fee to Decimal, using default",
                            )

            # Fallback to default Kraken taker fee
            logger.warning(
                "fee_tier_not_found",
                product_id=product_id,
                message="Using default taker fee rate (0.26%)",
            )
            return Decimal("0.0026")

        except Exception as e:
            # Clear cache on error to avoid returning stale data
            with self._fee_rate_cache_lock:
                self._fee_rate_cache = None
                self._fee_rate_cache_time = None
            logger.warning(
                "get_trading_fee_rate_failed",
                product_id=product_id,
                error=str(e),
                message="Using default taker fee rate (0.26%)",
            )
            return Decimal("0.0026")
