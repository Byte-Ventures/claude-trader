"""
Kraken API client implementation.

Implements the ExchangeClient protocol for Kraken exchange.
Uses HMAC-SHA512 authentication for private endpoints.
"""

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
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


class KrakenClient:
    """
    Kraken API client implementing the ExchangeClient protocol.

    Features:
    - HMAC-SHA512 authentication for private endpoints
    - Automatic symbol translation (BTC-USD -> XBT/USD)
    - Automatic retry with exponential backoff
    - Rate limiting awareness (15 calls/minute for private endpoints)
    """

    RETRY_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
    )

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
        self._last_request_time = 0
        self._min_request_interval = 1.0  # 1 second between requests for rate limiting

        logger.info("kraken_client_initialized")

    def _rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

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
        self._rate_limit()

        url = f"{BASE_URL}/0/public/{endpoint}"

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            error_msg = ", ".join(data["error"])
            logger.error("kraken_api_error", endpoint=endpoint, error=error_msg)
            raise Exception(f"Kraken API error: {error_msg}")

        return data.get("result", {})

    def _private_request(
        self,
        endpoint: str,
        data: Optional[dict] = None,
    ) -> dict:
        """
        Make a private (authenticated) request.

        Args:
            endpoint: API endpoint (e.g., Balance, AddOrder)
            data: POST data

        Returns:
            Response data
        """
        self._rate_limit()

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

        response = self.session.post(url, data=data, headers=headers)
        response.raise_for_status()

        result = response.json()

        if result.get("error"):
            error_msg = ", ".join(result["error"])
            logger.error("kraken_api_error", endpoint=endpoint, error=error_msg)
            raise Exception(f"Kraken API error: {error_msg}")

        return result.get("result", {})

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
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
