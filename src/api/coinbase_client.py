"""
Coinbase Advanced Trade API client with Ed25519 support.

Uses custom JWT authentication since the official SDK doesn't support Ed25519 yet.
Implements the ExchangeClient protocol for multi-exchange support.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.api.coinbase_auth import build_jwt, load_key_file
from src.api.exchange_protocol import Balance, MarketData, OrderResult

logger = structlog.get_logger(__name__)

BASE_URL = "https://api.coinbase.com"


class CoinbaseClient:
    """
    Coinbase Advanced Trade API client with Ed25519 support.

    Features:
    - Ed25519 JWT authentication (new CDP keys)
    - Automatic retry with exponential backoff
    - Balance and position management
    - Market order execution
    - OHLCV data fetching for indicators
    """

    RETRY_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        key_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize Coinbase client.

        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            key_file: Path to CDP API key JSON file (alternative to api_key/api_secret)
        """
        if key_file:
            api_key, api_secret = load_key_file(key_file)
            logger.info("loaded_key_from_file", path=str(key_file))

        if not api_key or not api_secret:
            raise ValueError("Must provide api_key/api_secret or valid key_file")

        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

        logger.info("coinbase_client_initialized")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """Make authenticated request to Coinbase API."""
        token = build_jwt(self.api_key, self.api_secret, method, path)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        url = f"{BASE_URL}{path}"

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
        )

        if response.status_code != 200:
            error_msg = f"API error {response.status_code}: {response.text}"
            logger.error("api_request_failed", status=response.status_code, error=response.text)
            raise Exception(error_msg)

        return response.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def get_accounts(self) -> dict:
        """Get all accounts."""
        return self._request("GET", "/api/v3/brokerage/accounts")

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
        accounts = self.get_accounts()

        for account in accounts.get("accounts", []):
            if account.get("currency") == currency:
                return Balance(
                    currency=currency,
                    available=Decimal(account.get("available_balance", {}).get("value", "0")),
                    hold=Decimal(account.get("hold", {}).get("value", "0")),
                )

        # Return zero balance if not found
        return Balance(currency=currency, available=Decimal("0"), hold=Decimal("0"))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def get_product(self, product_id: str = "BTC-USD") -> dict:
        """Get product details."""
        return self._request("GET", f"/api/v3/brokerage/products/{product_id}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def get_current_price(self, product_id: str = "BTC-USD") -> Decimal:
        """
        Get current market price.

        Args:
            product_id: Trading pair (e.g., BTC-USD)

        Returns:
            Current price
        """
        product = self.get_product(product_id)
        return Decimal(product.get("price", "0"))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def get_market_data(self, product_id: str = "BTC-USD") -> MarketData:
        """
        Get current market data including bid/ask.

        Args:
            product_id: Trading pair

        Returns:
            MarketData with current prices
        """
        product = self.get_product(product_id)

        return MarketData(
            symbol=product_id,
            price=Decimal(product.get("price", "0")),
            bid=Decimal(product.get("bid", "0")),
            ask=Decimal(product.get("ask", "0")),
            volume_24h=Decimal(product.get("volume_24h", "0")),
            timestamp=datetime.now(),
        )

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
            product_id: Trading pair
            granularity: Candle size (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                        THIRTY_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Calculate time range
        end = datetime.now(timezone.utc)

        # Map granularity to seconds for time calculation
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }

        seconds = granularity_seconds.get(granularity, 3600)
        start = end - timedelta(seconds=seconds * limit)

        params = {
            "start": str(int(start.timestamp())),
            "end": str(int(end.timestamp())),
            "granularity": granularity,
        }

        result = self._request("GET", f"/api/v3/brokerage/products/{product_id}/candles", params=params)

        # Convert to DataFrame
        data = []
        for candle in result.get("candles", []):
            data.append({
                "timestamp": datetime.fromtimestamp(int(candle["start"]), tz=timezone.utc),
                "open": Decimal(candle["open"]),
                "high": Decimal(candle["high"]),
                "low": Decimal(candle["low"]),
                "close": Decimal(candle["close"]),
                "volume": Decimal(candle["volume"]),
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

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            quote_size: Amount to spend in quote currency (USD)

        Returns:
            OrderResult with execution details
        """
        try:
            order_data = {
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {
                        "quote_size": str(quote_size),
                    }
                },
            }

            result = self._request("POST", "/api/v3/brokerage/orders", data=order_data)

            order = result.get("success_response", result)

            return OrderResult(
                order_id=order.get("order_id", ""),
                side="buy",
                size=Decimal(order.get("filled_size", "0")),
                filled_price=Decimal(order.get("average_filled_price", "0")) if order.get("average_filled_price") else None,
                status=order.get("status", "unknown"),
                fee=Decimal(order.get("total_fees", "0")),
                success=True,
            )

        except Exception as e:
            logger.error("market_buy_failed", error=str(e), product_id=product_id, quote_size=str(quote_size))
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
            order_data = {
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "SELL",
                "order_configuration": {
                    "market_market_ioc": {
                        "base_size": str(base_size),
                    }
                },
            }

            result = self._request("POST", "/api/v3/brokerage/orders", data=order_data)

            order = result.get("success_response", result)

            return OrderResult(
                order_id=order.get("order_id", ""),
                side="sell",
                size=Decimal(order.get("filled_size", "0")),
                filled_price=Decimal(order.get("average_filled_price", "0")) if order.get("average_filled_price") else None,
                status=order.get("status", "unknown"),
                fee=Decimal(order.get("total_fees", "0")),
                success=True,
            )

        except Exception as e:
            logger.error("market_sell_failed", error=str(e), product_id=product_id, base_size=str(base_size))
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

    def get_order(self, order_id: str) -> dict:
        """Get order details by ID."""
        return self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self._request("POST", "/api/v3/brokerage/orders/batch_cancel", data={"order_ids": [order_id]})
            return True
        except Exception as e:
            logger.error("cancel_order_failed", error=str(e), order_id=order_id)
            return False
