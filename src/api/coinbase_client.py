"""
Coinbase Advanced Trade API client with Ed25519 support.

Uses custom JWT authentication since the official SDK doesn't support Ed25519 yet.
Implements the ExchangeClient protocol for multi-exchange support.
"""

import time
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
            timeout=30,  # Prevent hanging on network issues
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
        # Get ticker data with best bid/ask
        ticker = self._request("GET", f"/api/v3/brokerage/market/products/{product_id}/ticker")

        # Get product data for volume
        product = self.get_product(product_id)

        # Extract best bid/ask from ticker
        best_bid = ticker.get("best_bid", "0")
        best_ask = ticker.get("best_ask", "0")

        # Calculate mid price from bid/ask, fallback to product price
        if best_bid and best_ask and best_bid != "" and best_ask != "":
            mid_price = (Decimal(best_bid) + Decimal(best_ask)) / Decimal("2")
        else:
            mid_price = Decimal(product.get("price", "0"))

        return MarketData(
            symbol=product_id,
            price=mid_price,
            bid=Decimal(best_bid) if best_bid else Decimal("0"),
            ask=Decimal(best_ask) if best_ask else Decimal("0"),
            volume_24h=Decimal(product.get("volume_24h", "0")),
            timestamp=datetime.now(timezone.utc),
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

    def _poll_order_fill(
        self,
        order_id: str,
        max_attempts: int = 10,
        poll_interval: float = 0.5,
    ) -> dict:
        """
        Poll for order fill status.

        Coinbase API returns order_id immediately but fill details (filled_size,
        average_filled_price) are populated asynchronously. This method polls
        until the order fills or times out.

        Args:
            order_id: The order ID to poll
            max_attempts: Maximum number of polling attempts
            poll_interval: Seconds between polls

        Returns:
            Order data dict with fill details
        """
        order_data = {}  # Initialize to prevent NameError if all attempts fail
        for attempt in range(max_attempts):
            try:
                order_status = self.get_order(order_id)
                order_data = order_status.get("order", order_status)
                status = order_data.get("status", "")
                filled_size = Decimal(order_data.get("filled_size", "0"))

                logger.debug(
                    "polling_order_status",
                    order_id=order_id,
                    attempt=attempt + 1,
                    status=status,
                    filled_size=str(filled_size),
                )

                if status in ["FILLED", "COMPLETED"] or filled_size > Decimal("0"):
                    return order_data

                time.sleep(poll_interval)

            except Exception as e:
                logger.warning("poll_order_failed", order_id=order_id, attempt=attempt + 1, error=str(e))
                time.sleep(poll_interval)

        # Timeout: order may have filled but we don't have confirmation
        # Raise exception to trigger retry logic or fail safely
        error_msg = f"Order polling timed out after {max_attempts} attempts"
        logger.error("order_poll_timeout", order_id=order_id, max_attempts=max_attempts)
        raise TimeoutError(error_msg)

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
            OrderResult with execution details (polls until filled)
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
            order_id = order.get("order_id", "")

            if not order_id:
                logger.error("market_buy_no_order_id", response=result)
                return OrderResult(
                    order_id="",
                    side="buy",
                    size=Decimal("0"),
                    filled_price=None,
                    status="failed",
                    fee=Decimal("0"),
                    success=False,
                    error="No order_id in response",
                )

            # Poll for fill details (Coinbase fills async)
            # Raises TimeoutError if polling exceeds max_attempts
            filled_order = self._poll_order_fill(order_id)

            filled_size = Decimal(filled_order.get("filled_size", "0"))
            avg_price = filled_order.get("average_filled_price")
            total_fees = Decimal(filled_order.get("total_fees", "0"))
            status = filled_order.get("status", "unknown")

            logger.info(
                "market_buy_completed",
                order_id=order_id,
                filled_size=str(filled_size),
                average_price=avg_price,
                status=status,
            )

            return OrderResult(
                order_id=order_id,
                side="buy",
                size=filled_size,
                filled_price=Decimal(avg_price) if avg_price else None,
                status=status,
                fee=total_fees,
                success=filled_size > Decimal("0"),
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
            OrderResult with execution details (polls until filled)
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
            order_id = order.get("order_id", "")

            if not order_id:
                logger.error("market_sell_no_order_id", response=result)
                return OrderResult(
                    order_id="",
                    side="sell",
                    size=Decimal("0"),
                    filled_price=None,
                    status="failed",
                    fee=Decimal("0"),
                    success=False,
                    error="No order_id in response",
                )

            # Poll for fill details (Coinbase fills async)
            # Raises TimeoutError if polling exceeds max_attempts
            filled_order = self._poll_order_fill(order_id)

            filled_size = Decimal(filled_order.get("filled_size", "0"))
            avg_price = filled_order.get("average_filled_price")
            total_fees = Decimal(filled_order.get("total_fees", "0"))
            status = filled_order.get("status", "unknown")

            logger.info(
                "market_sell_completed",
                order_id=order_id,
                filled_size=str(filled_size),
                average_price=avg_price,
                status=status,
            )

            return OrderResult(
                order_id=order_id,
                side="sell",
                size=filled_size,
                filled_price=Decimal(avg_price) if avg_price else None,
                status=status,
                fee=total_fees,
                success=filled_size > Decimal("0"),
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
            order_data = {
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "BUY",
                "order_configuration": {
                    "limit_limit_ioc": {
                        "base_size": str(base_size),
                        "limit_price": str(limit_price),
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
            logger.error(
                "limit_buy_ioc_failed",
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
            order_data = {
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "SELL",
                "order_configuration": {
                    "limit_limit_ioc": {
                        "base_size": str(base_size),
                        "limit_price": str(limit_price),
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
            logger.error(
                "limit_sell_ioc_failed",
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

    def get_trading_fee_rate(self, product_id: str) -> Decimal:
        """
        Get current trading fee rate for a product.

        Coinbase Advanced Trade fees are volume-tiered:
        - Taker: 0.6% (default), 0.4% (>$10K/30d), down to 0.05% (>$300M/30d)
        - Maker: 0.4% (default), 0.25% (>$10K/30d), down to 0.0% (>$300M/30d)

        Since the bot primarily uses IOC limit orders (which typically execute as taker),
        we use the taker fee rate for conservative profit margin validation.

        Args:
            product_id: Trading pair (e.g., BTC-USD)

        Returns:
            Current taker fee rate as decimal (e.g., Decimal("0.006") for 0.6%)
        """
        try:
            # Get transaction summary for fee tier information
            response = self._request("GET", "/api/v3/brokerage/transaction_summary")

            # Extract taker fee rate if available
            # Fee rates are returned as strings like "0.006" (0.6%)
            if "fee_tier" in response:
                fee_tier = response["fee_tier"]
                if not isinstance(fee_tier, dict):
                    logger.warning(
                        "fee_tier_invalid_type",
                        product_id=product_id,
                        fee_tier_type=type(fee_tier).__name__,
                        message="fee_tier is not a dict, using default",
                    )
                elif "taker_fee_rate" in fee_tier:
                    try:
                        fee_rate = Decimal(fee_tier["taker_fee_rate"])
                        # Sanity check: fee should be between 0.001% and 5%
                        if Decimal("0.00001") <= fee_rate <= Decimal("0.05"):
                            return fee_rate
                        else:
                            logger.warning(
                                "fee_rate_out_of_range",
                                product_id=product_id,
                                fee_rate=str(fee_rate),
                                message="Fee rate outside reasonable range, using default",
                            )
                    except (ValueError, TypeError, ArithmeticError) as e:
                        logger.warning(
                            "fee_rate_conversion_failed",
                            product_id=product_id,
                            taker_fee_rate=fee_tier.get("taker_fee_rate"),
                            error=str(e),
                            message="Failed to convert fee rate to Decimal, using default",
                        )

            # Fallback to default Coinbase Advanced taker fee
            logger.warning(
                "fee_tier_not_found",
                product_id=product_id,
                message="Using default taker fee rate (0.6%)",
            )
            return Decimal("0.006")

        except Exception as e:
            logger.warning(
                "get_trading_fee_rate_failed",
                product_id=product_id,
                error=str(e),
                message="Using default taker fee rate (0.6%)",
            )
            return Decimal("0.006")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def limit_buy(
        self,
        symbol: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> dict:
        """
        Place a Good-Till-Cancelled (GTC) limit buy order.

        Order remains in the book until filled or manually cancelled.
        Used primarily for integration testing.

        Args:
            symbol: Trading pair (e.g., BTC-USD)
            base_size: Amount to buy in base currency (BTC)
            limit_price: Maximum price willing to pay

        Returns:
            dict with order details including order_id

        Raises:
            ValueError: If order parameters are invalid
        """
        # Validate parameters
        if base_size <= Decimal("0"):
            raise ValueError(f"base_size must be positive, got {base_size}")
        if limit_price <= Decimal("0"):
            raise ValueError(f"limit_price must be positive, got {limit_price}")

        # Calculate order value for safety check
        order_value_usd = base_size * limit_price

        # Warn if order value seems excessive (likely a bug)
        # This method is primarily for testing, so large orders are suspicious
        if order_value_usd > Decimal("100"):
            logger.warning(
                "large_order_detected",
                symbol=symbol,
                base_size=str(base_size),
                limit_price=str(limit_price),
                order_value_usd=str(order_value_usd),
                message="Order value exceeds $100. This method is primarily for integration testing. "
                        "Verify this is intentional and not a calculation error."
            )

        order_data = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": symbol,
            "side": "BUY",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(base_size),
                    "limit_price": str(limit_price),
                    "post_only": False,
                }
            },
        }

        result = self._request("POST", "/api/v3/brokerage/orders", data=order_data)
        order = result.get("success_response", result)

        logger.info(
            "limit_buy_placed",
            order_id=order.get("order_id"),
            symbol=symbol,
            base_size=str(base_size),
            limit_price=str(limit_price),
        )

        return order

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    def limit_sell(
        self,
        symbol: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> dict:
        """
        Place a Good-Till-Cancelled (GTC) limit sell order.

        Order remains in the book until filled or manually cancelled.
        Used primarily for integration testing.

        Args:
            symbol: Trading pair (e.g., BTC-USD)
            base_size: Amount to sell in base currency (BTC)
            limit_price: Minimum price willing to accept

        Returns:
            dict with order details including order_id

        Raises:
            ValueError: If order parameters are invalid
        """
        # Validate parameters
        if base_size <= Decimal("0"):
            raise ValueError(f"base_size must be positive, got {base_size}")
        if limit_price <= Decimal("0"):
            raise ValueError(f"limit_price must be positive, got {limit_price}")

        # Calculate order value for safety check
        order_value_usd = base_size * limit_price

        # Warn if order value seems excessive (likely a bug)
        # This method is primarily for testing, so large orders are suspicious
        if order_value_usd > Decimal("100"):
            logger.warning(
                "large_order_detected",
                symbol=symbol,
                base_size=str(base_size),
                limit_price=str(limit_price),
                order_value_usd=str(order_value_usd),
                message="Order value exceeds $100. This method is primarily for integration testing. "
                        "Verify this is intentional and not a calculation error."
            )

        order_data = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": symbol,
            "side": "SELL",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(base_size),
                    "limit_price": str(limit_price),
                    "post_only": False,
                }
            },
        }

        result = self._request("POST", "/api/v3/brokerage/orders", data=order_data)
        order = result.get("success_response", result)

        logger.info(
            "limit_sell_placed",
            order_id=order.get("order_id"),
            symbol=symbol,
            base_size=str(base_size),
            limit_price=str(limit_price),
        )

        return order
