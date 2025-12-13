"""
Exchange protocol interface and shared data types.

Defines the common interface that all exchange clients must implement,
enabling easy switching between Coinbase, Kraken, and other exchanges.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Protocol

import pandas as pd


@dataclass
class Balance:
    """Account balance information."""

    currency: str
    available: Decimal
    hold: Decimal

    @property
    def total(self) -> Decimal:
        return self.available + self.hold


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: str
    side: str
    size: Decimal
    filled_price: Optional[Decimal]
    status: str
    fee: Decimal
    success: bool
    error: Optional[str] = None


@dataclass
class MarketData:
    """Current market data."""

    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    timestamp: datetime


class ExchangeClient(Protocol):
    """
    Protocol defining the interface for exchange clients.

    All exchange implementations (Coinbase, Kraken, etc.) must
    implement these methods to be interchangeable.
    """

    def get_balance(self, currency: str) -> Balance:
        """
        Get account balance for a currency.

        Args:
            currency: Currency code (BTC, USD, EUR, etc.)

        Returns:
            Balance information
        """
        ...

    def get_current_price(self, product_id: str) -> Decimal:
        """
        Get current market price.

        Args:
            product_id: Trading pair in normalized format (e.g., BTC-USD)

        Returns:
            Current price
        """
        ...

    def get_market_data(self, product_id: str) -> MarketData:
        """
        Get current market data including bid/ask.

        Args:
            product_id: Trading pair in normalized format

        Returns:
            MarketData with current prices
        """
        ...

    def get_candles(
        self,
        product_id: str,
        granularity: str,
        limit: int,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV candle data.

        Args:
            product_id: Trading pair in normalized format
            granularity: Candle size (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE,
                        THIRTY_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

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
        ...

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
        ...

    def limit_buy_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Execute a limit buy order with IOC (Immediate-Or-Cancel) time-in-force.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount to buy in base currency (BTC)
            limit_price: Maximum price willing to pay

        Returns:
            OrderResult with execution details
        """
        ...

    def limit_sell_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Execute a limit sell order with IOC (Immediate-Or-Cancel) time-in-force.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount to sell in base currency (BTC)
            limit_price: Minimum price willing to accept

        Returns:
            OrderResult with execution details
        """
        ...

    def get_order(self, order_id: str) -> dict:
        """
        Get order details by ID.

        Args:
            order_id: The order ID

        Returns:
            Order details dictionary
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.

        Args:
            order_id: The order ID

        Returns:
            True if cancelled successfully
        """
        ...
