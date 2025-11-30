"""
Paper trading client for simulated trading.

Uses real market data from any exchange but simulates order execution
with virtual balances. Perfect for strategy testing before live trading.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
import uuid

import pandas as pd
import structlog

from src.api.exchange_protocol import Balance, ExchangeClient, MarketData, OrderResult

logger = structlog.get_logger(__name__)


@dataclass
class PaperTrade:
    """Record of a paper trade."""

    trade_id: str
    timestamp: datetime
    side: str
    size: Decimal
    price: Decimal
    fee: Decimal
    slippage: Decimal


class PaperTradingClient:
    """
    Paper trading client that simulates trades with real market data.

    Features:
    - Uses real Coinbase market data
    - Simulates order fills with configurable slippage
    - Tracks virtual BTC and USD balances
    - Records all trades for analysis
    - Identical interface to CoinbaseClient for easy switching
    """

    # Simulated trading fees (Coinbase Advanced is typically 0.5%)
    MAKER_FEE = Decimal("0.004")  # 0.4%
    TAKER_FEE = Decimal("0.006")  # 0.6%

    # Simulated slippage for market orders
    SLIPPAGE = Decimal("0.001")  # 0.1%

    def __init__(
        self,
        real_client: ExchangeClient,
        initial_usd: float = 10000.0,
        initial_btc: float = 0.0,
    ):
        """
        Initialize paper trading client.

        Args:
            real_client: Real exchange client for market data (Coinbase, Kraken, etc.)
            initial_usd: Starting USD balance
            initial_btc: Starting BTC balance
        """
        self.real_client = real_client

        # Virtual balances
        self._usd_balance = Decimal(str(initial_usd))
        self._btc_balance = Decimal(str(initial_btc))
        self._usd_hold = Decimal("0")
        self._btc_hold = Decimal("0")

        # Trade history
        self._trades: list[PaperTrade] = []

        # Statistics
        self._total_fees = Decimal("0")
        self._total_volume = Decimal("0")

        logger.info(
            "paper_client_initialized",
            initial_usd=str(self._usd_balance),
            initial_btc=str(self._btc_balance),
        )

    def get_balance(self, currency: str = "BTC") -> Balance:
        """Get virtual account balance."""
        if currency == "BTC":
            return Balance(
                currency="BTC",
                available=self._btc_balance,
                hold=self._btc_hold,
            )
        elif currency == "USD":
            return Balance(
                currency="USD",
                available=self._usd_balance,
                hold=self._usd_hold,
            )
        else:
            return Balance(currency=currency, available=Decimal("0"), hold=Decimal("0"))

    def get_current_price(self, product_id: str = "BTC-USD") -> Decimal:
        """Get real current market price."""
        return self.real_client.get_current_price(product_id)

    def get_market_data(self, product_id: str = "BTC-USD") -> MarketData:
        """Get real current market data."""
        return self.real_client.get_market_data(product_id)

    def get_candles(
        self,
        product_id: str = "BTC-USD",
        granularity: str = "ONE_HOUR",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get real historical candle data."""
        return self.real_client.get_candles(product_id, granularity, limit)

    def market_buy(
        self,
        product_id: str,
        quote_size: Decimal,
    ) -> OrderResult:
        """
        Simulate a market buy order.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            quote_size: Amount to spend in USD

        Returns:
            OrderResult with simulated execution
        """
        # Get real market price
        try:
            market_price = self.get_current_price(product_id)
        except Exception as e:
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Failed to get market price: {e}",
            )

        # Check balance
        if quote_size > self._usd_balance:
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient USD balance. Need ${quote_size}, have ${self._usd_balance}",
            )

        # Apply slippage (price goes up for buys)
        fill_price = market_price * (Decimal("1") + self.SLIPPAGE)

        # Calculate fee
        fee = quote_size * self.TAKER_FEE

        # Calculate BTC received
        effective_usd = quote_size - fee
        btc_received = effective_usd / fill_price

        # Update balances
        self._usd_balance -= quote_size
        self._btc_balance += btc_received

        # Update statistics
        self._total_fees += fee
        self._total_volume += quote_size

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            side="buy",
            size=btc_received,
            price=fill_price,
            fee=fee,
            slippage=fill_price - market_price,
        )
        self._trades.append(trade)

        logger.info(
            "paper_buy_executed",
            size=str(btc_received),
            price=str(fill_price),
            fee=str(fee),
            usd_spent=str(quote_size),
            new_btc_balance=str(self._btc_balance),
            new_usd_balance=str(self._usd_balance),
        )

        return OrderResult(
            order_id=trade.trade_id,
            side="buy",
            size=btc_received,
            filled_price=fill_price,
            status="FILLED",
            fee=fee,
            success=True,
        )

    def market_sell(
        self,
        product_id: str,
        base_size: Decimal,
    ) -> OrderResult:
        """
        Simulate a market sell order.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            base_size: Amount of BTC to sell

        Returns:
            OrderResult with simulated execution
        """
        # Get real market price
        try:
            market_price = self.get_current_price(product_id)
        except Exception as e:
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Failed to get market price: {e}",
            )

        # Check balance
        if base_size > self._btc_balance:
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient BTC balance. Need {base_size}, have {self._btc_balance}",
            )

        # Apply slippage (price goes down for sells)
        fill_price = market_price * (Decimal("1") - self.SLIPPAGE)

        # Calculate USD received
        gross_usd = base_size * fill_price

        # Calculate fee
        fee = gross_usd * self.TAKER_FEE
        net_usd = gross_usd - fee

        # Update balances
        self._btc_balance -= base_size
        self._usd_balance += net_usd

        # Update statistics
        self._total_fees += fee
        self._total_volume += gross_usd

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            side="sell",
            size=base_size,
            price=fill_price,
            fee=fee,
            slippage=market_price - fill_price,
        )
        self._trades.append(trade)

        logger.info(
            "paper_sell_executed",
            size=str(base_size),
            price=str(fill_price),
            fee=str(fee),
            usd_received=str(net_usd),
            new_btc_balance=str(self._btc_balance),
            new_usd_balance=str(self._usd_balance),
        )

        return OrderResult(
            order_id=trade.trade_id,
            side="sell",
            size=base_size,
            filled_price=fill_price,
            status="FILLED",
            fee=fee,
            success=True,
        )

    def get_order(self, order_id: str) -> dict:
        """Get paper order details."""
        for trade in self._trades:
            if trade.trade_id == order_id:
                return {
                    "order_id": trade.trade_id,
                    "status": "FILLED",
                    "side": trade.side,
                    "filled_size": str(trade.size),
                    "average_filled_price": str(trade.price),
                    "total_fees": str(trade.fee),
                }
        return {}

    def cancel_order(self, order_id: str) -> bool:
        """Paper orders are instantly filled, cannot cancel."""
        return False

    # Paper trading specific methods

    def get_portfolio_value(self, product_id: str = "BTC-USD") -> Decimal:
        """Get total portfolio value in USD."""
        btc_price = self.get_current_price(product_id)
        btc_value = self._btc_balance * btc_price
        return self._usd_balance + btc_value

    def get_statistics(self) -> dict:
        """Get paper trading statistics."""
        return {
            "total_trades": len(self._trades),
            "total_fees": str(self._total_fees),
            "total_volume": str(self._total_volume),
            "current_btc": str(self._btc_balance),
            "current_usd": str(self._usd_balance),
        }

    def get_trade_history(self) -> list[PaperTrade]:
        """Get all paper trades."""
        return self._trades.copy()

    def reset(self, initial_usd: float = 10000.0, initial_btc: float = 0.0) -> None:
        """Reset paper trading account."""
        self._usd_balance = Decimal(str(initial_usd))
        self._btc_balance = Decimal(str(initial_btc))
        self._usd_hold = Decimal("0")
        self._btc_hold = Decimal("0")
        self._trades = []
        self._total_fees = Decimal("0")
        self._total_volume = Decimal("0")

        logger.info(
            "paper_client_reset",
            initial_usd=str(self._usd_balance),
            initial_btc=str(self._btc_balance),
        )
