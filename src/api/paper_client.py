"""
Paper trading client for simulated trading.

Uses real market data from any exchange but simulates order execution
with virtual balances. Perfect for strategy testing before live trading.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
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
    - Uses real exchange market data
    - Simulates order fills with configurable slippage
    - Tracks virtual base and quote currency balances
    - Records all trades for analysis
    - Identical interface to exchange clients for easy switching
    """

    # Simulated trading fee (Coinbase Advanced taker fee)
    TAKER_FEE = Decimal("0.006")  # 0.6%

    # Simulated slippage for market orders
    SLIPPAGE = Decimal("0.001")  # 0.1%

    def __init__(
        self,
        real_client: ExchangeClient,
        initial_quote: float = 10000.0,
        initial_base: float = 0.0,
        trading_pair: str = "BTC-USD",
    ):
        """
        Initialize paper trading client.

        Args:
            real_client: Real exchange client for market data (Coinbase, Kraken, etc.)
            initial_quote: Starting quote currency balance (e.g., USD, EUR)
            initial_base: Starting base currency balance (e.g., BTC)
            trading_pair: Trading pair symbol (e.g., BTC-USD, BTC-EUR)
        """
        self.real_client = real_client
        self.trading_pair = trading_pair

        # Parse base and quote currencies from trading pair
        parts = trading_pair.split("-")
        self._base_currency = parts[0] if len(parts) >= 1 else "BTC"
        self._quote_currency = parts[1] if len(parts) >= 2 else "USD"

        # Virtual balances
        self._quote_balance = Decimal(str(initial_quote))
        self._base_balance = Decimal(str(initial_base))
        self._quote_hold = Decimal("0")
        self._base_hold = Decimal("0")

        # Trade history
        self._trades: list[PaperTrade] = []

        # Statistics
        self._total_fees = Decimal("0")
        self._total_volume = Decimal("0")

        logger.info(
            "paper_client_initialized",
            initial_quote=str(self._quote_balance),
            initial_base=str(self._base_balance),
        )

    def get_balance(self, currency: str = "BTC") -> Balance:
        """Get virtual account balance."""
        if currency == self._base_currency:
            return Balance(
                currency=self._base_currency,
                available=self._base_balance,
                hold=self._base_hold,
            )
        elif currency == self._quote_currency:
            return Balance(
                currency=self._quote_currency,
                available=self._quote_balance,
                hold=self._quote_hold,
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
        allow_negative_quote: bool = False,
    ) -> OrderResult:
        """
        Simulate a market buy order.

        Args:
            product_id: Trading pair (e.g., BTC-USD)
            quote_size: Amount to spend in quote currency
            allow_negative_quote: If True, allow buying even with insufficient quote balance

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

        # Check balance (can be skipped with allow_negative_quote)
        if quote_size > self._quote_balance and not allow_negative_quote:
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient {self._quote_currency} balance. Need {quote_size}, have {self._quote_balance}",
            )

        # Apply slippage (price goes up for buys)
        fill_price = market_price * (Decimal("1") + self.SLIPPAGE)

        # Calculate fee
        fee = quote_size * self.TAKER_FEE

        # Calculate base currency received
        effective_quote = quote_size - fee
        base_received = effective_quote / fill_price

        # Update balances
        self._quote_balance -= quote_size
        self._base_balance += base_received

        # Update statistics
        self._total_fees += fee
        self._total_volume += quote_size

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            side="buy",
            size=base_received,
            price=fill_price,
            fee=fee,
            slippage=fill_price - market_price,
        )
        self._trades.append(trade)

        logger.info(
            "paper_buy_executed",
            size=str(base_received),
            price=str(fill_price),
            fee=str(fee),
            quote_spent=str(quote_size),
            new_base_balance=str(self._base_balance),
            new_quote_balance=str(self._quote_balance),
        )

        return OrderResult(
            order_id=trade.trade_id,
            side="buy",
            size=base_received,
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
            base_size: Amount of base currency to sell

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
        if base_size > self._base_balance:
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient {self._base_currency} balance. Need {base_size}, have {self._base_balance}",
            )

        # Apply slippage (price goes down for sells)
        fill_price = market_price * (Decimal("1") - self.SLIPPAGE)

        # Calculate quote currency received
        gross_quote = base_size * fill_price

        # Calculate fee
        fee = gross_quote * self.TAKER_FEE
        net_quote = gross_quote - fee

        # Update balances
        self._base_balance -= base_size
        self._quote_balance += net_quote

        # Update statistics
        self._total_fees += fee
        self._total_volume += gross_quote

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
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
            quote_received=str(net_quote),
            new_base_balance=str(self._base_balance),
            new_quote_balance=str(self._quote_balance),
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

    def limit_buy_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Simulate a limit buy order with IOC time-in-force.

        For paper trading, fills at limit_price if limit >= ask (favorable).
        Otherwise returns unfilled (IOC cancellation).
        """
        # Get real market data
        try:
            market_data = self.get_market_data(product_id)
        except Exception as e:
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Failed to get market data: {e}",
            )

        # IOC: only fill if limit_price >= ask (willing to pay at least the ask)
        if limit_price < market_data.ask:
            logger.info(
                "paper_limit_buy_ioc_cancelled",
                limit_price=str(limit_price),
                ask=str(market_data.ask),
                reason="Limit below ask",
            )
            return OrderResult(
                order_id=str(uuid.uuid4()),
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="CANCELLED",
                fee=Decimal("0"),
                success=True,  # Order submitted successfully, just didn't fill
            )

        # Calculate cost
        quote_needed = base_size * limit_price

        # Check balance
        if quote_needed > self._quote_balance:
            return OrderResult(
                order_id="",
                side="buy",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient {self._quote_currency} balance. Need {quote_needed}, have {self._quote_balance}",
            )

        # IOC orders crossing spread fill at market price (no price improvement)
        fill_price = market_data.ask

        # Calculate actual cost and fee
        gross_cost = base_size * fill_price
        fee = gross_cost * self.TAKER_FEE  # IOC orders crossing spread take liquidity

        # Update balances
        self._quote_balance -= (gross_cost + fee)
        self._base_balance += base_size

        # Update statistics
        self._total_fees += fee
        self._total_volume += gross_cost

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            side="buy",
            size=base_size,
            price=fill_price,
            fee=fee,
            slippage=Decimal("0"),  # Limit orders have no slippage
        )
        self._trades.append(trade)

        logger.info(
            "paper_limit_buy_ioc_executed",
            size=str(base_size),
            limit_price=str(limit_price),
            fill_price=str(fill_price),
            fee=str(fee),
            new_base_balance=str(self._base_balance),
            new_quote_balance=str(self._quote_balance),
        )

        return OrderResult(
            order_id=trade.trade_id,
            side="buy",
            size=base_size,
            filled_price=fill_price,
            status="FILLED",
            fee=fee,
            success=True,
        )

    def limit_sell_ioc(
        self,
        product_id: str,
        base_size: Decimal,
        limit_price: Decimal,
    ) -> OrderResult:
        """
        Simulate a limit sell order with IOC time-in-force.

        For paper trading, fills at limit_price if limit <= bid (favorable).
        Otherwise returns unfilled (IOC cancellation).
        """
        # Get real market data
        try:
            market_data = self.get_market_data(product_id)
        except Exception as e:
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Failed to get market data: {e}",
            )

        # IOC: only fill if limit_price <= bid (willing to accept at most the bid)
        if limit_price > market_data.bid:
            logger.info(
                "paper_limit_sell_ioc_cancelled",
                limit_price=str(limit_price),
                bid=str(market_data.bid),
                reason="Limit above bid",
            )
            return OrderResult(
                order_id=str(uuid.uuid4()),
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="CANCELLED",
                fee=Decimal("0"),
                success=True,  # Order submitted successfully, just didn't fill
            )

        # Check balance
        if base_size > self._base_balance:
            return OrderResult(
                order_id="",
                side="sell",
                size=Decimal("0"),
                filled_price=None,
                status="failed",
                fee=Decimal("0"),
                success=False,
                error=f"Insufficient {self._base_currency} balance. Need {base_size}, have {self._base_balance}",
            )

        # IOC orders crossing spread fill at market price (no price improvement)
        fill_price = market_data.bid

        # Calculate quote received and fee
        gross_quote = base_size * fill_price
        fee = gross_quote * self.TAKER_FEE  # IOC orders crossing spread take liquidity
        net_quote = gross_quote - fee

        # Update balances
        self._base_balance -= base_size
        self._quote_balance += net_quote

        # Update statistics
        self._total_fees += fee
        self._total_volume += gross_quote

        # Record trade
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            side="sell",
            size=base_size,
            price=fill_price,
            fee=fee,
            slippage=Decimal("0"),  # Limit orders have no slippage
        )
        self._trades.append(trade)

        logger.info(
            "paper_limit_sell_ioc_executed",
            size=str(base_size),
            limit_price=str(limit_price),
            fill_price=str(fill_price),
            fee=str(fee),
            quote_received=str(net_quote),
            new_base_balance=str(self._base_balance),
            new_quote_balance=str(self._quote_balance),
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

    def get_portfolio_value(self, product_id: Optional[str] = None) -> Decimal:
        """Get total portfolio value in quote currency."""
        target_pair = product_id or self.trading_pair
        price = self.get_current_price(target_pair)
        base_value = self._base_balance * price
        return self._quote_balance + base_value

    def get_statistics(self) -> dict:
        """Get paper trading statistics."""
        return {
            "total_trades": len(self._trades),
            "total_fees": str(self._total_fees),
            "total_volume": str(self._total_volume),
            "current_base": str(self._base_balance),
            "current_quote": str(self._quote_balance),
        }

    def get_trade_history(self) -> list[PaperTrade]:
        """Get all paper trades."""
        return self._trades.copy()

    def reset(self, initial_quote: float = 10000.0, initial_base: float = 0.0) -> None:
        """Reset paper trading account."""
        self._quote_balance = Decimal(str(initial_quote))
        self._base_balance = Decimal(str(initial_base))
        self._quote_hold = Decimal("0")
        self._base_hold = Decimal("0")
        self._trades = []
        self._total_fees = Decimal("0")
        self._total_volume = Decimal("0")

        logger.info(
            "paper_client_reset",
            initial_quote=str(self._quote_balance),
            initial_base=str(self._base_balance),
        )
