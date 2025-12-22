"""
Position tracking service for managing trading positions and P&L.

Handles:
- Position tracking after buys (weighted average cost)
- Realized P&L calculation on sells
- Portfolio value calculation

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from src.api.exchange_protocol import ExchangeClient
from src.state.database import Database, BotMode

logger = structlog.get_logger(__name__)


@dataclass
class PositionConfig:
    """Configuration for position service."""

    # Trading context
    trading_pair: str
    is_paper_trading: bool

    # Currency pair (derived from trading_pair)
    base_currency: str
    quote_currency: str


class PositionService:
    """
    Service for tracking positions and calculating P&L.

    Responsibilities:
    - Update position after buy (weighted average cost)
    - Calculate realized P&L on sell
    - Calculate portfolio value
    """

    def __init__(
        self,
        config: PositionConfig,
        db: Database,
        exchange_client: ExchangeClient,
    ):
        """
        Initialize position service.

        Args:
            config: Position service configuration
            db: Database for position storage
            exchange_client: Exchange client for balance queries
        """
        self.config = config
        self.db = db
        self.client = exchange_client

    def update_config(self, config: PositionConfig) -> None:
        """
        Update the position configuration.

        Used for hot-reload of settings.

        Args:
            config: New position configuration
        """
        self.config = config
        logger.info("position_service_config_updated")

    def update_after_buy(
        self,
        size: Decimal,
        price: Decimal,
        fee: Decimal,
        is_paper: bool,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> Decimal:
        """
        Update position with new buy, recalculating weighted average cost.

        Cost basis includes fees to accurately reflect break-even price.

        Args:
            size: Base currency amount bought (e.g., BTC)
            price: Price paid per unit
            fee: Trading fee paid (in quote currency)
            is_paper: Whether this is a paper trade
            bot_mode: Bot mode ("normal" or "inverted")

        Returns:
            The new weighted average cost for the position
        """
        current = self.db.get_current_position(
            self.config.trading_pair, is_paper=is_paper, bot_mode=bot_mode
        )

        if current:
            old_qty = current.get_quantity()
            old_cost = current.get_average_cost()
            old_value = old_qty * old_cost
        else:
            old_qty = Decimal("0")
            old_value = Decimal("0")

        new_qty = old_qty + size
        # Include fee in cost basis for accurate break-even calculation
        new_value = old_value + (size * price) + fee
        new_avg_cost = new_value / new_qty if new_qty > 0 else Decimal("0")

        self.db.update_position(
            symbol=self.config.trading_pair,
            quantity=new_qty,
            average_cost=new_avg_cost,
            is_paper=is_paper,
            bot_mode=bot_mode,
        )

        logger.debug(
            "position_updated_after_buy",
            old_qty=str(old_qty),
            new_qty=str(new_qty),
            new_avg_cost=str(new_avg_cost),
        )

        return new_avg_cost

    def calculate_realized_pnl(
        self,
        size: Decimal,
        sell_price: Decimal,
        fee: Decimal,
        is_paper: bool,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> Decimal:
        """
        Calculate realized PnL for a sell based on average cost.

        Net PnL = (sell_price - avg_cost) * size - sell_fee

        Args:
            size: Base currency amount sold (e.g., BTC)
            sell_price: Price received per unit
            fee: Trading fee paid on sell (in quote currency)
            is_paper: Whether this is a paper trade
            bot_mode: Bot mode ("normal" or "inverted")

        Returns:
            Realized profit/loss (positive = profit, negative = loss)
        """
        current = self.db.get_current_position(
            self.config.trading_pair, is_paper=is_paper, bot_mode=bot_mode
        )

        if not current or current.get_quantity() <= 0:
            return Decimal("0") - fee  # Still deduct fee even without position

        avg_cost = current.get_average_cost()
        # Deduct sell fee for accurate net P&L
        pnl = ((sell_price - avg_cost) * size) - fee

        # Update position (reduce quantity, keep avg cost)
        new_qty = current.get_quantity() - size
        if new_qty < Decimal("0"):
            new_qty = Decimal("0")

        self.db.update_position(
            symbol=self.config.trading_pair,
            quantity=new_qty,
            average_cost=avg_cost,
            is_paper=is_paper,
            bot_mode=bot_mode,
        )

        logger.info(
            "realized_pnl_calculated",
            size=str(size),
            sell_price=str(sell_price),
            avg_cost=str(avg_cost),
            pnl=str(pnl),
            remaining_qty=str(new_qty),
            bot_mode=bot_mode,
        )

        return pnl

    def get_portfolio_value(self, current_price: Optional[Decimal] = None) -> Decimal:
        """
        Get total portfolio value in quote currency.

        Args:
            current_price: Optional current price. If not provided, fetches from exchange.

        Returns:
            Total portfolio value (quote balance + base balance * price)
        """
        base_balance = self.client.get_balance(self.config.base_currency).available
        quote_balance = self.client.get_balance(self.config.quote_currency).available

        if current_price is None:
            current_price = self.client.get_current_price(self.config.trading_pair)

        base_value = base_balance * current_price
        return quote_balance + base_value

    def get_current_position(
        self, is_paper: Optional[bool] = None, bot_mode: BotMode = BotMode.NORMAL
    ):
        """
        Get current position from database.

        Args:
            is_paper: Whether to get paper position. Defaults to config setting.
            bot_mode: Bot mode ("normal" or "inverted")

        Returns:
            Position object or None
        """
        if is_paper is None:
            is_paper = self.config.is_paper_trading

        return self.db.get_current_position(
            self.config.trading_pair, is_paper=is_paper, bot_mode=bot_mode
        )
