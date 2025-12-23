"""
Cramer Mode service for inverse trading strategy comparison.

Handles:
- Cramer Mode client initialization and state management
- Inverse trade execution (opposite of normal bot)
- Independent trailing stop management
- Portfolio value tracking

Cramer Mode is PAPER TRADING ONLY - it uses virtual balance for strategy
comparison against the normal bot. Named after Jim Cramer for the inverse
trading meme.

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional, TYPE_CHECKING

import structlog

from src.api.paper_client import PaperTradingClient
from src.safety.trade_cooldown import TradeCooldown, TradeCooldownConfig
from src.state.database import Database, BotMode

if TYPE_CHECKING:
    import pandas as pd
    from src.api.exchange_protocol import ExchangeClient
    from src.strategy.position_sizer import PositionSizer
    from src.daemon.position_service import PositionService

logger = structlog.get_logger(__name__)


@dataclass
class CramerConfig:
    """Configuration for Cramer Mode service."""

    # Trading context
    trading_pair: str
    base_currency: str
    quote_currency: str

    # Enable/disable
    enable_cramer_mode: bool = False

    # Position sizing constraints
    min_trade_quote: Decimal = Decimal("10")
    min_trade_base: Decimal = Decimal("0.0001")

    # Take profit
    enable_take_profit: bool = False

    # Cooldown settings
    enable_cooldown: bool = False
    cooldown_period_hours: float = 4.0
    cooldown_min_price_move: float = 0.5


class CramerService:
    """
    Service for managing Cramer Mode (inverse trading).

    Cramer Mode trades the OPPOSITE of what the normal bot does:
    - When normal bot buys, Cramer sells
    - When normal bot sells, Cramer buys

    This is for strategy comparison only - purely paper trading.

    Responsibilities:
    - Manage Cramer Mode client and state
    - Execute inverse trades when normal bot trades
    - Execute independent trades when normal bot is blocked
    - Manage Cramer Mode trailing stops
    - Track Cramer Mode portfolio value
    """

    def __init__(
        self,
        config: CramerConfig,
        db: Database,
        position_sizer: "PositionSizer",
        position_service: "PositionService",
        real_client: "ExchangeClient",
        create_trailing_stop_callback: Callable,
    ):
        """
        Initialize Cramer Mode service.

        Args:
            config: Cramer Mode configuration
            db: Database for position/trade storage
            position_sizer: Position sizer for calculating trade sizes
            position_service: Position service for P&L calculation
            real_client: Real exchange client for market data queries
            create_trailing_stop_callback: Callback to create trailing stops
        """
        self.config = config
        self.db = db
        self.position_sizer = position_sizer
        self.position_service = position_service
        self.real_client = real_client
        self._create_trailing_stop = create_trailing_stop_callback

        # Cramer Mode state
        self.client: Optional[PaperTradingClient] = None
        self.trade_cooldown: Optional[TradeCooldown] = None
        self._disabled = False  # Disabled on balance mismatch

    def initialize(
        self,
        initial_quote: Optional[Decimal] = None,
        initial_base: Optional[Decimal] = None,
        normal_has_position: bool = False,
    ) -> bool:
        """
        Initialize Cramer Mode client with balance.

        Args:
            initial_quote: Initial quote currency balance
            initial_base: Initial base currency balance
            normal_has_position: Whether normal bot has an open position

        Returns:
            True if initialized successfully, False otherwise
        """
        if not self.config.enable_cramer_mode:
            logger.debug("cramer_mode_not_enabled")
            return False

        # Try to restore balance from database
        cramer_balance = self.db.get_last_paper_balance(
            self.config.trading_pair, bot_mode=BotMode.INVERTED
        )

        if cramer_balance:
            cramer_quote, cramer_base, _ = cramer_balance
            logger.info(
                "cramer_balance_restored_from_db",
                quote=str(cramer_quote),
                base=str(cramer_base),
            )
        elif initial_quote is not None and initial_base is not None:
            cramer_quote = initial_quote
            cramer_base = initial_base
            logger.info(
                "cramer_balance_copied_from_normal",
                quote=str(cramer_quote),
                base=str(cramer_base),
            )
        else:
            logger.warning("cramer_mode_no_initial_balance")
            return False

        # Warn if normal bot has position (Cramer starts fresh)
        if normal_has_position and not cramer_balance:
            logger.warning(
                "cramer_mode_starting_without_position",
                msg="Normal bot has open position but Cramer Mode starts fresh. "
                    "Consider disabling until position is closed for fair comparison.",
            )

        # Create paper trading client
        self.client = PaperTradingClient(
            real_client=self.real_client,
            trading_pair=self.config.trading_pair,
            initial_quote=float(cramer_quote),
            initial_base=float(cramer_base),
        )

        # Initialize cooldown if enabled
        if self.config.enable_cooldown:
            cooldown_config = TradeCooldownConfig(
                buy_cooldown_minutes=int(self.config.cooldown_period_hours * 60),
                sell_cooldown_minutes=int(self.config.cooldown_period_hours * 60),
                buy_price_change_percent=self.config.cooldown_min_price_move,
                sell_price_change_percent=self.config.cooldown_min_price_move,
            )
            self.trade_cooldown = TradeCooldown(
                config=cooldown_config,
                db=self.db,
                symbol=self.config.trading_pair,
                is_paper=True,
                bot_mode="inverted",
            )

        logger.info("cramer_mode_enabled", mode="inverted")
        return True

    def update_config(self, config: CramerConfig) -> None:
        """
        Update the Cramer Mode configuration.

        Note: Some settings (trading_pair, currencies) require restart.

        Args:
            config: New Cramer Mode configuration
        """
        self.config = config
        logger.info("cramer_service_config_updated")

    @property
    def is_enabled(self) -> bool:
        """Check if Cramer Mode is enabled and initialized."""
        return self.client is not None and not self._disabled

    @property
    def is_disabled(self) -> bool:
        """Check if Cramer Mode has been disabled due to error."""
        return self._disabled

    def disable(self, reason: str = "unknown") -> None:
        """
        Disable Cramer Mode for the session.

        Args:
            reason: Reason for disabling
        """
        self._disabled = True
        logger.warning("cramer_mode_disabled", reason=reason)

    def get_portfolio_value(self) -> Decimal:
        """Get Cramer Mode total portfolio value in quote currency."""
        if not self.client:
            return Decimal("0")

        base_balance = self.client.get_balance(self.config.base_currency).available
        quote_balance = self.client.get_balance(self.config.quote_currency).available
        current_price = self.real_client.get_current_price(self.config.trading_pair)

        base_value = base_balance * current_price
        return quote_balance + base_value

    def get_portfolio_info(self, current_price: Decimal) -> Optional[dict]:
        """
        Get Cramer Mode portfolio info for dashboard display.

        Args:
            current_price: Current market price

        Returns:
            Dict with balance info, or None if not enabled
        """
        if not self.client:
            return None

        try:
            base_balance = self.client.get_balance(self.config.base_currency).available
            quote_balance = self.client.get_balance(self.config.quote_currency).available
            base_value = base_balance * current_price
            portfolio_value = quote_balance + base_value

            position_percent = (
                float(base_value / portfolio_value * 100)
                if portfolio_value > 0 else 0.0
            )

            return {
                "quote_balance": str(quote_balance),
                "base_balance": str(base_balance),
                "portfolio_value": str(portfolio_value),
                "position_percent": position_percent,
            }
        except Exception as e:
            logger.warning("cramer_portfolio_fetch_error", error=str(e))
            return None

    def get_base_balance(self) -> Decimal:
        """Get Cramer Mode base currency balance."""
        if not self.client:
            return Decimal("0")
        return self.client.get_balance(self.config.base_currency).available

    def get_quote_balance(self) -> Decimal:
        """Get Cramer Mode quote currency balance."""
        if not self.client:
            return Decimal("0")
        return self.client.get_balance(self.config.quote_currency).available

    def can_trade(self, signal_direction: str) -> bool:
        """
        Check if Cramer Mode can trade the inverse direction.

        Args:
            signal_direction: Normal bot's signal direction ("buy" or "sell")

        Returns:
            True if Cramer can trade inverse direction
        """
        if not self.is_enabled:
            return False

        try:
            cramer_quote = self.get_quote_balance()
            cramer_base = self.get_base_balance()

            min_quote = self.config.min_trade_quote
            min_base = self.config.min_trade_base

            cramer_can_buy = cramer_quote > min_quote
            cramer_can_sell = cramer_base > min_base

            # Cramer trades INVERSE: signal=sell means Cramer buys
            if signal_direction == "sell":
                return cramer_can_buy
            else:  # signal=buy means Cramer sells
                return cramer_can_sell

        except Exception as e:
            logger.warning("cramer_balance_check_failed", error=str(e))
            return False

    def maybe_execute_independent(
        self,
        signal_side: str,
        candles: "pd.DataFrame",
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
        volatility: str = "normal",
    ) -> None:
        """
        Execute Cramer Mode trade (inverse of signal) if conditions allow.

        This is called when the normal bot CANNOT execute (blocked by balance,
        position limits, or cooldown) but Cramer may still trade independently.

        Args:
            signal_side: The direction the normal bot would have traded
            candles: OHLCV data for position sizing
            current_price: Current market price
            signal_score: Signal strength from scorer
            safety_multiplier: Combined safety multiplier
            volatility: Current volatility level
        """
        if not self.is_enabled:
            return

        # Cramer trades inverse: signal "buy" -> Cramer "sell", and vice versa
        cramer_side = "sell" if signal_side == "buy" else "buy"

        if safety_multiplier > 0:
            self.execute_trade(
                side=cramer_side,
                candles=candles,
                current_price=current_price,
                signal_score=signal_score,
                safety_multiplier=safety_multiplier,
                volatility=volatility,
            )
        else:
            logger.warning(
                "cramer_trade_blocked_by_safety",
                side=cramer_side,
                safety_multiplier=safety_multiplier,
                reason="circuit_breaker_active",
            )

    def execute_trade(
        self,
        side: str,
        candles: "pd.DataFrame",
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
        volatility: str = "normal",
    ) -> None:
        """
        Execute a trade for Cramer Mode (inverted mode).

        This executes the OPPOSITE trade of what the normal bot just did,
        using Cramer Mode's separate virtual balance.

        Args:
            side: Trade side ("buy" or "sell") for Cramer
            candles: OHLCV data for position sizing
            current_price: Current market price
            signal_score: Signal strength from scorer
            safety_multiplier: Combined safety multiplier
            volatility: Current volatility level
        """
        if not self.client:
            return

        if self._disabled:
            return

        cramer_quote_balance = self.get_quote_balance()
        cramer_base_balance = self.get_base_balance()

        logger.debug(
            "cramer_trade_attempt",
            side=side,
            quote_balance=str(cramer_quote_balance),
            base_balance=str(cramer_base_balance),
        )

        if side == "buy":
            self._execute_buy(
                candles=candles,
                current_price=current_price,
                signal_score=signal_score,
                safety_multiplier=safety_multiplier,
                quote_balance=cramer_quote_balance,
                base_balance=cramer_base_balance,
                volatility=volatility,
            )
        elif side == "sell":
            self._execute_sell(
                candles=candles,
                current_price=current_price,
                signal_score=signal_score,
                safety_multiplier=safety_multiplier,
                base_balance=cramer_base_balance,
                volatility=volatility,
            )

    def _execute_buy(
        self,
        candles: "pd.DataFrame",
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
        quote_balance: Decimal,
        base_balance: Decimal,
        volatility: str,
    ) -> None:
        """Execute a buy for Cramer Mode."""
        min_quote = self.config.min_trade_quote
        if quote_balance < min_quote:
            logger.info("cramer_skip_buy", reason="insufficient_quote_balance")
            return

        # Check cooldown
        if self.trade_cooldown:
            cooldown_ok, cooldown_reason = self.trade_cooldown.can_execute(
                "buy", current_price
            )
            if not cooldown_ok:
                logger.info("cramer_skip_buy", reason=f"cooldown: {cooldown_reason}")
                return

        # Calculate position size
        position = self.position_sizer.calculate_size(
            df=candles,
            current_price=current_price,
            quote_balance=quote_balance,
            base_balance=base_balance,
            signal_strength=abs(signal_score),
            side="buy",
            safety_multiplier=safety_multiplier,
        )

        if position.size_quote < min_quote:
            logger.info("cramer_skip_buy", reason="position_too_small")
            return

        # Execute buy
        result = self.client.market_buy(
            self.config.trading_pair,
            position.size_quote,
        )

        if result.success:
            filled_price = result.filled_price or current_price

            # Get new balances
            new_quote = self.client.get_balance(self.config.quote_currency).available
            new_base = self.client.get_balance(self.config.base_currency).available

            # Verify balance consistency
            if not self._verify_balance_consistency(new_quote, new_base):
                return

            # Update position
            new_avg_cost = self.position_service.update_after_buy(
                result.size, filled_price, result.fee,
                is_paper=True, bot_mode=BotMode.INVERTED
            )

            # Record trade
            self.db.record_trade(
                side="buy",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                symbol=self.config.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
                quote_balance_after=new_quote,
                base_balance_after=new_base,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=True, bot_mode=BotMode.INVERTED)

            # Create trailing stop
            take_profit_price = (
                position.take_profit_price
                if (self.config.enable_take_profit and position.take_profit_price)
                else None
            )
            self._create_trailing_stop(
                filled_price,
                candles,
                is_paper=True,
                avg_cost=new_avg_cost,
                volatility=volatility,
                take_profit_price=take_profit_price,
                bot_mode=BotMode.INVERTED,
            )

            logger.info(
                "cramer_buy_executed",
                size=str(result.size),
                price=str(filled_price),
                fee=str(result.fee),
            )

            # Update cooldown
            if self.trade_cooldown:
                self.trade_cooldown.record_trade("buy", filled_price)

            # Update daily stats
            self._update_daily_stats(current_price)
        else:
            logger.warning("cramer_buy_failed", error=result.error)

    def _execute_sell(
        self,
        candles: "pd.DataFrame",
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
        base_balance: Decimal,
        volatility: str,
    ) -> None:
        """Execute a sell for Cramer Mode."""
        min_base = self.config.min_trade_base
        if base_balance < min_base:
            logger.info("cramer_skip_sell", reason="insufficient_base_balance")
            return

        # Check cooldown
        if self.trade_cooldown:
            cooldown_ok, cooldown_reason = self.trade_cooldown.can_execute(
                "sell", current_price
            )
            if not cooldown_ok:
                logger.info("cramer_skip_sell", reason=f"cooldown: {cooldown_reason}")
                return

        # For aggressive selling, sell entire position on strong signal
        if abs(signal_score) >= 80:
            size_base = base_balance
        else:
            position = self.position_sizer.calculate_size(
                df=candles,
                current_price=current_price,
                quote_balance=Decimal("0"),
                base_balance=base_balance,
                signal_strength=abs(signal_score),
                side="sell",
                safety_multiplier=safety_multiplier,
            )
            size_base = position.size_base

        if size_base < min_base:
            logger.info("cramer_skip_sell", reason="position_too_small")
            return

        # Execute sell
        result = self.client.market_sell(
            self.config.trading_pair,
            size_base,
        )

        if result.success:
            filled_price = result.filled_price or current_price

            # Get new balances
            new_quote = self.client.get_balance(self.config.quote_currency).available
            new_base = self.client.get_balance(self.config.base_currency).available

            # Verify balance consistency
            if not self._verify_balance_consistency(new_quote, new_base):
                return

            # Calculate realized P&L
            realized_pnl = self.position_service.calculate_realized_pnl(
                result.size, filled_price, result.fee,
                is_paper=True, bot_mode=BotMode.INVERTED
            )

            # Deactivate trailing stop
            self.db.deactivate_trailing_stop(
                self.config.trading_pair, is_paper=True, bot_mode=BotMode.INVERTED
            )

            # Record trade
            self.db.record_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                realized_pnl=realized_pnl,
                symbol=self.config.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
                quote_balance_after=new_quote,
                base_balance_after=new_base,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=True, bot_mode=BotMode.INVERTED)

            logger.info(
                "cramer_sell_executed",
                size=str(result.size),
                price=str(filled_price),
                fee=str(result.fee),
                realized_pnl=str(realized_pnl),
            )

            # Update cooldown
            if self.trade_cooldown:
                self.trade_cooldown.record_trade("sell", filled_price)

            # Update daily stats
            self._update_daily_stats(current_price)
        else:
            logger.warning("cramer_sell_failed", error=result.error)

    def execute_trailing_stop_sell(
        self,
        _candles: "pd.DataFrame",
        current_price: Decimal,
        base_balance: Decimal,
    ) -> None:
        """
        Execute Cramer Mode sell from independent trailing stop trigger.

        This is NOT a mirror of normal bot action - it's Cramer Mode's own
        risk management exiting its position.

        Args:
            _candles: OHLCV data (unused, kept for interface consistency with normal bot)
            current_price: Current market price
            base_balance: Base balance to sell
        """
        if not self.client:
            return

        if self._disabled:
            return

        # Sell entire position (trailing stop = full exit)
        result = self.client.market_sell(
            self.config.trading_pair,
            base_balance,
        )

        if result.success:
            filled_price = result.filled_price or current_price

            # Get new balances
            new_quote = self.client.get_balance(self.config.quote_currency).available
            new_base = self.client.get_balance(self.config.base_currency).available

            # Verify balance consistency
            if not self._verify_balance_consistency(new_quote, new_base):
                return

            # Calculate realized P&L
            realized_pnl = self.position_service.calculate_realized_pnl(
                result.size, filled_price, result.fee,
                is_paper=True, bot_mode=BotMode.INVERTED
            )

            # Record trade
            self.db.record_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                realized_pnl=realized_pnl,
                symbol=self.config.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
                quote_balance_after=new_quote,
                base_balance_after=new_base,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=True, bot_mode=BotMode.INVERTED)

            # Deactivate trailing stop (critical: prevent repeated triggers)
            self.db.deactivate_trailing_stop(
                symbol=self.config.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
            )

            # Record to cooldown
            if self.trade_cooldown:
                self.trade_cooldown.record_trade("sell", filled_price)

            logger.info(
                "cramer_trailing_stop_sell_executed",
                size=str(result.size),
                price=str(filled_price),
                fee=str(result.fee),
                realized_pnl=str(realized_pnl),
            )

            # Update daily stats
            self._update_daily_stats(current_price)
        else:
            logger.warning("cramer_trailing_stop_sell_failed", error=result.error)

    def _verify_balance_consistency(
        self, expected_quote: Decimal, expected_base: Decimal
    ) -> bool:
        """
        Verify balance consistency after trade.

        If mismatch detected, disable Cramer Mode to preserve data integrity.

        This is a defensive check that re-fetches balances immediately after
        trade execution. While PaperTradingClient is synchronous and in-memory,
        this guards against potential future changes (async operations, caching)
        or bugs in balance calculation. The check is cheap and provides an
        important safety net for data integrity.

        Args:
            expected_quote: Expected quote balance
            expected_base: Expected base balance

        Returns:
            True if consistent, False if mismatch detected
        """
        actual_quote = self.client.get_balance(self.config.quote_currency).available
        actual_base = self.client.get_balance(self.config.base_currency).available

        if actual_quote != expected_quote or actual_base != expected_base:
            logger.error(
                "cramer_balance_mismatch",
                expected_quote=str(expected_quote),
                actual_quote=str(actual_quote),
                expected_base=str(expected_base),
                actual_base=str(actual_base),
            )
            self.disable(reason="balance_mismatch")
            return False

        return True

    def _update_daily_stats(self, current_price: Decimal) -> None:
        """Update Cramer Mode daily stats with current portfolio value."""
        ending_balance = self.get_portfolio_value()
        self.db.update_daily_stats(
            ending_balance=ending_balance,
            ending_price=current_price,
            is_paper=True,
            bot_mode=BotMode.INVERTED,
        )

    def initialize_daily_stats(self) -> None:
        """Initialize Cramer Mode daily stats at start of trading."""
        if not self.is_enabled:
            return

        portfolio_value = self.get_portfolio_value()
        # update_daily_stats creates the record if it doesn't exist
        self.db.update_daily_stats(
            starting_balance=portfolio_value,
            is_paper=True,
            bot_mode=BotMode.INVERTED,
        )
        logger.info(
            "cramer_daily_stats_initialized",
            starting_balance=str(portfolio_value),
        )

    def save_final_state(self, current_price: Decimal) -> None:
        """Save Cramer Mode final state on shutdown."""
        if not self.is_enabled:
            return

        portfolio_value = self.get_portfolio_value()
        self.db.update_daily_stats(
            ending_balance=portfolio_value,
            ending_price=current_price,
            is_paper=True,
            bot_mode=BotMode.INVERTED,
        )
        logger.info(
            "cramer_final_state_saved",
            portfolio_value=str(portfolio_value),
        )
