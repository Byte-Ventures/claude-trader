"""
Main daemon loop for the trading bot.

Orchestrates all components:
- Market data fetching
- Signal generation
- Order execution
- Safety system checks
- State persistence
"""

import signal
import time
from decimal import Decimal
from threading import Event
from typing import Optional, Union

import structlog

from config.settings import Settings, TradingMode
from src.api.exchange_factory import create_exchange_client, get_exchange_name
from src.api.exchange_protocol import ExchangeClient
from src.api.paper_client import PaperTradingClient
from src.notifications.telegram import TelegramNotifier
from src.safety.circuit_breaker import CircuitBreaker, BreakerLevel
from src.safety.kill_switch import KillSwitch
from src.safety.loss_limiter import LossLimiter
from src.safety.validator import OrderValidator, OrderRequest, ValidatorConfig
from src.state.database import Database
from src.strategy.signal_scorer import SignalScorer
from src.strategy.position_sizer import PositionSizer, PositionSizeConfig

logger = structlog.get_logger(__name__)


class TradingDaemon:
    """
    Main trading daemon that orchestrates all bot operations.

    Lifecycle:
    1. Initialize all components
    2. Run recovery if needed
    3. Start main trading loop
    4. Handle shutdown gracefully
    """

    def __init__(self, settings: Settings):
        """
        Initialize trading daemon with all components.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.shutdown_event = Event()
        self._running = False

        # Initialize database
        self.db = Database(settings.database_path)

        # Initialize exchange client using factory (Coinbase or Kraken)
        self.exchange_name = get_exchange_name(settings)
        self.real_client: ExchangeClient = create_exchange_client(settings)
        logger.info("exchange_client_initialized", exchange=self.exchange_name)

        # Validate trading pair against exchange
        self._validate_trading_pair(settings.trading_pair)

        # Parse base/quote currencies from trading pair
        pair_parts = settings.trading_pair.split("-")
        self._base_currency = pair_parts[0]
        self._quote_currency = pair_parts[1]

        # Initialize trading client (paper or live)
        self.client: Union[ExchangeClient, PaperTradingClient]
        if settings.is_paper_trading:
            # Use configured initial balances for paper trading
            self.client = PaperTradingClient(
                real_client=self.real_client,
                initial_quote=settings.paper_initial_quote,
                initial_base=settings.paper_initial_base,
                trading_pair=settings.trading_pair,
            )
            logger.info("using_paper_trading_client")
        else:
            self.client = self.real_client
            logger.info("using_live_trading_client")

        # Initialize Telegram notifier
        self.notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else "",
            chat_id=settings.telegram_chat_id or "",
            enabled=settings.telegram_enabled,
        )

        # Initialize safety systems
        self.kill_switch = KillSwitch(
            on_activate=lambda reason: self.notifier.notify_kill_switch(reason)
        )
        self.kill_switch.register_signal_handler()

        self.circuit_breaker = CircuitBreaker(
            on_trip=lambda level, reason: self.notifier.notify_circuit_breaker(
                level.name.lower(), reason
            )
        )

        self.loss_limiter = LossLimiter(
            on_limit_hit=lambda limit_type, percent: self.notifier.notify_loss_limit(
                limit_type, percent
            )
        )

        # Initialize order validator
        self.validator = OrderValidator(
            config=ValidatorConfig(
                min_trade_quote=10.0,
                max_position_percent=settings.max_position_percent,
            ),
            kill_switch=self.kill_switch,
            circuit_breaker=self.circuit_breaker,
            loss_limiter=self.loss_limiter,
        )

        # Initialize strategy components
        self.signal_scorer = SignalScorer(
            threshold=settings.signal_threshold,
            rsi_period=settings.rsi_period,
            rsi_oversold=settings.rsi_oversold,
            rsi_overbought=settings.rsi_overbought,
            macd_fast=settings.macd_fast,
            macd_slow=settings.macd_slow,
            macd_signal=settings.macd_signal,
            bollinger_period=settings.bollinger_period,
            bollinger_std=settings.bollinger_std,
            ema_fast=settings.ema_fast,
            ema_slow=settings.ema_slow,
            atr_period=settings.atr_period,
        )

        self.position_sizer = PositionSizer(
            config=PositionSizeConfig(
                max_position_percent=settings.position_size_percent,
                stop_loss_atr_multiplier=settings.stop_loss_atr_multiplier,
            ),
            atr_period=settings.atr_period,
            take_profit_atr_multiplier=settings.take_profit_atr_multiplier,
        )

        # Register shutdown handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _validate_trading_pair(self, trading_pair: str) -> None:
        """
        Validate trading pair format and against exchange.

        Args:
            trading_pair: Trading pair symbol (e.g., "BTC-USD", "BTC-EUR")

        Raises:
            ValueError: If trading pair format is invalid or not supported by exchange
        """
        # Validate format
        parts = trading_pair.split("-")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid trading pair format: '{trading_pair}'. "
                f"Expected format: BASE-QUOTE (e.g., BTC-USD, ETH-EUR)"
            )

        # Validate against exchange by attempting to fetch price
        try:
            price = self.real_client.get_current_price(trading_pair)
            logger.info(
                "trading_pair_validated",
                pair=trading_pair,
                exchange=self.exchange_name,
                current_price=str(price),
            )
        except Exception as e:
            raise ValueError(
                f"Trading pair '{trading_pair}' is not valid on {self.exchange_name}: {e}"
            ) from e

    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received", signal=signum)
        self.shutdown_event.set()

    def run(self) -> None:
        """Run the main trading loop."""
        self._running = True

        try:
            # Get initial portfolio value
            portfolio_value = self._get_portfolio_value()
            self.loss_limiter.set_starting_balance(portfolio_value)

            # Send startup notification
            mode = "PAPER" if self.settings.is_paper_trading else "LIVE"
            self.notifier.notify_startup(mode, portfolio_value, exchange=self.exchange_name)

            logger.info(
                "daemon_started",
                mode=mode,
                exchange=self.exchange_name,
                portfolio_value=str(portfolio_value),
                trading_pair=self.settings.trading_pair,
            )

            # Main loop
            while not self.shutdown_event.is_set():
                try:
                    self._trading_iteration()
                except Exception as e:
                    logger.error("trading_iteration_error", error=str(e))
                    self.notifier.notify_error(str(e), "Main trading loop")
                    # Continue running after non-fatal errors
                    time.sleep(60)
                    continue

                # Wait for next iteration
                self.shutdown_event.wait(self.settings.check_interval_seconds)

        except Exception as e:
            logger.critical("daemon_fatal_error", error=str(e))
            self.notifier.notify_error(str(e), "Fatal error - daemon stopping")
            raise

        finally:
            self._shutdown()

    def _trading_iteration(self) -> None:
        """Execute one trading iteration."""
        # Check kill switch first
        if self.kill_switch.is_active:
            logger.info("iteration_skipped", reason="kill_switch_active")
            return

        # Check circuit breaker
        if not self.circuit_breaker.can_trade:
            logger.info("iteration_skipped", reason="circuit_breaker_open")
            return

        # Check loss limits
        loss_status = self.loss_limiter.get_status()
        if not loss_status.can_trade:
            logger.info("iteration_skipped", reason="loss_limit_exceeded")
            return

        # Get market data
        try:
            current_price = self.client.get_current_price(self.settings.trading_pair)
            candles = self.client.get_candles(
                self.settings.trading_pair,
                granularity="ONE_HOUR",
                limit=100,
            )
        except Exception as e:
            logger.error("market_data_fetch_failed", error=str(e))
            self.circuit_breaker.record_api_failure()
            return

        self.circuit_breaker.record_api_success()

        # Update circuit breaker with price data
        self.circuit_breaker.check_price_movement(float(current_price))

        # Get current balances
        base_balance = self.client.get_balance(self._base_currency).available
        quote_balance = self.client.get_balance(self._quote_currency).available

        # Update validator with current state
        self.validator.update_balances(base_balance, quote_balance, current_price)

        # Calculate portfolio value for logging
        base_value = base_balance * current_price
        portfolio_value = quote_balance + base_value
        position_percent = float(base_value / portfolio_value * 100) if portfolio_value > 0 else 0

        # Calculate signal
        signal_result = self.signal_scorer.calculate_score(candles, current_price)

        logger.info(
            "trading_check",
            price=str(current_price),
            signal_score=signal_result.score,
            signal_action=signal_result.action,
            base_balance=str(base_balance),
            quote_balance=str(quote_balance),
            portfolio_value=str(portfolio_value),
            position_pct=f"{position_percent:.1f}%",
        )

        # Execute trade if signal is strong enough
        if signal_result.action == "hold":
            logger.info(
                "decision",
                action="hold",
                reason=f"signal_score={signal_result.score} below threshold",
            )
            return

        # Get safety multiplier
        safety_multiplier = self.validator.get_position_multiplier()

        if signal_result.action == "buy":
            if quote_balance > Decimal("10"):
                logger.info(
                    "decision",
                    action="buy",
                    signal_score=signal_result.score,
                    safety_multiplier=f"{safety_multiplier:.2f}",
                )
                self._execute_buy(
                    candles, current_price, quote_balance, base_balance,
                    signal_result.score, safety_multiplier
                )
            else:
                logger.info(
                    "decision",
                    action="skip_buy",
                    reason=f"insufficient_quote_balance ({quote_balance})",
                    signal_score=signal_result.score,
                )

        elif signal_result.action == "sell":
            if base_balance > Decimal("0.0001"):
                logger.info(
                    "decision",
                    action="sell",
                    signal_score=signal_result.score,
                    safety_multiplier=f"{safety_multiplier:.2f}",
                )
                self._execute_sell(
                    candles, current_price, base_balance,
                    signal_result.score, safety_multiplier
                )
            else:
                logger.info(
                    "decision",
                    action="skip_sell",
                    reason=f"insufficient_base_balance ({base_balance})",
                    signal_score=signal_result.score,
                )

    def _execute_buy(
        self,
        candles,
        current_price: Decimal,
        quote_balance: Decimal,
        base_balance: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """Execute a buy order."""
        # Calculate position size
        position = self.position_sizer.calculate_size(
            df=candles,
            current_price=current_price,
            quote_balance=quote_balance,
            base_balance=base_balance,
            signal_strength=signal_score,
            side="buy",
            safety_multiplier=safety_multiplier,
        )

        if position.size_quote < Decimal("10"):
            logger.info("buy_skipped", reason="position_too_small", size_quote=str(position.size_quote))
            return

        # Validate order
        order_request = OrderRequest(
            side="buy",
            size=position.size_base,
            order_type="market",
        )

        validation = self.validator.validate(order_request)
        if not validation.valid:
            logger.info("buy_rejected", reason=validation.reason)
            return

        # Execute order
        result = self.client.market_buy(
            self.settings.trading_pair,
            position.size_quote,
        )

        if result.success:
            # Record trade
            is_paper = self.settings.is_paper_trading
            filled_price = result.filled_price or current_price

            self.db.record_trade(
                side="buy",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                is_paper=is_paper,
            )

            # Update position tracking for cost basis
            self._update_position_after_buy(result.size, filled_price, result.fee)

            # Update loss limiter (no P&L on buy)
            self.loss_limiter.record_trade(
                realized_pnl=Decimal("0"),
                side="buy",
                size=result.size,
                price=filled_price,
            )

            # Send notification
            self.notifier.notify_trade(
                side="buy",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                is_paper=is_paper,
            )

            logger.info(
                "buy_executed",
                size=str(result.size),
                price=str(result.filled_price),
                fee=str(result.fee),
            )

            self.circuit_breaker.record_order_success()
        else:
            logger.error("buy_failed", error=result.error)
            self.circuit_breaker.record_order_failure()
            self.notifier.notify_order_failed("buy", position.size_base, result.error or "Unknown error")

    def _execute_sell(
        self,
        candles,
        current_price: Decimal,
        base_balance: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """Execute a sell order."""
        # For aggressive selling, sell entire position on strong signal
        if abs(signal_score) >= 80:
            size_base = base_balance
        else:
            # Calculate position size
            position = self.position_sizer.calculate_size(
                df=candles,
                current_price=current_price,
                quote_balance=Decimal("0"),
                base_balance=base_balance,
                signal_strength=signal_score,
                side="sell",
                safety_multiplier=safety_multiplier,
            )
            size_base = position.size_base

        if size_base < Decimal("0.0001"):
            logger.info("sell_skipped", reason="position_too_small", size_base=str(size_base))
            return

        # Validate order
        order_request = OrderRequest(
            side="sell",
            size=size_base,
            order_type="market",
        )

        validation = self.validator.validate(order_request)
        if not validation.valid:
            logger.info("sell_rejected", reason=validation.reason)
            return

        # Execute order
        result = self.client.market_sell(
            self.settings.trading_pair,
            size_base,
        )

        if result.success:
            is_paper = self.settings.is_paper_trading
            filled_price = result.filled_price or current_price

            # Calculate realized P&L based on average cost basis
            realized_pnl = self._calculate_realized_pnl(result.size, filled_price, result.fee)

            # Record trade
            self.db.record_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                realized_pnl=realized_pnl,
                is_paper=is_paper,
            )

            # Update loss limiter with actual PnL
            self.loss_limiter.record_trade(
                realized_pnl=realized_pnl,
                side="sell",
                size=result.size,
                price=filled_price,
            )

            # Send notification
            self.notifier.notify_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                is_paper=is_paper,
            )

            logger.info(
                "sell_executed",
                size=str(result.size),
                price=str(result.filled_price),
                fee=str(result.fee),
                realized_pnl=str(realized_pnl),
            )

            self.circuit_breaker.record_order_success()
        else:
            logger.error("sell_failed", error=result.error)
            self.circuit_breaker.record_order_failure()
            self.notifier.notify_order_failed("sell", size_base, result.error or "Unknown error")

    def _update_position_after_buy(self, size: Decimal, price: Decimal, fee: Decimal) -> None:
        """
        Update position with new buy, recalculating weighted average cost.

        Cost basis includes fees to accurately reflect break-even price.

        Args:
            size: Base currency amount bought (e.g., BTC)
            price: Price paid per unit
            fee: Trading fee paid (in quote currency)
        """
        current = self.db.get_current_position(self.settings.trading_pair)

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
            symbol=self.settings.trading_pair,
            quantity=new_qty,
            average_cost=new_avg_cost,
        )

        logger.debug(
            "position_updated_after_buy",
            old_qty=str(old_qty),
            new_qty=str(new_qty),
            new_avg_cost=str(new_avg_cost),
        )

    def _calculate_realized_pnl(self, size: Decimal, sell_price: Decimal, fee: Decimal) -> Decimal:
        """
        Calculate realized PnL for a sell based on average cost.

        Net PnL = (sell_price - avg_cost) * size - sell_fee

        Args:
            size: Base currency amount sold (e.g., BTC)
            sell_price: Price received per unit
            fee: Trading fee paid on sell (in quote currency)

        Returns:
            Realized profit/loss (positive = profit, negative = loss)
        """
        current = self.db.get_current_position(self.settings.trading_pair)

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
            symbol=self.settings.trading_pair,
            quantity=new_qty,
            average_cost=avg_cost,
        )

        logger.info(
            "realized_pnl_calculated",
            size=str(size),
            sell_price=str(sell_price),
            avg_cost=str(avg_cost),
            pnl=str(pnl),
            remaining_qty=str(new_qty),
        )

        return pnl

    def _get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in quote currency."""
        base_balance = self.client.get_balance(self._base_currency).available
        quote_balance = self.client.get_balance(self._quote_currency).available
        current_price = self.client.get_current_price(self.settings.trading_pair)

        base_value = base_balance * current_price
        return quote_balance + base_value

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False

        # Save final state
        try:
            portfolio_value = self._get_portfolio_value()
            self.db.update_daily_stats(ending_balance=portfolio_value)
        except Exception as e:
            logger.error("shutdown_state_save_failed", error=str(e))

        # Send shutdown notification
        self.notifier.notify_shutdown("Graceful shutdown")

        logger.info("daemon_stopped")
