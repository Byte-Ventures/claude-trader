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

        # Initialize trading client (paper or live)
        self.client: Union[ExchangeClient, PaperTradingClient]
        if settings.is_paper_trading:
            # Use configured initial balances for paper trading
            self.client = PaperTradingClient(
                real_client=self.real_client,
                initial_usd=settings.paper_initial_usd,
                initial_btc=settings.paper_initial_btc,
            )
            logger.info("using_paper_trading_client")
        else:
            self.client = self.real_client
            logger.info("using_live_trading_client")

        # Initialize Telegram notifier
        self.notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token or "",
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
                level.value, reason
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
                min_trade_usd=10.0,
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
            logger.debug("iteration_skipped", reason="kill_switch_active")
            return

        # Check circuit breaker
        if not self.circuit_breaker.can_trade:
            logger.debug("iteration_skipped", reason="circuit_breaker_open")
            return

        # Check loss limits
        loss_status = self.loss_limiter.get_status()
        if not loss_status.can_trade:
            logger.debug("iteration_skipped", reason="loss_limit_exceeded")
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
        btc_balance = self.client.get_balance("BTC").available
        usd_balance = self.client.get_balance("USD").available

        # Update validator with current state
        self.validator.update_balances(btc_balance, usd_balance, current_price)

        # Calculate signal
        signal_result = self.signal_scorer.calculate_score(candles, current_price)

        logger.debug(
            "signal_calculated",
            score=signal_result.score,
            action=signal_result.action,
            price=str(current_price),
        )

        # Execute trade if signal is strong enough
        if signal_result.action == "hold":
            return

        # Get safety multiplier
        safety_multiplier = self.validator.get_position_multiplier()

        if signal_result.action == "buy" and usd_balance > Decimal("10"):
            self._execute_buy(
                candles, current_price, usd_balance, btc_balance,
                signal_result.score, safety_multiplier
            )

        elif signal_result.action == "sell" and btc_balance > Decimal("0.0001"):
            self._execute_sell(
                candles, current_price, btc_balance,
                signal_result.score, safety_multiplier
            )

    def _execute_buy(
        self,
        candles,
        current_price: Decimal,
        usd_balance: Decimal,
        btc_balance: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """Execute a buy order."""
        # Calculate position size
        position = self.position_sizer.calculate_size(
            df=candles,
            current_price=current_price,
            usd_balance=usd_balance,
            btc_balance=btc_balance,
            signal_strength=signal_score,
            side="buy",
            safety_multiplier=safety_multiplier,
        )

        if position.size_usd < Decimal("10"):
            logger.debug("buy_skipped", reason="position_too_small")
            return

        # Validate order
        order_request = OrderRequest(
            side="buy",
            size=position.size_btc,
            order_type="market",
        )

        validation = self.validator.validate(order_request)
        if not validation.valid:
            logger.info("buy_rejected", reason=validation.reason)
            return

        # Execute order
        result = self.client.market_buy(
            self.settings.trading_pair,
            position.size_usd,
        )

        if result.success:
            # Record trade
            is_paper = self.settings.is_paper_trading
            self.db.record_trade(
                side="buy",
                size=result.size,
                price=result.filled_price or current_price,
                fee=result.fee,
                is_paper=is_paper,
            )

            # Update loss limiter
            self.loss_limiter.record_trade(
                realized_pnl=Decimal("0"),  # No P&L on buy
                side="buy",
                size=result.size,
                price=result.filled_price or current_price,
            )

            # Send notification
            self.notifier.notify_trade(
                side="buy",
                size=result.size,
                price=result.filled_price or current_price,
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
            self.notifier.notify_order_failed("buy", position.size_btc, result.error or "Unknown error")

    def _execute_sell(
        self,
        candles,
        current_price: Decimal,
        btc_balance: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """Execute a sell order."""
        # For aggressive selling, sell entire position on strong signal
        if abs(signal_score) >= 80:
            size_btc = btc_balance
        else:
            # Calculate position size
            position = self.position_sizer.calculate_size(
                df=candles,
                current_price=current_price,
                usd_balance=Decimal("0"),
                btc_balance=btc_balance,
                signal_strength=signal_score,
                side="sell",
                safety_multiplier=safety_multiplier,
            )
            size_btc = position.size_btc

        if size_btc < Decimal("0.0001"):
            logger.debug("sell_skipped", reason="position_too_small")
            return

        # Validate order
        order_request = OrderRequest(
            side="sell",
            size=size_btc,
            order_type="market",
        )

        validation = self.validator.validate(order_request)
        if not validation.valid:
            logger.info("sell_rejected", reason=validation.reason)
            return

        # Execute order
        result = self.client.market_sell(
            self.settings.trading_pair,
            size_btc,
        )

        if result.success:
            # Calculate realized P&L (simplified - should track cost basis)
            realized_pnl = Decimal("0")  # TODO: Implement proper P&L tracking

            # Record trade
            is_paper = self.settings.is_paper_trading
            self.db.record_trade(
                side="sell",
                size=result.size,
                price=result.filled_price or current_price,
                fee=result.fee,
                realized_pnl=realized_pnl,
                is_paper=is_paper,
            )

            # Update loss limiter
            self.loss_limiter.record_trade(
                realized_pnl=realized_pnl,
                side="sell",
                size=result.size,
                price=result.filled_price or current_price,
            )

            # Send notification
            self.notifier.notify_trade(
                side="sell",
                size=result.size,
                price=result.filled_price or current_price,
                fee=result.fee,
                is_paper=is_paper,
            )

            logger.info(
                "sell_executed",
                size=str(result.size),
                price=str(result.filled_price),
                fee=str(result.fee),
            )

            self.circuit_breaker.record_order_success()
        else:
            logger.error("sell_failed", error=result.error)
            self.circuit_breaker.record_order_failure()
            self.notifier.notify_order_failed("sell", size_btc, result.error or "Unknown error")

    def _get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in USD."""
        btc_balance = self.client.get_balance("BTC").available
        usd_balance = self.client.get_balance("USD").available
        current_price = self.client.get_current_price(self.settings.trading_pair)

        btc_value = btc_balance * current_price
        return usd_balance + btc_value

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
