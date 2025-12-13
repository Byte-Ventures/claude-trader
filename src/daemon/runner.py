"""
Main daemon loop for the trading bot.

Orchestrates all components:
- Market data fetching
- Signal generation
- Order execution
- Safety system checks
- State persistence
"""

import asyncio
import math
import signal
import time
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from threading import Event
from typing import Optional, Union

import structlog

from config.settings import Settings, TradingMode, VetoAction, request_reload, reload_pending, reload_settings
from src.api.exchange_factory import create_exchange_client, get_exchange_name
from src.api.exchange_protocol import ExchangeClient
from src.api.paper_client import PaperTradingClient
from src.notifications.telegram import TelegramNotifier
from src.safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, BreakerLevel
from src.safety.kill_switch import KillSwitch
from src.safety.loss_limiter import LossLimiter
from src.safety.validator import OrderValidator, OrderRequest, ValidatorConfig
from src.state.database import Database
from src.strategy.signal_scorer import SignalScorer
from src.strategy.position_sizer import PositionSizer, PositionSizeConfig
from src.strategy.regime import MarketRegime, RegimeConfig, RegimeAdjustments, get_cached_sentiment
from src.indicators.ema import get_ema_trend_from_values

logger = structlog.get_logger(__name__)

# Default timeout for async operations (AI reviews, sentiment fetch)
ASYNC_TIMEOUT_SECONDS = 120

# Interval for periodic stop protection checks (seconds)
STOP_PROTECTION_CHECK_INTERVAL_SECONDS = 300  # 5 minutes


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
        self._last_daily_report: Optional[date] = None
        self._last_weekly_report: Optional[date] = None
        self._last_monthly_report: Optional[date] = None
        self._last_volatility: str = "normal"  # For adaptive interval
        self._last_regime: str = "neutral"  # For regime change notifications
        self._last_hourly_analysis: Optional[datetime] = None  # For hourly market analysis
        self._pending_post_volatility_analysis: bool = False  # Trigger analysis when volatility calms
        self._last_stop_check: Optional[datetime] = None  # For periodic stop protection checks

        # Create persistent event loop for async operations (avoids repeated asyncio.run())
        self._loop = asyncio.new_event_loop()

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
            # Try to restore paper balance from database
            saved_balance = self.db.get_last_paper_balance(settings.trading_pair)
            if saved_balance:
                initial_quote, initial_base, _ = saved_balance
                logger.info(
                    "paper_balance_restored_from_db",
                    quote=str(initial_quote),
                    base=str(initial_base),
                )
            else:
                initial_quote = settings.paper_initial_quote
                initial_base = settings.paper_initial_base
                logger.info(
                    "paper_balance_using_config",
                    quote=str(initial_quote),
                    base=str(initial_base),
                )

            self.client = PaperTradingClient(
                real_client=self.real_client,
                initial_quote=float(initial_quote),
                initial_base=float(initial_base),
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
            db=self.db,
            is_paper=settings.is_paper_trading,
        )

        # Initialize multi-agent AI trade reviewer (optional, via OpenRouter)
        self.trade_reviewer = None
        if settings.ai_review_enabled and settings.openrouter_api_key:
            from src.ai.trade_reviewer import TradeReviewer
            self.trade_reviewer = TradeReviewer(
                api_key=settings.openrouter_api_key.get_secret_value(),
                db=self.db,
                reviewer_models=[
                    settings.reviewer_model_1,
                    settings.reviewer_model_2,
                    settings.reviewer_model_3,
                ],
                judge_model=settings.judge_model,
                veto_action=settings.veto_action,
                veto_threshold=settings.veto_threshold,
                position_reduction=settings.position_reduction,
                delay_minutes=settings.delay_minutes,
                interesting_hold_margin=settings.interesting_hold_margin,
                review_all=settings.ai_review_all,
                market_research_enabled=settings.market_research_enabled,
                ai_web_search_enabled=settings.ai_web_search_enabled,
                market_research_cache_minutes=settings.market_research_cache_minutes,
                candle_interval=settings.candle_interval,
            )
            logger.info(
                "multi_agent_trade_reviewer_initialized",
                reviewers=[settings.reviewer_model_1, settings.reviewer_model_2, settings.reviewer_model_3],
                judge=settings.judge_model,
                review_all=settings.ai_review_all,
            )
        else:
            logger.info("ai_trade_reviewer_disabled")

        # Hourly market analysis uses trade_reviewer (if AI review enabled)
        self._hourly_analysis_enabled = settings.hourly_analysis_enabled and self.trade_reviewer is not None
        if self._hourly_analysis_enabled:
            logger.info("hourly_market_analysis_enabled", uses_trade_reviewer=True)
        else:
            logger.info("hourly_market_analysis_disabled")

        # Initialize safety systems
        self.kill_switch = KillSwitch(
            on_activate=lambda reason: self.notifier.notify_kill_switch(reason)
        )
        self.kill_switch.register_signal_handler()

        self.circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                black_recovery_hours=settings.black_recovery_hours,
            ),
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
                min_trade_quote=100.0,
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

        # Initialize market regime detector
        self.market_regime = MarketRegime(
            config=RegimeConfig(
                enabled=settings.regime_adaptation_enabled,
                sentiment_enabled=settings.regime_sentiment_enabled,
                volatility_enabled=settings.regime_volatility_enabled,
                trend_enabled=settings.regime_trend_enabled,
                adjustment_scale=settings.regime_adjustment_scale,
            )
        )
        if settings.regime_adaptation_enabled:
            # Restore last regime from database
            last_regime = self.db.get_last_regime(is_paper=settings.is_paper_trading)
            if last_regime:
                self._last_regime = last_regime
                logger.info("regime_restored_from_db", regime=last_regime)
            logger.info("market_regime_initialized", scale=settings.regime_adjustment_scale)

        # AI recommendation state (for threshold adjustments from interesting holds)
        self._ai_recommendation: Optional[str] = None  # "accumulate", "reduce", "wait"
        self._ai_recommendation_confidence: float = 0.0
        self._ai_recommendation_time: Optional[datetime] = None
        self._ai_recommendation_ttl_minutes: int = settings.ai_recommendation_ttl_minutes

        # Register shutdown handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Register SIGUSR2 for config hot-reload
        signal.signal(signal.SIGUSR2, self._handle_reload_signal)
        logger.info("config_reload_signal_registered", signal="SIGUSR2")

    def _run_async_with_timeout(self, coro, timeout: int = ASYNC_TIMEOUT_SECONDS, default=None):
        """
        Run an async coroutine with timeout protection.

        Prevents daemon from hanging indefinitely if external APIs are slow.

        Args:
            coro: Async coroutine to execute
            timeout: Timeout in seconds (default: 120)
            default: Value to return on timeout (default: None)

        Returns:
            Coroutine result or default value on timeout
        """
        async def _with_timeout():
            task = asyncio.create_task(coro)
            try:
                return await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task  # Allow cleanup
                except asyncio.CancelledError:
                    pass
                logger.warning("async_operation_timeout", timeout=timeout)
                return default

        return self._loop.run_until_complete(_with_timeout())

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

    def _get_ai_threshold_adjustment(self, action: str) -> int:
        """
        Calculate threshold adjustment based on active AI recommendation.

        When the AI judge recommends "accumulate" or "reduce" on an interesting hold,
        this temporarily adjusts the threshold to make trading easier.

        Args:
            action: "buy" or "sell"

        Returns:
            Negative value to lower threshold (easier to trade), 0 otherwise.
            Decays linearly over TTL period.
        """
        if not self._ai_recommendation or not self._ai_recommendation_time:
            return 0

        # Check if recommendation has expired
        elapsed = (datetime.utcnow() - self._ai_recommendation_time).total_seconds() / 60
        if elapsed > self._ai_recommendation_ttl_minutes:
            # Clear expired recommendation
            self._ai_recommendation = None
            self._ai_recommendation_time = None
            return 0

        # Calculate decay factor (1.0 at start, 0.0 at TTL)
        decay = 1.0 - (elapsed / self._ai_recommendation_ttl_minutes)

        # Base adjustment of 15 points, scaled by confidence and decay
        base_adjustment = 15
        adjustment = int(base_adjustment * self._ai_recommendation_confidence * decay)

        if self._ai_recommendation == "accumulate" and action == "buy":
            return -adjustment  # Lower buy threshold (easier to buy)
        elif self._ai_recommendation == "reduce" and action == "sell":
            return -adjustment  # Lower sell threshold (easier to sell)

        return 0

    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received", signal=signum)
        self.shutdown_event.set()

    def _handle_reload_signal(self, signum: int, frame) -> None:
        """Handle SIGUSR2 signal for config reload."""
        logger.info("config_reload_signal_received")
        request_reload()

    def _reload_config(self) -> None:
        """
        Reload configuration from .env and update all components.

        Called from main loop when reload is requested.
        """
        try:
            new_settings, changes = reload_settings()

            if not changes:
                logger.info("config_reload_no_changes")
                return

            # Log what changed
            for field, (old, new) in changes.items():
                logger.info(
                    "config_value_changed",
                    field=field,
                    old_value=str(old),
                    new_value=str(new),
                )

            # Warn about structural changes that require restart
            restart_required = {"trading_mode", "exchange", "trading_pair", "database_path"}
            ignored_changes = restart_required & set(changes.keys())
            if ignored_changes:
                logger.warning(
                    "config_change_requires_restart",
                    fields=list(ignored_changes),
                )
                self.notifier.send_message(
                    f"Warning: {', '.join(ignored_changes)} changes require restart"
                )

            # Update settings reference
            self.settings = new_settings

            # Update SignalScorer
            self.signal_scorer.update_settings(
                threshold=new_settings.signal_threshold,
                rsi_period=new_settings.rsi_period,
                rsi_oversold=new_settings.rsi_oversold,
                rsi_overbought=new_settings.rsi_overbought,
                macd_fast=new_settings.macd_fast,
                macd_slow=new_settings.macd_slow,
                macd_signal=new_settings.macd_signal,
                bollinger_period=new_settings.bollinger_period,
                bollinger_std=new_settings.bollinger_std,
                ema_fast=new_settings.ema_fast,
                ema_slow=new_settings.ema_slow,
                atr_period=new_settings.atr_period,
            )

            # Update PositionSizer
            self.position_sizer.update_settings(
                max_position_percent=new_settings.position_size_percent,
                stop_loss_atr_multiplier=new_settings.stop_loss_atr_multiplier,
                take_profit_atr_multiplier=new_settings.take_profit_atr_multiplier,
                atr_period=new_settings.atr_period,
            )

            # Update OrderValidator
            self.validator.update_settings(
                max_position_percent=new_settings.max_position_percent,
            )

            # Update LossLimiter
            self.loss_limiter.update_settings(
                max_daily_loss_percent=new_settings.max_daily_loss_percent,
                max_hourly_loss_percent=new_settings.max_hourly_loss_percent,
            )

            # Update TradeReviewer (if enabled)
            if self.trade_reviewer:
                self.trade_reviewer.review_all = new_settings.ai_review_all
                self.trade_reviewer.veto_action = new_settings.veto_action
                self.trade_reviewer.veto_threshold = new_settings.veto_threshold
                self.trade_reviewer.position_reduction = new_settings.position_reduction
                self.trade_reviewer.delay_minutes = new_settings.delay_minutes
                self.trade_reviewer.interesting_hold_margin = new_settings.interesting_hold_margin
                self.trade_reviewer.candle_interval = new_settings.candle_interval

            # Update AI recommendation TTL
            self._ai_recommendation_ttl_minutes = new_settings.ai_recommendation_ttl_minutes

            logger.info(
                "config_reload_complete",
                changed_fields=list(changes.keys()),
            )

            # Notify via Telegram
            self.notifier.send_message(
                f"Config reloaded: {', '.join(changes.keys())} updated"
            )

        except Exception as e:
            logger.error("config_reload_failed", error=str(e))
            self.notifier.notify_error(str(e), "Config reload failed")

    def _get_adaptive_interval(self) -> int:
        """
        Get check interval based on current market volatility.

        Returns shorter intervals during high volatility (more opportunity/risk)
        and longer intervals during low volatility (save resources).
        """
        if not self.settings.adaptive_interval_enabled:
            return self.settings.check_interval_seconds

        interval_map = {
            "low": self.settings.interval_low_volatility,
            "normal": self.settings.interval_normal,
            "high": self.settings.interval_high_volatility,
            "extreme": self.settings.interval_extreme_volatility,
        }
        return interval_map.get(self._last_volatility, self.settings.check_interval_seconds)

    def run(self) -> None:
        """Run the main trading loop."""
        self._running = True

        try:
            # Get initial portfolio value and price
            portfolio_value = self._get_portfolio_value()
            starting_price = self.client.get_current_price(self.settings.trading_pair)
            self.loss_limiter.set_starting_balance(portfolio_value)

            # Record starting balance and price for daily stats
            self.db.update_daily_stats(
                starting_balance=portfolio_value,
                starting_price=starting_price,
                is_paper=self.settings.is_paper_trading,
            )

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

            # SAFETY CHECK: Verify positions have stop protection
            # This catches situations where the bot crashed after opening a position
            # but before creating the stop-loss
            self._verify_position_stop_protection()

            # Main loop
            while not self.shutdown_event.is_set():
                # Check for config reload request
                if reload_pending():
                    self._reload_config()

                # Check for performance reports
                self._check_daily_report()
                self._check_weekly_report()
                self._check_monthly_report()

                # Check for hourly market analysis
                self._check_hourly_analysis()

                # Periodic safety check: ensure positions have stop protection
                # This catches edge cases like manual position opens or recovery failures
                self._periodic_stop_protection_check()

                try:
                    self._trading_iteration()
                except Exception as e:
                    logger.error("trading_iteration_error", error=str(e))
                    self.notifier.notify_error(str(e), "Main trading loop")
                    # Continue running after non-fatal errors
                    time.sleep(60)
                    continue

                # Wait for next iteration (adaptive based on volatility)
                interval = self._get_adaptive_interval()
                logger.debug("next_check", interval=interval, volatility=self._last_volatility)
                self.shutdown_event.wait(interval)

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

        # Get market data and balances
        try:
            current_price = self.client.get_current_price(self.settings.trading_pair)
            candles = self.client.get_candles(
                self.settings.trading_pair,
                granularity=self.settings.candle_interval,
                limit=self.settings.candle_limit,
            )

            # Persist candles to rate_history (separate try/except - don't affect trading)
            if not candles.empty:
                try:
                    candle_dicts = candles.to_dict("records")
                    inserted = self.db.record_rates_bulk(
                        candles=candle_dicts,
                        symbol=self.settings.trading_pair,
                        exchange=self.exchange_name,
                        interval=self.settings.candle_interval,
                        is_paper=self.settings.is_paper_trading,
                    )
                    if inserted > 0:
                        logger.debug("rate_history_saved", count=inserted)
                except Exception as rate_err:
                    # Get sample timestamp for debugging (if available)
                    sample_ts = None
                    if candle_dicts:
                        sample_ts = str(candle_dicts[0].get("timestamp"))
                    logger.warning(
                        "rate_history_save_failed",
                        error=str(rate_err),
                        error_type=type(rate_err).__name__,
                        symbol=self.settings.trading_pair,
                        exchange=self.exchange_name,
                        candle_count=len(candles),
                        sample_timestamp=sample_ts,
                    )

            base_balance = self.client.get_balance(self._base_currency).available
            quote_balance = self.client.get_balance(self._quote_currency).available
        except Exception as e:
            logger.error("api_request_failed", error=str(e), error_type=type(e).__name__)
            self.circuit_breaker.record_api_failure()
            return

        self.circuit_breaker.record_api_success()

        # Check unrealized loss (early warning for underwater positions)
        try:
            position = self.db.get_current_position(
                self.settings.trading_pair,
                is_paper=self.settings.is_paper_trading
            )
            if position and position.get_quantity() > 0:
                avg_cost = position.get_average_cost()
                qty = position.get_quantity()
                unrealized_pnl = (current_price - avg_cost) * qty
                within_limit, combined_loss_pct = self.loss_limiter.check_limits_with_unrealized(
                    unrealized_pnl
                )
                if not within_limit:
                    logger.warning(
                        "iteration_skipped_unrealized_loss",
                        unrealized_pnl=str(unrealized_pnl),
                        combined_loss_pct=f"{combined_loss_pct:.1f}%",
                    )
                    self.notifier.notify_error(
                        f"Combined loss {combined_loss_pct:.1f}% exceeds daily limit",
                        "Unrealized loss warning"
                    )
                    return
        except Exception as e:
            # Don't block trading on unrealized check failure
            logger.debug("unrealized_loss_check_failed", error=str(e))

        # Update circuit breaker with price data
        self.circuit_breaker.check_price_movement(float(current_price))

        # Update validator with current state
        self.validator.update_balances(base_balance, quote_balance, current_price)

        # Check trailing stop before normal signal processing
        trailing_action = self._check_trailing_stop(current_price)
        if trailing_action == "sell" and base_balance > Decimal("0"):
            logger.info("trailing_stop_sell_triggered", base_balance=str(base_balance))
            # Execute sell for trailing stop (use full position, high priority)
            self._execute_sell(
                candles=candles,
                current_price=current_price,
                base_balance=base_balance,
                signal_score=80,  # Treat as strong sell signal
                safety_multiplier=1.0,  # No safety reduction for stops
            )
            return  # Exit iteration after trailing stop execution

        # Calculate portfolio value for logging
        base_value = base_balance * current_price
        portfolio_value = quote_balance + base_value
        position_percent = float(base_value / portfolio_value * 100) if portfolio_value > 0 else 0

        # Determine which trade directions are possible
        min_quote = Decimal(str(self.position_sizer.config.min_trade_quote))
        can_buy = quote_balance > min_quote and position_percent < self.position_sizer.config.max_position_percent
        can_sell = base_balance > Decimal("0.0001")

        # Calculate signal
        signal_result = self.signal_scorer.calculate_score(candles, current_price)

        # Log indicator values for debugging
        ind = signal_result.indicators
        logger.info(
            "indicators",
            rsi=f"{ind.rsi:.1f}" if ind.rsi else "N/A",
            macd_hist=f"{ind.macd_histogram:.2f}" if ind.macd_histogram else "N/A",
            bb_pct_b=f"{((float(current_price) - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)):.2f}" if ind.bb_upper and ind.bb_lower else "N/A",
            ema_gap=f"{((ind.ema_fast - ind.ema_slow) / ind.ema_slow * 100):.3f}%" if ind.ema_fast and ind.ema_slow else "N/A",
            volatility=ind.volatility,
        )

        # Track volatility for adaptive interval and post-volatility analysis
        if ind.volatility:
            old_volatility = self._last_volatility
            self._last_volatility = ind.volatility

            # Detect transition from high/extreme to normal/low
            if old_volatility in ("high", "extreme") and ind.volatility in ("normal", "low"):
                self._pending_post_volatility_analysis = True
                logger.info(
                    "post_volatility_analysis_pending",
                    from_volatility=old_volatility,
                    to_volatility=ind.volatility,
                )

        # Calculate market regime for strategy adaptation
        sentiment = None
        if self.settings.regime_sentiment_enabled:
            try:
                sentiment = self._run_async_with_timeout(get_cached_sentiment(), timeout=30)
            except Exception as e:
                logger.debug("sentiment_fetch_skipped", error=str(e))
        trend = get_ema_trend_from_values(ind.ema_fast, ind.ema_slow) if ind.ema_fast and ind.ema_slow else "neutral"

        regime = self.market_regime.calculate(
            sentiment=sentiment,
            volatility=ind.volatility or "normal",
            trend=trend,
            signal_action=signal_result.action,
        )

        # Notify on regime change
        if regime.regime_name != self._last_regime:
            logger.info(
                "regime_changed",
                old_regime=self._last_regime,
                new_regime=regime.regime_name,
                threshold_adj=regime.threshold_adjustment,
                position_mult=regime.position_multiplier,
            )

            # Extract component data for logging
            sentiment_value = None
            sentiment_category = None
            volatility_level = None
            trend_direction = None

            if "sentiment" in regime.components:
                sentiment_value = regime.components["sentiment"].get("value")
                sentiment_category = regime.components["sentiment"].get("category")
            if "volatility" in regime.components:
                volatility_level = regime.components["volatility"].get("level")
            if "trend" in regime.components:
                trend_direction = regime.components["trend"].get("direction")

            # Record to database
            self.db.record_regime_change(
                regime_name=regime.regime_name,
                threshold_adjustment=regime.threshold_adjustment,
                position_multiplier=regime.position_multiplier,
                sentiment_value=sentiment_value,
                sentiment_category=sentiment_category,
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                is_paper=self.settings.is_paper_trading,
            )

            # Send Telegram notification
            self.notifier.notify_regime_change(
                old_regime=self._last_regime,
                new_regime=regime.regime_name,
                threshold_adj=regime.threshold_adjustment,
                position_mult=regime.position_multiplier,
                components=regime.components,
            )

            # Save to dashboard notifications
            self.db.save_notification(
                type="regime_change",
                title=f"Regime: {self._last_regime} → {regime.regime_name}",
                message=f"Threshold adj: {regime.threshold_adjustment:+d}, Position mult: {regime.position_multiplier:.1f}x",
                is_paper=self.settings.is_paper_trading,
            )

            self._last_regime = regime.regime_name

        # Apply regime and AI threshold adjustments to determine effective action
        base_threshold = self.settings.signal_threshold + regime.threshold_adjustment
        ai_buy_adj = self._get_ai_threshold_adjustment("buy")
        ai_sell_adj = self._get_ai_threshold_adjustment("sell")

        effective_buy_threshold = base_threshold + ai_buy_adj
        effective_sell_threshold = base_threshold + ai_sell_adj

        if signal_result.score >= effective_buy_threshold:
            effective_action = "buy"
        elif signal_result.score <= -effective_sell_threshold:
            effective_action = "sell"
        else:
            effective_action = "hold"

        # Use the more restrictive threshold for logging (when no AI adjustment active)
        effective_threshold = base_threshold if ai_buy_adj == 0 and ai_sell_adj == 0 else effective_buy_threshold

        # Log when AI adjustment is active
        if ai_buy_adj != 0 or ai_sell_adj != 0:
            elapsed = (datetime.utcnow() - self._ai_recommendation_time).total_seconds() / 60
            decay_pct = (1.0 - elapsed / self._ai_recommendation_ttl_minutes) * 100
            logger.info(
                "ai_threshold_adjustment_active",
                recommendation=self._ai_recommendation,
                buy_adj=ai_buy_adj,
                sell_adj=ai_sell_adj,
                decay_remaining=f"{decay_pct:.0f}%",
                effective_buy_threshold=effective_buy_threshold,
                effective_sell_threshold=effective_sell_threshold,
            )

        logger.info(
            "trading_check",
            price=str(current_price),
            signal_score=signal_result.score,
            signal_action=effective_action,
            effective_threshold=effective_threshold,
            regime=regime.regime_name,
            regime_adj=regime.threshold_adjustment,
            regime_pos_mult=f"{regime.position_multiplier:.2f}",
            breakdown=signal_result.breakdown,
            base_balance=str(base_balance),
            quote_balance=str(quote_balance),
            portfolio_value=str(portfolio_value),
            position_pct=f"{position_percent:.1f}%",
        )

        # Persist state for dashboard
        ind = signal_result.indicators
        dashboard_state = {
            "timestamp": datetime.utcnow().isoformat(),
            "price": str(current_price),
            "signal": {
                "score": signal_result.score,
                "action": effective_action,
                "threshold": effective_threshold,
                "breakdown": signal_result.breakdown,
                "confidence": signal_result.confidence,
            },
            "indicators": {
                "rsi": ind.rsi,
                "macd_line": ind.macd_line,
                "macd_signal": ind.macd_signal,
                "macd_histogram": ind.macd_histogram,
                "bb_upper": float(ind.bb_upper) if ind.bb_upper else None,
                "bb_middle": float(ind.bb_middle) if ind.bb_middle else None,
                "bb_lower": float(ind.bb_lower) if ind.bb_lower else None,
                "ema_fast": float(ind.ema_fast) if ind.ema_fast else None,
                "ema_slow": float(ind.ema_slow) if ind.ema_slow else None,
                "atr": float(ind.atr) if ind.atr else None,
                "volatility": ind.volatility,
            },
            "portfolio": {
                "quote_balance": str(quote_balance),
                "base_balance": str(base_balance),
                "portfolio_value": str(portfolio_value),
                "position_percent": position_percent,
            },
            "regime": regime.regime_name,
            "safety": {
                "circuit_breaker": self.circuit_breaker.level.name,
                "can_trade": self.circuit_breaker.can_trade,
            },
            "trading_pair": self.settings.trading_pair,
            "is_paper": self.settings.is_paper_trading,
        }
        self.db.set_state("dashboard_state", dashboard_state)

        # Claude AI trade review (if enabled)
        claude_veto_multiplier = 1.0  # Default: no reduction
        if self.trade_reviewer:
            # Determine signal direction from score
            signal_direction = "buy" if signal_result.score > 0 else "sell" if signal_result.score < 0 else None

            # Check if this direction is tradeable
            direction_is_tradeable = (
                (signal_direction == "buy" and can_buy) or
                (signal_direction == "sell" and can_sell) or
                signal_direction is None  # Neutral score, no direction
            )

            if not direction_is_tradeable:
                logger.info(
                    "ai_review_skipped",
                    signal_direction=signal_direction,
                    reason="no_position" if signal_direction == "sell" else "fully_allocated",
                )
            else:
                should_review, review_type = self.trade_reviewer.should_review(
                    signal_result, self.settings.signal_threshold
                )

                if should_review:
                    try:
                        # Run async multi-agent review with timeout protection
                        review = self._run_async_with_timeout(
                            self.trade_reviewer.review_trade(
                                signal_result=signal_result,
                                current_price=current_price,
                                trading_pair=self.settings.trading_pair,
                                review_type=review_type,
                                position_percent=position_percent,
                            ),
                            timeout=ASYNC_TIMEOUT_SECONDS,
                        )
                        if review is None:
                            logger.warning("trade_review_timeout")
                            # Continue with trade on timeout (fail open)
                            raise TimeoutError("Trade review timed out")

                        # Always notify Telegram
                        self.notifier.notify_trade_review(review, review_type)

                        # Log multi-agent review summary
                        agent_summary = [
                            {
                                "model": r.model.split("/")[-1],
                                "stance": r.stance,
                                "approved": r.approved,
                                "confidence": f"{r.confidence:.2f}",
                                "summary": getattr(r, 'summary', '')[:50],
                            }
                            for r in review.reviews
                        ]
                        logger.info(
                            "multi_agent_trade_review",
                            review_type=review_type,
                            agents=agent_summary,
                            judge_decision="APPROVED" if review.judge_decision else "REJECTED",
                            judge_confidence=f"{review.judge_confidence:.2f}",
                            judge_recommendation=review.judge_recommendation,
                            judge_reasoning=review.judge_reasoning,
                            veto_action=review.final_veto_action or "none",
                        )

                        # Log full reasoning for each agent (separate entries for readability)
                        for r in review.reviews:
                            logger.debug(
                                "agent_full_reasoning",
                                model=r.model.split("/")[-1],
                                stance=r.stance,
                                reasoning=r.reasoning,
                            )

                        # Handle veto (only for actual trades, not interesting holds)
                        if review_type == "trade" and not review.judge_decision:
                            if review.final_veto_action == VetoAction.SKIP.value:
                                logger.info("trade_vetoed", reason=review.judge_reasoning)
                                return  # Skip this iteration
                            elif review.final_veto_action == VetoAction.REDUCE.value:
                                claude_veto_multiplier = self.settings.position_reduction
                                logger.info(
                                    "trade_reduced_by_review",
                                    multiplier=f"{claude_veto_multiplier:.2f}",
                                )
                            elif review.final_veto_action == VetoAction.DELAY.value:
                                logger.info(
                                    "trade_delayed_by_review",
                                    delay_minutes=self.settings.delay_minutes,
                                )
                                return  # Skip this iteration (user can check Telegram)
                            # VetoAction.INFO: log but proceed with trade

                        # For interesting holds, store recommendation for threshold adjustment
                        if review_type == "interesting_hold":
                            if review.judge_recommendation in ("accumulate", "reduce"):
                                self._ai_recommendation = review.judge_recommendation
                                self._ai_recommendation_confidence = review.judge_confidence
                                self._ai_recommendation_time = datetime.utcnow()
                                logger.info(
                                    "ai_recommendation_stored",
                                    recommendation=review.judge_recommendation,
                                    confidence=f"{review.judge_confidence:.2f}",
                                    ttl_minutes=self._ai_recommendation_ttl_minutes,
                                )
                            return

                    except Exception as e:
                        logger.error("claude_review_failed", error=str(e))
                        # Continue with trade on review failure (fail open)

        # Execute trade if signal is strong enough (using regime-adjusted threshold)
        if effective_action == "hold":
            logger.info(
                "decision",
                action="hold",
                reason=f"|score|={abs(signal_result.score)} < effective_threshold={effective_threshold}",
                regime=regime.regime_name,
            )
            return

        # Get safety multiplier (including Claude veto and regime adjustments)
        safety_multiplier = self.validator.get_position_multiplier() * claude_veto_multiplier * regime.position_multiplier

        if effective_action == "buy":
            if can_buy:
                logger.info(
                    "decision",
                    action="buy",
                    signal_score=signal_result.score,
                    effective_threshold=effective_threshold,
                    regime=regime.regime_name,
                    safety_multiplier=f"{safety_multiplier:.2f}",
                )
                self._execute_buy(
                    candles, current_price, quote_balance, base_balance,
                    signal_result.score, safety_multiplier,
                    rsi_value=signal_result.indicators.rsi,
                )
            else:
                logger.info(
                    "decision",
                    action="skip_buy",
                    reason=f"insufficient_balance_or_position_limit (quote={quote_balance}, position={position_percent:.1f}%)",
                    signal_score=signal_result.score,
                )

        elif effective_action == "sell":
            if can_sell:
                logger.info(
                    "decision",
                    action="sell",
                    signal_score=signal_result.score,
                    effective_threshold=effective_threshold,
                    regime=regime.regime_name,
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
        rsi_value: Optional[float] = None,
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

            # Get balance after trade for snapshot
            new_base_balance = self.client.get_balance(self._base_currency).available
            new_quote_balance = self.client.get_balance(self._quote_currency).available

            self.db.record_trade(
                side="buy",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                symbol=self.settings.trading_pair,
                is_paper=is_paper,
                quote_balance_after=new_quote_balance,
                base_balance_after=new_base_balance,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=is_paper)

            # Update position tracking and create stop protection
            # CRITICAL: If stop creation fails, immediately close position (fail-safe)
            try:
                new_avg_cost = self._update_position_after_buy(result.size, filled_price, result.fee, is_paper)
                self._create_trailing_stop(filled_price, candles, is_paper, avg_cost=new_avg_cost)
            except Exception as stop_error:
                # FAIL-SAFE: Cannot protect position, must close immediately
                logger.critical(
                    "stop_creation_failed_emergency_close",
                    error=str(stop_error),
                    action="closing_unprotected_position",
                )
                self.notifier.send_alert(
                    f"🚨 EMERGENCY: Stop creation failed, closing position immediately: {stop_error}"
                )
                # Emergency sell the position we just bought
                emergency_result = self.client.market_sell(
                    self.settings.trading_pair,
                    result.size,
                )
                if emergency_result.success:
                    logger.warning(
                        "emergency_position_closed",
                        size=str(result.size),
                        buy_price=str(filled_price),
                        sell_price=str(emergency_result.filled_price),
                    )
                    self.notifier.send_alert(
                        f"✅ Emergency close successful: sold {result.size} at {emergency_result.filled_price}"
                    )
                else:
                    # Double failure - halt trading
                    logger.critical(
                        "emergency_close_failed_halting",
                        error=emergency_result.error,
                    )
                    self.notifier.send_alert(
                        f"🔴 CRITICAL: Emergency close FAILED. Halting trading. Manual intervention required!"
                    )
                    self.kill_switch.activate(f"Emergency close failed: {emergency_result.error}")
                return  # Don't continue with normal post-buy flow

            # Update loss limiter (fee is a realized loss on buy)
            self.loss_limiter.record_trade(
                realized_pnl=-result.fee,  # Fees count as realized loss
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

            # Record oversold buy for crash protection rate limiting
            if rsi_value is not None and rsi_value < 25:
                self.signal_scorer.record_oversold_buy()

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

            # Deactivate any trailing stop since position is being sold
            self.db.deactivate_trailing_stop(
                symbol=self.settings.trading_pair,
                is_paper=is_paper,
            )

            # Calculate realized P&L based on average cost basis
            realized_pnl = self._calculate_realized_pnl(result.size, filled_price, result.fee, is_paper)

            # Get balance after trade for snapshot
            new_base_balance = self.client.get_balance(self._base_currency).available
            new_quote_balance = self.client.get_balance(self._quote_currency).available

            # Record trade
            self.db.record_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                realized_pnl=realized_pnl,
                symbol=self.settings.trading_pair,
                is_paper=is_paper,
                quote_balance_after=new_quote_balance,
                base_balance_after=new_base_balance,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=is_paper)

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

    def _update_position_after_buy(
        self, size: Decimal, price: Decimal, fee: Decimal, is_paper: bool = False
    ) -> Decimal:
        """
        Update position with new buy, recalculating weighted average cost.

        Cost basis includes fees to accurately reflect break-even price.

        Args:
            size: Base currency amount bought (e.g., BTC)
            price: Price paid per unit
            fee: Trading fee paid (in quote currency)
            is_paper: Whether this is a paper trade

        Returns:
            The new weighted average cost for the position
        """
        current = self.db.get_current_position(self.settings.trading_pair, is_paper=is_paper)

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
            is_paper=is_paper,
        )

        logger.debug(
            "position_updated_after_buy",
            old_qty=str(old_qty),
            new_qty=str(new_qty),
            new_avg_cost=str(new_avg_cost),
        )

        return new_avg_cost

    def _calculate_realized_pnl(
        self, size: Decimal, sell_price: Decimal, fee: Decimal, is_paper: bool = False
    ) -> Decimal:
        """
        Calculate realized PnL for a sell based on average cost.

        Net PnL = (sell_price - avg_cost) * size - sell_fee

        Args:
            size: Base currency amount sold (e.g., BTC)
            sell_price: Price received per unit
            fee: Trading fee paid on sell (in quote currency)
            is_paper: Whether this is a paper trade

        Returns:
            Realized profit/loss (positive = profit, negative = loss)
        """
        current = self.db.get_current_position(self.settings.trading_pair, is_paper=is_paper)

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
            is_paper=is_paper,
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

    def _check_daily_report(self) -> None:
        """Check if we should generate daily performance report."""
        today = date.today()

        # Only report once per day
        if self._last_daily_report == today:
            return

        # Get yesterday's stats (if exists)
        from datetime import timedelta
        yesterday = today - timedelta(days=1)
        stats = self.db.get_daily_stats(yesterday, is_paper=self.settings.is_paper_trading)

        if stats and stats.starting_balance and stats.ending_balance:
            try:
                starting_balance = Decimal(stats.starting_balance)
                ending_balance = Decimal(stats.ending_balance)
                starting_price = Decimal(stats.starting_price) if stats.starting_price else None
                ending_price = Decimal(stats.ending_price) if stats.ending_price else None

                # Calculate portfolio return
                if starting_balance > 0:
                    portfolio_return = float((ending_balance - starting_balance) / starting_balance * 100)
                else:
                    portfolio_return = 0.0

                # Calculate BTC return (buy-and-hold benchmark)
                btc_return = 0.0
                if starting_price and ending_price and starting_price > 0:
                    btc_return = float((ending_price - starting_price) / starting_price * 100)

                # Calculate alpha (outperformance vs buy-and-hold)
                alpha = portfolio_return - btc_return

                logger.info(
                    "daily_performance",
                    date=str(yesterday),
                    portfolio_return=f"{portfolio_return:+.2f}%",
                    btc_return=f"{btc_return:+.2f}%",
                    alpha=f"{alpha:+.2f}%",
                    starting_balance=str(starting_balance),
                    ending_balance=str(ending_balance),
                    trades=stats.total_trades or 0,
                )

                # Send Telegram notification
                self._send_daily_report(
                    yesterday,
                    portfolio_return,
                    btc_return,
                    alpha,
                    starting_balance,
                    ending_balance,
                    stats.total_trades or 0,
                )

            except Exception as e:
                logger.error("daily_report_failed", error=str(e))

        self._last_daily_report = today

    def _send_daily_report(
        self,
        report_date: date,
        portfolio_return: float,
        btc_return: float,
        alpha: float,
        starting_balance: Decimal,
        ending_balance: Decimal,
        trades: int,
    ) -> None:
        """Send daily performance report via Telegram."""
        # Determine performance emoji
        if alpha > 1:
            perf_emoji = "🚀"  # Beating BTC significantly
        elif alpha > 0:
            perf_emoji = "✅"  # Beating BTC
        elif alpha > -1:
            perf_emoji = "➖"  # Roughly matching
        else:
            perf_emoji = "📉"  # Underperforming

        mode = "PAPER" if self.settings.is_paper_trading else "LIVE"
        pnl = ending_balance - starting_balance

        message = (
            f"📊 <b>Daily Report</b> ({mode})\n"
            f"Date: {report_date}\n\n"
            f"<b>Portfolio</b>: {portfolio_return:+.2f}%\n"
            f"<b>BTC (HODL)</b>: {btc_return:+.2f}%\n"
            f"{perf_emoji} <b>Alpha</b>: {alpha:+.2f}%\n\n"
            f"P&L: €{pnl:+,.2f}\n"
            f"Balance: €{ending_balance:,.2f}\n"
            f"Trades: {trades}"
        )

        self.notifier.send_message_sync(message)

    def _check_weekly_report(self) -> None:
        """Check if we should generate weekly performance report (on Mondays)."""
        from datetime import timedelta

        today = date.today()

        # Only on Mondays
        if today.weekday() != 0:
            return

        # Only report once per week
        if self._last_weekly_report and (today - self._last_weekly_report).days < 7:
            return

        # Get last week's date range (Monday to Sunday)
        last_monday = today - timedelta(days=7)
        last_sunday = today - timedelta(days=1)

        self._generate_period_report("Weekly", last_monday, last_sunday)
        self._last_weekly_report = today

    def _check_monthly_report(self) -> None:
        """Check if we should generate monthly performance report (on 1st of month)."""
        from datetime import timedelta

        today = date.today()

        # Only on 1st of month
        if today.day != 1:
            return

        # Only report once per month
        if self._last_monthly_report and self._last_monthly_report.month == today.month:
            return

        # Get last month's date range
        last_day_prev_month = today - timedelta(days=1)
        first_day_prev_month = last_day_prev_month.replace(day=1)

        self._generate_period_report("Monthly", first_day_prev_month, last_day_prev_month)
        self._last_monthly_report = today

    def _generate_period_report(self, period: str, start_date: date, end_date: date) -> None:
        """Generate and send a performance report for a date range."""
        stats_list = self.db.get_daily_stats_range(
            start_date, end_date, is_paper=self.settings.is_paper_trading
        )

        if not stats_list:
            logger.debug(f"no_stats_for_{period.lower()}_report", start=str(start_date), end=str(end_date))
            return

        try:
            # Get first and last stats with valid data
            first_stats = None
            last_stats = None
            total_trades = 0

            for s in stats_list:
                if s.starting_balance and s.starting_price:
                    if first_stats is None:
                        first_stats = s
                if s.ending_balance and s.ending_price:
                    last_stats = s
                total_trades += s.total_trades or 0

            if not first_stats or not last_stats:
                logger.debug(f"incomplete_stats_for_{period.lower()}_report")
                return

            starting_balance = Decimal(first_stats.starting_balance)
            ending_balance = Decimal(last_stats.ending_balance)
            starting_price = Decimal(first_stats.starting_price)
            ending_price = Decimal(last_stats.ending_price)

            # Calculate returns
            portfolio_return = 0.0
            if starting_balance > 0:
                portfolio_return = float((ending_balance - starting_balance) / starting_balance * 100)

            btc_return = 0.0
            if starting_price > 0:
                btc_return = float((ending_price - starting_price) / starting_price * 100)

            alpha = portfolio_return - btc_return

            logger.info(
                f"{period.lower()}_performance",
                period=f"{start_date} to {end_date}",
                portfolio_return=f"{portfolio_return:+.2f}%",
                btc_return=f"{btc_return:+.2f}%",
                alpha=f"{alpha:+.2f}%",
                trades=total_trades,
            )

            # Send Telegram notification
            self._send_period_report(
                period,
                start_date,
                end_date,
                portfolio_return,
                btc_return,
                alpha,
                starting_balance,
                ending_balance,
                total_trades,
            )

        except Exception as e:
            logger.error(f"{period.lower()}_report_failed", error=str(e))

    def _send_period_report(
        self,
        period: str,
        start_date: date,
        end_date: date,
        portfolio_return: float,
        btc_return: float,
        alpha: float,
        starting_balance: Decimal,
        ending_balance: Decimal,
        trades: int,
    ) -> None:
        """Send weekly/monthly performance report via Telegram."""
        # Determine performance emoji
        if alpha > 2:
            perf_emoji = "🚀"
        elif alpha > 0:
            perf_emoji = "✅"
        elif alpha > -2:
            perf_emoji = "➖"
        else:
            perf_emoji = "📉"

        mode = "PAPER" if self.settings.is_paper_trading else "LIVE"
        pnl = ending_balance - starting_balance

        # Period-specific emoji
        period_emoji = "📅" if period == "Weekly" else "📆"

        message = (
            f"{period_emoji} <b>{period} Report</b> ({mode})\n"
            f"{start_date} → {end_date}\n\n"
            f"<b>Portfolio</b>: {portfolio_return:+.2f}%\n"
            f"<b>BTC (HODL)</b>: {btc_return:+.2f}%\n"
            f"{perf_emoji} <b>Alpha</b>: {alpha:+.2f}%\n\n"
            f"P&L: €{pnl:+,.2f}\n"
            f"Balance: €{ending_balance:,.2f}\n"
            f"Trades: {trades}"
        )

        self.notifier.send_message_sync(message)

    def _check_hourly_analysis(self) -> None:
        """Run hourly market analysis during volatile conditions or post-volatility."""
        # Skip if hourly analysis not enabled
        if not self._hourly_analysis_enabled:
            return

        now = datetime.now()
        is_post_volatility = self._pending_post_volatility_analysis

        # Check if we should run analysis
        should_run = False
        analysis_reason = ""

        if is_post_volatility:
            # Post-volatility analysis: run immediately (don't wait for hourly cooldown)
            should_run = True
            analysis_reason = "post_volatility"
            self._pending_post_volatility_analysis = False
        elif self._last_volatility in ("high", "extreme"):
            # During high/extreme volatility: run once per hour
            if self._last_hourly_analysis:
                elapsed = (now - self._last_hourly_analysis).total_seconds()
                if elapsed >= 3600:
                    should_run = True
                    analysis_reason = "hourly_volatile"
            else:
                should_run = True
                analysis_reason = "hourly_volatile"

        if not should_run:
            return

        # Run analysis
        try:
            # Get current market data for analysis
            current_price = self.client.get_current_price(self.settings.trading_pair)
            candles = self.client.get_candles(
                self.settings.trading_pair,
                granularity=self.settings.candle_interval,
                limit=self.settings.candle_limit,
            )

            # Calculate indicators
            signal_result = self.signal_scorer.calculate_score(candles, current_price)
            indicators = signal_result.indicators

            # Get sentiment
            sentiment_value = None
            sentiment_class = "Unknown"
            if self.settings.regime_sentiment_enabled:
                try:
                    sentiment_result = self._run_async_with_timeout(get_cached_sentiment(), timeout=30)
                    if sentiment_result and sentiment_result.value:
                        sentiment_value = sentiment_result.value
                        sentiment_class = sentiment_result.classification or "Unknown"
                except Exception as e:
                    logger.debug("sentiment_fetch_skipped", error=str(e))

            # Calculate price changes from candles
            price_change_1h = None
            price_change_24h = None
            if len(candles) >= 2:
                prev_close = candles["close"].iloc[-2]
                curr_close = candles["close"].iloc[-1]
                if prev_close > 0:
                    price_change_1h = ((curr_close - prev_close) / prev_close) * 100
            if len(candles) >= 24:
                prev_24h = candles["close"].iloc[-24]
                curr_close = candles["close"].iloc[-1]
                if prev_24h > 0:
                    price_change_24h = ((curr_close - prev_24h) / prev_24h) * 100

            # Build indicators dict for analyze_market
            indicators_dict = {
                "rsi": indicators.rsi,
                "macd_histogram": indicators.macd_histogram,
                "bb_percent_b": None,
                "ema_gap": None,
            }
            # Calculate Bollinger %B
            if indicators.bb_lower and indicators.bb_upper:
                bb_range = indicators.bb_upper - indicators.bb_lower
                if bb_range > 0:
                    indicators_dict["bb_percent_b"] = round(
                        (float(current_price) - indicators.bb_lower) / bb_range, 2
                    )
            # Calculate EMA gap
            if indicators.ema_slow and indicators.ema_slow > 0:
                indicators_dict["ema_gap"] = round(
                    ((float(current_price) - indicators.ema_slow) / indicators.ema_slow) * 100, 2
                )

            # Run multi-agent market analysis with timeout protection
            review = self._run_async_with_timeout(
                self.trade_reviewer.analyze_market(
                    indicators=indicators_dict,
                    current_price=current_price,
                    fear_greed=sentiment_value or 50,
                    fear_greed_class=sentiment_class,
                    regime=self._last_regime,
                    volatility=self._last_volatility,
                    price_change_1h=price_change_1h,
                    price_change_24h=price_change_24h,
                ),
                timeout=ASYNC_TIMEOUT_SECONDS,
            )
            if review is None:
                logger.warning("market_analysis_timeout")
                return  # Skip this analysis cycle

            # Log multi-agent analysis summary
            agent_summary = [
                {
                    "model": r.model.split("/")[-1],
                    "stance": r.stance,
                    "outlook": r.sentiment,
                    "confidence": f"{r.confidence:.2f}",
                    "summary": getattr(r, 'summary', '')[:50],
                }
                for r in review.reviews
            ]
            logger.info(
                "hourly_market_analysis",
                reason=analysis_reason,
                agents=agent_summary,
                judge_confidence=f"{review.judge_confidence:.2f}",
                judge_recommendation=review.judge_recommendation,
                judge_reasoning=review.judge_reasoning,
            )

            # Log full reasoning for each agent at debug level
            for r in review.reviews:
                logger.debug(
                    "market_analysis_agent_reasoning",
                    model=r.model.split("/")[-1],
                    stance=r.stance,
                    reasoning=r.reasoning,
                )

            # Send Telegram notification
            self.notifier.notify_market_analysis(
                review=review,
                indicators=indicators,
                volatility=self._last_volatility,
                fear_greed=sentiment_value or 50,
                fear_greed_class=sentiment_class,
                current_price=current_price,
                analysis_reason=analysis_reason,
            )

        except Exception as e:
            logger.error("hourly_analysis_failed", error=str(e))

        # Update timestamp regardless of success/failure to prevent spam on errors
        self._last_hourly_analysis = now

    def _create_trailing_stop(
        self,
        entry_price: Decimal,
        candles,  # pd.DataFrame with OHLCV data
        is_paper: bool = False,
        *,  # Force keyword-only for avg_cost
        avg_cost: Decimal,
    ) -> None:
        """Create or update trailing stop for a position.

        Args:
            entry_price: The price at which the buy was executed (for logging)
            candles: pandas DataFrame with OHLCV data for ATR calculation
            is_paper: Whether this is paper trading
            avg_cost: REQUIRED - Weighted average cost for hard stop calculation.
                     Must be passed from caller to avoid race conditions.
                     Do NOT query DB here - caller has the authoritative value.
        """
        from src.indicators.atr import calculate_atr

        try:
            # Calculate ATR for trailing stop distance
            high = candles["high"].astype(float)
            low = candles["low"].astype(float)
            close = candles["close"].astype(float)
            atr_result = calculate_atr(high, low, close, period=self.settings.atr_period)
            atr_value = atr_result.atr.iloc[-1]

            # Validate ATR before using
            if math.isnan(atr_value) or atr_value <= 0:
                logger.error("trailing_stop_atr_invalid", atr_value=atr_value)
                return

            atr = Decimal(str(atr_value))

            # Trailing activates at 1 ATR profit from average cost
            activation = avg_cost + atr
            # Trailing distance is based on config multiplier
            distance = atr * Decimal(str(self.settings.trailing_stop_atr_multiplier))

            # Hard stop: based on average cost, not latest entry
            hard_stop = avg_cost - (atr * Decimal(str(self.settings.stop_loss_atr_multiplier)))

            # Check if we're DCA'ing (stop already exists)
            # If so, UPDATE the existing stop in-place to avoid any window
            # where the position is unprotected during the transition
            existing_stop = self.db.get_active_trailing_stop(
                symbol=self.settings.trading_pair, is_paper=is_paper
            )

            if existing_stop:
                # DCA: Update existing stop without deactivating
                self.db.update_trailing_stop_for_dca(
                    symbol=self.settings.trading_pair,
                    entry_price=avg_cost,
                    trailing_activation=activation,
                    trailing_distance=distance,
                    hard_stop=hard_stop,
                    is_paper=is_paper,
                )
                logger.info(
                    "trailing_stop_updated_dca",
                    entry_price=str(entry_price),
                    avg_cost=str(avg_cost),
                    activation=str(activation),
                    distance=str(distance),
                    hard_stop=str(hard_stop),
                )
            else:
                # First buy: Create new stop
                self.db.create_trailing_stop(
                    symbol=self.settings.trading_pair,
                    side="buy",
                    entry_price=avg_cost,
                    trailing_activation=activation,
                    trailing_distance=distance,
                    is_paper=is_paper,
                    hard_stop=hard_stop,
                )
                logger.info(
                    "trailing_stop_created",
                    entry_price=str(entry_price),
                    avg_cost=str(avg_cost),
                    activation=str(activation),
                    distance=str(distance),
                    hard_stop=str(hard_stop),
                )
        except Exception as e:
            # CRITICAL: Re-raise to allow caller to handle (emergency close position)
            logger.error("trailing_stop_creation_failed", error=str(e))
            raise

    def _check_trailing_stop(self, current_price: Decimal) -> Optional[str]:
        """
        Check and update trailing stop, return action if stop triggered.

        Priority order (for buy positions):
        1. Hard stop (emergency capital protection, always active, never moves)
        2. Trailing stop activation (activates at 1 ATR profit above avg cost)
        3. Trailing stop update (follows price up, locks in gains)
        4. Trailing stop trigger (price drops to stop level, locks profit)

        The entry_price stored in trailing_stops is the weighted average cost,
        not the individual entry price, ensuring correct calculations for DCA.

        Returns:
            "sell" if trailing stop or hard stop triggered, None otherwise
        """
        is_paper = self.settings.is_paper_trading
        ts = self.db.get_active_trailing_stop(
            symbol=self.settings.trading_pair,
            is_paper=is_paper,
        )

        if not ts:
            return None

        entry_price = ts.get_entry_price()
        activation = ts.get_trailing_activation()
        distance = ts.get_trailing_distance()
        current_stop = ts.get_trailing_stop()
        hard_stop = ts.get_hard_stop()

        # For buy positions: check hard stop first, then trailing stop logic
        if ts.side == "buy":
            # CHECK HARD STOP FIRST (always active, never moves)
            # This is emergency capital protection - triggers before trailing can activate
            if hard_stop is not None and current_price <= hard_stop:
                # Determine if this was an emergency exit (trailing never activated)
                # or if trailing was active but hard stop was hit anyway (shouldn't happen normally)
                trailing_was_active = current_stop is not None
                exit_type = "emergency_exit" if not trailing_was_active else "hard_stop_below_trailing"
                # Calculate loss percentage
                # Use Decimal.quantize() for proper Decimal arithmetic (not round() which expects float)
                if entry_price <= 0:
                    # CRITICAL: Invalid entry_price indicates data corruption - halt trading
                    logger.critical(
                        "invalid_entry_price_halting_trading",
                        entry_price=str(entry_price),
                        context="hard_stop_check",
                    )
                    self.notifier.send_alert(
                        f"🔴 CRITICAL: Invalid entry_price ({entry_price}) detected. Halting trading!"
                    )
                    self.kill_switch.activate(f"Data corruption: entry_price={entry_price}")
                    loss_pct = Decimal("0")  # Proceed with sell to close position
                else:
                    loss_pct = ((entry_price - current_price) / entry_price * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                logger.warning(
                    "hard_stop_triggered",
                    exit_type=exit_type,
                    current_price=str(current_price),
                    hard_stop=str(hard_stop),
                    entry_price=str(entry_price),
                    trailing_was_active=trailing_was_active,
                    loss_percent=str(loss_pct),
                )
                self.db.deactivate_trailing_stop(
                    symbol=self.settings.trading_pair,
                    is_paper=is_paper,
                )
                return "sell"

            # Check if trailing stop is now activated
            if current_stop is None and activation and current_price >= activation:
                # Activate: set initial stop at current_price - distance
                # SAFETY: Never set trailing stop below hard stop
                new_stop = current_price - distance
                if hard_stop is not None and new_stop < hard_stop:
                    new_stop = hard_stop
                    logger.warning(
                        "trailing_stop_floored_at_hard_stop",
                        calculated_stop=str(current_price - distance),
                        floored_to=str(hard_stop),
                    )
                self.db.update_trailing_stop(ts.id, new_stop_level=new_stop)
                logger.info(
                    "trailing_stop_activated",
                    current_price=str(current_price),
                    stop_level=str(new_stop),
                )
                return None

            # If activated, update stop if price moved up
            if current_stop is not None and distance:
                potential_new_stop = current_price - distance
                # SAFETY: Never let trailing stop go below hard stop
                if hard_stop is not None and potential_new_stop < hard_stop:
                    potential_new_stop = hard_stop
                if potential_new_stop > current_stop:
                    self.db.update_trailing_stop(ts.id, new_stop_level=potential_new_stop)
                    logger.debug(
                        "trailing_stop_moved",
                        old_stop=str(current_stop),
                        new_stop=str(potential_new_stop),
                    )
                    current_stop = potential_new_stop

            # Check if stop is hit (profit protection - trailing only activates after profit)
            if current_stop is not None and current_price <= current_stop:
                # Calculate profit locked in (trailing stop is profit protection)
                # Use Decimal.quantize() for proper Decimal arithmetic
                if entry_price <= 0:
                    # CRITICAL: Invalid entry_price indicates data corruption - halt trading
                    logger.critical(
                        "invalid_entry_price_halting_trading",
                        entry_price=str(entry_price),
                        context="trailing_stop_check",
                    )
                    self.notifier.send_alert(
                        f"🔴 CRITICAL: Invalid entry_price ({entry_price}) detected. Halting trading!"
                    )
                    self.kill_switch.activate(f"Data corruption: entry_price={entry_price}")
                    profit_pct = Decimal("0")  # Proceed with sell to close position
                else:
                    profit_pct = ((current_price - entry_price) / entry_price * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                logger.info(
                    "trailing_stop_triggered",
                    exit_type="profit_protection",
                    current_price=str(current_price),
                    stop_level=str(current_stop),
                    entry_price=str(entry_price),
                    profit_percent=str(profit_pct),
                )
                self.db.deactivate_trailing_stop(
                    symbol=self.settings.trading_pair,
                    is_paper=is_paper,
                )
                return "sell"

        return None

    def _verify_position_stop_protection(self) -> None:
        """
        Safety check: Verify all open positions have active stop protection.

        This catches situations where the bot crashed after opening a position
        but before creating the trailing stop (e.g., between lines 912-915).

        If an unprotected position is found:
        1. Log a CRITICAL warning
        2. Send alert to user
        3. Attempt to create stop protection using current price

        This is a recovery mechanism, not a normal operation.
        """
        is_paper = self.settings.is_paper_trading

        # Get current position
        position = self.db.get_current_position(
            self.settings.trading_pair, is_paper=is_paper
        )

        if not position or position.get_quantity() <= Decimal("0"):
            return  # No position, nothing to check

        # Check if there's an active trailing stop
        ts = self.db.get_active_trailing_stop(
            symbol=self.settings.trading_pair, is_paper=is_paper
        )

        if ts is not None:
            return  # Stop exists, all good

        # CRITICAL: Position exists but no stop protection!
        avg_cost = position.get_average_cost()
        qty = position.get_quantity()

        logger.critical(
            "position_without_stop_protection",
            symbol=self.settings.trading_pair,
            quantity=str(qty),
            avg_cost=str(avg_cost),
            is_paper=is_paper,
        )

        self.notifier.send_alert(
            f"⚠️ CRITICAL: Position found without stop protection! "
            f"Qty: {qty}, Avg Cost: {avg_cost}. Creating emergency stop."
        )

        # Attempt to create stop protection
        try:
            candles = self.client.get_candles(
                self.settings.trading_pair,
                self.settings.candle_interval,
                limit=self.settings.candle_limit,
            )
            self._create_trailing_stop(
                entry_price=avg_cost,
                candles=candles,
                is_paper=is_paper,
                avg_cost=avg_cost,
            )
            logger.info("emergency_stop_created", avg_cost=str(avg_cost))
            self.notifier.send_alert("✅ Emergency stop protection created successfully.")
        except Exception as e:
            logger.error("emergency_stop_creation_failed", error=str(e))
            self.notifier.send_alert(
                f"❌ FAILED to create emergency stop: {e}. Manual intervention required!"
            )

    def _periodic_stop_protection_check(self) -> None:
        """
        Periodic safety check: verify positions have stop protection.

        Runs every STOP_PROTECTION_CHECK_INTERVAL_SECONDS (after startup check)
        to catch edge cases:
        - Manual position opens
        - Recovery failures
        - Database corruption
        """
        now = datetime.now()

        # Skip if not enough time has passed (initialized to None in __init__)
        if self._last_stop_check is not None:
            elapsed = (now - self._last_stop_check).total_seconds()
            if elapsed < STOP_PROTECTION_CHECK_INTERVAL_SECONDS:
                return

        self._last_stop_check = now

        # Run the full verification
        try:
            self._verify_position_stop_protection()
        except Exception as e:
            logger.error("periodic_stop_check_failed", error=str(e))

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False

        # Save final state
        try:
            portfolio_value = self._get_portfolio_value()
            ending_price = self.client.get_current_price(self.settings.trading_pair)
            self.db.update_daily_stats(
                ending_balance=portfolio_value,
                ending_price=ending_price,
                is_paper=self.settings.is_paper_trading,
            )
        except Exception as e:
            logger.error("shutdown_state_save_failed", error=str(e))

        # Close the async event loop
        try:
            self._loop.close()
        except Exception as e:
            logger.debug("event_loop_close_failed", error=str(e))

        # Send shutdown notification
        self.notifier.notify_shutdown("Graceful shutdown")

        logger.info("daemon_stopped")
