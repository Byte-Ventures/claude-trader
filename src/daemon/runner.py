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
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Optional, Union

import structlog

from config.settings import Settings, TradingMode, VetoAction, AIFailureMode, request_reload, reload_pending, reload_settings
from src.api.exchange_factory import create_exchange_client, get_exchange_name
from src.api.exchange_protocol import ExchangeClient
from src.api.paper_client import PaperTradingClient
from src.notifications.telegram import TelegramNotifier
from src.safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, BreakerLevel
from src.safety.kill_switch import KillSwitch
from src.safety.loss_limiter import LossLimiter, LossLimitConfig
from src.safety.trade_cooldown import TradeCooldown, TradeCooldownConfig
from src.safety.validator import OrderValidator, OrderRequest, ValidatorConfig
from sqlalchemy.exc import SQLAlchemyError
from src.state.database import BotMode, Database, SignalHistory
from src.strategy.signal_scorer import SignalScorer
from src.strategy.weight_profile_selector import (
    WeightProfileSelector,
    ProfileSelectorConfig,
    WEIGHT_PROFILES,
)
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

    # Signal scores for stop types - high values (>100) to distinguish from normal signals (-100 to +100)
    TRAILING_STOP_SIGNAL_SCORE = 255  # Trailing stop triggered
    HARD_STOP_SIGNAL_SCORE = 256      # Hard stop loss triggered
    TAKE_PROFIT_SIGNAL_SCORE = 257    # Take profit triggered

    def __init__(self, settings: Settings):
        """
        Initialize trading daemon with all components.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.shutdown_event = Event()
        self._running = False
        self._cramer_mode_disabled = False  # Disables Cramer Mode for session on balance mismatch
        self.cramer_trade_cooldown: Optional[TradeCooldown] = None  # Independent cooldown for Cramer Mode

        # Check for deprecated AI_FAILURE_MODE setting
        import os
        if "AI_FAILURE_MODE" in os.environ and (
            "AI_FAILURE_MODE_BUY" not in os.environ and "AI_FAILURE_MODE_SELL" not in os.environ
        ):
            logger.warning(
                "deprecated_setting_detected",
                setting="AI_FAILURE_MODE",
                message="AI_FAILURE_MODE is deprecated. Use AI_FAILURE_MODE_BUY and AI_FAILURE_MODE_SELL instead. "
                        "New defaults: BUY=safe (skip on AI failure), SELL=open (proceed on AI failure)."
            )
        self._last_daily_report: Optional[date] = None
        self._last_weekly_report: Optional[date] = None
        self._last_monthly_report: Optional[date] = None
        # For adaptive interval and emergency stop creation. Default "normal" is intentionally
        # conservative - if bot restarts during extreme volatility before first iteration,
        # emergency stops use tighter (1.5x ATR) rather than wider (2.0x) multiplier.
        # This is safer than no stop at all; volatility updates after first iteration.
        self._last_volatility: str = "normal"
        self._last_regime: str = "neutral"  # For regime change notifications
        self._pending_regime: Optional[str] = None  # Flap protection: pending regime change
        self._last_hourly_analysis: Optional[datetime] = None  # For hourly market analysis
        self._pending_post_volatility_analysis: bool = False  # Trigger analysis when volatility calms
        self._last_stop_check: Optional[datetime] = None  # For periodic stop protection checks
        self._last_ai_failure_notification: Optional[datetime] = None  # Cooldown for AI failure notifications
        self._last_interesting_hold_review: Optional[datetime] = None  # Cooldown for interesting_hold reviews
        self._last_interesting_hold_score: Optional[int] = None  # Track signal changes
        self._last_veto_timestamp: Optional[datetime] = None  # Cooldown after AI veto rejection
        self._last_veto_direction: Optional[str] = None  # Signal direction at veto ("buy"/"sell")

        # Multi-Timeframe state (for HTF bias caching)
        self._daily_trend: str = "neutral"
        self._daily_last_fetch: Optional[datetime] = None
        self._6h_trend: str = "neutral"
        self._6h_last_fetch: Optional[datetime] = None

        # Signal history tracking for marking executed trades.
        # Thread-safety note: This is safe because TradingDaemon runs single-threaded.
        # The ID is set in _trading_iteration() before any trade execution, and used
        # in _execute_buy()/_execute_sell() within the same iteration. No concurrent
        # iterations can occur because the main loop is synchronous.
        self._current_signal_id: Optional[int] = None
        self._signal_history_failures: int = 0  # Track consecutive storage failures

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

        # Validate Cramer Mode configuration (must be paper trading only)
        if settings.enable_cramer_mode and not settings.is_paper_trading:
            logger.error("cramer_mode_requires_paper_trading")
            raise ValueError("Cramer Mode can only be enabled in paper trading mode")

        # Initialize trading client (paper or live)
        self.client: Union[ExchangeClient, PaperTradingClient]
        if settings.is_paper_trading:
            # Try to restore paper balance from database (explicit bot_mode for clarity)
            saved_balance = self.db.get_last_paper_balance(settings.trading_pair, bot_mode=BotMode.NORMAL)
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

            # Initialize Cramer Mode client if enabled
            self.cramer_client: Optional[PaperTradingClient] = None
            if settings.enable_cramer_mode:
                # Try to restore Cramer Mode balance from database
                cramer_balance = self.db.get_last_paper_balance(settings.trading_pair, bot_mode=BotMode.INVERTED)
                if cramer_balance:
                    cramer_quote, cramer_base, _ = cramer_balance
                    logger.info(
                        "cramer_balance_restored_from_db",
                        quote=str(cramer_quote),
                        base=str(cramer_base),
                    )
                else:
                    # First-time enable: use normal bot's CURRENT state
                    cramer_quote = self.client.get_balance(self._quote_currency).available
                    cramer_base = self.client.get_balance(self._base_currency).available
                    logger.info(
                        "cramer_balance_copied_from_normal",
                        quote=str(cramer_quote),
                        base=str(cramer_base),
                    )

                    # Warn if normal bot has open position (unfair comparison)
                    normal_position = self.db.get_current_position(
                        settings.trading_pair, is_paper=True, bot_mode=BotMode.NORMAL
                    )
                    if normal_position and normal_position.get_quantity() > Decimal("0"):
                        logger.warning(
                            "cramer_mode_starting_without_position",
                            msg="Normal bot has open position but Cramer Mode starts fresh. Consider disabling until position is closed for fair comparison.",
                            normal_position_size=str(normal_position.get_quantity()),
                        )

                self.cramer_client = PaperTradingClient(
                    real_client=self.real_client,
                    initial_quote=float(cramer_quote),
                    initial_base=float(cramer_base),
                    trading_pair=settings.trading_pair,
                )
                # Initialize independent cooldown for Cramer Mode (uses same config as normal bot)
                if settings.trade_cooldown_enabled:
                    self.cramer_trade_cooldown = TradeCooldown(
                        config=TradeCooldownConfig(
                            buy_cooldown_minutes=settings.buy_cooldown_minutes,
                            sell_cooldown_minutes=settings.sell_cooldown_minutes,
                            buy_price_change_percent=settings.buy_price_change_percent,
                            sell_price_change_percent=settings.sell_price_change_percent,
                        ),
                        db=self.db,
                        is_paper=True,  # Cramer Mode is always paper trading
                        symbol=settings.trading_pair,
                        bot_mode="inverted",  # Track separately from normal bot
                    )
                logger.info("cramer_mode_enabled", mode="inverted")
        else:
            self.client = self.real_client
            self.cramer_client = None
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
                veto_reduce_threshold=settings.veto_reduce_threshold,
                veto_skip_threshold=settings.veto_skip_threshold,
                position_reduction=settings.position_reduction,
                interesting_hold_margin=settings.interesting_hold_margin,
                review_all=settings.ai_review_all,
                market_research_enabled=settings.market_research_enabled,
                ai_web_search_enabled=settings.ai_web_search_enabled,
                market_research_cache_minutes=settings.market_research_cache_minutes,
                candle_interval=settings.candle_interval,
                signal_threshold=settings.signal_threshold,
                max_tokens=settings.ai_max_tokens,
                api_timeout=settings.ai_api_timeout,
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
            ),
            candle_interval=settings.candle_interval,
        )

        self.loss_limiter = LossLimiter(
            config=LossLimitConfig(
                throttle_at_percent=settings.loss_throttle_start_percent,
                throttle_min_multiplier=settings.loss_throttle_min_multiplier,
                max_daily_loss_percent=settings.max_daily_loss_percent,
                max_hourly_loss_percent=settings.max_hourly_loss_percent,
            ),
            on_limit_hit=lambda limit_type, percent: self.notifier.notify_loss_limit(
                limit_type, percent
            )
        )

        # Initialize order validator
        self.validator = OrderValidator(
            config=ValidatorConfig(
                estimated_fee_percent=settings.estimated_fee_percent,
                profit_margin_multiplier=settings.profit_margin_multiplier,
                min_trade_quote=settings.min_trade_quote,
                max_position_percent=settings.max_position_percent,
            ),
            kill_switch=self.kill_switch,
            circuit_breaker=self.circuit_breaker,
            loss_limiter=self.loss_limiter,
        )

        # Initialize trade cooldown (optional, prevents rapid consecutive trades)
        self.trade_cooldown: Optional[TradeCooldown] = None
        if settings.trade_cooldown_enabled:
            self.trade_cooldown = TradeCooldown(
                config=TradeCooldownConfig(
                    buy_cooldown_minutes=settings.buy_cooldown_minutes,
                    sell_cooldown_minutes=settings.sell_cooldown_minutes,
                    buy_price_change_percent=settings.buy_price_change_percent,
                    sell_price_change_percent=settings.sell_price_change_percent,
                ),
                db=self.db,
                is_paper=settings.is_paper_trading,
                symbol=settings.trading_pair,
            )

        # Enable Telegram command handling (for /reset, /status, /help)
        self.notifier.set_safety_systems(
            circuit_breaker=self.circuit_breaker,
            kill_switch=self.kill_switch,
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
            candle_interval=settings.candle_interval,
            trading_pair=settings.trading_pair,
            momentum_trend_strength_cap=settings.momentum_trend_strength_cap,
            whale_volume_threshold=settings.whale_volume_threshold,
            whale_direction_threshold=settings.whale_direction_threshold,
            whale_candle_bullish_threshold=settings.whale_candle_bullish_threshold,
            whale_candle_bearish_threshold=settings.whale_candle_bearish_threshold,
            whale_boost_percent=settings.whale_boost_percent,
            high_volume_boost_percent=settings.high_volume_boost_percent,
            mtf_aligned_boost=settings.mtf_aligned_boost,
            mtf_counter_penalty=settings.mtf_counter_penalty,
            max_oversold_buys_24h=settings.max_oversold_buys_24h,
            price_stabilization_window=settings.price_stabilization_window,
            volume_sma_window=settings.volume_sma_window,
            high_volume_threshold=settings.high_volume_threshold,
            low_volume_threshold=settings.low_volume_threshold,
            low_volume_penalty=settings.low_volume_penalty,
            extreme_rsi_lower=settings.extreme_rsi_lower,
            extreme_rsi_upper=settings.extreme_rsi_upper,
            trend_filter_penalty=settings.trend_filter_penalty,
            macd_interval_multipliers=settings.macd_interval_multipliers,
        )

        self.position_sizer = PositionSizer(
            config=PositionSizeConfig(
                risk_per_trade_percent=settings.risk_per_trade_percent,
                min_trade_base=settings.min_trade_base,
                max_position_percent=settings.position_size_percent,
                stop_loss_atr_multiplier=settings.stop_loss_atr_multiplier,
                min_stop_loss_percent=settings.min_stop_loss_percent,
                min_trade_quote=settings.min_trade_quote,
                max_trade_quote=settings.max_trade_quote,
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
            ),
            custom_modifiers=settings.sentiment_trend_modifiers,
        )
        if settings.regime_adaptation_enabled:
            # Restore last regime from database
            last_regime = self.db.get_last_regime(is_paper=settings.is_paper_trading)
            if last_regime:
                self._last_regime = last_regime
                logger.info("regime_restored_from_db", regime=last_regime)
            logger.info("market_regime_initialized", scale=settings.regime_adjustment_scale)

        # Initialize AI weight profile selector (optional, requires OpenRouter key)
        self.weight_profile_selector: Optional[WeightProfileSelector] = None
        self._last_weight_profile: str = "default"
        self._last_weight_profile_confidence: float = 0.0
        self._last_weight_profile_reasoning: str = ""
        self._pending_weight_profile: Optional[str] = None  # Flap protection: pending profile change
        if settings.ai_weight_profile_enabled and settings.openrouter_api_key:
            self.weight_profile_selector = WeightProfileSelector(
                api_key=settings.openrouter_api_key.get_secret_value(),
                config=ProfileSelectorConfig(
                    enabled=settings.ai_weight_profile_enabled,
                    cache_minutes=self._get_candle_interval_minutes(),
                    fallback_profile=settings.ai_weight_fallback_profile,
                    model=settings.ai_weight_profile_model,
                    max_tokens=settings.ai_max_tokens,
                ),
            )
            # Restore last weight profile from database
            last_profile = self.db.get_last_weight_profile(is_paper=settings.is_paper_trading)
            if last_profile and last_profile in WEIGHT_PROFILES:
                self._last_weight_profile = last_profile
                self.signal_scorer.update_weights(WEIGHT_PROFILES[last_profile])
                logger.info("weight_profile_restored_from_db", profile=last_profile)
            logger.info(
                "ai_weight_profile_selector_initialized",
                model=settings.ai_weight_profile_model,
                fallback=settings.ai_weight_fallback_profile,
            )
        else:
            logger.info("ai_weight_profile_selector_disabled")

        # AI recommendation state (for threshold adjustments from interesting holds)
        self._ai_recommendation: Optional[str] = None  # "accumulate", "reduce", "wait"
        self._ai_recommendation_confidence: float = 0.0
        self._ai_recommendation_time: Optional[datetime] = None
        self._ai_recommendation_ttl_minutes: int = settings.ai_recommendation_ttl_minutes

        # Check postmortem requirements if enabled
        self._postmortem_available = False
        self._postmortem_executor: Optional[ThreadPoolExecutor] = None
        if settings.postmortem_enabled:
            self._postmortem_available = self._check_postmortem_requirements()
            if self._postmortem_available:
                # Single-worker executor prevents resource exhaustion
                self._postmortem_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="postmortem")

        # Register shutdown handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Register SIGUSR2 for config hot-reload
        signal.signal(signal.SIGUSR2, self._handle_reload_signal)
        logger.info("config_reload_signal_registered", signal="SIGUSR2")

    def _check_postmortem_requirements(self) -> bool:
        """
        Check if postmortem analysis requirements are met.

        Returns True if Claude CLI is accessible, False otherwise.
        Logs warnings if requirements are not met.
        """
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("postmortem_claude_cli_available", version=result.stdout.strip())
                return True
            else:
                logger.warning(
                    "postmortem_claude_cli_error",
                    error="Claude CLI returned non-zero exit code",
                    stderr=result.stderr[:200] if result.stderr else None,
                )
                return False
        except FileNotFoundError:
            logger.warning(
                "postmortem_disabled_claude_not_found",
                hint="Install Claude CLI or disable POSTMORTEM_ENABLED",
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("postmortem_disabled_claude_timeout")
            return False
        except Exception as e:
            logger.warning("postmortem_disabled_error", error=str(e))
            return False

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

    def _get_candle_interval_minutes(self) -> int:
        """Convert candle interval string to minutes."""
        interval_map = {
            "ONE_MINUTE": 1,
            "FIVE_MINUTE": 5,
            "FIFTEEN_MINUTE": 15,
            "THIRTY_MINUTE": 30,
            "ONE_HOUR": 60,
            "TWO_HOUR": 120,
            "SIX_HOUR": 360,
            "ONE_DAY": 1440,
        }
        return interval_map.get(self.settings.candle_interval, 60)

    def _calculate_bb_percent_b(self, price: Decimal, indicators) -> Optional[float]:
        """Calculate Bollinger %B for weight profile selection."""
        if indicators.bb_upper and indicators.bb_lower:
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                return (float(price) - indicators.bb_lower) / bb_range
        return None

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
        elapsed = (datetime.now(timezone.utc) - self._ai_recommendation_time).total_seconds() / 60
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
                candle_interval=new_settings.candle_interval,
            )

            # Update PositionSizer
            self.position_sizer.update_settings(
                max_position_percent=new_settings.position_size_percent,
                stop_loss_atr_multiplier=new_settings.stop_loss_atr_multiplier,
                take_profit_atr_multiplier=new_settings.take_profit_atr_multiplier,
                atr_period=new_settings.atr_period,
                min_stop_loss_percent=new_settings.min_stop_loss_percent,
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

            # Update TradeCooldown (if enabled)
            if self.trade_cooldown:
                self.trade_cooldown.update_settings(
                    buy_cooldown_minutes=new_settings.buy_cooldown_minutes,
                    sell_cooldown_minutes=new_settings.sell_cooldown_minutes,
                    buy_price_change_percent=new_settings.buy_price_change_percent,
                    sell_price_change_percent=new_settings.sell_price_change_percent,
                )

            # Update TradeReviewer (if enabled)
            if self.trade_reviewer:
                self.trade_reviewer.review_all = new_settings.ai_review_all
                self.trade_reviewer.veto_reduce_threshold = new_settings.veto_reduce_threshold
                self.trade_reviewer.veto_skip_threshold = new_settings.veto_skip_threshold
                self.trade_reviewer.position_reduction = new_settings.position_reduction
                self.trade_reviewer.interesting_hold_margin = new_settings.interesting_hold_margin
                self.trade_reviewer.candle_interval = new_settings.candle_interval

            # Update circuit breaker candle interval for adaptive flash crash detection
            self.circuit_breaker.set_candle_interval(new_settings.candle_interval)

            # Update AI recommendation TTL
            self._ai_recommendation_ttl_minutes = new_settings.ai_recommendation_ttl_minutes

            # Invalidate HTF cache if MTF settings changed
            mtf_settings = {"mtf_enabled", "mtf_4h_enabled", "mtf_candle_limit",
                           "mtf_daily_cache_minutes", "mtf_4h_cache_minutes",
                           "mtf_aligned_boost", "mtf_counter_penalty"}
            if mtf_settings & set(changes.keys()):
                self._invalidate_htf_cache()

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

    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """
        Get start of the candle period containing the given timestamp.

        Args:
            timestamp: Any datetime within a candle period

        Returns:
            datetime representing the start of that candle period
        """
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
        seconds = granularity_seconds.get(self.settings.candle_interval, 3600)
        ts = timestamp.timestamp()
        candle_start_ts = (ts // seconds) * seconds
        return datetime.fromtimestamp(candle_start_ts, tz=timezone.utc)

    def _should_skip_review_after_veto(self, signal_action: str) -> tuple[bool, Optional[str]]:
        """
        Check if AI review should be skipped due to recent veto rejection.

        After a SKIP or REDUCE veto, skip further reviews until:
        - A new candle period begins, OR
        - The signal direction changes (BUY -> SELL or vice versa)

        Args:
            signal_action: Current signal action ("buy" or "sell")

        Returns:
            Tuple of (should_skip, reason) where reason explains why if skipping
        """
        if not self.settings.ai_review_rejection_cooldown:
            return False, None

        if self._last_veto_timestamp is None:
            return False, None

        # Direction changed - allow review
        if self._last_veto_direction != signal_action:
            return False, None

        # Check if we're in a new candle period
        now = datetime.now(timezone.utc)
        current_candle_start = self._get_candle_start(now)
        veto_candle_start = self._get_candle_start(self._last_veto_timestamp)

        if current_candle_start > veto_candle_start:
            # New candle, reset cooldown state
            self._last_veto_timestamp = None
            self._last_veto_direction = None
            return False, None

        # Same candle, same direction - skip review
        return True, f"veto_cooldown (same candle, direction={signal_action})"

    def _get_timeframe_trend(self, granularity: str, cache_minutes: int) -> str:
        """
        Get trend for a specific timeframe with caching.

        Args:
            granularity: Candle granularity (e.g., "ONE_DAY", "SIX_HOUR")
            cache_minutes: Cache TTL in minutes

        Returns:
            Trend direction: "bullish", "bearish", or "neutral"
        """
        # Select appropriate cache based on granularity
        if granularity == "ONE_DAY":
            last_fetch = self._daily_last_fetch
            cached_trend = self._daily_trend
        else:  # SIX_HOUR
            last_fetch = self._6h_last_fetch
            cached_trend = self._6h_trend

        now = datetime.now(timezone.utc)
        if last_fetch and (now - last_fetch) < timedelta(minutes=cache_minutes):
            return cached_trend

        try:
            candles = self.client.get_candles(
                self.settings.trading_pair,
                granularity=granularity,
                limit=self.settings.mtf_candle_limit,
            )

            # Validate candles before processing - need enough data for trend calculation
            if candles is None or candles.empty or len(candles) < self.signal_scorer.ema_slow_period:
                logger.warning(
                    "htf_insufficient_data",
                    timeframe=granularity,
                    candle_count=len(candles) if candles is not None and not candles.empty else 0,
                    required=self.signal_scorer.ema_slow_period,
                )
                return cached_trend or "neutral"

            trend = self.signal_scorer.get_trend(candles)

            # Update cache
            if granularity == "ONE_DAY":
                self._daily_trend = trend
                self._daily_last_fetch = now
            else:
                self._6h_trend = trend
                self._6h_last_fetch = now

            logger.info("htf_trend_updated", timeframe=granularity, trend=trend)
            return trend
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, NotImplementedError) as e:
            # Expected failures: network issues, API errors, data parsing issues,
            # or unsupported granularity (NotImplementedError).
            # Fail-open: return cached trend or neutral, never block trading
            logger.warning("htf_fetch_failed", timeframe=granularity, error=str(e), error_type=type(e).__name__)
            return cached_trend or "neutral"
        except Exception as e:
            # Unexpected errors - log at error level but still fail-open
            # Financial bot should never crash due to HTF analysis failure
            logger.error("htf_fetch_unexpected_error", timeframe=granularity, error=str(e), error_type=type(e).__name__)
            return cached_trend or "neutral"

    def _get_htf_bias(self) -> tuple[str, str, str]:
        """
        Get combined HTF bias from daily + 6-hour trends.

        Returns:
            Tuple of (combined_bias, daily_trend, 6h_trend)
            - Both bullish → "bullish"
            - Both bearish → "bearish"
            - Mixed/neutral → "neutral"
        """
        if not self.settings.mtf_enabled:
            return "neutral", "neutral", "neutral"

        daily = self._get_timeframe_trend("ONE_DAY", self.settings.mtf_daily_cache_minutes)

        # 4H is optional - when disabled, just use daily trend directly
        if not self.settings.mtf_4h_enabled:
            # Daily-only mode: simpler, fewer API calls
            return daily, daily, None

        # Use FOUR_HOUR instead of SIX_HOUR for broader exchange compatibility
        # (Kraken doesn't support 6-hour candles, only 4-hour)
        four_hour = self._get_timeframe_trend("FOUR_HOUR", self.settings.mtf_4h_cache_minutes)

        # Combine: both must agree for strong bias
        if daily == "bullish" and four_hour == "bullish":
            combined = "bullish"
        elif daily == "bearish" and four_hour == "bearish":
            combined = "bearish"
        else:
            combined = "neutral"

        return combined, daily, four_hour

    def _invalidate_htf_cache(self) -> None:
        """
        Invalidate HTF trend cache when settings change.

        Called when MTF-related settings are updated at runtime to ensure
        the next iteration uses fresh data with the new parameters.
        """
        self._daily_last_fetch = None
        self._6h_last_fetch = None
        logger.info("htf_cache_invalidated")

    def _store_signal_history(
        self,
        signal_result,
        current_price: Decimal,
        htf_bias: str,
        daily_trend: str,
        four_hour_trend: str,
        threshold: int,
        trade_executed: bool = False,
    ) -> Optional[int]:
        """
        Store signal calculation for historical analysis.

        Called every iteration to enable post-mortem analysis of trades.

        Returns:
            The signal history record ID, or None if storage failed.
        """
        try:
            breakdown = signal_result.breakdown
            with self.db.session() as session:
                history = SignalHistory(
                    symbol=self.settings.trading_pair,
                    is_paper=self.settings.is_paper_trading,
                    current_price=str(current_price),
                    rsi_score=breakdown.get("rsi", 0),
                    macd_score=breakdown.get("macd", 0),
                    bollinger_score=breakdown.get("bollinger", 0),
                    ema_score=breakdown.get("ema", 0),
                    volume_score=breakdown.get("volume", 0),
                    rsi_value=breakdown.get("_rsi_value"),
                    macd_histogram=breakdown.get("_macd_histogram"),
                    bb_position=breakdown.get("_bb_position"),
                    ema_gap_percent=breakdown.get("_ema_gap_percent"),
                    volume_ratio=breakdown.get("_volume_ratio"),
                    trend_filter_adj=breakdown.get("trend_filter", 0),
                    momentum_mode_adj=breakdown.get("momentum_mode", 0),
                    whale_activity_adj=breakdown.get("whale_activity", 0),
                    htf_bias_adj=breakdown.get("htf_bias", 0),
                    htf_bias=htf_bias,
                    htf_daily_trend=daily_trend,
                    htf_4h_trend=four_hour_trend,
                    raw_score=breakdown.get("_raw_score", signal_result.score),
                    final_score=signal_result.score,
                    action=signal_result.action,
                    threshold_used=threshold,
                    trade_executed=trade_executed,
                )
                session.add(history)
                session.commit()
                self._signal_history_failures = 0  # Reset on success
                return history.id
        except SQLAlchemyError as e:
            # Database errors are non-critical - don't block trading for history storage
            self._signal_history_failures += 1
            logger.warning(
                "signal_history_store_failed",
                error=str(e),
                error_type=type(e).__name__,
                consecutive_failures=self._signal_history_failures,
            )
            # Alert at threshold failures, then every 50 additional failures
            threshold = self.settings.signal_history_failure_threshold
            if self._signal_history_failures == threshold or (
                self._signal_history_failures > threshold
                and (self._signal_history_failures - threshold) % 50 == 0
            ):
                self.notifier.notify_error(
                    f"Signal history storage failing ({self._signal_history_failures} consecutive failures)",
                    context=f"Last error: {e}",
                )
            return None

    def _mark_signal_trade_executed(self, signal_id: Optional[int]) -> None:
        """
        Mark a specific signal history record as having resulted in a trade.

        Called after successful buy/sell to enable accurate post-mortem analysis.

        Args:
            signal_id: The ID of the signal history record to mark.
                       If None, the operation is skipped (signal storage may have failed).
        """
        if signal_id is None:
            return

        try:
            with self.db.session() as session:
                session.query(SignalHistory).filter(
                    SignalHistory.id == signal_id
                ).update({"trade_executed": True})
                session.commit()
                logger.debug("signal_history_trade_marked", signal_id=signal_id)
        except SQLAlchemyError as e:
            # Non-critical - don't block trading for history update
            logger.warning("signal_history_trade_flag_update_failed", error=str(e))

    def _run_postmortem_async(self, trade_id: Optional[int] = None) -> None:
        """
        Run post-mortem analysis asynchronously after a trade.

        Spawns a background thread to run the postmortem script without blocking trading.
        Only runs if postmortem is enabled AND Claude CLI is available.

        Args:
            trade_id: Optional specific trade ID. If None, analyzes the last trade.
        """
        if not self.settings.postmortem_enabled or not self._postmortem_available:
            return

        # Validate trade_id type (defense-in-depth against injection)
        if trade_id is not None and not isinstance(trade_id, int):
            logger.error("postmortem_invalid_trade_id_type", trade_id_type=type(trade_id).__name__)
            return

        def run_postmortem():
            try:
                # Build command
                script_path = Path(__file__).parent.parent.parent / "tools" / "postmortem.py"
                cmd = ["python3", str(script_path)]

                # Add mode flag
                if self.settings.is_paper_trading:
                    cmd.append("--paper")
                else:
                    cmd.append("--live")

                # Add trade selection
                if trade_id:
                    cmd.extend(["--trade-id", str(trade_id)])
                else:
                    cmd.extend(["--last", "1"])

                # Add discussion flag if enabled
                if self.settings.postmortem_create_discussion:
                    cmd.append("--create-discussion")

                logger.info("postmortem_starting", cmd=" ".join(cmd))

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,  # 15 minute timeout
                )

                if result.returncode == 0:
                    logger.info("postmortem_completed")
                    # Notify via Telegram if we created a discussion
                    if self.settings.postmortem_create_discussion and "GitHub Discussion created" in result.stdout:
                        # Extract URL from output
                        for line in result.stdout.split("\n"):
                            if "github.com" in line and "discussions" in line:
                                self.notifier.send_message(f"📊 Post-mortem analysis: {line.strip()}")
                                break
                else:
                    logger.warning(
                        "postmortem_failed",
                        returncode=result.returncode,
                        stderr=result.stderr[:500] if result.stderr else None,
                    )

            except subprocess.TimeoutExpired:
                logger.warning("postmortem_timeout")
            except Exception as e:
                logger.warning("postmortem_error", error=str(e))

        # Submit to executor (single-worker prevents resource exhaustion)
        if self._postmortem_executor:
            self._postmortem_executor.submit(run_postmortem)

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

            # Initialize Cramer Mode daily stats if enabled
            if self.cramer_client and self.settings.enable_cramer_mode:
                cramer_portfolio_value = self._get_cramer_portfolio_value()
                self.db.update_daily_stats(
                    starting_balance=cramer_portfolio_value,
                    starting_price=starting_price,
                    is_paper=True,  # Cramer Mode is paper-only
                    bot_mode=BotMode.INVERTED,
                )
                logger.info(
                    "cramer_daily_stats_initialized",
                    starting_balance=str(cramer_portfolio_value),
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
                # Check for Telegram commands (/reset, /status, /help)
                # Pass event loop to avoid creating a new one each time
                self.notifier.check_commands(loop=self._loop)

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
                is_paper=self.settings.is_paper_trading,
                bot_mode=BotMode.NORMAL
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

        # Update validator with current state (normal bot only)
        # NOTE: Cramer Mode shares safety systems with normal bot intentionally.
        # Cramer Mode is for strategy comparison only - it should NOT trigger
        # kill switch, circuit breaker, or loss limiter independently.
        # Only normal bot's performance affects safety thresholds.
        #
        # WHY Cramer Mode doesn't update loss_limiter:
        # 1. Purpose: Compare strategy performance, not add to risk exposure
        # 2. If Cramer loses money, we don't want to halt normal bot trading
        # 3. Paper trading only - no real financial impact from Cramer losses
        # 4. Cramer PnL is tracked separately in database for analysis
        self.validator.update_balances(base_balance, quote_balance, current_price)

        # Log warning if both bots have positions simultaneously (for visibility)
        if self.cramer_client and not self._cramer_mode_disabled:
            cramer_base = self.cramer_client.get_balance(self._base_currency).available
            if base_balance > Decimal("0") and cramer_base > Decimal("0"):
                normal_value = base_balance * current_price
                cramer_value = cramer_base * current_price
                total_exposure = normal_value + cramer_value
                net_exposure = abs(normal_value - cramer_value)
                logger.info(
                    "dual_bot_position_exposure",
                    normal_base=str(base_balance),
                    normal_value_usd=str(normal_value),
                    cramer_base=str(cramer_base),
                    cramer_value_usd=str(cramer_value),
                    total_exposure_usd=str(total_exposure),
                    net_exposure_usd=str(net_exposure),
                    msg="Both bots have open positions - paper trading only",
                )

        # Check BOTH trailing stops before executing either (avoid race condition)
        # We must check both BEFORE executing either because normal bot may return early,
        # which would skip Cramer Mode's trailing stop check if done sequentially.
        trailing_action = self._check_trailing_stop(current_price)
        cramer_trailing_action = None
        cramer_base_balance = Decimal("0")

        if self.cramer_client and not self._cramer_mode_disabled:
            # Get balance BEFORE stop check to ensure consistency
            cramer_base_balance = self.cramer_client.get_balance(self._base_currency).available
            cramer_trailing_action = self._check_trailing_stop(current_price, bot_mode=BotMode.INVERTED)

        # Execute Cramer Mode trailing stop FIRST (before normal bot's early return)
        # Why Cramer executes first:
        # 1. Normal bot returns early after its trailing stop (line 1245)
        # 2. If we executed normal first, Cramer's stop would never run
        # 3. Both bots have INDEPENDENT risk management - not mirrored entries
        # 4. This ensures fair comparison: both bots can exit via their own stops
        if self.cramer_client and not self._cramer_mode_disabled and cramer_trailing_action == "sell" and cramer_base_balance > Decimal("0"):
            logger.info(
                "cramer_trailing_stop_triggered",
                base_balance=str(cramer_base_balance),
            )
            self._execute_cramer_trailing_stop_sell(
                candles=candles,
                current_price=current_price,
                base_balance=cramer_base_balance,
            )

        # Execute normal bot trailing stop (may return early)
        if trailing_action == "sell" and base_balance > Decimal("0"):
            logger.info("trailing_stop_sell_triggered", base_balance=str(base_balance))
            # Execute sell for trailing stop (use full position, high priority)
            self._execute_sell(
                candles=candles,
                current_price=current_price,
                base_balance=base_balance,
                signal_score=self.TRAILING_STOP_SIGNAL_SCORE,
                safety_multiplier=1.0,  # No safety reduction for stops
            )
            # Note: Trailing stops are NOT mirrored to Cramer Mode as entries.
            # Cramer Mode has independent trailing stops checked above.
            return  # Exit iteration after trailing stop execution

        # Calculate portfolio value for logging
        base_value = base_balance * current_price
        portfolio_value = quote_balance + base_value
        position_percent = float(base_value / portfolio_value * 100) if portfolio_value > 0 else 0

        # Determine which trade directions are possible
        min_quote = Decimal(str(self.position_sizer.config.min_trade_quote))
        min_base = Decimal(str(self.position_sizer.config.min_trade_base))
        # Use validator's hard limit (MAX_POSITION_PERCENT), not position sizer's soft target
        max_position_pct = self.validator.config.max_position_percent

        # Check if there's meaningful room to buy (at least 1% of portfolio or min_trade_quote)
        # This prevents running AI review when position is nearly at limit
        available_room_pct = float(max_position_pct) - position_percent
        min_room_pct = max(1.0, float(min_quote / portfolio_value * 100)) if portfolio_value > 0 else 1.0
        has_room = available_room_pct >= min_room_pct
        can_buy = quote_balance > min_quote and has_room
        can_sell = base_balance > min_base

        # Note: Cramer Mode balance is checked in _execute_cramer_trade
        # Cramer only acts when judge approves AND normal bot executes

        if not has_room and quote_balance > min_quote:
            logger.info(
                "buy_blocked_insufficient_room",
                position_pct=f"{position_percent:.1f}%",
                max_position_pct=f"{max_position_pct:.1f}%",
                available_room_pct=f"{available_room_pct:.2f}%",
                min_room_pct=f"{min_room_pct:.2f}%",
            )

        # Get HTF bias for multi-timeframe confirmation
        htf_bias, daily_trend, four_hour_trend = self._get_htf_bias()

        # Fetch sentiment before signal calculation to enable extreme fear override in MTF logic.
        # This must happen BEFORE calculate_score() because sentiment_category is used to
        # determine whether to apply full vs half counter-penalties when daily/4H disagree.
        sentiment = None
        sentiment_category = None
        if self.settings.regime_sentiment_enabled:
            try:
                sentiment = self._run_async_with_timeout(get_cached_sentiment(), timeout=30)
                if sentiment and sentiment.value is not None:
                    # Reuse existing classification logic from MarketRegime
                    sentiment_category = self.market_regime._classify_sentiment(sentiment.value)
                    logger.debug(
                        "sentiment_fetch_success",
                        category=sentiment_category,
                        value=sentiment.value,
                    )
                else:
                    logger.warning(
                        "sentiment_unavailable_for_trade_evaluation",
                        reason="fetch_returned_none",
                        impact="extreme_fear_override_disabled",
                    )
            except Exception as e:
                logger.warning(
                    "sentiment_fetch_failed_during_trade_evaluation",
                    error=str(e),
                    impact="extreme_fear_override_disabled",
                )

        # Calculate signal with HTF context and sentiment
        signal_result = self.signal_scorer.calculate_score(
            candles, current_price,
            htf_bias=htf_bias,
            htf_daily=daily_trend,
            htf_4h=four_hour_trend,
            sentiment_category=sentiment_category,
        )

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

        # Record whale activity to database if detected
        if signal_result.breakdown.get("_whale_activity"):
            try:
                self.db.record_whale_event(
                    symbol=self.settings.trading_pair,
                    volume_ratio=signal_result.breakdown.get("_volume_ratio", 0),
                    direction=signal_result.breakdown.get("_whale_direction", "unknown"),
                    price_change_pct=signal_result.breakdown.get("_price_change_pct"),
                    signal_score=signal_result.score,
                    signal_action=signal_result.action,
                    is_paper=self.settings.is_paper_trading,
                )
            except SQLAlchemyError as e:
                # Database errors - expected failure mode, non-critical
                logger.warning("whale_event_record_failed", error=str(e), exc_info=True)
            except Exception as e:
                # Unexpected errors - log as error for visibility
                logger.error("whale_event_record_unexpected", error=str(e), exc_info=True)

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
        # Note: sentiment already fetched above before signal calculation
        trend = get_ema_trend_from_values(ind.ema_fast, ind.ema_slow) if ind.ema_fast and ind.ema_slow else "neutral"

        regime = self.market_regime.calculate(
            sentiment=sentiment,
            volatility=ind.volatility or "normal",
            trend=trend,
            signal_action=signal_result.action,
        )

        # Handle regime change with optional flap protection
        should_apply_regime_change = False

        if regime.regime_name != self._last_regime:
            # Flap protection: require 2 consecutive detections before changing
            if self.settings.regime_flap_protection:
                if self._pending_regime == regime.regime_name:
                    # Second consecutive detection - confirm change
                    logger.debug(
                        "regime_change_confirmed",
                        pending=self._pending_regime,
                        new=regime.regime_name,
                    )
                    self._pending_regime = None
                    should_apply_regime_change = True
                else:
                    # First detection - mark as pending, don't change yet
                    logger.debug(
                        "regime_change_pending",
                        current=self._last_regime,
                        pending=regime.regime_name,
                    )
                    self._pending_regime = regime.regime_name
                    # Don't apply change yet
            else:
                # No flap protection - immediate change
                should_apply_regime_change = True
        else:
            # Same regime as before - clear any pending change
            self._pending_regime = None

        # Apply regime change if confirmed
        if should_apply_regime_change:
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

            # Send notification (Telegram + dashboard via unified method)
            self.notifier.notify_regime_change(
                old_regime=self._last_regime,
                new_regime=regime.regime_name,
                threshold_adj=regime.threshold_adjustment,
                position_mult=regime.position_multiplier,
                components=regime.components,
            )

            self._last_regime = regime.regime_name

        # Update weight profile if AI selector enabled and should update
        if self.weight_profile_selector and self.weight_profile_selector.should_update():
            try:
                # Build indicator context for AI profile selection
                indicator_context = {
                    "rsi": ind.rsi,
                    "macd_histogram": ind.macd_histogram,
                    "bb_percent_b": self._calculate_bb_percent_b(current_price, ind),
                }

                # Get Fear & Greed value for context
                fear_greed_value = None
                if sentiment:
                    fear_greed_value = sentiment.value

                selection = self._run_async_with_timeout(
                    self.weight_profile_selector.select_profile(
                        indicators=indicator_context,
                        volatility=ind.volatility or "normal",
                        trend=trend,
                        current_price=current_price,
                        fear_greed=fear_greed_value,
                    ),
                    timeout=30,
                    default=None,
                )

                if selection:
                    # Handle weight profile change with optional flap protection
                    should_apply_profile_change = False

                    if selection.profile_name != self._last_weight_profile:
                        # Flap protection: require 2 consecutive detections before changing
                        if self.settings.weight_profile_flap_protection:
                            if self._pending_weight_profile == selection.profile_name:
                                # Second consecutive detection - confirm change
                                logger.debug(
                                    "weight_profile_change_confirmed",
                                    pending=self._pending_weight_profile,
                                    new=selection.profile_name,
                                )
                                self._pending_weight_profile = None
                                should_apply_profile_change = True
                            else:
                                # First detection - mark as pending, don't change yet
                                logger.debug(
                                    "weight_profile_change_pending",
                                    current=self._last_weight_profile,
                                    pending=selection.profile_name,
                                )
                                self._pending_weight_profile = selection.profile_name
                                # Don't apply weights yet - keep using current profile
                                # But still update confidence/reasoning for dashboard visibility
                                self._last_weight_profile_confidence = selection.confidence
                                self._last_weight_profile_reasoning = selection.reasoning
                        else:
                            # No flap protection - immediate change
                            should_apply_profile_change = True
                    else:
                        # Same profile as before - clear any pending change
                        self._pending_weight_profile = None

                    # Apply profile change if confirmed (or no flap protection)
                    if should_apply_profile_change:
                        # Update weights in signal scorer for NEXT iteration
                        self.signal_scorer.update_weights(selection.weights)

                        # Update confidence/reasoning for dashboard display
                        self._last_weight_profile_confidence = selection.confidence
                        self._last_weight_profile_reasoning = selection.reasoning

                        logger.info(
                            "weight_profile_changed",
                            old=self._last_weight_profile,
                            new=selection.profile_name,
                            confidence=selection.confidence,
                        )

                        # Record to database
                        self.db.record_weight_profile_change(
                            profile_name=selection.profile_name,
                            confidence=selection.confidence,
                            reasoning=selection.reasoning,
                            market_context=selection.market_context,
                            is_paper=self.settings.is_paper_trading,
                        )

                        # Send notification (Telegram + dashboard via unified method)
                        self.notifier.notify_weight_profile(
                            old_profile=self._last_weight_profile,
                            new_profile=selection.profile_name,
                            confidence=selection.confidence,
                            reasoning=selection.reasoning,
                        )

                        # Update stored profile name
                        self._last_weight_profile = selection.profile_name
                    elif selection.profile_name == self._last_weight_profile:
                        # Same profile - still update confidence/reasoning for dashboard
                        self._last_weight_profile_confidence = selection.confidence
                        self._last_weight_profile_reasoning = selection.reasoning

            except Exception as e:
                logger.warning("weight_profile_update_failed", error=str(e))

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
            elapsed = (datetime.now(timezone.utc) - self._ai_recommendation_time).total_seconds() / 60
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "cramer_portfolio": self._get_cramer_portfolio_info(current_price) if self.cramer_client else None,
            "regime": regime.regime_name,
            "weight_profile": {
                "name": self._last_weight_profile,
                "confidence": self._last_weight_profile_confidence,
                "reasoning": self._last_weight_profile_reasoning,
            } if self.weight_profile_selector else None,
            "safety": {
                "circuit_breaker": self.circuit_breaker.level.name,
                "can_trade": self.circuit_breaker.can_trade,
            },
            "trading_pair": self.settings.trading_pair,
            "is_paper": self.settings.is_paper_trading,
        }
        self.db.set_state("dashboard_state", dashboard_state)

        # Store signal for historical analysis (every iteration)
        # Store the ID to mark as executed if a trade occurs (avoids race condition)
        self._current_signal_id = self._store_signal_history(
            signal_result=signal_result,
            current_price=current_price,
            htf_bias=htf_bias,
            daily_trend=daily_trend,
            four_hour_trend=four_hour_trend,
            threshold=effective_threshold,
            trade_executed=False,  # Will be updated if trade executes
        )

        # Claude AI trade review (if enabled)
        claude_veto_multiplier = 1.0  # Default: no reduction
        if self.trade_reviewer:
            # Determine signal direction from score
            signal_direction = "buy" if signal_result.score > 0 else "sell" if signal_result.score < 0 else None

            # Check if normal bot can trade this direction
            normal_can_trade = (
                (signal_direction == "buy" and can_buy) or
                (signal_direction == "sell" and can_sell)
            )

            # Check if Cramer Mode can trade the INVERSE direction
            # Cramer buys when signal says sell, and vice versa
            cramer_can_trade = False
            if self.cramer_client and not self._cramer_mode_disabled:
                try:
                    cramer_quote_balance = self.cramer_client.get_balance(self._quote_currency)
                    cramer_base_balance = self.cramer_client.get_balance(self._base_currency)
                    if cramer_quote_balance and cramer_base_balance:
                        cramer_quote = cramer_quote_balance.available
                        cramer_base = cramer_base_balance.available
                        min_quote = Decimal(str(self.position_sizer.config.min_trade_quote))
                        min_base = Decimal(str(self.position_sizer.config.min_trade_base))
                        cramer_can_buy = cramer_quote > min_quote
                        cramer_can_sell = cramer_base > min_base
                        # Cramer trades INVERSE: signal=sell means Cramer buys, signal=buy means Cramer sells
                        cramer_can_trade = (
                            (signal_direction == "sell" and cramer_can_buy) or
                            (signal_direction == "buy" and cramer_can_sell)
                        )
                except Exception as e:
                    logger.warning("cramer_balance_check_failed", error=str(e))
                    cramer_can_trade = False

            direction_is_tradeable = (
                normal_can_trade or
                cramer_can_trade or
                signal_direction is None  # Neutral score, no direction
            )

            if not direction_is_tradeable:
                # Neither normal nor Cramer can trade this direction
                if signal_direction == "sell":
                    reason = "no_position_either_bot" if self.cramer_client else "no_position"
                else:
                    reason = "fully_allocated_both_bots" if self.cramer_client else "fully_allocated"
                logger.info(
                    "ai_review_skipped",
                    signal_direction=signal_direction,
                    reason=reason,
                )
            else:
                should_review, review_type = self.trade_reviewer.should_review(
                    signal_result, self.settings.signal_threshold
                )

                # Cooldown for interesting_hold reviews (informational only, save tokens)
                # Skip if: signal unchanged OR cooldown not expired (15 min)
                if should_review and review_type == "interesting_hold":
                    now = datetime.now(timezone.utc)
                    score_unchanged = self._last_interesting_hold_score == signal_result.score
                    cooldown_active = (
                        self._last_interesting_hold_review is not None
                        and (now - self._last_interesting_hold_review).total_seconds() < 900  # 15 min
                    )
                    if score_unchanged or cooldown_active:
                        skip_reason = "signal_unchanged" if score_unchanged else "cooldown_active"
                        logger.debug(
                            "interesting_hold_review_skipped",
                            reason=skip_reason,
                            score=signal_result.score,
                            last_score=self._last_interesting_hold_score,
                        )
                        should_review = False

                # Veto cooldown: skip trade reviews if recently rejected for same direction
                if should_review and review_type == "trade":
                    skip_veto, veto_reason = self._should_skip_review_after_veto(
                        signal_result.action
                    )
                    if skip_veto:
                        logger.info(
                            "ai_review_skipped_veto_cooldown",
                            action=signal_result.action,
                            score=signal_result.score,
                            reason=veto_reason,
                        )
                        should_review = False

                if should_review:
                    try:
                        # Estimate trade size for context (before AI veto adjustments)
                        # This gives users visibility into what the trade would be
                        # Use 1.0 multiplier for estimation - actual safety check happens later
                        estimated_size = None
                        if signal_result.action == "buy":
                            est_position = self.position_sizer.calculate_size(
                                df=candles,
                                current_price=current_price,
                                quote_balance=quote_balance,
                                base_balance=base_balance,
                                signal_strength=signal_result.score,
                                side="buy",
                                safety_multiplier=1.0,
                            )
                            if est_position.size_quote > 0:
                                estimated_size = {
                                    "side": "buy",
                                    "size_base": float(est_position.size_base),
                                    "size_quote": float(est_position.size_quote),
                                }
                        elif signal_result.action == "sell":
                            # For sells, estimate based on position
                            sell_size = base_balance if abs(signal_result.score) >= 80 else base_balance * Decimal("0.5")
                            estimated_size = {
                                "side": "sell",
                                "size_base": float(sell_size),
                                "size_quote": float(sell_size * current_price),
                            }

                        # Run async multi-agent review with timeout protection
                        # When Cramer Mode is enabled, hide balance info from reviewers
                        # so judge evaluates signal quality, not execution feasibility
                        review = self._run_async_with_timeout(
                            self.trade_reviewer.review_trade(
                                signal_result=signal_result,
                                current_price=current_price,
                                trading_pair=self.settings.trading_pair,
                                review_type=review_type,
                                position_percent=position_percent,
                                candles=candles,
                                quote_balance=quote_balance,
                                base_balance=base_balance,
                                estimated_size=estimated_size,
                                hide_balance_info=self.settings.enable_cramer_mode,
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
                                # Record veto for cooldown (skip reviews until next candle)
                                self._last_veto_timestamp = datetime.now(timezone.utc)
                                self._last_veto_direction = signal_result.action
                                return  # Skip this iteration
                            elif review.final_veto_action == VetoAction.REDUCE.value:
                                claude_veto_multiplier = self.settings.position_reduction
                                logger.info(
                                    "trade_reduced_by_review",
                                    multiplier=f"{claude_veto_multiplier:.2f}",
                                )
                                # Record veto for cooldown (skip reviews until next candle)
                                self._last_veto_timestamp = datetime.now(timezone.utc)
                                self._last_veto_direction = signal_result.action
                            # Tiered system: skip or reduce only (no delay)

                        # For interesting holds, store recommendation for threshold adjustment
                        if review_type == "interesting_hold":
                            if review.judge_recommendation in ("accumulate", "reduce"):
                                self._ai_recommendation = review.judge_recommendation
                                self._ai_recommendation_confidence = review.judge_confidence
                                self._ai_recommendation_time = datetime.now(timezone.utc)
                                logger.info(
                                    "ai_recommendation_stored",
                                    recommendation=review.judge_recommendation,
                                    confidence=f"{review.judge_confidence:.2f}",
                                    ttl_minutes=self._ai_recommendation_ttl_minutes,
                                )
                            # Update cooldown tracking for interesting_hold
                            self._last_interesting_hold_review = datetime.now(timezone.utc)
                            self._last_interesting_hold_score = signal_result.score
                            return

                    except Exception as e:
                        logger.error("claude_review_failed", error=str(e), exc_info=True)

                        # Check AI failure mode setting (per-action)
                        # Only apply failure mode logic to actual trades, not holds
                        if effective_action == "buy":
                            failure_mode = self.settings.ai_failure_mode_buy
                        elif effective_action == "sell":
                            failure_mode = self.settings.ai_failure_mode_sell
                        else:
                            # Hold - AI review was for "interesting_hold" or debug mode
                            # AI failure only affects threshold adjustments, not trade execution
                            # No trade to skip or proceed with, so return early
                            logger.info("ai_review_failed_for_hold", action=effective_action)
                            return

                        if failure_mode == AIFailureMode.SAFE:
                            logger.warning(
                                "trade_skipped_ai_unavailable",
                                signal_score=signal_result.score,
                                action=effective_action,
                                failure_mode="safe",
                            )

                            # Rate-limit notifications to avoid spam (max once per 15 minutes)
                            # Note: _last_ai_failure_notification is intentionally ephemeral (in-memory only)
                            # It resets on bot restart, which is acceptable for notification cooldown
                            now = datetime.now(timezone.utc)
                            should_notify = (
                                self._last_ai_failure_notification is None
                                or (now - self._last_ai_failure_notification).total_seconds() >= 900  # 15 min
                            )
                            if should_notify:
                                self._last_ai_failure_notification = now
                                self.notifier.send_message(
                                    f"⚠️ Trade skipped: AI review unavailable\n"
                                    f"Signal: {effective_action} (score: {signal_result.score})"
                                )

                            return  # Skip this iteration

                        # AIFailureMode.OPEN: Continue with trade (fail-open, current default)
                        logger.info(
                            "trade_proceeding_without_ai_review",
                            signal_score=signal_result.score,
                            action=effective_action,
                            failure_mode="open",
                        )

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

        # Check for dual-extreme conditions before attempting any buy
        # Post-mortem #135: These conditions create unfavorable risk/reward for entries
        if (
            effective_action == "buy"
            and self.settings.block_trades_extreme_conditions
            and regime.components.get("sentiment", {}).get("category") == "extreme_fear"
            and regime.components.get("volatility", {}).get("level") == "extreme"
        ):
            logger.warning(
                "trade_blocked_extreme_conditions",
                reason="extreme_fear_and_extreme_volatility",
                sentiment=regime.components.get("sentiment", {}).get("value"),
                volatility=regime.components.get("volatility", {}).get("level"),
            )
            self.notifier.notify_trade_rejected(
                side="buy",
                reason="extreme_fear + extreme_volatility",
                price=current_price,
                signal_score=signal_result.score,
                is_paper=self.settings.is_paper_trading,
            )
            return

        if effective_action == "buy":
            normal_bot_blocked_by_cooldown = False
            if can_buy:
                # Check trade cooldown
                if self.trade_cooldown:
                    cooldown_ok, cooldown_reason = self.trade_cooldown.can_execute("buy", current_price)
                    if not cooldown_ok:
                        logger.info(
                            "decision",
                            action="skip_buy",
                            reason=f"trade_cooldown: {cooldown_reason}",
                            signal_score=signal_result.score,
                        )
                        self.notifier.notify_trade_rejected(
                            side="buy",
                            reason=cooldown_reason,
                            price=current_price,
                            signal_score=signal_result.score,
                            is_paper=self.settings.is_paper_trading,
                        )
                        normal_bot_blocked_by_cooldown = True

            if can_buy and not normal_bot_blocked_by_cooldown:
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
                    volatility=signal_result.indicators.volatility or "normal",
                )
                # Execute opposite trade for Cramer Mode (if enabled)
                if self.cramer_client:
                    self._execute_cramer_trade(
                        side="sell",  # Opposite of buy
                        candles=candles,
                        current_price=current_price,
                        signal_score=signal_result.score,
                        safety_multiplier=safety_multiplier,
                    )
            # Handle cases where normal bot skips
            if not can_buy or normal_bot_blocked_by_cooldown:
                if not can_buy:
                    logger.info(
                        "decision",
                        action="skip_buy",
                        reason=f"insufficient_balance_or_position_limit (quote={quote_balance}, position={position_percent:.1f}%)",
                        signal_score=signal_result.score,
                    )
                # Cramer Mode may trade independently (see _maybe_execute_cramer_independent docstring)
                self._maybe_execute_cramer_independent(
                    signal_side="buy",
                    candles=candles,
                    current_price=current_price,
                    signal_score=signal_result.score,
                    safety_multiplier=safety_multiplier,
                )

        elif effective_action == "sell":
            normal_bot_blocked_by_cooldown = False
            if can_sell:
                # Check trade cooldown (disabled by default for sells, but here for completeness)
                if self.trade_cooldown:
                    cooldown_ok, cooldown_reason = self.trade_cooldown.can_execute("sell", current_price)
                    if not cooldown_ok:
                        logger.info(
                            "decision",
                            action="skip_sell",
                            reason=f"trade_cooldown: {cooldown_reason}",
                            signal_score=signal_result.score,
                        )
                        self.notifier.notify_trade_rejected(
                            side="sell",
                            reason=cooldown_reason,
                            price=current_price,
                            signal_score=signal_result.score,
                            is_paper=self.settings.is_paper_trading,
                        )
                        normal_bot_blocked_by_cooldown = True

            if can_sell and not normal_bot_blocked_by_cooldown:
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
                # Execute opposite trade for Cramer Mode (if enabled)
                if self.cramer_client:
                    self._execute_cramer_trade(
                        side="buy",  # Opposite of sell
                        candles=candles,
                        current_price=current_price,
                        signal_score=signal_result.score,
                        safety_multiplier=safety_multiplier,
                    )

            # Handle cases where normal bot skips
            if not can_sell or normal_bot_blocked_by_cooldown:
                if not can_sell:
                    logger.info(
                        "decision",
                        action="skip_sell",
                        reason=f"insufficient_base_balance ({base_balance})",
                        signal_score=signal_result.score,
                    )
                # Cramer Mode may trade independently (see _maybe_execute_cramer_independent docstring)
                self._maybe_execute_cramer_independent(
                    signal_side="sell",
                    candles=candles,
                    current_price=current_price,
                    signal_score=signal_result.score,
                    safety_multiplier=safety_multiplier,
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
        volatility: str = "normal",
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

        min_trade_quote = Decimal(str(self.position_sizer.config.min_trade_quote))
        if position.size_quote < min_trade_quote:
            logger.info("buy_skipped", reason="position_too_small", size_quote=str(position.size_quote))
            self.notifier.notify_trade_rejected(
                side="buy",
                reason=f"Position too small (< ${min_trade_quote})",
                price=current_price,
                signal_score=signal_score,
                size_quote=position.size_quote,
                is_paper=self.settings.is_paper_trading,
            )
            return

        # Validate order
        order_request = OrderRequest(
            side="buy",
            size=position.size_base,
            order_type="market",
        )

        # Calculate stop distance for profit margin validation
        stop_distance_percent = None
        if current_price > 0 and position.stop_loss_price > 0:
            stop_distance_percent = float(
                abs(current_price - position.stop_loss_price) / current_price
            )

        validation = self.validator.validate(order_request, stop_distance_percent)
        if not validation.valid:
            logger.info("buy_rejected", reason=validation.reason)
            self.notifier.notify_trade_rejected(
                side="buy",
                reason=validation.reason,
                price=current_price,
                signal_score=signal_score,
                size_quote=position.size_quote,
                is_paper=self.settings.is_paper_trading,
            )
            return

        # Execute order (IOC limit -> market fallback)
        if self.settings.use_limit_orders:
            try:
                # Get market data for limit price calculation
                market_data = self.client.get_market_data(self.settings.trading_pair)
                offset = Decimal(str(self.settings.limit_order_offset_percent)) / 100

                # Try IOC at ask + offset
                limit_price = (market_data.ask * (1 + offset)).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                ).normalize()
                base_size = (position.size_quote / limit_price).quantize(Decimal("0.00000001"))

                result = self.client.limit_buy_ioc(
                    self.settings.trading_pair,
                    base_size,
                    limit_price,
                )

                if result.success and result.size > Decimal("0"):
                    logger.info(
                        "limit_buy_ioc_filled",
                        limit_price=str(limit_price),
                        filled_price=str(result.filled_price),
                        size=str(result.size),
                    )
                else:
                    # IOC failed, fall back to market
                    logger.warning(
                        "limit_buy_ioc_fallback",
                        reason="unfilled" if result.success else result.error,
                        limit_price=str(limit_price),
                        fallback="market",
                    )
                    result = self.client.market_buy(
                        self.settings.trading_pair,
                        position.size_quote,
                    )
            except Exception as e:
                # Fall back to market order if limit order setup fails
                logger.warning(
                    "limit_buy_setup_failed",
                    error=str(e),
                    fallback="market_order",
                )
                result = self.client.market_buy(
                    self.settings.trading_pair,
                    position.size_quote,
                )
        else:
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
                bot_mode=BotMode.NORMAL,
                quote_balance_after=new_quote_balance,
                base_balance_after=new_base_balance,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=is_paper, bot_mode=BotMode.NORMAL)

            # Update trade cooldown
            if self.trade_cooldown:
                self.trade_cooldown.record_trade("buy", filled_price)

            # Update position tracking and create stop protection
            # CRITICAL: If stop creation fails, immediately close position (fail-safe)
            try:
                new_avg_cost = self._update_position_after_buy(result.size, filled_price, result.fee, is_paper, bot_mode=BotMode.NORMAL)
                take_profit_price = position.take_profit_price if (self.settings.enable_take_profit and position.take_profit_price) else None
                self._create_trailing_stop(
                    filled_price,
                    candles,
                    is_paper,
                    avg_cost=new_avg_cost,
                    volatility=volatility,
                    take_profit_price=take_profit_price,
                    bot_mode=BotMode.NORMAL,
                )
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
                signal_score=signal_score,
                stop_loss=position.stop_loss_price,
                take_profit=position.take_profit_price,
                position_percent=position.position_percent,
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

            # Mark signal history as having resulted in trade (for post-mortem analysis)
            self._mark_signal_trade_executed(self._current_signal_id)

            # Run post-mortem analysis asynchronously (if enabled)
            self._run_postmortem_async()
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

        min_base = Decimal(str(self.position_sizer.config.min_trade_base))
        if size_base < min_base:
            logger.info("sell_skipped", reason="position_too_small", size_base=str(size_base), min_base=str(min_base))
            self.notifier.notify_trade_rejected(
                side="sell",
                reason=f"Position too small (< {min_base} BTC)",
                price=current_price,
                signal_score=signal_score,
                size_quote=size_base * current_price,
                is_paper=self.settings.is_paper_trading,
            )
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
            self.notifier.notify_trade_rejected(
                side="sell",
                reason=validation.reason,
                price=current_price,
                signal_score=signal_score,
                size_quote=size_base * current_price,
                is_paper=self.settings.is_paper_trading,
            )
            return

        # Execute order (IOC limit -> market fallback)
        if self.settings.use_limit_orders:
            try:
                # Get market data for limit price calculation
                market_data = self.client.get_market_data(self.settings.trading_pair)
                offset = Decimal(str(self.settings.limit_order_offset_percent)) / 100

                # Try IOC at bid - offset
                limit_price = (market_data.bid * (1 - offset)).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                ).normalize()

                result = self.client.limit_sell_ioc(
                    self.settings.trading_pair,
                    size_base,
                    limit_price,
                )

                if result.success and result.size > Decimal("0"):
                    logger.info(
                        "limit_sell_ioc_filled",
                        limit_price=str(limit_price),
                        filled_price=str(result.filled_price),
                        size=str(result.size),
                    )
                else:
                    # IOC failed, fall back to market
                    logger.warning(
                        "limit_sell_ioc_fallback",
                        reason="unfilled" if result.success else result.error,
                        limit_price=str(limit_price),
                        fallback="market",
                    )
                    result = self.client.market_sell(
                        self.settings.trading_pair,
                        size_base,
                    )
            except Exception as e:
                # Fall back to market order if limit order setup fails
                logger.warning(
                    "limit_sell_setup_failed",
                    error=str(e),
                    fallback="market_order",
                )
                result = self.client.market_sell(
                    self.settings.trading_pair,
                    size_base,
                )
        else:
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
                bot_mode=BotMode.NORMAL,
            )

            # Get entry price before calculating realized PnL
            # (realized PnL calculation updates the position, reducing quantity)
            current_position = self.db.get_current_position(self.settings.trading_pair, is_paper=is_paper, bot_mode=BotMode.NORMAL)
            entry_price = current_position.get_average_cost() if current_position else None

            # Calculate realized P&L based on average cost basis
            realized_pnl = self._calculate_realized_pnl(result.size, filled_price, result.fee, is_paper, bot_mode=BotMode.NORMAL)

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
                bot_mode=BotMode.NORMAL,
                quote_balance_after=new_quote_balance,
                base_balance_after=new_base_balance,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=is_paper, bot_mode=BotMode.NORMAL)

            # Update loss limiter with actual PnL
            self.loss_limiter.record_trade(
                realized_pnl=realized_pnl,
                side="sell",
                size=result.size,
                price=filled_price,
            )

            # Update trade cooldown
            if self.trade_cooldown:
                self.trade_cooldown.record_trade("sell", filled_price)

            # Send notification
            self.notifier.notify_trade(
                side="sell",
                size=result.size,
                price=filled_price,
                fee=result.fee,
                is_paper=is_paper,
                signal_score=signal_score,
                realized_pnl=realized_pnl,
                entry_price=entry_price,
            )

            logger.info(
                "sell_executed",
                size=str(result.size),
                price=str(result.filled_price),
                fee=str(result.fee),
                realized_pnl=str(realized_pnl),
            )

            self.circuit_breaker.record_order_success()

            # Mark signal history as having resulted in trade (for post-mortem analysis)
            self._mark_signal_trade_executed(self._current_signal_id)

            # Run post-mortem analysis asynchronously (if enabled)
            self._run_postmortem_async()
        else:
            logger.error("sell_failed", error=result.error)
            self.circuit_breaker.record_order_failure()
            self.notifier.notify_order_failed("sell", size_base, result.error or "Unknown error")

    def _maybe_execute_cramer_independent(
        self,
        signal_side: str,
        candles,
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """
        Execute Cramer Mode trade (inverse of signal) if conditions allow.

        This is called when the normal bot CANNOT execute (blocked by balance,
        position limits, or cooldown) but Cramer may still trade independently.

        Cramer Mode is for PAPER MODE strategy comparison only - it uses virtual
        balance and does not affect real funds.

        Args:
            signal_side: The direction the normal bot would have traded ("buy" or "sell")
            candles: OHLCV data for position sizing
            current_price: Current market price
            signal_score: Signal strength from scorer
            safety_multiplier: Combined safety multiplier (circuit breaker, judge, regime)
        """
        if not self.cramer_client:
            return

        # Cramer trades inverse: signal "buy" -> Cramer "sell", and vice versa
        cramer_side = "sell" if signal_side == "buy" else "buy"

        if safety_multiplier > 0:
            self._execute_cramer_trade(
                side=cramer_side,
                candles=candles,
                current_price=current_price,
                signal_score=signal_score,
                safety_multiplier=safety_multiplier,
            )
        else:
            logger.warning(
                "cramer_trade_blocked_by_safety",
                side=cramer_side,
                safety_multiplier=safety_multiplier,
                reason="circuit_breaker_active",
            )

    def _execute_cramer_trade(
        self,
        side: str,
        candles,
        current_price: Decimal,
        signal_score: int,
        safety_multiplier: float,
    ) -> None:
        """
        Execute a trade for the Cramer Mode (inverted mode).

        This executes the OPPOSITE trade of what the normal bot just did,
        using the Cramer Mode's separate virtual balance.
        """
        if not self.cramer_client:
            return

        if self._cramer_mode_disabled:
            return  # Cramer Mode disabled due to balance mismatch

        # Get Cramer Mode balances
        cramer_quote_balance = self.cramer_client.get_balance(self._quote_currency).available
        cramer_base_balance = self.cramer_client.get_balance(self._base_currency).available

        logger.debug(
            "cramer_trade_attempt",
            side=side,
            quote_balance=str(cramer_quote_balance),
            base_balance=str(cramer_base_balance),
        )

        if side == "buy":
            # Check if Cramer Mode can buy (same constraints as normal bot)
            min_quote = Decimal(str(self.position_sizer.config.min_trade_quote))
            if cramer_quote_balance < min_quote:
                logger.info("cramer_skip_buy", reason="insufficient_quote_balance")
                return

            # Check Cramer Mode independent cooldown
            if self.cramer_trade_cooldown:
                cooldown_ok, cooldown_reason = self.cramer_trade_cooldown.can_execute("buy", current_price)
                if not cooldown_ok:
                    logger.info("cramer_skip_buy", reason=f"cooldown: {cooldown_reason}")
                    return

            # Calculate position size for Cramer Mode buy
            position = self.position_sizer.calculate_size(
                df=candles,
                current_price=current_price,
                quote_balance=cramer_quote_balance,
                base_balance=cramer_base_balance,
                signal_strength=abs(signal_score),
                side="buy",
                safety_multiplier=safety_multiplier,
            )

            if position.size_quote < min_quote:
                logger.info("cramer_skip_buy", reason="position_too_small")
                return

            # Execute buy on Cramer Mode client
            result = self.cramer_client.market_buy(
                self.settings.trading_pair,
                position.size_quote,
            )

            if result.success:
                filled_price = result.filled_price or current_price

                # Get new balances
                new_quote = self.cramer_client.get_balance(self._quote_currency).available
                new_base = self.cramer_client.get_balance(self._base_currency).available

                # Verify balance consistency BEFORE DB writes (defensive check)
                # If mismatch detected, skip all DB operations to preserve data integrity
                actual_quote = self.cramer_client.get_balance(self._quote_currency).available
                actual_base = self.cramer_client.get_balance(self._base_currency).available
                if actual_quote != new_quote or actual_base != new_base:
                    logger.error(
                        "cramer_balance_mismatch",
                        expected_quote=str(new_quote),
                        actual_quote=str(actual_quote),
                        expected_base=str(new_base),
                        actual_base=str(actual_base),
                    )
                    logger.warning("cramer_mode_disabled_due_to_mismatch")
                    self._cramer_mode_disabled = True
                    return

                # Update Cramer Mode position
                new_avg_cost = self._update_position_after_buy(
                    result.size, filled_price, result.fee,
                    is_paper=True, bot_mode=BotMode.INVERTED
                )

                # Record trade
                self.db.record_trade(
                    side="buy",
                    size=result.size,
                    price=filled_price,
                    fee=result.fee,
                    symbol=self.settings.trading_pair,
                    is_paper=True,
                    bot_mode=BotMode.INVERTED,
                    quote_balance_after=new_quote,
                    base_balance_after=new_base,
                    spot_rate=current_price,
                )
                self.db.increment_daily_trade_count(is_paper=True, bot_mode=BotMode.INVERTED)

                # Create trailing stop for Cramer Mode
                take_profit_price = position.take_profit_price if (self.settings.enable_take_profit and position.take_profit_price) else None
                self._create_trailing_stop(
                    filled_price,
                    candles,
                    is_paper=True,
                    avg_cost=new_avg_cost,
                    volatility=self._last_volatility,
                    take_profit_price=take_profit_price,
                    bot_mode=BotMode.INVERTED,
                )

                logger.info(
                    "cramer_buy_executed",
                    size=str(result.size),
                    price=str(filled_price),
                    fee=str(result.fee),
                )

                # Update Cramer cooldown
                if self.cramer_trade_cooldown:
                    self.cramer_trade_cooldown.record_trade("buy", filled_price)

                # Update Cramer Mode daily stats with new ending balance
                cramer_ending_balance = self._get_cramer_portfolio_value()
                self.db.update_daily_stats(
                    ending_balance=cramer_ending_balance,
                    ending_price=current_price,
                    is_paper=True,
                    bot_mode=BotMode.INVERTED,
                )
            else:
                logger.warning("cramer_buy_failed", error=result.error)

        elif side == "sell":
            # Check if Cramer Mode can sell
            min_base = Decimal(str(self.position_sizer.config.min_trade_base))
            if cramer_base_balance < min_base:
                logger.info("cramer_skip_sell", reason="insufficient_base_balance")
                return

            # Check Cramer Mode independent cooldown
            if self.cramer_trade_cooldown:
                cooldown_ok, cooldown_reason = self.cramer_trade_cooldown.can_execute("sell", current_price)
                if not cooldown_ok:
                    logger.info("cramer_skip_sell", reason=f"cooldown: {cooldown_reason}")
                    return

            # For aggressive selling, sell entire position on strong signal
            if abs(signal_score) >= 80:
                size_base = cramer_base_balance
            else:
                position = self.position_sizer.calculate_size(
                    df=candles,
                    current_price=current_price,
                    quote_balance=Decimal("0"),
                    base_balance=cramer_base_balance,
                    signal_strength=abs(signal_score),
                    side="sell",
                    safety_multiplier=safety_multiplier,
                )
                size_base = position.size_base

            if size_base < min_base:
                logger.info("cramer_skip_sell", reason="position_too_small")
                return

            # Execute sell on Cramer Mode client
            result = self.cramer_client.market_sell(
                self.settings.trading_pair,
                size_base,
            )

            if result.success:
                filled_price = result.filled_price or current_price

                # Get new balances
                new_quote = self.cramer_client.get_balance(self._quote_currency).available
                new_base = self.cramer_client.get_balance(self._base_currency).available

                # Verify balance consistency BEFORE DB writes (defensive check)
                # If mismatch detected, skip all DB operations to preserve data integrity
                actual_quote = self.cramer_client.get_balance(self._quote_currency).available
                actual_base = self.cramer_client.get_balance(self._base_currency).available
                if actual_quote != new_quote or actual_base != new_base:
                    logger.error(
                        "cramer_balance_mismatch",
                        expected_quote=str(new_quote),
                        actual_quote=str(actual_quote),
                        expected_base=str(new_base),
                        actual_base=str(actual_base),
                    )
                    logger.warning("cramer_mode_disabled_due_to_mismatch")
                    self._cramer_mode_disabled = True
                    return

                # Calculate realized P&L for Cramer Mode
                realized_pnl = self._calculate_realized_pnl(
                    result.size, filled_price, result.fee,
                    is_paper=True, bot_mode=BotMode.INVERTED
                )

                # Deactivate trailing stop
                self.db.deactivate_trailing_stop(
                    self.settings.trading_pair, is_paper=True, bot_mode=BotMode.INVERTED
                )

                # Record trade
                self.db.record_trade(
                    side="sell",
                    size=result.size,
                    price=filled_price,
                    fee=result.fee,
                    realized_pnl=realized_pnl,
                    symbol=self.settings.trading_pair,
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

                # Update Cramer cooldown
                if self.cramer_trade_cooldown:
                    self.cramer_trade_cooldown.record_trade("sell", filled_price)

                # Update Cramer Mode daily stats with new ending balance
                cramer_ending_balance = self._get_cramer_portfolio_value()
                self.db.update_daily_stats(
                    ending_balance=cramer_ending_balance,
                    ending_price=current_price,
                    is_paper=True,
                    bot_mode=BotMode.INVERTED,
                )
            else:
                logger.warning("cramer_sell_failed", error=result.error)

    def _execute_cramer_trailing_stop_sell(
        self,
        candles,
        current_price: Decimal,
        base_balance: Decimal,
    ) -> None:
        """
        Execute Cramer Mode sell from independent trailing stop trigger.

        This is NOT a mirror of normal bot action - it's Cramer Mode's own
        risk management exiting its position.
        """
        if not self.cramer_client:
            return

        if self._cramer_mode_disabled:
            return  # Cramer Mode disabled due to balance mismatch

        # Sell entire position (trailing stop = full exit)
        result = self.cramer_client.market_sell(
            self.settings.trading_pair,
            base_balance,
        )

        if result.success:
            filled_price = result.filled_price or current_price

            # Get new balances
            new_quote = self.cramer_client.get_balance(self._quote_currency).available
            new_base = self.cramer_client.get_balance(self._base_currency).available

            # Verify balance consistency BEFORE DB writes (defensive check)
            # If mismatch detected, skip all DB operations to preserve data integrity
            actual_quote = self.cramer_client.get_balance(self._quote_currency).available
            actual_base = self.cramer_client.get_balance(self._base_currency).available
            if actual_quote != new_quote or actual_base != new_base:
                logger.error(
                    "cramer_balance_mismatch",
                    expected_quote=str(new_quote),
                    actual_quote=str(actual_quote),
                    expected_base=str(new_base),
                    actual_base=str(actual_base),
                )
                logger.warning("cramer_mode_disabled_due_to_mismatch")
                self._cramer_mode_disabled = True
                return

            # Calculate realized P&L
            realized_pnl = self._calculate_realized_pnl(
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
                symbol=self.settings.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
                quote_balance_after=new_quote,
                base_balance_after=new_base,
                spot_rate=current_price,
            )
            self.db.increment_daily_trade_count(is_paper=True, bot_mode=BotMode.INVERTED)

            # Deactivate trailing stop (critical: prevent repeated triggers)
            self.db.deactivate_trailing_stop(
                symbol=self.settings.trading_pair,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
            )

            # Record to Cramer Mode cooldown
            if self.cramer_trade_cooldown:
                self.cramer_trade_cooldown.record_trade("sell", filled_price)

            logger.info(
                "cramer_trailing_stop_sell_executed",
                size=str(result.size),
                price=str(filled_price),
                fee=str(result.fee),
                realized_pnl=str(realized_pnl),
            )

            # Update Cramer Mode daily stats with new ending balance
            cramer_ending_balance = self._get_cramer_portfolio_value()
            self.db.update_daily_stats(
                ending_balance=cramer_ending_balance,
                ending_price=current_price,
                is_paper=True,
                bot_mode=BotMode.INVERTED,
            )
        else:
            logger.warning("cramer_trailing_stop_sell_failed", error=result.error)

    def _update_position_after_buy(
        self, size: Decimal, price: Decimal, fee: Decimal, is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
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
        current = self.db.get_current_position(self.settings.trading_pair, is_paper=is_paper, bot_mode=bot_mode)

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
            bot_mode=bot_mode,
        )

        logger.debug(
            "position_updated_after_buy",
            old_qty=str(old_qty),
            new_qty=str(new_qty),
            new_avg_cost=str(new_avg_cost),
        )

        return new_avg_cost

    def _calculate_realized_pnl(
        self, size: Decimal, sell_price: Decimal, fee: Decimal, is_paper: bool = False, bot_mode: BotMode = BotMode.NORMAL
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
        current = self.db.get_current_position(self.settings.trading_pair, is_paper=is_paper, bot_mode=bot_mode)

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

    def _get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in quote currency."""
        base_balance = self.client.get_balance(self._base_currency).available
        quote_balance = self.client.get_balance(self._quote_currency).available
        current_price = self.client.get_current_price(self.settings.trading_pair)

        base_value = base_balance * current_price
        return quote_balance + base_value

    def _get_cramer_portfolio_value(self) -> Decimal:
        """Get Cramer Mode total portfolio value in quote currency."""
        if not self.cramer_client:
            return Decimal("0")

        base_balance = self.cramer_client.get_balance(self._base_currency).available
        quote_balance = self.cramer_client.get_balance(self._quote_currency).available
        current_price = self.client.get_current_price(self.settings.trading_pair)

        base_value = base_balance * current_price
        return quote_balance + base_value

    def _get_cramer_portfolio_info(self, current_price: Decimal) -> Optional[dict]:
        """Get Cramer Mode portfolio info for dashboard display."""
        if not self.cramer_client:
            return None

        try:
            base_balance = self.cramer_client.get_balance(self._base_currency).available
            quote_balance = self.cramer_client.get_balance(self._quote_currency).available
            base_value = base_balance * current_price
            portfolio_value = quote_balance + base_value

            # Calculate position percent (how much is in BTC)
            position_percent = float(base_value / portfolio_value * 100) if portfolio_value > 0 else 0.0

            return {
                "quote_balance": str(quote_balance),
                "base_balance": str(base_balance),
                "portfolio_value": str(portfolio_value),
                "position_percent": position_percent,
            }
        except Exception as e:
            logger.warning("cramer_portfolio_fetch_error", error=str(e))
            return None

    def _check_daily_report(self) -> None:
        """Check if we should generate daily performance report (UTC)."""
        today = datetime.now(timezone.utc).date()

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
        """Send daily performance report via unified notification method."""
        pnl = ending_balance - starting_balance

        self.notifier.notify_periodic_report(
            period="Daily",
            report_date=f"Date: {report_date}",
            portfolio_return=portfolio_return,
            btc_return=btc_return,
            alpha=alpha,
            pnl=pnl,
            ending_balance=ending_balance,
            trades=trades,
            is_paper=self.settings.is_paper_trading,
        )

    def _check_weekly_report(self) -> None:
        """Check if we should generate weekly performance report (on Mondays, UTC)."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()

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
        """Check if we should generate monthly performance report (on 1st of month, UTC)."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()

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
        """Send weekly/monthly performance report via unified notification method."""
        pnl = ending_balance - starting_balance

        self.notifier.notify_periodic_report(
            period=period,
            report_date=f"{start_date} → {end_date}",
            portfolio_return=portfolio_return,
            btc_return=btc_return,
            alpha=alpha,
            pnl=pnl,
            ending_balance=ending_balance,
            trades=trades,
            is_paper=self.settings.is_paper_trading,
        )

    def _check_hourly_analysis(self) -> None:
        """Run hourly market analysis during volatile conditions or post-volatility."""
        # Skip if hourly analysis not enabled
        if not self._hourly_analysis_enabled:
            return

        now = datetime.now(timezone.utc)
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
        volatility: str = "normal",
        take_profit_price: Optional[Decimal] = None,
        bot_mode: BotMode = BotMode.NORMAL,
    ) -> None:
        """Create or update trailing stop for a position.

        Args:
            entry_price: The price at which the buy was executed (for logging)
            candles: pandas DataFrame with OHLCV data for ATR calculation
            is_paper: Whether this is paper trading
            avg_cost: REQUIRED - Weighted average cost for hard stop calculation.
                     Must be passed from caller to avoid race conditions.
                     Do NOT query DB here - caller has the authoritative value.
            volatility: Current volatility level (low/normal/high/extreme)
            take_profit_price: Optional take profit target price
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

            # Hard stop: based on MARKET entry price, NOT fee-inflated cost basis
            # Using avg_cost caused immediate stop triggers because cost basis is ~0.6%
            # higher than market price due to fee inclusion
            #
            # Use the LARGER of ATR-based distance or minimum percentage distance
            # This ensures stop is never too tight on low-volatility timeframes
            #
            # During extreme volatility, use wider stop multiplier to avoid
            # being stopped out by normal price fluctuations
            stop_multiplier = self.settings.stop_loss_atr_multiplier
            if volatility == "extreme":
                stop_multiplier = self.settings.stop_loss_atr_multiplier_extreme
                logger.info(
                    "using_extreme_volatility_stop",
                    volatility=volatility,
                    stop_multiplier=stop_multiplier,
                )
            atr_stop_distance = atr * Decimal(str(stop_multiplier))
            min_pct_distance = entry_price * Decimal(str(self.settings.min_stop_loss_percent)) / Decimal("100")
            stop_distance = max(atr_stop_distance, min_pct_distance)
            hard_stop = entry_price - stop_distance

            # Log which method determined the stop
            if min_pct_distance > atr_stop_distance:
                logger.info(
                    "hard_stop_using_min_percent",
                    atr_distance=str(atr_stop_distance),
                    min_pct_distance=str(min_pct_distance),
                    min_pct=self.settings.min_stop_loss_percent,
                )

            # Check if we're DCA'ing (stop already exists)
            # If so, UPDATE the existing stop in-place to avoid any window
            # where the position is unprotected during the transition
            existing_stop = self.db.get_active_trailing_stop(
                symbol=self.settings.trading_pair, is_paper=is_paper, bot_mode=bot_mode
            )

            if existing_stop:
                # DCA: Recalculate take profit based on new avg_cost (not original entry)
                if self.settings.enable_take_profit:
                    take_profit_price = avg_cost + (atr * Decimal(str(self.settings.take_profit_atr_multiplier)))

                # DCA: Update existing stop without deactivating
                self.db.update_trailing_stop_for_dca(
                    symbol=self.settings.trading_pair,
                    entry_price=avg_cost,
                    trailing_activation=activation,
                    trailing_distance=distance,
                    hard_stop=hard_stop,
                    take_profit_price=take_profit_price,
                    is_paper=is_paper,
                    bot_mode=bot_mode,
                )
                logger.info(
                    "trailing_stop_updated_dca",
                    entry_price=str(entry_price),
                    avg_cost=str(avg_cost),
                    activation=str(activation),
                    distance=str(distance),
                    hard_stop=str(hard_stop),
                    take_profit_price=str(take_profit_price) if take_profit_price else None,
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
                    take_profit_price=take_profit_price,
                    bot_mode=bot_mode,
                )
                logger.info(
                    "trailing_stop_created",
                    entry_price=str(entry_price),
                    avg_cost=str(avg_cost),
                    activation=str(activation),
                    distance=str(distance),
                    hard_stop=str(hard_stop),
                    take_profit_price=str(take_profit_price) if take_profit_price else None,
                )
        except Exception as e:
            # CRITICAL: Re-raise to allow caller to handle (emergency close position)
            logger.error("trailing_stop_creation_failed", error=str(e))
            raise

    def _check_trailing_stop(self, current_price: Decimal, bot_mode: BotMode = BotMode.NORMAL) -> Optional[str]:
        """
        Check and update trailing stop, return action if stop triggered.

        Priority order (for buy positions):
        1. Hard stop (capital protection, triggers sell)
        2. Take profit (profit target, triggers sell)
        3. Break-even trigger (moves hard stop to entry at +0.5 ATR, no sell)
        4. Trailing activation (activates at +1 ATR, no sell)
        5. Trailing update (moves stop up, no sell)
        6. Trailing trigger (locks profit, triggers sell)

        The entry_price stored in trailing_stops is the weighted average cost,
        not the individual entry price, ensuring correct calculations for DCA.

        Args:
            current_price: Current market price
            bot_mode: "normal" or "inverted" (Cramer Mode)

        Returns:
            "sell" if trailing stop or hard stop triggered, None otherwise
        """
        is_paper = self.settings.is_paper_trading
        ts = self.db.get_active_trailing_stop(
            symbol=self.settings.trading_pair,
            is_paper=is_paper,
            bot_mode=bot_mode,
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

                # Send hard stop notification with missed TP target
                take_profit = ts.get_take_profit_price()
                tp_info = f" | TP Target: ¤{take_profit:,.2f}" if take_profit else ""
                self.notifier.send_message(
                    f"🛑 Hard Stop Triggered (-{loss_pct}%)\n"
                    f"Entry: ¤{entry_price:,.2f} | Exit: ¤{current_price:,.2f}\n"
                    f"Stop: ¤{hard_stop:,.2f}{tp_info}"
                )

                self.db.deactivate_trailing_stop(
                    symbol=self.settings.trading_pair,
                    is_paper=is_paper,
                    bot_mode=bot_mode,
                )
                return "sell"

            # CHECK TAKE PROFIT (if enabled) - Priority #2
            if self.settings.enable_take_profit:
                take_profit = ts.get_take_profit_price()
                if take_profit is not None and current_price >= take_profit:
                    # Calculate profit percentage (with validation)
                    if entry_price <= 0:
                        logger.critical(
                            "invalid_entry_price_halting_trading",
                            entry_price=str(entry_price),
                            context="take_profit_check",
                        )
                        self.notifier.send_alert(
                            f"🔴 CRITICAL: Invalid entry_price ({entry_price}) detected. Halting trading!"
                        )
                        self.kill_switch.activate(f"Data corruption: entry_price={entry_price}")
                        profit_pct = Decimal("0")
                    else:
                        profit_pct = ((current_price - entry_price) / entry_price * 100).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )

                    logger.info(
                        "take_profit_triggered",
                        exit_type="profit_target",
                        current_price=str(current_price),
                        take_profit_price=str(take_profit),
                        entry_price=str(entry_price),
                        profit_percent=str(profit_pct),
                    )

                    self.notifier.send_message(
                        f"🎯 Take Profit Target Reached\n"
                        f"Entry: ¤{entry_price:,.2f} | Exit: ¤{current_price:,.2f} (+{profit_pct}%)\n"
                        f"Target: ¤{take_profit:,.2f}"
                    )

                    self.db.deactivate_trailing_stop(
                        symbol=self.settings.trading_pair,
                        is_paper=is_paper,
                        bot_mode=bot_mode,
                    )

                    return "sell"

            # Check break-even trigger (protects capital once in moderate profit)
            # This triggers BEFORE trailing activation (0.5 ATR vs 1 ATR)
            # TODO: Add equivalent logic for short positions when implemented
            if not ts.is_breakeven_active() and distance is not None:
                # Calculate break-even activation: entry + (ATR * breakeven_multiplier)
                # ATR = distance / trailing_stop_atr_multiplier
                breakeven_multiplier = Decimal(str(self.settings.breakeven_atr_multiplier))
                trailing_multiplier = Decimal(str(self.settings.trailing_stop_atr_multiplier))
                atr = distance / trailing_multiplier
                breakeven_activation = entry_price + (atr * breakeven_multiplier)

                if current_price >= breakeven_activation:
                    # Move hard stop to break-even (entry price)
                    self.db.update_trailing_stop_breakeven(ts.id, new_hard_stop=entry_price)
                    # Update local state to prevent duplicate triggers this iteration
                    ts.breakeven_triggered = True
                    profit_pct = ((current_price - entry_price) / entry_price * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    logger.info(
                        "breakeven_stop_activated",
                        current_price=str(current_price),
                        breakeven_activation=str(breakeven_activation),
                        entry_price=str(entry_price),
                        profit_percent=str(profit_pct),
                        breakeven_multiplier=str(breakeven_multiplier),
                        atr=str(atr),
                    )
                    self.notifier.send_message(
                        f"🛡️ Break-even protection activated\n"
                        f"Entry: ¤{entry_price:,.2f} | Current: ¤{current_price:,.2f} (+{profit_pct}%)\n"
                        f"Position now protected at entry"
                    )
                    # Update local hard_stop for subsequent checks this iteration
                    hard_stop = entry_price

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
                    bot_mode=bot_mode,
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
            self.settings.trading_pair, is_paper=is_paper, bot_mode=BotMode.NORMAL
        )

        if not position or position.get_quantity() <= Decimal("0"):
            return  # No position, nothing to check

        # Check if there's an active trailing stop
        ts = self.db.get_active_trailing_stop(
            symbol=self.settings.trading_pair, is_paper=is_paper, bot_mode=BotMode.NORMAL
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

            # Calculate take profit price for emergency recovery
            take_profit_price = None
            if self.settings.enable_take_profit:
                from src.indicators.atr import calculate_atr

                high = candles["high"].astype(float)
                low = candles["low"].astype(float)
                close = candles["close"].astype(float)
                atr_result = calculate_atr(high, low, close, period=self.settings.atr_period)
                atr_value = atr_result.atr.iloc[-1]
                if not math.isnan(atr_value) and atr_value > 0:
                    atr = Decimal(str(atr_value))
                    take_profit_price = avg_cost + (atr * Decimal(str(self.settings.take_profit_atr_multiplier)))

            self._create_trailing_stop(
                entry_price=avg_cost,
                candles=candles,
                is_paper=is_paper,
                avg_cost=avg_cost,
                volatility=self._last_volatility,  # Use last known volatility for appropriate stop width
                take_profit_price=take_profit_price,
                bot_mode=BotMode.NORMAL,
            )
            logger.info("emergency_stop_created", avg_cost=str(avg_cost), take_profit_price=str(take_profit_price) if take_profit_price else None)
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
        now = datetime.now(timezone.utc)

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

            # Save Cramer Mode final state if enabled
            if self.cramer_client and self.settings.enable_cramer_mode:
                cramer_portfolio_value = self._get_cramer_portfolio_value()
                self.db.update_daily_stats(
                    ending_balance=cramer_portfolio_value,
                    ending_price=ending_price,
                    is_paper=True,
                    bot_mode=BotMode.INVERTED,
                )
        except Exception as e:
            logger.error("shutdown_state_save_failed", error=str(e))

        # Shutdown postmortem executor if active
        if self._postmortem_executor:
            try:
                self._postmortem_executor.shutdown(wait=False)
                logger.debug("postmortem_executor_shutdown")
            except Exception as e:
                logger.debug("postmortem_executor_shutdown_failed", error=str(e))

        # Close the async event loop
        try:
            self._loop.close()
        except Exception as e:
            logger.debug("event_loop_close_failed", error=str(e))

        # Send shutdown notification
        self.notifier.notify_shutdown("Graceful shutdown")

        logger.info("daemon_stopped")
