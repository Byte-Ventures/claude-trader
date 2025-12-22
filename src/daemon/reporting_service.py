"""
Reporting service for performance reports and market analysis.

Handles:
- Daily/weekly/monthly performance reports
- Signal history cleanup
- Hourly market analysis during volatile conditions

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Optional, TYPE_CHECKING

import structlog

from src.api.exchange_protocol import ExchangeClient
from src.notifications.telegram import TelegramNotifier
from src.state.database import Database
from src.strategy.regime import get_cached_sentiment
from src.strategy.signal_scorer import SignalScorer

if TYPE_CHECKING:
    from src.ai.trade_reviewer import TradeReviewer

logger = structlog.get_logger(__name__)

# Default timeout for async operations (AI reviews, sentiment fetch)
ASYNC_TIMEOUT_SECONDS = 120


@dataclass
class ReportingConfig:
    """Configuration for reporting service."""

    # Trading context
    is_paper_trading: bool
    trading_pair: str

    # Signal history cleanup
    signal_history_retention_days: int = 90

    # Hourly analysis configuration
    hourly_analysis_enabled: bool = True
    candle_interval: str = "ONE_HOUR"
    candle_limit: int = 100
    regime_sentiment_enabled: bool = True


class ReportingService:
    """
    Service for generating performance reports and market analysis.

    Responsibilities:
    - Daily/weekly/monthly performance reports
    - Signal history cleanup
    - Hourly market analysis during volatile conditions
    """

    def __init__(
        self,
        config: ReportingConfig,
        notifier: TelegramNotifier,
        db: Database,
        exchange_client: ExchangeClient,
        signal_scorer: SignalScorer,
        trade_reviewer: Optional["TradeReviewer"] = None,
        on_sentiment_success: Optional[Callable[[], None]] = None,
        on_sentiment_failure: Optional[Callable[[str], None]] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize reporting service.

        Args:
            config: Reporting configuration
            notifier: Telegram notifier for sending messages
            db: Database for querying stats
            exchange_client: Exchange client for fetching candles
            signal_scorer: Signal scorer for calculating indicators
            trade_reviewer: Optional AI trade reviewer for hourly analysis
            on_sentiment_success: Callback when sentiment fetch succeeds
            on_sentiment_failure: Callback when sentiment fetch fails
            event_loop: Optional event loop to use (if None, creates a new one)
        """
        self.config = config
        self.notifier = notifier
        self.db = db
        self.client = exchange_client
        self.signal_scorer = signal_scorer
        self.trade_reviewer = trade_reviewer

        # Callbacks for sentiment tracking (owned by TradingDaemon)
        self._on_sentiment_success = on_sentiment_success
        self._on_sentiment_failure = on_sentiment_failure

        # Report state (migrated from TradingDaemon)
        self._last_daily_report: Optional[date] = None
        self._last_weekly_report: Optional[date] = None
        self._last_monthly_report: Optional[date] = None
        self._last_cleanup_date: Optional[date] = None

        # Hourly analysis state
        self._last_hourly_analysis: Optional[datetime] = None
        self._pending_post_volatility_analysis: bool = False

        # Use provided event loop or create a new one
        self._loop = event_loop if event_loop is not None else asyncio.new_event_loop()
        self._owns_loop = event_loop is None  # Track if we own the loop (for cleanup)

    def close(self) -> None:
        """
        Clean up resources.

        Only closes the event loop if this service created it (not shared).
        """
        if self._owns_loop:
            try:
                self._loop.close()
                logger.debug("reporting_service_loop_closed")
            except Exception as e:
                logger.debug("reporting_service_loop_close_failed", error=str(e))

    def _run_async_with_timeout(self, coro, timeout: int = ASYNC_TIMEOUT_SECONDS, default=None):
        """
        Run an async coroutine with timeout protection.

        Prevents service from hanging indefinitely if external APIs are slow.

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

    # -------------------------------------------------------------------------
    # Public Check Methods
    # -------------------------------------------------------------------------

    def check_daily_report(self) -> None:
        """Check if we should generate daily performance report (UTC)."""
        today = datetime.now(timezone.utc).date()

        # Only report once per day
        if self._last_daily_report == today:
            return

        # Get yesterday's stats (if exists)
        yesterday = today - timedelta(days=1)
        stats = self.db.get_daily_stats(yesterday, is_paper=self.config.is_paper_trading)

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

    def check_weekly_report(self) -> None:
        """Check if we should generate weekly performance report (on Mondays, UTC)."""
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

    def check_monthly_report(self) -> None:
        """Check if we should generate monthly performance report (on 1st of month, UTC)."""
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

    def check_signal_history_cleanup(self) -> None:
        """Check if we should clean up old signal history records (once per day, UTC)."""
        today = datetime.now(timezone.utc).date()

        # Only clean up once per day
        if self._last_cleanup_date == today:
            return

        try:
            deleted = self.db.cleanup_signal_history(
                retention_days=self.config.signal_history_retention_days,
                is_paper=self.config.is_paper_trading
            )
            if deleted > 0:
                logger.info("signal_history_cleanup", deleted_count=deleted)
            self._last_cleanup_date = today
        except Exception as e:
            logger.error("signal_history_cleanup_failed", error=str(e))

    def check_hourly_analysis(self, volatility: str, regime: str) -> None:
        """
        Run hourly market analysis during volatile conditions or post-volatility.

        Args:
            volatility: Current volatility level (low/normal/high/extreme)
            regime: Current market regime
        """
        # Skip if hourly analysis not enabled
        if not self.config.hourly_analysis_enabled or self.trade_reviewer is None:
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
        elif volatility in ("high", "extreme"):
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
            current_price = self.client.get_current_price(self.config.trading_pair)
            candles = self.client.get_candles(
                self.config.trading_pair,
                granularity=self.config.candle_interval,
                limit=self.config.candle_limit,
            )

            # Calculate indicators
            signal_result = self.signal_scorer.calculate_score(candles, current_price)
            indicators = signal_result.indicators

            # Get sentiment
            sentiment_value = None
            sentiment_class = "Unknown"
            if self.config.regime_sentiment_enabled:
                try:
                    sentiment_result = self._run_async_with_timeout(get_cached_sentiment(), timeout=30)
                    if sentiment_result and sentiment_result.value:
                        sentiment_value = sentiment_result.value
                        sentiment_class = sentiment_result.classification or "Unknown"
                        # Record success and handle recovery notification
                        if self._on_sentiment_success:
                            self._on_sentiment_success()
                    else:
                        logger.warning(
                            "sentiment_unavailable_for_dashboard",
                            reason="fetch_returned_none",
                            impact="extreme_fear_override_disabled",
                        )
                        # Record failure atomically (increments counter and checks threshold)
                        if self._on_sentiment_failure:
                            self._on_sentiment_failure("dashboard")
                except Exception as e:
                    logger.warning(
                        "sentiment_fetch_failed_during_dashboard",
                        error=str(e),
                        impact="extreme_fear_override_disabled",
                    )
                    # Record failure atomically (increments counter and checks threshold)
                    if self._on_sentiment_failure:
                        self._on_sentiment_failure("dashboard")

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
                    regime=regime,
                    volatility=volatility,
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
                volatility=volatility,
                fear_greed=sentiment_value or 50,
                fear_greed_class=sentiment_class,
                current_price=current_price,
                analysis_reason=analysis_reason,
            )

        except Exception as e:
            logger.error("hourly_analysis_failed", error=str(e))

        # Update timestamp regardless of success/failure to prevent spam on errors
        self._last_hourly_analysis = now

    def set_pending_post_volatility_analysis(self, value: bool) -> None:
        """Set the pending post-volatility analysis flag."""
        self._pending_post_volatility_analysis = value

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

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
            is_paper=self.config.is_paper_trading,
        )

    def _generate_period_report(self, period: str, start_date: date, end_date: date) -> None:
        """Generate and send a performance report for a date range."""
        stats_list = self.db.get_daily_stats_range(
            start_date, end_date, is_paper=self.config.is_paper_trading
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
            is_paper=self.config.is_paper_trading,
        )
