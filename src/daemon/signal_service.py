"""
Signal history service for storing and managing trading signals.

Handles:
- Signal history storage for post-mortem analysis
- Signal trade execution marking
- Failure tracking and alerting

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy.exc import SQLAlchemyError

from src.notifications.telegram import TelegramNotifier
from src.state.database import Database, SignalHistory
from src.strategy.signal_scorer import SignalResult

logger = structlog.get_logger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal history service."""

    # Trading context
    trading_pair: str
    is_paper_trading: bool

    # Failure alerting
    signal_history_failure_threshold: int = 10


class SignalService:
    """
    Service for storing and managing signal history.

    Responsibilities:
    - Store signal calculations for historical analysis
    - Mark signals that resulted in trades
    - Track and alert on consecutive storage failures
    """

    def __init__(
        self,
        config: SignalConfig,
        db: Database,
        notifier: TelegramNotifier,
    ):
        """
        Initialize signal service.

        Args:
            config: Signal service configuration
            db: Database for storing signal history
            notifier: Telegram notifier for failure alerts
        """
        self.config = config
        self.db = db
        self.notifier = notifier

        # Failure tracking
        self._signal_history_failures: int = 0

    def update_config(self, config: SignalConfig) -> None:
        """
        Update the signal configuration.

        Used for hot-reload of settings.

        Args:
            config: New signal configuration
        """
        self.config = config
        logger.info("signal_service_config_updated")

    def store_signal(
        self,
        signal_result: SignalResult,
        current_price: Decimal,
        htf_bias: str,
        daily_trend: Optional[str],
        four_hour_trend: Optional[str],
        threshold: int,
        trade_executed: bool = False,
    ) -> Optional[int]:
        """
        Store signal calculation for historical analysis.

        Called every iteration to enable post-mortem analysis of trades.

        Args:
            signal_result: The calculated signal with scores and metadata
            current_price: Current market price
            htf_bias: Higher timeframe bias (bullish/bearish/neutral)
            daily_trend: Daily trend direction
            four_hour_trend: 4-hour trend direction
            threshold: Signal threshold used for trading decision
            trade_executed: Whether a trade was executed based on this signal

        Returns:
            The signal history record ID, or None if storage failed.
        """
        try:
            breakdown = signal_result.breakdown
            with self.db.session() as session:
                history = SignalHistory(
                    symbol=self.config.trading_pair,
                    is_paper=self.config.is_paper_trading,
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
                    momentum_mode_adj=breakdown.get("_momentum_active", 0),
                    whale_activity_adj=breakdown.get("_whale_activity", 0),
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
            threshold_val = self.config.signal_history_failure_threshold
            if self._signal_history_failures == threshold_val or (
                self._signal_history_failures > threshold_val
                and (self._signal_history_failures - threshold_val) % 50 == 0
            ):
                self.notifier.notify_error(
                    f"Signal history storage failing ({self._signal_history_failures} consecutive failures)",
                    context=f"Last error: {str(e)}",
                )
            return None

    def mark_trade_executed(self, signal_id: Optional[int]) -> None:
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

    @property
    def consecutive_failures(self) -> int:
        """Get the current count of consecutive storage failures."""
        return self._signal_history_failures
