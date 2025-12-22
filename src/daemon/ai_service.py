"""
AI integration service for managing AI recommendations, veto cooldowns, and sentiment tracking.

Handles:
- AI recommendation state and threshold adjustments
- Veto cooldown after AI rejections
- Sentiment fetch failure tracking and alerting

Extracted from TradingDaemon as part of the runner.py refactoring (Issue #58).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import pandas as pd
import structlog

from src.notifications.telegram import TelegramNotifier

logger = structlog.get_logger(__name__)


# Candle granularity to seconds mapping
GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}


@dataclass
class AIConfig:
    """Configuration for AI service."""

    # Candle interval for veto cooldown calculation
    candle_interval: str = "ONE_HOUR"

    # Veto cooldown
    ai_review_rejection_cooldown: bool = True

    # AI recommendation TTL
    ai_recommendation_ttl_minutes: int = 60

    # Sentiment alerting
    sentiment_failure_alert_threshold: int = 5


class AIService:
    """
    Service for managing AI-related state and logic.

    Responsibilities:
    - Track AI recommendations and calculate threshold adjustments
    - Manage veto cooldown after AI rejections
    - Track sentiment fetch failures and recovery
    """

    def __init__(
        self,
        config: AIConfig,
        notifier: TelegramNotifier,
    ):
        """
        Initialize AI service.

        Args:
            config: AI service configuration
            notifier: Telegram notifier for alerts
        """
        self.config = config
        self.notifier = notifier

        # AI recommendation state
        self._ai_recommendation: Optional[str] = None  # "accumulate", "reduce", "wait"
        self._ai_recommendation_confidence: float = 0.0
        self._ai_recommendation_time: Optional[datetime] = None

        # Veto cooldown state
        self._last_veto_timestamp: Optional[datetime] = None
        self._last_veto_direction: Optional[str] = None  # "buy" or "sell"

        # Sentiment tracking state
        self._sentiment_fetch_failures: int = 0
        self._last_sentiment_fetch_success: Optional[datetime] = None
        self._sentiment_lock = Lock()

    def update_config(self, config: AIConfig) -> None:
        """
        Update the AI configuration.

        Used for hot-reload of settings.

        Args:
            config: New AI configuration
        """
        self.config = config
        logger.info("ai_service_config_updated")

    def get_threshold_adjustment(self, action: str) -> int:
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
        if elapsed > self.config.ai_recommendation_ttl_minutes:
            # Clear expired recommendation
            self._ai_recommendation = None
            self._ai_recommendation_time = None
            return 0

        # Calculate decay factor (1.0 at start, 0.0 at TTL)
        decay = 1.0 - (elapsed / self.config.ai_recommendation_ttl_minutes)

        # Base adjustment of 15 points, scaled by confidence and decay
        base_adjustment = 15
        adjustment = int(base_adjustment * self._ai_recommendation_confidence * decay)

        if self._ai_recommendation == "accumulate" and action == "buy":
            return -adjustment  # Lower buy threshold (easier to buy)
        elif self._ai_recommendation == "reduce" and action == "sell":
            return -adjustment  # Lower sell threshold (easier to sell)

        return 0

    def set_recommendation(
        self,
        recommendation: str,
        confidence: float,
    ) -> None:
        """
        Set a new AI recommendation.

        Args:
            recommendation: "accumulate", "reduce", or "wait"
            confidence: Confidence level (0.0 to 1.0)
        """
        self._ai_recommendation = recommendation
        self._ai_recommendation_confidence = confidence
        self._ai_recommendation_time = datetime.now(timezone.utc)

        logger.info(
            "ai_recommendation_set",
            recommendation=recommendation,
            confidence=confidence,
            ttl_minutes=self.config.ai_recommendation_ttl_minutes,
        )

    def get_recommendation_info(self) -> Optional[dict]:
        """
        Get current AI recommendation info for dashboard/logging.

        Returns:
            Dict with recommendation, decay_percent, and ttl_minutes, or None if no active recommendation.
        """
        if not self._ai_recommendation or not self._ai_recommendation_time:
            return None

        elapsed = (datetime.now(timezone.utc) - self._ai_recommendation_time).total_seconds() / 60
        if elapsed > self.config.ai_recommendation_ttl_minutes:
            return None

        decay_pct = (1.0 - elapsed / self.config.ai_recommendation_ttl_minutes) * 100

        return {
            "recommendation": self._ai_recommendation,
            "decay_percent": decay_pct,
            "ttl_minutes": self.config.ai_recommendation_ttl_minutes,
        }

    def should_skip_review_after_veto(self, signal_action: str) -> tuple[bool, Optional[str]]:
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
        if not self.config.ai_review_rejection_cooldown:
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

    def record_veto(self, candles: pd.DataFrame, direction: str) -> None:
        """
        Record a veto rejection for cooldown tracking.

        Uses candle's market time instead of wall-clock time to avoid
        edge cases at candle boundaries.

        Args:
            candles: DataFrame containing candle data with timestamp column
            direction: Signal direction at veto ("buy" or "sell")
        """
        # Use candle timestamp (market time) instead of wall-clock time
        if not candles.empty:
            candle_timestamp = candles.iloc[-1]["timestamp"]
            # Normalize pd.Timestamp to datetime for consistent handling
            if isinstance(candle_timestamp, pd.Timestamp):
                candle_timestamp = candle_timestamp.to_pydatetime()
            elif not isinstance(candle_timestamp, datetime):
                logger.warning(
                    "candle_timestamp_type_mismatch",
                    type=type(candle_timestamp).__name__,
                    message="Expected datetime, falling back to current time"
                )
                candle_timestamp = datetime.now(timezone.utc)
        else:
            candle_timestamp = datetime.now(timezone.utc)

        self._last_veto_timestamp = candle_timestamp
        self._last_veto_direction = direction

    def record_sentiment_failure(self, context: str = "unknown") -> None:
        """
        Record sentiment fetch failure and alert if threshold is exceeded.

        Args:
            context: The context where failure occurred ("trading" or "dashboard")

        Thread-safe: All counter operations are atomic under lock.
        """
        threshold = self.config.sentiment_failure_alert_threshold

        # Atomically increment and check threshold under lock
        with self._sentiment_lock:
            self._sentiment_fetch_failures += 1
            failures_count = self._sentiment_fetch_failures

            # Only alert when we first cross the threshold
            if failures_count != threshold:
                if failures_count > threshold:
                    logger.debug(
                        "sentiment_fetch_failures_above_threshold",
                        consecutive_failures=failures_count,
                        threshold=threshold,
                        context=context,
                    )
                return

            last_success_timestamp = self._last_sentiment_fetch_success

        # Calculate time since last success outside lock
        try:
            if last_success_timestamp is None:
                time_since_success = "none (check API connectivity)"
            else:
                minutes_ago = (datetime.now(timezone.utc) - last_success_timestamp).total_seconds() / 60
                if minutes_ago < 0:
                    time_since_success = "unknown (clock skew detected)"
                elif minutes_ago < 1:
                    time_since_success = "just now (< 1 minute ago)"
                else:
                    time_since_success = f"{minutes_ago:.0f} minutes ago"
        except Exception:
            time_since_success = "unknown"

        logger.error(
            "sentiment_fetch_failure_threshold_exceeded",
            consecutive_failures=failures_count,
            threshold=threshold,
            last_success=time_since_success,
            impact="extreme_fear_override_disabled",
            context=context,
        )

        self.notifier.notify_error(
            f"Sentiment API failing: {failures_count} consecutive failures. "
            f"Last success: {time_since_success}. "
            f"Extreme fear override is disabled until sentiment API recovers.",
            "Sentiment Fetch Failure Alert"
        )

    def record_sentiment_success(self) -> None:
        """
        Record successful sentiment fetch and notify recovery if needed.

        This method:
        1. Checks if we were previously in an alert state
        2. Resets the failure counter
        3. Updates the last success timestamp
        4. Notifies recovery if we were previously failing
        """
        threshold = self.config.sentiment_failure_alert_threshold

        # Atomically check and reset state under lock
        with self._sentiment_lock:
            was_in_alert_state = self._sentiment_fetch_failures >= threshold
            self._sentiment_fetch_failures = 0
            self._last_sentiment_fetch_success = datetime.now(timezone.utc)

        # Notify recovery if we were previously failing
        if was_in_alert_state:
            self.notifier.notify_info(
                "Sentiment API has recovered. Extreme fear override is now active.",
                "Sentiment API Recovery"
            )

    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """
        Get start of the candle period containing the given timestamp.

        Args:
            timestamp: Any datetime within a candle period

        Returns:
            datetime representing the start of that candle period
        """
        seconds = GRANULARITY_SECONDS.get(self.config.candle_interval, 3600)
        ts = timestamp.timestamp()
        candle_start_ts = (ts // seconds) * seconds
        return datetime.fromtimestamp(candle_start_ts, tz=timezone.utc)
