"""
Loss limiter to prevent excessive losses.

Tracks:
- Daily realized P&L
- Hourly realized P&L
- Current drawdown

Throttles or halts trading when limits are approached or exceeded.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LossLimitConfig:
    """Configuration for loss limits."""

    # Maximum loss percentages
    max_daily_loss_percent: float = 10.0  # Stop trading for the day
    max_hourly_loss_percent: float = 3.0  # Pause for 1 hour

    # Throttling thresholds
    throttle_at_percent: float = 50.0  # Start reducing position at 50% of limit

    # Cooldown after hitting limit
    hourly_cooldown_seconds: int = 3600  # 1 hour
    daily_reset_hour_utc: int = 0  # Reset at midnight UTC


@dataclass
class LossLimitStatus:
    """Current loss limit status."""

    can_trade: bool
    position_multiplier: float  # 0.0 to 1.0
    daily_loss_percent: float
    hourly_loss_percent: float
    daily_limit_hit: bool
    hourly_limit_hit: bool
    cooldown_until: Optional[datetime] = None
    reason: Optional[str] = None


@dataclass
class TradeRecord:
    """Record of a single trade for P&L tracking."""

    timestamp: datetime
    realized_pnl: Decimal
    side: str  # "buy" or "sell"
    size: Decimal
    price: Decimal


class LossLimiter:
    """
    Tracks and limits trading losses.

    Features:
    - Daily loss limit with automatic trading halt
    - Hourly loss limit with temporary pause
    - Progressive throttling as limits approach
    - Automatic reset at configured times
    """

    def __init__(
        self,
        config: Optional[LossLimitConfig] = None,
        starting_balance: Decimal = Decimal("0"),
        on_limit_hit: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize loss limiter.

        Args:
            config: Loss limit configuration
            starting_balance: Starting portfolio balance for percentage calculations
            on_limit_hit: Callback when a limit is hit
        """
        self.config = config or LossLimitConfig()
        self._on_limit_hit = on_limit_hit

        # Balance tracking
        self._starting_balance = starting_balance
        self._daily_starting_balance = starting_balance

        # Trade history
        self._trades: list[TradeRecord] = []

        # State
        self._daily_limit_hit = False
        self._hourly_limit_hit = False
        self._cooldown_until: Optional[datetime] = None
        self._last_daily_reset: Optional[datetime] = None

    def set_starting_balance(self, balance: Decimal) -> None:
        """
        Set the starting balance for percentage calculations.

        Args:
            balance: Current portfolio balance
        """
        self._starting_balance = balance
        if self._daily_starting_balance == Decimal("0"):
            self._daily_starting_balance = balance
        logger.info(
            "loss_limiter_balance_set",
            starting_balance=str(balance),
            daily_starting_balance=str(self._daily_starting_balance),
        )

    def update_settings(
        self,
        max_daily_loss_percent: Optional[float] = None,
        max_hourly_loss_percent: Optional[float] = None,
    ) -> None:
        """
        Update loss limiter settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        Note: This does NOT reset current loss tracking.
        """
        if max_daily_loss_percent is not None:
            self.config.max_daily_loss_percent = max_daily_loss_percent
        if max_hourly_loss_percent is not None:
            self.config.max_hourly_loss_percent = max_hourly_loss_percent

        logger.info("loss_limiter_settings_updated")

    def record_trade(
        self,
        realized_pnl: Decimal,
        side: str,
        size: Decimal,
        price: Decimal,
    ) -> LossLimitStatus:
        """
        Record a trade and check limits.

        Args:
            realized_pnl: Realized profit/loss from the trade
            side: Trade side ("buy" or "sell")
            size: Trade size
            price: Trade price

        Returns:
            Current loss limit status
        """
        trade = TradeRecord(
            timestamp=datetime.now(),
            realized_pnl=realized_pnl,
            side=side,
            size=size,
            price=price,
        )
        self._trades.append(trade)

        # Clean old trades
        self._cleanup_old_trades()

        # Check limits
        return self._check_limits()

    def _cleanup_old_trades(self) -> None:
        """Remove trades older than 24 hours."""
        cutoff = datetime.now() - timedelta(hours=24)
        self._trades = [t for t in self._trades if t.timestamp > cutoff]

    def _get_daily_pnl(self) -> Decimal:
        """Get total realized P&L for today."""
        today = datetime.now().date()
        daily_trades = [
            t for t in self._trades if t.timestamp.date() == today
        ]
        return sum((t.realized_pnl for t in daily_trades), Decimal("0"))

    def _get_hourly_pnl(self) -> Decimal:
        """Get total realized P&L for the last hour."""
        hour_ago = datetime.now() - timedelta(hours=1)
        hourly_trades = [
            t for t in self._trades if t.timestamp > hour_ago
        ]
        return sum((t.realized_pnl for t in hourly_trades), Decimal("0"))

    def _calculate_loss_percent(self, pnl: Decimal, balance: Decimal) -> float:
        """Calculate loss as percentage of balance."""
        if balance == Decimal("0"):
            return 0.0
        # Negative P&L = loss, so we negate for loss percentage
        return float(-pnl / balance * 100)

    def _check_limits(self) -> LossLimitStatus:
        """Check all loss limits and update state."""
        # Check for daily reset
        self._check_daily_reset()

        # Check cooldown expiry
        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            self._hourly_limit_hit = False
            self._cooldown_until = None
            logger.info("loss_limiter_hourly_cooldown_expired")

        # Calculate current losses
        daily_pnl = self._get_daily_pnl()
        hourly_pnl = self._get_hourly_pnl()

        daily_loss_percent = self._calculate_loss_percent(
            daily_pnl, self._daily_starting_balance
        )
        hourly_loss_percent = self._calculate_loss_percent(
            hourly_pnl, self._daily_starting_balance
        )

        # Check daily limit
        if daily_loss_percent >= self.config.max_daily_loss_percent:
            if not self._daily_limit_hit:
                self._daily_limit_hit = True
                logger.warning(
                    "loss_limiter_daily_limit_hit",
                    daily_loss_percent=daily_loss_percent,
                    max_allowed=self.config.max_daily_loss_percent,
                )
                if self._on_limit_hit:
                    self._on_limit_hit("daily", daily_loss_percent)

        # Check hourly limit
        if hourly_loss_percent >= self.config.max_hourly_loss_percent:
            if not self._hourly_limit_hit:
                self._hourly_limit_hit = True
                self._cooldown_until = datetime.now() + timedelta(
                    seconds=self.config.hourly_cooldown_seconds
                )
                logger.warning(
                    "loss_limiter_hourly_limit_hit",
                    hourly_loss_percent=hourly_loss_percent,
                    max_allowed=self.config.max_hourly_loss_percent,
                    cooldown_until=self._cooldown_until.isoformat(),
                )
                if self._on_limit_hit:
                    self._on_limit_hit("hourly", hourly_loss_percent)

        # Calculate position multiplier
        position_multiplier = self._calculate_position_multiplier(
            daily_loss_percent, hourly_loss_percent
        )

        # Determine if can trade
        can_trade = not self._daily_limit_hit and not self._hourly_limit_hit

        reason = None
        if self._daily_limit_hit:
            reason = f"Daily loss limit exceeded ({daily_loss_percent:.1f}%)"
        elif self._hourly_limit_hit:
            reason = f"Hourly loss limit exceeded ({hourly_loss_percent:.1f}%)"

        return LossLimitStatus(
            can_trade=can_trade,
            position_multiplier=position_multiplier,
            daily_loss_percent=daily_loss_percent,
            hourly_loss_percent=hourly_loss_percent,
            daily_limit_hit=self._daily_limit_hit,
            hourly_limit_hit=self._hourly_limit_hit,
            cooldown_until=self._cooldown_until,
            reason=reason,
        )

    def _calculate_position_multiplier(
        self, daily_loss_percent: float, hourly_loss_percent: float
    ) -> float:
        """
        Calculate position size multiplier based on current losses.

        Progressively reduces position size as losses approach limits.
        """
        if self._daily_limit_hit or self._hourly_limit_hit:
            return 0.0

        # Use the higher of the two loss percentages
        max_loss_percent = max(daily_loss_percent, hourly_loss_percent)

        # Determine which limit we're approaching
        if daily_loss_percent > hourly_loss_percent:
            limit = self.config.max_daily_loss_percent
        else:
            limit = self.config.max_hourly_loss_percent

        # Calculate throttle threshold
        throttle_threshold = limit * (self.config.throttle_at_percent / 100)

        if max_loss_percent < throttle_threshold:
            return 1.0

        # Linear reduction from 1.0 to 0.3 as we approach limit
        progress = (max_loss_percent - throttle_threshold) / (limit - throttle_threshold)
        multiplier = max(0.3, 1.0 - (progress * 0.7))

        logger.debug(
            "loss_limiter_throttling",
            loss_percent=max_loss_percent,
            multiplier=multiplier,
        )

        return multiplier

    def _check_daily_reset(self) -> None:
        """Check if we should reset daily counters."""
        now = datetime.now()
        reset_time = now.replace(
            hour=self.config.daily_reset_hour_utc,
            minute=0,
            second=0,
            microsecond=0,
        )

        # If reset time is in the future, it means we should use yesterday's reset time
        if reset_time > now:
            reset_time -= timedelta(days=1)

        if self._last_daily_reset is None or self._last_daily_reset < reset_time:
            self._daily_limit_hit = False
            self._daily_starting_balance = self._starting_balance
            self._last_daily_reset = reset_time
            logger.info("loss_limiter_daily_reset")

    def get_status(self) -> LossLimitStatus:
        """Get current loss limit status."""
        return self._check_limits()

    def check_and_raise(self) -> None:
        """Check limits and raise exception if trading not allowed."""
        status = self.get_status()
        if not status.can_trade:
            raise LossLimitExceededError(status.reason or "Loss limit exceeded")


class LossLimitExceededError(Exception):
    """Exception raised when loss limit prevents trading."""

    pass
