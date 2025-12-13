"""
Circuit breaker for automatic trading halt on anomalous conditions.

Monitors for:
- Price crashes/spikes
- API failures
- Order failures
- Unusual volume

Multi-level status: GREEN, YELLOW, RED, BLACK
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class BreakerLevel(IntEnum):
    """Circuit breaker status levels (ordered by severity)."""

    GREEN = 1   # Normal operation
    YELLOW = 2  # Warning - reduced trading
    RED = 3     # Trading halted - auto-reset after cooldown
    BLACK = 4   # Trading halted - requires manual reset


@dataclass
class BreakerStatus:
    """Current circuit breaker status."""

    level: BreakerLevel
    reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    can_trade: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker thresholds."""

    # Price movement thresholds (percentage)
    price_drop_yellow: float = 5.0  # 5% drop triggers YELLOW
    price_drop_red: float = 10.0  # 10% drop triggers RED
    price_spike_yellow: float = 8.0  # 8% spike triggers YELLOW
    price_spike_red: float = 15.0  # 15% spike triggers RED

    # API failure thresholds (generous to handle transient issues)
    api_failures_yellow: int = 5  # 5 consecutive failures
    api_failures_red: int = 10  # 10 consecutive failures

    # Order failure thresholds
    order_failures_yellow: int = 2
    order_failures_black: int = 3  # Requires manual reset

    # Cooldown periods (seconds)
    yellow_cooldown: int = 300  # 5 minutes
    red_cooldown: int = 14400  # 4 hours

    # Time window for price change detection (seconds)
    price_window: int = 3600  # 1 hour
    price_window_24h: int = 86400  # 24 hours for sustained crash detection

    # Extended window thresholds (24-hour)
    price_drop_red_24h: float = 20.0  # 20% drop in 24h triggers RED

    # BLACK state recovery (None = manual only, hours value = auto-recovery)
    black_recovery_hours: Optional[int] = None


# Adaptive flash crash detection parameters by candle interval
# Format: (drop_threshold_percent, window_minutes)
# Shorter candles need faster detection with lower thresholds
RAPID_DROP_PARAMS = {
    "ONE_MINUTE": (1.0, 2),      # 1% in 2 min
    "FIVE_MINUTE": (2.0, 5),     # 2% in 5 min
    "FIFTEEN_MINUTE": (3.0, 10),  # 3% in 10 min
    "THIRTY_MINUTE": (4.0, 15),   # 4% in 15 min
    "ONE_HOUR": (5.0, 20),        # 5% in 20 min
    "TWO_HOUR": (6.0, 30),        # 6% in 30 min
    "SIX_HOUR": (8.0, 60),        # 8% in 1 hour
    "ONE_DAY": (10.0, 120),       # 10% in 2 hours
}
_DEFAULT_RAPID_DROP_PARAMS = (5.0, 20)  # Default: 5% in 20 minutes

# Valid candle intervals for validation
_VALID_CANDLE_INTERVALS = set(RAPID_DROP_PARAMS.keys())


def get_rapid_drop_params(candle_interval: Optional[str] = None) -> tuple[float, int]:
    """
    Get flash crash detection parameters appropriate for the candle interval.

    Shorter candles have smaller normal price movements, so flash crashes
    are detected with smaller thresholds over shorter windows. Longer candles
    expect larger normal movements, so thresholds are higher.

    Args:
        candle_interval: The candle interval string (e.g., "FIFTEEN_MINUTE")

    Returns:
        Tuple of (drop_threshold_percent, window_minutes)
    """
    if candle_interval is None:
        return _DEFAULT_RAPID_DROP_PARAMS
    if candle_interval not in _VALID_CANDLE_INTERVALS:
        logger.warning(
            "invalid_candle_interval",
            interval=candle_interval,
            using="default",
            valid_intervals=list(_VALID_CANDLE_INTERVALS)
        )
    return RAPID_DROP_PARAMS.get(candle_interval, _DEFAULT_RAPID_DROP_PARAMS)


class CircuitBreaker:
    """
    Multi-level circuit breaker for trading protection.

    Levels:
    - GREEN: Normal trading
    - YELLOW: Reduced trading (smaller positions, longer intervals)
    - RED: Trading halted, auto-reset after cooldown
    - BLACK: Trading halted, requires manual reset
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        on_trip: Optional[Callable[[BreakerLevel, str], None]] = None,
        candle_interval: Optional[str] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_trip: Callback when breaker is tripped
            candle_interval: Candle interval for adaptive flash crash detection
        """
        self.config = config or CircuitBreakerConfig()
        self._on_trip = on_trip
        self._candle_interval = candle_interval

        # Current state
        self._level = BreakerLevel.GREEN
        self._reason: Optional[str] = None
        self._triggered_at: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

        # Failure counters
        self._api_failures = 0
        self._order_failures = 0

        # Price history for change detection
        self._price_history: list[tuple[datetime, float]] = []
        self._price_history_24h: list[tuple[datetime, float]] = []
        self._price_history_rapid: list[tuple[datetime, float]] = []  # For adaptive flash crash

    def set_candle_interval(self, candle_interval: str) -> None:
        """
        Update the candle interval for adaptive flash crash detection.

        Args:
            candle_interval: The candle interval string (e.g., "FIFTEEN_MINUTE")
        """
        # Only clear history and log if interval actually changed
        if candle_interval != self._candle_interval:
            self._candle_interval = candle_interval
            # Clear rapid price history on interval change to avoid false triggers
            self._price_history_rapid = []
            logger.debug("circuit_breaker_interval_updated", candle_interval=candle_interval)

    @property
    def level(self) -> BreakerLevel:
        """Get current breaker level, accounting for cooldown expiry."""
        self._check_cooldown_expiry()
        return self._level

    @property
    def status(self) -> BreakerStatus:
        """Get full circuit breaker status."""
        self._check_cooldown_expiry()
        return BreakerStatus(
            level=self._level,
            reason=self._reason,
            triggered_at=self._triggered_at,
            cooldown_until=self._cooldown_until,
            can_trade=self._level in (BreakerLevel.GREEN, BreakerLevel.YELLOW),
        )

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        self._check_cooldown_expiry()
        return self._level in (BreakerLevel.GREEN, BreakerLevel.YELLOW)

    @property
    def position_multiplier(self) -> float:
        """Get position size multiplier based on current level."""
        if self._level == BreakerLevel.GREEN:
            return 1.0
        elif self._level == BreakerLevel.YELLOW:
            return 0.5  # Half position size during warning
        return 0.0  # No trading at RED or BLACK

    def _check_cooldown_expiry(self) -> None:
        """Check if cooldown has expired and reset if so."""
        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            if self._level == BreakerLevel.RED:
                self._reset_to_green("Cooldown expired")
            elif self._level == BreakerLevel.YELLOW:
                self._reset_to_green("Warning cooldown expired")

        # Check for BLACK state auto-recovery (if configured)
        if self._level == BreakerLevel.BLACK and self.config.black_recovery_hours:
            if self._triggered_at:
                hours_since = (datetime.now() - self._triggered_at).total_seconds() / 3600
                if hours_since >= self.config.black_recovery_hours:
                    logger.info(
                        "circuit_breaker_black_auto_recovery",
                        hours_since=f"{hours_since:.1f}",
                        recovery_hours=self.config.black_recovery_hours,
                    )
                    # Downgrade to RED (which will then auto-recover to GREEN after cooldown)
                    self._level = BreakerLevel.RED
                    self._reason = f"Auto-recovery from BLACK after {self.config.black_recovery_hours}h"
                    self._cooldown_until = datetime.now() + timedelta(
                        seconds=self.config.red_cooldown
                    )

    def _reset_to_green(self, reason: str) -> None:
        """Reset breaker to GREEN state."""
        old_level = self._level
        self._level = BreakerLevel.GREEN
        self._reason = None
        self._triggered_at = None
        self._cooldown_until = None
        self._api_failures = 0
        self._order_failures = 0

        logger.info(
            "circuit_breaker_reset",
            from_level=old_level.name,
            reason=reason,
        )

    def trip(self, level: BreakerLevel, reason: str) -> None:
        """
        Trip the circuit breaker to a specific level.

        Args:
            level: Breaker level to set
            reason: Reason for tripping
        """
        # Don't downgrade severity (IntEnum comparison works correctly)
        if self._level >= level and self._level != BreakerLevel.GREEN:
            logger.debug(
                "circuit_breaker_skip",
                current=self._level.name,
                requested=level.name,
            )
            return

        self._level = level
        self._reason = reason
        self._triggered_at = datetime.now()

        # Set cooldown based on level
        if level == BreakerLevel.YELLOW:
            self._cooldown_until = datetime.now() + timedelta(
                seconds=self.config.yellow_cooldown
            )
        elif level == BreakerLevel.RED:
            self._cooldown_until = datetime.now() + timedelta(
                seconds=self.config.red_cooldown
            )
        else:  # BLACK - no auto cooldown
            self._cooldown_until = None

        logger.warning(
            "circuit_breaker_tripped",
            level=level.name,
            reason=reason,
            cooldown_until=self._cooldown_until.isoformat() if self._cooldown_until else None,
        )

        # Call notification callback
        if self._on_trip:
            try:
                self._on_trip(level, reason)
            except Exception as e:
                logger.error("circuit_breaker_callback_failed", error=str(e))

    def record_price(self, price: float) -> None:
        """
        Record a price point for change detection.

        Args:
            price: Current price
        """
        now = datetime.now()
        self._price_history.append((now, price))
        self._price_history_24h.append((now, price))
        self._price_history_rapid.append((now, price))

        # Clean old entries (1-hour window)
        cutoff = now - timedelta(seconds=self.config.price_window)
        self._price_history = [
            (ts, p) for ts, p in self._price_history if ts > cutoff
        ]

        # Clean old entries (24-hour window)
        cutoff_24h = now - timedelta(seconds=self.config.price_window_24h)
        self._price_history_24h = [
            (ts, p) for ts, p in self._price_history_24h if ts > cutoff_24h
        ]

        # Clean old entries (rapid window - adaptive based on candle interval)
        _, rapid_window_minutes = get_rapid_drop_params(self._candle_interval)
        cutoff_rapid = now - timedelta(minutes=rapid_window_minutes)
        self._price_history_rapid = [
            (ts, p) for ts, p in self._price_history_rapid if ts > cutoff_rapid
        ]

    def check_price_movement(self, current_price: float) -> Optional[BreakerStatus]:
        """
        Check for anomalous price movements.

        Args:
            current_price: Current market price

        Returns:
            BreakerStatus if triggered, None otherwise
        """
        self.record_price(current_price)

        # Check 1-hour window (flash crash detection)
        if len(self._price_history) >= 2:
            oldest_price = self._price_history[0][1]
            change_percent = ((current_price - oldest_price) / oldest_price) * 100

            # Check for price drop
            if change_percent <= -self.config.price_drop_red:
                self.trip(BreakerLevel.RED, f"Price dropped {abs(change_percent):.1f}% in {self.config.price_window // 60} minutes")
            elif change_percent <= -self.config.price_drop_yellow:
                self.trip(BreakerLevel.YELLOW, f"Price dropped {abs(change_percent):.1f}% in {self.config.price_window // 60} minutes")

            # Check for price spike
            if change_percent >= self.config.price_spike_red:
                self.trip(BreakerLevel.RED, f"Price spiked {change_percent:.1f}% in {self.config.price_window // 60} minutes")
            elif change_percent >= self.config.price_spike_yellow:
                self.trip(BreakerLevel.YELLOW, f"Price spiked {change_percent:.1f}% in {self.config.price_window // 60} minutes")

        # Check rapid window (adaptive flash crash detection based on candle interval)
        if len(self._price_history_rapid) >= 2:
            drop_threshold, window_minutes = get_rapid_drop_params(self._candle_interval)
            oldest_rapid_price = self._price_history_rapid[0][1]
            rapid_change_percent = ((current_price - oldest_rapid_price) / oldest_rapid_price) * 100

            # Rapid drop triggers YELLOW (early warning for flash crash)
            if rapid_change_percent <= -drop_threshold:
                self.trip(
                    BreakerLevel.YELLOW,
                    f"Rapid price drop: {abs(rapid_change_percent):.1f}% in {window_minutes} minutes "
                    f"(threshold: {drop_threshold}% for {self._candle_interval or 'default'} interval)"
                )

        # Check 24-hour window (sustained crash detection)
        if len(self._price_history_24h) >= 2:
            oldest_price_24h = self._price_history_24h[0][1]
            change_percent_24h = ((current_price - oldest_price_24h) / oldest_price_24h) * 100

            if change_percent_24h <= -self.config.price_drop_red_24h:
                self.trip(BreakerLevel.RED, f"Price dropped {abs(change_percent_24h):.1f}% in 24 hours (sustained crash)")

        return self.status

    def record_api_success(self) -> None:
        """Record a successful API call, resetting failure counter."""
        self._api_failures = 0

    def record_api_failure(self) -> BreakerStatus:
        """
        Record an API failure.

        Returns:
            Current breaker status
        """
        self._api_failures += 1

        if self._api_failures >= self.config.api_failures_red:
            self.trip(BreakerLevel.RED, f"{self._api_failures} consecutive API failures")
        elif self._api_failures >= self.config.api_failures_yellow:
            self.trip(BreakerLevel.YELLOW, f"{self._api_failures} consecutive API failures")

        return self.status

    def record_order_success(self) -> None:
        """Record a successful order, resetting failure counter."""
        self._order_failures = 0

    def record_order_failure(self) -> BreakerStatus:
        """
        Record an order failure.

        Returns:
            Current breaker status
        """
        self._order_failures += 1

        if self._order_failures >= self.config.order_failures_black:
            self.trip(BreakerLevel.BLACK, f"{self._order_failures} consecutive order failures - manual reset required")
        elif self._order_failures >= self.config.order_failures_yellow:
            self.trip(BreakerLevel.YELLOW, f"{self._order_failures} consecutive order failures")

        return self.status

    def manual_reset(self, confirm_code: str = "RESET_CONFIRMED") -> bool:
        """
        Manually reset the circuit breaker (for BLACK level).

        Args:
            confirm_code: Confirmation code to prevent accidental resets

        Returns:
            True if reset successful
        """
        if confirm_code != "RESET_CONFIRMED":
            logger.warning("circuit_breaker_reset_failed", reason="Invalid confirmation code")
            return False

        self._reset_to_green("Manual reset")
        return True

    def check_and_raise(self) -> None:
        """Check circuit breaker and raise exception if trading not allowed."""
        if not self.can_trade:
            raise CircuitBreakerOpenError(self._level, self._reason or "Circuit breaker open")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker prevents trading."""

    def __init__(self, level: BreakerLevel, reason: str):
        self.level = level
        self.reason = reason
        super().__init__(f"Circuit breaker at {level.name}: {reason}")
