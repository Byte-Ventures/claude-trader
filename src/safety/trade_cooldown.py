"""
Trade cooldown to prevent rapid consecutive trades.

Prevents:
- Rapid averaging into falling knives (buy after buy)
- Excessive fees from multiple small trades at similar prices

Features:
- Time-based cooldown: Minimum minutes between same-direction trades
- Price-based cooldown: Only trade if price moved X% from last trade
- Per-direction tracking: Independent cooldowns for buys vs sells

Logic: Either condition can unlock the cooldown (OR, not AND).
- Time passed: After X minutes, cooldown expires regardless of price
- Price moved: If price moved Y%, can trade regardless of time
This prevents old trades from blocking new ones indefinitely.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from src.state.database import Database

logger = structlog.get_logger(__name__)


@dataclass
class TradeCooldownConfig:
    """Configuration for trade cooldowns."""

    # Time-based cooldown (minutes, 0 = disabled)
    buy_cooldown_minutes: int = 15
    sell_cooldown_minutes: int = 0  # Disabled by default (safety: allow quick exits)

    # Price-based cooldown (percent change required, 0 = disabled)
    buy_price_change_percent: float = 1.0  # Price must drop 1% to buy again
    sell_price_change_percent: float = 0.0  # Disabled for sells


@dataclass
class CooldownStatus:
    """Current cooldown status."""

    can_buy: bool
    can_sell: bool
    buy_blocked_reason: Optional[str] = None
    sell_blocked_reason: Optional[str] = None
    last_buy_price: Optional[Decimal] = None
    last_sell_price: Optional[Decimal] = None
    buy_cooldown_remaining_seconds: int = 0
    sell_cooldown_remaining_seconds: int = 0


class TradeCooldown:
    """
    Enforces cooldowns between same-direction trades.

    Features:
    - Time-based: Must wait X minutes between same-direction trades
    - Price-based: Price must move Y% from last trade to trade again
    - Per-direction: Buy and sell cooldowns are independent
    - DB-backed: Survives restarts by querying last trade from database
    """

    def __init__(
        self,
        config: Optional[TradeCooldownConfig] = None,
        db: Optional["Database"] = None,
        is_paper: bool = False,
        symbol: str = "BTC-USD",
    ):
        """
        Initialize trade cooldown.

        Args:
            config: Cooldown configuration
            db: Database for querying last trades
            is_paper: Whether this is paper trading
            symbol: Trading pair symbol
        """
        self.config = config or TradeCooldownConfig()
        self.db = db
        self.is_paper = is_paper
        self.symbol = symbol

        # In-memory cache of last trades (populated on first check)
        self._last_buy_time: Optional[datetime] = None
        self._last_buy_price: Optional[Decimal] = None
        self._last_sell_time: Optional[datetime] = None
        self._last_sell_price: Optional[Decimal] = None
        self._cache_initialized = False

        logger.info(
            "trade_cooldown_initialized",
            buy_cooldown_minutes=self.config.buy_cooldown_minutes,
            sell_cooldown_minutes=self.config.sell_cooldown_minutes,
            buy_price_change_percent=self.config.buy_price_change_percent,
            sell_price_change_percent=self.config.sell_price_change_percent,
        )

    def _ensure_cache_initialized(self) -> None:
        """Load last trades from database if not already cached."""
        if self._cache_initialized or not self.db:
            return

        # Load last buy
        last_buy = self.db.get_last_trade_by_side(
            side="buy", symbol=self.symbol, is_paper=self.is_paper
        )
        if last_buy:
            self._last_buy_time = last_buy.executed_at
            self._last_buy_price = Decimal(last_buy.price)

        # Load last sell
        last_sell = self.db.get_last_trade_by_side(
            side="sell", symbol=self.symbol, is_paper=self.is_paper
        )
        if last_sell:
            self._last_sell_time = last_sell.executed_at
            self._last_sell_price = Decimal(last_sell.price)

        self._cache_initialized = True

        logger.debug(
            "trade_cooldown_cache_loaded",
            last_buy_time=self._last_buy_time.isoformat() if self._last_buy_time else None,
            last_buy_price=str(self._last_buy_price) if self._last_buy_price else None,
            last_sell_time=self._last_sell_time.isoformat() if self._last_sell_time else None,
            last_sell_price=str(self._last_sell_price) if self._last_sell_price else None,
        )

    def can_execute(self, side: str, current_price: Decimal) -> tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on cooldown rules.

        Either time OR price condition can unlock the cooldown:
        - If enough time has passed since last trade: allowed
        - If price has moved enough from last trade: allowed
        - Otherwise: blocked

        Args:
            side: Trade side ("buy" or "sell")
            current_price: Current market price

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        # Validate input price
        if current_price is None or current_price <= 0:
            logger.error("invalid_price_for_cooldown", price=str(current_price))
            return False, "invalid price"

        self._ensure_cache_initialized()

        if side == "buy":
            return self._check_buy_cooldown(current_price)
        elif side == "sell":
            return self._check_sell_cooldown(current_price)
        else:
            return True, None

    def _check_buy_cooldown(self, current_price: Decimal) -> tuple[bool, Optional[str]]:
        """Check buy cooldown conditions.

        Logic: Either time OR price condition can unlock the cooldown.
        - If enough time has passed since last buy: allowed
        - If price has dropped enough from last buy: allowed
        - Otherwise: blocked
        """
        # No previous buy = no cooldown
        if self._last_buy_time is None:
            return True, None

        now = datetime.now(timezone.utc)

        # Ensure last_buy_time is timezone-aware
        last_buy_time = self._last_buy_time
        if last_buy_time.tzinfo is None:
            last_buy_time = last_buy_time.replace(tzinfo=timezone.utc)

        elapsed_seconds = (now - last_buy_time).total_seconds()
        elapsed_minutes = elapsed_seconds / 60

        # Check time cooldown
        time_ok = True
        time_reason = None
        if self.config.buy_cooldown_minutes > 0:
            if elapsed_minutes < self.config.buy_cooldown_minutes:
                remaining = self.config.buy_cooldown_minutes - elapsed_minutes
                time_ok = False
                time_reason = f"wait {remaining:.1f}min"

        # Check price cooldown
        price_ok = True
        price_reason = None
        if self.config.buy_price_change_percent > 0 and self._last_buy_price:
            price_change = ((current_price - self._last_buy_price) / self._last_buy_price) * 100
            # For buys, price must DROP (negative change)
            if price_change > -self.config.buy_price_change_percent:
                price_ok = False
                price_reason = f"price drop {-price_change:.2f}% < {self.config.buy_price_change_percent}%"

        # Either condition passing unlocks the cooldown
        if time_ok or price_ok:
            return True, None

        # Both failed - return combined reason
        return False, f"{time_reason}, {price_reason}"

    def _check_sell_cooldown(self, current_price: Decimal) -> tuple[bool, Optional[str]]:
        """Check sell cooldown conditions.

        Logic: Either time OR price condition can unlock the cooldown.
        - If enough time has passed since last sell: allowed
        - If price has risen enough from last sell: allowed
        - Otherwise: blocked
        """
        # No previous sell = no cooldown
        if self._last_sell_time is None:
            return True, None

        now = datetime.now(timezone.utc)

        # Ensure last_sell_time is timezone-aware
        last_sell_time = self._last_sell_time
        if last_sell_time.tzinfo is None:
            last_sell_time = last_sell_time.replace(tzinfo=timezone.utc)

        elapsed_seconds = (now - last_sell_time).total_seconds()
        elapsed_minutes = elapsed_seconds / 60

        # Check time cooldown
        time_ok = True
        time_reason = None
        if self.config.sell_cooldown_minutes > 0:
            if elapsed_minutes < self.config.sell_cooldown_minutes:
                remaining = self.config.sell_cooldown_minutes - elapsed_minutes
                time_ok = False
                time_reason = f"wait {remaining:.1f}min"

        # Check price cooldown
        price_ok = True
        price_reason = None
        if self.config.sell_price_change_percent > 0 and self._last_sell_price:
            price_change = ((current_price - self._last_sell_price) / self._last_sell_price) * 100
            # For sells, price must RISE (positive change)
            if price_change < self.config.sell_price_change_percent:
                price_ok = False
                price_reason = f"price rise {price_change:.2f}% < {self.config.sell_price_change_percent}%"

        # Either condition passing unlocks the cooldown
        if time_ok or price_ok:
            return True, None

        # Both failed - return combined reason
        return False, f"{time_reason}, {price_reason}"

    def record_trade(self, side: str, price: Decimal) -> None:
        """
        Record a trade execution to update cooldown state.

        Args:
            side: Trade side ("buy" or "sell")
            price: Execution price
        """
        now = datetime.now(timezone.utc)

        if side == "buy":
            self._last_buy_time = now
            self._last_buy_price = price
            logger.debug(
                "trade_cooldown_buy_recorded",
                price=str(price),
            )
        elif side == "sell":
            self._last_sell_time = now
            self._last_sell_price = price
            logger.debug(
                "trade_cooldown_sell_recorded",
                price=str(price),
            )

    def get_status(self) -> CooldownStatus:
        """Get current cooldown status."""
        self._ensure_cache_initialized()

        now = datetime.now(timezone.utc)

        # Calculate buy cooldown remaining
        buy_remaining = 0
        if self._last_buy_time and self.config.buy_cooldown_minutes > 0:
            last_buy_time = self._last_buy_time
            if last_buy_time.tzinfo is None:
                last_buy_time = last_buy_time.replace(tzinfo=timezone.utc)
            elapsed = (now - last_buy_time).total_seconds()
            cooldown_seconds = self.config.buy_cooldown_minutes * 60
            if elapsed < cooldown_seconds:
                buy_remaining = int(cooldown_seconds - elapsed)

        # Calculate sell cooldown remaining
        sell_remaining = 0
        if self._last_sell_time and self.config.sell_cooldown_minutes > 0:
            last_sell_time = self._last_sell_time
            if last_sell_time.tzinfo is None:
                last_sell_time = last_sell_time.replace(tzinfo=timezone.utc)
            elapsed = (now - last_sell_time).total_seconds()
            cooldown_seconds = self.config.sell_cooldown_minutes * 60
            if elapsed < cooldown_seconds:
                sell_remaining = int(cooldown_seconds - elapsed)

        # For full status, we'd need current price to check price condition
        # This is simplified - full check happens in can_execute()
        return CooldownStatus(
            can_buy=buy_remaining == 0,  # Simplified, doesn't check price
            can_sell=sell_remaining == 0,  # Simplified, doesn't check price
            last_buy_price=self._last_buy_price,
            last_sell_price=self._last_sell_price,
            buy_cooldown_remaining_seconds=buy_remaining,
            sell_cooldown_remaining_seconds=sell_remaining,
        )

    def update_settings(
        self,
        buy_cooldown_minutes: Optional[int] = None,
        sell_cooldown_minutes: Optional[int] = None,
        buy_price_change_percent: Optional[float] = None,
        sell_price_change_percent: Optional[float] = None,
    ) -> None:
        """
        Update cooldown settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        """
        if buy_cooldown_minutes is not None:
            self.config.buy_cooldown_minutes = buy_cooldown_minutes
        if sell_cooldown_minutes is not None:
            self.config.sell_cooldown_minutes = sell_cooldown_minutes
        if buy_price_change_percent is not None:
            self.config.buy_price_change_percent = buy_price_change_percent
        if sell_price_change_percent is not None:
            self.config.sell_price_change_percent = sell_price_change_percent

        logger.info(
            "trade_cooldown_settings_updated",
            buy_cooldown_minutes=self.config.buy_cooldown_minutes,
            sell_cooldown_minutes=self.config.sell_cooldown_minutes,
            buy_price_change_percent=self.config.buy_price_change_percent,
            sell_price_change_percent=self.config.sell_price_change_percent,
        )

    def invalidate_cache(self) -> None:
        """Invalidate cached trade data (e.g., after config reload)."""
        self._cache_initialized = False
        self._last_buy_time = None
        self._last_buy_price = None
        self._last_sell_time = None
        self._last_sell_price = None
        logger.debug("trade_cooldown_cache_invalidated")
