"""
Pre-trade order validator.

Validates all orders before execution to ensure they pass safety checks.
All checks must pass for an order to be executed.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from src.safety.circuit_breaker import CircuitBreaker, BreakerLevel
from src.safety.kill_switch import KillSwitch
from src.safety.loss_limiter import LossLimiter

logger = structlog.get_logger(__name__)


@dataclass
class OrderRequest:
    """Order request to be validated."""

    side: str  # "buy" or "sell"
    size: Decimal  # Amount of BTC
    price: Optional[Decimal] = None  # Limit price (None for market orders)
    order_type: str = "market"  # "market" or "limit"


@dataclass
class ValidationResult:
    """Result of order validation."""

    valid: bool
    reason: Optional[str] = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ValidatorConfig:
    """Configuration for order validator."""

    min_trade_quote: float = 100.0  # Minimum trade size in quote currency
    # Hard safety limit for position size (catches edge cases).
    # The position sizer has a lower "soft limit" (40%) for normal sizing.
    # Two-tier design: sizer targets 40%, validator enforces 80% hard stop.
    max_position_percent: float = 80.0
    price_sanity_percent: float = 5.0  # Max deviation from market price
    # Estimated round-trip fee for profit margin check (buy + sell fees)
    # Coinbase Advanced: ~0.6% (0.3% each way), Kraken: ~0.5%
    estimated_fee_percent: float = 0.006


class OrderValidator:
    """
    Comprehensive pre-trade validation.

    All checks must pass for an order to be approved:
    1. Kill switch not active
    2. Circuit breaker allows trading
    3. Loss limits not exceeded
    4. Sufficient balance
    5. Within position limits
    6. Price sanity check
    7. Minimum size check
    8. Profit margin check (stop > 2x fees for positive EV)
    """

    def __init__(
        self,
        config: Optional[ValidatorConfig] = None,
        kill_switch: Optional[KillSwitch] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        loss_limiter: Optional[LossLimiter] = None,
    ):
        """
        Initialize order validator.

        Args:
            config: Validator configuration
            kill_switch: Kill switch instance
            circuit_breaker: Circuit breaker instance
            loss_limiter: Loss limiter instance
        """
        self.config = config or ValidatorConfig()
        self.kill_switch = kill_switch
        self.circuit_breaker = circuit_breaker
        self.loss_limiter = loss_limiter

        # Current state (updated externally)
        self._base_balance = Decimal("0")
        self._quote_balance = Decimal("0")
        self._current_price = Decimal("0")

    def update_settings(
        self,
        max_position_percent: Optional[float] = None,
    ) -> None:
        """
        Update validator settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        """
        if max_position_percent is not None:
            self.config.max_position_percent = max_position_percent

        logger.info("order_validator_settings_updated")

    def update_balances(
        self,
        base_balance: Decimal,
        quote_balance: Decimal,
        current_price: Decimal,
    ) -> None:
        """
        Update current balances for validation.

        Args:
            base_balance: Current base currency balance (e.g., BTC)
            quote_balance: Current quote currency balance (e.g., USD/EUR)
            current_price: Current price in quote currency
        """
        self._base_balance = base_balance
        self._quote_balance = quote_balance
        self._current_price = current_price

    def validate(
        self,
        order: OrderRequest,
        stop_distance_percent: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate an order against all safety checks.

        Args:
            order: Order to validate
            stop_distance_percent: Stop-loss distance as percentage of price (e.g., 0.02 = 2%).
                                   If provided, validates that trade has positive expected value.

        Returns:
            ValidationResult with valid=True if all checks pass
        """
        warnings = []

        # Check 1: Kill switch
        result = self._check_kill_switch()
        if not result.valid:
            return result

        # Check 2: Circuit breaker
        result = self._check_circuit_breaker()
        if not result.valid:
            return result
        if result.warnings:
            warnings.extend(result.warnings)

        # Check 3: Loss limits
        result = self._check_loss_limits()
        if not result.valid:
            return result
        if result.warnings:
            warnings.extend(result.warnings)

        # Check 4: Balance check
        result = self._check_balance(order)
        if not result.valid:
            return result

        # Check 5: Position limits
        result = self._check_position_limits(order)
        if not result.valid:
            return result
        if result.warnings:
            warnings.extend(result.warnings)

        # Check 6: Price sanity
        result = self._check_price_sanity(order)
        if not result.valid:
            return result

        # Check 7: Minimum size
        result = self._check_minimum_size(order)
        if not result.valid:
            return result

        # Check 8: Profit margin (only for buys with stop distance provided)
        if order.side == "buy" and stop_distance_percent is not None:
            result = self._check_profit_margin(stop_distance_percent)
            if not result.valid:
                return result
            if result.warnings:
                warnings.extend(result.warnings)

        logger.info(
            "order_validated",
            side=order.side,
            size=str(order.size),
            order_type=order.order_type,
            warnings=warnings if warnings else None,
        )

        return ValidationResult(valid=True, warnings=warnings)

    def _check_kill_switch(self) -> ValidationResult:
        """Check if kill switch is active."""
        if self.kill_switch and self.kill_switch.is_active:
            return ValidationResult(
                valid=False,
                reason=f"Kill switch active: {self.kill_switch.reason}",
            )
        return ValidationResult(valid=True)

    def _check_circuit_breaker(self) -> ValidationResult:
        """Check circuit breaker status."""
        if not self.circuit_breaker:
            return ValidationResult(valid=True)

        status = self.circuit_breaker.status

        if not status.can_trade:
            return ValidationResult(
                valid=False,
                reason=f"Circuit breaker at {status.level.name}: {status.reason}",
            )

        warnings = []
        if status.level == BreakerLevel.YELLOW:
            warnings.append(
                f"Circuit breaker warning: {status.reason}. Position reduced to {self.circuit_breaker.position_multiplier * 100:.0f}%"
            )

        return ValidationResult(valid=True, warnings=warnings)

    def _check_loss_limits(self) -> ValidationResult:
        """Check loss limit status."""
        if not self.loss_limiter:
            return ValidationResult(valid=True)

        status = self.loss_limiter.get_status()

        if not status.can_trade:
            return ValidationResult(
                valid=False,
                reason=status.reason or "Loss limit exceeded",
            )

        warnings = []
        if status.position_multiplier < 1.0:
            warnings.append(
                f"Loss throttling active: Position reduced to {status.position_multiplier * 100:.0f}%"
            )

        return ValidationResult(valid=True, warnings=warnings)

    def _check_balance(self, order: OrderRequest) -> ValidationResult:
        """Check if sufficient balance for order."""
        if order.side == "buy":
            # For buy orders, check quote currency balance
            order_value = order.size * (order.price or self._current_price)
            if order_value > self._quote_balance:
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient quote balance. Need {order_value:.2f}, have {self._quote_balance:.2f}",
                )
        else:  # sell
            # For sell orders, check base currency balance
            if order.size > self._base_balance:
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient base balance. Need {order.size:.8f}, have {self._base_balance:.8f}",
                )

        return ValidationResult(valid=True)

    def _check_position_limits(self, order: OrderRequest) -> ValidationResult:
        """Check if order would exceed position limits."""
        # Calculate total portfolio value
        base_value = self._base_balance * self._current_price
        total_value = base_value + self._quote_balance

        if total_value == Decimal("0"):
            return ValidationResult(valid=True)

        if order.side == "buy":
            # Calculate new position after buy
            order_value = order.size * (order.price or self._current_price)
            new_base_value = base_value + order_value
            new_position_percent = float(new_base_value / total_value * 100)

            if new_position_percent > self.config.max_position_percent:
                return ValidationResult(
                    valid=False,
                    reason=f"Order would exceed position limit. New position: {new_position_percent:.1f}%, limit: {self.config.max_position_percent:.1f}%",
                )

            warnings = []
            if new_position_percent >= self.config.max_position_percent * 0.9:
                warnings.append(
                    f"Position nearing limit: {new_position_percent:.1f}%"
                )
            return ValidationResult(valid=True, warnings=warnings)

        return ValidationResult(valid=True)

    def _check_price_sanity(self, order: OrderRequest) -> ValidationResult:
        """Check if limit price is within reasonable range of market price."""
        if order.order_type != "limit" or order.price is None:
            return ValidationResult(valid=True)

        if self._current_price == Decimal("0"):
            return ValidationResult(valid=True)

        deviation_percent = abs(
            float((order.price - self._current_price) / self._current_price * 100)
        )

        if deviation_percent >= self.config.price_sanity_percent:
            return ValidationResult(
                valid=False,
                reason=f"Limit price {order.price:.2f} deviates {deviation_percent:.1f}% from market price {self._current_price:.2f}",
            )

        return ValidationResult(valid=True)

    def _check_minimum_size(self, order: OrderRequest) -> ValidationResult:
        """Check if order meets minimum size requirements."""
        order_value_quote = float(order.size * (order.price or self._current_price))

        if order_value_quote < self.config.min_trade_quote:
            return ValidationResult(
                valid=False,
                reason=f"Order value {order_value_quote:.2f} below minimum {self.config.min_trade_quote:.2f}",
            )

        return ValidationResult(valid=True)

    def _check_profit_margin(self, stop_distance_percent: float) -> ValidationResult:
        """
        Check if trade has positive expected value after fees.

        A trade needs the stop distance to be at least 2x the round-trip fees,
        otherwise even a 50% win rate results in net loss due to fees.

        Args:
            stop_distance_percent: Stop-loss distance as percentage (e.g., 0.02 = 2%)

        Returns:
            ValidationResult with valid=True if profit margin is sufficient
        """
        # Minimum required margin is 2x round-trip fees for break-even at 50% win rate
        min_margin = self.config.estimated_fee_percent * 2

        if stop_distance_percent < min_margin:
            return ValidationResult(
                valid=False,
                reason=f"Stop too tight for fees. Stop: {stop_distance_percent:.2%}, min required: {min_margin:.2%}",
            )

        # Warn if margin is tight (between 2x and 3x fees)
        warnings = []
        if stop_distance_percent < min_margin * 1.5:
            warnings.append(
                f"Tight profit margin: {stop_distance_percent:.2%} stop vs {self.config.estimated_fee_percent:.2%} fees"
            )

        return ValidationResult(valid=True, warnings=warnings)

    def get_position_multiplier(self) -> float:
        """
        Get combined position multiplier from all safety systems.

        Returns:
            Multiplier between 0.0 and 1.0
        """
        multiplier = 1.0

        if self.circuit_breaker:
            multiplier *= self.circuit_breaker.position_multiplier

        if self.loss_limiter:
            status = self.loss_limiter.get_status()
            multiplier *= status.position_multiplier

        return multiplier
