"""
Pre-trade order validator.

Validates all orders before execution to ensure they pass safety checks.
All checks must pass for an order to be executed.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from src.safety.circuit_breaker import CircuitBreaker
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

    min_trade_usd: float = 10.0  # Minimum trade size in USD
    max_position_percent: float = 80.0  # Maximum position as % of portfolio
    price_sanity_percent: float = 5.0  # Max deviation from market price


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
        self._btc_balance = Decimal("0")
        self._usd_balance = Decimal("0")
        self._current_price = Decimal("0")

    def update_balances(
        self,
        btc_balance: Decimal,
        usd_balance: Decimal,
        current_price: Decimal,
    ) -> None:
        """
        Update current balances for validation.

        Args:
            btc_balance: Current BTC balance
            usd_balance: Current USD balance
            current_price: Current BTC price
        """
        self._btc_balance = btc_balance
        self._usd_balance = usd_balance
        self._current_price = current_price

    def validate(self, order: OrderRequest) -> ValidationResult:
        """
        Validate an order against all safety checks.

        Args:
            order: Order to validate

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
                reason=f"Circuit breaker at {status.level.value}: {status.reason}",
            )

        warnings = []
        if status.level.value == "yellow":
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
            # For buy orders, check USD balance
            order_value = order.size * (order.price or self._current_price)
            if order_value > self._usd_balance:
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient USD balance. Need ${order_value:.2f}, have ${self._usd_balance:.2f}",
                )
        else:  # sell
            # For sell orders, check BTC balance
            if order.size > self._btc_balance:
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient BTC balance. Need {order.size:.8f}, have {self._btc_balance:.8f}",
                )

        return ValidationResult(valid=True)

    def _check_position_limits(self, order: OrderRequest) -> ValidationResult:
        """Check if order would exceed position limits."""
        # Calculate total portfolio value
        btc_value = self._btc_balance * self._current_price
        total_value = btc_value + self._usd_balance

        if total_value == Decimal("0"):
            return ValidationResult(valid=True)

        if order.side == "buy":
            # Calculate new position after buy
            order_value = order.size * (order.price or self._current_price)
            new_btc_value = btc_value + order_value
            new_position_percent = float(new_btc_value / total_value * 100)

            if new_position_percent > self.config.max_position_percent:
                return ValidationResult(
                    valid=False,
                    reason=f"Order would exceed position limit. New position: {new_position_percent:.1f}%, limit: {self.config.max_position_percent:.1f}%",
                )

            warnings = []
            if new_position_percent > self.config.max_position_percent * 0.9:
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

        if deviation_percent > self.config.price_sanity_percent:
            return ValidationResult(
                valid=False,
                reason=f"Limit price ${order.price:.2f} deviates {deviation_percent:.1f}% from market price ${self._current_price:.2f}",
            )

        return ValidationResult(valid=True)

    def _check_minimum_size(self, order: OrderRequest) -> ValidationResult:
        """Check if order meets minimum size requirements."""
        order_value_usd = float(order.size * (order.price or self._current_price))

        if order_value_usd < self.config.min_trade_usd:
            return ValidationResult(
                valid=False,
                reason=f"Order value ${order_value_usd:.2f} below minimum ${self.config.min_trade_usd:.2f}",
            )

        return ValidationResult(valid=True)

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
