"""
ATR-based dynamic position sizing.

Calculates optimal position size based on:
- Account balance
- Current volatility (ATR)
- Risk per trade
- Signal strength
- Safety system multipliers
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pandas as pd
import structlog

from src.indicators.atr import calculate_atr, get_position_size_multiplier

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing."""

    # Target position limit during order calculation (conservative default).
    # This is the "soft limit" that guides normal order sizing.
    # The validator has a separate "hard limit" (80%) that catches edge cases.
    # Two-tier design: 40% target prevents over-concentration, 80% hard stop for safety.
    max_position_percent: float = 40.0
    risk_per_trade_percent: float = 0.5  # Risk 0.5% per trade (conservative)
    stop_loss_atr_multiplier: float = 1.5  # Stop at 1.5x ATR
    min_trade_quote: float = 100.0  # Minimum trade size in quote currency
    min_trade_base: float = 0.0001  # Minimum trade size in base currency (e.g., BTC)
    min_stop_loss_percent: float = 1.5  # Minimum stop as % of price (floor for short timeframes)


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    size_base: Decimal  # Size in base currency (e.g., BTC)
    size_quote: Decimal  # Size in quote currency (e.g., USD/EUR)
    stop_loss_price: Decimal
    take_profit_price: Decimal
    risk_amount_quote: Decimal  # Risk in quote currency
    position_percent: float


class PositionSizer:
    """
    ATR-based position sizing calculator.

    Determines optimal trade size based on:
    1. Account balance and max position
    2. Current market volatility (ATR)
    3. Risk tolerance (max loss per trade)
    4. Signal strength (stronger = larger position)
    5. Safety system multipliers
    """

    def __init__(
        self,
        config: Optional[PositionSizeConfig] = None,
        atr_period: int = 14,
        take_profit_atr_multiplier: float = 2.0,
    ):
        """
        Initialize position sizer.

        Args:
            config: Position sizing configuration
            atr_period: ATR calculation period
            take_profit_atr_multiplier: Take profit distance as ATR multiple
        """
        self.config = config or PositionSizeConfig()
        self.atr_period = atr_period
        self.take_profit_multiplier = take_profit_atr_multiplier

    def update_settings(
        self,
        max_position_percent: Optional[float] = None,
        stop_loss_atr_multiplier: Optional[float] = None,
        take_profit_atr_multiplier: Optional[float] = None,
        atr_period: Optional[int] = None,
        min_stop_loss_percent: Optional[float] = None,
    ) -> None:
        """
        Update position sizer settings at runtime.

        Only updates parameters that are explicitly provided (not None).
        """
        if max_position_percent is not None:
            self.config.max_position_percent = max_position_percent
        if stop_loss_atr_multiplier is not None:
            self.config.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        if take_profit_atr_multiplier is not None:
            self.take_profit_multiplier = take_profit_atr_multiplier
        if atr_period is not None:
            self.atr_period = atr_period
        if min_stop_loss_percent is not None:
            self.config.min_stop_loss_percent = min_stop_loss_percent

        logger.info("position_sizer_settings_updated")

    def calculate_size(
        self,
        df: pd.DataFrame,
        current_price: Decimal,
        quote_balance: Decimal,
        base_balance: Decimal,
        signal_strength: int,
        side: str = "buy",
        safety_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price in quote currency
            quote_balance: Available quote currency balance (e.g., USD/EUR)
            base_balance: Current base currency holdings (e.g., BTC)
            signal_strength: Signal score (0-100)
            side: Trade side ("buy" or "sell")
            safety_multiplier: Multiplier from safety systems (0-1)

        Returns:
            PositionSizeResult with size and price targets
        """
        # Calculate current ATR
        if df.empty or len(df) < self.atr_period + 1:
            return self._zero_result(current_price, side)

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        atr_result = calculate_atr(high, low, close, self.atr_period)
        current_atr = atr_result.atr.iloc[-1]

        if pd.isna(current_atr) or current_atr <= 0:
            return self._zero_result(current_price, side)

        atr_decimal = Decimal(str(current_atr))

        # Calculate total portfolio value
        base_value = base_balance * current_price
        total_value = quote_balance + base_value

        if total_value <= 0:
            return self._zero_result(current_price, side)

        # Step 1: Calculate risk amount (% of portfolio willing to lose)
        risk_amount = total_value * Decimal(str(self.config.risk_per_trade_percent)) / Decimal("100")

        # Step 2: Calculate stop-loss distance
        # Use the LARGER of ATR-based distance or minimum percentage distance
        # This ensures stop is never too tight on short timeframes (e.g., 15-min candles)
        atr_stop_distance = atr_decimal * Decimal(str(self.config.stop_loss_atr_multiplier))
        min_pct_distance = current_price * Decimal(str(self.config.min_stop_loss_percent)) / Decimal("100")
        stop_distance = max(atr_stop_distance, min_pct_distance)

        # Step 3: Calculate base position size from risk
        # size = risk_amount / stop_distance
        if stop_distance > 0:
            size_base = risk_amount / stop_distance
        else:
            size_base = Decimal("0")

        # Step 4: Apply signal strength multiplier (60-100 -> 0.6-1.0)
        strength_multiplier = Decimal(str(abs(signal_strength) / 100))
        size_base *= strength_multiplier

        # Step 5: Apply volatility multiplier (reduce size in high volatility)
        volatility_multiplier = get_position_size_multiplier(atr_result)
        size_base *= Decimal(str(volatility_multiplier))

        # Step 6: Apply safety system multiplier
        size_base *= Decimal(str(safety_multiplier))

        # Step 7: Apply maximum position limit (accounting for existing position)
        max_position_base = (total_value * Decimal(str(self.config.max_position_percent)) / Decimal("100")) / current_price

        if side == "buy":
            # Calculate how much more we can buy before hitting the position limit
            # Note: This uses position_sizer's max (40% default) as the target.
            # The validator has a separate hard limit (80% default) as a safety net.
            max_additional_base = max_position_base - base_balance
            if max_additional_base <= 0:
                # Already at or above position limit
                current_position_pct = float(base_balance * current_price / total_value * 100)
                logger.info(
                    "buy_blocked_at_position_limit",
                    current_position_pct=f"{current_position_pct:.1f}%",
                    max_position_pct=f"{self.config.max_position_percent:.1f}%",
                    base_balance=str(base_balance),
                    max_position_base=str(max_position_base),
                )
                return self._zero_result(current_price, side)
            # Also cap at available quote currency
            max_from_balance = quote_balance / current_price
            size_base = min(size_base, max_additional_base, max_from_balance)
        else:  # sell
            # For sells, only cap at available base currency (no position limit for exits)
            size_base = min(size_base, base_balance)

        # Step 8: Ensure minimum trade size
        size_quote = size_base * current_price
        if size_quote < Decimal(str(self.config.min_trade_quote)):
            logger.info(
                "position_size_below_minimum",
                size_quote=str(size_quote),
                min_trade_quote=self.config.min_trade_quote,
                size_base=str(size_base),
                safety_multiplier=safety_multiplier,
            )
            return self._zero_result(current_price, side)

        # Calculate stop-loss and take-profit prices
        if side == "buy":
            stop_loss = current_price - stop_distance
            take_profit = current_price + (atr_decimal * Decimal(str(self.take_profit_multiplier)))
        else:  # sell
            stop_loss = current_price + stop_distance
            take_profit = current_price - (atr_decimal * Decimal(str(self.take_profit_multiplier)))

        # Calculate actual position percentage
        position_percent = float(size_quote / total_value * 100)

        result = PositionSizeResult(
            size_base=size_base.quantize(Decimal("0.00000001")),
            size_quote=size_quote.quantize(Decimal("0.01")),
            stop_loss_price=stop_loss.quantize(Decimal("0.01")),
            take_profit_price=take_profit.quantize(Decimal("0.01")),
            risk_amount_quote=risk_amount.quantize(Decimal("0.01")),
            position_percent=position_percent,
        )

        # Log which method determined the stop distance
        used_min_pct = min_pct_distance > atr_stop_distance
        if used_min_pct:
            logger.info(
                "stop_using_min_percent",
                atr_distance=str(atr_stop_distance),
                min_pct_distance=str(min_pct_distance),
                min_pct=self.config.min_stop_loss_percent,
            )

        logger.debug(
            "position_size_calculated",
            size_base=str(result.size_base),
            size_quote=str(result.size_quote),
            stop_loss=str(result.stop_loss_price),
            take_profit=str(result.take_profit_price),
            atr=str(atr_decimal),
            stop_distance=str(stop_distance),
            stop_method="min_pct" if used_min_pct else "atr",
            volatility_mult=volatility_multiplier,
            safety_mult=safety_multiplier,
        )

        return result

    def _zero_result(self, current_price: Decimal, side: str) -> PositionSizeResult:
        """Return a zero-size result."""
        return PositionSizeResult(
            size_base=Decimal("0"),
            size_quote=Decimal("0"),
            stop_loss_price=current_price,
            take_profit_price=current_price,
            risk_amount_quote=Decimal("0"),
            position_percent=0.0,
        )

    def calculate_sell_all_size(
        self,
        base_balance: Decimal,
        current_price: Decimal,
    ) -> PositionSizeResult:
        """
        Calculate size for selling entire position.

        Args:
            base_balance: Current base currency holdings (e.g., BTC)
            current_price: Current price in quote currency

        Returns:
            PositionSizeResult for full position exit
        """
        size_quote = base_balance * current_price

        return PositionSizeResult(
            size_base=base_balance,
            size_quote=size_quote,
            stop_loss_price=Decimal("0"),
            take_profit_price=Decimal("0"),
            risk_amount_quote=Decimal("0"),
            position_percent=100.0,
        )
