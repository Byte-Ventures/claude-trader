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

    max_position_percent: float = 75.0  # Maximum 75% of portfolio
    risk_per_trade_percent: float = 2.0  # Risk 2% per trade
    stop_loss_atr_multiplier: float = 1.5  # Stop at 1.5x ATR
    min_trade_usd: float = 10.0  # Minimum trade size


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    size_btc: Decimal
    size_usd: Decimal
    stop_loss_price: Decimal
    take_profit_price: Decimal
    risk_amount_usd: Decimal
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

    def calculate_size(
        self,
        df: pd.DataFrame,
        current_price: Decimal,
        usd_balance: Decimal,
        btc_balance: Decimal,
        signal_strength: int,
        side: str = "buy",
        safety_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current BTC price
            usd_balance: Available USD balance
            btc_balance: Current BTC holdings
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
        btc_value = btc_balance * current_price
        total_value = usd_balance + btc_value

        if total_value <= 0:
            return self._zero_result(current_price, side)

        # Step 1: Calculate risk amount (% of portfolio willing to lose)
        risk_amount = total_value * Decimal(str(self.config.risk_per_trade_percent / 100))

        # Step 2: Calculate stop-loss distance
        stop_distance = atr_decimal * Decimal(str(self.config.stop_loss_atr_multiplier))

        # Step 3: Calculate base position size from risk
        # size = risk_amount / stop_distance
        if stop_distance > 0:
            size_btc = risk_amount / stop_distance
        else:
            size_btc = Decimal("0")

        # Step 4: Apply signal strength multiplier (60-100 -> 0.6-1.0)
        strength_multiplier = Decimal(str(abs(signal_strength) / 100))
        size_btc *= strength_multiplier

        # Step 5: Apply volatility multiplier (reduce size in high volatility)
        volatility_multiplier = get_position_size_multiplier(atr_result)
        size_btc *= Decimal(str(volatility_multiplier))

        # Step 6: Apply safety system multiplier
        size_btc *= Decimal(str(safety_multiplier))

        # Step 7: Apply maximum position limit
        max_position_btc = (total_value * Decimal(str(self.config.max_position_percent / 100))) / current_price

        if side == "buy":
            # For buys, also cap at available USD
            max_from_balance = usd_balance / current_price
            size_btc = min(size_btc, max_position_btc, max_from_balance)
        else:  # sell
            # For sells, cap at available BTC
            size_btc = min(size_btc, btc_balance, max_position_btc)

        # Step 8: Ensure minimum trade size
        size_usd = size_btc * current_price
        if size_usd < Decimal(str(self.config.min_trade_usd)):
            return self._zero_result(current_price, side)

        # Calculate stop-loss and take-profit prices
        if side == "buy":
            stop_loss = current_price - stop_distance
            take_profit = current_price + (atr_decimal * Decimal(str(self.take_profit_multiplier)))
        else:  # sell
            stop_loss = current_price + stop_distance
            take_profit = current_price - (atr_decimal * Decimal(str(self.take_profit_multiplier)))

        # Calculate actual position percentage
        position_percent = float(size_usd / total_value * 100)

        result = PositionSizeResult(
            size_btc=size_btc.quantize(Decimal("0.00000001")),
            size_usd=size_usd.quantize(Decimal("0.01")),
            stop_loss_price=stop_loss.quantize(Decimal("0.01")),
            take_profit_price=take_profit.quantize(Decimal("0.01")),
            risk_amount_usd=risk_amount.quantize(Decimal("0.01")),
            position_percent=position_percent,
        )

        logger.debug(
            "position_size_calculated",
            size_btc=str(result.size_btc),
            size_usd=str(result.size_usd),
            stop_loss=str(result.stop_loss_price),
            take_profit=str(result.take_profit_price),
            atr=str(atr_decimal),
            volatility_mult=volatility_multiplier,
            safety_mult=safety_multiplier,
        )

        return result

    def _zero_result(self, current_price: Decimal, side: str) -> PositionSizeResult:
        """Return a zero-size result."""
        return PositionSizeResult(
            size_btc=Decimal("0"),
            size_usd=Decimal("0"),
            stop_loss_price=current_price,
            take_profit_price=current_price,
            risk_amount_usd=Decimal("0"),
            position_percent=0.0,
        )

    def calculate_sell_all_size(
        self,
        btc_balance: Decimal,
        current_price: Decimal,
    ) -> PositionSizeResult:
        """
        Calculate size for selling entire position.

        Args:
            btc_balance: Current BTC holdings
            current_price: Current BTC price

        Returns:
            PositionSizeResult for full position exit
        """
        size_usd = btc_balance * current_price

        return PositionSizeResult(
            size_btc=btc_balance,
            size_usd=size_usd,
            stop_loss_price=Decimal("0"),
            take_profit_price=Decimal("0"),
            risk_amount_usd=Decimal("0"),
            position_percent=100.0,
        )
