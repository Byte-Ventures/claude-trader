"""Pydantic models for dashboard API responses."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class IndicatorValues(BaseModel):
    """Technical indicator values."""

    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None
    volatility: str = "normal"


class SignalInfo(BaseModel):
    """Trading signal information."""

    score: int
    action: str
    threshold: int
    breakdown: dict[str, int]
    confidence: float


class PortfolioInfo(BaseModel):
    """Portfolio balance information."""

    quote_balance: str
    base_balance: str
    portfolio_value: str
    position_percent: float


class SafetyStatus(BaseModel):
    """Safety system status."""

    circuit_breaker: str
    can_trade: bool


class DashboardState(BaseModel):
    """Complete dashboard state for WebSocket broadcast."""

    timestamp: str
    price: str
    signal: SignalInfo
    indicators: IndicatorValues
    portfolio: PortfolioInfo
    regime: str
    safety: SafetyStatus
    trading_pair: str
    is_paper: bool


class CandleData(BaseModel):
    """OHLCV candle data."""

    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: str


class TradeRecord(BaseModel):
    """Trade record for display."""

    id: int
    side: str
    size: str
    price: str
    fee: str
    realized_pnl: Optional[str]
    executed_at: str


class PositionInfo(BaseModel):
    """Current position information."""

    symbol: str
    quantity: str
    average_cost: str
    unrealized_pnl: str
    is_paper: bool


class DailyStatsInfo(BaseModel):
    """Daily trading statistics."""

    date: str
    starting_balance: str
    ending_balance: str
    realized_pnl: str
    total_trades: int
    is_paper: bool


class NotificationRecord(BaseModel):
    """Notification record for display."""

    id: int
    type: str
    title: str
    message: str
    created_at: str
