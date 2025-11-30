"""
Market sentiment data fetching.

Provides:
- Bitcoin Fear & Greed Index from Alternative.me (free API)
- Historical trade performance summaries
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Fear & Greed API (free, no key needed)
FEAR_GREED_API = "https://api.alternative.me/fng/"


@dataclass
class FearGreedResult:
    """Fear & Greed Index result."""

    value: int  # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime
    error: Optional[str] = None

    @property
    def is_fear(self) -> bool:
        """Check if market is in fear territory."""
        return self.value < 47

    @property
    def is_greed(self) -> bool:
        """Check if market is in greed territory."""
        return self.value > 54

    @property
    def is_extreme(self) -> bool:
        """Check if sentiment is extreme (< 25 or > 75)."""
        return self.value < 25 or self.value > 75


@dataclass
class TradeSummary:
    """Summary of recent trading performance."""

    total_trades: int
    wins: int
    losses: int
    win_rate: float  # 0.0 to 1.0
    net_pnl: Decimal
    avg_pnl_per_trade: Decimal
    best_trade: Decimal
    worst_trade: Decimal
    days_analyzed: int


async def fetch_fear_greed_index(timeout: float = 10.0) -> FearGreedResult:
    """
    Fetch Bitcoin Fear & Greed Index from Alternative.me.

    The index aggregates sentiment from:
    - Volatility (25%)
    - Market volume (25%)
    - Social media (15%)
    - Surveys (15%)
    - Bitcoin dominance (10%)
    - Google trends (10%)

    Returns:
        FearGreedResult with value 0-100 and classification
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(FEAR_GREED_API)
            response.raise_for_status()

            data = response.json()

            if "data" not in data or not data["data"]:
                return FearGreedResult(
                    value=50,
                    classification="Neutral",
                    timestamp=datetime.utcnow(),
                    error="No data in response",
                )

            fng_data = data["data"][0]
            value = int(fng_data["value"])
            classification = fng_data["value_classification"]
            timestamp = datetime.fromtimestamp(int(fng_data["timestamp"]))

            logger.debug(
                "fear_greed_fetched",
                value=value,
                classification=classification,
            )

            return FearGreedResult(
                value=value,
                classification=classification,
                timestamp=timestamp,
            )

    except httpx.HTTPError as e:
        logger.warning("fear_greed_fetch_failed", error=str(e))
        return FearGreedResult(
            value=50,
            classification="Neutral",
            timestamp=datetime.utcnow(),
            error=str(e),
        )
    except Exception as e:
        logger.error("fear_greed_fetch_error", error=str(e))
        return FearGreedResult(
            value=50,
            classification="Neutral",
            timestamp=datetime.utcnow(),
            error=str(e),
        )


def get_trade_summary(db, days: int = 7, is_paper: Optional[bool] = None) -> TradeSummary:
    """
    Get summary of recent trading performance.

    Args:
        db: Database instance
        days: Number of days to analyze
        is_paper: Filter by paper/live trades (None = all)

    Returns:
        TradeSummary with win rate, P&L, etc.
    """
    from src.state.database import Trade

    cutoff = datetime.utcnow() - timedelta(days=days)

    with db.session() as session:
        query = session.query(Trade).filter(
            Trade.executed_at >= cutoff,
            Trade.side == "sell",  # Only sells have realized P&L
        )

        if is_paper is not None:
            query = query.filter(Trade.is_paper == is_paper)

        trades = query.order_by(Trade.executed_at.desc()).all()

    if not trades:
        return TradeSummary(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            net_pnl=Decimal("0"),
            avg_pnl_per_trade=Decimal("0"),
            best_trade=Decimal("0"),
            worst_trade=Decimal("0"),
            days_analyzed=days,
        )

    # Calculate statistics
    pnls = []
    wins = 0
    losses = 0

    for trade in trades:
        pnl = Decimal(trade.realized_pnl) if trade.realized_pnl else Decimal("0")
        pnls.append(pnl)

        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    total_trades = len(trades)
    net_pnl = sum(pnls)
    avg_pnl = net_pnl / total_trades if total_trades > 0 else Decimal("0")
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return TradeSummary(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        net_pnl=net_pnl,
        avg_pnl_per_trade=avg_pnl,
        best_trade=max(pnls) if pnls else Decimal("0"),
        worst_trade=min(pnls) if pnls else Decimal("0"),
        days_analyzed=days,
    )
