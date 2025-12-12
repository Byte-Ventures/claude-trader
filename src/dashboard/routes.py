"""REST API routes for dashboard."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from config.settings import get_settings
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.state.database import Database

from .models import (
    CandleData,
    DailyStatsInfo,
    DashboardState,
    NotificationRecord,
    PositionInfo,
    TradeRecord,
)

router = APIRouter()

# Rate limiter instance - limits are intentionally permissive for single-user dashboard:
# - /api/state: 60/min allows initial polling before WebSocket connects
# - /api/candles: 30/min is sufficient for page loads and refreshes
# - /api/performance: 20/min is stricter as it's the most expensive query
# Note: Without authentication, these limits assume trusted network access.
# For public deployment, add authentication or tighten limits.
limiter = Limiter(key_func=get_remote_address)


@lru_cache(maxsize=1)
def get_db() -> Database:
    """Get singleton database instance.

    Uses lru_cache for singleton pattern. The Database instance is disposed
    in server.py lifespan handler on shutdown. This is acceptable for SQLite
    which handles connection pooling internally.
    """
    settings = get_settings()
    return Database(settings.database_path)


# TODO: Add integration tests for dashboard endpoints (see tests/dashboard/)
# Priority tests: paper/live data separation, WebSocket reconnection, notification deduplication


@router.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text())
    return HTMLResponse(content="<h1>Dashboard template not found</h1>", status_code=404)


@router.get("/api/state")
@limiter.limit("60/minute")
async def get_current_state(request: Request) -> Optional[DashboardState]:
    """Get the current trading state from the daemon."""
    db = get_db()
    state = db.get_state("dashboard_state")
    if state:
        return DashboardState(**state)
    return None


@router.get("/api/candles")
@limiter.limit("30/minute")
async def get_candles(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CandleData]:
    """Get historical candle data (most recent N candles, oldest first for charting)."""
    settings = get_settings()
    db = get_db()

    # DB has inconsistent interval formats (legacy: ONE_MINUTE, new: 1m) - try both.
    # TODO: Migrate database to consistent format and remove this workaround.
    interval_map = {
        "ONE_MINUTE": "1m",
        "FIVE_MINUTE": "5m",
        "FIFTEEN_MINUTE": "15m",
        "THIRTY_MINUTE": "30m",
        "ONE_HOUR": "1h",
        "TWO_HOUR": "2h",
        "SIX_HOUR": "6h",
        "ONE_DAY": "1d",
    }
    intervals_to_try = [settings.candle_interval, interval_map.get(settings.candle_interval, "1h")]

    # Query directly to get LATEST N candles (db.get_rates returns oldest N)
    from src.state.database import RateHistory
    rates = []
    with db.session() as session:
        for interval in intervals_to_try:
            rates = (
                session.query(RateHistory)
                .filter(
                    RateHistory.symbol == settings.trading_pair,
                    RateHistory.exchange == settings.exchange.value.capitalize(),
                    RateHistory.interval == interval,
                    RateHistory.is_paper == settings.is_paper_trading,
                )
                .order_by(RateHistory.timestamp.desc())
                .limit(limit)
                .all()
            )
            if rates:
                break

    # Reverse to get oldest-first for charting
    return [
        CandleData(
            timestamp=r.timestamp.isoformat() if r.timestamp else "",
            open=r.open_price or "0",
            high=r.high_price or "0",
            low=r.low_price or "0",
            close=r.close_price or "0",
            volume=r.volume or "0",
        )
        for r in reversed(rates)
    ]


@router.get("/api/trades")
@limiter.limit("30/minute")
async def get_trades(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[TradeRecord]:
    """Get recent trades."""
    settings = get_settings()
    db = get_db()

    trades = db.get_recent_trades(
        limit=limit,
        is_paper=settings.is_paper_trading,
        symbol=settings.trading_pair,
    )

    return [
        TradeRecord(
            id=t.id,
            side=t.side,
            size=t.size,
            price=t.filled_price or t.price or "0",
            fee=t.fee or "0",
            realized_pnl=t.realized_pnl,
            executed_at=t.executed_at.isoformat() if t.executed_at else "",
        )
        for t in trades
    ]


@router.get("/api/position")
@limiter.limit("30/minute")
async def get_position(request: Request) -> Optional[PositionInfo]:
    """Get current position."""
    settings = get_settings()
    db = get_db()

    position = db.get_current_position(
        symbol=settings.trading_pair,
        is_paper=settings.is_paper_trading,
    )

    if position:
        return PositionInfo(
            symbol=position.symbol,
            quantity=position.quantity,
            average_cost=position.average_cost,
            unrealized_pnl=position.unrealized_pnl or "0",
            is_paper=position.is_paper,
        )
    return None


@router.get("/api/stats")
@limiter.limit("30/minute")
async def get_daily_stats(request: Request) -> Optional[DailyStatsInfo]:
    """Get today's trading statistics."""
    settings = get_settings()
    db = get_db()

    stats = db.get_daily_stats(is_paper=settings.is_paper_trading)

    if stats:
        return DailyStatsInfo(
            date=str(stats.date),
            starting_balance=stats.starting_balance or "0",
            ending_balance=stats.ending_balance or "0",
            realized_pnl=stats.realized_pnl or "0",
            total_trades=stats.total_trades or 0,
            is_paper=stats.is_paper,
        )
    return None


@router.get("/api/config")
@limiter.limit("30/minute")
async def get_config(request: Request) -> dict:
    """Get non-sensitive configuration info."""
    settings = get_settings()
    return {
        "trading_pair": settings.trading_pair,
        "trading_mode": settings.trading_mode.value,
        "exchange": settings.exchange.value,
        "signal_threshold": settings.signal_threshold,
        "candle_interval": settings.candle_interval,
        "position_size_percent": settings.position_size_percent,
    }


@router.get("/api/notifications")
@limiter.limit("30/minute")
async def get_notifications(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[NotificationRecord]:
    """Get recent notifications."""
    settings = get_settings()
    db = get_db()

    notifications = db.get_recent_notifications(
        limit=limit,
        is_paper=settings.is_paper_trading,
    )

    return [
        NotificationRecord(
            id=n.id,
            type=n.type,
            title=n.title,
            message=n.message,
            created_at=n.created_at.isoformat() if n.created_at else "",
        )
        for n in notifications
    ]


@router.get("/api/performance")
@limiter.limit("20/minute")
async def get_performance(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
) -> list[dict]:
    """Get daily performance stats for charting."""
    from datetime import date, timedelta

    settings = get_settings()
    db = get_db()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    stats = db.get_daily_stats_range(
        start_date=start_date,
        end_date=end_date,
        is_paper=settings.is_paper_trading,
    )

    return [
        {
            "date": str(s.date),
            "starting_balance": s.starting_balance or "0",
            "ending_balance": s.ending_balance or "0",
            "starting_price": s.starting_price or "0",
            "ending_price": s.ending_price or "0",
            "realized_pnl": s.realized_pnl or "0",
        }
        for s in stats
    ]
