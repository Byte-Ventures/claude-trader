"""FastAPI dashboard server with WebSocket support."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from config.settings import get_settings
from src.state.database import BotMode, Database

from .routes import router, get_db, limiter
from .websocket import manager


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

logger = structlog.get_logger(__name__)

# Background task handle
_broadcaster_task = None
_db = None
_broadcaster_healthy = True  # Track broadcaster health for /health endpoint


async def state_broadcaster(db: Database):
    """Background task to poll DB and broadcast state updates."""
    global _broadcaster_healthy
    settings = get_settings()
    seen_notification_ids: set[int] = set()
    max_seen_ids = 500  # Prevent unbounded growth
    error_count = 0
    max_backoff = 30  # Max backoff in seconds

    logger.info("state_broadcaster_started")

    while True:
        try:
            if manager.connection_count > 0:
                state = db.get_state("dashboard_state")
                if state:
                    # Check for new notifications
                    notifications = db.get_recent_notifications(
                        limit=10,
                        is_paper=settings.is_paper_trading,
                    )
                    new_notifications = []
                    if notifications:
                        for n in notifications:
                            if n.id not in seen_notification_ids:
                                new_notifications.append({
                                    "id": n.id,
                                    "type": n.type,
                                    "title": n.title,
                                    "message": n.message,
                                    "created_at": n.created_at.isoformat() if n.created_at else "",
                                })
                                seen_notification_ids.add(n.id)

                        # Prevent unbounded growth - keep only recent IDs
                        if len(seen_notification_ids) > max_seen_ids:
                            seen_notification_ids.clear()
                            seen_notification_ids.update(n.id for n in notifications)

                    # Include new notifications in broadcast
                    state["new_notifications"] = new_notifications

                    # Include recent trades in broadcast (normal + Cramer if enabled)
                    normal_trades = db.get_recent_trades(
                        symbol=settings.trading_pair,
                        limit=20,
                        is_paper=settings.is_paper_trading,
                        bot_mode=BotMode.NORMAL,
                    )
                    all_trades = list(normal_trades)

                    # Include Cramer trades if enabled
                    if settings.enable_cramer_mode:
                        cramer_trades = db.get_recent_trades(
                            symbol=settings.trading_pair,
                            limit=20,
                            is_paper=True,
                            bot_mode=BotMode.INVERTED,
                        )
                        all_trades.extend(cramer_trades)

                    # Sort by time and limit
                    all_trades.sort(key=lambda t: t.executed_at or "", reverse=True)
                    all_trades = all_trades[:20]

                    state["recent_trades"] = [
                        {
                            "id": t.id,
                            "side": t.side,
                            "size": str(t.size),
                            "price": str(t.price),
                            "realized_pnl": str(t.realized_pnl) if t.realized_pnl else None,
                            "executed_at": t.executed_at.isoformat() if t.executed_at else "",
                            "bot_mode": t.bot_mode if hasattr(t, 'bot_mode') else "normal",
                        }
                        for t in all_trades
                    ]

                    await manager.broadcast(state)

            # Reset error count on success
            error_count = 0
            _broadcaster_healthy = True
            await asyncio.sleep(1.5)  # Normal poll interval

        except (sqlite3.Error, OSError, ConnectionError) as e:
            # Database or I/O errors - use backoff
            error_count += 1
            _broadcaster_healthy = error_count < 5  # Unhealthy after 5 consecutive errors
            backoff = min(1.5 * (2 ** error_count), max_backoff)
            logger.error("broadcast_error", error=str(e), backoff=backoff, error_count=error_count)
            await asyncio.sleep(backoff)
        except Exception as e:
            # Unexpected errors - log with full traceback but continue
            error_count += 1
            _broadcaster_healthy = error_count < 5
            backoff = min(1.5 * (2 ** error_count), max_backoff)
            logger.error("broadcast_unexpected_error", error=str(e), exc_info=True, backoff=backoff)
            await asyncio.sleep(backoff)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _broadcaster_task, _db

    # Startup
    logger.info("dashboard_starting")
    _db = get_db()  # Use singleton from routes
    _broadcaster_task = asyncio.create_task(state_broadcaster(_db))

    yield

    # Shutdown
    logger.info("dashboard_stopping")
    if _broadcaster_task:
        _broadcaster_task.cancel()
        try:
            await _broadcaster_task
        except asyncio.CancelledError:
            pass
    if _db and hasattr(_db, 'engine'):
        _db.engine.dispose()
        logger.info("database_connections_closed")


app = FastAPI(
    title="Claude Bitcoin Trader",
    description="Live trading dashboard with real-time updates",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include REST routes
app.include_router(router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    if not await manager.connect(websocket):
        return  # Rate limited, connection already closed
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
            # Could handle client commands here if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except (RuntimeError, asyncio.CancelledError) as e:
        # Expected errors during shutdown or connection issues
        logger.debug("websocket_closed", reason=str(e))
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("unexpected_websocket_error", error=str(e), exc_info=True)
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "ok" if _broadcaster_healthy else "degraded"
    return {
        "status": status,
        "connections": manager.connection_count,
        "broadcaster": "healthy" if _broadcaster_healthy else "unhealthy",
    }


def main():
    """Run the dashboard server."""
    settings = get_settings()

    logger.info(
        "starting_dashboard_server",
        host=settings.dashboard_host,
        port=settings.dashboard_port,
    )

    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
