"""FastAPI dashboard server with WebSocket support."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from src.state.database import Database

from .routes import router, get_db
from .websocket import manager

logger = structlog.get_logger(__name__)

# Background task handle
_broadcaster_task = None
_db = None


async def state_broadcaster(db: Database):
    """Background task to poll DB and broadcast state updates."""
    settings = get_settings()
    last_notification_id = 0
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
                            if n.id > last_notification_id:
                                new_notifications.append({
                                    "id": n.id,
                                    "type": n.type,
                                    "title": n.title,
                                    "message": n.message,
                                    "created_at": n.created_at.isoformat() if n.created_at else "",
                                })
                        if notifications:
                            last_notification_id = max(n.id for n in notifications)

                    # Include new notifications in broadcast
                    state["new_notifications"] = new_notifications
                    await manager.broadcast(state)

            # Reset error count on success
            error_count = 0
            await asyncio.sleep(1.5)  # Normal poll interval

        except Exception as e:
            error_count += 1
            backoff = min(1.5 * (2 ** error_count), max_backoff)
            logger.error("broadcast_error", error=str(e), backoff=backoff, error_count=error_count)
            await asyncio.sleep(backoff)  # Exponential backoff on errors


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
    title="Claude Trader Dashboard",
    description="Live trading dashboard with real-time updates",
    lifespan=lifespan,
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include REST routes
app.include_router(router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
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
    return {"status": "ok", "connections": manager.connection_count}


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
