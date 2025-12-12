"""WebSocket connection manager for real-time updates."""

from collections import defaultdict

from fastapi import WebSocket
from starlette.websockets import WebSocketState
import structlog

logger = structlog.get_logger(__name__)

# Rate limits for WebSocket connections
MAX_CONNECTIONS_PER_IP = 5
MAX_TOTAL_CONNECTIONS = 50


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._connections_by_ip: dict[str, list[WebSocket]] = defaultdict(list)

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Extract client IP from WebSocket connection."""
        # Check for forwarded header (behind reverse proxy)
        forwarded = websocket.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Fall back to direct client
        if websocket.client:
            return websocket.client.host
        return "unknown"

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept and track a new WebSocket connection.

        Returns True if connection accepted, False if rate limited.
        """
        client_ip = self._get_client_ip(websocket)

        # Check per-IP limit
        if len(self._connections_by_ip[client_ip]) >= MAX_CONNECTIONS_PER_IP:
            logger.warning("websocket_rate_limited", ip=client_ip, reason="per_ip_limit")
            await websocket.close(code=1008, reason="Too many connections from this IP")
            return False

        # Check global limit
        if len(self.active_connections) >= MAX_TOTAL_CONNECTIONS:
            logger.warning("websocket_rate_limited", ip=client_ip, reason="global_limit")
            await websocket.close(code=1008, reason="Server at capacity")
            return False

        await websocket.accept()
        self.active_connections.append(websocket)
        self._connections_by_ip[client_ip].append(websocket)
        logger.info("websocket_connected", ip=client_ip, total=len(self.active_connections))
        return True

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

            # Clean up IP tracking
            for ip, connections in self._connections_by_ip.items():
                if websocket in connections:
                    connections.remove(websocket)
                    break

            logger.info("websocket_disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except (RuntimeError, ConnectionError, OSError) as e:
                # Connection closed or network error - mark for cleanup
                logger.debug("broadcast_send_failed", error=str(e))
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()
