#!/bin/bash
# Hot-reload claude-trader configuration without restarting
#
# Usage: sudo ./reload-config.sh
#
# Sends SIGUSR2 to the running service, triggering config reload.
# The service will re-read .env and update settings on the fly.

SERVICE_NAME="claude-trader"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Get the main PID of the service
PID=$(systemctl show -p MainPID --value "$SERVICE_NAME")

if [ -z "$PID" ] || [ "$PID" -eq 0 ]; then
    echo "ERROR: Service '$SERVICE_NAME' is not running"
    exit 1
fi

echo "Sending SIGUSR2 to $SERVICE_NAME (PID: $PID)..."
kill -USR2 "$PID"

echo "Config reload triggered. Check logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
