#!/bin/bash
# Update claude-trader on the server
set -e

INSTALL_DIR="/opt/claude-trader"
SERVICE_NAME="claude-trader"

echo "=== Updating Crypto Trading Bot ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Stop the service
echo "Stopping service..."
systemctl stop "$SERVICE_NAME"

# Pull latest code
echo "Pulling latest code..."
cd "$INSTALL_DIR"
sudo -u trader git pull

# Update dependencies
echo "Updating dependencies..."
"$INSTALL_DIR/venv/bin/pip" install -r requirements.txt --quiet

# Restart service
echo "Starting service..."
systemctl start "$SERVICE_NAME"

echo ""
echo "=== Update Complete ==="
echo "Check status: sudo systemctl status $SERVICE_NAME"
echo "View logs:    sudo journalctl -u $SERVICE_NAME -f"
