#!/bin/bash
# Update claude-trader on the server
#
# Usage: sudo ./update.sh [source_directory]
#
# If source_directory is not provided, uses the directory containing this script.
# This script copies updated files from source to the installation directory.
set -e

INSTALL_DIR="/opt/claude-trader"
SERVICE_NAME="claude-trader"
SERVICE_USER="trader"

echo "=== Updating Claude Trader ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Determine source directory
if [ -n "$1" ]; then
    SOURCE_DIR="$1"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Validate source directory
if [ ! -d "$SOURCE_DIR/src" ]; then
    echo "ERROR: Source directory must contain 'src' folder"
    echo "Usage: sudo $0 /path/to/claude-trader"
    exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Target: $INSTALL_DIR"
echo ""

# Stop the service
echo "Stopping service..."
systemctl stop "$SERVICE_NAME" || true

# Backup current .env (contains secrets)
if [ -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env" "$INSTALL_DIR/.env.backup"
fi

# Copy updated files
echo "Updating files..."
cp -r "$SOURCE_DIR/src" "$INSTALL_DIR/"
cp -r "$SOURCE_DIR/config" "$INSTALL_DIR/"
cp "$SOURCE_DIR/requirements.txt" "$INSTALL_DIR/"

# Update service file if changed
if ! cmp -s "$SOURCE_DIR/scripts/claude-trader.service" "/etc/systemd/system/$SERVICE_NAME.service"; then
    echo "Updating service file..."
    cp "$SOURCE_DIR/scripts/claude-trader.service" "/etc/systemd/system/$SERVICE_NAME.service"
    systemctl daemon-reload
fi

# Clear Python bytecode cache to ensure fresh code runs
echo "Clearing bytecode cache..."
find "$INSTALL_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$INSTALL_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Restore .env
if [ -f "$INSTALL_DIR/.env.backup" ]; then
    mv "$INSTALL_DIR/.env.backup" "$INSTALL_DIR/.env"
fi

# Update dependencies
echo "Updating dependencies..."
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" --quiet

# Fix ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Restart service
echo "Starting service..."
systemctl start "$SERVICE_NAME"

echo ""
echo "=== Update Complete ==="
echo "Check status: sudo systemctl status $SERVICE_NAME"
echo "View logs:    sudo journalctl -u $SERVICE_NAME -f"
