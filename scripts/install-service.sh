#!/bin/bash
# Install coinbase-trader as a systemd service on Ubuntu
set -e

INSTALL_DIR="/opt/coinbase-trader"
SERVICE_USER="trader"
SERVICE_FILE="/etc/systemd/system/coinbase-trader.service"

echo "=== Crypto Trading Bot - Service Installation ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# Create service user if doesn't exist
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating user: $SERVICE_USER"
    useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
fi

# Create install directory
echo "Setting up $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"

# Copy files (assumes script is run from repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "Copying files from $REPO_DIR"
cp -r "$REPO_DIR/src" "$INSTALL_DIR/"
cp -r "$REPO_DIR/config" "$INSTALL_DIR/"
cp "$REPO_DIR/requirements.txt" "$INSTALL_DIR/"

# Copy .env if exists, otherwise create template
if [ -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
else
    echo "WARNING: No .env file found. Create $INSTALL_DIR/.env before starting."
fi

# Create virtual environment and install dependencies
echo "Setting up Python virtual environment"
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Install systemd service
echo "Installing systemd service"
cp "$SCRIPT_DIR/coinbase-trader.service" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable coinbase-trader

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit configuration: sudo nano $INSTALL_DIR/.env"
echo "  2. Start the service:  sudo systemctl start coinbase-trader"
echo "  3. Check status:       sudo systemctl status coinbase-trader"
echo "  4. View logs:          sudo journalctl -u coinbase-trader -f"
echo ""
