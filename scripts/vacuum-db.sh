#!/bin/bash
# Reclaim disk space after large database cleanups
#
# Usage: ./scripts/vacuum-db.sh [db_path]
#
# Default: data/trading.db
#
# Note: VACUUM requires exclusive lock - run when bot is stopped or during maintenance.
# This operation rebuilds the database file to reclaim space from deleted records.
# Particularly useful after signal history cleanup or other large deletions.

set -e

# Check if sqlite3 is installed
if ! command -v sqlite3 &> /dev/null; then
    echo "Error: sqlite3 command not found. Install with: sudo apt-get install sqlite3"
    exit 1
fi

# Default database path with absolute path resolution
DB_PATH="$(readlink -f "${1:-data/trading.db}")"

# Check if database file exists
if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    exit 1
fi

# Safety check: Verify bot service is stopped to prevent database locking issues
# In a financial trading system, running VACUUM while the bot is active could:
# - Block active trades mid-execution
# - Cause transaction timeouts
# - Lead to inconsistent state
if systemctl is-active --quiet claude-trader 2>/dev/null; then
    echo "WARNING: claude-trader service is currently running!"
    echo "Running VACUUM while the bot is active may cause database lock issues."
    echo ""
    read -p "Do you want to continue anyway? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Operation cancelled. Stop the service first with: sudo systemctl stop claude-trader"
        exit 0
    fi
fi

# Get file size before VACUUM
SIZE_BEFORE=$(du -h "$DB_PATH" | cut -f1)

echo "Running VACUUM on $DB_PATH..."
echo "Size before: $SIZE_BEFORE"

# Run VACUUM command with explicit error handling
# VACUUM can fail due to:
# - Database locked (another process has exclusive lock)
# - Insufficient disk space (needs temp space ~equal to database size)
# - Database corruption
if ! sqlite3 "$DB_PATH" "VACUUM;" 2>/tmp/vacuum_error.txt; then
    echo "ERROR: VACUUM operation failed!"
    echo ""
    echo "Common causes:"
    echo "  1. Database is locked by another process (bot still running?)"
    echo "  2. Insufficient disk space (VACUUM needs temp space ~equal to DB size)"
    echo "  3. Database corruption (run: sqlite3 $DB_PATH 'PRAGMA integrity_check;')"
    echo ""
    if [ -s /tmp/vacuum_error.txt ]; then
        echo "Error details:"
        cat /tmp/vacuum_error.txt
    fi
    rm -f /tmp/vacuum_error.txt
    exit 1
fi
rm -f /tmp/vacuum_error.txt

# Get file size after VACUUM
SIZE_AFTER=$(du -h "$DB_PATH" | cut -f1)

echo "Size after:  $SIZE_AFTER"
echo "Done. Database optimized."
