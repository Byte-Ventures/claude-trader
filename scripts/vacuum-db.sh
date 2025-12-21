#!/bin/bash
# Reclaim disk space after large database cleanups
#
# Usage: ./scripts/vacuum-db.sh [db_path] [--force]
#
# Default: data/trading.db
# Options:
#   --force    Skip confirmation prompt (for automation)
#
# IMPORTANT: Consider backing up the database before running in production.
# While VACUUM is generally safe, extra caution with financial data is prudent.
#
# Note: VACUUM requires exclusive lock - run when bot is stopped or during maintenance.
# This operation rebuilds the database file to reclaim space from deleted records.
# Particularly useful after signal history cleanup or other large deletions.
# Requires: Linux (uses readlink -f)

set -e

# Parse arguments
FORCE_MODE=false
DB_PATH_ARG=""
DB_PATH_COUNT=0
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_MODE=true
    else
        DB_PATH_ARG="$arg"
        DB_PATH_COUNT=$((DB_PATH_COUNT + 1))
    fi
done

# Reject multiple database path arguments to prevent confusion
if [ $DB_PATH_COUNT -gt 1 ]; then
    echo "Error: Multiple database paths provided. Please specify only one database path."
    echo "Usage: ./scripts/vacuum-db.sh [db_path] [--force]"
    exit 1
fi

# Check if sqlite3 is installed
if ! command -v sqlite3 &> /dev/null; then
    echo "Error: sqlite3 command not found. Install with: sudo apt-get install sqlite3"
    exit 1
fi

# Default database path with absolute path resolution
DB_PATH="$(readlink -f "${DB_PATH_ARG:-data/trading.db}")"

# Create lock file to prevent concurrent executions
# This prevents multiple simultaneous VACUUM operations which could cause "database is locked" errors
LOCK_FILE="/tmp/vacuum-db.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Error: Another vacuum operation is already running"
    echo "If you're sure no other vacuum is running, remove: $LOCK_FILE"
    exit 1
fi

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
    if [ "$FORCE_MODE" = true ]; then
        echo "Force mode enabled - proceeding despite service running"
    else
        read -p "Do you want to continue anyway? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo "Operation cancelled. Stop the service first with: sudo systemctl stop claude-trader"
            exit 0
        fi
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
# Use mktemp for safer temp file handling (handles /tmp being read-only or full)
ERROR_FILE=$(mktemp)
if ! sqlite3 "$DB_PATH" "VACUUM;" 2>"$ERROR_FILE"; then
    echo "ERROR: VACUUM operation failed!"
    echo ""
    echo "Common causes:"
    echo "  1. Database is locked by another process (bot still running?)"
    echo "  2. Insufficient disk space (VACUUM needs temp space ~equal to DB size)"
    echo "  3. Database corruption (run: sqlite3 $DB_PATH 'PRAGMA integrity_check;')"
    echo ""
    if [ -s "$ERROR_FILE" ]; then
        echo "Error details:"
        cat "$ERROR_FILE"
    fi
    rm -f "$ERROR_FILE"
    exit 1
fi
rm -f "$ERROR_FILE"

# Get file size after VACUUM
SIZE_AFTER=$(du -h "$DB_PATH" | cut -f1)

echo "Size after:  $SIZE_AFTER"
echo "Done. Database optimized."
