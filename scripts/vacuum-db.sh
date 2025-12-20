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

# Get file size before VACUUM
SIZE_BEFORE=$(du -h "$DB_PATH" | cut -f1)

echo "Running VACUUM on $DB_PATH..."
echo "Size before: $SIZE_BEFORE"

# Run VACUUM command
sqlite3 "$DB_PATH" "VACUUM;"

# Get file size after VACUUM
SIZE_AFTER=$(du -h "$DB_PATH" | cut -f1)

echo "Size after:  $SIZE_AFTER"
echo "Done. Database optimized."
