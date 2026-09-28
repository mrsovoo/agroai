#!/usr/bin/env bash
# Reset the local AgroAI database and re-ingest the knowledge directory.
# Usage: bash scripts/reset_database.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Removing database files under data/database/ ..."
rm -f data/database/agroai.db data/database/agroai.db-journal
rm -f data/database/agroai.db-wal data/database/agroai.db-shm

echo "Re-ingesting knowledge/ ..."
agroai ingest-dir ./knowledge

echo "Done. Try: agroai search \"pomidor\""
