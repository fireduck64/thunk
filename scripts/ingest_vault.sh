#!/bin/bash
set -euo pipefail

VENV_DIR="${THUNK_VENV_DIR:-/var/nvme/thunk_venv}"
VAULT_DIR="/vault/ebook"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR."
    echo "Please run ./scripts/setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Starting recursive directory ingestion for $VAULT_DIR..."
PYTHONPATH=. python3 src/tools/ingest_dir.py "$VAULT_DIR"
