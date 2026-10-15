#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at ./venv."
    echo "Please run ./scripts/setup.sh first."
    exit 1
fi

source "venv/bin/activate"

PYTHONPATH=. python3 src/tools/ingest_dir.py "$@"
