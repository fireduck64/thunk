#!/bin/bash
set -euo pipefail

VENV_DIR="${THUNK_VENV_DIR:-/var/nvme/thunk_venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR."
    echo "Please run ./scripts/setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# We will run the main entrypoint here eventually
echo "Starting Thunk Agent..."
# python -m src.main
