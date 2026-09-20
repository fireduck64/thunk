#!/bin/bash
set -euo pipefail

# Allow user to override venv location via environment variable
VENV_DIR="${THUNK_VENV_DIR:-/var/nvme/thunk_venv}"

echo "Setting up venv at ${VENV_DIR}..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate and install requirements
source "$VENV_DIR/bin/activate"

# We will create requirements.txt later, but touch it now so pip install doesn't fail
touch requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Virtual environment is ready at $VENV_DIR"
