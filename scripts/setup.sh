#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# If THUNK_VENV_DIR is set, create the venv externally and symlink ./venv to it.
# Otherwise, create ./venv directly in the project root.
if [ -n "${THUNK_VENV_DIR:-}" ]; then
    echo "Setting up external venv at ${THUNK_VENV_DIR}..."
    if [ ! -d "$THUNK_VENV_DIR" ]; then
        python3 -m venv "$THUNK_VENV_DIR"
    fi
    ln -sfn "$THUNK_VENV_DIR" venv
    echo "Linked ./venv -> ${THUNK_VENV_DIR}"
else
    echo "Setting up local venv at ./venv..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
fi

# Activate and install requirements
source "venv/bin/activate"

touch requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Virtual environment is ready at ./venv"
