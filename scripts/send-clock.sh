#!/bin/bash
set -euo pipefail

# go into directory where the config.toml lives
cd "$(dirname "$0")/.."

(echo "clock notification"; ddate; date) | ./scripts/send-notification.py -c data/thunk/config.toml

