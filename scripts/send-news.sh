#!/bin/bash
set -euo pipefail

# go into directory where the config.toml lives
cd "$(dirname "$0")/.."

(echo "daily news report"; cat reports/news-daily-thunk.txt | ./scripts/ask-oracle.py -c data/thunk/config.toml ) | ./scripts/send-notification.py -c data/thunk/config.toml

(echo "daily news report"; cat reports/news-daily-wisp.txt | ./scripts/ask-oracle.py -c data/thunk/config.toml --model gemini-4-argon) | ./scripts/send-notification.py -c data/wisp/config.toml


