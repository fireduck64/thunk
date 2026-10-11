#!/bin/bash

# go into directory where the config.toml lives
cd "$(dirname $0)"
cd ..

(echo "daily news report"; cat reports/news-daily-thunk.txt | ./scripts/ask-oracle.py ) | ./scripts/send-notification.py

