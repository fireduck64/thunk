#!/bin/bash

# go into directory where the config.toml lives
cd "$(dirname $0)"
cd ..

(echo "clock notification"; ddate; date) | ./scripts/send-notification.py

