#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_log_file.jsonl>"
    exit 1
fi

jq -r 'select(.event == "message_added" and .payload.message.role == "assistant" and .payload.message.content != null and .payload.message.content != "") | .payload.message.content' "$1"
