#!/usr/bin/env bash
# Restart the v3 voice chatbot (free port 8020, then start again).
set -euo pipefail
CHATBOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Let the POST /api/admin/restart response reach the browser first.
sleep 2
fuser -k 8020/tcp 2>/dev/null || true
sleep 1
nohup bash "$CHATBOT_DIR/scripts/start_server.sh" >>"$CHATBOT_DIR/restart.log" 2>&1 &
