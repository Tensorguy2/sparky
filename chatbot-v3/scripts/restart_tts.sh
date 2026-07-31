#!/usr/bin/env bash
# Restart the v3 TTS server (faster-qwen3-tts) on port 25568.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHATBOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sleep 1
fuser -k 25568/tcp 2>/dev/null || true
sleep 2
cd "$ROOT/src"
nohup "$ROOT/venv-tts-v3/bin/python" v3_server.py --port 25568 >>"$CHATBOT_DIR/restart.log" 2>&1 &
