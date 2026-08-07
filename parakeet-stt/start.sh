#!/usr/bin/env bash
# Start the standalone Parakeet STT server (creates venv on first run).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "Creating venv..."
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PORT="${PORT:-8100}"
exec ./venv/bin/python server.py
