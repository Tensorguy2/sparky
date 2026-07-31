#!/usr/bin/env bash
# Start the v3 voice chatbot with CUDA CTranslate2 on the library path.
# Shares the original chatbot's venv (dependencies are identical).
set -euo pipefail
cd "$(dirname "$0")/.."
source ../chatbot/venv/bin/activate

CT2_LIB="${HOME}/.local/ctranslate2/lib"
CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
export LD_LIBRARY_PATH="${CT2_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"

# Parakeet onnx-asr downloads / reads weights from HF_HOME
export HF_HOME="${HF_HOME:-$(cd .. && pwd)/.hf_cache}"
mkdir -p "$HF_HOME"

exec python server.py "$@"
