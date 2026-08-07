#!/usr/bin/env bash
# Serve a local LLM on the GB10 via vLLM's OpenAI-compatible API.
#
#   scripts/start_local_llm.sh nemotron-3-nano-30b-a3b
#   scripts/start_local_llm.sh qwen3.6-35b-a3b
#
# Ports match services/local_llm.py, so both models can be resident at once.
# First launch downloads weights (~30 GB) into HF_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "usage: $0 <nemotron-3-nano-30b-a3b|qwen3.6-35b-a3b>" >&2
  exit 1
fi

case "$MODEL" in
  nemotron-3-nano-30b-a3b)
    REPO="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    PORT=8101
    EXTRA_ENV=(-e VLLM_USE_FLASHINFER_MOE_FP8=1)
    EXTRA_ARGS=(--tool-call-parser qwen3_coder --enable-auto-tool-choice)
    ;;
  qwen3.6-35b-a3b)
    REPO="Qwen/Qwen3.6-35B-A3B-FP8"
    PORT=8100
    EXTRA_ENV=()
    EXTRA_ARGS=(--tool-call-parser hermes --enable-auto-tool-choice)
    ;;
  *)
    echo "unknown model: $MODEL" >&2
    exit 1
    ;;
esac

HF_CACHE="${HF_HOME:-$(cd .. && pwd)/.hf_cache}"
mkdir -p "$HF_CACHE"

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:cu130-nightly}"

# Only allocate a TTY when there is one, so the script also works when
# launched detached or from a log-redirected background job.
TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)

# 32k is well past what a voice turn needs; the savings go to KV cache
# headroom so both models plus the TTS and STT sessions fit in the 128 GB
# unified pool. Prefix caching is what keeps time-to-first-token low, since
# the system prompt is large and identical on every turn.
exec docker run --rm "${TTY_FLAGS[@]}" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/hf" \
  -e HF_HOME=/hf \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "${EXTRA_ENV[@]}" \
  --name "vllm-${MODEL}" \
  "$IMAGE" \
  --model "$REPO" \
  --served-model-name "$MODEL" \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.35 \
  --enable-prefix-caching \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"
