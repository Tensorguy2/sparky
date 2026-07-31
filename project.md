# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qwen3-TTS server project with two main components:

1. **`src/server.py`** — A FastAPI/WebSocket server for low-latency, sentence-streamed TTS with voice profile management.
2. **`qwen_tts_package/`** — A local copy of the `qwen-tts` Python package (the Alibaba Qwen Team's library), installed in editable mode. Do not edit this unless intentionally patching upstream code.
3. **`playground/`** — Experimental scripts (not production).

## Running the Server

```bash
# Activate the local venv first
source env/Scripts/activate   # Windows/bash

# Start the TTS server (port 25565) — must run from inside src/
cd src
python server.py
```

The server loads `Qwen/Qwen3-TTS-12Hz-1.7B-Base` from HuggingFace on startup, pre-computes VoiceClonePromptItems for **all** registered voices and keeps them in RAM, then runs a CUDA warmup inference. Startup takes ~30–60 seconds on first run.

## Installing / Reinstalling the Package

The `qwen_tts` package is installed from the local copy:

```bash
pip install -e qwen_tts_package/
```

## Quick Inference (no server)

```bash
cd playground
python test.py        # basic voice clone to output_voice_clone.wav
python test2.py       # alternative test
```

## Architecture

### Module layout (`src/`)

```
src/
├── server.py              # FastAPI app, lifespan, /health
├── config.py              # All constants (model, device, paths, built-in voices)
├── db/database.py         # SQLite CRUD — only metadata (paths, ref_text); no tensors
├── services/
│   ├── model_service.py   # Qwen3TTSModel + single-worker ThreadPoolExecutor singleton
│   ├── voice_service.py   # In-RAM dict[voice_id → VoiceEntry] + DB bridge
│   └── qwen_streamer.py   # Custom QwenAudioStreamer yielding decoded audio chunks
├── routers/
│   ├── tts.py             # WS /ws/tts — real-time sub-sentence chunk streaming
│   └── voices.py          # REST /voices — CRUD
├── utils/text.py          # split_into_sentences()
├── assets/                # Built-in reference WAVs (e.g. mikey_sample.wav)
└── voices/                # User-uploaded WAVs (auto-created, one file per voice_id)
```

### Startup sequence (`server.py` lifespan)

1. `db.init()` — creates `voices.db` if absent
2. `model_service.load()` — loads model onto GPU, creates single-worker `ThreadPoolExecutor`
3. Seed built-in voices into DB (idempotent `INSERT`)
4. `voice_service.load_all()` — reads every DB record, calls `compute_prompt_sync` via executor for each, stores `VoiceClonePromptItem` in RAM dict
5. CUDA warmup — one dummy `generate_voice_clone` to compile kernels

### Voice service design

- `VoiceService._voices: Dict[str, VoiceEntry]` — all prompts live in system RAM permanently (100 GB RAM, no eviction needed). Tensors are explicitly pushed to `.cpu()` after compilation.
- Thread-scoped cloned prompts are shifted dynamically `.to(device)` during inference to prevent overlapping race conditions and prevent VRAM bloating.
- SQLite stores only metadata; used to rebuild RAM cache on restart
- `asyncio.Lock` guards concurrent `POST /voices` registrations
- Built-in voices (defined in `config.BUILTIN_VOICES`) are seeded at startup and cannot be deleted

### Inference path (WS /ws/tts)

`split_into_sentences()` → per-sentence `infer_stream(...)` → `QwenAudioStreamer::get_async()`. 
As `qwen_streamer.py` natively accumulates internal `codec_ids` frame-by-frame from the underlying model, it offsets the newly generated audio slices seamlessly. The websocket consumer loop instantly streams these `chunk_header` payloads back-to-back alongside their raw `pcm_f32le` bytes, drastically optimizing Time-To-First-Byte (TTFB) latency to sub-second real-time bounds.

### qwen_tts Package (`qwen_tts_package/qwen_tts/`)

- **`core/models/modeling_qwen3_tts.py`** — Contains the core model wrapper `Qwen3TTSForConditionalGeneration`. Patched to forward Huggingface `streamer` argument parameters deep into the `Talker` layer to continually pipe isolated `codec_ids` outside the internal pipeline during the recursive generation (`self._streamer.put()`).
- **`inference/qwen3_tts_model.py`** — `Qwen3TTSModel` wrapper. Key methods:
  - `from_pretrained(model_id, device_map, dtype, attn_implementation)`
  - `create_voice_clone_prompt(ref_audio, ref_text)` → `List[VoiceClonePromptItem]` — expensive, must be cached
  - `generate_voice_clone(text, language, voice_clone_prompt)` → `(List[np.ndarray], int)`
- **`core/tokenizer_12hz/`** and **`core/tokenizer_25hz/`** — Two tokenizer variants. 12Hz is the default for the 1.7B-Base model.

### Realtime Voice Changer (`playground/realtime_voice_changer.py`)

Standalone pipeline: mic → WebRTC VAD → faster-whisper STT → Qwen3-TTS → playback. Four daemon threads via queues. Contains hardcoded Linux paths — update `REF_AUDIO_PATH` before running.

## Key Configuration (`src/config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model |
| `DEVICE` | `cuda:0` | GPU target |
| `DTYPE` | `torch.bfloat16` | Inference precision |
| `BUILTIN_VOICES` | `{"mikey": {...}}` | Always-loaded voices |
| `PORT` | `25565` | Server port |

## WebSocket Protocol Summary

```
Client → {"text": "...", "language": "English", "voice_id": "mikey"}
Server → {"type": "metadata", "sample_rate": 24000, "channels": 1, "encoding": "pcm_f32le", "num_sentences": N}
Server → {"type": "chunk_header", "index": 0, "num_samples": X, "size_bytes": Y}
Server → <raw bytes: float32 LE PCM>
... (repeated per sentence)
Server → {"type": "done"}
```

Audio output is always mono float32 little-endian PCM at 24000 Hz (server notifies if actual SR differs).
