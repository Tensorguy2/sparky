# Parakeet STT — standalone

Self-contained speech-to-text server built around **NVIDIA Parakeet TDT 0.6B v2**
running through [onnx-asr](https://github.com/istupakov/onnx-asr) on CPU ONNX
Runtime, so it uses zero GPU VRAM and won't contend with TTS/LLM workloads.
Optional fallback: faster-whisper `large-v3-turbo` (greedy, `beam_size=1`).

Independent of chatbot-v3 — no shared code or config.

## Quick start

```bash
./start.sh            # creates ./venv, installs deps, serves on :8100
```

Open http://localhost:8100 — click **Start microphone**. A simple energy VAD
(onset 96 ms, endpoint 340 ms) segments your speech; each utterance is
transcribed and shown with `stt_ms` (model compute time) and end-to-end
latency (speech start → transcript, includes the VAD endpoint wait).

The first run downloads the ONNX model (~2.4 GB) into `./.hf_cache`. To reuse
an existing cache instead:

```bash
HF_HOME=/path/to/.hf_cache ./start.sh
```

## API

- `GET /api/status` — model/backend status
- `POST /api/transcribe` — multipart WAV (16-bit PCM) upload:

```bash
curl -F file=@clip.wav localhost:8100/api/transcribe
# {"text": "...", "stt_ms": 85.2, "audio_s": 3.1, "model": "parakeet"}
```

- `WS /ws` — send `{"type":"start","sampleRate":16000,"model":"parakeet"}`,
  then binary PCM16 mono frames, then `{"type":"stop"}`; receive
  `{"type":"transcript","text":...,"stt_ms":...}`.

## Whisper fallback

```bash
./venv/bin/pip install faster-whisper
```

Then pick "whisper large-v3-turbo" in the UI or pass `model=whisper` to the
API. Loads lazily on first use (CUDA fp16, falls back to CPU int8).

## Benchmark

```bash
./venv/bin/python scripts/bench_stt.py [optional_clip.wav]
```

Reports median wall ms per model on 0.5–8 s clips (5 runs each).

## Env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `PORT` | `8100` | server port |
| `HF_HOME` | `./.hf_cache` | model cache |
| `PARAKEET_ONNX_MODEL` | `nemo-parakeet-tdt-0.6b-v2` | onnx-asr model id |
| `WHISPER_MODEL` | `large-v3-turbo` | fallback model |
