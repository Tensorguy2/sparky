#!/usr/bin/env python3
"""
End-to-end voice-pipeline latency benchmark.

Measures the three stages that sit between a user finishing a sentence and
hearing a reply, then adds them into a turn budget:

  STT  transcription of the captured utterance (local Parakeet, GPU)
  LLM  time-to-first-token from the chat model
  TTS  time-to-first-audio for the first clause of the reply

TTS is measured at clause length, not sentence length, because client-side
streaming ships the first few words as soon as they arrive -- that fragment is
what sets perceived latency.

  python scripts/bench_pipeline.py                 # everything
  python scripts/bench_pipeline.py --skip-stt      # servers only
  python scripts/bench_pipeline.py --models gpt-4.1-mini,claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
TTS_URL = "ws://localhost:25568/ws/tts"

# Roughly what the client emits: the first clause, then a full sentence.
TTS_CASES = [
    ("first clause", "Sure, let me check that."),
    ("full sentence", "Sure, let me check that for you and see what options are available."),
]

STT_DURATIONS = (2.0, 4.0, 8.0)


def _stats(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "no data"
    s = sorted(xs)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return f"p50 {statistics.median(s):7.1f} ms | p95 {p95:7.1f} ms | min {min(s):7.1f} ms"


# ---------------------------------------------------------------- STT

def _load_sample() -> tuple["np.ndarray", int]:
    import numpy as np

    path = Path("/tmp/parakeet_sample.wav")
    if not path.exists():
        urllib.request.urlretrieve(SAMPLE_URL, path)
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        a = a.astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def bench_stt(trials: int) -> dict:
    import numpy as np
    from services import parakeet_stt as pk

    audio, sr = _load_sample()
    if sr != 16000:
        n = int(len(audio) / sr * 16000)
        audio = np.interp(
            np.linspace(0, 1, n, endpoint=False),
            np.linspace(0, 1, len(audio), endpoint=False),
            audio,
        ).astype(np.float32)

    def as_pcm16(a: "np.ndarray") -> bytes:
        return (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()

    # transcribe_sync takes PCM16 chunks at the capture rate, matching how the
    # server feeds it from the websocket.
    pk.transcribe_sync([as_pcm16(audio[:16000])], sample_rate=16000)  # warm kernels

    out = {}
    for secs in STT_DURATIONS:
        need = int(secs * 16000)
        clip = np.tile(audio, int(np.ceil(need / len(audio))))[:need]
        pcm = as_pcm16(clip)
        xs = []
        for _ in range(trials):
            t0 = time.perf_counter()
            pk.transcribe_sync([pcm], sample_rate=16000)
            xs.append((time.perf_counter() - t0) * 1000)
        out[secs] = xs
        rtf = statistics.median(xs) / (secs * 1000)
        print(f"    {secs:>4.0f}s utterance : {_stats(xs)} | RTF {rtf:.3f}")
    return out


# ---------------------------------------------------------------- TTS

async def _tts_ttfa(text: str) -> float | None:
    import websockets

    t0 = time.perf_counter()
    async with websockets.connect(TTS_URL) as ws:
        await ws.send(json.dumps({"text": text, "voice_id": "mikey", "language": "English"}))
        async for m in ws:
            if isinstance(m, (bytes, bytearray)):
                return (time.perf_counter() - t0) * 1000
            if json.loads(m).get("type") == "error":
                return None
    return None


async def bench_tts(trials: int, settle: float) -> dict:
    """Two things will silently corrupt these numbers if ignored.

    The server caches finished audio per (sentence, voice, language), so
    repeating one string times a dict lookup rather than synthesis -- hence
    unique wording per trial.

    It also keeps generating the rest of the sentence for 1.5-3 s after the
    first chunk is on the wire. A request issued before that finishes queues
    behind it and reports ~2.8x the real figure, so trials are spaced by
    ``settle``. The queued case is measured separately because client-side
    streaming does issue fragments back-to-back within a turn.
    """
    out = {}
    for label, text in TTS_CASES:
        xs = []
        for i in range(trials):
            await asyncio.sleep(settle)
            xs.append(await _tts_ttfa(f"{text} Variant number {i} alpha."))
        out[label] = xs
        print(f"    {label:<16} (idle server) : {_stats(xs)}")

    xs = []
    for i in range(trials):
        xs.append(await _tts_ttfa(f"Another reply fragment, item {i} omega epsilon."))
    out["back-to-back"] = xs
    print(f"    {'back-to-back':<16} (queued)      : {_stats(xs)}")

    warm = "Sure, let me check that."
    await _tts_ttfa(warm)
    await asyncio.sleep(settle)
    hits = [await _tts_ttfa(warm) for _ in range(trials)]
    out["cache hit"] = hits
    print(f"    {'repeated phrase':<16} (cached)      : {_stats(hits)}")
    return out


# ---------------------------------------------------------------- LLM

async def bench_llm(models: list[str], trials: int, gap: float) -> dict:
    import services.llm_service as L
    from models.instructions import ModelParams

    out = {}
    for model in models:
        await L.prewarm(model)
        xs = []
        for _ in range(trials):
            t0 = time.perf_counter()
            ttft = None
            try:
                async for _tok in L.stream_chat(
                    model,
                    "You are a friendly voice assistant. Reply in one short sentence.",
                    [{"role": "user", "content": "What time do you open on Saturdays?"}],
                    ModelParams(temperature=0.7, max_tokens=40),
                ):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f"    {model:<22}: FAILED ({type(e).__name__})")
                xs = []
                break
            xs.append(ttft)
            # A real turn has a pause here; without it the pool stays hot and
            # the numbers flatter the connection reuse more than reality would.
            await asyncio.sleep(gap)
        if xs:
            out[model] = xs
            print(f"    {model:<22}: {_stats(xs)}")
    return out


# ---------------------------------------------------------------- main

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--llm-trials", type=int, default=5)
    ap.add_argument("--gap", type=float, default=6.0, help="pause between LLM turns")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="pause between TTS trials so the server is idle")
    ap.add_argument("--models", default="gpt-4.1-mini,claude-haiku-4-5")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()

    stt = tts = llm = {}

    if not args.skip_stt:
        print("\n  STT  (local Parakeet TDT 0.6b v2, GPU)")
        try:
            stt = bench_stt(args.trials)
        except Exception as e:
            print(f"    unavailable: {type(e).__name__}: {e}")

    if not args.skip_llm:
        print("\n  LLM  (time to first token)")
        llm = await bench_llm([m.strip() for m in args.models.split(",") if m.strip()],
                              args.llm_trials, args.gap)

    if not args.skip_tts:
        print("\n  TTS  (time to first audio)")
        try:
            tts = await bench_tts(args.trials, args.settle)
        except Exception as e:
            print(f"    unavailable: {type(e).__name__}: {e}")

    # Turn budget: a 4 s utterance, then the model, then the first clause.
    stt_p50 = statistics.median(stt[4.0]) if stt.get(4.0) else None
    tts_p50 = statistics.median(tts["first clause"]) if tts.get("first clause") else None
    if stt_p50 is not None and tts_p50 is not None and llm:
        print("\n  Turn budget  (4 s utterance -> first audible word)")
        for model, xs in llm.items():
            l = statistics.median(xs)
            total = stt_p50 + l + tts_p50
            print(f"    {model:<22}: STT {stt_p50:5.0f} + LLM {l:6.0f} + TTS {tts_p50:6.0f} "
                  f"= {total:7.0f} ms")
        print("\n  Excludes VAD endpoint detection (~200 ms) and network to the browser.")


if __name__ == "__main__":
    asyncio.run(main())
