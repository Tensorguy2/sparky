#!/usr/bin/env python3
"""Benchmark Parakeet (onnx-asr) vs faster-whisper on short clips.

Usage: python scripts/bench_stt.py [path/to/clip.wav]
Without an argument it downloads a public LibriSpeech sample.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import stt_service  # noqa: E402

SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def main() -> None:
    if len(sys.argv) > 1:
        sample = Path(sys.argv[1])
    else:
        sample = Path("/tmp/parakeet_sample.wav")
        if not sample.exists():
            print(f"Downloading sample clip to {sample} ...")
            urllib.request.urlretrieve(SAMPLE_URL, sample)
    base, sr = _read_wav(sample)
    base16 = stt_service.resample(base, sr)

    print("Loading + warming Parakeet...")
    t0 = time.perf_counter()
    stt_service.preload()
    print(f"  ready in {(time.perf_counter()-t0)*1000:.0f} ms")

    have_whisper = stt_service.whisper_available()
    if have_whisper:
        print("Loading Whisper...")
        stt_service.transcribe_sync(
            [np.zeros(8000, dtype=np.int16).tobytes()], sample_rate=16000, model="whisper"
        )
    else:
        print("faster-whisper not installed; benchmarking Parakeet only.")

    print("\nSeconds | Parakeet ms | Whisper ms | Parakeet text")
    for sec in (0.5, 1.0, 2.0, 4.0, 8.0):
        need = int(16000 * sec)
        audio = np.tile(base16, int(np.ceil(need / len(base16))))[:need]
        pcm = [(np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()]

        pk_times, text = [], ""
        for _ in range(5):
            r = stt_service.transcribe_sync(pcm, sample_rate=16000, model="parakeet")
            pk_times.append(r["stt_ms"])
            text = r["text"]

        wh_times = []
        if have_whisper:
            for _ in range(5):
                r = stt_service.transcribe_sync(pcm, sample_rate=16000, model="whisper")
                wh_times.append(r["stt_ms"])

        wh = f"{np.median(wh_times):8.1f}" if wh_times else "     n/a"
        print(f"{sec:7.1f} | {np.median(pk_times):11.1f} | {wh} | {text[:60]!r}")


if __name__ == "__main__":
    main()
