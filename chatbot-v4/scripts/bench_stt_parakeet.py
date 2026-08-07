#!/usr/bin/env python3
"""Benchmark Parakeet (onnx-asr) vs faster-whisper on short clips."""

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
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf_cache"))

SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def _write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_sr)
    n = max(1, int(round(duration * dst_sr)))
    x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    y = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(y, x, audio).astype(np.float32)


def main() -> None:
    sample = Path("/tmp/parakeet_sample.wav")
    if not sample.exists():
        urllib.request.urlretrieve(SAMPLE_URL, sample)
    base, sr = _read_wav(sample)
    base16 = _resample(base, sr, 16000)

    import onnx_asr

    print("Loading Parakeet...")
    t0 = time.perf_counter()
    pk = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
    print(f"  load {(time.perf_counter()-t0)*1000:.0f} ms")
    pk.recognize(base16[:8000], sample_rate=16000)

    whisper = None
    try:
        from faster_whisper import WhisperModel

        print("Loading Whisper large-v3-turbo...")
        t0 = time.perf_counter()
        whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        print(f"  load {(time.perf_counter()-t0)*1000:.0f} ms")
        list(whisper.transcribe(np.zeros(8000, dtype=np.float32), beam_size=1, vad_filter=False)[0])
    except Exception as exc:
        print(f"Whisper unavailable: {exc}")

    print("\nSeconds | Parakeet ms | Whisper ms")
    for sec in (0.5, 1.0, 2.0, 4.0, 8.0):
        need = int(16000 * sec)
        audio = np.tile(base16, int(np.ceil(need / len(base16))))[:need]
        path = Path(f"/tmp/bench_pk_{sec}.wav")
        _write_wav(path, audio)

        pk_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            pk.recognize(audio, sample_rate=16000)
            pk_times.append((time.perf_counter() - t0) * 1000)

        wh_times = []
        if whisper is not None:
            for _ in range(5):
                t0 = time.perf_counter()
                list(whisper.transcribe(audio, beam_size=1, language="en", vad_filter=False)[0])
                wh_times.append((time.perf_counter() - t0) * 1000)

        wh = f"{np.median(wh_times):8.1f}" if wh_times else "     n/a"
        print(f"{sec:7.1f} | {np.median(pk_times):11.1f} | {wh}")


if __name__ == "__main__":
    main()
