#!/usr/bin/env python3
"""
Can a sentence be locked to the speaker the previous sentence actually rendered?

A reply is spoken as several independent generations, one per sentence, each
conditioned only on the voice's original reference recording. Nothing carries the
speaker that was actually produced, so every sentence is free to land somewhere
else -- which is what callers hear as the voice changing mid-reply.

The engine can build a clone prompt from any audio plus its transcript. So the
first sentence can be generated normally, and the rest conditioned on the first
sentence's own output, pinning them to whatever speaker it settled on.

Method: generate a sentence, then regenerate the same text twice --
  A  conditioned on the original reference (what the server does now)
  B  conditioned on the first take's audio (the proposed fix)
and measure how far each lands from the first take. Text is identical
throughout, so the difference is not phonetic.

Also times prompt construction, since it would sit on the critical path.

  ../venv-tts-v3/bin/python scripts/test_speaker_carryover.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

MODEL_PATH = str(BASE.parent / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
DB_PATH = str(BASE / "voices.db")

SENTENCE = "Yeah, it's Shiva. Sorry about that —"
VOICE = "Shiva6"


def ltas(audio: np.ndarray, sr: int, nmel: int = 40):
    a = np.asarray(audio, dtype=np.float32)
    a = a[np.abs(a) > 0.01]
    if len(a) < int(0.2 * sr):
        return None
    win, hop = 1024, 512
    w = np.hanning(win)
    acc = np.zeros(win // 2 + 1)
    n = 0
    for i in range(0, len(a) - win, hop):
        acc += np.abs(np.fft.rfft(a[i:i + win] * w))
        n += 1
    if n == 0:
        return None
    spec = acc / n
    freqs = np.fft.rfftfreq(win, 1 / sr)
    to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    from_mel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = from_mel(np.linspace(to_mel(80), to_mel(8000), nmel + 1))
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        bands.append(spec[m].mean() if m.any() else 0.0)
    b = np.log(np.array(bands) + 1e-8)
    b -= b.mean()
    n2 = np.linalg.norm(b)
    return b / n2 if n2 > 0 else None


def cos_dist(x, y) -> float:
    return 1.0 - float(np.dot(x, y))


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    import sqlite3

    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print("  loading engine ...", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        MODEL_PATH, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM voices WHERE voice_id=?", (VOICE,)).fetchone()
    conn.close()

    t0 = time.perf_counter()
    ref_prompt = model.model.create_voice_clone_prompt(
        ref_audio=row["ref_audio_path"], ref_text=row["ref_text"],
    )
    print(f"  reference prompt built in {(time.perf_counter()-t0)*1000:.0f} ms "
          f"(from {row['ref_audio_path'].split('/')[-1]})")

    def gen(prompt):
        chunks, sr = [], 24000
        for chunk, s, _t in model.generate_voice_clone_streaming(
            text=SENTENCE, language="English", voice_clone_prompt=prompt,
            chunk_size=8, max_new_tokens=min(600, max(150, len(SENTENCE) * 3)),
        ):
            chunks.append(chunk)
            sr = s
        return (np.concatenate(chunks) if chunks else None), sr

    a_dists, b_dists, build_ms = [], [], []
    tmpdir = tempfile.mkdtemp(prefix="carryover-")

    for trial in range(args.trials):
        first, sr = gen(ref_prompt)
        if first is None:
            continue
        fp_first = ltas(first, sr)
        if fp_first is None:
            continue

        # A: what happens today -- conditioned on the original reference only.
        a, sr_a = gen(ref_prompt)
        fp_a = ltas(a, sr_a) if a is not None else None
        if fp_a is not None:
            a_dists.append(cos_dist(fp_first, fp_a))

        # B: conditioned on the audio the first take actually produced.
        wav = str(Path(tmpdir) / f"first_{trial}.wav")
        write_wav(wav, first, sr)
        t1 = time.perf_counter()
        carry_prompt = model.model.create_voice_clone_prompt(
            ref_audio=wav, ref_text=SENTENCE,
        )
        build_ms.append((time.perf_counter() - t1) * 1000)
        b, sr_b = gen(carry_prompt)
        fp_b = ltas(b, sr_b) if b is not None else None
        if fp_b is not None:
            b_dists.append(cos_dist(fp_first, fp_b))

        print(f"    trial {trial+1}: reference-conditioned {a_dists[-1]:.4f} | "
              f"carry-over {b_dists[-1]:.4f}", flush=True)

    if not a_dists or not b_dists:
        print("  not enough usable takes")
        return

    print(f"\n  distance from the first take (lower = same speaker):")
    print(f"    A  conditioned on original reference : p50 {np.median(a_dists):.4f} "
          f"max {max(a_dists):.4f}")
    print(f"    B  conditioned on first take's audio : p50 {np.median(b_dists):.4f} "
          f"max {max(b_dists):.4f}")
    improve = (1 - np.median(b_dists) / np.median(a_dists)) * 100
    print(f"    carry-over reduces drift by {improve:.0f}%")
    print(f"\n  prompt construction cost: p50 {np.median(build_ms):.0f} ms "
          f"(would sit on the critical path for sentence 2 onward)")


if __name__ == "__main__":
    main()
