#!/usr/bin/env python3
"""
Does sampling temperature control speaker identity drift in the voice clone?

Callers report the voice changing mid-reply. Each sentence of a reply is a
separate autoregressive generation, and in a codec-token model the speaker is
largely fixed by the tokens sampled at the start of the sequence -- so a high
sampling temperature can land a different speaker on each sentence.

The engine defaults are temperature=0.9, top_k=50, do_sample=True, and the
server has been using them unchanged.

Method: one fixed sentence, generated repeatedly at each temperature. Text is
held constant so spectral differences between takes cannot be phonetic; they are
generation variance. The reference scale is the distance between deliberately
different voices on that same sentence -- drift is only a real problem when it
approaches that.

Also reports duration spread, since driving temperature too low makes the
decoder degenerate (flat prosody, repetition, truncation).

  ../venv-tts-v3/bin/python scripts/sweep_tts_temperature.py
  ../venv-tts-v3/bin/python scripts/sweep_tts_temperature.py --takes 6
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

MODEL_PATH = str(BASE.parent / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
DB_PATH = str(BASE / "voices.db")

SENTENCE = "Yeah, it's Shiva. Sorry about that —"
VOICE = "Shiva6"
OTHER_VOICES = ["Shiva1", "Shiva3", "morgan", "kevin"]


def ltas(audio: np.ndarray, sr: int, nmel: int = 40):
    """Energy-normalised log-mel long-term average spectrum.

    Captures the formant structure that carries speaker identity, and unlike
    autocorrelation pitch it cannot produce octave errors.
    """
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
    norm = np.linalg.norm(b)
    return b / norm if norm > 0 else None


def cos_dist(x, y) -> float:
    return 1.0 - float(np.dot(x, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--temps", default="0.9,0.7,0.5,0.3")
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"  loading engine ...", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        MODEL_PATH, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    import sqlite3
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = {r["voice_id"]: r for r in conn.execute("SELECT * FROM voices").fetchall()}
    conn.close()

    def prompt_for(vid: str):
        r = rows[vid]
        return model.model.create_voice_clone_prompt(
            ref_audio=r["ref_audio_path"], ref_text=r["ref_text"],
        )

    def gen(vid_prompt, temperature: float, top_p: float):
        chunks = []
        sr = 24000
        for chunk, s, _t in model.generate_voice_clone_streaming(
            text=SENTENCE, language="English", voice_clone_prompt=vid_prompt,
            temperature=temperature, top_p=top_p, chunk_size=8,
            max_new_tokens=min(600, max(150, len(SENTENCE) * 3)),
        ):
            chunks.append(chunk)
            sr = s
        return (np.concatenate(chunks) if chunks else None), sr

    print(f"  computing clone prompts ...", flush=True)
    p_main = prompt_for(VOICE)
    p_others = {v: prompt_for(v) for v in OTHER_VOICES if v in rows}

    # Reference scale: how far apart genuinely different speakers sit on this
    # exact sentence, at the engine default temperature.
    other_fps = []
    for v, p in p_others.items():
        a, sr = gen(p, 0.9, args.top_p)
        f = ltas(a, sr) if a is not None else None
        if f is not None:
            other_fps.append((v, f))

    main_ref, sr0 = gen(p_main, 0.9, args.top_p)
    main_ref_fp = ltas(main_ref, sr0)
    cross = [cos_dist(main_ref_fp, f) for _, f in other_fps] if main_ref_fp is not None else []
    cross_med = float(np.median(cross)) if cross else float("nan")
    print(f"\n  reference scale -- {VOICE} vs a different voice, same sentence:")
    print(f"    median cosine distance {cross_med:.4f} "
          f"(min {min(cross):.4f}, max {max(cross):.4f})\n")

    print(f"  {'temp':>5}  {'drift p50':>10}{'drift max':>11}"
          f"{'% of a real voice change':>26}{'dur spread':>12}")
    results = {}
    for temp in [float(t) for t in args.temps.split(",")]:
        fps, durs = [], []
        for _ in range(args.takes):
            a, sr = gen(p_main, temp, args.top_p)
            if a is None:
                continue
            durs.append(len(a) / sr)
            f = ltas(a, sr)
            if f is not None:
                fps.append(f)
        if len(fps) < 2:
            print(f"  {temp:5.2f}  insufficient usable takes")
            continue
        d = [cos_dist(x, y) for x, y in itertools.combinations(fps, 2)]
        pct = np.median(d) / cross_med * 100 if cross_med == cross_med else float("nan")
        results[temp] = (float(np.median(d)), float(max(d)), pct)
        print(f"  {temp:5.2f}  {np.median(d):10.4f}{max(d):11.4f}{pct:25.0f}%"
              f"{min(durs):6.2f}-{max(durs):.2f}s")

    if results:
        best = min(results.items(), key=lambda kv: kv[1][0])
        print(f"\n  lowest drift at temperature {best[0]:.2f} "
              f"(p50 {best[1][0]:.4f}, {best[1][2]:.0f}% of a real voice change)")
        print(f"  engine default is 0.90 -> "
              f"{results.get(0.9, (float('nan'),))[0]:.4f}")


if __name__ == "__main__":
    main()
