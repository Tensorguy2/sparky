#!/usr/bin/env python3
"""
Analyze VAD latency metrics exported from the chatbot client.

Usage:
  1. In the browser console after 30+ turns, run:
       copy(JSON.stringify(window.__vadMetrics))
  2. Paste into a file (e.g. metrics.json)
  3. Run: python3 scripts/analyze_vad_metrics.py metrics.json

Or pipe directly:
  echo '<json>' | python3 scripts/analyze_vad_metrics.py -
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def analyze(metrics: list[dict]) -> None:
    total = len(metrics)
    if total == 0:
        print("No metrics to analyze.")
        return

    valid = [m for m in metrics if not m.get("aborted") and not m.get("empty")]
    empty = [m for m in metrics if m.get("empty") and not m.get("aborted")]
    aborted = [m for m in metrics if m.get("aborted")]

    print(f"{'='*60}")
    print(f" VAD Latency Report — {total} turns")
    print(f"{'='*60}")
    print(f"  Valid transcripts:  {len(valid)} ({100*len(valid)/total:.1f}%)")
    print(f"  Empty transcripts:  {len(empty)} ({100*len(empty)/total:.1f}%)")
    print(f"  Aborted turns:      {len(aborted)} ({100*len(aborted)/total:.1f}%)")
    print()

    if not valid:
        print("  No valid turns to compute latency stats.")
        return

    ep = np.array([m["endpoint_ms"] for m in valid])
    stt = np.array([m["stt_ms"] for m in valid])
    tot = np.array([m["total_ms"] for m in valid])

    print(f"  {'Metric':<20} {'p50':>8} {'p95':>8} {'mean':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [("Endpoint (VAD)", ep), ("STT (server)", stt), ("Total", tot)]:
        print(
            f"  {label:<20} {np.percentile(arr,50):>7.0f}ms"
            f" {np.percentile(arr,95):>7.0f}ms"
            f" {np.mean(arr):>7.0f}ms"
            f" {np.min(arr):>7.0f}ms"
            f" {np.max(arr):>7.0f}ms"
        )

    print()
    preroll_count = sum(1 for m in valid if m.get("had_preroll"))
    print(f"  Pre-roll used: {preroll_count}/{len(valid)} turns ({100*preroll_count/len(valid):.0f}%)")
    print()

    # Acceptance criteria from plan
    print("  Acceptance criteria:")
    median_improvement = 529 - np.percentile(tot, 50)
    empty_rate = 100 * len(empty) / total
    print(f"    Target: ≥150 ms median improvement from 529 ms baseline")
    print(f"    Actual: {median_improvement:.0f} ms improvement (p50 total = {np.percentile(tot,50):.0f} ms)")
    print(f"    {'✓' if median_improvement >= 150 else '✗'} {'PASS' if median_improvement >= 150 else 'FAIL'}")
    print()
    print(f"    Target: <5% empty/false endpoints")
    print(f"    Actual: {empty_rate:.1f}%")
    print(f"    {'✓' if empty_rate < 5 else '✗'} {'PASS' if empty_rate < 5 else 'FAIL'}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    if src == "-":
        data = json.loads(sys.stdin.read())
    else:
        data = json.loads(Path(src).read_text())

    analyze(data)


if __name__ == "__main__":
    main()
