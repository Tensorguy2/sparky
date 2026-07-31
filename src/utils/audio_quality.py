"""
Shared TTS audio quality / instability checks.

Used by v2 and v3 servers to decide when to abort a sentence and retry.
Designed to catch intermittent soft failures that pure NaN/clipping miss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_CLIP_THRESHOLD = 0.98
_CLIP_MAX_FRACTION = 0.15
_PEAK_SPIKE = 1.0  # any sample beyond full-scale → runaway / crazy-loud
_DC_BIAS_MAX = 0.25
_RMS_MAX = 0.65
_CLICK_DELTA = 1.2
_FLATLINE_STD = 1e-5
_MIN_SAMPLES_FOR_STATS = 240  # 10 ms at 24 kHz


def is_unstable(chunk: np.ndarray) -> Optional[str]:
    """Return a short reason string if *chunk* looks like bad TTS output."""
    if chunk is None or len(chunk) == 0:
        return None

    x = np.asarray(chunk, dtype=np.float32)
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return "nan_or_inf"

    # Few-sample blasts that the 15% clipping rule would miss.
    peak = float(np.max(np.abs(x)))
    if peak > _PEAK_SPIKE:
        return f"peak_spike_{peak:.2f}"

    clipped_frac = float(np.mean(np.abs(x) > _CLIP_THRESHOLD))
    if clipped_frac > _CLIP_MAX_FRACTION:
        return f"clipping_{clipped_frac:.0%}"

    if len(x) < _MIN_SAMPLES_FOR_STATS:
        return None

    # Hard DC offset / runaway bias (sounds like a thump or mute).
    dc = float(np.abs(np.mean(x)))
    if dc > _DC_BIAS_MAX:
        return f"dc_bias_{dc:.2f}"

    # Overall energy too hot — usually harsh distortion even if not clipped.
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > _RMS_MAX:
        return f"rms_{rms:.2f}"

    # Stuck / silent decoder output.
    std = float(np.std(x))
    if std < _FLATLINE_STD:
        return "flatline"

    # Single-sample clicks / discontinuities.
    if len(x) >= 2:
        max_delta = float(np.max(np.abs(np.diff(x))))
        if max_delta > _CLICK_DELTA:
            return f"click_{max_delta:.2f}"

    return None
