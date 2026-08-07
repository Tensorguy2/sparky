"""
Output-sanity guard for cloned TTS.

The clone intermittently returns a fragment of audio far too short for the text
it was given -- measured on this deployment, 2 of 30 short fragments came back
at 8-21 ms per character where the normal range is 40-120. At 0.16 s for "Here
is the answer." the listener hears an unintelligible blip, which is easily
mistaken for the voice changing.

Two properties make this recoverable:

  * It is not deterministic. The same text regenerated gave 0.24 s, 0.96 s,
    1.20 s and 0.96 s, so a retry usually succeeds.
  * Text with no terminal punctuation fails far more often. "Here is the
    answer" truncated every time; adding the period fixed it.

A pitch-based speaker check was tried first and abandoned: neither median f0 nor
log-mel spectral envelope separates these voices (cross-voice similarity reached
0.993 while within-voice fell to 0.901), and autocorrelation f0 suffers octave
errors that look exactly like a voice change. Duration separates cleanly with a
wide margin, so that is what is gated on.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Measured on this deployment: takes that were audibly broken ran 8-23 ms/char,
# while healthy output never fell below 28. 25 splits the two populations.
#
# It is deliberately at the low end of that gap. A false positive is expensive --
# it burns a retry, and if every attempt trips the check the fragment has to fall
# back to the best take -- whereas a marginally-short take that slips through is
# still intelligible. An earlier value of 30 rejected the terse-but-correct 28
# ms/char renderings of the default voice on every attempt.
MIN_MS_PER_CHAR = 25.0

# Runaway generations also occur -- one 15-word sentence produced 20.4 s of
# audio. Well clear of the 120 ms/char ceiling seen in healthy output.
MAX_MS_PER_CHAR = 400.0

# Below this there is too little text for the ratio to mean anything.
_MIN_CHARS = 8

_ALPHANUM = re.compile(r"[^\w\s]")


def _billable_chars(text: str) -> int:
    """Length excluding punctuation, which costs no speaking time."""
    return len(_ALPHANUM.sub("", text).strip())


def expected_min_seconds(text: str) -> float:
    """Shortest plausible duration for ``text`` before it counts as truncated."""
    return _billable_chars(text) * MIN_MS_PER_CHAR / 1000.0


def check_duration(text: str, audio_seconds: float) -> Optional[str]:
    """Return a reason string if the audio length is implausible for the text.

    Returns None when the output looks healthy or when the text is too short to
    judge -- the guard must not reject audio it cannot assess, since a false
    positive costs a retry and risks dropping a fragment.
    """
    chars = _billable_chars(text)
    if chars < _MIN_CHARS or audio_seconds <= 0:
        return None
    ratio = audio_seconds * 1000.0 / chars
    if ratio < MIN_MS_PER_CHAR:
        return f"truncated_{ratio:.0f}ms_per_char"
    if ratio > MAX_MS_PER_CHAR:
        return f"runaway_{ratio:.0f}ms_per_char"
    return None


def audio_seconds(chunks: list[np.ndarray], sample_rate: int) -> float:
    if not chunks or sample_rate <= 0:
        return 0.0
    return sum(len(c) for c in chunks) / float(sample_rate)


def needs_terminal_punctuation(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] not in ".!?…,;:"
