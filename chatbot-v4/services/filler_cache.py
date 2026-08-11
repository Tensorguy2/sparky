"""Short backchannel fillers — only Aha / Ok, for dead-air masking.

Clips are cached per voice_id. Chat turns use delayed fillers
(``FILLER_DELAY_MS``) so these play only when reply TTS is late.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set

import config
from services.tts_client_v2 import TTSInstability, stream_tts_v2

logger = logging.getLogger(__name__)

# Keep it simple — dead-air acknowledgments only.
FILLER_PHRASES = [
    "Aha.",
    "Ok.",
]

_QUESTION_STARTERS = re.compile(
    r"^(who|what|when|where|why|how|is|are|do|does|did|can|could|would|will|"
    r"should|have|has|am|was|were|may|might)\b",
    re.IGNORECASE,
)


def is_direct_question(text: str) -> bool:
    """True when the utterance is a direct question — skip fillers for these."""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first = re.split(r"[.!]\s+", t, maxsplit=1)[0].strip()
    return bool(_QUESTION_STARTERS.match(first))


@dataclass
class FillerClip:
    phrase: str
    pcm_bytes: bytes
    voice_id: str
    sample_rate: int = 24000

    @property
    def num_samples(self) -> int:
        return len(self.pcm_bytes) // 2


_cache: Dict[str, List[FillerClip]] = {}
_warming: Set[str] = set()
_warm_lock = asyncio.Lock()
_recent: Dict[str, Deque[str]] = {}


async def warm_cache(voice_id: str = "", language: str = "English") -> None:
    """Generate filler clips for a voice (no-op if fully cached or in progress)."""
    voice = voice_id or config.DEFAULT_VOICE_ID
    async with _warm_lock:
        existing = _cache.get(voice) or []
        have = {c.phrase for c in existing}
        missing = [p for p in FILLER_PHRASES if p not in have]
        if not missing or voice in _warming:
            return
        _warming.add(voice)

    logger.info("Warming filler cache (%d phrases, voice=%s)...", len(missing), voice)
    new_clips: List[FillerClip] = []
    try:
        for phrase in missing:
            try:
                pcm_parts: List[bytes] = []
                async for event in stream_tts_v2(
                    phrase, voice, language, count_toward_breaker=False
                ):
                    if isinstance(event, TTSInstability):
                        continue
                    pcm_parts.append(event.pcm_bytes)
                if pcm_parts:
                    new_clips.append(
                        FillerClip(
                            phrase=phrase,
                            pcm_bytes=b"".join(pcm_parts),
                            voice_id=voice,
                        )
                    )
            except Exception:
                logger.warning(
                    "Failed to generate filler %r (voice=%s)",
                    phrase,
                    voice,
                    exc_info=True,
                )
    finally:
        async with _warm_lock:
            merged = list(_cache.get(voice) or [])
            seen = {c.phrase for c in merged}
            for clip in new_clips:
                if clip.phrase not in seen:
                    merged.append(clip)
                    seen.add(clip.phrase)
            # Drop legacy phrases if any remain from a prior process.
            merged = [c for c in merged if c.phrase in FILLER_PHRASES]
            _cache[voice] = merged
            _warming.discard(voice)
            total = len(merged)
        logger.info("Filler cache ready: %d clips for voice=%s.", total, voice)


def _remember(voice: str, phrase: str) -> None:
    hist = _recent.setdefault(voice, deque(maxlen=4))
    hist.append(phrase)


def get_random_filler(voice_id: str = "") -> Optional[FillerClip]:
    """Pick a cached filler, avoiding recently used phrases when possible."""
    voice = voice_id or config.DEFAULT_VOICE_ID
    clips = _cache.get(voice) or []
    if not clips:
        return None
    recent = set(_recent.get(voice) or ())
    candidates = [c for c in clips if c.phrase not in recent] or list(clips)
    clip = random.choice(candidates)
    _remember(voice, clip.phrase)
    return clip


async def get_filler_for_voice(
    voice_id: str = "", language: str = "English"
) -> Optional[FillerClip]:
    """Return a cached filler instantly; warm in background on miss."""
    voice = voice_id or config.DEFAULT_VOICE_ID
    clip = get_random_filler(voice)
    if clip:
        return clip
    logger.info("Filler cache miss for voice=%s — skipping this turn, warming.", voice)
    asyncio.create_task(warm_cache(voice, language))
    return None


def get_filler_as_events(clip: FillerClip) -> List[dict]:
    """Convert a filler clip into the same JSON event format the browser expects."""
    b64 = base64.b64encode(clip.pcm_bytes).decode("ascii")
    return [
        {
            "type": "tts_start",
            "num_sentences": 1,
            "sample_rate": clip.sample_rate,
            "is_filler": True,
            "voice_id": clip.voice_id,
            "phrase": clip.phrase,
        },
        {
            "type": "tts_audio",
            "data": b64,
            "sample_rate": clip.sample_rate,
            "num_samples": clip.num_samples,
            "index": 0,
            "is_filler": True,
        },
        {"type": "tts_sentence_done", "is_filler": True},
        {
            "type": "filler_done",
            "voice_id": clip.voice_id,
            "phrase": clip.phrase,
        },
    ]
