"""
Sentence-level LRU audio cache.

Stores generated PCM audio keyed on (text, voice_id, language) to skip
GPU inference entirely for repeated sentences.

Thread-safe via threading.Lock (cache ops are fast dict lookups).
Bounded by config.AUDIO_CACHE_MAX_BYTES (default 10 GB ≈ 100 000 s of audio).
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    audio: np.ndarray
    sample_rate: int
    byte_size: int
    created_at: float = field(default_factory=time.monotonic)


class AudioCache:
    """Bounded LRU cache mapping (sentence, voice, lang) → PCM audio."""

    def __init__(self, max_bytes: int = 0) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._current_bytes: int = 0
        self._max_bytes: int = max_bytes or config.AUDIO_CACHE_MAX_BYTES
        self._hits: int = 0
        self._misses: int = 0
        logger.info(
            "AudioCache initialised | max=%.1f GB",
            self._max_bytes / (1024 ** 3),
        )

    # ── Key construction ──────────────────────────────────────────────────────

    @staticmethod
    def _key(text: str, voice_id: Optional[str], language: str) -> str:
        raw = f"{text}|{voice_id or ''}|{language}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self, text: str, voice_id: Optional[str], language: str,
    ) -> Optional[Tuple[np.ndarray, int]]:
        key = self._key(text, voice_id, language)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
        logger.debug("Cache HIT | key=%s…", key[:12])
        return entry.audio.copy(), entry.sample_rate

    def put(
        self,
        text: str,
        voice_id: Optional[str],
        language: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        key = self._key(text, voice_id, language)
        nbytes = audio.nbytes
        if nbytes > self._max_bytes:
            return

        with self._lock:
            if key in self._store:
                old = self._store.pop(key)
                self._current_bytes -= old.byte_size

            while (
                self._current_bytes + nbytes > self._max_bytes
                and self._store
            ):
                _, evicted = self._store.popitem(last=False)
                self._current_bytes -= evicted.byte_size

            self._store[key] = _CacheEntry(
                audio=audio.copy(),
                sample_rate=sample_rate,
                byte_size=nbytes,
            )
            self._current_bytes += nbytes

        logger.debug(
            "Cache PUT | key=%s… bytes=%d total=%d/%d entries=%d",
            key[:12], nbytes, self._current_bytes, self._max_bytes,
            len(self._store),
        )

    def clear(self) -> int:
        """Drop all entries.  Returns the number removed."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._current_bytes = 0
        logger.info("Cache cleared (%d entries removed)", n)
        return n

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "current_bytes": self._current_bytes,
                "max_bytes": self._max_bytes,
                "utilization_pct": round(
                    100 * self._current_bytes / self._max_bytes, 2
                ) if self._max_bytes else 0,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(100 * self._hits / total, 2) if total else 0,
            }


audio_cache = AudioCache()
