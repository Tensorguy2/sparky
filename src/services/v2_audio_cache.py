"""
v2 audio cache with zero-copy reads and bulk lookup.

Improvements over audio_cache.py:
  - get() returns a read-only view (no numpy copy per hit)
  - get_many() performs batch lookups in one lock acquisition
  - Tracks entry age for staleness monitoring
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    audio: np.ndarray
    sample_rate: int
    byte_size: int
    created_at: float = field(default_factory=time.monotonic)


class V2AudioCache:
    """Bounded LRU cache with zero-copy reads and batch operations."""

    def __init__(self, max_bytes: int = 0) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._current_bytes: int = 0
        self._max_bytes: int = max_bytes or config.AUDIO_CACHE_MAX_BYTES
        self._hits: int = 0
        self._misses: int = 0
        logger.info(
            "V2AudioCache initialised | max=%.1f GB",
            self._max_bytes / (1024 ** 3),
        )

    @staticmethod
    def _key(text: str, voice_id: Optional[str], language: str) -> str:
        raw = f"{text}|{voice_id or ''}|{language}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self, text: str, voice_id: Optional[str], language: str,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Return (audio_view, sample_rate) or None. The array is read-only."""
        key = self._key(text, voice_id, language)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
        # Return a read-only view -- no allocation
        view = entry.audio.view()
        view.flags.writeable = False
        return view, entry.sample_rate

    def get_many(
        self,
        keys: List[Tuple[str, Optional[str], str]],
    ) -> Dict[int, Tuple[np.ndarray, int]]:
        """Batch cache lookup. Returns {index: (audio_view, sample_rate)} for hits."""
        results: Dict[int, Tuple[np.ndarray, int]] = {}
        with self._lock:
            for idx, (text, voice_id, language) in enumerate(keys):
                k = self._key(text, voice_id, language)
                entry = self._store.get(k)
                if entry is not None:
                    self._store.move_to_end(k)
                    self._hits += 1
                    view = entry.audio.view()
                    view.flags.writeable = False
                    results[idx] = (view, entry.sample_rate)
                else:
                    self._misses += 1
        return results

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

            stored = audio.copy()
            stored.flags.writeable = False
            self._store[key] = _CacheEntry(
                audio=stored,
                sample_rate=sample_rate,
                byte_size=nbytes,
            )
            self._current_bytes += nbytes

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._current_bytes = 0
        logger.info("V2 cache cleared (%d entries removed)", n)
        return n

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            now = time.monotonic()
            ages = [now - e.created_at for e in self._store.values()]
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
                "avg_age_s": round(sum(ages) / len(ages), 1) if ages else 0,
                "oldest_s": round(max(ages), 1) if ages else 0,
            }


v2_audio_cache = V2AudioCache()
