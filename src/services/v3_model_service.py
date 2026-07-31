"""
v3 model service — faster-qwen3-tts (CUDA graph) engine behind the same
streaming surface the v2 wire protocol needs.

Self-contained: reads the existing voices.db (read-only) and voices/ WAVs,
recomputes clone prompts with the new engine at startup. No imports from the
v2 service stack, so this runs in the dedicated venv-tts-v3.
"""

import asyncio
import hashlib
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class VoiceEntry:
    voice_id: str
    name: str
    ref_audio_path: str
    ref_text: str
    builtin: bool
    prompt: object = field(default=None, repr=False)


class SentenceCache:
    """In-RAM LRU cache of synthesized sentences (same keying as v2)."""

    def __init__(self, max_bytes: int = 10 * 1024**3) -> None:
        self._data: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(text: str, voice_id: Optional[str], language: str) -> str:
        raw = f"{text}|{voice_id or ''}|{language}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, voice_id: Optional[str], language: str):
        key = self._key(text, voice_id, language)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return entry

    def put(self, text: str, voice_id: Optional[str], language: str,
            audio: np.ndarray, sr: int) -> None:
        key = self._key(text, voice_id, language)
        size = audio.nbytes
        with self._lock:
            if key in self._data:
                return
            while self._bytes + size > self._max_bytes and self._data:
                _, (old, _sr) = self._data.popitem(last=False)
                self._bytes -= old.nbytes
            self._data[key] = (audio, sr)
            self._bytes += size

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            self._bytes = 0
            return n

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._data),
                "bytes": self._bytes,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


class V3ModelService:
    """Loads FasterQwen3TTS once and serializes GPU work on one thread."""

    def __init__(self) -> None:
        self.model = None
        self.voices: dict[str, VoiceEntry] = {}
        self.cache = SentenceCache()
        # Single worker doubles as the GPU serialization point.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3-gpu")
        self.load_time_s: float = 0.0

    # -- Startup ---------------------------------------------------------------

    def load_model(self, model_path: str, attn_impl: str = "sdpa") -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        t0 = time.perf_counter()
        logger.info("v3: loading %s ...", model_path)
        self.model = FasterQwen3TTS.from_pretrained(
            model_path, device="cuda", dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self.load_time_s = time.perf_counter() - t0
        logger.info("v3: model loaded in %.1f s", self.load_time_s)

    def load_voices(self, db_path: str) -> None:
        """Read the existing v2 voices.db (read-only) and recompute prompts."""
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM voices ORDER BY builtin DESC, created_at ASC"
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            ref_path = row["ref_audio_path"]
            if not Path(ref_path).exists():
                logger.warning("v3: skipping voice %r — ref audio missing: %s",
                               row["voice_id"], ref_path)
                continue
            t0 = time.perf_counter()
            try:
                prompt = self.model.model.create_voice_clone_prompt(
                    ref_audio=ref_path, ref_text=row["ref_text"],
                )
            except Exception:
                logger.exception("v3: prompt compute failed for %r", row["voice_id"])
                continue
            self.voices[row["voice_id"]] = VoiceEntry(
                voice_id=row["voice_id"],
                name=row["name"],
                ref_audio_path=ref_path,
                ref_text=row["ref_text"],
                builtin=bool(row["builtin"]),
                prompt=prompt,
            )
            logger.info("v3: voice %r ready (%.1f s)",
                        row["voice_id"], time.perf_counter() - t0)
        logger.info("v3: %d voice(s) loaded", len(self.voices))

    def warmup(self) -> None:
        """First generation captures the CUDA graphs — keep it off the request path."""
        prompt = next(iter(self.voices.values())).prompt if self.voices else None
        t0 = time.perf_counter()
        self.model.generate_voice_clone(
            text="Warmup sentence to capture CUDA graphs.",
            language="English", voice_clone_prompt=prompt,
        )
        logger.info("v3: warmup / graph capture done in %.1f s",
                    time.perf_counter() - t0)

    # -- Streaming inference -----------------------------------------------------

    async def stream_sentence(
        self,
        text: str,
        language: str,
        prompt: object,
        chunk_size: int = 8,
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Bridge the sync streaming generator onto asyncio.

        Yields (audio_chunk float32 ndarray, sample_rate). Generation runs on
        the single GPU executor thread; abandoning this iterator (e.g. client
        disconnect) sets the cancel flag, stopping the producer at the next chunk.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        cancel = threading.Event()
        max_new_tokens = min(600, max(150, len(text) * 3))

        def producer() -> None:
            try:
                for chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                    text=text, language=language, voice_clone_prompt=prompt,
                    chunk_size=chunk_size, max_new_tokens=max_new_tokens,
                ):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, (chunk, sr))
            except Exception as exc:  # surfaced to the consumer
                logger.exception("v3: generation error for %r", text[:60])
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        future = loop.run_in_executor(self.executor, producer)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                chunk, sr = item
                yield np.asarray(chunk, dtype=np.float32), int(sr)
        finally:
            cancel.set()
            # Drain so the producer never blocks on a full queue.
            while not queue.empty():
                queue.get_nowait()
            await future


v3_model_service = V3ModelService()
