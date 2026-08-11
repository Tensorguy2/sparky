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
import os
import sqlite3
import tempfile
import threading
import time
import wave
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
    def _key(
        text: str,
        voice_id: Optional[str],
        language: str,
        instruct: str = "",
    ) -> str:
        raw = f"{text}|{voice_id or ''}|{language}|{instruct or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        text: str,
        voice_id: Optional[str],
        language: str,
        instruct: str = "",
    ):
        key = self._key(text, voice_id, language, instruct)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return entry

    def put(
        self,
        text: str,
        voice_id: Optional[str],
        language: str,
        audio: np.ndarray,
        sr: int,
        instruct: str = "",
    ) -> None:
        key = self._key(text, voice_id, language, instruct)
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

    # -- Speaker carry-over ------------------------------------------------------

    async def build_carryover_prompt(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: str,
    ) -> Optional[object]:
        """Clone prompt taken from audio this engine just produced.

        Conditioning a sentence on the original reference recording leaves the
        speaker free to re-roll on every generation: measured on this
        deployment, consecutive takes of one voice sat 0.12 apart in log-mel
        cosine distance while two genuinely different voices sat 0.08 apart --
        the same voice was landing further from itself than from someone else.
        Conditioning on the previous sentence's own output instead pins it, and
        brings that distance to 0.01.

        Runs on the GPU executor, so it serialises behind generation rather than
        competing with it. Costs ~40 ms and is only needed from the second
        sentence of a turn onward, leaving first-audio latency untouched.
        """
        if self.model is None or audio is None or len(audio) == 0:
            return None

        def work() -> Optional[object]:
            path = None
            try:
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="carryover-")
                os.close(fd)
                clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sample_rate)
                    w.writeframes((clipped * 32767).astype(np.int16).tobytes())
                return self.model.model.create_voice_clone_prompt(
                    ref_audio=path, ref_text=text,
                )
            except Exception:
                # Never fatal: the caller falls back to the reference prompt,
                # which is the current behaviour.
                logger.exception("v3: carry-over prompt failed for %r", text[:60])
                return None
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, work)

    # -- Streaming inference -----------------------------------------------------

    async def stream_sentence(
        self,
        text: str,
        language: str,
        prompt: object,
        chunk_size: int = 8,
        instruct: str = "",
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Bridge the sync streaming generator onto asyncio.

        Yields (audio_chunk float32 ndarray, sample_rate). Generation runs on
        the single GPU executor thread; abandoning this iterator (e.g. client
        disconnect) sets the cancel flag, stopping the producer at the next chunk.
        Optional ``instruct`` is experimental on Base voice-clone.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        cancel = threading.Event()
        max_new_tokens = min(600, max(150, len(text) * 3))
        instruct = (instruct or "").strip() or None

        def producer() -> None:
            try:
                kwargs = dict(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                    chunk_size=chunk_size,
                    max_new_tokens=max_new_tokens,
                )
                if instruct:
                    kwargs["instruct"] = instruct
                    logger.info("v3: applying instruct=%r for %r", instruct[:80], text[:40])
                for chunk, sr, _timing in self.model.generate_voice_clone_streaming(**kwargs):
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
