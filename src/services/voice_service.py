"""
In-RAM voice store backed by SQLite for persistence.

Every registered voice has its VoiceClonePromptItem (speaker embedding + ICL
codes) pre-computed and kept in a plain dict.  With 100 GB RAM there is no
eviction — all prompts stay live for zero-latency lookups during inference.

Thread / concurrency notes:
  - Dict reads happen on the asyncio event loop (single thread) → no lock needed.
  - Registration awaits the executor (GPU work) and then writes the dict from
    the event loop, so it is also safe without a lock in practice.  An asyncio
    Lock is added anyway to guard against concurrent POST /voices requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qwen_tts import VoiceClonePromptItem

from db import database as db
from services.model_service import model_service

logger = logging.getLogger(__name__)


@dataclass
class VoiceEntry:
    voice_id: str
    name: str
    ref_audio_path: str
    ref_text: str
    builtin: bool
    prompt_item: Optional[VoiceClonePromptItem] = field(default=None, repr=False)


class VoiceService:
    def __init__(self) -> None:
        self._voices: Dict[str, VoiceEntry] = {}
        self._lock = asyncio.Lock()

    # ── Startup ────────────────────────────────────────────────────────────────

    async def load_all(self) -> None:
        """
        Read every voice from SQLite and pre-compute its prompt.
        Called once during server lifespan before the first request.
        """
        records = db.get_all()
        total = len(records)
        logger.info("Loading %d voice(s) into RAM ...", total)
        t_all = time.perf_counter()
        loop = asyncio.get_running_loop()

        ok = 0
        for idx, rec in enumerate(records, start=1):
            logger.info(
                "[%d/%d] Computing prompt for voice '%s' (%s) ...",
                idx, total, rec.voice_id, rec.name,
            )
            t0 = time.perf_counter()
            try:
                prompt_item: VoiceClonePromptItem = await loop.run_in_executor(
                    model_service.executor,
                    lambda r=rec: model_service.compute_prompt_sync(r.ref_audio_path, r.ref_text),
                )
                self._voices[rec.voice_id] = VoiceEntry(
                    voice_id=rec.voice_id,
                    name=rec.name,
                    ref_audio_path=rec.ref_audio_path,
                    ref_text=rec.ref_text,
                    builtin=rec.builtin,
                    prompt_item=prompt_item,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "[%d/%d] Voice '%s' cached in %.2f s",
                    idx, total, rec.voice_id, elapsed,
                )
                ok += 1
            except Exception:
                elapsed = time.perf_counter() - t0
                logger.exception(
                    "[%d/%d] Failed to load voice '%s' after %.2f s — skipping.",
                    idx, total, rec.voice_id, elapsed,
                )

        total_elapsed = time.perf_counter() - t_all
        logger.info(
            "Voice preload complete: %d/%d loaded in %.2f s total.",
            ok, total, total_elapsed,
        )

    # ── Registration ───────────────────────────────────────────────────────────

    async def register(
        self,
        voice_id: str,
        name: str,
        ref_audio_path: str,
        ref_text: str,
        builtin: bool = False,
    ) -> VoiceEntry:
        """Compute prompt (GPU), cache in RAM, persist to SQLite."""
        logger.info(
            "Registering voice '%s' (name=%r, builtin=%s, audio=%r) ...",
            voice_id, name, builtin, ref_audio_path,
        )
        t_total = time.perf_counter()

        async with self._lock:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            prompt_item: VoiceClonePromptItem = await loop.run_in_executor(
                model_service.executor,
                lambda: model_service.compute_prompt_sync(ref_audio_path, ref_text),
            )
            prompt_elapsed = time.perf_counter() - t0
            logger.debug("Prompt computation done in %.2f s", prompt_elapsed)

            entry = VoiceEntry(
                voice_id=voice_id,
                name=name,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                builtin=builtin,
                prompt_item=prompt_item,
            )
            self._voices[voice_id] = entry
            logger.debug("Voice '%s' added to in-memory store (total: %d)", voice_id, len(self._voices))

            if not db.exists(voice_id):
                db.insert(voice_id, name, ref_audio_path, ref_text, builtin)
                logger.debug("Voice '%s' persisted to SQLite.", voice_id)
            else:
                logger.debug("Voice '%s' already in SQLite — skipping insert.", voice_id)

        total_elapsed = time.perf_counter() - t_total
        logger.info(
            "Voice '%s' registered and ready in %.2f s.",
            voice_id, total_elapsed,
        )
        return entry

    # ── Queries ────────────────────────────────────────────────────────────────

    def get(self, voice_id: str) -> Optional[VoiceEntry]:
        entry = self._voices.get(voice_id)
        if entry is None:
            logger.debug("get(%r) → miss (available: %s)", voice_id, list(self._voices))
        else:
            logger.debug("get(%r) → hit (prompt_ready=%s)", voice_id, entry.prompt_item is not None)
        return entry

    def list_all(self) -> List[VoiceEntry]:
        entries = list(self._voices.values())
        logger.debug("list_all() → %d voice(s)", len(entries))
        return entries

    def exists(self, voice_id: str) -> bool:
        result = voice_id in self._voices
        logger.debug("exists(%r) → %s", voice_id, result)
        return result

    # ── Deletion ───────────────────────────────────────────────────────────────

    def remove(self, voice_id: str) -> Optional[VoiceEntry]:
        entry = self._voices.pop(voice_id, None)
        if entry is None:
            logger.warning("remove(%r) called but voice not in RAM store.", voice_id)
            return None
        db.delete(voice_id)
        logger.info(
            "Voice '%s' removed from RAM and SQLite (remaining: %d).",
            voice_id, len(self._voices),
        )
        return entry


voice_service = VoiceService()
