"""
STT router: dispatches to OpenAI Realtime, OpenAI batch, local Parakeet, or Whisper.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, List, Optional

import config
from services import openai_stt, parakeet_stt, stt_service
from services.openai_stt import RealtimeTranscriber
from services.parakeet_stt import PARAKEET_MODEL_ID

logger = logging.getLogger(__name__)

RealtimeDeltaCallback = Callable[[str], Awaitable[None]]

OPENAI_BATCH_MODELS = frozenset({
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "whisper-1",
})

PARAKEET_ENTRY = {
    "id": PARAKEET_MODEL_ID,
    "label": "Local Parakeet TDT 0.6B v2 (offline, fast English)",
    "provider": "local",
    "streaming": False,
}


def _ensure_parakeet_listed() -> None:
    ids = {m["id"] for m in config.AVAILABLE_STT_MODELS}
    if PARAKEET_MODEL_ID not in ids:
        # Prefer Parakeet near the top of local options (after cloud streaming).
        insert_at = 0
        for i, m in enumerate(config.AVAILABLE_STT_MODELS):
            if m["id"].startswith("local:"):
                insert_at = i
                break
        else:
            insert_at = len(config.AVAILABLE_STT_MODELS)
        config.AVAILABLE_STT_MODELS.insert(insert_at, PARAKEET_ENTRY)


_ensure_parakeet_listed()


def is_realtime_model(model_id: str) -> bool:
    return model_id == "gpt-realtime-whisper"


def is_openai_batch_model(model_id: str) -> bool:
    return model_id in OPENAI_BATCH_MODELS


def is_local_model(model_id: str) -> bool:
    return model_id.startswith("local:")


def is_parakeet_model(model_id: str) -> bool:
    return model_id == PARAKEET_MODEL_ID or "parakeet" in model_id


def local_whisper_size(model_id: str) -> str:
    return model_id.split(":", 1)[1]


def resolve_stt_model(model_id: Optional[str]) -> str:
    _ensure_parakeet_listed()
    model = (model_id or config.DEFAULT_STT_MODEL).strip()
    valid = {m["id"] for m in config.AVAILABLE_STT_MODELS}
    if model not in valid:
        logger.warning("Unknown STT model %r — using %s.", model, config.DEFAULT_STT_MODEL)
        return config.DEFAULT_STT_MODEL
    return model


def get_stt_info(active_model: Optional[str] = None) -> dict:
    model = resolve_stt_model(active_model)
    info = {
        "default_model": config.DEFAULT_STT_MODEL,
        "active_model": model,
        "models": config.AVAILABLE_STT_MODELS,
        "realtime_delay": config.STT_REALTIME_DELAY,
        "language": config.STT_LANGUAGE,
        "streaming": is_realtime_model(model),
        "provider": _provider_for(model),
    }
    if is_parakeet_model(model):
        info.update(parakeet_stt.get_stt_status())
    elif is_local_model(model):
        info.update(stt_service.get_stt_status(model_size=local_whisper_size(model)))
    else:
        info["loaded"] = True
    return info


def _provider_for(model_id: str) -> str:
    if is_local_model(model_id):
        return "local"
    return "openai"


async def create_realtime_session(
    language: Optional[str] = None,
    on_delta: Optional[RealtimeDeltaCallback] = None,
    on_vad_stopped: Optional[Callable[[], Awaitable[None]]] = None,
) -> RealtimeTranscriber:
    lang = language or config.STT_LANGUAGE
    session = RealtimeTranscriber(
        language=lang,
        delay=config.STT_REALTIME_DELAY,
        on_delta=on_delta,
        on_vad_stopped=on_vad_stopped,
        server_vad=config.STT_SERVER_VAD,
        vad_silence_ms=config.STT_VAD_SILENCE_MS,
    )
    await session.start()
    return session


async def transcribe(
    pcm_chunks: List[bytes],
    model_id: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: int = 24000,
) -> str:
    model = resolve_stt_model(model_id)
    lang = language or config.STT_LANGUAGE

    if is_realtime_model(model):
        raise RuntimeError("Use create_realtime_session() for gpt-realtime-whisper.")

    if is_openai_batch_model(model):
        return await openai_stt.transcribe_batch(
            pcm_chunks, model, lang, sample_rate=sample_rate,
        )

    if is_parakeet_model(model):
        return await parakeet_stt.transcribe_async(
            pcm_chunks,
            sample_rate=sample_rate,
            language=lang,
        )

    if is_local_model(model):
        return await stt_service.transcribe_async(
            pcm_chunks,
            sample_rate=sample_rate,
            language=lang,
            model_size=local_whisper_size(model),
        )

    raise ValueError(f"Unsupported STT model: {model}")
