"""
REST endpoints for voice profile management.

GET    /voices            → list all registered voices
GET    /voices/{voice_id} → detail for a single voice
POST   /voices/{voice_id} → register a new voice (multipart: file + name + ref_text)
DELETE /voices/{voice_id} → remove a voice (built-ins are protected)
"""

import logging
import os
import time
from typing import Annotated, List, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from services.voice_service import voice_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voices", tags=["voices"])


# ── Response / request models ──────────────────────────────────────────────────

class VoiceSummary(BaseModel):
    voice_id: str = Field(description="Unique identifier for the voice.", examples=["mikey"])
    name: str = Field(description="Human-readable display name.", examples=["Mikey"])
    builtin: bool = Field(description="True for voices shipped with the server; these cannot be deleted.")
    prompt_ready: bool = Field(description="True when the speaker embedding is loaded in RAM and ready for inference.")

    model_config = {"json_schema_extra": {"example": {"voice_id": "mikey", "name": "Mikey", "builtin": True, "prompt_ready": True}}}


class VoiceDetail(VoiceSummary):
    ref_audio_path: str = Field(description="Absolute path to the reference WAV file on disk.")
    ref_text: str = Field(description="Transcript of the reference audio used for speaker matching.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "voice_id": "mikey",
                "name": "Mikey",
                "builtin": True,
                "prompt_ready": True,
                "ref_audio_path": "/app/src/assets/mikey_sample.wav",
                "ref_text": "So I was going through that today ...",
            }
        }
    }


class RegisteredResponse(BaseModel):
    voice_id: str = Field(examples=["alice"])
    name: str = Field(examples=["Alice"])
    status: Literal["registered"]

    model_config = {"json_schema_extra": {"example": {"voice_id": "alice", "name": "Alice", "status": "registered"}}}


class DeletedResponse(BaseModel):
    voice_id: str = Field(examples=["alice"])
    status: Literal["deleted"]

    model_config = {"json_schema_extra": {"example": {"voice_id": "alice", "status": "deleted"}}}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=List[VoiceSummary],
    summary="List all voices",
    description=(
        "Returns every voice profile currently registered, including built-in voices. "
        "Use the `voice_id` values from this list in the `/ws/tts` WebSocket request."
    ),
    response_description="Array of voice summaries, built-ins first.",
)
async def list_voices() -> List[VoiceSummary]:
    t0 = time.perf_counter()
    entries = voice_service.list_all()
    result = [
        VoiceSummary(
            voice_id=e.voice_id,
            name=e.name,
            builtin=e.builtin,
            prompt_ready=e.prompt_item is not None,
        )
        for e in entries
    ]
    logger.info("GET /voices → %d voice(s) in %.1f ms", len(result), (time.perf_counter() - t0) * 1000)
    return result


@router.get(
    "/{voice_id}",
    response_model=VoiceDetail,
    summary="Get voice detail",
    description="Returns full metadata for a single voice, including the reference audio path and transcript.",
    response_description="Voice detail including ref_audio_path and ref_text.",
    responses={
        404: {"description": "Voice not found."},
    },
)
async def get_voice(voice_id: str) -> VoiceDetail:
    t0 = time.perf_counter()
    logger.debug("GET /voices/%s", voice_id)
    entry = voice_service.get(voice_id)
    if entry is None:
        logger.warning("GET /voices/%s → 404", voice_id)
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
    logger.info("GET /voices/%s → found (%s) in %.1f ms", voice_id, entry.name, (time.perf_counter() - t0) * 1000)
    return VoiceDetail(
        voice_id=entry.voice_id,
        name=entry.name,
        builtin=entry.builtin,
        prompt_ready=entry.prompt_item is not None,
        ref_audio_path=entry.ref_audio_path,
        ref_text=entry.ref_text,
    )


@router.post(
    "/{voice_id}",
    status_code=201,
    response_model=RegisteredResponse,
    summary="Register a new voice",
    description=(
        "Upload a reference WAV clip and its transcript to register a new voice profile. "
        "The server immediately computes the speaker embedding and ICL codes and caches them "
        "in RAM — subsequent TTS requests using this `voice_id` incur zero additional overhead. "
        "The profile is also persisted to SQLite so it survives restarts.\n\n"
        "**Form fields**\n"
        "- `name` — human-readable label (e.g. `Alice`)\n"
        "- `ref_text` — verbatim transcript of the uploaded audio; longer is better\n"
        "- `file` — mono or stereo WAV, any sample rate (server resamples as needed)\n\n"
        "**`voice_id`** must be URL-safe (letters, digits, hyphens, underscores)."
    ),
    response_description="Confirmation that the voice was registered and its prompt is cached.",
    responses={
        201: {"description": "Voice registered and prompt cached in RAM."},
        409: {"description": "A voice with this `voice_id` already exists."},
        422: {"description": "Audio processing failed (bad file, corrupted WAV, etc.)."},
    },
)
async def register_voice(
    voice_id: str,
    name: Annotated[str, Form(description="Human-readable display name.", examples=["Alice"])],
    ref_text: Annotated[str, Form(description="Verbatim transcript of the reference audio.")],
    file: Annotated[UploadFile, File(description="Reference WAV audio clip (any sample rate, mono or stereo).")],
) -> RegisteredResponse:
    t0 = time.perf_counter()
    logger.info(
        "POST /voices/%s | name=%r  ref_text_len=%d  filename=%r",
        voice_id, name, len(ref_text), file.filename,
    )

    if voice_service.exists(voice_id):
        logger.warning("POST /voices/%s → 409 already exists.", voice_id)
        raise HTTPException(status_code=409, detail=f"Voice '{voice_id}' already exists.")

    audio_path = os.path.join(config.VOICES_DIR, f"{voice_id}.wav")
    data = await file.read()
    logger.debug("POST /voices/%s | received %d bytes, writing to %s", voice_id, len(data), audio_path)
    with open(audio_path, "wb") as f:
        f.write(data)

    try:
        await voice_service.register(
            voice_id=voice_id,
            name=name,
            ref_audio_path=audio_path,
            ref_text=ref_text,
        )
    except Exception as exc:
        logger.exception("POST /voices/%s | prompt computation failed, rolling back WAV.", voice_id)
        try:
            os.remove(audio_path)
        except OSError:
            logger.warning("POST /voices/%s | could not remove partial WAV at %s", voice_id, audio_path)
        raise HTTPException(status_code=422, detail=f"Failed to process voice audio: {exc}")

    elapsed = time.perf_counter() - t0
    logger.info("POST /voices/%s → 201 registered in %.2f s", voice_id, elapsed)
    return RegisteredResponse(voice_id=voice_id, name=name, status="registered")


@router.delete(
    "/{voice_id}",
    status_code=200,
    response_model=DeletedResponse,
    summary="Delete a voice",
    description=(
        "Removes a user-registered voice from RAM, SQLite, and disk. "
        "Built-in voices (e.g. `mikey`) are protected and cannot be deleted."
    ),
    response_description="Confirmation that the voice was removed.",
    responses={
        400: {"description": "Attempted to delete a built-in voice."},
        404: {"description": "Voice not found."},
    },
)
async def delete_voice(voice_id: str) -> DeletedResponse:
    t0 = time.perf_counter()
    logger.info("DELETE /voices/%s", voice_id)

    entry = voice_service.get(voice_id)
    if entry is None:
        logger.warning("DELETE /voices/%s → 404", voice_id)
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
    if entry.builtin:
        logger.warning("DELETE /voices/%s → 400 built-in voice.", voice_id)
        raise HTTPException(status_code=400, detail=f"Built-in voice '{voice_id}' cannot be deleted.")

    audio_path = entry.ref_audio_path
    voice_service.remove(voice_id)

    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            logger.debug("DELETE /voices/%s | removed WAV at %s", voice_id, audio_path)
        except OSError:
            logger.warning("DELETE /voices/%s | could not remove WAV at %s", voice_id, audio_path)

    elapsed = time.perf_counter() - t0
    logger.info("DELETE /voices/%s → 200 deleted in %.1f ms", voice_id, elapsed * 1000)
    return DeletedResponse(voice_id=voice_id, status="deleted")
