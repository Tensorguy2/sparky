"""
WebSocket endpoint: /ws/tts

Protocol
--------
Client  → JSON  { "text": "...", "language": "English", "voice_id": "mikey" }
                 OR inline clone: { ..., "ref_audio": "<path|base64>", "ref_text": "..." }

Server  → JSON  { "type": "metadata", "sample_rate": 24000, "channels": 1,
                  "encoding": "pcm_f32le", "num_sentences": N }
Server  → JSON  { "type": "chunk_header", "index": i, "num_samples": N, "size_bytes": B }
Server  → bytes  raw PCM float32 LE
        ... repeated per sentence ...
Server  → JSON  { "type": "done" }
Server  → JSON  { "type": "error", "message": "..." }   (on failure, then closes)

See GET /ws/tts/schema for the machine-readable protocol schema.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from qwen_tts import VoiceClonePromptItem
from services.model_service import model_service
from services.voice_service import voice_service
from utils.text import split_into_sentences

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tts"])


# ── Protocol schema (for /ws/tts/schema) ──────────────────────────────────────

_WS_SCHEMA = {
    "endpoint": "ws://<host>/ws/tts",
    "description": (
        "Bidirectional WebSocket for sentence-streamed TTS. "
        "The client sends one JSON request; the server streams back audio chunks, "
        "one per sentence, as interleaved JSON headers + raw PCM bytes."
    ),
    "client_message": {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text":      {"type": "string",  "description": "Text to synthesize."},
            "language":  {"type": "string",  "default": "English", "description": "Target language."},
            "voice_id":  {"type": "string",  "description": "ID of a pre-registered voice (see GET /voices). Mutually exclusive with ref_audio."},
            "ref_audio": {"type": "string",  "description": "Inline reference audio: absolute path or base64-encoded WAV. Mutually exclusive with voice_id."},
            "ref_text":  {"type": "string",  "description": "Transcript of ref_audio. Recommended for best speaker matching."},
        },
        "example": {"text": "Hello world!", "language": "English", "voice_id": "mikey"},
    },
    "server_messages": [
        {
            "type": "metadata",
            "description": "Sent once before any audio, immediately after the request is validated.",
            "fields": {
                "sample_rate":   24000,
                "channels":      1,
                "encoding":      "pcm_f32le",
                "num_sentences": "<int> number of sentences the text was split into",
            },
        },
        {
            "type": "sample_rate_correction",
            "description": "Sent at most once, only when the model returns a sample rate other than 24000.",
            "fields": {"sample_rate": "<int>"},
        },
        {
            "type": "chunk_header",
            "description": "JSON header preceding each binary audio chunk.",
            "fields": {
                "index":       "<int> 0-based sentence index",
                "num_samples": "<int> number of float32 samples in the following bytes message",
                "size_bytes":  "<int> byte length of the following bytes message",
            },
        },
        {
            "type": "<binary frame>",
            "description": "Raw PCM float32 little-endian audio, immediately follows its chunk_header.",
        },
        {
            "type": "done",
            "description": "Sent after all sentences are streamed. Server closes the connection.",
        },
        {
            "type": "error",
            "description": "Sent on any failure. Server closes the connection.",
            "fields": {"message": "<str>"},
        },
    ],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_conn_id() -> str:
    return uuid.uuid4().hex[:8]


async def _send_error(ws: WebSocket, conn_id: str, message: str) -> None:
    logger.warning("[%s] Sending error to client: %s", conn_id, message)
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}))
        await ws.close()
    except Exception:
        pass


# ── REST: WebSocket protocol schema ───────────────────────────────────────────

@router.get(
    "/ws/tts/schema",
    summary="WebSocket TTS — protocol schema",
    description=(
        "Returns the full JSON schema describing the `/ws/tts` WebSocket protocol: "
        "client request shape, server frame sequence, and field definitions. "
        "WebSocket endpoints cannot appear in OpenAPI directly; use this endpoint "
        "to inspect or generate client code."
    ),
    response_description="WebSocket protocol schema as a JSON object.",
    tags=["tts"],
)
async def ws_schema() -> JSONResponse:
    return JSONResponse(_WS_SCHEMA)


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@router.websocket("/ws/tts")
async def ws_tts(ws: WebSocket) -> None:
    conn_id = _new_conn_id()
    client = ws.client
    t_connect = time.perf_counter()
    logger.info("[%s] WebSocket connection from %s:%s", conn_id, client.host, client.port)

    await ws.accept()

    loop = asyncio.get_running_loop()

    # ── 1. Receive and validate request ───────────────────────────────────────
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        req: dict = json.loads(raw)
        logger.debug("[%s] Request received: %s", conn_id, raw[:200])
    except asyncio.TimeoutError:
        logger.warning("[%s] Client did not send a request within 10 s.", conn_id)
        await _send_error(ws, conn_id, "Request timed out (10 s)")
        return
    except Exception as exc:
        logger.warning("[%s] Failed to parse request JSON: %s", conn_id, exc)
        await _send_error(ws, conn_id, f"Invalid JSON: {exc}")
        return

    text: str = req.get("text", "").strip()
    language: str = req.get("language", "English")
    voice_id: Optional[str] = req.get("voice_id")

    if not text:
        logger.warning("[%s] Empty 'text' field.", conn_id)
        await _send_error(ws, conn_id, "'text' field is required")
        return

    logger.info(
        "[%s] Request | voice_id=%r  language=%s  text_len=%d  preview=%r",
        conn_id, voice_id, language, len(text), text[:80],
    )

    # ── 2. Resolve voice clone prompt ─────────────────────────────────────────
    prompt_item: Optional[VoiceClonePromptItem] = None

    if voice_id:
        entry = voice_service.get(voice_id)
        if entry is None:
            available = [e.voice_id for e in voice_service.list_all()]
            logger.warning("[%s] Voice '%s' not found. Available: %s", conn_id, voice_id, available)
            await _send_error(
                ws, conn_id,
                f"Voice '{voice_id}' not found. Available: {available}. "
                "Register a new voice via POST /voices/<id>.",
            )
            return
        prompt_item = entry.prompt_item
        logger.debug("[%s] Resolved voice '%s' (%s) from RAM cache.", conn_id, voice_id, entry.name)

    elif "ref_audio" in req:
        logger.info("[%s] Inline ref_audio provided — computing prompt on the fly.", conn_id)
        t0 = time.perf_counter()
        try:
            prompt_item = await loop.run_in_executor(
                model_service.executor,
                lambda: model_service.compute_prompt_sync(
                    req["ref_audio"], req.get("ref_text")
                ),
            )
            logger.info("[%s] Inline prompt computed in %.2f s.", conn_id, time.perf_counter() - t0)
        except Exception as exc:
            logger.exception("[%s] Failed to process ref_audio.", conn_id)
            await _send_error(ws, conn_id, f"Failed to process ref_audio: {exc}")
            return
    else:
        logger.info("[%s] No voice_id or ref_audio — using x_vector_only mode.", conn_id)

    # ── 3. Split text and send metadata ───────────────────────────────────────
    sentences = split_into_sentences(text)
    logger.info("[%s] Text split into %d sentence(s).", conn_id, len(sentences))
    logger.debug("[%s] Sentences: %s", conn_id, sentences)

    await ws.send_text(json.dumps({
        "type": "metadata",
        "sample_rate": 24000,
        "channels": 1,
        "encoding": "pcm_f32le",
        "num_sentences": len(sentences),
    }))

    # ── 4. Stream sentences ───────────────────────────────────────────────────
    t_stream_start = time.perf_counter()
    notified_sr = False
    total_audio_s = 0.0

    for idx, sentence in enumerate(sentences):
        logger.info(
            "[%s] [%d/%d] Synthesising: %r",
            conn_id, idx + 1, len(sentences), sentence[:80],
        )
        t0 = time.perf_counter()
        try:
            streamer = model_service.infer_stream(sentence, language, prompt_item)
            
            async for audio_chunk, sr in streamer.get_async():
                if len(audio_chunk) == 0:
                    continue
                    
                if not notified_sr and sr != 24000:
                    notified_sr = True
                    logger.debug("[%s] Model returned sr=%d (expected 24000) — notifying client.", conn_id, sr)
                    await ws.send_text(json.dumps({"type": "sample_rate_correction", "sample_rate": sr}))

                pcm: bytes = audio_chunk.astype(np.float32).tobytes()
                await ws.send_text(json.dumps({
                    "type": "chunk_header",
                    "index": idx,
                    "num_samples": len(audio_chunk),
                    "size_bytes": len(pcm),
                }))
                await ws.send_bytes(pcm)
                
                audio_duration_s = len(audio_chunk) / sr
                total_audio_s += audio_duration_s

        except WebSocketDisconnect:
            logger.info("[%s] Client disconnected during inference of sentence %d.", conn_id, idx)
            return
        except Exception as exc:
            logger.exception("[%s] Inference failed on sentence %d.", conn_id, idx)
            await _send_error(ws, conn_id, f"Inference error on chunk {idx}: {exc}")
            return

        infer_elapsed = time.perf_counter() - t0
        logger.debug(
            "[%s] [%d/%d] Sentence done | wall=%.2f s",
            conn_id, idx + 1, len(sentences),
            infer_elapsed
        )

    await ws.send_text(json.dumps({"type": "done"}))
    try:
        await ws.close()
    except Exception:
        pass

    total_wall = time.perf_counter() - t_connect
    logger.info(
        "[%s] Done | total_wall=%.2f s  audio_generated=%.2f s  sentences=%d  RTF=%.3f",
        conn_id,
        total_wall,
        total_audio_s,
        len(sentences),
        total_wall / total_audio_s if total_audio_s else 0,
    )
