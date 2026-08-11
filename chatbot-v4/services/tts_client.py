"""
Async WebSocket client for the existing Qwen3-TTS server.

Connects to the TTS server's /ws/tts endpoint, sends text, and yields
audio chunks (base64-encoded PCM) back to the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Union

import websockets

import config

logger = logging.getLogger(__name__)

EXCUSE_ME_PHRASE = "Excuse me."


@dataclass
class TTSAudioChunk:
    """A single audio chunk from the TTS server."""
    index: int
    num_samples: int
    pcm_bytes: bytes
    sample_rate: int = 24000


@dataclass
class TTSInstability:
    """Server detected bad audio; client should flush playback."""
    sentence_index: int
    reason: str
    will_retry: bool


TTSEvent = Union[TTSAudioChunk, TTSInstability]


async def reset_tts_server() -> None:
    """Clear sentence cache and drop stale GPU-side state between recoveries."""
    import httpx

    url = f"{config.TTS_REST_URL}/cache/clear"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url)
            if resp.status_code == 200:
                cleared = resp.json().get("cleared", 0)
                logger.info("TTS cache cleared (%d entries).", cleared)
            else:
                logger.warning("TTS cache clear failed: HTTP %d", resp.status_code)
    except Exception:
        logger.exception("TTS cache clear request failed.")


async def stream_tts(
    text: str,
    voice_id: str = "",
    language: str = "English",
    instruct: str = "",
    turn_id: str = "",
) -> AsyncGenerator[TTSEvent, None]:
    """
    Connect to the TTS server, synthesize text, and yield audio chunks.

    Each chunk contains raw PCM float32 LE bytes at 24 kHz mono.
    May yield TTSInstability when the fast server detects bad audio.
    Optional ``instruct`` is natural-language delivery guidance for Qwen TTS.
    """
    voice_id = voice_id or config.DEFAULT_VOICE_ID
    url = config.TTS_WS_URL
    instruct = (instruct or "").strip()

    logger.info(
        "TTS request | voice=%s lang=%s text_len=%d instruct=%r url=%s",
        voice_id, language, len(text), instruct[:80] if instruct else "", url,
    )

    try:
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            payload = {
                "text": text,
                "language": language,
                "voice_id": voice_id,
            }
            if instruct:
                payload["instruct"] = instruct
            if turn_id:
                payload["turn_id"] = str(turn_id)
            await ws.send(json.dumps(payload))

            sample_rate = 24000
            expecting_binary = False
            current_header: Optional[dict] = None

            async for msg in ws:
                if isinstance(msg, str):
                    pkt = json.loads(msg)
                    pkt_type = pkt.get("type")

                    if pkt_type == "metadata":
                        sample_rate = pkt.get("sample_rate", 24000)
                        logger.debug(
                            "TTS metadata: sr=%d sentences=%d",
                            sample_rate, pkt.get("num_sentences", 0),
                        )

                    elif pkt_type == "sample_rate_correction":
                        sample_rate = pkt["sample_rate"]

                    elif pkt_type == "chunk_header":
                        current_header = pkt
                        expecting_binary = True

                    elif pkt_type == "instability":
                        yield TTSInstability(
                            sentence_index=pkt.get("sentence_index", -1),
                            reason=pkt.get("reason", "unknown"),
                            will_retry=bool(pkt.get("will_retry", False)),
                        )

                    elif pkt_type == "done":
                        logger.debug("TTS stream done.")
                        break

                    elif pkt_type == "error":
                        logger.error("TTS server error: %s", pkt.get("message"))
                        break

                elif isinstance(msg, bytes) and expecting_binary and current_header:
                    yield TTSAudioChunk(
                        index=current_header["index"],
                        num_samples=current_header["num_samples"],
                        pcm_bytes=msg,
                        sample_rate=sample_rate,
                    )
                    expecting_binary = False
                    current_header = None

    except websockets.ConnectionClosed:
        logger.warning("TTS WebSocket closed unexpectedly.")
    except ConnectionRefusedError:
        logger.error("Cannot connect to TTS server at %s", url)
    except Exception:
        logger.exception("TTS client error.")


async def fetch_voices() -> list[dict]:
    """Fetch the voice list from the TTS server's REST API."""
    import httpx
    url = f"{config.TTS_REST_URL}/voices"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Failed to fetch voices: HTTP %d", resp.status_code)
            return []
    except Exception:
        logger.warning("Failed to fetch voices from %s", url)
        return []
