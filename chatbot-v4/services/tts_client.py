"""
Async WebSocket client for the Qwen3-TTS server.

Uses a process-wide pooled multi-request connection so instruct-capable
sentences do not pay a fresh TCP/WS handshake each time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Union

import websockets
from websockets.exceptions import ConnectionClosed

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
    # Drop pooled socket so the next request reconnects cleanly.
    await _pool.reset()


class _TTSPool:
    """One shared multi-request WS; serialized so utterances don't interleave."""

    def __init__(self) -> None:
        self._ws = None
        self._lock = asyncio.Lock()

    async def reset(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

    async def _ensure_unlocked(self):
        if self._ws is not None:
            return self._ws
        url = config.TTS_WS_URL
        logger.info("TTS pool connecting | url=%s", url)
        self._ws = await websockets.connect(url, max_size=16 * 1024 * 1024)
        return self._ws

    async def stream(
        self,
        text: str,
        voice_id: str,
        language: str,
        instruct: str,
        turn_id: str,
    ) -> AsyncGenerator[TTSEvent, None]:
        async with self._lock:
            for attempt in range(2):
                try:
                    ws = await self._ensure_unlocked()
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
                                return

                            elif pkt_type == "error":
                                logger.error("TTS server error: %s", pkt.get("message"))
                                await self._close_unlocked()
                                return

                        elif isinstance(msg, bytes) and expecting_binary and current_header:
                            yield TTSAudioChunk(
                                index=current_header["index"],
                                num_samples=current_header["num_samples"],
                                pcm_bytes=msg,
                                sample_rate=sample_rate,
                            )
                            expecting_binary = False
                            current_header = None
                    # Socket ended without done — reconnect once.
                    await self._close_unlocked()
                except ConnectionClosed:
                    logger.warning("TTS pooled WS closed; retry=%s", attempt == 0)
                    await self._close_unlocked()
                    if attempt == 0:
                        continue
                    return
                except ConnectionRefusedError:
                    logger.error("Cannot connect to TTS server at %s", config.TTS_WS_URL)
                    await self._close_unlocked()
                    return
                except Exception:
                    logger.exception("TTS client error.")
                    await self._close_unlocked()
                    return


_pool = _TTSPool()


async def stream_tts(
    text: str,
    voice_id: str = "",
    language: str = "English",
    instruct: str = "",
    turn_id: str = "",
) -> AsyncGenerator[TTSEvent, None]:
    """
    Synthesize text via the pooled multi-request TTS socket.

    Each chunk contains raw PCM float32 LE bytes at 24 kHz mono.
    Optional ``instruct`` is natural-language delivery guidance for Qwen TTS.
    """
    voice_id = voice_id or config.DEFAULT_VOICE_ID
    instruct = (instruct or "").strip()

    logger.info(
        "TTS request | voice=%s lang=%s text_len=%d instruct=%r pooled=1",
        voice_id, language, len(text), instruct[:80] if instruct else "",
    )

    async for event in _pool.stream(
        text=text,
        voice_id=voice_id,
        language=language,
        instruct=instruct,
        turn_id=turn_id,
    ):
        yield event


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


async def shutdown_v2_client() -> None:
    """Compatibility shim — close the pooled socket on process shutdown."""
    await _pool.reset()
