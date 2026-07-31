"""
OpenAI speech-to-text: streaming (gpt-realtime-whisper) and batch (4o transcribe).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from typing import Awaitable, Callable, List, Optional

import openai
import websockets
from websockets.exceptions import ConnectionClosed

import config

logger = logging.getLogger(__name__)

RealtimeDeltaCallback = Callable[[str], Awaitable[None]]

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


def _require_api_key() -> None:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI STT models.")


def _get_openai() -> openai.AsyncOpenAI:
    _require_api_key()
    return openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)


def pcm16_to_wav_bytes(chunks: List[bytes], sample_rate: int) -> bytes:
    pcm = b"".join(chunks)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class RealtimeTranscriber:
    """
    Streaming transcription via OpenAI Realtime API (gpt-realtime-whisper).

    Audio is buffered locally during recording. On finish(), all buffered audio
    is flushed to the API, committed, and we wait for the final transcript.
    Deltas are emitted as the API processes the audio.
    """

    def __init__(
        self,
        language: str = "en",
        delay: str = "low",
        on_delta: Optional[RealtimeDeltaCallback] = None,
        on_vad_stopped: Optional[Callable[[], Awaitable[None]]] = None,
        server_vad: bool = True,
        vad_silence_ms: int = 500,
    ):
        self.language = language
        self.delay = delay
        self.on_delta = on_delta
        self.on_vad_stopped = on_vad_stopped
        self.server_vad = server_vad
        self.vad_silence_ms = vad_silence_ms
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._final_transcript = ""
        self._completed = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._configured = asyncio.Event()
        self._error: Optional[str] = None
        self._closed = False
        self._audio_buffer: List[bytes] = []

    async def start(self) -> None:
        _require_api_key()
        self._ws = await websockets.connect(
            REALTIME_WS_URL,
            additional_headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            },
            max_size=16 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        )
        self._listener_task = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            await self.close()
            raise RuntimeError("Timed out waiting for OpenAI Realtime session.")
        if self._error:
            err = self._error
            await self.close()
            raise RuntimeError(err)
        await self._send_session_update()
        try:
            await asyncio.wait_for(self._configured.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.debug("session.updated not received within 5s; proceeding.")
        if self._error:
            err = self._error
            await self.close()
            raise RuntimeError(err)
        logger.info("Realtime STT session ready (lang=%s, delay=%s).", self.language, self.delay)

    async def _send_session_update(self) -> None:
        assert self._ws is not None
        payload = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": config.MIC_SAMPLE_RATE},
                        "turn_detection": None,
                        "transcription": {
                            "model": "gpt-realtime-whisper",
                            "language": self.language,
                        },
                    },
                },
            },
        }
        await self._ws.send(json.dumps(payload))

    async def append(self, pcm16_bytes: bytes) -> None:
        """Send audio directly to the API for real-time processing."""
        if not pcm16_bytes or self._closed or self._ws is None:
            return
        self._audio_buffer.append(pcm16_bytes)
        try:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16_bytes).decode("ascii"),
            }))
        except Exception as exc:
            logger.debug("Failed to send audio chunk: %s", exc)

    async def finish(self, timeout_s: float = 15.0, already_committed: bool = False) -> str:
        """Commit audio buffer and wait for final transcript.
        
        If already_committed=True (server VAD triggered), skip sending the commit.
        """
        if self._closed or self._ws is None:
            return self._final_transcript

        total_bytes = sum(len(c) for c in self._audio_buffer)
        total_ms = (total_bytes / 2) / config.MIC_SAMPLE_RATE * 1000
        logger.info("Realtime STT: finishing (%.1f ms audio, committed=%s).", total_ms, already_committed)

        self._audio_buffer.clear()

        if not already_committed:
            if total_ms < 100:
                logger.warning("Audio too short (%.1f ms); skipping commit.", total_ms)
                return ""
            try:
                self._completed.clear()
                self._error = None
                self._final_transcript = ""
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as exc:
                logger.error("Failed to commit audio: %s", exc)
                return ""

        try:
            await asyncio.wait_for(self._completed.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning("Realtime STT commit timed out after %.1f s.", timeout_s)
        if self._error:
            raise RuntimeError(self._error)
        return self._final_transcript.strip()

    def reset(self) -> None:
        """Reset state for reuse across utterances (keeps connection alive)."""
        self._final_transcript = ""
        self._completed.clear()
        self._error = None
        self._audio_buffer.clear()

    @property
    def is_alive(self) -> bool:
        return not self._closed and self._ws is not None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._audio_buffer.clear()
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    @staticmethod
    def _error_message(event: dict) -> str:
        err = event.get("error")
        if isinstance(err, dict):
            return err.get("message") or str(err)
        return event.get("message") or str(event)

    async def _listen(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                event = json.loads(raw)
                etype = event.get("type", "")

                if etype in ("session.created", "transcription_session.created"):
                    self._session_ready.set()
                elif etype == "session.updated":
                    self._configured.set()

                elif etype == "input_audio_buffer.committed":
                    logger.debug("Server VAD committed audio buffer.")
                    if self.on_vad_stopped:
                        await self.on_vad_stopped()

                elif etype == "input_audio_buffer.speech_started":
                    logger.debug("Server VAD: speech started.")

                elif etype == "input_audio_buffer.speech_stopped":
                    logger.debug("Server VAD: speech stopped.")

                elif etype == "conversation.item.input_audio_transcription.delta":
                    delta = event.get("delta") or ""
                    if delta and self.on_delta:
                        await self.on_delta(delta)

                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = (event.get("transcript") or "").strip()
                    if transcript:
                        self._final_transcript = transcript
                    self._completed.set()

                elif etype == "error":
                    self._error = self._error_message(event)
                    logger.error("Realtime STT error: %s", self._error)
                    self._session_ready.set()
                    self._configured.set()
                    self._completed.set()

        except ConnectionClosed as exc:
            if not self._completed.is_set():
                self._error = f"WebSocket closed: {exc}"
                self._completed.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Realtime STT listener failed.")
            self._error = str(exc)
            self._completed.set()


async def transcribe_batch(
    pcm_chunks: List[bytes],
    model: str,
    language: str,
    sample_rate: int = 24000,
) -> str:
    """Batch transcription via OpenAI Audio API (gpt-4o-transcribe, etc.)."""
    _require_api_key()
    if not pcm_chunks:
        return ""

    wav_bytes = pcm16_to_wav_bytes(pcm_chunks, sample_rate)
    client = _get_openai()
    response = await client.audio.transcriptions.create(
        model=model,
        file=("audio.wav", wav_bytes, "audio/wav"),
        language=language,
        response_format="text",
    )
    if isinstance(response, str):
        return response.strip()
    text = getattr(response, "text", None)
    return (text or "").strip()
