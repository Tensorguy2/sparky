"""
v3 TTS server — faster-qwen3-tts engine, identical /ws/tts wire protocol to v2.

Runs in the dedicated venv:  venv-tts-v3 (faster-qwen3-tts, torch 2.12)
Default port: 25568 (v2 stays on 25567).

    cd src && ../venv-tts-v3/bin/python v3_server.py --port 25568

Wire protocol (unchanged from routers/v2_tts.py, so chatbot's tts_client_v2
works as-is):
  client -> {"text", "language", "voice_id"}
  server -> metadata | sample_rate_correction | chunk_header + <binary f32le>
            | instability | done | error

Voices are read from the existing src/voices.db + src/voices/*.wav
(read-only); clone prompts are recomputed with the new engine at startup.
"""

import argparse
import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.v3_model_service import v3_model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-20s : %(message)s",
)
logger = logging.getLogger("v3_server")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR.parent / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
DB_PATH = str(BASE_DIR / "voices.db")
DEFAULT_PORT = 25568

_MAX_RETRIES = 2
_SENTENCE_TIMEOUT_S = 15.0

# Prefer shared helpers (long-sentence split + richer instability heuristics).
try:
    from utils.audio_quality import is_unstable as _is_unstable
    from utils.text import split_into_sentences
except ImportError:  # pragma: no cover — fallback if launched outside src/
    _SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

    def split_into_sentences(text: str, min_chars: int = 40) -> List[str]:
        raw = _SENTENCE_RE.split(text.strip())
        merged: List[str] = []
        buf = ""
        for part in raw:
            buf = f"{buf} {part}".strip() if buf else part
            if len(buf) >= min_chars:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] = f"{merged[-1]} {buf}"
            else:
                merged.append(buf)
        return merged or [text]

    def _is_unstable(chunk: np.ndarray) -> Optional[str]:
        if len(chunk) == 0:
            return None
        if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
            return "nan_or_inf"
        clipped = float(np.mean(np.abs(chunk) > 0.98))
        if clipped > 0.15:
            return f"clipping_{clipped:.0%}"
        return None


# -- Latency metrics --------------------------------------------------------------

_metrics_lock = threading.Lock()
_ttfa_history: deque = deque(maxlen=200)
_rtf_history: deque = deque(maxlen=200)
_request_count = 0
_error_count = 0


def _record(ttfa: float, rtf: float) -> None:
    global _request_count
    with _metrics_lock:
        _ttfa_history.append(ttfa)
        _rtf_history.append(rtf)
        _request_count += 1


def _stats() -> dict:
    def pct(data):
        if not data:
            return {"p50": 0, "p95": 0, "p99": 0}
        s = sorted(data)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.5)], 4),
            "p95": round(s[min(int(n * 0.95), n - 1)], 4),
            "p99": round(s[min(int(n * 0.99), n - 1)], 4),
        }

    with _metrics_lock:
        return {
            "request_count": _request_count,
            "error_count": _error_count,
            "ttfa_seconds": pct(_ttfa_history),
            "rtf": pct(_rtf_history),
            "cache": v3_model_service.cache.stats(),
        }


# -- App ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, v3_model_service.load_model, MODEL_PATH)
    await loop.run_in_executor(None, v3_model_service.load_voices, DB_PATH)
    await loop.run_in_executor(None, v3_model_service.warmup)
    logger.info(
        "=== v3 Server ready in %.2f s | voices=%s ===",
        time.perf_counter() - t0, list(v3_model_service.voices),
    )
    yield


app = FastAPI(title="Qwen3-TTS v3 (faster-qwen3-tts)", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engine": "faster-qwen3-tts",
        "model_loaded": v3_model_service.model is not None,
        "voices": len(v3_model_service.voices),
        "cache": v3_model_service.cache.stats(),
    }


@app.get("/voices")
async def list_voices() -> list[dict]:
    return [
        {
            "voice_id": v.voice_id,
            "name": v.name,
            "builtin": v.builtin,
            "prompt_ready": v.prompt is not None,
        }
        for v in v3_model_service.voices.values()
    ]


@app.get("/v2/stats")
async def stats() -> dict:
    return _stats()


@app.get("/cache/stats")
async def cache_stats() -> dict:
    return v3_model_service.cache.stats()


@app.post("/cache/clear")
async def cache_clear() -> dict:
    return {"cleared": v3_model_service.cache.clear()}


async def _send_error(ws: WebSocket, conn_id: str, message: str) -> None:
    logger.warning("[%s] error -> client: %s", conn_id, message)
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}))
        await ws.close()
    except Exception:
        pass


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket) -> None:
    conn_id = uuid.uuid4().hex[:8]
    t_connect = time.perf_counter()
    await ws.accept()

    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        req: dict = json.loads(raw)
    except asyncio.TimeoutError:
        await _send_error(ws, conn_id, "Request timed out (10 s)")
        return
    except Exception as exc:
        await _send_error(ws, conn_id, f"Invalid JSON: {exc}")
        return

    text: str = req.get("text", "").strip()
    language: str = req.get("language", "English")
    voice_id: Optional[str] = req.get("voice_id")

    if not text:
        await _send_error(ws, conn_id, "'text' field is required")
        return

    prompt = None
    if voice_id:
        entry = v3_model_service.voices.get(voice_id)
        if entry is None:
            await _send_error(
                ws, conn_id,
                f"Voice '{voice_id}' not found. Available: {list(v3_model_service.voices)}",
            )
            return
        prompt = entry.prompt

    sentences = split_into_sentences(text)
    logger.info("[%s] voice=%r lang=%s %d sentence(s) text=%r",
                conn_id, voice_id, language, len(sentences), text[:80])

    await ws.send_text(json.dumps({
        "type": "metadata",
        "sample_rate": 24000,
        "channels": 1,
        "encoding": "pcm_f32le",
        "num_sentences": len(sentences),
    }))

    t_first_audio: Optional[float] = None
    notified_sr = False
    total_audio_s = 0.0

    async def send_chunk(idx: int, audio: np.ndarray, sr: int) -> None:
        nonlocal t_first_audio, notified_sr, total_audio_s
        if not notified_sr and sr != 24000:
            notified_sr = True
            await ws.send_text(json.dumps({
                "type": "sample_rate_correction", "sample_rate": sr,
            }))
        # Hard safety: nothing above full-scale leaves the server.
        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = audio.tobytes()
        await ws.send_text(json.dumps({
            "type": "chunk_header",
            "index": idx,
            "num_samples": len(audio),
            "size_bytes": len(pcm),
        }))
        await ws.send_bytes(pcm)
        total_audio_s += len(audio) / sr
        if t_first_audio is None:
            t_first_audio = time.perf_counter() - t_connect

    for idx, sentence in enumerate(sentences):
        tag = f"[{conn_id}] [{idx + 1}/{len(sentences)}]"

        cached = v3_model_service.cache.get(sentence, voice_id, language)
        if cached is not None:
            audio, sr = cached
            logger.info("%s CACHE HIT (%d samples)", tag, len(audio))
            try:
                await send_chunk(idx, audio, sr)
            except Exception:
                return
            continue

        for attempt in range(_MAX_RETRIES + 1):
            sentence_chunks: List[np.ndarray] = []
            sentence_sr = 24000
            unstable_reason: Optional[str] = None
            timed_out = False
            deadline = time.perf_counter() + _SENTENCE_TIMEOUT_S

            try:
                async for chunk, sr in v3_model_service.stream_sentence(
                    sentence, language, prompt,
                ):
                    if time.perf_counter() > deadline:
                        timed_out = True
                        break
                    if len(chunk) == 0:
                        continue
                    reason = _is_unstable(chunk)
                    if reason:
                        unstable_reason = reason
                        # Mute this slice so the blast never plays; keep timeline.
                        try:
                            await send_chunk(idx, np.zeros_like(chunk), sr)
                        except Exception:
                            return
                        break
                    sentence_sr = sr
                    await send_chunk(idx, chunk, sr)
                    sentence_chunks.append(chunk)
            except WebSocketDisconnect:
                logger.info("[%s] client disconnected at sentence %d", conn_id, idx)
                return
            except Exception as exc:
                logger.exception("[%s] inference error at sentence %d", conn_id, idx)
                global _error_count
                with _metrics_lock:
                    _error_count += 1
                await _send_error(ws, conn_id, f"Inference error on chunk {idx}: {exc}")
                return

            if timed_out or unstable_reason:
                # Anti-stutter: only restart if the listener has not already
                # heard good audio from this sentence.
                already_streamed = len(sentence_chunks) > 0
                will_retry = (attempt < _MAX_RETRIES) and not already_streamed
                reason = "timeout" if timed_out else unstable_reason
                if already_streamed:
                    logger.warning(
                        "%s %s on attempt %d — muted; skip restart (already streamed)",
                        tag, reason, attempt + 1,
                    )
                else:
                    logger.warning(
                        "%s %s on attempt %d — %s",
                        tag, reason, attempt + 1,
                        "retrying" if will_retry else "giving up",
                    )
                try:
                    await ws.send_text(json.dumps({
                        "type": "instability",
                        "sentence_index": idx,
                        "reason": reason,
                        "will_retry": will_retry,
                    }))
                except Exception:
                    return
                if will_retry:
                    await asyncio.sleep(0.1)
                    continue
                break

            if sentence_chunks:
                v3_model_service.cache.put(
                    sentence, voice_id, language,
                    np.concatenate(sentence_chunks), sentence_sr,
                )
            break

    await ws.send_text(json.dumps({"type": "done"}))
    try:
        await ws.close()
    except Exception:
        pass

    wall = time.perf_counter() - t_connect
    rtf = wall / total_audio_s if total_audio_s else 0
    ttfa = t_first_audio if t_first_audio else wall
    _record(ttfa, rtf)
    logger.info("[%s] v3 Done | wall=%.2f s audio=%.2f s RTF=%.3f TTFA=%.3f s",
                conn_id, wall, total_audio_s, rtf, ttfa)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, ws_ping_interval=20,
                ws_ping_timeout=30)
