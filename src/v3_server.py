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
import os
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

# Truncation guard. The clone intermittently returns audio far too short for the
# text; holding the opening slices lets a bad take be retried before the
# listener hears the blip.
#
# Nothing is held once this much audio has accumulated. The engine emits 0.64 s
# chunks, so any cap below that clears on the first chunk and costs no
# first-audio latency (measured: 524 ms with the guard vs 521 ms without).
# Raising it past 0.64 would make every fragment wait for a second chunk, which
# arrives ~780 ms later. Fragments longer than ~20 characters can therefore
# still slip a marginally-short take through; severe truncation, which is what
# is actually audible, lands well under the cap.
_DURATION_GUARD = os.getenv("TTS_DURATION_GUARD", "true").lower() in ("1", "true", "yes")
_GUARD_CAP_S = float(os.getenv("TTS_GUARD_CAP_S", "0.60"))

# Truncation gets one attempt more than the other failure modes. A rejected take
# is caught at the moment generation ends, so retrying costs roughly 300 ms,
# where a timeout retry costs up to _SENTENCE_TIMEOUT_S. Truncation is also
# stochastic -- the same text regenerated is usually fine -- so the extra attempt
# converts most of the residual into clean audio.
_TRUNC_EXTRA_RETRIES = 1

# Speaker lock. Every sentence of a reply is a separate generation conditioned on
# the voice's reference recording, and nothing carries the speaker that was
# actually rendered -- so each sentence re-rolls the identity, which callers hear
# as the voice changing mid-reply. Conditioning later sentences on the first
# one's own audio pins them to it.
#
# Scoped by the turn id the client sends, so concurrent callers never share a
# lock and each new turn starts from the reference again.
_SPEAKER_LOCK = os.getenv("TTS_SPEAKER_LOCK", "true").lower() in ("1", "true", "yes")
_LOCK_TTL_S = 120.0
_LOCK_MAX = 64

# (voice_id, turn_id) -> (created_at, clone prompt)
_speaker_locks: dict[tuple[str, str], tuple[float, object]] = {}
_locks_guard = threading.Lock()


def _lock_get(key: tuple[str, str]) -> Optional[object]:
    now = time.time()
    with _locks_guard:
        for k, (at, _) in list(_speaker_locks.items()):
            if now - at > _LOCK_TTL_S:
                _speaker_locks.pop(k, None)
        entry = _speaker_locks.get(key)
        return entry[1] if entry else None


def _lock_put(key: tuple[str, str], prompt: object) -> None:
    with _locks_guard:
        if len(_speaker_locks) >= _LOCK_MAX:
            oldest = min(_speaker_locks, key=lambda k: _speaker_locks[k][0])
            _speaker_locks.pop(oldest, None)
        _speaker_locks[key] = (time.time(), prompt)

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

try:
    from utils.voice_guard import (
        check_duration,
        expected_min_seconds,
        needs_terminal_punctuation,
    )
except ImportError:  # pragma: no cover — guard is optional
    _DURATION_GUARD = False

    def check_duration(text: str, audio_seconds: float) -> Optional[str]:
        return None

    def expected_min_seconds(text: str) -> float:
        return 0.0

    def needs_terminal_punctuation(text: str) -> bool:
        return False


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
    # Optional: groups the fragments of one reply so they can share a speaker.
    # Absent (e.g. the server-side TTS path) means no lock, i.e. prior behaviour.
    turn_id: Optional[str] = req.get("turn_id")

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

    # A lock already exists once an earlier fragment of this turn has been
    # spoken; later fragments inherit the speaker it actually produced.
    lock_key = (voice_id, turn_id) if (_SPEAKER_LOCK and voice_id and turn_id) else None
    locked_prompt = _lock_get(lock_key) if lock_key else None
    if locked_prompt is not None:
        prompt = locked_prompt
        logger.info("[%s] speaker lock applied (turn=%s)", conn_id, turn_id)

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

    # Once a speaker is locked, cached audio cannot be served: it was rendered
    # under different conditioning, so replaying it is exactly the mid-reply
    # voice change the lock exists to prevent. Only unlocked (first) sentences
    # use the cache, which is where the repeated greetings and fillers live.
    locked = locked_prompt is not None

    for idx, sentence in enumerate(sentences):
        tag = f"[{conn_id}] [{idx + 1}/{len(sentences)}]"

        cached = None if locked else v3_model_service.cache.get(sentence, voice_id, language)
        if cached is not None:
            audio, sr = cached
            logger.info("%s CACHE HIT (%d samples)", tag, len(audio))
            try:
                await send_chunk(idx, audio, sr)
            except Exception:
                return
            # Lock to the cached audio as well. It is what the listener just
            # heard, so the rest of the turn has to match it -- and a repeated
            # greeting is the most likely first sentence to be a cache hit.
            if lock_key and not locked:
                carry = await v3_model_service.build_carryover_prompt(audio, sr, sentence)
                if carry is not None:
                    _lock_put(lock_key, carry)
                    prompt = carry
                    locked = True
                    logger.info("%s speaker locked from cache for turn=%s", tag, turn_id)
            continue

        # Text ending mid-clause truncates far more often -- "Here is the answer"
        # came back at 0.16 s every time, where the same text with a period was
        # fine. Cheaper to terminate it than to retry the take.
        gen_text = sentence
        if _DURATION_GUARD and needs_terminal_punctuation(sentence):
            gen_text = sentence.rstrip() + "."

        # Audio is withheld until this much has accumulated, at which point the
        # take is too long to be a truncation and streaming proceeds live. The
        # opening slices normally clear it immediately, so the common path pays
        # nothing.
        gate_s = min(expected_min_seconds(sentence), _GUARD_CAP_S) if _DURATION_GUARD else 0.0

        # Longest take seen for this sentence. If every attempt is rejected the
        # words still have to be spoken, so the best one is played rather than
        # dropping the fragment and leaving a hole in the reply.
        best_take: List[np.ndarray] = []
        best_take_s = 0.0

        for attempt in range(_MAX_RETRIES + _TRUNC_EXTRA_RETRIES + 1):
            sentence_chunks: List[np.ndarray] = []
            sentence_sr = 24000
            unstable_reason: Optional[str] = None
            timed_out = False
            deadline = time.perf_counter() + _SENTENCE_TIMEOUT_S
            held: List[np.ndarray] = []
            held_s = 0.0
            flushed = gate_s <= 0.0

            async def flush_held(sr: int) -> None:
                nonlocal held, held_s, flushed
                for pending in held:
                    await send_chunk(idx, pending, sr)
                held = []
                held_s = 0.0
                flushed = True

            try:
                async for chunk, sr in v3_model_service.stream_sentence(
                    gen_text, language, prompt,
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
                        # Nothing is emitted while still holding, since a retry
                        # will replace the whole sentence.
                        if flushed:
                            try:
                                await send_chunk(idx, np.zeros_like(chunk), sr)
                            except Exception:
                                return
                        break
                    sentence_sr = sr
                    sentence_chunks.append(chunk)
                    if flushed:
                        await send_chunk(idx, chunk, sr)
                    else:
                        held.append(chunk)
                        held_s += len(chunk) / sr
                        if held_s >= gate_s:
                            await flush_held(sr)

                if not timed_out and not unstable_reason and not flushed:
                    # Generation ended with everything still held, so the take is
                    # shorter than the text can account for.
                    unstable_reason = check_duration(sentence, held_s)
                    if not unstable_reason:
                        await flush_held(sentence_sr)
                    elif held_s > best_take_s:
                        best_take, best_take_s = list(held), held_s
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
                already_streamed = flushed
                reason = "timeout" if timed_out else unstable_reason
                budget = _MAX_RETRIES + (
                    _TRUNC_EXTRA_RETRIES if str(reason).startswith("truncated") else 0
                )
                will_retry = (attempt < budget) and not already_streamed
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
                # Out of retries. Play the longest take rather than nothing, so
                # the reply never loses words; a short take is still better than
                # a silent gap.
                if not flushed:
                    if held_s >= best_take_s:
                        best_take, best_take_s = list(held), held_s
                    if best_take:
                        logger.warning(
                            "%s retries exhausted (%s) — playing best take (%.2f s)",
                            tag, reason, best_take_s,
                        )
                        held, held_s = best_take, best_take_s
                        await flush_held(sentence_sr)
                break

            if sentence_chunks:
                spoken = np.concatenate(sentence_chunks)
                if not locked:
                    v3_model_service.cache.put(
                        sentence, voice_id, language, spoken, sentence_sr,
                    )
                # Pin the rest of this turn to the speaker just rendered. Built
                # after the audio is already streaming, so it never delays what
                # the caller hears.
                if lock_key and not locked:
                    carry = await v3_model_service.build_carryover_prompt(
                        spoken, sentence_sr, sentence,
                    )
                    if carry is not None:
                        _lock_put(lock_key, carry)
                        prompt = carry
                        locked = True
                        logger.info("%s speaker locked for turn=%s", tag, turn_id)
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
