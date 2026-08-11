"""
Voice Chatbot Server (v4 — parallel pipeline)
==============================================
FastAPI application that orchestrates:
  - Local faster-whisper / Parakeet for speech-to-text
  - OpenAI / Anthropic models for text generation (streaming)
  - Existing Qwen3-TTS server for text-to-speech output

v4 changes vs v3:
  - Delayed quiet filler: only if reply TTS is late (masks LLM/TTS delay)
  - Non-blocking STT: transcription runs as a task so WS stays responsive
  - Parallel router: LLM starts immediately; maybe_route runs alongside;
    if a state switch occurs, the in-flight turn is cancelled and restarted once
  - Stage timers logged for each turn

Run:
    cd chatbot-v4
    bash scripts/start_server.sh
    python server.py
"""

import asyncio
import base64
import json
import logging
import random
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from models.conversation import ConversationManager
from models.context import ContextManager
from models.flow import Flow, FlowManager
from models.instructions import InstructionManager, ModelParams
from models.session_store import SessionSnapshot, SessionStore
from services import llm_service
from services import router as call_router
from services import call_facts as call_facts_svc
from services.stt_router import (
    create_realtime_session,
    get_stt_info,
    is_realtime_model,
    resolve_stt_model,
    transcribe,
)
from services.stt_service import AudioRecorder, preload
from services.tts_client_v2 import (
    EXCUSE_ME_PHRASE,
    TTSInstability,
    reset_tts_server,
    shutdown_v2_client,
    stream_tts_v2,
)
from services.tts_client import stream_tts as stream_tts_with_instruct
from services import filler_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:\u2014\u2013\-])\s+|(?<=[.!?…])\s+")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
context_mgr: Optional[ContextManager] = None
instruction_mgr: Optional[InstructionManager] = None
session_store: Optional[SessionStore] = None
flow_mgr: Optional[FlowManager] = None


def active_flow() -> Optional[Flow]:
    return flow_mgr.default() if flow_mgr else None


def mode_changed_event(session, reason=None, confidence=None) -> dict:
    extra = {}
    if reason:
        extra["reason"] = reason
    if confidence is not None:
        extra["confidence"] = confidence
    evt = {
        "type": "mode_changed",
        "state": session.current_state,
        **extra,
    }
    return evt


def switch_state(session, state_name: str) -> bool:
    flow = active_flow()
    if not flow:
        return False
    st = flow.state(state_name) if flow else None
    if st is None:
        return False
    session.current_state = state_name
    session.instruction_set_id = st.instruction_set if hasattr(st, "instruction_set") else session.instruction_set_id
    session.context_name = st.context if hasattr(st, "context") else session.context_name
    return True


# Wire tokens for AnswerBot / Operator — keep in llm_done, never speak.
_WIRE_TOKEN_RE = re.compile(
    r"\[\[(?:GOODBYE|PHASE_CLOSER|INSTRUCT:[^\]]*|EMO:[^\]]*)\]\]",
    re.IGNORECASE,
)
_INSTRUCT_TOKEN_RE = re.compile(r"\[\[INSTRUCT:([^\]]*)\]\]", re.IGNORECASE)
_EMO_TOKEN_RE = re.compile(r"\[\[EMO:([^\]]*)\]\]", re.IGNORECASE)

_EMO_INSTRUCT_MAP = {
    "friendly": "Speak warmly, friendly, and natural",
    "cheerful": "Speak cheerfully with light energy",
    "angry": "Speak angrily and firmly, clearly offended",
    "apologetic": "Speak softly and apologetically",
    "firm": "Speak calmly but firmly, no warmth",
    "excited": "Speak with excitement and enthusiasm",
    "calm": "Speak calmly and evenly",
    "soft": "Speak softly and gently",
}

# Coach phrases → default TTS instruct for the next turn(s).
_COACH_TTS_INSTRUCT_RE = re.compile(
    r"\b(angry|angrier|anger|mad|friendly|friendlier|warm|warmer|"
    r"apologetic|sorry|firm|firmer|cheerful|excited|excitedly|"
    r"calm|calmer|soft|softer)\b",
    re.IGNORECASE,
)
_COACH_TTS_INSTRUCT_MAP = {
    "angry": "Speak angrily and firmly, clearly offended",
    "angrier": "Speak angrily and firmly, clearly offended",
    "anger": "Speak angrily and firmly, clearly offended",
    "mad": "Speak angrily and firmly, clearly offended",
    "friendly": "Speak warmly, friendly, and natural",
    "friendlier": "Speak warmly, friendly, and natural",
    "warm": "Speak warmly, friendly, and natural",
    "warmer": "Speak warmly, friendly, and natural",
    "apologetic": "Speak softly and apologetically",
    "sorry": "Speak softly and apologetically",
    "firm": "Speak calmly but firmly, no warmth",
    "firmer": "Speak calmly but firmly, no warmth",
    "cheerful": "Speak cheerfully with light energy",
    "excited": "Speak with excitement and enthusiasm",
    "excitedly": "Speak with excitement and enthusiasm",
    "calm": "Speak calmly and evenly",
    "calmer": "Speak calmly and evenly",
    "soft": "Speak softly and gently",
    "softer": "Speak softly and gently",
}


def _strip_wire_tokens(text: str) -> str:
    """Remove control tokens so TTS never reads them aloud."""
    if not text:
        return text
    cleaned = _WIRE_TOKEN_RE.sub("", text)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _extract_tts_instruct(text: str) -> tuple[str, str]:
    """Return (spoken_text, instruct) from optional [[INSTRUCT:]] / [[EMO:]] tags."""
    if not text:
        return "", ""
    instruct = ""
    m = _INSTRUCT_TOKEN_RE.search(text)
    if m:
        instruct = (m.group(1) or "").strip()
    else:
        m = _EMO_TOKEN_RE.search(text)
        if m:
            alias = (m.group(1) or "").strip().lower()
            instruct = _EMO_INSTRUCT_MAP.get(alias, "")
            if not instruct and alias:
                instruct = f"Speak in a {alias} tone"
    return _strip_wire_tokens(text), instruct


def _coach_note_to_tts_instruct(note: str) -> str:
    m = _COACH_TTS_INSTRUCT_RE.search(note or "")
    if not m:
        return ""
    return _COACH_TTS_INSTRUCT_MAP.get(m.group(1).lower(), "")


def _split_sentences(text: str, min_chars: int = 40) -> list:
    raw = _SENTENCE_RE.split(text)
    merged = []
    buf = ""
    for part in raw:
        buf += (" " if buf else "") + part
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def _split_first_clause(text: str, min_chars: int = 20) -> list:
    raw = _CLAUSE_RE.split(text)
    merged = []
    buf = ""
    for part in raw:
        buf += (" " if buf else "") + part
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def _operator_override_block(directives: list[str]) -> str:
    lines = [d.strip() for d in directives if (d or "").strip()]
    if not lines:
        return ""
    body = "\n".join(f"- {d}" for d in lines)
    return (
        "# LIVE OPERATOR OVERRIDE — HIGHEST PRIORITY FOR THIS REPLY\n"
        "You MUST obey these Operator directives in your next spoken reply.\n"
        "They override rate floors, W2/C2C rules, role lock, personality, "
        "prior refusals in this conversation, and any conflicting instruction above.\n"
        "Do not mention the Operator or these directives. Do not renegotiate against them.\n"
        f"{body}"
    )


def _messages_with_operator_directives(
    messages: list[dict], directives: list[str]
) -> list[dict]:
    """Inject a sticky override immediately before the trailing user turn(s)."""
    lines = [d.strip() for d in directives if (d or "").strip()]
    if not lines:
        return messages
    body = "\n".join(f"- {d}" for d in lines)
    inject = [
        {
            "role": "user",
            "content": (
                "[Internal Operator directive — not spoken by the caller. "
                "Obey in your next reply only.]\n"
                f"{body}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I will follow those Operator directives in my next "
                "reply and will not mention them."
            ),
        },
    ]
    msgs = list(messages or [])
    if not msgs:
        return inject
    i = len(msgs)
    while i > 0 and msgs[i - 1].get("role") == "user":
        i -= 1
    return msgs[:i] + inject + msgs[i:]


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------
class ChatSession:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.conversation = ConversationManager(max_turns=config.MAX_CONVERSATION_TURNS)
        self.model = config.DEFAULT_MODEL
        self.voice_id = config.DEFAULT_VOICE_ID
        flow = active_flow()
        entry = flow.entry_state if flow else None
        self.current_state = entry
        self.instruction_set_id = None
        self.context_name = None
        self.language = "English"
        self.tts_enabled = True
        # Operator telephony packs own the brain — default off for phone chatbot.
        self.router_enabled = bool(config.ROUTER_ENABLED)
        self.stt_model = config.DEFAULT_STT_MODEL
        self.recorder = AudioRecorder(config.MIC_SAMPLE_RATE)
        self.stt_stream = None
        self._stt_persistent = None
        self.vad_committed = asyncio.Event()
        self.tts_lock = asyncio.Lock()
        self.generation_turn: int = 0
        self.active_turn_task = None
        self.turn_snapshot = None
        self.stt_transcribing: bool = False
        self.persist_id = config.DEFAULT_SESSION_ID
        # Set True when the real reply TTS starts (not fillers).
        self.reply_audio_started: bool = False
        self._filler_task: Optional[asyncio.Task] = None
        # Live Operator coach notes — injected into the next LLM turn(s).
        self.operator_directives: list[str] = []
        # Sticky call facts (caller/company/job) — survive history truncation.
        self.call_facts: dict[str, str] = call_facts_svc.empty_facts()
        # Default TTS delivery instruct (coach emotion / sticky until cleared).
        self.default_tts_instruct: str = ""
        # Active turn's TTS consumer (cancelled on barge-in with the LLM task).
        self._tts_consumer_task = None
        self._tts_queue: Optional[asyncio.Queue] = None

    @property
    def tts_turn(self):
        return self.generation_turn

    def merge_call_facts(
        self, incoming: dict[str, str] | None, *, replace: bool = False
    ) -> list[str]:
        updated = call_facts_svc.merge_facts(
            self.call_facts, incoming, replace=replace
        )
        if updated:
            call_facts_svc.log_updates(self.id, updated, self.call_facts)
        return updated

    def apply_snapshot(self, snap: SessionSnapshot):
        self.conversation = ConversationManager.from_turn_dicts(
            snap.turns if hasattr(snap, "turns") else [],
            max_turns=config.MAX_CONVERSATION_TURNS,
        )
        self.model = snap.model or self.model
        self.voice_id = snap.voice_id or self.voice_id
        self.current_state = snap.current_state or self.current_state
        self.instruction_set_id = getattr(snap, "instruction_set_id", None) or self.instruction_set_id
        self.context_name = getattr(snap, "context_name", None) or self.context_name
        self.persist_id = snap.session_id or self.persist_id

    def to_snapshot(self) -> SessionSnapshot:
        title = ""
        turn = self.conversation.turn_count
        return SessionSnapshot(
            session_id=self.persist_id,
            model=self.model,
            voice_id=self.voice_id,
            current_state=self.current_state,
            instruction_set_id=self.instruction_set_id or "default",
            context_name=self.context_name or "default",
            turns=self.conversation.to_turn_dicts(),
            title=title,
        )

    def session_loaded_event(self, resumed: bool) -> dict:
        turns = self.conversation.turn_count
        return {
            "type": "session_loaded",
            "session_id": self.persist_id,
            "resumed": resumed,
            "turns": turns,
        }

    @property
    def state_label(self) -> str:
        flow = active_flow()
        if not flow:
            return self.current_state or "default"
        st = flow.state(self.current_state) if flow else None
        return st.label if st and hasattr(st, "label") else (self.current_state or "default")

    def build_system_prompt(self) -> str:
        parts = []
        ctx = context_mgr.get(self.context_name) if context_mgr and self.context_name else None
        iset = instruction_mgr.get(self.instruction_set_id) if instruction_mgr and self.instruction_set_id else None
        instr = ""
        if iset and hasattr(iset, "system_prompt"):
            instr = iset.system_prompt or ""
        elif iset and isinstance(iset, str):
            instr = iset
        # Operator live directives must lead; they are prepended to instruction
        # and often also pinned at the top of context.
        directives_first = (instr or "").lstrip().startswith(
            "# Operator directives (IMPERATIVE"
        ) or (isinstance(ctx, str) and ctx.lstrip().startswith("# Operator directives (IMPERATIVE"))
        if directives_first:
            if instr:
                parts.append(instr)
            if ctx:
                parts.append(ctx)
        else:
            if ctx:
                parts.append(ctx)
            if instr:
                parts.append(instr)
        prompt = "\n\n".join(parts) if parts else ""
        facts_block = call_facts_svc.call_facts_block(self.call_facts)
        if facts_block:
            prompt = f"{prompt}\n\n{facts_block}" if prompt else facts_block
        # Sticky live coach: also pin at the END so it wins over long packs/history.
        overlay = _operator_override_block(self.operator_directives)
        if overlay:
            prompt = f"{prompt}\n\n{overlay}" if prompt else overlay
        return prompt

    def get_model_params(self) -> ModelParams:
        if instruction_mgr:
            iset = instruction_mgr.get(self.instruction_set_id) if self.instruction_set_id else None
            if iset:
                return iset.params_for(self.model)
        return ModelParams()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def persist_chat_session(session: ChatSession):
    if session_store and session.persist_id:
        session_store.save(session.to_snapshot())


def cancel_active_turn(session: ChatSession) -> bool:
    """Stop in-flight LLM/TTS for this session (barge-in). Returns True if a turn was active."""
    task = session.active_turn_task
    consumer = session._tts_consumer_task
    q = session._tts_queue
    cancelled = False
    if task and not task.done():
        session.generation_turn += 1
        task.cancel()
        cancelled = True
    if consumer and not consumer.done():
        consumer.cancel()
        cancelled = True
    if q is not None:
        try:
            while True:
                q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        session._tts_queue = None
    session._tts_consumer_task = None
    ft = session._filler_task
    if ft and not ft.done():
        ft.cancel()
        cancelled = True
    return cancelled


def _quiet_filler_events(clip, gain: float):
    """Yield filler WS events with scaled-down PCM and filler=True tags."""
    import numpy as np

    gain = max(0.0, min(1.0, float(gain)))
    for evt in filler_cache.get_filler_as_events(clip):
        out = dict(evt)
        out["filler"] = True
        if out.get("type") == "tts_audio" and out.get("data") and gain != 1.0:
            raw = base64.b64decode(out["data"])
            ns = int(out.get("num_samples") or 0)
            if ns and len(raw) == ns * 4:
                samples = np.frombuffer(raw, dtype=np.float32) * gain
                out["data"] = base64.b64encode(samples.astype(np.float32).tobytes()).decode("ascii")
            elif ns and len(raw) == ns * 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * gain
                out["data"] = base64.b64encode(
                    np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
                ).decode("ascii")
        yield out


async def _delayed_filler(session: "ChatSession", send, turn_id: int, transcript: str) -> None:
    """Play a quiet filler only if the real reply TTS is still late."""
    try:
        await asyncio.sleep(max(0.0, config.FILLER_DELAY_MS) / 1000.0)
    except asyncio.CancelledError:
        return

    if session.generation_turn != turn_id or session.reply_audio_started:
        return
    if not (session.tts_enabled and config.FILLERS_ENABLED):
        return
    if filler_cache.is_direct_question(transcript):
        return
    if random.random() >= config.FILLER_PROBABILITY:
        return

    try:
        clip = await filler_cache.get_filler_for_voice(session.voice_id, session.language)
        if not clip:
            return
        if session.generation_turn != turn_id or session.reply_audio_started:
            return
        for evt in _quiet_filler_events(clip, config.FILLER_GAIN):
            if session.generation_turn != turn_id or session.reply_audio_started:
                return
            await send(evt)
        logger.info(
            "[%s] Delayed filler played (delay_ms=%.0f gain=%.2f phrase=%r)",
            session.id,
            config.FILLER_DELAY_MS,
            config.FILLER_GAIN,
            getattr(clip, "phrase", "")[:40],
        )
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("[%s] Delayed filler failed", session.id, exc_info=True)


def _close_stt_stream(session: ChatSession):
    stream = session.stt_stream
    if stream:
        session.stt_stream = None
        asyncio.ensure_future(stream.close())


# ---------------------------------------------------------------------------
# Turn management
# ---------------------------------------------------------------------------
async def start_user_turn(session: ChatSession, ws: WebSocket, text: str):
    """Spawn one chat turn as a background task (non-blocking).

    The WS message loop keeps reading while the turn runs, so interrupt /
    start_stt messages cancel mid-turn via cancel_active_turn().
    """
    if session.active_turn_task and not session.active_turn_task.done():
        cancel_active_turn(session)

    task = asyncio.create_task(handle_user_text(session, ws, text))
    session.active_turn_task = task

    def _on_done(t):
        if session.active_turn_task is t:
            session.active_turn_task = None

    task.add_done_callback(_on_done)


async def start_greeting_turn(session: ChatSession, ws: WebSocket, greeting: str):
    """Same as start_user_turn but for greeting playback."""
    if session.active_turn_task and not session.active_turn_task.done():
        cancel_active_turn(session)

    task = asyncio.create_task(handle_greeting(session, ws, greeting))
    session.active_turn_task = task

    def _on_done(t):
        if session.active_turn_task is t:
            session.active_turn_task = None

    task.add_done_callback(_on_done)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    global context_mgr, instruction_mgr, session_store, flow_mgr
    logger.info("=== Voice Chatbot v4 (parallel pipeline) starting ===")
    context_mgr = ContextManager(config.CONTEXT_DIR)
    instruction_mgr = InstructionManager(config.INSTRUCTIONS_DIR)
    flow_mgr = FlowManager(config.FLOWS_DIR, config.DEFAULT_FLOW)
    session_store = SessionStore(config.SESSIONS_DIR)

    flow = flow_mgr.default()
    logger.info(
        "Loaded %d context file(s), %d instruction set(s), %d flow(s); active flow=%s",
        len(context_mgr.list_all()) if hasattr(context_mgr, "list_all") else 0,
        len(instruction_mgr.list_all()) if hasattr(instruction_mgr, "list_all") else 0,
        len(flow_mgr.list_all()) if hasattr(flow_mgr, "list_all") else 0,
        flow.id if flow else None,
    )

    loop = asyncio.get_running_loop()
    if not config.DEFAULT_STT_MODEL.startswith("none"):
        try:
            await loop.run_in_executor(None, preload)
        except Exception:
            logger.exception("Failed to preload STT (will load on first use).")

    if config.FILLERS_ENABLED:
        asyncio.create_task(filler_cache.warm_cache(config.DEFAULT_VOICE_ID))

    yield

    logger.info("=== Voice Chatbot v4 shutting down ===")
    await shutdown_v2_client()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Voice Chatbot", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "stt": get_stt_info()}


DEFAULT_SESSION_ID = config.DEFAULT_SESSION_ID


@app.get("/api/session")
async def get_saved_session(session_id: str = DEFAULT_SESSION_ID):
    snap = session_store.load(session_id) if session_store else None
    if snap:
        return snap.__dict__ if hasattr(snap, "__dict__") else snap
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/api/sessions")
async def list_saved_sessions():
    return session_store.list_sessions() if session_store else []


def _admin_restart_denied():
    if not getattr(config, "ALLOW_ADMIN_RESTART", False):
        return JSONResponse(status_code=403, content={"error": "admin restart disabled"})
    return None


def _spawn_restart_script(script_name: str):
    script = config.BASE_DIR / "scripts" / script_name
    if not script.exists():
        return JSONResponse(status_code=404, content={"error": f"script not found: {script_name}"})
    subprocess.Popen(["bash", str(script)], start_new_session=True)
    return {"status": "restarting"}


@app.post("/api/admin/restart")
async def admin_restart_chatbot():
    denied = _admin_restart_denied()
    if denied:
        return denied
    try:
        return _spawn_restart_script("restart_chatbot.sh")
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/admin/restart-tts")
async def admin_restart_tts():
    denied = _admin_restart_denied()
    if denied:
        return denied
    try:
        return _spawn_restart_script("restart_tts.sh")
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/tts-health")
async def tts_health():
    try:
        import httpx
        url = config.TTS_REST_URL + "/health" if hasattr(config, "TTS_REST_URL") else f"http://localhost:{config.TTS_SERVER_PORT}/health"
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url)
            return r.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/api/stt-models")
async def list_stt_models():
    return {"models": config.AVAILABLE_STT_MODELS, "default_model": config.DEFAULT_STT_MODEL}


@app.get("/api/models")
async def list_models():
    return config.AVAILABLE_MODELS


@app.get("/api/voices")
async def list_voices():
    try:
        from services.tts_client_v2 import fetch_voices
        return await fetch_voices()
    except Exception:
        return []


@app.get("/api/flow")
async def get_flow():
    flow = active_flow()
    if flow:
        return flow.__dict__ if hasattr(flow, "__dict__") else {"id": flow.id}
    return {}


@app.get("/api/instructions")
async def list_instructions():
    if not instruction_mgr:
        return []
    return [iset.to_dict() for iset in instruction_mgr.list_all()]


@app.get("/api/instructions/{set_id}")
async def get_instruction_set(set_id: str):
    iset = instruction_mgr.get(set_id) if instruction_mgr else None
    if iset:
        return iset.to_dict()
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.put("/api/instructions/{set_id}")
async def save_instruction_set(set_id: str, body: dict):
    if not instruction_mgr:
        return JSONResponse(status_code=500, content={"error": "not initialized"})
    body_id = body.get("id", set_id)
    if body_id != set_id:
        return JSONResponse(status_code=400, content={"error": "id in body must match URL"})
    try:
        iset = instruction_mgr.save_from_dict(body)
        return {"status": "saved", "instruction": iset.to_dict()}
    except (KeyError, ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.exception("Failed to save instruction set %s", set_id)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/instructions")
async def create_instruction_set(body: dict):
    if not instruction_mgr:
        return JSONResponse(status_code=500, content={"error": "not initialized"})
    body.setdefault("id", uuid.uuid4().hex[:8])
    try:
        iset = instruction_mgr.save_from_dict(body)
        return {"status": "saved", "instruction": iset.to_dict()}
    except (KeyError, ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.exception("Failed to create instruction set")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.delete("/api/instructions/{set_id}")
async def delete_instruction_set(set_id: str):
    if instruction_mgr:
        instruction_mgr.delete(set_id)
    return {"status": "deleted"}


@app.get("/api/context")
async def list_context_files():
    return context_mgr.list_all() if context_mgr else []


@app.get("/api/context/{name}")
async def get_context(name: str):
    content = context_mgr.get(name) if context_mgr else None
    if content is not None:
        return {"name": name, "content": content}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.put("/api/context/{name}")
async def save_context(name: str, body: dict):
    if context_mgr:
        context_mgr.save(name, body.get("content", ""))
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# WebSocket handler — main chat loop
# ---------------------------------------------------------------------------
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session = ChatSession()
    logger.info("[%s] Chat session opened.", session.id)

    async def send_event(event: dict):
        await ws.send_text(json.dumps(event))

    try:
        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "bytes" in raw:
                pcm = raw["bytes"]
                if session.recorder.active:
                    session.recorder.append(pcm)
                if session.stt_stream:
                    session.stt_stream.append(pcm)
                continue

            data = raw.get("text", "")
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await send_event({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            # ------ load_session ------
            if msg_type == "load_session":
                cancel_active_turn(session)
                session.conversation.turn_count  # ensure attribute
                sid = msg.get("session_id", config.DEFAULT_SESSION_ID)
                snap = session_store.load(sid) if session_store else None
                if snap:
                    session.apply_snapshot(snap)
                    logger.info("[%s] Restored session %r (%d turns).", session.id, sid, session.conversation.turn_count)
                await send_event(session.session_loaded_event(snap is not None))

            # ------ save_session ------
            elif msg_type == "save_session":
                new_id = msg.get("session_id") or session.persist_id
                session.persist_id = new_id
                persist_chat_session(session)
                await send_event({"type": "session_saved", "session_id": new_id})

            # ------ config ------
            elif msg_type == "config":
                prev_voice = session.voice_id
                if "model" in msg:
                    session.model = msg["model"]
                if "voice_id" in msg:
                    session.voice_id = msg["voice_id"]
                if "instruction_set" in msg:
                    session.instruction_set_id = msg["instruction_set"]
                if "context" in msg:
                    session.context_name = msg["context"]
                    ctx_body = (
                        context_mgr.get(session.context_name)
                        if context_mgr and session.context_name
                        else None
                    )
                    if isinstance(ctx_body, str) and ctx_body.strip():
                        session.merge_call_facts(
                            call_facts_svc.seed_from_context_text(ctx_body)
                        )
                if "language" in msg:
                    session.language = msg["language"]
                if "tts_enabled" in msg:
                    session.tts_enabled = msg["tts_enabled"]
                if "router_enabled" in msg:
                    session.router_enabled = bool(msg["router_enabled"])
                if "stt_model" in msg:
                    session.stt_model = resolve_stt_model(msg["stt_model"])
                await send_event({"type": "config_ack"})
                if config.FILLERS_ENABLED and session.voice_id != prev_voice:
                    asyncio.create_task(filler_cache.warm_cache(session.voice_id))
                # Prefill Anthropic prompt cache for large Operator packs so the
                # first spoken turn (and greeting LLM path) hit a warm prefix.
                if (
                    getattr(config, "ANTHROPIC_CACHE_WARMUP", True)
                    and any(k in msg for k in ("instruction_set", "context", "model"))
                ):
                    asyncio.create_task(
                        llm_service.warm_prompt_cache(
                            session.model,
                            session.build_system_prompt(),
                            session.get_model_params(),
                        )
                    )

            # ------ coach (Operator live imperative; never interrupts TTS) ------
            elif msg_type == "coach":
                note = (msg.get("text") or "").strip()
                if note:
                    session.operator_directives.append(note)
                    tts_instruct = _coach_note_to_tts_instruct(note)
                    if tts_instruct:
                        session.default_tts_instruct = tts_instruct
                        logger.info(
                            "[%s] Coach set default_tts_instruct=%r",
                            session.id,
                            tts_instruct,
                        )
                    logger.info(
                        "[%s] Operator coach directive (%d): %s",
                        session.id,
                        len(session.operator_directives),
                        note[:120],
                    )
                    await send_event(
                        {
                            "type": "coach_ack",
                            "count": len(session.operator_directives),
                            "tts_instruct": session.default_tts_instruct or "",
                        }
                    )
                    if getattr(config, "ANTHROPIC_CACHE_WARMUP", True):
                        asyncio.create_task(
                            llm_service.warm_prompt_cache(
                                session.model,
                                session.build_system_prompt(),
                                session.get_model_params(),
                            )
                        )

            # ------ tts_instruct (Operator / bridge sticky delivery) ------
            elif msg_type == "tts_instruct":
                instruct = (msg.get("text") or msg.get("instruct") or "").strip()
                session.default_tts_instruct = instruct
                logger.info(
                    "[%s] default_tts_instruct=%r",
                    session.id,
                    instruct[:120] if instruct else "",
                )
                await send_event(
                    {
                        "type": "tts_instruct_ack",
                        "instruct": session.default_tts_instruct,
                    }
                )

            # ------ call_facts / call_fact (Operator seed or mid-call pin) ------
            elif msg_type == "call_facts":
                facts = msg.get("facts") if isinstance(msg.get("facts"), dict) else {}
                replace = bool(msg.get("replace"))
                updated = session.merge_call_facts(facts, replace=replace)
                await send_event(
                    {
                        "type": "call_facts_ack",
                        "updated": updated,
                        "facts": {
                            k: v for k, v in session.call_facts.items() if v
                        },
                    }
                )
                if updated and getattr(config, "ANTHROPIC_CACHE_WARMUP", True):
                    asyncio.create_task(
                        llm_service.warm_prompt_cache(
                            session.model,
                            session.build_system_prompt(),
                            session.get_model_params(),
                        )
                    )

            elif msg_type == "call_fact":
                key = (msg.get("key") or "").strip()
                value = (msg.get("value") or "").strip()
                replace = bool(msg.get("replace"))
                updated = []
                if key and value:
                    updated = session.merge_call_facts(
                        {key: value}, replace=replace
                    )
                await send_event(
                    {
                        "type": "call_fact_ack",
                        "updated": updated,
                        "facts": {
                            k: v for k, v in session.call_facts.items() if v
                        },
                    }
                )

            # ------ text_input ------
            elif msg_type == "text_input":
                text = msg.get("text", "").strip()
                if text:
                    cancel_active_turn(session)
                    await start_user_turn(session, ws, text)

            # ------ interrupt ------
            elif msg_type == "interrupt":
                if cancel_active_turn(session):
                    logger.info("[%s] Generation interrupted.", session.id)
                    await send_event({"type": "turn_cancelled"})

            # ------ excuse_me ------
            elif msg_type == "excuse_me":
                await _handle_excuse_me(session, ws, send_event)

            # ------ start_stt ------
            elif msg_type == "start_stt":
                cancel_active_turn(session)
                await send_event({"type": "turn_cancelled"})
                logger.info("[%s] Barge-in: stopped assistant output.", session.id)

                stt_model = resolve_stt_model(session.stt_model)
                if is_realtime_model(stt_model):
                    # stop_stt clears session.stt_stream after finish(); reuse must
                    # re-bind it or the next utterance hits "Realtime STT session
                    # is not active" and audio never reaches OpenAI.
                    if session._stt_persistent and session._stt_persistent.is_alive:
                        session._stt_persistent.reset()
                        session.stt_stream = session._stt_persistent
                    else:
                        async def _on_delta(delta):
                            await send_event({"type": "transcript_delta", "delta": delta})

                        async def _on_vad_stopped():
                            session.vad_committed.set()

                        try:
                            stream = await create_realtime_session(
                                language=config.STT_LANGUAGE,
                                on_delta=_on_delta,
                                on_vad_stopped=_on_vad_stopped,
                                model=stt_model,
                            )
                            session.stt_stream = stream
                            session._stt_persistent = stream
                        except Exception as exc:
                            logger.exception("[%s] Failed to start realtime STT.", session.id)
                            await send_event({"type": "error", "message": "STT unavailable: " + str(exc)})
                            await send_event({"type": "stt_stopped"})
                            continue

                session.recorder.start()
                session.vad_committed.clear()
                await send_event({"type": "stt_ready"})
                await send_event({"type": "speech_started"})
                logger.info("[%s] Recording started (stt=%s).", session.id, stt_model)

            # ------ stop_stt (v4: early filler + async STT) ------
            elif msg_type == "stop_stt":
                if session.stt_transcribing:
                    logger.debug("[%s] Ignoring duplicate stop_stt.", session.id)
                    continue

                if not session.recorder.active and not session.stt_stream:
                    logger.debug("[%s] stop_stt with no active recording.", session.id)
                    await send_event({"type": "stt_stopped"})
                    continue

                chunks = session.recorder.stop()
                await send_event({"type": "speech_stopped"})

                n_samples = sum(len(c) // 2 for c in chunks)
                dur = n_samples / config.MIC_SAMPLE_RATE if n_samples else 0
                stt_model = resolve_stt_model(session.stt_model)
                logger.info("[%s] Recording stopped | %.2f s | %d chunk(s) | stt=%s", session.id, dur, len(chunks), stt_model)

                if not chunks and not session.stt_stream:
                    await send_event({"type": "stt_stopped"})
                    continue

                if is_realtime_model(stt_model) and not session.stt_stream:
                    await send_event({"type": "error", "message": "Realtime STT session is not active. Try speaking again."})
                    await send_event({"type": "stt_stopped"})
                    continue

                session.stt_transcribing = True

                # === v4: NON-BLOCKING STT ===
                t_stt_start = time.perf_counter()

                if session.stt_stream:
                    stream = session.stt_stream
                    session.stt_stream = None
                    vad_did_commit = session.vad_committed.is_set()
                    transcript = await stream.finish(already_committed=vad_did_commit)
                else:
                    await send_event({"type": "transcript_delta", "delta": "Transcribing\u2026"})
                    transcript = await transcribe(
                        chunks,
                        model_id=stt_model,
                        language=config.STT_LANGUAGE,
                        sample_rate=config.MIC_SAMPLE_RATE,
                    )

                stt_ms = (time.perf_counter() - t_stt_start) * 1000
                logger.info("[%s] STT completed in %.0f ms", session.id, stt_ms)

                if transcript and transcript.strip():
                    await send_event({"type": "transcript_done", "text": transcript})
                    session.stt_transcribing = False
                    await send_event({"type": "stt_stopped"})

                    if cancel_active_turn(session):
                        await send_event({"type": "turn_cancelled"})

                    # Fillers are delay-gated inside handle_user_text (only if
                    # reply TTS is late), not fired immediately after STT.
                    await start_user_turn(session, ws, transcript)
                else:
                    await send_event({"type": "error", "message": "No speech detected. Try speaking longer or closer to the mic."})
                    session.stt_transcribing = False
                    await send_event({"type": "stt_stopped"})

            # ------ audio_chunk ------
            elif msg_type == "audio_chunk":
                pcm = base64.b64decode(msg.get("data", ""))
                if session.recorder.active:
                    session.recorder.append(pcm)
                if session.stt_stream:
                    session.stt_stream.append(pcm)

            # ------ set_mode ------
            elif msg_type == "set_mode":
                state = msg.get("state")
                if state and switch_state(session, state):
                    await send_event(mode_changed_event(session))
                else:
                    await send_event({"type": "error", "message": "Unknown call state: " + str(state)})

            # ------ start_call ------
            elif msg_type == "start_call":
                flow = active_flow()
                if flow:
                    entry = flow.entry_state
                    session.current_state = entry
                    if hasattr(flow, "greeting") and flow.greeting:
                        await start_greeting_turn(session, ws, flow.greeting)

            # ------ clear_history ------
            elif msg_type == "clear_history":
                session.conversation = ConversationManager(max_turns=config.MAX_CONVERSATION_TURNS)
                await send_event({"type": "history_cleared"})

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected.", session.id)
    except Exception:
        logger.exception("[%s] Unexpected error.", session.id)
    finally:
        _close_stt_stream(session)
        cancel_active_turn(session)
        logger.info("[%s] Session cleaned up.", session.id)


# ---------------------------------------------------------------------------
# Excuse-me handling
# ---------------------------------------------------------------------------
async def _handle_excuse_me(session: ChatSession, ws: WebSocket, send_event):
    snapshot = session.turn_snapshot
    assistant_text = ""
    if snapshot and isinstance(snapshot, dict):
        assistant_text = snapshot.get("full_response", "")
    await run_excuse_recovery(session, ws, assistant_text)


async def run_excuse_recovery(session: ChatSession, ws: WebSocket, assistant_text: str):
    task = asyncio.create_task(_play_excuse_recovery(session, ws, assistant_text))
    session.active_turn_task = task


async def _play_excuse_recovery(session: ChatSession, ws: WebSocket, assistant_text: str):
    sid = session.id

    async def send(evt):
        await ws.send_text(json.dumps(evt))

    turn_id = session.generation_turn
    replay_text = EXCUSE_ME_PHRASE + " " + assistant_text if assistant_text else EXCUSE_ME_PHRASE
    await send({"type": "llm_start", "model": session.model})
    await send({"type": "llm_delta", "delta": replay_text})
    await send({"type": "llm_done", "text": replay_text})

    if session.tts_enabled:
        async with session.tts_lock:
            await _dispatch_tts(ws, session, replay_text, session.voice_id, session.language, turn_id)

    await send({"type": "tts_done"})
    await send({"type": "turn_done"})


# ---------------------------------------------------------------------------
# Greeting
# ---------------------------------------------------------------------------
async def handle_greeting(session: ChatSession, ws: WebSocket, greeting: str):
    sid = session.id

    async def send(evt):
        await ws.send_text(json.dumps(evt))

    def turn_active(turn_id):
        return turn_id == session.generation_turn

    session.generation_turn += 1
    turn_id = session.generation_turn

    await send({"type": "llm_start", "model": session.model})
    await send({"type": "llm_delta", "delta": greeting})
    session.conversation.add_assistant_message(greeting, model=session.model)
    await send({"type": "llm_done", "text": greeting})

    if session.tts_enabled and turn_active(turn_id):
        await send({"type": "tts_start", "num_sentences": 1, "sample_rate": 24000})
        async with session.tts_lock:
            if turn_active(turn_id):
                await _dispatch_tts(ws, session, greeting, session.voice_id, session.language, turn_id)
        await send({"type": "tts_done"})

    if turn_active(turn_id):
        await send({"type": "turn_done"})
    persist_chat_session(session)


# ---------------------------------------------------------------------------
# maybe_route (v4: exposed as standalone coroutine for parallel execution)
# ---------------------------------------------------------------------------
async def maybe_route(session: ChatSession, send) -> Optional[str]:
    """Run the call router for this turn and switch state if warranted.

    Routes from reception always; from a specialist only when re-routing is
    allowed (confidence-guarded inside the router). Emits mode_changed on a
    switch. Failures are swallowed so a routing hiccup never blocks the turn.

    Returns the new state name if switched, else None.
    """
    if not config.ROUTER_ENABLED:
        return None
    if getattr(session, "router_enabled", True) is False:
        return None
    flow = active_flow()
    if not flow:
        return None

    in_specialist = session.current_state != flow.entry_state
    if in_specialist and not config.ALLOW_REROUTE:
        return None

    messages = session.conversation.get_messages(model=session.model)

    try:
        decision = await call_router.route_intent(
            flow,
            messages,
            session.current_state,
            allow_reroute=config.ALLOW_REROUTE,
            reroute_min_confidence=config.REROUTE_MIN_CONFIDENCE,
        )
    except Exception:
        logger.exception("[%s] Routing failed; keeping current state.", session.id)
        return None

    if decision.should_switch and switch_state(session, decision.target):
        logger.info(
            "[%s] Routed %s -> %s (conf=%.2f, %s)",
            session.id,
            flow.entry_state if not in_specialist else "specialist",
            decision.target,
            decision.confidence,
            decision.reason,
        )
        await send(mode_changed_event(session, reason=decision.reason, confidence=decision.confidence))
        return decision.target

    return None


# ---------------------------------------------------------------------------
# handle_user_text (v4: parallel router + stage timers)
# ---------------------------------------------------------------------------
async def handle_user_text(session: ChatSession, ws: WebSocket, text: str):
    """Process a user utterance: routing -> LLM generation -> TTS -> audio stream.

    v4 changes:
      - LLM stream starts immediately on current state's prompt
      - maybe_route runs in parallel; if a switch occurs, the current generation
        is cancelled and restarted with the new state's prompt (single restart)
      - Stage timers are logged
    """
    sid = session.id
    logger.info("[%s] User: %s", sid, text[:100])

    session.conversation.add_or_coalesce_user_message(text)
    updated = call_facts_svc.extract_from_user(text, session.call_facts)
    if updated:
        call_facts_svc.log_updates(sid, updated, session.call_facts)

    async def send(evt: dict):
        await ws.send_text(json.dumps(evt))

    def turn_active(turn_id: int) -> bool:
        return turn_id == session.generation_turn

    session.generation_turn += 1
    turn_id = session.generation_turn
    session.reply_audio_started = False

    # Quiet filler only if real reply TTS is still late after FILLER_DELAY_MS.
    if session._filler_task and not session._filler_task.done():
        session._filler_task.cancel()
    session._filler_task = asyncio.create_task(
        _delayed_filler(session, send, turn_id, text)
    )

    await send({"type": "llm_start", "model": session.model})

    # === v4: Start LLM immediately + maybe_route in parallel ===
    t_turn_start = time.perf_counter()

    # Fire router as a parallel task (non-blocking)
    route_task = asyncio.create_task(maybe_route(session, send))

    if not turn_active(turn_id):
        route_task.cancel()
        return

    # Prepare LLM call with current state's prompt
    system_prompt = session.build_system_prompt()
    messages = _messages_with_operator_directives(
        session.conversation.get_messages(model=session.model),
        session.operator_directives,
    )
    params = session.get_model_params()
    if session.operator_directives:
        logger.info(
            "[%s] Applying %d Operator directive(s) to this turn",
            sid,
            len(session.operator_directives),
        )
    known = {k: v for k, v in session.call_facts.items() if v}
    if known:
        logger.info("[%s] call_facts pinned: %s", sid, known)

    full_response = ""
    sentence_buffer = ""
    sentences_queued = []
    tts_sentence_count = 0
    first_tts_dispatched = False
    session.turn_snapshot = {"full_response": "", "sentence_buffer": ""}

    tts_queue: asyncio.Queue = asyncio.Queue()
    session._tts_queue = tts_queue

    async def tts_consumer():
        nonlocal tts_sentence_count
        try:
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    return
                if not session.tts_enabled or not turn_active(turn_id):
                    continue

                batch = [sentence]
                ended = False
                while not tts_queue.empty():
                    nxt = tts_queue.get_nowait()
                    if nxt is None:
                        ended = True
                        break
                    batch.append(nxt)

                tts_sentence_count += 1
                if tts_sentence_count == 1:
                    session.reply_audio_started = True
                    ft = session._filler_task
                    if ft and not ft.done():
                        ft.cancel()
                    await send({"type": "tts_start", "num_sentences": 1, "sample_rate": 24000})

                spoken, instruct = _extract_tts_instruct(" ".join(batch))
                if not instruct:
                    instruct = session.default_tts_instruct or ""
                if not spoken:
                    if ended:
                        return
                    continue

                async with session.tts_lock:
                    if turn_active(turn_id):
                        await _dispatch_tts(
                            ws, session, spoken, session.voice_id, session.language,
                            turn_id, instruct=instruct,
                        )

                if ended:
                    return
        except asyncio.CancelledError:
            return

    consumer_task = asyncio.create_task(tts_consumer()) if session.tts_enabled else None
    session._tts_consumer_task = consumer_task

    # Track whether route caused a restart
    route_switched = False
    llm_ttft_ms = None
    t_first_token = None

    try:
        async for token in llm_service.stream_chat(
            model=session.model,
            system_prompt=system_prompt,
            messages=messages,
            params=params,
        ):
            if not turn_active(turn_id):
                break

            # === v4: Check if router finished and switched state ===
            if not route_switched and route_task.done():
                try:
                    new_state = route_task.result()
                except Exception:
                    new_state = None
                if new_state:
                    # Router switched state — cancel this generation and restart
                    route_switched = True
                    route_ms = (time.perf_counter() - t_turn_start) * 1000
                    logger.info("[%s] Route switch mid-stream -> %s (route_ms=%.0f); restarting turn.", sid, new_state, route_ms)
                    break

            if t_first_token is None:
                t_first_token = time.perf_counter()
                llm_ttft_ms = (t_first_token - t_turn_start) * 1000

            full_response += token
            session.turn_snapshot = {"full_response": full_response, "sentence_buffer": sentence_buffer}

            await send({"type": "llm_delta", "delta": token})

            if not session.tts_enabled:
                continue

            sentence_buffer += token
            splitter = _split_first_clause if not first_tts_dispatched else _split_sentences
            parts = splitter(sentence_buffer)

            if len(parts) > 1:
                for s in parts[:-1]:
                    sentences_queued.append(s)
                    tts_queue.put_nowait(s)
                    first_tts_dispatched = True
                sentence_buffer = parts[-1]

            session.turn_snapshot = {"full_response": full_response, "sentence_buffer": sentence_buffer}

    except asyncio.CancelledError:
        logger.info("[%s] Turn %d cancelled (barge-in).", sid, turn_id)
        raise
    except Exception as exc:
        logger.exception("[%s] LLM generation error.", sid)
        await send({"type": "error", "message": "LLM error: " + str(exc)})

    # === v4: If route switched, restart the turn with new prompt ===
    if route_switched and turn_active(turn_id):
        # Cancel TTS consumer
        if consumer_task and not consumer_task.done():
            consumer_task.cancel()
        # Discard partial response, restart with new state
        session.conversation.add_assistant_message(full_response, model=session.model) if full_response.strip() else None
        # Remove the partial assistant message — we'll redo the full turn
        if full_response.strip():
            # Pop the partial assistant message we just added
            try:
                session.conversation._history.pop()
            except (IndexError, AttributeError):
                pass

        # Rebuild with new state's prompt and restart LLM
        system_prompt = session.build_system_prompt()
        messages = _messages_with_operator_directives(
            session.conversation.get_messages(model=session.model),
            session.operator_directives,
        )
        params = session.get_model_params()

        full_response = ""
        sentence_buffer = ""
        sentences_queued = []
        tts_sentence_count = 0
        first_tts_dispatched = False
        session.turn_snapshot = {"full_response": "", "sentence_buffer": ""}
        tts_queue = asyncio.Queue()
        session._tts_queue = tts_queue
        consumer_task = asyncio.create_task(tts_consumer()) if session.tts_enabled else None
        session._tts_consumer_task = consumer_task
        t_first_token = None

        try:
            async for token in llm_service.stream_chat(
                model=session.model,
                system_prompt=system_prompt,
                messages=messages,
                params=params,
            ):
                if not turn_active(turn_id):
                    break

                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    llm_ttft_ms = (t_first_token - t_turn_start) * 1000

                full_response += token
                session.turn_snapshot = {"full_response": full_response, "sentence_buffer": sentence_buffer}
                await send({"type": "llm_delta", "delta": token})

                if not session.tts_enabled:
                    continue

                sentence_buffer += token
                splitter = _split_first_clause if not first_tts_dispatched else _split_sentences
                parts = splitter(sentence_buffer)
                if len(parts) > 1:
                    for s in parts[:-1]:
                        sentences_queued.append(s)
                        tts_queue.put_nowait(s)
                        first_tts_dispatched = True
                    sentence_buffer = parts[-1]
                session.turn_snapshot = {"full_response": full_response, "sentence_buffer": sentence_buffer}

        except asyncio.CancelledError:
            logger.info("[%s] Turn %d cancelled (barge-in) after route restart.", sid, turn_id)
            raise
        except Exception as exc:
            logger.exception("[%s] LLM generation error (post-route).", sid)
            await send({"type": "error", "message": "LLM error: " + str(exc)})

    # --- Finalize ---

    if not turn_active(turn_id):
        logger.info("[%s] Turn %d superseded; skipping history commit.", sid, turn_id)
        if consumer_task and not consumer_task.done():
            consumer_task.cancel()
        return

    # Flush remaining sentence buffer to TTS
    if session.tts_enabled and sentence_buffer.strip():
        sentences_queued.append(sentence_buffer.strip())
        tts_queue.put_nowait(sentence_buffer.strip())

    await send({"type": "llm_done", "text": full_response})

    # Keep wire tokens in llm_done for AnswerBot; strip from history/TTS memory.
    history_text = _strip_wire_tokens(full_response) or full_response
    session.conversation.add_assistant_message(history_text, model=session.model)
    if full_response.strip():
        asst_updated = call_facts_svc.extract_from_assistant(
            full_response, session.call_facts
        )
        if asst_updated:
            call_facts_svc.log_updates(sid, asst_updated, session.call_facts)

    # === v4: Stage timers ===
    route_ms = None
    if route_task.done() and not route_task.cancelled():
        try:
            route_task.result()
        except Exception:
            pass
        # Route task ran in parallel with LLM; measure from turn start
        # (actual duration is hidden by overlap, but we know it finished)
        route_ms = 0

    logger.info(
        "[%s] v4 Done | chars=%d tts_sentences=%d llm_ttft_ms=%.0f route_ms=%s model=%s",
        sid,
        len(full_response),
        len(sentences_queued),
        llm_ttft_ms or 0,
        f"{route_ms:.0f}" if route_ms is not None else "pending",
        session.model,
    )

    # Wait for TTS consumer to finish
    if consumer_task:
        tts_queue.put_nowait(None)
        await consumer_task

    if not turn_active(turn_id):
        if consumer_task and not consumer_task.done():
            consumer_task.cancel()
        return

    if session.tts_enabled:
        await send({"type": "tts_done"})

    session.turn_snapshot = None
    await send({"type": "turn_done"})
    persist_chat_session(session)

    if consumer_task and not consumer_task.done():
        consumer_task.cancel()


# ---------------------------------------------------------------------------
# TTS dispatch
# ---------------------------------------------------------------------------
async def _dispatch_tts(
    ws: WebSocket,
    session: ChatSession,
    text: str,
    voice_id: str,
    language: str,
    turn_id: int,
    instruct: str = "",
):
    """Send one sentence to the TTS server and relay audio (caller holds tts_lock)."""
    text, tag_instruct = _extract_tts_instruct(text)
    instruct = (instruct or tag_instruct or session.default_tts_instruct or "").strip()
    if not text:
        return
    if instruct:
        logger.info("[%s] TTS instruct=%r text=%r", session.id, instruct[:80], text[:60])
    try:
        # Prefer instruct-capable client when delivery guidance is set; otherwise
        # keep the pooled v2 client for lower reconnect latency.
        if instruct:
            stream = stream_tts_with_instruct(
                text,
                voice_id=voice_id,
                language=language,
                instruct=instruct,
                turn_id=str(turn_id),
            )
        else:
            stream = stream_tts_v2(
                text,
                voice_id=voice_id,
                language=language,
            )
        async for event in stream:
            if session.generation_turn != turn_id:
                return
            if isinstance(event, TTSInstability):
                await ws.send_text(json.dumps({
                    "type": "tts_instability",
                    "reason": event.reason,
                    "sentence_index": event.sentence_index,
                    "will_retry": event.will_retry,
                }))
            else:
                await ws.send_text(json.dumps({
                    "type": "tts_audio",
                    "data": base64.b64encode(event.pcm_bytes).decode("ascii"),
                    "sample_rate": event.sample_rate,
                    "num_samples": event.num_samples,
                    "index": event.index,
                }))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("TTS dispatch error for: %s", text[:60])


# ---------------------------------------------------------------------------
# Static files + fallback
# ---------------------------------------------------------------------------
WEB_DIR = config.BASE_DIR / "web"
if WEB_DIR.is_dir():
    app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ssl_certfile = config.BASE_DIR / "cert.pem"
    ssl_keyfile = config.BASE_DIR / "key.pem"
    ssl_kwargs = {}
    if ssl_certfile.exists() and ssl_keyfile.exists():
        ssl_kwargs = {"ssl_certfile": str(ssl_certfile), "ssl_keyfile": str(ssl_keyfile)}
        logger.info("HTTPS enabled (cert=%s)", ssl_certfile)

    uvicorn.run(
        app,
        host=config.CHATBOT_HOST,
        port=config.CHATBOT_PORT,
        **ssl_kwargs,
    )
