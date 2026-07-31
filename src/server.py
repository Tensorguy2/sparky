"""
Qwen3-TTS Streaming Server
==========================
Entry point.  Run from the src/ directory:

    python server.py

Startup sequence
----------------
1. Configure logging (console INFO + rotating file DEBUG → src/logs/server.log)
2. Init SQLite (creates voices.db if absent)
3. Load TTS model onto GPU
4. Ensure built-in voices exist in DB
5. Pre-compute VoiceClonePromptItem for every registered voice → all in RAM
6. CUDA warmup inference (eliminates cold-start on first real request)

Swagger UI:  http://localhost:25565/docs
ReDoc:       http://localhost:25565/redoc
WS schema:   http://localhost:25565/ws/tts/schema
"""

# ── Logging must be configured before any module that calls getLogger() ────────
import utils.logging_setup as _log_bootstrap  # noqa: E402  (import before fastapi)

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config

_log_bootstrap.configure(log_dir=config.LOG_DIR)

from db import database as db                          # noqa: E402
from routers import tts as tts_router                  # noqa: E402
from routers import voices as voices_router            # noqa: E402
from services.model_service import model_service       # noqa: E402
from services.voice_service import voice_service       # noqa: E402

logger = logging.getLogger(__name__)


# ── OpenAPI metadata ───────────────────────────────────────────────────────────

_DESCRIPTION = """
## Qwen3-TTS Streaming API

Real-time, sentence-streamed Text-to-Speech backed by **Qwen3-TTS-12Hz-1.7B-Base**.

### Features
- **WebSocket streaming** — audio is returned sentence-by-sentence as raw
  `pcm_f32le` bytes; the first chunk arrives in < 1 s for short sentences.
- **Voice cloning** — register any speaker with a short WAV clip; embeddings
  are pre-computed and cached in RAM for zero-overhead inference.
- **Persistent profiles** — voices survive server restarts via SQLite.

### WebSocket endpoint
The main synthesis endpoint is a WebSocket at **`/ws/tts`** (not shown in
OpenAPI — use [`GET /ws/tts/schema`](/ws/tts/schema) for the protocol schema).

### Quick start
```json
// 1. Pick a voice
GET /voices

// 2. Connect and synthesize
ws://localhost:25565/ws/tts
→ send: {"text": "Hello!", "language": "English", "voice_id": "mikey"}
← recv: {"type":"metadata","sample_rate":24000,...}
← recv: {"type":"chunk_header","index":0,"num_samples":...,"size_bytes":...}
← recv: <binary PCM float32 LE>
← recv: {"type":"done"}
```

### Register a custom voice
```bash
curl -X POST http://localhost:25565/voices/alice \\
  -F name="Alice" \\
  -F ref_text="The quick brown fox jumps over the lazy dog." \\
  -F file=@alice_clip.wav
```
"""

_TAGS = [
    {
        "name": "tts",
        "description": (
            "Text-to-speech synthesis. "
            "The primary endpoint is the **WebSocket** `/ws/tts`; "
            "this tag also exposes `GET /ws/tts/schema` for the protocol documentation."
        ),
    },
    {
        "name": "voices",
        "description": (
            "CRUD operations for voice profiles. "
            "Each profile stores a reference WAV + transcript; "
            "the speaker embedding is pre-computed at registration and kept in RAM."
        ),
    },
    {
        "name": "system",
        "description": "Health check and server diagnostics.",
    },
]


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t_boot = time.perf_counter()
    logger.info("=== Qwen3-TTS Server starting up ===")

    # 1. Storage dirs + DB
    os.makedirs(config.VOICES_DIR, exist_ok=True)
    logger.debug("Voices directory: %s", config.VOICES_DIR)
    db.init(config.DB_PATH)

    # 2. Model
    model_service.load(config.MODEL_ID, config.DEVICE, config.DTYPE, config.ATTN_IMPL)
    loop = asyncio.get_running_loop()

    # 3. Seed built-in voices (idempotent)
    for vid, info in config.BUILTIN_VOICES.items():
        if not db.exists(vid):
            logger.info("Seeding built-in voice '%s' into DB.", vid)
            db.insert(vid, info["name"], info["ref_audio"], info["ref_text"], builtin=True)
        else:
            logger.debug("Built-in voice '%s' already in DB.", vid)

    # 4. Pre-compute all voice prompts into RAM
    await voice_service.load_all()

    # 5. CUDA warmup — use first built-in voice
    first_id = next(iter(config.BUILTIN_VOICES))
    first_entry = voice_service.get(first_id)
    if first_entry and first_entry.prompt_item:
        logger.info("Running CUDA warmup inference ...")
        await loop.run_in_executor(
            model_service.executor,
            lambda: model_service.warmup_sync(first_entry.prompt_item),
        )
    else:
        logger.warning("No prompt available for warmup — skipping. First request may be slow.")

    boot_elapsed = time.perf_counter() - t_boot
    loaded_ids = [e.voice_id for e in voice_service.list_all()]
    logger.info(
        "=== Server ready in %.2f s | port=%d | voices=%s ===",
        boot_elapsed, config.PORT, loaded_ids,
    )

    yield

    logger.info("=== Server shutting down ===")
    model_service.shutdown()
    logger.info("=== Shutdown complete ===")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qwen3-TTS API",
    version="2.0.0",
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    contact={"name": "Qwen3-TTS Server"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(tts_router.router)
app.include_router(voices_router.router)


# ── Static web UI (after API routes; do not shadow /docs, /health, /voices, /ws) ─

if os.path.isdir(config.WEB_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(config.WEB_DIR, "css")), name="web-css")
    app.mount("/js", StaticFiles(directory=os.path.join(config.WEB_DIR, "js")), name="web-js")

    @app.get("/", include_in_schema=False)
    async def web_index() -> FileResponse:
        return FileResponse(os.path.join(config.WEB_DIR, "index.html"))
else:
    logger.warning("Web UI directory not found: %s", config.WEB_DIR)


# ── Narrator avatar app (served under /narrator) ─────────────────────────────

if os.path.isdir(config.NARRATOR_DIR):
    @app.get("/narrator", include_in_schema=False)
    async def narrator_index() -> FileResponse:
        return FileResponse(os.path.join(config.NARRATOR_DIR, "index.html"))

    app.mount(
        "/narrator",
        StaticFiles(directory=config.NARRATOR_DIR),
        name="narrator",
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    description="Returns server status, model load state, and the list of voices currently in RAM.",
    response_description="Server health snapshot.",
    tags=["system"],
)
async def health() -> JSONResponse:
    voices_snapshot = [
        {"voice_id": e.voice_id, "name": e.name, "builtin": e.builtin, "prompt_ready": e.prompt_item is not None}
        for e in voice_service.list_all()
    ]
    payload = {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "voices_in_ram": len(voices_snapshot),
        "voices": voices_snapshot,
    }
    logger.debug("GET /health → %s", payload)
    return JSONResponse(payload)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        ws_ping_interval=config.WS_PING_INTERVAL,
        ws_ping_timeout=config.WS_PING_TIMEOUT,
        log_level="warning",          # uvicorn's own logger; ours is already configured
    )
