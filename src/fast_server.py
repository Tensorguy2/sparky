"""
Qwen3-TTS **Optimised** Streaming Server
==========================================
Drop-in replacement for server.py with lower time-to-first-audio and
sentence-level caching.  Run from the src/ directory:

    python fast_server.py                     # default port 25566
    python fast_server.py --port 25565        # override port

Improvements over server.py (no existing files are modified):

  O1 – Sentence-level audio cache
       Cache hits skip GPU entirely; 10 GB budget ≈ 100 000 s of audio.
  O2 – Overlapped sentence pipeline
       asyncio.Lock serialises GPU access so cache hits are instant and
       the next miss starts the moment the previous inference exits.
  O3 – Adaptive first-chunk size
       First decode fires after 4 frames (≈330 ms) instead of 12 (≈1 s).
  O4 – Pre-computed ref_audio_samples
       Voice-clone sample offset calculated once at streamer init.

Startup sequence is identical to server.py:
  1. Configure logging
  2. Init SQLite
  3. Load TTS model onto GPU
  4. Seed built-in voices
  5. Pre-compute voice prompts → RAM
  6. CUDA warmup inference
"""

# ── Logging must be configured before any module that calls getLogger() ────
import utils.logging_setup as _log_bootstrap  # noqa: E402

import argparse
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

from db import database as db                           # noqa: E402
from routers import fast_tts                            # noqa: E402
from routers import voices as voices_router             # noqa: E402
from services.audio_cache import audio_cache            # noqa: E402
from services.model_service import model_service        # noqa: E402
from services.voice_service import voice_service        # noqa: E402

logger = logging.getLogger(__name__)

# ── Default port (different from server.py so both can coexist in dev) ─────
DEFAULT_PORT = 25566
_active_port = DEFAULT_PORT  # set by __main__ before uvicorn.run()


# ── OpenAPI metadata ──────────────────────────────────────────────────────────

_DESCRIPTION = """
## Qwen3-TTS Optimised Streaming API

Same wire protocol as the standard server, with lower latency and caching.

### Optimisations
- **Sentence-level audio cache** — repeated sentences skip GPU entirely.
- **Adaptive first-chunk** — first audio arrives ~500-700 ms sooner.
- **Overlapped pipeline** — cache hits are served instantly between GPU
  sentences; no idle gaps.
- **Pre-computed ref offset** — voice-clone decoder overhead removed from
  the first-chunk critical path.

### Extra endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /cache/stats` | Cache hit rate, byte usage, entry count |
| `POST /cache/clear` | Evict all cached audio |

### WebSocket endpoint
The synthesis endpoint is a WebSocket at **`/ws/tts`** — wire-compatible
with the standard server.  See [`GET /ws/tts/schema`](/ws/tts/schema).
"""

_TAGS = [
    {
        "name": "tts",
        "description": "Text-to-speech synthesis (optimised pipeline).",
    },
    {
        "name": "voices",
        "description": "CRUD for voice profiles (unchanged from standard server).",
    },
    {
        "name": "cache",
        "description": "Audio cache management.",
    },
    {
        "name": "system",
        "description": "Health check and diagnostics.",
    },
]


# ── Lifespan (identical to server.py — reuses the same singletons) ───────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    t_boot = time.perf_counter()
    logger.info("=== Qwen3-TTS FAST Server starting up ===")

    os.makedirs(config.VOICES_DIR, exist_ok=True)
    db.init(config.DB_PATH)

    model_service.load(config.MODEL_ID, config.DEVICE, config.DTYPE, config.ATTN_IMPL)
    loop = asyncio.get_running_loop()

    for vid, info in config.BUILTIN_VOICES.items():
        if not db.exists(vid):
            logger.info("Seeding built-in voice '%s'.", vid)
            db.insert(vid, info["name"], info["ref_audio"], info["ref_text"], builtin=True)

    await voice_service.load_all()

    first_id = next(iter(config.BUILTIN_VOICES))
    first_entry = voice_service.get(first_id)
    if first_entry and first_entry.prompt_item:
        logger.info("Running CUDA warmup inference ...")
        await loop.run_in_executor(
            model_service.executor,
            lambda: model_service.warmup_sync(first_entry.prompt_item),
        )

    boot_elapsed = time.perf_counter() - t_boot
    loaded_ids = [e.voice_id for e in voice_service.list_all()]
    logger.info(
        "=== FAST Server ready in %.2f s | port=%d | voices=%s ===",
        boot_elapsed, _active_port, loaded_ids,
    )

    yield

    logger.info("=== FAST Server shutting down ===")
    model_service.shutdown()
    logger.info("=== Shutdown complete ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qwen3-TTS Optimised API",
    version="2.1.0",
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    contact={"name": "Qwen3-TTS Fast Server"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(fast_tts.router)
app.include_router(voices_router.router)


# ── Static web UI ─────────────────────────────────────────────────────────────

if os.path.isdir(config.WEB_DIR):
    app.mount(
        "/css",
        StaticFiles(directory=os.path.join(config.WEB_DIR, "css")),
        name="web-css",
    )
    app.mount(
        "/js",
        StaticFiles(directory=os.path.join(config.WEB_DIR, "js")),
        name="web-js",
    )

    @app.get("/", include_in_schema=False)
    async def web_index() -> FileResponse:
        return FileResponse(os.path.join(config.WEB_DIR, "index.html"))


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


# ── Health (standard + cache stats) ───────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    description="Server status, model state, voices in RAM, and audio cache stats.",
    tags=["system"],
)
async def health() -> JSONResponse:
    voices_snapshot = [
        {
            "voice_id": e.voice_id,
            "name": e.name,
            "builtin": e.builtin,
            "prompt_ready": e.prompt_item is not None,
        }
        for e in voice_service.list_all()
    ]
    return JSONResponse({
        "status": "ok",
        "server": "fast",
        "model_loaded": model_service.model is not None,
        "voices_in_ram": len(voices_snapshot),
        "voices": voices_snapshot,
        "cache": audio_cache.stats(),
    })


# ── Cache management ──────────────────────────────────────────────────────────

@app.get(
    "/cache/stats",
    summary="Audio cache statistics",
    description="Returns hit/miss counts, byte usage, and utilisation.",
    tags=["cache"],
)
async def cache_stats() -> JSONResponse:
    return JSONResponse(audio_cache.stats())


@app.post(
    "/cache/clear",
    summary="Clear audio cache",
    description="Evict all cached sentence audio.  Returns the number of entries removed.",
    tags=["cache"],
)
async def cache_clear() -> JSONResponse:
    n = audio_cache.clear()
    return JSONResponse({"cleared": n})


# ── Entrypoint ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS optimised server.")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Listen port (default {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--host", default=config.HOST,
        help=f"Bind address (default {config.HOST}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _active_port = args.port
    uvicorn.run(
        "fast_server:app",
        host=args.host,
        port=args.port,
        ws_ping_interval=config.WS_PING_INTERVAL,
        ws_ping_timeout=config.WS_PING_TIMEOUT,
        log_level="warning",
    )
