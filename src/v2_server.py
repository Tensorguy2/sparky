"""
Qwen3-TTS v2 Streaming Server
===============================
Parallel deployment with latency/reliability improvements.
Runs on port 25567 by default (coexists with fast_server.py on 25565).

    python v2_server.py                # default port 25567
    python v2_server.py --port 25570   # override

Improvements over fast_server.py:
  - asyncio-native streamer output (no thread-pool per chunk read)
  - Parallel cache pre-check (hits served without GPU lock)
  - Per-sentence inference timeout (15s)
  - GPU thread liveness detection
  - Increased instability retries (2) with CUDA cache clear
  - Zero-copy cache reads
  - Latency metrics endpoint (/v2/stats)
"""

# Logging must be configured before any module that calls getLogger()
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
from routers import v2_tts                              # noqa: E402
from routers import voices as voices_router             # noqa: E402
from services.v2_audio_cache import v2_audio_cache      # noqa: E402
from services.model_service import model_service        # noqa: E402
from services.voice_service import voice_service        # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PORT = 25567
_active_port = DEFAULT_PORT


# -- OpenAPI metadata -----------------------------------------------------------

_DESCRIPTION = """
## Qwen3-TTS v2 Streaming API

Same wire protocol as v1, with lower latency and improved reliability.

### Improvements over v1 (fast_server.py)
- **asyncio-native streaming** — no thread-pool overhead per audio chunk
- **Parallel cache pre-check** — cache hits served instantly, no GPU lock wait
- **Per-sentence timeout (15s)** — prevents hung inferences from blocking pipeline
- **GPU thread liveness detection** — detects and handles stuck generation threads
- **Increased retries (2)** with CUDA cache clear between attempts
- **Zero-copy cache reads** — reduced allocation on cache hits
- **Latency metrics** — real-time TTFA and RTF percentiles

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /v2/stats` | TTFA/RTF percentiles, cache stats, request/error counts |
| `GET /cache/stats` | Cache hit rate, byte usage, entry count |
| `POST /cache/clear` | Evict all cached audio |
| `WS /ws/tts` | Synthesis endpoint (wire-compatible with v1) |
"""

_TAGS = [
    {"name": "tts", "description": "Text-to-speech synthesis (v2 pipeline)."},
    {"name": "voices", "description": "CRUD for voice profiles."},
    {"name": "cache", "description": "Audio cache management."},
    {"name": "system", "description": "Health, metrics, and diagnostics."},
]


# -- Lifespan (same startup as fast_server.py, reuses singletons) ---------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    t_boot = time.perf_counter()
    logger.info("=== Qwen3-TTS v2 Server starting up ===")

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
        "=== v2 Server ready in %.2f s | port=%d | voices=%s ===",
        boot_elapsed, _active_port, loaded_ids,
    )

    yield

    logger.info("=== v2 Server shutting down ===")
    model_service.shutdown()
    logger.info("=== Shutdown complete ===")


# -- App ------------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-TTS v2 API",
    version="2.2.0",
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    contact={"name": "Qwen3-TTS v2 Server"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(v2_tts.router)
app.include_router(voices_router.router)


# -- Static web UI (reuses existing web/) ----------------------------------------

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


# -- Narrator avatar app (served under /narrator) ---------------------------

if os.path.isdir(config.NARRATOR_DIR):
    @app.get("/narrator", include_in_schema=False)
    async def narrator_index() -> FileResponse:
        return FileResponse(os.path.join(config.NARRATOR_DIR, "index.html"))

    app.mount(
        "/narrator",
        StaticFiles(directory=config.NARRATOR_DIR),
        name="narrator",
    )


# -- Health ---------------------------------------------------------------------

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
        "server": "v2",
        "model_loaded": model_service.model is not None,
        "voices_in_ram": len(voices_snapshot),
        "voices": voices_snapshot,
        "cache": v2_audio_cache.stats(),
    })


# -- v2 Metrics -----------------------------------------------------------------

@app.get(
    "/v2/stats",
    summary="Latency metrics",
    description="TTFA/RTF percentiles (p50/p95/p99), cache stats, request and error counts.",
    tags=["system"],
)
async def v2_stats() -> JSONResponse:
    return JSONResponse(v2_tts.get_latency_stats())


# -- Cache management -----------------------------------------------------------

@app.get(
    "/cache/stats",
    summary="Audio cache statistics",
    tags=["cache"],
)
async def cache_stats() -> JSONResponse:
    return JSONResponse(v2_audio_cache.stats())


@app.post(
    "/cache/clear",
    summary="Clear audio cache",
    tags=["cache"],
)
async def cache_clear() -> JSONResponse:
    n = v2_audio_cache.clear()
    return JSONResponse({"cleared": n})


# -- Entrypoint -----------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS v2 server.")
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
        "v2_server:app",
        host=args.host,
        port=args.port,
        ws_ping_interval=config.WS_PING_INTERVAL,
        ws_ping_timeout=config.WS_PING_TIMEOUT,
        log_level="warning",
    )
