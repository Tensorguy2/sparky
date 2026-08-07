"""
Standalone Parakeet STT server.

  GET  /                 mic demo page
  GET  /api/status       backend/model status
  POST /api/transcribe   multipart WAV upload -> {text, stt_ms, audio_s}
  WS   /ws               JSON control + binary PCM16 frames

WebSocket protocol (client -> server):
  {"type": "start", "sampleRate": 16000, "model": "parakeet"}   begin utterance
  <binary>                                                      PCM16 mono frames
  {"type": "stop"}                                              end utterance -> transcribe
  {"type": "cancel"}                                            drop buffered audio

Server -> client:
  {"type": "ready"}
  {"type": "transcript", "text": ..., "stt_ms": ..., "audio_s": ..., "model": ...}
  {"type": "error", "message": ...}
"""

from __future__ import annotations

import io
import logging
import os
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import stt_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("parakeet-stt")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Parakeet STT (standalone)")


@app.on_event("startup")
async def _startup() -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stt_service.preload)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse(stt_service.get_status())


@app.post("/api/transcribe")
async def transcribe_upload(file: UploadFile, model: str = "parakeet") -> JSONResponse:
    data = await file.read()
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except wave.Error as exc:
        return JSONResponse({"error": f"invalid WAV: {exc}"}, status_code=400)
    if width != 2:
        return JSONResponse({"error": "only 16-bit PCM WAV supported"}, status_code=400)
    if ch > 1:
        audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, ch).mean(axis=1)
        frames = audio.astype(np.int16).tobytes()
    result = await stt_service.transcribe_async([frames], sample_rate=sr, model=model)
    return JSONResponse(result)


@app.websocket("/ws")
async def ws_stt(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "ready"})
    chunks: list[bytes] = []
    sample_rate = stt_service.TARGET_SR
    model = "parakeet"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                chunks.append(msg["bytes"])
                continue
            if msg.get("text") is None:
                continue
            import json

            try:
                cmd = json.loads(msg["text"])
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "bad JSON"})
                continue

            ctype = cmd.get("type")
            if ctype == "start":
                chunks = []
                sample_rate = int(cmd.get("sampleRate", stt_service.TARGET_SR))
                model = cmd.get("model", "parakeet")
            elif ctype == "cancel":
                chunks = []
            elif ctype == "stop":
                try:
                    result = await stt_service.transcribe_async(
                        chunks, sample_rate=sample_rate, model=model
                    )
                    await ws.send_json({"type": "transcript", **result})
                except Exception as exc:
                    logger.exception("transcribe failed")
                    await ws.send_json({"type": "error", "message": str(exc)})
                chunks = []
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
