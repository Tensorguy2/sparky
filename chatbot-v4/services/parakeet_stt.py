"""
Local NVIDIA Parakeet TDT 0.6B v2 STT via onnx-asr (CPU ONNX Runtime).

Runs off-GPU so it does not contend with Qwen TTS VRAM. Input is 24 kHz
PCM16 from the client; Parakeet expects 16 kHz float/PCM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

PARAKEET_MODEL_ID = "local:parakeet-tdt-0.6b-v2"
ONNX_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"
TARGET_SR = 16000

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet-stt")
_model = None
_model_lock = __import__("threading").Lock()
_loaded = False


def _ensure_hf_home() -> None:
    """Prefer a project-writable HF cache (home cache may be unwritable)."""
    if os.environ.get("HF_HOME"):
        return
    cache = Path(config.BASE_DIR).resolve().parent / ".hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)


def _load_model():
    global _model, _loaded
    with _model_lock:
        if _model is not None:
            return _model
        _ensure_hf_home()
        import onnx_asr

        logger.info("Loading Parakeet ONNX model=%s", ONNX_MODEL_NAME)
        t0 = time.perf_counter()
        # int8 is same speed as fp32 here; keep default weights.
        _model = onnx_asr.load_model(ONNX_MODEL_NAME)
        _loaded = True
        logger.info("Parakeet loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        return _model


def get_stt_status() -> dict:
    return {
        "model": PARAKEET_MODEL_ID,
        "backend": "onnx-asr",
        "onnx_model": ONNX_MODEL_NAME,
        "device": "cpu",
        "compute_type": "onnx",
        "cuda_devices": 0,
        "loaded": _loaded and _model is not None,
        "streaming": False,
    }


def preload() -> None:
    """Load model and warm up with a short silent clip."""
    model = _load_model()
    silence = np.zeros(TARGET_SR // 2, dtype=np.float32)
    t0 = time.perf_counter()
    _ = model.recognize(silence, sample_rate=TARGET_SR)
    logger.info("Parakeet warmup complete (%.0f ms).", (time.perf_counter() - t0) * 1000)


def _pcm16_chunks_to_float32(chunks: List[bytes], sample_rate: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate == TARGET_SR or len(audio) == 0:
        return audio
    # Linear resample 24 kHz → 16 kHz (same approach as Whisper path).
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(round(duration * TARGET_SR)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def transcribe_sync(
    pcm_chunks: List[bytes],
    sample_rate: int = 24000,
    language: Optional[str] = None,
) -> str:
    """Transcribe buffered PCM16 audio on the worker thread."""
    del language  # Parakeet v2 is English-only; language hint unused.
    audio = _pcm16_chunks_to_float32(pcm_chunks, sample_rate)
    min_samples = max(800, getattr(config, "WHISPER_MIN_SAMPLES", 1600) * TARGET_SR // 16000)
    if len(audio) < min_samples:
        return ""

    model = _load_model()
    t0 = time.perf_counter()
    text = model.recognize(audio, sample_rate=TARGET_SR)
    stt_ms = (time.perf_counter() - t0) * 1000
    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    logger.info(
        "Parakeet STT | audio=%.2fs stt_ms=%.0f text=%r",
        len(audio) / TARGET_SR,
        stt_ms,
        text[:80],
    )
    return text


async def transcribe_async(
    pcm_chunks: List[bytes],
    sample_rate: int = 24000,
    language: Optional[str] = None,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: transcribe_sync(pcm_chunks, sample_rate=sample_rate, language=language),
    )
