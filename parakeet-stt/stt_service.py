"""
Standalone local STT service.

Primary backend: NVIDIA Parakeet TDT 0.6B v2 via onnx-asr (CPU ONNX Runtime,
so it never contends with GPU workloads such as TTS/LLM).
Optional fallback: faster-whisper large-v3-turbo with greedy decoding
(beam_size=1), only if faster-whisper is installed.

No dependencies on the chatbot codebase; configuration is via env vars:
  HF_HOME              model cache dir (default: <project>/.hf_cache)
  PARAKEET_ONNX_MODEL  onnx-asr model name (default: nemo-parakeet-tdt-0.6b-v2)
  WHISPER_MODEL        faster-whisper model name (default: large-v3-turbo)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TARGET_SR = 16000
MIN_SAMPLES = 1600  # 100 ms @ 16 kHz; shorter clips return ""

PARAKEET_ONNX_MODEL = os.environ.get("PARAKEET_ONNX_MODEL", "nemo-parakeet-tdt-0.6b-v2")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Single worker thread: onnxruntime sessions are not re-entrant per session,
# and serialized inference keeps latency predictable.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
_lock = threading.Lock()
_parakeet = None
_whisper = None
_whisper_failed: Optional[str] = None


def _ensure_hf_home() -> None:
    if not os.environ.get("HF_HOME"):
        cache = BASE_DIR / ".hf_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)


def _load_parakeet():
    global _parakeet
    with _lock:
        if _parakeet is not None:
            return _parakeet
        _ensure_hf_home()
        import onnx_asr

        logger.info("Loading Parakeet ONNX model=%s", PARAKEET_ONNX_MODEL)
        t0 = time.perf_counter()
        _parakeet = onnx_asr.load_model(PARAKEET_ONNX_MODEL)
        logger.info("Parakeet loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        return _parakeet


def _load_whisper():
    """Lazy-load faster-whisper; returns None if unavailable."""
    global _whisper, _whisper_failed
    with _lock:
        if _whisper is not None or _whisper_failed is not None:
            return _whisper
        _ensure_hf_home()
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model=%s", WHISPER_MODEL)
            t0 = time.perf_counter()
            try:
                _whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            except Exception:
                _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            logger.info("Whisper loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            _whisper_failed = str(exc)
            logger.warning("Whisper unavailable: %s", exc)
        return _whisper


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return _whisper_failed is None
    except ImportError:
        return False


def get_status() -> dict:
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception:
        providers = []
    return {
        "default_model": "parakeet",
        "parakeet": {
            "onnx_model": PARAKEET_ONNX_MODEL,
            "backend": "onnx-asr",
            "device": "cuda" if "CUDAExecutionProvider" in providers else "cpu",
            "providers": providers,
            "loaded": _parakeet is not None,
        },
        "whisper": {
            "model": WHISPER_MODEL,
            "available": whisper_available(),
            "loaded": _whisper is not None,
            "error": _whisper_failed,
        },
        "sample_rate": TARGET_SR,
    }


def preload() -> None:
    """Load Parakeet and warm it up so the first real request is fast."""
    model = _load_parakeet()
    silence = np.zeros(TARGET_SR // 2, dtype=np.float32)
    t0 = time.perf_counter()
    model.recognize(silence, sample_rate=TARGET_SR)
    logger.info("Parakeet warmup complete (%.0f ms)", (time.perf_counter() - t0) * 1000)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    """Linear resample; adequate for speech going 24k/48k -> 16k."""
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_sr)
    n = max(1, int(round(duration * dst_sr)))
    x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    y = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(y, x, audio).astype(np.float32)


def pcm16_to_float32(chunks: List[bytes], sample_rate: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    return resample(audio, sample_rate, TARGET_SR)


def _transcribe_parakeet(audio: np.ndarray) -> str:
    model = _load_parakeet()
    text = model.recognize(audio, sample_rate=TARGET_SR)
    return (text if isinstance(text, str) else str(text or "")).strip()


def _transcribe_whisper(audio: np.ndarray) -> str:
    model = _load_whisper()
    if model is None:
        raise RuntimeError(f"Whisper unavailable: {_whisper_failed}")
    segments, _info = model.transcribe(audio, beam_size=1, language="en", vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_sync(
    pcm_chunks: List[bytes],
    sample_rate: int = TARGET_SR,
    model: str = "parakeet",
) -> dict:
    """Transcribe buffered PCM16 audio. Returns text plus timing metadata."""
    audio = pcm16_to_float32(pcm_chunks, sample_rate)
    audio_s = len(audio) / TARGET_SR
    if len(audio) < MIN_SAMPLES:
        return {"text": "", "stt_ms": 0.0, "audio_s": audio_s, "model": model}

    t0 = time.perf_counter()
    if model == "whisper":
        text = _transcribe_whisper(audio)
    else:
        text = _transcribe_parakeet(audio)
    stt_ms = (time.perf_counter() - t0) * 1000
    logger.info("STT %s | audio=%.2fs stt_ms=%.0f text=%r", model, audio_s, stt_ms, text[:80])
    return {"text": text, "stt_ms": round(stt_ms, 1), "audio_s": round(audio_s, 2), "model": model}


async def transcribe_async(
    pcm_chunks: List[bytes],
    sample_rate: int = TARGET_SR,
    model: str = "parakeet",
) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: transcribe_sync(pcm_chunks, sample_rate=sample_rate, model=model),
    )
