"""
Local faster-whisper STT (CUDA CTranslate2 when available).
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


def _preload_ct2_libraries() -> None:
    """Load CUDA CT2 shared libs early so import finds them on aarch64."""
    ct2_lib = os.path.expanduser("~/.local/ctranslate2/lib")
    if not os.path.isdir(ct2_lib):
        return
    for name in sorted(os.listdir(ct2_lib)):
        if name.endswith(".so") or ".so." in name:
            try:
                ctypes.CDLL(os.path.join(ct2_lib, name), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.debug("Skip preload %s: %s", name, exc)


_preload_ct2_libraries()

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-stt")
_model = None
_resolved_device: Optional[str] = None
_resolved_compute: Optional[str] = None
_loaded_size: Optional[str] = None
_model_lock = __import__("threading").Lock()


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_whisper_device() -> tuple[str, str]:
    """Pick device/compute_type from env and runtime CUDA support."""
    want = (config.WHISPER_DEVICE or "auto").lower()
    compute = (config.WHISPER_COMPUTE_TYPE or "auto").lower()
    if want == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    elif want == "cuda" and not _cuda_available():
        logger.warning(
            "WHISPER_DEVICE=cuda but CTranslate2 has no CUDA devices; using CPU. "
            "On aarch64, install CUDA ctranslate2 (see scripts/install_stt_cuda.sh)."
        )
        device = "cpu"
    else:
        device = want

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _load_model(model_size: Optional[str] = None):
    global _model, _loaded_size, _resolved_device, _resolved_compute
    size = model_size or config.WHISPER_MODEL_SIZE
    with _model_lock:
        if _model is not None and _loaded_size == size:
            return _model
        from faster_whisper import WhisperModel

        device, compute = resolve_whisper_device()
        logger.info(
            "Loading faster-whisper model=%s device=%s compute=%s",
            size,
            device,
            compute,
        )
        _model = WhisperModel(size, device=device, compute_type=compute)
        _loaded_size = size
        _resolved_device = device
        _resolved_compute = compute
        logger.info("faster-whisper model loaded (%s / %s).", device, compute)
        return _model


def get_stt_status(model_size: Optional[str] = None) -> dict:
    """Runtime STT device info for health checks."""
    size = model_size or config.WHISPER_MODEL_SIZE
    device, compute = resolve_whisper_device()
    try:
        import ctranslate2

        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_devices = 0
    return {
        "model": f"local:{size}",
        "device": _resolved_device or device,
        "compute_type": _resolved_compute or compute,
        "cuda_devices": cuda_devices,
        "loaded": _model is not None and _loaded_size == size,
        "streaming": False,
    }


def preload() -> None:
    """Load the active local STT backend at startup and warm it up."""
    default = (config.DEFAULT_STT_MODEL or "").strip()
    if "parakeet" in default:
        from services import parakeet_stt

        parakeet_stt.preload()
        return

    model = _load_model()
    audio = np.zeros(config.WHISPER_SAMPLE_RATE // 2, dtype=np.float32)
    list(
        model.transcribe(
            audio,
            beam_size=1,
            language=config.STT_LANGUAGE,
            vad_filter=False,
        )[0]
    )
    logger.info("faster-whisper warmup complete.")


def _pcm16_chunks_to_float32(chunks: List[bytes], sample_rate: int) -> np.ndarray:
    """Concatenate int16 LE PCM chunks and convert to float32 mono @ Whisper SR."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
    target_sr = config.WHISPER_SAMPLE_RATE
    if sample_rate == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _merge_segment_texts(parts: List[str]) -> str:
    out: List[str] = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        if out and t[0].islower() and not out[-1].endswith((".", "?", "!", ":")):
            out.append(" " + t)
        else:
            if out:
                out.append(" ")
            out.append(t)
    return "".join(out).strip()


def transcribe_sync(
    pcm_chunks: List[bytes],
    sample_rate: int = 24000,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
) -> str:
    """
    Transcribe buffered PCM16 audio. Runs on the worker thread.
    """
    import time

    audio = _pcm16_chunks_to_float32(pcm_chunks, sample_rate)
    if len(audio) < config.WHISPER_MIN_SAMPLES:
        return ""

    model = _load_model(model_size)
    lang = language or config.STT_LANGUAGE
    # Client-side VAD already segments utterances. Keep Whisper vad_filter off.
    # Greedy decode (beam_size=1) for lower latency on short turns.
    t0 = time.perf_counter()
    segments, _info = model.transcribe(
        audio,
        beam_size=1,
        best_of=1,
        temperature=0,
        language=lang,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    parts = [seg.text for seg in segments]
    text = _merge_segment_texts(parts)
    stt_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Whisper STT | model=%s audio=%.2fs stt_ms=%.0f text=%r",
        model_size or config.WHISPER_MODEL_SIZE,
        len(audio) / config.WHISPER_SAMPLE_RATE,
        stt_ms,
        text[:80],
    )
    return text


async def transcribe_async(
    pcm_chunks: List[bytes],
    sample_rate: int = 24000,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: transcribe_sync(
            pcm_chunks,
            sample_rate=sample_rate,
            language=language,
            model_size=model_size,
        ),
    )


class AudioRecorder:
    """Simple PCM16 chunk buffer used by the WebSocket voice path."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._chunks: List[bytes] = []
        self.active = False

    def start(self) -> None:
        self._chunks.clear()
        self.active = True

    def append(self, data: bytes) -> None:
        if self.active:
            self._chunks.append(bytes(data))

    def stop(self) -> List[bytes]:
        self.active = False
        return list(self._chunks)

    @property
    def duration_s(self) -> float:
        n = sum(len(c) for c in self._chunks)
        # int16 mono → 2 bytes/sample
        return (n / 2) / float(self.sample_rate)
