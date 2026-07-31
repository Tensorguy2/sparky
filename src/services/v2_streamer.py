"""
v2 streamer for Qwen3-TTS with lower latency output delivery.

Improvements over FastAudioStreamer:
  - asyncio.Queue output bridge (eliminates run_in_executor per chunk)
  - asyncio.Event cancellation for immediate abort on timeout/barge-in
  - Initial decode after 2 frames (vs 4 in FastAudioStreamer)
  - Per-decode timing instrumentation
"""

import asyncio
import logging
import queue
import time
from typing import Optional

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

logger = logging.getLogger(__name__)


class V2AudioStreamer(BaseStreamer):
    """Drop-in streamer with asyncio-native output and faster first audio."""

    def __init__(
        self,
        model: Qwen3TTSModel,
        voice_clone_prompt_dict: Optional[dict] = None,
        decode_chunk_frames: int = 12,
        initial_decode_frames: int = 2,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.model = model
        self.voice_clone_prompt_dict = voice_clone_prompt_dict or {}
        self.decode_chunk_frames = decode_chunk_frames
        self.initial_decode_frames = initial_decode_frames

        self.accumulated_codes: Optional[torch.Tensor] = None
        self.last_decoded_frames: int = 0
        self.last_yielded_samples: int = 0
        self.fs: Optional[int] = None
        self._first_decode_done: bool = False

        self._cancelled: bool = False
        self._cancel_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = loop

        # asyncio.Queue for zero-overhead async consumption
        self._async_queue: Optional[asyncio.Queue] = None
        # fallback thread-safe queue for put() calls from the GPU thread
        self._thread_queue: queue.Queue = queue.Queue()

        self._ref_audio_samples = self._precompute_ref_offset()
        self.last_yielded_samples = max(self._ref_audio_samples, 0)

        # Instrumentation
        self._decode_times: list = []
        self._first_chunk_time: Optional[float] = None
        self._start_time: float = time.perf_counter()

    def bind_async(self, loop: asyncio.AbstractEventLoop, async_queue: asyncio.Queue, cancel_event: asyncio.Event):
        """Bind the asyncio primitives after construction (must be called from the event loop thread)."""
        self._loop = loop
        self._async_queue = async_queue
        self._cancel_event = cancel_event

    # -- Pre-computation --------------------------------------------------------

    def _precompute_ref_offset(self) -> int:
        ref_code_list = self.voice_clone_prompt_dict.get("ref_code")
        if not ref_code_list or ref_code_list[0] is None:
            return 0
        try:
            dummy_wav, _ = self.model.model.speech_tokenizer.decode(
                [{"audio_codes": ref_code_list[0]}]
            )
            n = len(dummy_wav[0])
            logger.debug("v2 pre-computed ref_audio_samples=%d", n)
            return n
        except Exception:
            logger.warning("v2 ref_audio_samples pre-computation failed.", exc_info=True)
            return -1

    # -- Cancellation -----------------------------------------------------------

    def cancel(self) -> None:
        """Signal stop. Safe from any thread."""
        self._cancelled = True
        self._enqueue_sentinel()

    def _enqueue_sentinel(self) -> None:
        """Put None to unblock any waiting consumer."""
        if self._loop and self._async_queue is not None:
            try:
                self._loop.call_soon_threadsafe(self._async_queue.put_nowait, None)
            except RuntimeError:
                pass
        else:
            self._thread_queue.put(None)

    # -- BaseStreamer interface --------------------------------------------------

    def put(self, value: torch.Tensor) -> None:
        if self._cancelled:
            return

        c_ids = value.clone()
        if c_ids.dim() == 2 and c_ids.shape[0] == 1:
            c_ids = c_ids.squeeze(0)
        c_ids = c_ids.unsqueeze(0)

        try:
            eos_id = self.model.model.config.talker_config.codec_eos_token_id
            if int(c_ids[0, 0].item()) == eos_id:
                return
        except AttributeError:
            pass

        if self.accumulated_codes is None:
            self.accumulated_codes = c_ids
        else:
            self.accumulated_codes = torch.cat([self.accumulated_codes, c_ids], dim=0)

        new_frames = self.accumulated_codes.shape[0] - self.last_decoded_frames
        threshold = (
            self.initial_decode_frames
            if not self._first_decode_done
            else self.decode_chunk_frames
        )
        if new_frames >= threshold:
            self.last_decoded_frames = self.accumulated_codes.shape[0]
            self._decode_and_enqueue()
            self._first_decode_done = True

    def end(self) -> None:
        if self._cancelled:
            return
        if (
            self.accumulated_codes is not None
            and self.accumulated_codes.shape[0] > self.last_decoded_frames
        ):
            self._decode_and_enqueue()
        self._enqueue_sentinel()

    # -- Internal decode --------------------------------------------------------

    def _decode_and_enqueue(self) -> None:
        t0 = time.perf_counter()
        codes = self.accumulated_codes

        ref_code_list = self.voice_clone_prompt_dict.get("ref_code")
        has_ref = (
            ref_code_list is not None
            and len(ref_code_list) > 0
            and ref_code_list[0] is not None
        )

        if has_ref:
            ref_code = ref_code_list[0].to(codes.device)
            codes_with_ref = torch.cat([ref_code, codes], dim=0)
            wavs_all, fs = self.model.model.speech_tokenizer.decode(
                [{"audio_codes": codes_with_ref}]
            )
            wav = wavs_all[0]

            if self._ref_audio_samples < 0:
                dummy_wav, _ = self.model.model.speech_tokenizer.decode(
                    [{"audio_codes": ref_code}]
                )
                self._ref_audio_samples = len(dummy_wav[0])
                self.last_yielded_samples = self._ref_audio_samples
        else:
            wavs_all, fs = self.model.model.speech_tokenizer.decode(
                [{"audio_codes": codes}]
            )
            wav = wavs_all[0]

        self.fs = fs

        if len(wav) > self.last_yielded_samples:
            chunk = wav[self.last_yielded_samples:]
            self.last_yielded_samples = len(wav)

            elapsed = time.perf_counter() - t0
            self._decode_times.append(elapsed)
            if self._first_chunk_time is None:
                self._first_chunk_time = time.perf_counter() - self._start_time
                logger.info(
                    "v2 first chunk: %.3f s (decode %.3f s, %d samples)",
                    self._first_chunk_time, elapsed, len(chunk),
                )

            # Deliver via asyncio queue (thread-safe)
            if self._loop and self._async_queue is not None:
                try:
                    self._loop.call_soon_threadsafe(
                        self._async_queue.put_nowait, chunk
                    )
                except RuntimeError:
                    self._thread_queue.put(chunk)
            else:
                self._thread_queue.put(chunk)

    # -- Async consumer ---------------------------------------------------------

    async def get_async(self):
        """Yield (audio_chunk, sample_rate) tuples until stream ends or cancelled."""
        q = self._async_queue
        cancel_evt = self._cancel_event

        if q is None:
            # Fallback to run_in_executor if bind_async wasn't called
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(None, self._thread_queue.get)
                if chunk is None:
                    break
                yield chunk, self.fs
            return

        while True:
            if cancel_evt and cancel_evt.is_set():
                break

            try:
                chunk = await asyncio.wait_for(q.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if cancel_evt and cancel_evt.is_set():
                    break
                continue

            if chunk is None:
                break
            yield chunk, self.fs

    # -- Metrics ----------------------------------------------------------------

    @property
    def metrics(self) -> dict:
        return {
            "first_chunk_s": self._first_chunk_time,
            "decode_count": len(self._decode_times),
            "decode_avg_ms": (
                1000 * sum(self._decode_times) / len(self._decode_times)
                if self._decode_times else 0
            ),
            "decode_max_ms": (
                1000 * max(self._decode_times) if self._decode_times else 0
            ),
        }
