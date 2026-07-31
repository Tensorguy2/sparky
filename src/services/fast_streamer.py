"""
Optimised streamer for Qwen3-TTS.

Improvements over QwenAudioStreamer:
  - **Adaptive first-chunk threshold**: decodes after fewer initial frames
    (default 4 vs 12) so the first audio reaches the client ~500-700 ms sooner.
  - **Pre-computed ref_audio_samples**: the sample offset for voice-clone
    ref_code is calculated once in __init__, not on the hot path of the first
    decode call.
"""

import asyncio
import logging
import queue
from typing import Optional

import numpy as np
import torch
from transformers.generation.streamers import BaseStreamer

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

logger = logging.getLogger(__name__)


class FastAudioStreamer(BaseStreamer):
    """Drop-in replacement for QwenAudioStreamer with lower TTFA."""

    def __init__(
        self,
        model: Qwen3TTSModel,
        voice_clone_prompt_dict: Optional[dict] = None,
        decode_chunk_frames: int = 12,
        initial_decode_frames: int = 4,
    ):
        self.model = model
        self.voice_clone_prompt_dict = voice_clone_prompt_dict or {}
        self.decode_chunk_frames = decode_chunk_frames
        self.initial_decode_frames = initial_decode_frames

        self.accumulated_codes: Optional[torch.Tensor] = None
        self.last_decoded_frames: int = 0
        self.last_yielded_samples: int = 0
        self.out_queue: queue.Queue = queue.Queue()
        self.fs: Optional[int] = None
        self._first_decode_done: bool = False

        self._cancelled: bool = False

        self._ref_audio_samples = self._precompute_ref_offset()
        self.last_yielded_samples = max(self._ref_audio_samples, 0)

    # ── Pre-computation ───────────────────────────────────────────────────────

    def _precompute_ref_offset(self) -> int:
        """Decode ref_code once to learn its sample length.

        Returns -1 if pre-computation fails (will fall back at first decode).
        """
        ref_code_list = self.voice_clone_prompt_dict.get("ref_code")
        if not ref_code_list or ref_code_list[0] is None:
            return 0
        try:
            dummy_wav, _ = self.model.model.speech_tokenizer.decode(
                [{"audio_codes": ref_code_list[0]}]
            )
            n = len(dummy_wav[0])
            logger.debug(
                "Pre-computed ref_audio_samples=%d (ref_code shape=%s)",
                n, ref_code_list[0].shape,
            )
            return n
        except Exception:
            logger.warning(
                "ref_audio_samples pre-computation failed; will retry on first decode.",
                exc_info=True,
            )
            return -1

    # ── Cancellation ──────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal the streamer to stop emitting chunks.

        Safe to call from any thread or coroutine. The next call to
        ``get_async()`` will stop iterating immediately.
        """
        self._cancelled = True
        self.out_queue.put(None)  # unblock any waiting get_async() consumer

    # ── BaseStreamer interface ─────────────────────────────────────────────────

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
            self.accumulated_codes = torch.cat(
                [self.accumulated_codes, c_ids], dim=0,
            )

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
            return  # cancel() already put None in the queue
        if (
            self.accumulated_codes is not None
            and self.accumulated_codes.shape[0] > self.last_decoded_frames
        ):
            self._decode_and_enqueue()
        self.out_queue.put(None)

    # ── Internal decode ───────────────────────────────────────────────────────

    def _decode_and_enqueue(self) -> None:
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
            self.out_queue.put(chunk)

    # ── Async consumer ────────────────────────────────────────────────────────

    async def get_async(self):
        """Yield ``(audio_chunk, sample_rate)`` tuples until the stream ends."""
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.out_queue.get)
            if chunk is None:
                break
            yield chunk, self.fs
