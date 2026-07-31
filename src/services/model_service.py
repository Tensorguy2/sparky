"""
Singleton that owns the Qwen3TTSModel and the single-worker ThreadPoolExecutor.

GPU ops are blocking and must be serialized — one executor thread is intentional.
With 100 GB RAM there's no pressure to evict anything; all tensors stay live.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
import threading

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from services.qwen_streamer import QwenAudioStreamer

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model: Optional[Qwen3TTSModel] = None
        self.executor: Optional[ThreadPoolExecutor] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self, model_id: str, device: str, dtype: torch.dtype, attn_impl: str) -> None:
        logger.info(
            "Loading model | id=%s  device=%s  dtype=%s  attn=%s",
            model_id, device, dtype, attn_impl,
        )
        t0 = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        # One worker: GPU ops serialize anyway; extra workers waste VRAM.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2f s", elapsed)

    def shutdown(self) -> None:
        logger.info("Shutting down executor ...")
        if self.executor:
            self.executor.shutdown(wait=False)
        logger.info("Executor shut down.")

    # ── Sync inference helpers (call via run_in_executor) ──────────────────────

    def compute_prompt_sync(
        self,
        ref_audio: str,
        ref_text: Optional[str],
    ) -> VoiceClonePromptItem:
        """
        Blocking — computes speaker embedding + ICL codes for a reference clip.
        Expensive (~1–3 s); always cache the result via VoiceService.
        """
        logger.debug(
            "compute_prompt_sync | ref_audio=%r  ref_text_len=%s",
            ref_audio,
            len(ref_text) if ref_text else 0,
        )
        t0 = time.perf_counter()
        item = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )[0]
        elapsed = time.perf_counter() - t0
        logger.info("Prompt computed in %.2f s | ref_audio=%r", elapsed, ref_audio)
        
        # Move tensors to CPU to save VRAM 
        if item.ref_code is not None:
            item.ref_code = item.ref_code.cpu()
        if item.ref_spk_embedding is not None:
            item.ref_spk_embedding = item.ref_spk_embedding.cpu()
            
        return item

    def infer_sync(
        self,
        text: str,
        language: str,
        prompt_item: Optional[VoiceClonePromptItem],
    ) -> Tuple[np.ndarray, int]:
        """Blocking TTS inference. Returns (waveform, sample_rate)."""
        mode = "voice_clone" if prompt_item is not None else "x_vector_only"
        logger.debug(
            "infer_sync | mode=%s  language=%s  text_len=%d  preview=%r",
            mode, language, len(text), text[:50],
        )
        t0 = time.perf_counter()

        if prompt_item is not None:
            try:
                # Move to GPU for inference
                if prompt_item.ref_code is not None:
                    prompt_item.ref_code = prompt_item.ref_code.to(self.model.device)
                if prompt_item.ref_spk_embedding is not None:
                    prompt_item.ref_spk_embedding = prompt_item.ref_spk_embedding.to(self.model.device)

                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=[prompt_item],
                )
            finally:
                # Move back to CPU after
                if prompt_item.ref_code is not None:
                    prompt_item.ref_code = prompt_item.ref_code.cpu()
                if prompt_item.ref_spk_embedding is not None:
                    prompt_item.ref_spk_embedding = prompt_item.ref_spk_embedding.cpu()
        else:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=None,
                ref_audio=None,
                x_vector_only_mode=True,
            )

        audio = wavs[0]
        elapsed = time.perf_counter() - t0
        duration_s = len(audio) / sr
        rtf = elapsed / duration_s if duration_s > 0 else 0
        logger.info(
            "Inference done | %.2f s wall  →  %.2f s audio  (RTF %.3f)  sr=%d",
            elapsed, duration_s, rtf, sr,
        )
        return audio, sr

    def infer_stream(
        self,
        text: str,
        language: str,
        prompt_item: Optional[VoiceClonePromptItem] = None,
    ) -> QwenAudioStreamer:
        """
        Starts text-to-speech generation on a background thread using QwenAudioStreamer,
        and immediately returns the streamer to allow asynchronous reading of chunks via `get_async()`.
        """
        # Wait for any previous streaming thread to fully exit before launching a new one.
        # Concurrent GPU access from a lingering thread can corrupt model state.
        prev = getattr(self, "_last_infer_thread", None)
        if prev is not None and prev.is_alive():
            prev.join(timeout=2.0)
        self._last_infer_thread = None

        # Cap max_new_tokens proportional to text length to prevent runaway generation.
        # At 12 Hz, ~0.6 codec frames/char observed; 5x safety factor still beats the
        # default 2048 (≈170 s) by an order of magnitude when EOS is never generated.
        _max_new_tokens = min(600, max(150, len(text) * 3))

        # Create a thread-local prompt mapped to GPU to prevent shared-cache race-conditions
        local_prompt = None
        prompt_dict = {}
        if prompt_item is not None:
            local_prompt = VoiceClonePromptItem(
                ref_code=prompt_item.ref_code.to(self.model.device) if prompt_item.ref_code is not None else None,
                ref_spk_embedding=prompt_item.ref_spk_embedding.to(self.model.device) if prompt_item.ref_spk_embedding is not None else None,
                x_vector_only_mode=prompt_item.x_vector_only_mode,
                icl_mode=prompt_item.icl_mode,
                ref_text=prompt_item.ref_text,
            )
            prompt_dict = self.model._prompt_items_to_voice_clone_prompt([local_prompt])

        streamer = QwenAudioStreamer(model=self.model, voice_clone_prompt_dict=prompt_dict, decode_chunk_frames=12)

        def _target(prompt=local_prompt):
            # `prompt` is a default-arg copy of local_prompt captured at call time.
            # Using a default arg avoids the UnboundLocalError that `del local_prompt`
            # would cause if it were referenced as a closure variable.
            try:
                if prompt is not None:
                    self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=[prompt],
                        streamer=streamer,
                        max_new_tokens=_max_new_tokens,
                    )
                else:
                    self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=None,
                        ref_audio=None,
                        x_vector_only_mode=True,
                        streamer=streamer,
                        max_new_tokens=_max_new_tokens,
                    )
            finally:
                if prompt is not None:
                    if prompt.ref_code is not None:
                        del prompt.ref_code
                    if prompt.ref_spk_embedding is not None:
                        del prompt.ref_spk_embedding
                    del prompt

        t = threading.Thread(target=_target, daemon=True)
        self._last_infer_thread = t
        t.start()
        return streamer

    def warmup_sync(self, prompt_item: VoiceClonePromptItem) -> None:
        """Fire a short inference to compile CUDA kernels before the first real request."""
        logger.debug("warmup_sync | starting kernel compilation ...")
        t0 = time.perf_counter()
        
        try:
            if prompt_item.ref_code is not None:
                prompt_item.ref_code = prompt_item.ref_code.to(self.model.device)
            if prompt_item.ref_spk_embedding is not None:
                prompt_item.ref_spk_embedding = prompt_item.ref_spk_embedding.to(self.model.device)

            self.model.generate_voice_clone(
                text="Warmup.",
                language="English",
                voice_clone_prompt=[prompt_item],
            )
        finally:
            if prompt_item.ref_code is not None:
                prompt_item.ref_code = prompt_item.ref_code.cpu()
            if prompt_item.ref_spk_embedding is not None:
                prompt_item.ref_spk_embedding = prompt_item.ref_spk_embedding.cpu()
                
        elapsed = time.perf_counter() - t0
        logger.info("CUDA warmup complete in %.2f s", elapsed)


model_service = ModelService()
