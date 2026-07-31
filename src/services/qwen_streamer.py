import asyncio
import queue
import logging
from typing import Optional

import torch
import numpy as np
from transformers.generation.streamers import BaseStreamer
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

logger = logging.getLogger(__name__)

class QwenAudioStreamer(BaseStreamer):
    """
    A custom streamer for Qwen3-TTS that intercepts the autoregressively 
    generated talker codes, accumulates them, and performs sliding/cumulative 
    decoding to yield raw PCM chunks via an internal thread-safe queue.
    """
    def __init__(
        self,
        model: Qwen3TTSModel,
        voice_clone_prompt_dict: dict = None,
        decode_chunk_frames: int = 12,
    ):
        """
        Args:
           model: Qwen3TTSModel instance containing the speech_tokenizer.
           voice_clone_prompt_dict: Voice clone dict to prepend ref_codes.
           decode_chunk_frames: Number of talker codes (frames) to accumulate 
                                before yielding a new audio chunk.
        """
        self.model = model
        self.voice_clone_prompt_dict = voice_clone_prompt_dict or {}
        self.decode_chunk_frames = decode_chunk_frames
        
        self.accumulated_codes = None
        self.last_decoded_frames = 0
        self.last_yielded_samples = 0
        self.out_queue = queue.Queue()
        self.fs = None # Will be set on first decode

    def put(self, value: torch.Tensor):
        c_ids = value.clone()
        if c_ids.dim() == 2 and c_ids.shape[0] == 1:
            c_ids = c_ids.squeeze(0)

        c_ids = c_ids.unsqueeze(0) # Force to (1, num_codebooks)

        # Drop EOS frames — the sub-codebook values are undefined for the EOS token and
        # decode to noise/garbage audio.  HuggingFace's generate() stops AFTER calling
        # forward() with the EOS token, so the frame has already been emitted here.
        try:
            eos_id = self.model.model.config.talker_config.codec_eos_token_id
            if int(c_ids[0, 0].item()) == eos_id:
                return
        except AttributeError:
            pass  # If config path doesn't exist, fall through and accumulate normally

        if self.accumulated_codes is None:
            self.accumulated_codes = c_ids
        else:
            self.accumulated_codes = torch.cat([self.accumulated_codes, c_ids], dim=0)

        current_frames = self.accumulated_codes.shape[0]
        if current_frames - self.last_decoded_frames >= self.decode_chunk_frames:
            self.last_decoded_frames = current_frames
            self._decode_and_enqueue()

    def end(self):
        if self.accumulated_codes is not None and self.accumulated_codes.shape[0] > self.last_decoded_frames:
            self._decode_and_enqueue()
        self.out_queue.put(None)

    def _decode_and_enqueue(self):
        codes_for_decode = self.accumulated_codes
        
        # Prepend reference code if voice clone is active
        ref_code_list = self.voice_clone_prompt_dict.get("ref_code", None)
        if ref_code_list is not None and len(ref_code_list) > 0 and ref_code_list[0] is not None:
            ref_code = ref_code_list[0].to(codes_for_decode.device)
            codes_for_decode_with_ref = torch.cat([ref_code, codes_for_decode], dim=0)
            
            wavs_all, fs = self.model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode_with_ref}])
            wav = wavs_all[0]
            if not hasattr(self, 'ref_audio_samples'):
                # Initialize the sample cutoff explicitly to sequence perfectly
                dummy_wav, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": ref_code}])
                self.ref_audio_samples = len(dummy_wav[0])
                self.last_yielded_samples = self.ref_audio_samples
        else:
            wavs_all, fs = self.model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
            wav = wavs_all[0]
            if not hasattr(self, 'ref_audio_samples'):
                self.ref_audio_samples = 0
                self.last_yielded_samples = 0
            
        self.fs = fs
        
        if len(wav) > self.last_yielded_samples:
            chunk = wav[self.last_yielded_samples:]
            self.last_yielded_samples = len(wav)
            self.out_queue.put(chunk)

    async def get_async(self):
        """Asynchronously yield audio chunks and sample rate."""
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.out_queue.get)
            if chunk is None:
                break
            yield chunk, self.fs
