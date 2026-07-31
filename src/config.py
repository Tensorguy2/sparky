import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
NARRATOR_DIR = os.path.join(PROJECT_ROOT, "narrator")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "voices.db")

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = os.path.join(os.path.dirname(BASE_DIR), "models", "Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "sdpa"

# ── Server ─────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 25565
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 30

# ── Audio cache (sentence-level, stored in RAM) ────────────────────────────────
# With 100 GB system RAM and the model using ~6.8 GB VRAM, 10 GB for audio cache
# leaves ample headroom.  Each cached sentence is a float32 numpy array; at
# 24 kHz mono, 1 second ≈ 96 KB, so 10 GB ≈ ~100 000 seconds of cached audio.
AUDIO_CACHE_MAX_BYTES: int = 10 * 1024 ** 3  # 10 GB

# ── Built-in voices (always loaded into RAM at startup) ────────────────────────
BUILTIN_VOICES: dict = {
    "mikey": {
        "name": "Mikey",
        "ref_audio": os.path.join(ASSETS_DIR, "mikey_sample.wav"),
        "ref_text": (
            "So I was going through that today. uhmm Actually, I said today I am still doing it. "
            "I just I pulled up when you said 5 minutes I was like cool. I am going to up my laptop "
            "see how started the training job"
        ),
    }
}
