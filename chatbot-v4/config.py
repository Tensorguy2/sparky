"""
Configuration for the v4 voice chatbot (parallel pipeline, external TTS).

Readable replacement for the bytecode-only config.  All values come from
.env / environment variables with sensible defaults matching the v3/v4
deployment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Directories -------------------------------------------------------------
CONTEXT_DIR = BASE_DIR / "context"
INSTRUCTIONS_DIR = BASE_DIR / "instructions"
FLOWS_DIR = BASE_DIR / "flows"
SESSIONS_DIR = BASE_DIR / "sessions"
WEB_DIR = BASE_DIR / "web"

HF_CACHE_DIR = BASE_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# --- Server ------------------------------------------------------------------
CHATBOT_HOST = os.getenv("CHATBOT_HOST", "0.0.0.0")
CHATBOT_PORT = int(os.getenv("CHATBOT_PORT", "8020"))
ALLOW_ADMIN_RESTART = os.getenv("ALLOW_ADMIN_RESTART", "true").lower() in ("1", "true", "yes")

# --- LLM ---------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "30"))

AVAILABLE_MODELS: dict = {
    "openai": [
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5-mini", "gpt-5-nano",
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-4o", "gpt-4o-mini", "o3", "o4-mini",
    ],
    "anthropic": [
        "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
    ],
}


def provider_for_model(model_id: str) -> str:
    """Return 'openai' or 'anthropic' for a given model ID."""
    for provider, models in AVAILABLE_MODELS.items():
        if model_id in models:
            return provider
    if model_id.startswith(("gpt", "o")):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    raise ValueError(f"Unknown model: {model_id}")


# --- STT ---------------------------------------------------------------------
DEFAULT_STT_MODEL = os.getenv("DEFAULT_STT_MODEL", "local:parakeet-tdt-0.6b-v2")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
STT_REALTIME_DELAY = os.getenv("STT_REALTIME_DELAY", "low")
STT_SERVER_VAD: bool = os.getenv("STT_SERVER_VAD", "true").lower() in ("1", "true", "yes")
STT_VAD_SILENCE_MS = int(os.getenv("STT_VAD_SILENCE_MS", "500"))
MIC_SAMPLE_RATE = 24000

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_SAMPLE_RATE = 16000
WHISPER_MIN_SAMPLES = 1600

AVAILABLE_STT_MODELS: list = [
    {"id": "local:parakeet-tdt-0.6b-v2", "label": "Local Parakeet TDT 0.6B v2 (offline, fastest)", "provider": "local", "streaming": False},
    # July 2026 generation — OpenAI's recommended STT models.
    {"id": "gpt-transcribe", "label": "GPT-Transcribe (API, high accuracy, batch)", "provider": "openai", "streaming": False},
    {"id": "gpt-live-transcribe", "label": "GPT-Live-Transcribe (API, streaming, low latency)", "provider": "openai", "streaming": True},
    # Older generation — kept for compatibility.
    {"id": "gpt-realtime-whisper", "label": "GPT Realtime Whisper (streaming, legacy)", "provider": "openai", "streaming": True},
    {"id": "gpt-4o-transcribe", "label": "GPT-4o Transcribe (legacy)", "provider": "openai", "streaming": False},
    {"id": "gpt-4o-mini-transcribe", "label": "GPT-4o Mini Transcribe (legacy)", "provider": "openai", "streaming": False},
    {"id": "local:large-v3-turbo", "label": "Local Whisper large-v3-turbo (offline, best quality)", "provider": "local", "streaming": False},
    {"id": "local:medium.en", "label": "Local Whisper medium.en (offline)", "provider": "local", "streaming": False},
    {"id": "local:base.en", "label": "Local Whisper base.en (offline, fast)", "provider": "local", "streaming": False},
]

# --- TTS (external server, typically v3_server.py on port 25568) -------------
TTS_SERVER_HOST = os.getenv("TTS_SERVER_HOST", "localhost")
TTS_SERVER_PORT = int(os.getenv("TTS_SERVER_PORT", "25568"))
TTS_WS_URL: str = os.getenv("TTS_WS_URL", f"ws://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/ws/tts")
TTS_REST_URL: str = os.getenv("TTS_REST_URL", f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}")
DEFAULT_VOICE_ID = os.getenv("DEFAULT_VOICE_ID", "mikey")

# --- Router / flow -----------------------------------------------------------
ROUTER_ENABLED = os.getenv("ROUTER_ENABLED", "true").lower() in ("1", "true", "yes")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1-mini")
ALLOW_REROUTE = os.getenv("ALLOW_REROUTE", "false").lower() in ("1", "true", "yes")
REROUTE_MIN_CONFIDENCE: float = float(os.getenv("REROUTE_MIN_CONFIDENCE", "0.7"))
DEFAULT_FLOW = os.getenv("DEFAULT_FLOW", "calltypes")

# --- Fillers -----------------------------------------------------------------
FILLERS_ENABLED = os.getenv("FILLERS_ENABLED", "true").lower() in ("1", "true", "yes")
FILLER_PROBABILITY: float = float(os.getenv("FILLER_PROBABILITY", "0.5"))

# --- Sessions ----------------------------------------------------------------
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "last")
