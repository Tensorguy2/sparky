"""
Centralised logging configuration.

Call ``configure()`` once at process startup (before any other import that
touches a logger).  After that every ``logging.getLogger(__name__)`` call
across the codebase picks up the right handlers and format automatically.

Output
------
- Console (stdout): INFO and above, human-readable one-liner.
- Rotating log file (src/logs/server.log): DEBUG and above, same format.
  Each file caps at 10 MB; up to 5 backups are kept.

Format
------
  2026-04-13 10:23:45.123 | INFO     | voice_service     : Voice 'mikey' cached in 1.82 s
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  timestamp (ms)            level     logger name (padded)   message
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional

# Guard against double-configure (uvicorn re-imports server.py in its worker
# process on Windows, which would hit the module-level configure() call twice).
_configured: bool = False


# ── Formatter ──────────────────────────────────────────────────────────────────

_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s: %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_FMT, datefmt=_DATE)


# ── Third-party loggers to quiet down ─────────────────────────────────────────

_QUIET = {
    "uvicorn.access": logging.WARNING,      # per-request HTTP access lines are noise
    "uvicorn.error": logging.INFO,
    "transformers": logging.WARNING,
    "torch": logging.WARNING,
    "accelerate": logging.WARNING,
    "filelock": logging.WARNING,
    "urllib3": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    # torio probes for FFmpeg 4/5/6 on every import and logs each failure at DEBUG
    "torio._extension.utils": logging.ERROR,
    "torio": logging.WARNING,
    # huggingface_hub logs retry attempts at WARNING; promote to ERROR so
    # transient network timeouts during model load don't flood the console
    "huggingface_hub.utils._http": logging.ERROR,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def configure(
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure the root logger.  Safe to call multiple times — only the first
    call has any effect (subsequent calls are no-ops).

    Args:
        log_dir:       Directory for the rotating log file.  Omit to disable
                       file logging (useful in tests / interactive use).
        console_level: Minimum level printed to stdout.  Default: INFO.
        file_level:    Minimum level written to the log file.  Default: DEBUG.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # root captures everything; handlers filter
    root.handlers.clear()                 # wipe any handlers set by basicConfig

    fmt = _make_formatter()

    # Console ──────────────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file ────────────────────────────────────────────────────────────
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, "server.log"),
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Silence noisy third-party libraries ──────────────────────────────────────
    for name, level in _QUIET.items():
        logging.getLogger(name).setLevel(level)

    logging.getLogger(__name__).debug(
        "Logging configured — console=%s, file=%s, log_dir=%s",
        logging.getLevelName(console_level),
        logging.getLevelName(file_level) if log_dir else "disabled",
        log_dir or "—",
    )
