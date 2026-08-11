"""
Persist chat sessions to disk so conversations survive page reloads and reconnects.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionSnapshot:
    """Serializable chat session (conversation + UI settings)."""

    session_id: str = "last"
    model: str = ""
    voice_id: str = ""
    instruction_set_id: str = "default"
    context_name: str = "default"
    current_state: str = ""
    language: str = "English"
    tts_enabled: bool = True
    turns: List[dict] = field(default_factory=list)
    updated_at: float = 0.0
    title: str = ""


class SessionStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "last"
        return self.directory / f"{safe}.json"

    def save(self, snapshot: SessionSnapshot) -> None:
        snapshot.updated_at = time.time()
        path = self._path(snapshot.session_id)
        path.write_text(
            json.dumps(asdict(snapshot), indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved session %r (%d turns) -> %s",
            snapshot.session_id,
            len(snapshot.turns),
            path.name,
        )

    def load(self, session_id: str = "last") -> Optional[SessionSnapshot]:
        path = self._path(session_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionSnapshot(**data)
        except Exception:
            logger.exception("Failed to load session from %s", path)
            return None

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[dict]:
        out: List[dict] = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append({
                    "session_id": data.get("session_id", path.stem),
                    "title": data.get("title", path.stem),
                    "turn_count": len(data.get("turns", [])),
                    "updated_at": data.get("updated_at", 0),
                    "model": data.get("model", ""),
                })
            except Exception:
                continue
        return out
