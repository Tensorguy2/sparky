"""
Context file manager.

Loads markdown reference files from the context/ directory and makes them
available for injection into system prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages reference context files that augment the system prompt."""

    def __init__(self, directory: Path):
        self._dir = directory
        self._files: Dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self._files.clear()
        self._dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self._dir.glob("*.md")):
            try:
                content = path.read_text(encoding="utf-8")
                name = path.stem
                self._files[name] = content
                logger.info("Loaded context file: %s (%d chars)", name, len(content))
            except Exception:
                logger.exception("Failed to load context file: %s", path)

    def get(self, name: str) -> Optional[str]:
        return self._files.get(name)

    def list_all(self) -> List[str]:
        return list(self._files.keys())

    def save(self, name: str, content: str) -> None:
        path = self._dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        self._files[name] = content
        logger.info("Saved context file: %s (%d chars)", name, len(content))

    def delete(self, name: str) -> bool:
        path = self._dir / f"{name}.md"
        if path.exists():
            path.unlink()
        removed = self._files.pop(name, None) is not None
        return removed
