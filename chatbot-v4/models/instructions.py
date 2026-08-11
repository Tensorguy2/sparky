"""
Custom instruction set manager.

Loads instruction profiles from JSON files in the instructions/ directory.
Each profile defines a system prompt and per-model generation parameters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelParams:
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class InstructionSet:
    id: str
    name: str
    system_prompt: str
    models: Dict[str, ModelParams] = field(default_factory=dict)

    def params_for(self, model_id: str) -> ModelParams:
        """Return params for a specific model, falling back to defaults."""
        return self.models.get(model_id, ModelParams())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "models": {
                k: {"temperature": v.temperature, "max_tokens": v.max_tokens}
                for k, v in self.models.items()
            },
        }


class InstructionManager:
    """Loads and manages instruction set profiles from disk."""

    def __init__(self, directory: Path):
        self._dir = directory
        self._sets: Dict[str, InstructionSet] = {}
        self.reload()

    def reload(self) -> None:
        self._sets.clear()
        self._dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                iset = self._parse(data)
                self._sets[iset.id] = iset
                logger.info("Loaded instruction set: %s (%s)", iset.id, iset.name)
            except Exception:
                logger.exception("Failed to load instruction set from %s", path)

    def get(self, set_id: str) -> Optional[InstructionSet]:
        return self._sets.get(set_id)

    def list_all(self) -> List[InstructionSet]:
        return list(self._sets.values())

    def save(self, iset: InstructionSet) -> None:
        path = self._dir / f"{iset.id}.json"
        path.write_text(
            json.dumps(iset.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._sets[iset.id] = iset
        logger.info("Saved instruction set: %s", iset.id)

    def save_from_dict(self, data: dict) -> InstructionSet:
        """Parse, persist, and register an instruction set from a JSON object."""
        iset = self._parse(data)
        if not iset.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("id must contain only letters, numbers, underscores, and hyphens")
        self.save(iset)
        return iset

    def delete(self, set_id: str) -> bool:
        path = self._dir / f"{set_id}.json"
        if path.exists():
            path.unlink()
        removed = self._sets.pop(set_id, None)
        return removed is not None

    @staticmethod
    def _parse(data: dict) -> InstructionSet:
        models: Dict[str, ModelParams] = {}
        for model_id, params in data.get("models", {}).items():
            models[model_id] = ModelParams(
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 2048),
            )
        return InstructionSet(
            id=data["id"],
            name=data["name"],
            system_prompt=data["system_prompt"],
            models=models,
        )
