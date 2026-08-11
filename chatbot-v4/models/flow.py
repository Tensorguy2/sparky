"""
Call-flow manager.

A "flow" describes a single inbound call experience: an entry state
(reception) that greets the caller generically, plus a set of specialist
states. Each state maps to an instruction set + context file (the existing
prompt-building primitives). The router decides when to move between states;
the voice and conversation history are shared across the whole call.

Flows are loaded from JSON files in the flows/ directory. Example schema:

    {
      "id": "calltypes",
      "name": "...",
      "entry_state": "reception",
      "greeting": "Hi, thanks for calling...",
      "allow_reroute": true,
      "states": {
        "reception": {"label": "Reception", "instruction_set": "reception",
                      "context": "reception", "routable": false},
        "typeA": {"label": "Type A", "instruction_set": "typeA",
                  "context": "typeA", "route_when": "..."}
      }
    }

A state is a *route target* (selectable by the router) when it defines a
non-empty "route_when" and is not explicitly marked "routable": false.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FlowState:
    name: str
    label: str
    instruction_set: str
    context: str
    route_when: str = ""
    routable: bool = True

    @property
    def is_route_target(self) -> bool:
        return self.routable and bool(self.route_when.strip())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "instruction_set": self.instruction_set,
            "context": self.context,
            "route_when": self.route_when,
            "is_route_target": self.is_route_target,
        }


@dataclass
class Flow:
    id: str
    name: str
    entry_state: str
    greeting: str
    allow_reroute: bool = True
    states: Dict[str, FlowState] = field(default_factory=dict)

    def state(self, name: str) -> Optional[FlowState]:
        return self.states.get(name)

    def entry(self) -> FlowState:
        return self.states[self.entry_state]

    def route_targets(self) -> List[FlowState]:
        """States the router may switch to, in stable (sorted) order."""
        return [s for _, s in sorted(self.states.items()) if s.is_route_target]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entry_state": self.entry_state,
            "greeting": self.greeting,
            "allow_reroute": self.allow_reroute,
            "states": {name: s.to_dict() for name, s in self.states.items()},
        }


class FlowManager:
    """Loads and manages call flows from disk."""

    def __init__(self, directory: Path, default_flow: str):
        self._dir = directory
        self._default_flow = default_flow
        self._flows: Dict[str, Flow] = {}
        self.reload()

    def reload(self) -> None:
        self._flows.clear()
        self._dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                flow = self._parse(data)
                self._flows[flow.id] = flow
                logger.info(
                    "Loaded flow: %s (entry=%s, %d state(s), %d route target(s))",
                    flow.id, flow.entry_state, len(flow.states),
                    len(flow.route_targets()),
                )
            except Exception:
                logger.exception("Failed to load flow from %s", path)

    def get(self, flow_id: str) -> Optional[Flow]:
        return self._flows.get(flow_id)

    def default(self) -> Optional[Flow]:
        return self._flows.get(self._default_flow) or next(
            iter(self._flows.values()), None
        )

    def list_all(self) -> List[Flow]:
        return list(self._flows.values())

    @staticmethod
    def _parse(data: dict) -> Flow:
        states: Dict[str, FlowState] = {}
        for name, st in data.get("states", {}).items():
            states[name] = FlowState(
                name=name,
                label=st.get("label", name),
                instruction_set=st["instruction_set"],
                context=st.get("context", ""),
                route_when=st.get("route_when", ""),
                routable=st.get("routable", True),
            )
        entry_state = data["entry_state"]
        if entry_state not in states:
            raise ValueError(
                f"entry_state '{entry_state}' is not defined in states"
            )
        return Flow(
            id=data["id"],
            name=data.get("name", data["id"]),
            entry_state=entry_state,
            greeting=data.get("greeting", ""),
            allow_reroute=bool(data.get("allow_reroute", True)),
            states=states,
        )
