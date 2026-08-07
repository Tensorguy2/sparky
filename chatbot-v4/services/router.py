"""
Call router.

Decides whether an in-progress call should switch to a specialist state.
Uses a single forced tool-call (via llm_service.call_tool) on a dedicated,
cheap model so routing latency/cost stays decoupled from the conversation
model. The router only changes which instruction set + context are active;
the voice and conversation history are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import config
from models.flow import Flow
from services import llm_service

logger = logging.getLogger(__name__)

_TOOL_NAME = "route_call"
_STAY = "stay"


@dataclass
class RouteDecision:
    target: Optional[str]  # state name to switch to, or None to stay
    confidence: float
    reason: str

    @property
    def should_switch(self) -> bool:
        return self.target is not None


def _build_system_prompt(flow: Flow, current_state: str) -> str:
    lines = [
        "You are the silent call router for an inbound voice line.",
        "Read the conversation so far and decide which specialist should handle",
        "the caller, by calling the route_call tool. Do not talk to the caller.",
        "",
        f"The call is currently in the '{current_state}' state.",
        "",
        "Available call types:",
    ]
    for st in flow.route_targets():
        lines.append(f"- {st.name} ({st.label}): {st.route_when}")
    lines.append(
        f"- {_STAY}: the caller's need is unclear, off-topic, or the current "
        "state is already correct. Choose this when no specialist clearly fits."
    )
    lines.append("")
    lines.append(
        "Set confidence in [0,1] to reflect how sure you are. Prefer 'stay' "
        "when the caller has not clearly expressed a need that matches a type."
    )
    return "\n".join(lines)


def _tool_parameters(flow: Flow) -> dict:
    enum = [st.name for st in flow.route_targets()] + [_STAY]
    return {
        "type": "object",
        "properties": {
            "call_type": {
                "type": "string",
                "enum": enum,
                "description": "The state to route the call to, or 'stay'.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this routing decision, 0 to 1.",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for the decision (a few words).",
            },
        },
        "required": ["call_type", "confidence"],
    }


async def route_intent(
    flow: Flow,
    messages: List[dict],
    current_state: str,
    *,
    allow_reroute: bool = True,
    reroute_min_confidence: float = 0.7,
) -> RouteDecision:
    """Return the routing decision for the current conversation.

    - From the entry (reception) state: honor the model's choice directly.
    - From a specialist state: only switch when allow_reroute is set and the
      model is confident (>= reroute_min_confidence), to avoid flapping.
    """
    targets = {st.name for st in flow.route_targets()}
    if not targets or not messages:
        return RouteDecision(None, 1.0, "no targets / empty conversation")

    in_specialist = current_state in targets

    try:
        args = await llm_service.call_tool(
            model=config.ROUTER_MODEL,
            system_prompt=_build_system_prompt(flow, current_state),
            messages=messages,
            tool_name=_TOOL_NAME,
            tool_description="Route the inbound call to the right specialist.",
            parameters=_tool_parameters(flow),
        )
    except Exception:
        logger.exception("Router tool-call failed; staying in current state.")
        return RouteDecision(None, 0.0, "router error")

    if not args:
        return RouteDecision(None, 0.0, "no tool call")

    choice = str(args.get("call_type", _STAY)).strip()
    try:
        confidence = float(args.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(args.get("reason", "")).strip()

    # Normalize: unknown choice or current state -> stay.
    if choice == _STAY or choice == current_state or choice not in targets:
        return RouteDecision(None, confidence, reason or "stay")

    # Guard specialist -> specialist re-routes against flapping.
    if in_specialist:
        if not allow_reroute:
            return RouteDecision(None, confidence, "reroute disabled")
        if confidence < reroute_min_confidence:
            return RouteDecision(
                None, confidence,
                f"reroute below threshold ({confidence:.2f})",
            )

    return RouteDecision(choice, confidence, reason or f"route to {choice}")
