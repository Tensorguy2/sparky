"""
Call router — deferred mode.

Decides whether an in-progress call should switch to a specialist state.
Uses a single forced tool-call (via llm_service.call_tool) on a dedicated,
cheap model so routing latency/cost stays decoupled from the conversation
model. The router only changes which instruction set + context are active;
the voice and conversation history are untouched.

In deferred mode (the default), route_intent returns a "stay" decision
immediately and runs the tool-call in the background. The completed decision
is picked up on the following turn. This removes the router from the critical
path at the cost of routing landing one turn late — acceptable because routing
changes persona and context, not the answer itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from models.flow import Flow
from services import llm_service

logger = logging.getLogger(__name__)

_TOOL_NAME = "route_call"
_STAY = "stay"

_DEFER = os.getenv("ROUTER_DEFER", "true").lower() in ("1", "true", "yes")

# Pending deferred decisions keyed by session id.
_pending: Dict[str, RouteDecision] = {}
_pending_tasks: Dict[str, asyncio.Task] = {}


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


def _parse_decision(
    args: Optional[dict],
    targets: set,
    current_state: str,
    in_specialist: bool,
    allow_reroute: bool,
    reroute_min_confidence: float,
) -> RouteDecision:
    """Shared logic to turn tool-call output into a RouteDecision."""
    if not args:
        return RouteDecision(None, 0.0, "no tool call")

    choice = str(args.get("call_type", _STAY)).strip()
    try:
        confidence = float(args.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(args.get("reason", "")).strip()

    if choice == _STAY or choice == current_state or choice not in targets:
        return RouteDecision(None, confidence, reason or "stay")

    if in_specialist:
        if not allow_reroute:
            return RouteDecision(None, confidence, "reroute disabled")
        if confidence < reroute_min_confidence:
            return RouteDecision(
                None, confidence,
                f"reroute below threshold ({confidence:.2f})",
            )

    return RouteDecision(choice, confidence, reason or f"route to {choice}")


async def _run_deferred(
    session_id: str,
    flow: Flow,
    messages: List[dict],
    current_state: str,
    allow_reroute: bool,
    reroute_min_confidence: float,
) -> None:
    """Background coroutine that completes the router tool-call."""
    targets = {st.name for st in flow.route_targets()}
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
        logger.exception("Deferred router tool-call failed; staying.")
        _pending[session_id] = RouteDecision(None, 0.0, "router error (deferred)")
        return
    finally:
        _pending_tasks.pop(session_id, None)

    decision = _parse_decision(
        args, targets, current_state, in_specialist,
        allow_reroute, reroute_min_confidence,
    )
    _pending[session_id] = decision
    logger.info("Deferred route for session=%s: target=%s conf=%.2f (%s)",
                session_id, decision.target, decision.confidence, decision.reason)


async def route_intent(
    flow: Flow,
    messages: List[dict],
    current_state: str,
    *,
    allow_reroute: bool = True,
    reroute_min_confidence: float = 0.7,
    session_id: str = "",
) -> RouteDecision:
    """Return the routing decision for the current conversation.

    When ROUTER_DEFER is true (the default), this returns immediately with a
    "stay" decision and runs the actual tool-call in the background. Any
    pending decision from a prior turn is consumed and returned instead, so
    routing takes effect one turn late.

    - From the entry (reception) state: honor the model's choice directly.
    - From a specialist state: only switch when allow_reroute is set and the
      model is confident (>= reroute_min_confidence), to avoid flapping.
    """
    targets = {st.name for st in flow.route_targets()}
    if not targets or not messages:
        return RouteDecision(None, 1.0, "no targets / empty conversation")

    # Check for a completed deferred decision from a prior turn.
    deferred = _pending.pop(session_id, None) if session_id else None
    if deferred is not None:
        logger.info("Applying deferred route for session=%s: target=%s (%s)",
                    session_id, deferred.target, deferred.reason)
        # Still fire a new deferred for this turn's context, then return.
        if _DEFER:
            old = _pending_tasks.pop(session_id, None)
            if old and not old.done():
                old.cancel()
            task = asyncio.create_task(
                _run_deferred(
                    session_id, flow, list(messages), current_state,
                    allow_reroute, reroute_min_confidence,
                )
            )
            _pending_tasks[session_id] = task
        return deferred

    if _DEFER:
        # Cancel any already-running deferred task for this session.
        old = _pending_tasks.pop(session_id, None)
        if old and not old.done():
            old.cancel()

        task = asyncio.create_task(
            _run_deferred(
                session_id, flow, list(messages), current_state,
                allow_reroute, reroute_min_confidence,
            )
        )
        _pending_tasks[session_id] = task
        logger.info("Router deferred for session=%s (will apply next turn)", session_id)
        return RouteDecision(None, 0.0, "deferred")

    # Synchronous fallback (ROUTER_DEFER=false)
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

    return _parse_decision(
        args, targets, current_state, in_specialist,
        allow_reroute, reroute_min_confidence,
    )
