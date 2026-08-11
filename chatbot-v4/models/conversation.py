"""
Conversation history manager.

Maintains a rolling list of user/assistant turns with token-aware truncation
so the full history always fits within the target model's context window.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import tiktoken

# Rough per-model context limits (input tokens we budget for history).
# We leave headroom for the system prompt + new response.
_CONTEXT_BUDGETS: dict[str, int] = {
    # OpenAI
    "gpt-5.5": 200_000,
    "gpt-5.4": 200_000,
    "gpt-5.4-mini": 200_000,
    "gpt-5.4-nano": 120_000,
    "gpt-5-mini": 120_000,
    "gpt-5-nano": 120_000,
    "gpt-4.1": 900_000,
    "gpt-4.1-mini": 900_000,
    "gpt-4.1-nano": 900_000,
    "gpt-4o": 120_000,
    "gpt-4o-mini": 120_000,
    "o3": 180_000,
    "o4-mini": 180_000,
    # Anthropic
    "claude-opus-4-7": 900_000,
    "claude-sonnet-4-6": 900_000,
    "claude-haiku-4-5": 180_000,
}

_DEFAULT_BUDGET = 100_000

_enc: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def estimate_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    tokens: int = 0

    def to_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """Thread-safe, token-aware conversation buffer."""

    def __init__(self, max_turns: int = 30):
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    # -- mutators -------------------------------------------------------------

    def add_user_message(self, text: str) -> Turn:
        turn = Turn(role="user", content=text, tokens=estimate_tokens(text))
        self._turns.append(turn)
        self._enforce_max_turns()
        return turn

    def add_or_coalesce_user_message(
        self, text: str, *, window_s: float = 2.5
    ) -> Turn:
        """Append to the prior user turn if it arrived within window_s (STT fragments)."""
        content = (text or "").strip()
        if not content:
            return self.add_user_message(text)
        if self._turns:
            last = self._turns[-1]
            if (
                last.role == "user"
                and (time.time() - float(last.timestamp or 0)) <= window_s
            ):
                merged = f"{last.content.rstrip()} {content}".strip()
                last.content = merged
                last.tokens = estimate_tokens(merged)
                last.timestamp = time.time()
                return last
        return self.add_user_message(content)

    def add_assistant_message(self, text: str, model: str = "") -> Turn:
        turn = Turn(
            role="assistant",
            content=text,
            model=model,
            tokens=estimate_tokens(text),
        )
        self._turns.append(turn)
        self._enforce_max_turns()
        return turn

    def clear(self) -> None:
        self._turns.clear()

    # -- accessors ------------------------------------------------------------

    def get_messages(self, model: str = "") -> List[dict]:
        """Return message list trimmed to fit the model's context budget."""
        budget = _CONTEXT_BUDGETS.get(model, _DEFAULT_BUDGET)
        messages: list[dict] = []
        total = 0
        for turn in reversed(self._turns):
            total += turn.tokens
            if total > budget:
                break
            messages.append(turn.to_message())
        messages.reverse()
        return messages

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_assistant_message(self) -> Optional[str]:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    # -- serialization --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps([asdict(t) for t in self._turns], indent=2)

    @classmethod
    def from_json(cls, data: str, max_turns: int = 30) -> "ConversationManager":
        mgr = cls(max_turns=max_turns)
        for item in json.loads(data):
            mgr._turns.append(Turn(**item))
        mgr._enforce_max_turns()
        return mgr

    @classmethod
    def from_turn_dicts(
        cls, items: List[dict], max_turns: int = 30,
    ) -> "ConversationManager":
        mgr = cls(max_turns=max_turns)
        for item in items:
            content = item.get("content", "")
            mgr._turns.append(
                Turn(
                    role=item.get("role", "user"),
                    content=content,
                    timestamp=item.get("timestamp", time.time()),
                    model=item.get("model", ""),
                    tokens=item.get("tokens") or estimate_tokens(content),
                )
            )
        mgr._enforce_max_turns()
        return mgr

    def to_turn_dicts(self) -> List[dict]:
        return [asdict(t) for t in self._turns]

    # -- internals ------------------------------------------------------------

    def _enforce_max_turns(self) -> None:
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
