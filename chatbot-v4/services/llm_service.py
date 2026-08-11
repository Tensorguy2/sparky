"""
Multi-provider LLM service.

Routes generation requests to OpenAI or Anthropic based on the requested
model, yielding streamed text tokens as an async generator.

On connect/timeout/API failures before the first token, automatically tries
``config.LLM_FALLBACK_MODELS`` so phone calls do not go silent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, List, Optional, Union

import anthropic
import httpx
import openai

import config
from models.instructions import ModelParams

logger = logging.getLogger(__name__)

_openai_client: openai.AsyncOpenAI | None = None
_anthropic_client: anthropic.AsyncAnthropic | None = None

# Skip cache_control on tiny system prompts (below Haiku's practical floor).
_CACHE_MIN_CHARS = 6000

_RETRYABLE = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.TransportError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _http_timeout() -> httpx.Timeout:
    connect = float(getattr(config, "LLM_CONNECT_TIMEOUT_S", 5) or 5)
    total = float(getattr(config, "LLM_REQUEST_TIMEOUT_S", 20) or 20)
    return httpx.Timeout(total, connect=connect)


def _get_openai() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=_http_timeout(),
            max_retries=int(getattr(config, "LLM_MAX_RETRIES", 1) or 0),
        )
    return _openai_client


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=_http_timeout(),
            max_retries=int(getattr(config, "LLM_MAX_RETRIES", 1) or 0),
        )
    return _anthropic_client


def _model_has_key(model: str) -> bool:
    try:
        provider = config.provider_for_model(model)
    except ValueError:
        return False
    if provider == "openai":
        return bool(config.OPENAI_API_KEY)
    if provider == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    return False


def _fallback_chain(primary: str) -> list[str]:
    """Primary first, then configured fallbacks (deduped, key-available only)."""
    seen: set[str] = set()
    chain: list[str] = []
    for model in [primary, *(getattr(config, "LLM_FALLBACK_MODELS", None) or [])]:
        m = (model or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        if not _model_has_key(m):
            logger.warning("Skipping LLM candidate %s (missing API key)", m)
            continue
        chain.append(m)
    return chain


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    # Anthropic/OpenAI sometimes wrap status 529 / 503
    status = getattr(exc, "status_code", None)
    if status in (408, 429, 500, 502, 503, 504, 529):
        return True
    return False


def _openai_uses_max_completion_tokens(model: str) -> bool:
    """GPT-5.x, GPT-4.1+, and o-series models use max_completion_tokens."""
    m = model.lower()
    if m.startswith(("gpt-5", "gpt-4.1")):
        return True
    if m.startswith(("o1", "o3", "o4")):
        return True
    return False


def _openai_supports_temperature(model: str) -> bool:
    """Reasoning / GPT-5 family models often reject custom temperature."""
    m = model.lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return False
    return True


def _anthropic_supports_temperature(model: str) -> bool:
    """Opus 4.7+ deprecates the temperature parameter."""
    m = model.lower()
    if "opus" in m:
        return False
    return True


def _cache_control_block() -> dict[str, str]:
    block: dict[str, str] = {"type": "ephemeral"}
    if getattr(config, "ANTHROPIC_CACHE_TTL", "5m") == "1h":
        block["ttl"] = "1h"
    return block


def _anthropic_system(
    system_prompt: str,
    *,
    dynamic_suffix: str = "",
) -> Union[str, List[dict[str, Any]]]:
    """Wrap system prompt with prompt-cache breakpoint when enabled.

    When ``dynamic_suffix`` is provided (coach / call_facts), only the static
    prefix is cached so Operator coaching does not bust the pack cache.
    """
    static = system_prompt or ""
    dynamic = (dynamic_suffix or "").strip()
    if not getattr(config, "ANTHROPIC_PROMPT_CACHE", True):
        if static and dynamic:
            return f"{static}\n\n{dynamic}"
        return static or dynamic
    # Tiny prompts: no cache_control.
    if len(static) < _CACHE_MIN_CHARS and not dynamic:
        return static
    if len(static) < _CACHE_MIN_CHARS:
        # Nothing worth caching — send combined plain text.
        return f"{static}\n\n{dynamic}".strip() if dynamic else static
    blocks: List[dict[str, Any]] = [
        {
            "type": "text",
            "text": static,
            "cache_control": _cache_control_block(),
        }
    ]
    if dynamic:
        blocks.append({"type": "text", "text": dynamic})
    return blocks


def _log_cache_usage(usage: Any, *, model: str, kind: str) -> None:
    if usage is None:
        return
    created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    logger.info(
        "Anthropic %s usage | model=%s input=%d output=%d cache_write=%d cache_read=%d",
        kind,
        model,
        inp,
        out,
        created,
        read,
    )


async def warm_prompt_cache(
    model: str,
    system_prompt: str,
    params: ModelParams | None = None,
) -> None:
    """Prefill Anthropic cache so the first real turn hits a warm prefix.

    Uses max_tokens=1 with a dummy user message; breakpoint stays on system
    (not the placeholder) so the next turn shares the same cache key.
    Temperature must match the live turn or the cache entry may not hit.
    """
    if not getattr(config, "ANTHROPIC_PROMPT_CACHE", True):
        return
    if not getattr(config, "ANTHROPIC_CACHE_WARMUP", True):
        return
    if not system_prompt or len(system_prompt) < _CACHE_MIN_CHARS:
        return
    try:
        if config.provider_for_model(model) != "anthropic":
            return
    except ValueError:
        return
    if not config.ANTHROPIC_API_KEY:
        return

    client = _get_anthropic()
    system = _anthropic_system(system_prompt)
    if isinstance(system, str):
        return

    p = params or ModelParams()
    create_kwargs: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
    }
    if _anthropic_supports_temperature(model):
        create_kwargs["temperature"] = p.temperature

    try:
        resp = await client.messages.create(**create_kwargs)
        _log_cache_usage(getattr(resp, "usage", None), model=model, kind="warmup")
    except Exception:
        logger.exception("Anthropic prompt-cache warmup failed (model=%s)", model)


async def _stream_provider(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams,
    *,
    dynamic_system: str = "",
) -> AsyncGenerator[str, None]:
    provider = config.provider_for_model(model)
    if provider == "openai":
        combined = system_prompt
        if dynamic_system:
            combined = f"{system_prompt}\n\n{dynamic_system}" if system_prompt else dynamic_system
        async for token in _stream_openai(model, combined, messages, params):
            yield token
    elif provider == "anthropic":
        async for token in _stream_anthropic(
            model, system_prompt, messages, params, dynamic_suffix=dynamic_system,
        ):
            yield token
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def stream_chat(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams | None = None,
    *,
    dynamic_system: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens.

    Tries ``model`` first, then ``LLM_FALLBACK_MODELS`` if the provider fails
    before producing any tokens (timeout / connection / overloaded). Mid-stream
    failures after the first token are not retried (avoids double speech).

    Args:
        model: Model ID (e.g. "gpt-4o", "claude-sonnet-4-20250514")
        system_prompt: Static system prefix (pack) — Anthropic-cached when large
        messages: Conversation history as [{"role": ..., "content": ...}]
        params: Temperature and max_tokens overrides
        dynamic_system: Uncached suffix (coach / call_facts)

    Yields:
        Individual text tokens/chunks as they arrive.
    """
    p = params or ModelParams()
    chain = _fallback_chain(model)
    if not chain:
        raise RuntimeError(f"No LLM candidates available for model={model!r}")

    last_err: BaseException | None = None
    for idx, candidate in enumerate(chain):
        produced = False
        try:
            async for token in _stream_provider(
                candidate, system_prompt, messages, p, dynamic_system=dynamic_system,
            ):
                if not produced:
                    produced = True
                    if idx > 0:
                        logger.warning(
                            "LLM fallback engaged: %s -> %s",
                            model,
                            candidate,
                        )
                yield token
            return
        except Exception as exc:
            last_err = exc
            if produced:
                # Already spoke — do not restart on another model.
                raise
            if idx + 1 >= len(chain) or not _is_retryable(exc):
                raise
            logger.warning(
                "LLM %s failed before tokens (%s: %s); trying fallback %s",
                candidate,
                type(exc).__name__,
                exc,
                chain[idx + 1],
            )
            continue

    assert last_err is not None
    raise last_err


async def _stream_openai(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams,
) -> AsyncGenerator[str, None]:
    client = _get_openai()

    api_messages = [{"role": "system", "content": system_prompt}] + messages

    use_max_completion = _openai_uses_max_completion_tokens(model)
    logger.info(
        "OpenAI stream | model=%s temp=%.2f limit=%d (%s) turns=%d",
        model,
        params.temperature,
        params.max_tokens,
        "max_completion_tokens" if use_max_completion else "max_tokens",
        len(messages),
    )

    create_kwargs: dict = {
        "model": model,
        "messages": api_messages,
        "stream": True,
    }
    if use_max_completion:
        create_kwargs["max_completion_tokens"] = params.max_tokens
    else:
        create_kwargs["max_tokens"] = params.max_tokens
    if _openai_supports_temperature(model):
        create_kwargs["temperature"] = params.temperature

    stream = await client.chat.completions.create(**create_kwargs)

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


async def call_tool(
    model: str,
    system_prompt: str,
    messages: List[dict],
    *,
    tool_name: str,
    tool_description: str,
    parameters: dict,
    max_tokens: int = 256,
) -> Optional[dict]:
    """Force a single tool call and return its parsed arguments (or None).

    Non-streaming. Used by the router to get a structured routing decision
    without complicating the streaming chat path. ``parameters`` is a JSON
    Schema object describing the tool's arguments.
    """
    provider = config.provider_for_model(model)
    if provider == "openai":
        return await _call_tool_openai(
            model, system_prompt, messages,
            tool_name, tool_description, parameters, max_tokens,
        )
    elif provider == "anthropic":
        return await _call_tool_anthropic(
            model, system_prompt, messages,
            tool_name, tool_description, parameters, max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


async def _call_tool_openai(
    model: str,
    system_prompt: str,
    messages: List[dict],
    tool_name: str,
    tool_description: str,
    parameters: dict,
    max_tokens: int,
) -> Optional[dict]:
    client = _get_openai()
    api_messages = [{"role": "system", "content": system_prompt}] + messages
    create_kwargs: dict = {
        "model": model,
        "messages": api_messages,
        "tools": [{
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": parameters,
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
    }
    if _openai_uses_max_completion_tokens(model):
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens

    resp = await client.chat.completions.create(**create_kwargs)
    choice = resp.choices[0] if resp.choices else None
    tool_calls = getattr(choice.message, "tool_calls", None) if choice else None
    if not tool_calls:
        return None
    try:
        return json.loads(tool_calls[0].function.arguments or "{}")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("call_tool: failed to parse OpenAI tool arguments")
        return None


async def _call_tool_anthropic(
    model: str,
    system_prompt: str,
    messages: List[dict],
    tool_name: str,
    tool_description: str,
    parameters: dict,
    max_tokens: int,
) -> Optional[dict]:
    client = _get_anthropic()
    resp = await client.messages.create(
        model=model,
        system=_anthropic_system(system_prompt),
        messages=messages,
        max_tokens=max_tokens,
        tools=[{
            "name": tool_name,
            "description": tool_description,
            "input_schema": parameters,
        }],
        tool_choice={"type": "tool", "name": tool_name},
    )
    _log_cache_usage(getattr(resp, "usage", None), model=model, kind="tool")
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            inp = getattr(block, "input", None)
            if isinstance(inp, dict):
                return inp
    return None


async def _stream_anthropic(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams,
    *,
    dynamic_suffix: str = "",
) -> AsyncGenerator[str, None]:
    client = _get_anthropic()
    system = _anthropic_system(system_prompt, dynamic_suffix=dynamic_suffix)
    cached = isinstance(system, list)

    logger.info(
        "Anthropic stream | model=%s temp=%.2f max_tokens=%d turns=%d cache=%s",
        model,
        params.temperature,
        params.max_tokens,
        len(messages),
        "on" if cached else "off",
    )

    create_kwargs: dict = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": params.max_tokens,
    }
    if _anthropic_supports_temperature(model):
        create_kwargs["temperature"] = params.temperature

    async with client.messages.stream(**create_kwargs) as stream:
        async for text in stream.text_stream:
            yield text
        try:
            final = await stream.get_final_message()
            _log_cache_usage(getattr(final, "usage", None), model=model, kind="stream")
        except Exception:
            logger.debug("Anthropic stream: could not read final usage", exc_info=True)
