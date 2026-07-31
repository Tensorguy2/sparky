"""
Multi-provider LLM service.

Routes generation requests to OpenAI or Anthropic based on the requested
model, yielding streamed text tokens as an async generator.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List, Optional

import anthropic
import openai

import config
from models.instructions import ModelParams

logger = logging.getLogger(__name__)

_openai_client: openai.AsyncOpenAI | None = None
_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_openai() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


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


async def stream_chat(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response tokens.

    Args:
        model: Model ID (e.g. "gpt-4o", "claude-sonnet-4-20250514")
        system_prompt: The full system prompt (instructions + context)
        messages: Conversation history as [{"role": ..., "content": ...}]
        params: Temperature and max_tokens overrides

    Yields:
        Individual text tokens/chunks as they arrive.
    """
    provider = config.provider_for_model(model)
    p = params or ModelParams()

    if provider == "openai":
        async for token in _stream_openai(model, system_prompt, messages, p):
            yield token
    elif provider == "anthropic":
        async for token in _stream_anthropic(model, system_prompt, messages, p):
            yield token
    else:
        raise ValueError(f"Unsupported provider: {provider}")


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
        system=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        tools=[{
            "name": tool_name,
            "description": tool_description,
            "input_schema": parameters,
        }],
        tool_choice={"type": "tool", "name": tool_name},
    )
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
) -> AsyncGenerator[str, None]:
    client = _get_anthropic()

    logger.info(
        "Anthropic stream | model=%s temp=%.2f max_tokens=%d turns=%d",
        model, params.temperature, params.max_tokens, len(messages),
    )

    create_kwargs: dict = {
        "model": model,
        "system": system_prompt,
        "messages": messages,
        "max_tokens": params.max_tokens,
    }
    if _anthropic_supports_temperature(model):
        create_kwargs["temperature"] = params.temperature

    async with client.messages.stream(**create_kwargs) as stream:
        async for text in stream.text_stream:
            yield text
