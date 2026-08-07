"""
Local LLM backend (vLLM, OpenAI-compatible).

Serves GPU-resident models from the DGX Spark alongside the cloud providers.
Each model gets its own port so more than one can be resident at a time.

Disabled entirely by setting LOCAL_LLM_ENABLED=false, in which case no local
models are registered and nothing here is ever reached.
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncGenerator, List, Optional

import openai

from models.instructions import ModelParams

logger = logging.getLogger(__name__)

PROVIDER = "local"

ENABLED = os.getenv("LOCAL_LLM_ENABLED", "true").lower() in ("1", "true", "yes")


# Both are MoE models with ~3B active parameters, which is what makes them
# viable here: GB10 decode is memory-bandwidth-bound, so active parameter
# count -- not total -- sets tokens/sec.
LOCAL_MODELS: dict[str, dict] = {
    "qwen3.6-35b-a3b": {
        "repo": "Qwen/Qwen3.6-35B-A3B-FP8",
        "port": 8100,
        # vLLM --served-model-name; keeping it identical to the UI id means
        # the request needs no translation.
        "served_name": "qwen3.6-35b-a3b",
        "supports_thinking": True,
    },
    "nemotron-3-nano-30b-a3b": {
        "repo": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "port": 8101,
        "served_name": "nemotron-3-nano-30b-a3b",
        "supports_thinking": True,
    },
}

_clients: dict[str, openai.AsyncOpenAI] = {}


def is_local_model(model: str) -> bool:
    return ENABLED and model in LOCAL_MODELS


def available_model_ids() -> List[str]:
    return list(LOCAL_MODELS.keys()) if ENABLED else []


def base_url_for(model: str) -> str:
    """Per-model override wins, then a global override, then the default port."""
    env_key = f"LOCAL_LLM_URL_{model.upper().replace('-', '_').replace('.', '_')}"
    if os.getenv(env_key):
        return os.environ[env_key]
    if os.getenv("LOCAL_LLM_BASE_URL"):
        return os.environ["LOCAL_LLM_BASE_URL"]
    port = LOCAL_MODELS[model]["port"]
    return f"http://127.0.0.1:{port}/v1"


def _get_client(model: str) -> openai.AsyncOpenAI:
    url = base_url_for(model)
    if url not in _clients:
        _clients[url] = openai.AsyncOpenAI(
            base_url=url,
            api_key="EMPTY",
            max_retries=0,
            timeout=120.0,
        )
    return _clients[url]


def _extra_body(model: str) -> dict:
    """Suppress reasoning traces.

    Both families emit a chain-of-thought block by default. In a voice loop
    that text would be spoken aloud and would delay the first audible token by
    seconds, so thinking is turned off at the chat-template level.
    """
    spec = LOCAL_MODELS[model]
    body: dict = {}
    if spec.get("supports_thinking"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


async def health(model: str) -> dict:
    """Report whether this model's vLLM server is up and serving it."""
    spec = LOCAL_MODELS.get(model)
    if not spec:
        return {"model": model, "ok": False, "error": "unknown model"}
    try:
        client = _get_client(model)
        resp = await client.models.list()
        served = [m.id for m in resp.data]
        return {
            "model": model,
            "ok": True,
            "url": base_url_for(model),
            "served": served,
        }
    except Exception as e:
        return {
            "model": model,
            "ok": False,
            "url": base_url_for(model),
            "error": f"{type(e).__name__}: {e}",
        }


async def health_all() -> List[dict]:
    return [await health(m) for m in available_model_ids()]


async def stream_chat(
    model: str,
    system_prompt: str,
    messages: List[dict],
    params: ModelParams,
) -> AsyncGenerator[str, None]:
    client = _get_client(model)
    served = LOCAL_MODELS[model]["served_name"]
    api_messages = [{"role": "system", "content": system_prompt}] + messages

    logger.info(
        "Local stream | model=%s url=%s temp=%.2f max_tokens=%d turns=%d",
        model, base_url_for(model), params.temperature,
        params.max_tokens, len(messages),
    )

    try:
        stream = await client.chat.completions.create(
            model=served,
            messages=api_messages,
            stream=True,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            extra_body=_extra_body(model),
        )
    except openai.APIConnectionError as e:
        logger.error("Local LLM unreachable at %s: %s", base_url_for(model), e)
        raise RuntimeError(
            f"Local model '{model}' is not running. Start it with "
            f"scripts/start_local_llm.sh {model}"
        ) from e

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


async def call_tool(
    model: str,
    system_prompt: str,
    messages: List[dict],
    tool_name: str,
    tool_description: str,
    parameters: dict,
    max_tokens: int,
) -> Optional[dict]:
    """Return structured arguments matching ``parameters``.

    Uses response_format/json_schema rather than the tool-call API: the grammar
    is enforced during sampling, so a malformed response is not representable.
    Forced tool_choice on local models has no such guarantee.

    Note that vLLM's older ``guided_json`` field is silently ignored on current
    builds -- the model free-forms and the caller gets plausible-looking JSON
    with entirely invented keys. response_format is the supported path.
    """
    client = _get_client(model)
    served = LOCAL_MODELS[model]["served_name"]

    instruction = (
        f"{system_prompt}\n\n"
        f"Task: {tool_description}\n"
        f"Respond with a single JSON object matching the required schema. "
        f"Output no other text."
    )
    api_messages = [{"role": "system", "content": instruction}] + messages

    # Constrain output to the declared keys so the decoder cannot spend tokens
    # on fields the caller will discard.
    schema = dict(parameters)
    schema.setdefault("additionalProperties", False)

    try:
        resp = await client.chat.completions.create(
            model=served,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": tool_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            extra_body=_extra_body(model),
        )
    except openai.APIConnectionError as e:
        logger.error("Local LLM unreachable for tool call: %s", e)
        return None

    choice = resp.choices[0] if resp.choices else None
    content = getattr(choice.message, "content", None) if choice else None
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("call_tool: local model returned non-JSON: %.200s", content)
        return None
    return parsed if isinstance(parsed, dict) else None
