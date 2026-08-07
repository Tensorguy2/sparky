#!/usr/bin/env python3
"""
Time-to-first-token across every reachable chat model.

TTFT is the number that matters for a voice agent: it sets how long the caller
waits in silence. Total generation time is secondary, because TTS consumes
tokens as they stream.

Models are discovered from both providers so newly released ones show up
without editing this file. Each trial is separated by a pause, since
back-to-back requests reuse a hot connection and flatter the result.

  python scripts/bench_llm_latency.py                    # config models + all Anthropic
  python scripts/bench_llm_latency.py --all-openai       # every OpenAI chat model too
  python scripts/bench_llm_latency.py --trials 6
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

SYSTEM = "You are a friendly voice assistant. Reply in one short sentence."
QUESTION = "What time do you open on Saturdays?"

# Not conversational models: reasoning-heavy "pro" tiers, plus the audio,
# image, transcription and realtime endpoints that do not serve chat.
_EXCLUDE_SUBSTRINGS = (
    "image", "audio", "transcribe", "tts", "realtime", "search", "codex",
    "instruct", "-pro", "whisper", "diarize", "live",
)


async def discover(all_openai: bool) -> list[tuple[str, str]]:
    import anthropic
    import openai

    out: list[tuple[str, str]] = []

    a = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    for m in (await a.models.list(limit=100)).data:
        out.append(("anthropic", m.id))

    if all_openai:
        o = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        ids = [m.id for m in (await o.models.list()).data]
        for mid in sorted(ids):
            if not mid.startswith(("gpt-", "o1", "o3", "o4")):
                continue
            if any(s in mid for s in _EXCLUDE_SUBSTRINGS):
                continue
            # Skip pinned date variants; the alias covers the same weights.
            if any(c.isdigit() for c in mid.split("-")[-1]) and len(mid.split("-")[-1]) >= 4:
                continue
            out.append(("openai", mid))
    else:
        for mid in config.AVAILABLE_MODELS.get("openai", []):
            out.append(("openai", mid))

    seen = set()
    uniq = []
    for provider, mid in out:
        if mid not in seen:
            seen.add(mid)
            uniq.append((provider, mid))
    return uniq


async def ttft_once(model: str, timeout: float) -> tuple[float | None, str | None]:
    import services.llm_service as L
    from models.instructions import ModelParams

    t0 = time.perf_counter()
    try:
        # Reasoning models emit a thinking block before any text. A small budget
        # is consumed entirely by that block and yields no tokens at all, so the
        # cap has to be generous even though only the first token is timed.
        # Their TTFT therefore includes thinking time, which is the honest
        # figure for a voice turn -- the caller waits through it in silence.
        agen = L.stream_chat(model, SYSTEM,
                             [{"role": "user", "content": QUESTION}],
                             ModelParams(temperature=0.7, max_tokens=1024))

        async def first_token() -> float:
            async for tok in agen:
                if tok:
                    return (time.perf_counter() - t0) * 1000
            raise RuntimeError("no tokens")

        ms = await asyncio.wait_for(first_token(), timeout=timeout)
        await agen.aclose()
        return ms, None
    except asyncio.TimeoutError:
        return None, f"timeout>{timeout:.0f}s"
    except Exception as e:
        msg = str(e).replace("\n", " ")[:60]
        return None, f"{type(e).__name__}: {msg}"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--all-openai", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated subset to test")
    args = ap.parse_args()

    models = await discover(args.all_openai)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        known = dict((mid, p) for p, mid in models)
        models = [(known.get(m, "anthropic" if m.startswith("claude") else "openai"), m)
                  for m in wanted]

    # provider_for_model walks AVAILABLE_MODELS, so anything newly discovered
    # has to be registered before llm_service will route it.
    for provider, mid in models:
        bucket = config.AVAILABLE_MODELS.setdefault(provider, [])
        if mid not in bucket:
            bucket.append(mid)

    import services.llm_service as L

    print(f"\n  Benchmarking {len(models)} models, {args.trials} trials each "
          f"(TTFT, lower is better)\n")

    results: dict[str, dict] = {}
    for provider, model in models:
        await L.prewarm(model)
        xs, err = [], None
        for _ in range(args.trials):
            ms, e = await ttft_once(model, args.timeout)
            if ms is None:
                err = e
                break
            xs.append(ms)
            await asyncio.sleep(args.gap)
        if xs:
            results[model] = {"provider": provider, "xs": xs}
            s = sorted(xs)
            print(f"    {model:<32} {provider:<10} p50 {statistics.median(s):7.0f} ms | "
                  f"min {min(s):7.0f} | max {max(s):7.0f}")
        else:
            print(f"    {model:<32} {provider:<10} -- {err}")

    if not results:
        return

    print("\n  Ranked by p50 TTFT:\n")
    ranked = sorted(results.items(), key=lambda kv: statistics.median(kv[1]["xs"]))
    for rank, (model, d) in enumerate(ranked, 1):
        p50 = statistics.median(d["xs"])
        # Under ~700 ms the model is no longer the dominant term in the turn;
        # TTS first-audio (~500 ms here) starts to matter more.
        verdict = "good for voice" if p50 < 700 else ("usable" if p50 < 1200 else "too slow")
        print(f"    {rank:>2}. {model:<32} {p50:7.0f} ms  {verdict}")

    fastest = ranked[0]
    print(f"\n  Fastest: {fastest[0]} at {statistics.median(fastest[1]['xs']):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
