#!/usr/bin/env python3
"""
Live per-turn latency for a voice call, reconstructed from the two server logs.

Uses explicit reply boundaries (`Assistant (model): N chars, M TTS sentence(s)`)
as the turn delimiter rather than idle-gap timing, which previously produced
impossible negative values when turns overlapped or logs arrived out of order.

  stt      transcription (local Parakeet, on GPU)
  router   the tool-call to OpenAI that classifies the turn
  llm      reply model time-to-first-token
  tts      first-audio latency reported by the TTS server
  ---------
  reply    recording stopped -> caller hears the first word

VAD endpoint delay happens in the browser before audio is even sent here, so
perceived latency is a further ~200 ms on top. window.__vadSummary() has it.

  python scripts/monitor_turns.py
  python scripts/monitor_turns.py --chatbot /tmp/chatbot_v3.log --tts /tmp/tts_v3.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

RE_STOP = re.compile(TS + r".*Recording stopped \| ([\d.]+) s")
RE_STT = re.compile(TS + r".*Parakeet STT \| audio=([\d.]+)s stt_ms=(\d+) text='(.*)'")
RE_ROUTER = re.compile(TS + r".*POST https://api\.openai\.com/v1/chat/completions")
RE_LLM_START = re.compile(TS + r".*(?:Anthropic|OpenAI|Local) stream \| model=(\S+)")
RE_REPLY = re.compile(TS + r".*Assistant \(([^)]+)\): (\d+) chars, (\d+) TTS sentence")

RE_DONE = re.compile(
    TS + r".*\[(\w+)\] v3 Done \| wall=([\d.]+) s audio=([\d.]+) s "
    r"RTF=([\d.]+) TTFA=([\d.]+) s"
)
RE_FRAG = re.compile(TS + r".*\[(\w+)\] voice='([^']+)' lang=\S+ \d+ sentence")
RE_GUARD = re.compile(TS + r".*\[[\d/]+\] (truncated\S*|runaway\S*) on attempt (\d+)")
RE_EXHAUST = re.compile(TS + r".*retries exhausted \((\S+)\)")
RE_LOCK_SET = re.compile(TS + r".*speaker locked(?: from cache)? for turn=(\S+)")
RE_LOCK_USED = re.compile(TS + r".*speaker lock applied \(turn=(\S+)\)")

# After the reply log line, allow this much wall time for TTS fragments
# to arrive before declaring the turn closed.
TTS_COLLECT_S = 30.0


def parse_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def follow(paths: list[Path]):
    """Yield appended lines, or None when both logs go quiet."""
    handles = {}
    for p in paths:
        f = p.open("r", errors="replace")
        f.seek(0, 2)
        handles[p] = f
    while True:
        idle = True
        for f in handles.values():
            line = f.readline()
            if line:
                idle = False
                yield line.rstrip("\n")
        if idle:
            time.sleep(0.15)
            yield None


class Turn:
    def __init__(self, stop_at: float, heard_s: float):
        self.stop_at = stop_at
        self.heard_s = heard_s
        self.text = ""
        self.stt_ms: float | None = None
        self.router_at: float | None = None
        self.llm_at: float | None = None
        self.model = ""
        self.reply_at: float | None = None
        self.sentences = 0
        self.first_audio_at: float | None = None
        self.tts_ttfa_ms: float | None = None
        self.voices: list[str] = []
        self.fragments = 0
        self.spoken_s = 0.0
        self.retries = 0
        self.exhausted = 0
        self.lock_set = 0
        self.lock_used = 0
        self.last_event_at = stop_at

    def add_fragment(self, done_at: float, wall: float, audio: float, ttfa: float) -> None:
        started = done_at - wall
        if self.first_audio_at is None:
            self.first_audio_at = started + ttfa
            self.tts_ttfa_ms = ttfa * 1000
        self.fragments += 1
        self.spoken_s += audio
        self.last_event_at = max(self.last_event_at, done_at)

    @property
    def reply_ms(self) -> float | None:
        if self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.stop_at) * 1000

    @property
    def collecting_tts(self) -> bool:
        """True while we still expect TTS fragments for this turn."""
        if self.reply_at is None:
            return False
        return time.time() - self.reply_at < TTS_COLLECT_S

    def render(self) -> str:
        out = [f"TURN  heard {self.heard_s:.1f}s: {self.text[:56]!r}"]
        if self.reply_ms is None:
            out.append("        (no audio yet)")
            return "\n".join(out)

        reply = self.reply_ms
        if reply < 0:
            out.append(f"        reply {reply:.0f} ms (clock skew between logs)")
            return "\n".join(out)

        stt = self.stt_ms or 0.0
        router = 0.0
        if self.router_at and self.llm_at and self.llm_at >= self.router_at:
            router = (self.router_at - (self.stop_at + stt / 1000)) * 1000
        llm = 0.0
        if self.llm_at and self.reply_at:
            llm = (self.reply_at - self.llm_at) * 1000
        elif self.llm_at and self.first_audio_at:
            llm = (self.first_audio_at - self.llm_at) * 1000 - (self.tts_ttfa_ms or 0)
        tts = self.tts_ttfa_ms or 0.0

        parts = [f"stt {stt:4.0f}"]
        if router > 0:
            parts.append(f"router {router:5.0f}")
        parts.append(f"llm {llm:5.0f}")
        parts.append(f"tts {tts:4.0f}")
        line = "        " + " | ".join(parts) + f"  ->  reply {reply:5.0f} ms"
        out.append(line)

        detail = (f"        {self.sentences} sentence(s) -> {self.fragments} TTS "
                  f"generation(s), {self.spoken_s:.1f}s spoken")
        distinct = sorted(set(self.voices))
        if len(distinct) > 1:
            detail += f"  !! MIXED VOICES: {distinct}"
        elif distinct:
            detail += f", voice={distinct[0]}"
        if self.lock_set or self.lock_used:
            unlocked = max(0, self.fragments - self.lock_set - self.lock_used)
            detail += f", speaker locked ({self.lock_used} inherited"
            detail += f", {unlocked} UNLOCKED)" if unlocked else ")"
        elif self.fragments > 1:
            detail += f"  !! NO SPEAKER LOCK across {self.fragments} generations"
        if self.retries:
            detail += f"  guard retried x{self.retries}"
        if self.exhausted:
            detail += f"  BEST-TAKE FALLBACK x{self.exhausted}"
        out.append(detail)
        return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chatbot", default="/tmp/chatbot_v3.log")
    ap.add_argument("--tts", default="/tmp/tts_v3.log")
    args = ap.parse_args()

    paths = [Path(args.chatbot), Path(args.tts)]
    for p in paths:
        if not p.exists():
            print(f"missing log: {p}", file=sys.stderr)
            sys.exit(1)

    print("  Monitoring turns. Waiting for the call...\n", flush=True)

    current: Turn | None = None
    done: list[Turn] = []

    def close_turn() -> None:
        nonlocal current
        if current is None:
            return
        print(current.render(), flush=True)
        if current.reply_ms is not None and current.reply_ms >= 0:
            done.append(current)
            if len(done) > 1:
                xs = [t.reply_ms for t in done]
                print(f"        running p50 reply {statistics.median(xs):.0f} ms "
                      f"over {len(done)} turns", flush=True)
        print(flush=True)
        current = None

    try:
        for line in follow(paths):
            if line is None:
                # Close turn once TTS collection window expires
                if (current is not None
                        and current.reply_at is not None
                        and not current.collecting_tts):
                    close_turn()
                # Also close turns that never got a reply after a long pause
                elif (current is not None
                      and current.reply_at is None
                      and time.time() - current.last_event_at > TTS_COLLECT_S):
                    close_turn()
                continue

            # -- Chatbot log events --

            m = RE_STOP.search(line)
            if m:
                close_turn()
                ts, heard = m.groups()
                current = Turn(parse_ts(ts), float(heard))
                continue

            if current is None:
                continue

            m = RE_STT.search(line)
            if m:
                _ts, _audio, stt_ms, text = m.groups()
                current.stt_ms = float(stt_ms)
                current.text = text
                current.last_event_at = parse_ts(m.group(1))
                continue

            m = RE_ROUTER.search(line)
            if m and current.llm_at is None:
                current.router_at = parse_ts(m.group(1))
                current.last_event_at = current.router_at
                continue

            m = RE_LLM_START.search(line)
            if m:
                if current.llm_at is None:
                    current.llm_at = parse_ts(m.group(1))
                    current.model = m.group(2)
                    current.last_event_at = current.llm_at
                continue

            m = RE_REPLY.search(line)
            if m:
                ts_str, model, _chars, n_sentences = m.groups()
                current.model = model
                current.sentences = int(n_sentences)
                current.reply_at = parse_ts(ts_str)
                current.last_event_at = current.reply_at
                continue

            # -- TTS log events --

            m = RE_FRAG.search(line)
            if m:
                current.voices.append(m.group(3))
                current.last_event_at = parse_ts(m.group(1))
                continue

            m = RE_LOCK_SET.search(line)
            if m:
                current.lock_set += 1
                continue

            m = RE_LOCK_USED.search(line)
            if m:
                current.lock_used += 1
                continue

            m = RE_GUARD.search(line)
            if m:
                current.retries += 1
                continue

            m = RE_EXHAUST.search(line)
            if m:
                current.exhausted += 1
                continue

            m = RE_DONE.search(line)
            if m:
                ts_str, _cid, wall, audio, _rtf, ttfa = m.groups()
                current.add_fragment(
                    parse_ts(ts_str), float(wall), float(audio), float(ttfa)
                )
                continue

    except KeyboardInterrupt:
        close_turn()
        if done:
            xs = [t.reply_ms for t in done]
            print(f"\n  {len(done)} turns | reply p50 {statistics.median(xs):.0f} ms | "
                  f"min {min(xs):.0f} | max {max(xs):.0f}")


if __name__ == "__main__":
    main()
