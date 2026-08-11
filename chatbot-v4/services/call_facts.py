"""Sticky per-call facts (caller name, company, job) that survive history truncation."""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

FACT_KEYS = (
    "caller_name",
    "company",
    "job_description",
    "employment_type",
    "rate_mentioned",
)

_LABELS = {
    "caller_name": "Caller name",
    "company": "Company",
    "job_description": "Job",
    "employment_type": "Employment type",
    "rate_mentioned": "Rate mentioned",
}

# Agent identity — never treat as caller.
_AGENT_NAME_BLOCKLIST = {
    "bert",
    "bertrand",
    "bertrand kalisa",
    "kalisa",
}

_JOB_MAX = 160

_NAME_USER_RE = re.compile(
    r"(?i)\b(?:my name is|this is|i'?m|i am)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"
)
_NAME_ASST_RE = re.compile(
    r"(?i)\b(?:thanks|thank you|nice to meet you|hi|hey|hello),?\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"
)
_CORRECTION_RE = re.compile(
    r"(?i)\b(?:actually(?: it'?s)?|it'?s|name is)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
    r"(?:\s*,?\s*not\b)?"
)
_COMPANY_STOP = {
    "about",
    "regarding",
    "for",
    "on",
    "with",
    "and",
    "who",
    "that",
    "this",
    "looking",
    "calling",
    "here",
}
_COMPANY_RE = re.compile(
    r"(?i)\b(?:calling from|i'?m with|we'?re with|from|with)\s+"
    r"([A-Z][A-Za-z0-9&.'-]{1,40}(?:\s+(?:Staffing|Recruiting|Recruiters?|Solutions|"
    r"Inc\.?|LLC|Corp\.?|[A-Z][A-Za-z0-9&.']{1,40})){0,3})"
)
_EMPLOYMENT_RE = re.compile(r"(?i)\b(W-?2|C2C|1099)\b")
_RATE_RE = re.compile(r"\$\s*(\d{2,3}(?:\.\d{1,2})?)\s*(?:/\s*h(?:ou)?r|an hour|per hour|hr)?")
_JOB_HINT_RE = re.compile(
    r"(?i)\b((?:agentic\s+)?(?:ai|ml)\s+(?:engineer|developer|role|position)"
    r"|rag\s+pipeline[^.]{0,40}"
    r"|(?:senior\s+)?(?:software|data|machine learning|mlops|llmops)\s+"
    r"(?:engineer|developer|architect|role|position)"
    r"|contract(?:\s+role|\s+position)?\s+(?:for|as)\s+[^.]{3,60})\b"
)


def empty_facts() -> dict[str, str]:
    return {k: "" for k in FACT_KEYS}


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", (raw or "").strip(" .,!?;:"))
    if not name or len(name) > 48:
        return ""
    if name.lower() in _AGENT_NAME_BLOCKLIST:
        return ""
    # Reject common false positives
    if name.lower() in {"here", "calling", "regarding", "about", "just", "looking"}:
        return ""
    return name


def _clean_company(raw: str) -> str:
    company = re.sub(r"\s+", " ", (raw or "").strip(" .,!?;:"))
    if not company or len(company) > 64:
        return ""
    tokens = company.split()
    kept: list[str] = []
    for tok in tokens:
        if tok.lower() in _COMPANY_STOP:
            break
        kept.append(tok)
    company = " ".join(kept).strip(" .,!?;:")
    if not company:
        return ""
    low = company.lower()
    if low in _AGENT_NAME_BLOCKLIST or low in {
        "here",
        "there",
        "you",
        "us",
        "them",
        "the",
        "this",
        "that",
        "our",
        "my",
    }:
        return ""
    return company


def _clean_job(raw: str) -> str:
    job = re.sub(r"\s+", " ", (raw or "").strip())
    if not job:
        return ""
    if len(job) > _JOB_MAX:
        job = job[: _JOB_MAX - 1].rsplit(" ", 1)[0] + "…"
    return job


def merge_facts(
    dest: dict[str, str],
    incoming: dict[str, str] | None,
    *,
    replace: bool = False,
) -> list[str]:
    """Fill-if-empty merge (or replace). Returns list of updated keys."""
    if not incoming:
        return []
    updated: list[str] = []
    for key in FACT_KEYS:
        if key not in incoming:
            continue
        val = (incoming.get(key) or "").strip()
        if not val:
            continue
        if key == "caller_name":
            val = _clean_name(val) or val
        elif key == "company":
            val = _clean_company(val) or val
        elif key == "job_description":
            val = _clean_job(val)
        if not val:
            continue
        cur = (dest.get(key) or "").strip()
        if replace or not cur:
            if cur != val:
                dest[key] = val
                updated.append(key)
        elif key == "caller_name" and val.lower() != cur.lower():
            # Explicit corrections handled by caller via replace=True
            pass
    return updated


def call_facts_block(facts: dict[str, str]) -> str:
    known_lines: list[str] = []
    missing: list[str] = []
    for key in ("caller_name", "company", "job_description"):
        label = _LABELS[key]
        val = (facts.get(key) or "").strip()
        if val:
            known_lines.append(f"- {label}: {val}")
        else:
            missing.append(label)
    for key in ("employment_type", "rate_mentioned"):
        val = (facts.get(key) or "").strip()
        if val:
            known_lines.append(f"- {_LABELS[key]}: {val}")

    parts: list[str] = []
    if known_lines:
        parts.append(
            "# Established this call\n"
            + "\n".join(known_lines)
            + "\nUse these. Do not re-ask for known items. Do not invent missing ones."
        )
    if missing:
        parts.append(
            "# Still need (ask naturally once, then pin)\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\nIf Still need is non-empty, ask for those naturally early; "
            "once filled, reuse and never re-ask."
        )
    elif known_lines:
        parts.append("All key call facts are known — do not re-verify them.")
    return "\n\n".join(parts)


def extract_from_user(text: str, facts: dict[str, str]) -> list[str]:
    """Update facts from a caller utterance. Returns updated keys."""
    incoming: dict[str, str] = {}
    t = text or ""

    # Corrections can overwrite caller_name
    corr = _CORRECTION_RE.search(t)
    if corr and re.search(r"(?i)\bactually\b|\bnot\b", t):
        name = _clean_name(corr.group(1))
        if name:
            merge_facts(facts, {"caller_name": name}, replace=True)

    if not (facts.get("caller_name") or "").strip():
        m = _NAME_USER_RE.search(t)
        if m:
            name = _clean_name(m.group(1))
            if name:
                incoming["caller_name"] = name

    if not (facts.get("company") or "").strip():
        m = _COMPANY_RE.search(t)
        if m:
            company = _clean_company(m.group(1))
            if company:
                incoming["company"] = company

    if not (facts.get("employment_type") or "").strip():
        m = _EMPLOYMENT_RE.search(t)
        if m:
            et = m.group(1).upper().replace("W2", "W2").replace("W-2", "W2")
            if et in {"W2", "W-2"}:
                et = "W2"
            incoming["employment_type"] = et

    if not (facts.get("rate_mentioned") or "").strip():
        m = _RATE_RE.search(t)
        if m:
            incoming["rate_mentioned"] = f"${m.group(1)}/hr"

    if not (facts.get("job_description") or "").strip():
        m = _JOB_HINT_RE.search(t)
        if m:
            job = _clean_job(m.group(1))
            if job:
                incoming["job_description"] = job

    return merge_facts(facts, incoming)


def extract_from_assistant(text: str, facts: dict[str, str]) -> list[str]:
    """Update facts from an assistant reply (e.g. addressed caller by name)."""
    incoming: dict[str, str] = {}
    t = text or ""
    if not (facts.get("caller_name") or "").strip():
        m = _NAME_ASST_RE.search(t)
        if m:
            name = _clean_name(m.group(1))
            if name:
                incoming["caller_name"] = name
    return merge_facts(facts, incoming)


def seed_from_context_text(context: str) -> dict[str, str]:
    """Pull company/job from Operator context section headers."""
    facts = empty_facts()
    text = context or ""
    if not text.strip():
        return facts

    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[2:].strip().lower()
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()

    for key in ("company",):
        if sections.get(key):
            facts["company"] = _clean_company(sections[key].splitlines()[0]) or ""

    for key in ("job", "opportunity"):
        body = sections.get(key) or ""
        if body:
            # First non-empty line / short summary
            first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
            facts["job_description"] = _clean_job(first or body)
            break

    # Established block from a prior write
    est = sections.get("established this call") or ""
    for line in est.splitlines():
        m = re.match(r"-\s*(Caller name|Company|Job):\s*(.+)\s*$", line.strip())
        if not m:
            continue
        label, val = m.group(1), m.group(2).strip()
        if label == "Caller name" and not facts["caller_name"]:
            facts["caller_name"] = _clean_name(val)
        elif label == "Company" and not facts["company"]:
            facts["company"] = _clean_company(val)
        elif label == "Job" and not facts["job_description"]:
            facts["job_description"] = _clean_job(val)

    return {k: v for k, v in facts.items() if v}


def extract_established_section(context: str) -> str:
    """Return trailing '# Established this call' section if present."""
    text = (context or "").strip()
    marker = "# Established this call"
    idx = text.find(marker)
    if idx < 0:
        return ""
    return text[idx:].strip()


def strip_established_section(context: str) -> str:
    text = (context or "").rstrip()
    marker = "\n# Established this call"
    idx = text.find(marker)
    if idx >= 0:
        return text[:idx].rstrip()
    if text.startswith("# Established this call"):
        return ""
    return text


def format_established_for_context(facts: dict[str, str]) -> str:
    """Persist known facts into context markdown (survives coach re-upload)."""
    lines: list[str] = []
    for key in ("caller_name", "company", "job_description"):
        val = (facts.get(key) or "").strip()
        if val:
            lines.append(f"- {_LABELS[key]}: {val}")
    if not lines:
        return ""
    return "# Established this call\n" + "\n".join(lines)


def log_updates(session_id: str, updated: Iterable[str], facts: dict[str, str]) -> None:
    for key in updated:
        logger.info(
            "[%s] call_facts.updated key=%s value=%s",
            session_id,
            key,
            (facts.get(key) or "")[:80],
        )
