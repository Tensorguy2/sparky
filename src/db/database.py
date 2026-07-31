"""
SQLite persistence layer for voice profiles.

Only metadata is stored here (voice_id, name, paths, ref_text).
The actual VoiceClonePromptItem tensors live in RAM inside VoiceService.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_DB_PATH: str = ""


# ── Init ───────────────────────────────────────────────────────────────────────

def init(db_path: str) -> None:
    global _DB_PATH
    _DB_PATH = db_path
    logger.debug("Initialising SQLite database at %s", db_path)
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                voice_id       TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                ref_audio_path TEXT NOT NULL,
                ref_text       TEXT NOT NULL,
                builtin        INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    logger.info("Database ready: %s", db_path)


# ── Internal connection helper ─────────────────────────────────────────────────

@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class VoiceRecord:
    voice_id: str
    name: str
    ref_audio_path: str
    ref_text: str
    builtin: bool
    created_at: str


def _row_to_record(row: sqlite3.Row) -> VoiceRecord:
    return VoiceRecord(
        voice_id=row["voice_id"],
        name=row["name"],
        ref_audio_path=row["ref_audio_path"],
        ref_text=row["ref_text"],
        builtin=bool(row["builtin"]),
        created_at=row["created_at"],
    )


# ── CRUD ───────────────────────────────────────────────────────────────────────

def get_all() -> List[VoiceRecord]:
    t0 = time.perf_counter()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM voices ORDER BY builtin DESC, created_at ASC"
        ).fetchall()
    records = [_row_to_record(r) for r in rows]
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug("get_all() returned %d record(s) in %.1f ms", len(records), elapsed)
    return records


def get(voice_id: str) -> Optional[VoiceRecord]:
    t0 = time.perf_counter()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM voices WHERE voice_id = ?", (voice_id,)
        ).fetchone()
    elapsed = (time.perf_counter() - t0) * 1000
    if row:
        logger.debug("get(%r) → hit (%.1f ms)", voice_id, elapsed)
        return _row_to_record(row)
    logger.debug("get(%r) → miss (%.1f ms)", voice_id, elapsed)
    return None


def exists(voice_id: str) -> bool:
    t0 = time.perf_counter()
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM voices WHERE voice_id = ?", (voice_id,)
        ).fetchone()
    elapsed = (time.perf_counter() - t0) * 1000
    result = row is not None
    logger.debug("exists(%r) → %s (%.1f ms)", voice_id, result, elapsed)
    return result


def insert(
    voice_id: str,
    name: str,
    ref_audio_path: str,
    ref_text: str,
    builtin: bool = False,
) -> None:
    t0 = time.perf_counter()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO voices (voice_id, name, ref_audio_path, ref_text, builtin)
               VALUES (?, ?, ?, ?, ?)""",
            (voice_id, name, ref_audio_path, ref_text, int(builtin)),
        )
        conn.commit()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(
        "insert(%r, name=%r, builtin=%s) done in %.1f ms",
        voice_id, name, builtin, elapsed,
    )


def delete(voice_id: str) -> None:
    t0 = time.perf_counter()
    with _conn() as conn:
        conn.execute("DELETE FROM voices WHERE voice_id = ?", (voice_id,))
        conn.commit()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug("delete(%r) done in %.1f ms", voice_id, elapsed)
