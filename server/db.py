"""SQLite 기록 저장. 호출마다 커넥션을 새로 열어 스레드 안전성을 확보한다."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL,          -- queued | running | done | error
    model         TEXT,
    duration_sec  REAL DEFAULT 0,         -- 오디오 길이
    elapsed_sec   REAL DEFAULT 0,         -- 변환에 걸린 시간
    audio_path    TEXT,
    segments_json TEXT DEFAULT '[]',
    text          TEXT DEFAULT '',
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON transcripts(created_at DESC);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)


def create_job(job_id: str, title: str, audio_path: str, model: str,
               duration_sec: float) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO transcripts (id,title,created_at,status,model,duration_sec,audio_path)"
            " VALUES (?,?,?,?,?,?,?)",
            (job_id, title, datetime.now().isoformat(timespec="seconds"),
             "queued", model, duration_sec, audio_path),
        )


def set_status(job_id: str, status: str, error: str | None = None) -> None:
    with conn() as c:
        c.execute("UPDATE transcripts SET status=?, error=? WHERE id=?",
                  (status, error, job_id))


def finish(job_id: str, segments: list[dict], text: str, elapsed_sec: float) -> None:
    with conn() as c:
        c.execute(
            "UPDATE transcripts SET status='done', segments_json=?, text=?, elapsed_sec=?"
            " WHERE id=?",
            (json.dumps(segments, ensure_ascii=False), text, elapsed_sec, job_id),
        )


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["segments"] = json.loads(d.pop("segments_json") or "[]")
    return d


def get(job_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM transcripts WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(r) if r else None


def list_recent(limit: int = 50) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT id,title,created_at,status,model,duration_sec,elapsed_sec,"
            "substr(text,1,120) AS preview"
            " FROM transcripts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_segments(job_id: str, segments: list[dict], text: str) -> None:
    """사용자가 고친 구간과, 그걸로 다시 만든 전문을 함께 저장 (내보내기가 둘 다 쓰므로)."""
    with conn() as c:
        c.execute(
            "UPDATE transcripts SET segments_json=?, text=? WHERE id=?",
            (json.dumps(segments, ensure_ascii=False), text, job_id),
        )


def delete(job_id: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM transcripts WHERE id=?", (job_id,))
