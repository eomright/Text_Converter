"""SQLite 기록 저장. 호출마다 커넥션을 새로 열어 스레드 안전성을 확보한다.

상태 흐름
  recording   실시간 녹음 중 (초안이 drafts_json 에 계속 쌓임)
  queued      정밀 변환 대기
  running     정밀 변환 중 (결과가 segments_json 에 조금씩 쌓이고 progress_sec 가 늘어남)
  done        완료
  interrupted 서버가 꺼지거나 연결이 끊겨 멈춤 → "이어서 변환" 가능
  error       오류
"""
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
    status        TEXT NOT NULL,
    model         TEXT,
    duration_sec  REAL DEFAULT 0,         -- 오디오 길이
    elapsed_sec   REAL DEFAULT 0,         -- 변환에 걸린 시간 (이어서 변환하면 누적)
    audio_path    TEXT,
    segments_json TEXT DEFAULT '[]',
    text          TEXT DEFAULT '',
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON transcripts(created_at DESC);
"""

# 나중에 추가된 컬럼. 옛 DB 에도 ALTER TABLE 로 붙인다 (기존 기록은 그대로 유지)
MIGRATIONS = {
    "preset": "TEXT",                     # 녹음 환경 — 이어서 변환할 때 같은 설정을 쓰려고
    "progress_sec": "REAL DEFAULT 0",     # 정밀 변환이 여기(초)까지 저장됨
    "drafts_json": "TEXT DEFAULT '[]'",   # 실시간 모드 초안
}

ACTIVE = ("recording", "queued", "running")


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
        have = {r["name"] for r in c.execute("PRAGMA table_info(transcripts)")}
        for col, decl in MIGRATIONS.items():
            if col not in have:
                c.execute(f"ALTER TABLE transcripts ADD COLUMN {col} {decl}")


def create_job(job_id: str, title: str, audio_path: str, model: str,
               duration_sec: float, preset: str, status: str = "queued") -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO transcripts (id,title,created_at,status,model,duration_sec,audio_path,preset)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (job_id, title, datetime.now().isoformat(timespec="seconds"),
             status, model, duration_sec, audio_path, preset),
        )


def set_status(job_id: str, status: str, error: str | None = None) -> None:
    with conn() as c:
        c.execute("UPDATE transcripts SET status=?, error=? WHERE id=?",
                  (status, error, job_id))


def set_recorded(job_id: str, *, title: str | None, duration_sec: float,
                 status: str, error: str | None = None) -> None:
    """실시간 녹음이 끝났을 때 (정상 종료든 끊김이든) 길이와 상태를 확정."""
    with conn() as c:
        if title:
            c.execute("UPDATE transcripts SET title=? WHERE id=?", (title, job_id))
        c.execute("UPDATE transcripts SET duration_sec=?, status=?, error=? WHERE id=?",
                  (duration_sec, status, error, job_id))


def set_drafts(job_id: str, drafts: list[dict]) -> None:
    with conn() as c:
        c.execute("UPDATE transcripts SET drafts_json=? WHERE id=?",
                  (json.dumps(drafts, ensure_ascii=False), job_id))


def save_partial(job_id: str, segments: list[dict], progress_sec: float) -> None:
    """정밀 변환 도중 중간 저장. 서버가 꺼져도 여기까지는 남는다."""
    with conn() as c:
        c.execute(
            "UPDATE transcripts SET segments_json=?, progress_sec=? WHERE id=?",
            (json.dumps(segments, ensure_ascii=False), progress_sec, job_id),
        )


def finish(job_id: str, segments: list[dict], text: str, elapsed_sec: float) -> None:
    with conn() as c:
        c.execute(
            "UPDATE transcripts SET status='done', error=NULL, segments_json=?, text=?,"
            " elapsed_sec=?, progress_sec=duration_sec WHERE id=?",
            (json.dumps(segments, ensure_ascii=False), text, elapsed_sec, job_id),
        )


def mark_interrupted() -> list[dict]:
    """서버가 시작될 때 호출. 지난번에 하던 작업은 전부 끊긴 것이므로 '중단됨'으로 바꾼다."""
    with conn() as c:
        rows = c.execute(
            f"SELECT * FROM transcripts WHERE status IN ({','.join('?' * len(ACTIVE))})",
            ACTIVE).fetchall()
        c.execute(
            f"UPDATE transcripts SET status='interrupted',"
            f" error='서버가 꺼져서 멈췄습니다. 저장된 곳까지는 남아 있고, 이어서 변환할 수 있습니다.'"
            f" WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE)
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["segments"] = json.loads(d.pop("segments_json") or "[]")
    d["drafts"] = json.loads(d.pop("drafts_json", None) or "[]")
    d["progress_sec"] = d.get("progress_sec") or 0.0
    return d


def get(job_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM transcripts WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(r) if r else None


def list_recent(limit: int = 50) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT id,title,created_at,status,model,duration_sec,elapsed_sec,progress_sec,"
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
