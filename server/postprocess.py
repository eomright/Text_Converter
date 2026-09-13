"""후처리: 반복 환각 제거 + 사용자 사전 치환 + 문단 정리 (DESIGN.md §5.4)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import GLOSSARY_PATH

# VAD를 켜도 가끔 새어나오는 유튜브 자막 오염 문구들.
# 세그먼트 전체가 이것뿐이면 통째로 버린다.
HALLUCINATION_PHRASES = [
    "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다",
    "구독과 좋아요", "구독 좋아요 부탁드립니다",
    "다음 영상에서 만나요", "한글자막 by", "자막 제공",
    "MBC 뉴스", "이덕영입니다",
    "Thanks for watching", "Please subscribe",
]

DEFAULT_GLOSSARY = {
    # initial_prompt 으로 모델에 주입되는 문맥 힌트.
    # 고유명사/전문용어를 여기 적어두면 인식률이 눈에 띄게 오른다.
    "prompt": "",
    # 후처리 문자열 치환. {"잘못 인식된 표기": "올바른 표기"}
    "replacements": {},
}


# ── 용어집 ────────────────────────────────────────────────────────────
def load_glossary() -> dict:
    if GLOSSARY_PATH.exists():
        try:
            data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_GLOSSARY, **data}
        except Exception:
            pass
    return dict(DEFAULT_GLOSSARY)


def save_glossary(data: dict) -> dict:
    merged = {
        "prompt": str(data.get("prompt", "") or ""),
        "replacements": dict(data.get("replacements", {}) or {}),
    }
    GLOSSARY_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged


# ── 개별 세그먼트 정리 ────────────────────────────────────────────────
def _collapse_inner_repeat(text: str) -> str:
    """한 세그먼트 안에서 같은 어절이 3회 이상 이어지면 1회로 줄인다.

    '네 네 네 네 네' → '네'
    """
    # 어절 단위
    text = re.sub(r"\b(\S{1,12}?)(?:\s+\1){2,}\b", r"\1", text)
    # 공백 없이 붙은 반복 ('감사합니다감사합니다감사합니다')
    text = re.sub(r"(.{2,15}?)\1{2,}", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    for phrase in HALLUCINATION_PHRASES:
        if phrase in t and len(t) <= len(phrase) + 8:
            return True
    return False


def clean_segment(text: str, replacements: dict[str, str] | None = None) -> str:
    text = _collapse_inner_repeat(text)
    for wrong, right in (replacements or {}).items():
        if wrong:
            text = text.replace(wrong, right)
    return text


# ── 전체 트랜스크립트 정리 ────────────────────────────────────────────
def clean_transcript(segments: list[dict], replacements: dict | None = None) -> list[dict]:
    """세그먼트 리스트를 받아 환각/중복을 걷어낸 새 리스트를 돌려준다."""
    out: list[dict] = []
    repeat_count = 0
    prev = None

    for seg in segments:
        text = clean_segment(seg.get("text", ""), replacements)
        if _is_hallucination(text):
            continue
        # 동일 문장이 연속 3회 이상이면 그 이후는 버린다
        if text == prev:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        prev = text
        out.append({**seg, "text": text})
    return out


def to_paragraphs(segments: list[dict], gap_sec: float = 1.5) -> str:
    """무음 gap_sec 이상을 경계로 문단을 나눈 평문."""
    paras: list[list[str]] = []
    cur: list[str] = []
    last_end = None
    for seg in segments:
        if last_end is not None and seg["start"] - last_end >= gap_sec and cur:
            paras.append(cur)
            cur = []
        cur.append(seg["text"])
        last_end = seg["end"]
    if cur:
        paras.append(cur)
    return "\n\n".join(" ".join(p).strip() for p in paras).strip()


# ── 내보내기 ──────────────────────────────────────────────────────────
def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(
            f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(lines)


def to_markdown(segments: list[dict], title: str = "받아쓰기") -> str:
    body = "\n".join(
        f"- `[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}]` {s['text']}"
        for s in segments
    )
    return f"# {title}\n\n## 전문\n\n{to_paragraphs(segments)}\n\n## 타임스탬프\n\n{body}\n"
