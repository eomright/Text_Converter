"""교정 모드: 사용자가 고친 구간 검증, 모델 비교 결과 정렬, 평가용 파일 이름 정리."""
from __future__ import annotations

import bisect
import re

MAX_SEGMENTS = 20000
MAX_TEXT_LEN = 2000

_PUNCT = re.compile(r"[\s.,!?~\"'`\-—…·:;()\[\]{}「」『』]+")


# ── 사용자가 보낸 구간 목록 검증 ───────────────────────────────────────
def sanitize_segments(raw) -> list[dict]:
    """화면에서 온 구간 목록을 믿지 않고 허용된 필드만 타입을 맞춰 남긴다."""
    if not isinstance(raw, list) or len(raw) > MAX_SEGMENTS:
        raise ValueError("segments 는 목록이어야 합니다.")
    out: list[dict] = []
    for s in raw:
        if not isinstance(s, dict):
            raise ValueError("구간 형식이 올바르지 않습니다.")
        try:
            start = round(float(s["start"]), 2)
            end = round(float(s["end"]), 2)
        except (KeyError, TypeError, ValueError):
            raise ValueError("구간에 start/end 가 필요합니다.")
        text = str(s.get("text", ""))
        text = re.sub(r"\s*\n\s*", " ", text).strip()[:MAX_TEXT_LEN]
        seg = {"start": start, "end": end, "text": text,
               "checked": bool(s.get("checked", False))}
        for key in ("avg_logprob", "no_speech_prob"):
            if isinstance(s.get(key), (int, float)):
                seg[key] = float(s[key])
        alts = s.get("alts")
        if isinstance(alts, dict):
            seg["alts"] = {str(k)[:40]: str(v)[:MAX_TEXT_LEN] for k, v in alts.items()}
        if s.get("flag") in ("disagree",):
            seg["flag"] = s["flag"]
        out.append(seg)
    return out


# ── 모델 비교 ─────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """비교용: 띄어쓰기·문장부호·대소문자 차이는 같은 것으로 본다."""
    return _PUNCT.sub("", text or "").lower()


def align_words(segments: list[dict], words: list[dict]) -> list[str]:
    """다른 모델의 단어들을 원래 구간에 나눠 담는다.

    모델마다 구간을 자르는 위치가 달라서 문장 단위로는 비교할 수 없다.
    단어별 타임스탬프의 가운데 시점이 들어가는 구간에 넣고,
    구간 사이 틈에 떨어진 단어는 가장 가까운 구간에 붙인다.
    """
    buckets: list[list[str]] = [[] for _ in segments]
    if not segments:
        return []
    starts = [s["start"] for s in segments]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        i = bisect.bisect_right(starts, mid) - 1
        if i < 0:
            i = 0
        elif mid >= segments[i]["end"] and i + 1 < len(segments):
            # 틈에 떨어짐 → 앞 구간 끝과 다음 구간 시작 중 가까운 쪽
            if segments[i + 1]["start"] - mid < mid - segments[i]["end"]:
                i += 1
        buckets[i].append(w["word"])
    return ["".join(b).strip() for b in buckets]


def apply_comparison(segments: list[dict], alts_by_model: dict[str, list[str]],
                     reference_only: set[str]) -> int:
    """구간마다 다른 모델의 결과를 붙이고, 큰 모델끼리 다르면 ⚠ 표시. ⚠ 개수를 돌려준다."""
    flagged = 0
    for i, seg in enumerate(segments):
        alts = {m: texts[i] for m, texts in alts_by_model.items() if i < len(texts)}
        seg["alts"] = alts
        base = normalize(seg.get("text", ""))
        judges = [m for m in alts if m not in reference_only] or list(alts)
        if any(normalize(alts[m]) != base for m in judges):
            seg["flag"] = "disagree"
            flagged += 1
        else:
            seg.pop("flag", None)
    return flagged


# ── 평가용 파일 이름 ──────────────────────────────────────────────────
def safe_eval_name(name: str, fallback: str = "강의") -> str:
    """경로 조작을 막기 위해 한글·영문·숫자·공백·-_ 만 남긴다. 점도 뺀다 (확장자 혼동 방지)."""
    cleaned = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ _\-]", "", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()[:60]
    return cleaned or fallback
