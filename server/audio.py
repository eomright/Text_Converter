"""ffmpeg 전처리: 어떤 포맷이 들어와도 Whisper가 원하는 16kHz 모노 WAV로 만든다."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import config

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# 필터 체인은 녹음 환경별로 config.PRESETS 에 있다 (DESIGN.md §5.3, §5.6)
#   highpass   : 에어컨/책상 진동 등 저주파 제거
#   loudnorm   : EBU R128 음량 정규화 — 파일 전체 음량을 맞춘다
#   dynaudnorm : 구간별 동적 증폭 — 작게 말한 구간만 따로 끌어올린다 (강의실)


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(path: Path) -> float:
    """오디오 길이(초). 실패하면 0.0."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def to_wav(src: Path, dst: Path, *, preset: str = config.DEFAULT_PRESET,
           denoise: bool = True) -> Path:
    """src(webm/mp3/m4a/wav/…) → dst(16kHz mono pcm_s16le wav)."""
    p = config.PRESETS.get(preset, config.PRESETS[config.DEFAULT_PRESET])
    chain = p["chain"] if denoise else "loudnorm=I=-16:TP=-1.5:LRA=11"
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src.resolve()),
        "-af", chain,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst.resolve()),
    ]
    # arnndn 은 모델 파일을 상대경로로 받는다 — 'C:' 콜론이 필터 문법과 충돌하기 때문
    cwd = str(config.RNNOISE_DIR) if "arnndn" in chain and config.RNNOISE_DIR.exists() else None
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0 or not dst.exists():
        raise AudioError(
            f"ffmpeg 전처리 실패 (code {proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    return dst
