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
#   (dynaudnorm·speechnorm·RNNoise 는 강의실 녹음에서 오히려 CER 이 나빠져 쓰지 않는다)


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


def to_listening_wav(src: Path, dst: Path) -> Path:
    """교정 모드 재생용. 필터 없이 모노 WAV 로만 바꾼다.

    브라우저로 녹음한 webm 은 탐색 정보가 없어 특정 시점으로 정확히 이동하지 못한다.
    WAV 는 항상 정확히 이동된다. 듣기용이라 음질을 위해 샘플레이트는 원본 그대로 둔다.
    """
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src.resolve()), "-ac", "1", "-c:a", "pcm_s16le", str(dst.resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise AudioError(f"재생용 변환 실패: {proc.stderr.strip()[:300]}")
    return dst


def to_wav(src: Path, dst: Path, *, preset: str = config.DEFAULT_PRESET,
           denoise: bool = True, start_sec: float = 0.0) -> Path:
    """src(webm/mp3/m4a/wav/…) → dst(16kHz mono pcm_s16le wav).

    start_sec > 0 이면 그 지점부터만 만든다 — 이어서 변환할 때 앞부분을 다시 처리하지 않으려고.
    """
    p = config.PRESETS.get(preset, config.PRESETS[config.DEFAULT_PRESET])
    chain = p["chain"] if denoise else "loudnorm=I=-16:TP=-1.5:LRA=11"
    seek = ["-ss", f"{start_sec:.3f}"] if start_sec > 0 else []
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        *seek, "-i", str(src.resolve()),
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


def pcm_to_wav(pcm: Path, dst: Path, sample_rate: int = 16000) -> Path:
    """실시간 녹음 중 디스크에 이어 붙인 raw PCM(16bit 모노)을 WAV 로 감싼다.

    녹음 도중 서버가 꺼져도 그때까지의 소리는 .pcm 파일에 남아 있어서 되살릴 수 있다.
    """
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm.resolve()),
           "-c:a", "pcm_s16le", str(dst.resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise AudioError(f"녹음 복구 실패: {proc.stderr.strip()[:300]}")
    return dst
