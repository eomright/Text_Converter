"""전역 설정. 여기 값만 바꾸면 파이프라인 동작이 달라진다."""
from __future__ import annotations

import os
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────
# 모델(수백 MB)과 녹음 파일은 OneDrive 동기화 대상에서 빼야 한다.
# DESIGN.md §12 리스크 참고.
RUNTIME_DIR = Path(
    os.environ.get("KSTT_RUNTIME", Path.home() / "korean-stt-runtime")
).resolve()

DATA_DIR = RUNTIME_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
EVAL_DIR = DATA_DIR / "eval"
DB_PATH = DATA_DIR / "app.db"
GLOSSARY_PATH = DATA_DIR / "glossary.json"

for _d in (UPLOAD_DIR, MODEL_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# ── 모델 ──────────────────────────────────────────────────────────────
# large-v3-turbo: 인코더는 large-v3와 동일, 디코더만 32→4 레이어.
# 품질은 large급 / 속도는 medium급 → GPU 없는 CPU 환경의 최적해.
DEFAULT_MODEL = os.environ.get("KSTT_MODEL", "large-v3-turbo")

AVAILABLE_MODELS = {
    "small": {"repo": "small", "size_mb": 250, "note": "가장 빠름, 정확도 아쉬움"},
    "medium": {"repo": "medium", "size_mb": 800, "note": "무난"},
    "large-v3-turbo": {"repo": "large-v3-turbo", "size_mb": 800, "note": "권장 기본값"},
    "large-v3": {"repo": "large-v3", "size_mb": 2900, "note": "최고 품질, 2배 느림 (강의실 모드 기본)"},
}

# NVIDIA GPU가 없으므로 CPU + int8 양자화 고정.
DEVICE = os.environ.get("KSTT_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("KSTT_COMPUTE", "int8")
# Core Ultra 7 155H = 16코어. 0이면 CTranslate2가 알아서 잡는다.
CPU_THREADS = int(os.environ.get("KSTT_THREADS", "0"))

# ── 추론 파라미터 (DESIGN.md §5.2) ────────────────────────────────────
# 한국어에서 Whisper가 터지는 전형적 패턴을 막기 위한 값들이다.
# 각 항목의 근거는 DESIGN.md §5.1 표 참고. 함부로 바꾸면 환각이 돌아온다.
TRANSCRIBE_OPTS: dict = {
    "language": "ko",                 # 자동감지 금지 — 짧은 오디오에서 일본어로 샌다
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "condition_on_previous_text": False,  # ★ "네네네네…" 반복 환각 차단
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "vad_filter": True,               # ★ 무음 구간 자막 환각 차단
    "vad_parameters": {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "threshold": 0.5,
    },
    "word_timestamps": True,
}

# 동시 처리 1건. 16코어를 한 작업에 몰아줘야 개별 응답이 빠르다.
MAX_CONCURRENT_JOBS = 1

# ── 녹음 환경 프리셋 ─────────────────────────────────────────────────
# 내 목소리를 가까이서 녹음할 때와, 강의실 뒤에서 교수님 목소리를 녹음할 때는
# 필요한 처리가 다르다. 강의실 체인은 scripts/preprocess_bench.py 로 실측해서 골랐다.
RNNOISE_DIR = MODEL_DIR / "rnnoise"
_LOUD = "loudnorm=I=-16:TP=-1.5:LRA=11"

PRESETS: dict[str, dict] = {
    "standard": {
        "label": "가까이 (내 목소리)",
        "chain": f"highpass=f=80,{_LOUD},afftdn=nf=-25",
        "model": None,         # None = 설정 화면에서 고른 모델 (기본 large-v3-turbo)
        "vad_threshold": 0.5,
        "live_gain": False,
    },
    "lecture": {
        "label": "강의실 (멀리 있는 작은 목소리)",
        # 실측(2026-09-13, 가상 뒷자리 녹음 CER): standard 21.6% / dynaudnorm 22.6% /
        # speechnorm 25.5% / RNNoise+증폭 49.4% / 무처리 31.3%.
        # 음량 정규화는 효과가 크지만, 추가 증폭·잡음제거는 오히려 나빠졌다 → standard 와 같은 체인.
        "chain": f"highpass=f=80,{_LOUD},afftdn=nf=-25",
        # 뒷자리 녹음 CER: turbo 21.6% → large-v3 13.5%. 오류 대부분이 "잘못 알아듣기"라서
        # 디코더가 32층인 large-v3 가 문맥으로 훨씬 잘 복원한다. 대신 약 2배 느리다.
        # VAD 기준·무음 판정 완화·일반 주제 프롬프트는 효과 없었다 (모두 21.6~21.9%).
        "model": "large-v3",
        "vad_threshold": 0.5,
        "live_gain": True,     # 실시간 모드에서 VAD 전에 자동 게인 — 안 하면 말소리 15%를 놓친다
    },
}
DEFAULT_PRESET = "standard"

# ── 실시간(하이브리드) 모드 ──────────────────────────────────────────
# 측정 결과(2026-09-11): Whisper 인코더는 실제 길이와 무관하게 항상 30초 윈도우를
# 패딩 처리하므로 호출당 고정 비용이 붙는다. large-v3-turbo 는 인코더가 large-v3와
# 같아서 3초 청크도 6.9초가 걸린다 → 실시간 불가.
# small 은 고정 비용이 1.8초뿐이라 발화당 ~2초 지연으로 실시간이 가능하다.
#   청크길이  large-v3-turbo   small
#     3초        6.87s         1.86s
#    12초        7.85s         2.41s
LIVE_MODEL = os.environ.get("KSTT_LIVE_MODEL", "small")

# 실시간 초안용 파라미터. beam을 1로 낮춰 속도를 확보한다.
# 어차피 녹음을 멈추면 DEFAULT_MODEL 이 전체를 다시 정밀 변환해 교체한다.
LIVE_TRANSCRIBE_OPTS: dict = {
    "language": "ko",
    "task": "transcribe",
    "beam_size": 1,
    "best_of": 1,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,       # 이미 VAD로 잘라서 넘기므로 중복 불필요
    "word_timestamps": False,
}

LIVE_SAMPLE_RATE = 16000
# 발화 끝을 판정하는 무음 길이. 짧으면 문장이 토막나고, 길면 반응이 느려진다.
LIVE_SILENCE_MS = 500
# 무음 없이 계속 말할 때 강제로 끊는 한계 (Whisper 윈도우가 30초라 그 안쪽으로)
LIVE_MAX_UTTERANCE_S = 20.0
# 이 길이 미만의 조각은 너무 짧아 인식이 불안정하므로 다음 발화와 합친다
LIVE_MIN_UTTERANCE_S = 0.6

HOST = os.environ.get("KSTT_HOST", "127.0.0.1")
PORT = int(os.environ.get("KSTT_PORT", "8000"))
