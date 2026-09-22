"""faster-whisper 래퍼.

모델은 프로세스당 1회만 로드해서 메모리에 상주시킨다.
매 요청마다 로드하면 요청당 20~30초를 그냥 버린다.

용도별로 **여러 모델을 동시에 상주**시킨다 (안 쓰는 모델은 _evict_unused 가 내림).
  - LIVE_MODEL (small)      : 녹음 중 발화 단위 초안. 고정 비용 1.8초
  - DEFAULT_MODEL (turbo)   : 녹음 종료 후 전체 정밀 변환. 고정 비용 7초
  - large-v3                : 강의실 모드 정밀 변환, 교정 모드 비교 분석
근거 수치는 config.LIVE_MODEL / config.PRESETS 주석 참고.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from . import config


class Transcriber:
    def __init__(self) -> None:
        self._models: OrderedDict[str, object] = OrderedDict()
        self._load_lock = threading.Lock()              # 로딩 직렬화
        self._infer_locks: dict[str, threading.Lock] = {}  # 모델별 추론 락
        self._batch_model = config.DEFAULT_MODEL        # UI에서 고른 정밀 변환용 모델

    # ── 모델 관리 ────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        """정밀 변환에 쓰는 현재 모델 이름."""
        return self._batch_model

    @property
    def loaded(self) -> bool:
        return self._batch_model in self._models

    def loaded_names(self) -> list[str]:
        return list(self._models.keys())

    def _infer_lock(self, name: str) -> threading.Lock:
        with self._load_lock:
            return self._infer_locks.setdefault(name, threading.Lock())

    def _evict_unused(self, keep: set[str]) -> None:
        """상주 대상이 아닌 모델을 내려 메모리를 돌려준다 (RAM 16GB 환경)."""
        for name in [n for n in self._models if n not in keep]:
            lock = self._infer_locks.get(name)
            # 추론 중이면 건드리지 않는다 — 다음 기회에 정리된다
            if lock is not None and not lock.acquire(blocking=False):
                continue
            try:
                self._models.pop(name, None)
            finally:
                if lock is not None:
                    lock.release()

    def get_model(self, name: str):
        """이름으로 모델을 얻는다. 없으면 로드한다 (최초 1회만 비용 발생)."""
        if name not in config.AVAILABLE_MODELS:
            raise ValueError(f"알 수 없는 모델: {name}")
        m = self._models.get(name)
        if m is not None:
            return m
        with self._load_lock:
            m = self._models.get(name)
            if m is not None:
                return m
            from faster_whisper import WhisperModel

            kwargs = dict(
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                download_root=str(config.MODEL_DIR),
            )
            if config.CPU_THREADS > 0:
                kwargs["cpu_threads"] = config.CPU_THREADS

            m = WhisperModel(config.AVAILABLE_MODELS[name]["repo"], **kwargs)
            self._models[name] = m
            self._infer_locks.setdefault(name, threading.Lock())
        # 실시간용과 현재 정밀용만 남긴다
        self._evict_unused({config.LIVE_MODEL, self._batch_model, name})
        return m

    def load(self, name: str | None = None) -> str:
        """정밀 변환용 모델을 교체(선택)하고 로드한다."""
        name = name or config.DEFAULT_MODEL
        if name not in config.AVAILABLE_MODELS:
            raise ValueError(f"알 수 없는 모델: {name}")
        self._batch_model = name
        self.get_model(name)
        self._evict_unused({config.LIVE_MODEL, name})
        return name

    def preload_live(self) -> None:
        """실시간 모드용 모델을 미리 올려둔다 (첫 발화가 느려지지 않도록)."""
        self.get_model(config.LIVE_MODEL)

    # ── 추론 ─────────────────────────────────────────────────────────
    def transcribe(
        self,
        wav_path: Path,
        *,
        initial_prompt: str = "",
        model_name: str | None = None,
        vad_threshold: float | None = None,
        on_segment: Callable[[dict, float], None] | None = None,
    ) -> tuple[list[dict], float, float]:
        """(segments, audio_duration_sec, elapsed_sec)

        on_segment(segment, percent) 은 세그먼트가 나올 때마다 즉시 호출된다.
        → 사용자는 전체가 끝나기 전에 첫 문장을 볼 수 있다.
        """
        name = model_name or self._batch_model
        model = self.get_model(name)

        opts = dict(config.TRANSCRIBE_OPTS)
        # 전역 설정을 건드리지 않도록 복사본에만 반영 (동시 작업 간 간섭 방지)
        opts["vad_parameters"] = dict(opts["vad_parameters"])
        if vad_threshold is not None:
            opts["vad_parameters"]["threshold"] = vad_threshold
        if initial_prompt.strip():
            # 용어집을 문맥으로 주입 — 고유명사 인식률에 가장 크게 기여한다
            opts["initial_prompt"] = initial_prompt.strip()

        started = time.time()
        with self._infer_lock(name):
            seg_iter, info = model.transcribe(str(wav_path), **opts)

            total = float(getattr(info, "duration", 0.0)) or 0.0
            out: list[dict] = []
            for s in seg_iter:              # 지연 생성 → 여기서 실제 연산이 돈다
                item = {
                    "start": round(float(s.start), 2),
                    "end": round(float(s.end), 2),
                    "text": (s.text or "").strip(),
                    "no_speech_prob": round(float(getattr(s, "no_speech_prob", 0.0)), 3),
                    "avg_logprob": round(float(getattr(s, "avg_logprob", 0.0)), 3),
                }
                out.append(item)
                if on_segment:
                    pct = min(99.0, (item["end"] / total * 100.0) if total else 0.0)
                    on_segment(item, pct)

        return out, total, time.time() - started

    def transcribe_words(
        self,
        wav_path: Path,
        model_name: str,
        *,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[dict]:
        """교정 모드의 모델 비교용: 단어별 타임스탬프만 뽑는다.

        원래 결과와 구간 경계가 달라도 단어 시점으로 맞춰 비교하기 위해서다.
        용어집은 넣지 않는다 — 같은 힌트를 주면 같은 방향으로 틀려서 비교 의미가 줄어든다.
        """
        model = self.get_model(model_name)
        opts = dict(config.TRANSCRIBE_OPTS)
        opts["vad_parameters"] = dict(opts["vad_parameters"])
        opts["word_timestamps"] = True
        words: list[dict] = []
        with self._infer_lock(model_name):
            seg_iter, info = model.transcribe(str(wav_path), **opts)
            total = float(getattr(info, "duration", 0.0)) or 0.0
            for s in seg_iter:
                for w in (s.words or []):
                    words.append({"start": float(w.start), "end": float(w.end),
                                  "word": w.word or ""})
                if on_progress and total:
                    on_progress(min(99.0, float(s.end) / total * 100.0))
        return words

    def transcribe_array(self, audio, *, initial_prompt: str = "") -> str:
        """실시간용: numpy float32 배열 하나를 받아 텍스트만 빠르게 돌려준다.

        이미 VAD로 잘라온 한 발화라서 파일 I/O도 VAD도 거치지 않는다.
        """
        model = self.get_model(config.LIVE_MODEL)
        opts = dict(config.LIVE_TRANSCRIBE_OPTS)
        if initial_prompt.strip():
            opts["initial_prompt"] = initial_prompt.strip()
        with self._infer_lock(config.LIVE_MODEL):
            seg_iter, _ = model.transcribe(audio, **opts)
            return " ".join((s.text or "").strip() for s in seg_iter).strip()


# 프로세스 전역 싱글턴
transcriber = Transcriber()
