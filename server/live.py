"""실시간(하이브리드) 모드의 발화 단위 절단 로직.

브라우저가 16kHz mono int16 PCM 을 계속 흘려보내면,
여기서 Silero VAD 로 "말이 끝난 지점"을 찾아 한 발화씩 잘라 넘긴다.

왜 고정 길이(예: 5초)로 안 자르나:
문장 중간을 자르면 Whisper 가 문맥을 잃어 정확도가 크게 떨어진다.
무음 경계에서 자르면 발화가 온전히 보존되어 초안 품질이 훨씬 낫다.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from . import config

SR = config.LIVE_SAMPLE_RATE
SILENCE_SAMPLES = int(SR * config.LIVE_SILENCE_MS / 1000)
MAX_SAMPLES = int(SR * config.LIVE_MAX_UTTERANCE_S)
MIN_SAMPLES = int(SR * config.LIVE_MIN_UTTERANCE_S)

# 자동 게인: 최근 10초의 상위 0.5% 진폭을 이 값에 맞춘다.
# 실측 — 뒷자리 강의 녹음(-45dBFS)을 그대로 VAD 에 넣으면 말소리의 85%만 잡혔고,
# 이 보정 후엔 100% 잡혔다. 최대 +30dB 로 제한해 무음 구간 잡음이 폭주하지 않게 한다.
GAIN_TARGET = 0.2
GAIN_MAX = 10 ** (30 / 20)
GAIN_WINDOW = SR * 10

# 경계 탐지용 VAD. 여기서는 "말이 있는 구간"만 알면 되므로 공격적으로 잡는다.
_VAD = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=200,
    min_silence_duration_ms=300,
    speech_pad_ms=100,
)


class LiveSession:
    """녹음 1회분의 상태. WebSocket 연결 하나당 하나."""

    def __init__(self, *, live_gain: bool = False, pcm_path: Path | None = None) -> None:
        self.live_gain = live_gain
        # 받은 소리를 디스크에도 바로 이어 붙인다. 메모리에만 두면 녹음 도중
        # 서버가 꺼지거나 탭이 닫힐 때 그때까지의 녹음이 통째로 사라진다.
        self.pcm_path = pcm_path
        self._pcm_file = open(pcm_path, "ab") if pcm_path else None
        self._unflushed = 0
        self.gain = 1.0          # 마지막으로 적용한 배율 (UI 표시용)
        self.pcm = bytearray()   # 전체 녹음 (int16 PCM). 종료 후 정밀 변환에 그대로 쓴다
        self.cursor = 0          # 샘플 인덱스 — 여기까지는 이미 초안 처리 완료
        self.segments: list[dict] = []
        self.closed = False

    # ── 입력 ─────────────────────────────────────────────────────────
    def feed(self, data: bytes) -> None:
        self.pcm.extend(data)
        if self._pcm_file:
            self._pcm_file.write(data)
            self._unflushed += len(data)
            if self._unflushed >= SR * 2:        # 약 1초 분량마다 OS 로 넘긴다
                self._pcm_file.flush()
                self._unflushed = 0

    def close_file(self) -> None:
        if self._pcm_file:
            self._pcm_file.close()
            self._pcm_file = None

    @property
    def total_samples(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_sec(self) -> float:
        return self.total_samples / SR

    def _slice(self, start: int, end: int) -> np.ndarray:
        raw = bytes(self.pcm[start * 2:end * 2])
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # ── 발화 경계 찾기 ───────────────────────────────────────────────
    def next_utterance(self, *, final: bool = False) -> tuple[int, int, np.ndarray] | None:
        """확정된 발화가 있으면 (시작샘플, 끝샘플, 오디오)를 돌려주고 커서를 옮긴다.

        final=True 면 남은 걸 무조건 비운다 (녹음 종료 시).
        """
        pending_len = self.total_samples - self.cursor
        if pending_len <= 0:
            return None
        if not final and pending_len < MIN_SAMPLES:
            return None

        pending = self._slice(self.cursor, self.total_samples)
        if self.live_gain:
            pending = self._apply_gain(pending)

        try:
            ts = get_speech_timestamps(pending, _VAD, sampling_rate=SR)
        except Exception:
            ts = []

        if not ts:
            # 전부 무음. 버퍼가 쌓이지 않게 커서를 당겨두되, 꼬리는 조금 남긴다
            if pending_len > SR * 2:
                self.cursor = self.total_samples - SILENCE_SAMPLES
            return None

        last_end = ts[-1]["end"]
        tail_silence = pending_len - last_end

        if final:
            cut = pending_len
        elif tail_silence >= SILENCE_SAMPLES:
            cut = min(last_end + SILENCE_SAMPLES // 2, pending_len)
        elif pending_len >= MAX_SAMPLES:
            cut = pending_len          # 쉬지 않고 말하는 중 — 강제로 끊는다
        else:
            return None

        if cut < MIN_SAMPLES and not final:
            return None

        start, end = self.cursor, self.cursor + cut
        audio = pending[:cut]
        self.cursor = end

        # 잘라낸 구간에 실제 말소리가 거의 없으면 버린다
        speech = sum(t["end"] - t["start"] for t in ts if t["start"] < cut)
        if speech < SR * 0.2:
            return None
        return start, end, audio

    def _apply_gain(self, x: np.ndarray) -> np.ndarray:
        ref = self._slice(max(0, self.total_samples - GAIN_WINDOW), self.total_samples)
        peak = float(np.percentile(np.abs(ref), 99.5)) if len(ref) else 0.0
        self.gain = float(np.clip(GAIN_TARGET / (peak + 1e-9), 1.0, GAIN_MAX))
        return np.clip(x * self.gain, -1.0, 1.0)

    # ── 저장 ─────────────────────────────────────────────────────────
    def save_wav(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(bytes(self.pcm))
        return path
