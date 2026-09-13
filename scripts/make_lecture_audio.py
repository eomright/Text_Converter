"""깨끗한 음성을 "멀리서 녹음한 강의" 처럼 망가뜨린다 — 전처리 효과를 숫자로 재기 위한 평가셋.

실제 강의 녹음의 세 가지 문제를 재현한다.
  1. 잔향  : 강의실 벽에 반사된 소리가 자음을 뭉갠다 (RT60)
  2. 잡음  : 에어컨/프로젝터 팬 소음 (핑크 노이즈 + 저주파 험)
  3. 작은 음량 : 마이크가 교수님과 멀어 입력 레벨 자체가 낮다

사용법
    python -m scripts.make_lecture_audio data/eval/long_ko_16k.wav
    → data/eval/long_ko_lecture_mild.wav / long_ko_lecture_hard.wav (+ 정답 txt 복사)
"""
from __future__ import annotations

import shutil
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
RNG = np.random.default_rng(42)  # 재현 가능하게 고정

PRESETS = {
    # 앞자리, 조용한 강의실
    "mild": dict(rt60=0.5, drr_db=3.0, snr_db=15.0, level_dbfs=-35.0),
    # 뒷자리, 에어컨 켜진 큰 강의실, 교수님 목소리 작음
    "hard": dict(rt60=1.0, drr_db=-3.0, snr_db=5.0, level_dbfs=-45.0),
}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, "16kHz mono 만 지원"
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return x.astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    y = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def active_rms(x: np.ndarray) -> float:
    """무음 구간을 빼고 말소리 구간의 RMS 만 잰다 (SNR 기준)."""
    frame = SR // 50
    n = len(x) // frame
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    thr = np.max(e) * 0.1
    act = e[e > thr]
    return float(np.sqrt(np.mean(act ** 2))) if len(act) else rms(x)


def room_ir(rt60: float, drr_db: float) -> np.ndarray:
    """직접음 + 지수 감쇠하는 확산 잔향으로 된 간이 강의실 임펄스 응답."""
    n = int(SR * rt60 * 1.2)
    t = np.arange(n) / SR
    decay = np.exp(-6.908 * t / rt60)            # 60dB 감쇠 = ln(1000)
    tail = RNG.standard_normal(n) * decay
    tail[: int(SR * 0.01)] = 0                    # 첫 반사까지 10ms
    ir = np.zeros(n, np.float32)
    ir[0] = 1.0
    # 직접음 대비 잔향 에너지 비율(DRR) 맞추기
    tail *= 1.0 / (np.sqrt(np.sum(tail ** 2)) * 10 ** (drr_db / 20))
    ir += tail.astype(np.float32)
    return ir


def pink_noise(n: int) -> np.ndarray:
    white = np.fft.rfft(RNG.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    pink = np.fft.irfft(white / np.sqrt(f), n)
    t = np.arange(n) / SR
    hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)
    x = pink / rms(pink) + hum / rms(hum) * 0.3
    return (x / rms(x)).astype(np.float32)


def degrade(clean: np.ndarray, rt60: float, drr_db: float,
            snr_db: float, level_dbfs: float) -> np.ndarray:
    wet = np.convolve(clean, room_ir(rt60, drr_db))[: len(clean)]
    noise = pink_noise(len(wet))
    noise *= active_rms(wet) / 10 ** (snr_db / 20)
    mix = wet + noise
    mix *= 10 ** (level_dbfs / 20) / active_rms(mix)
    return mix


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    clean = read_wav(src)
    ref = src.with_suffix(".txt")
    if not ref.exists():  # long_ko_16k.wav → long_ko.txt
        ref = src.with_name(src.stem.replace("_16k", "") + ".txt")
    base = src.stem.replace("_16k", "")

    for name, p in PRESETS.items():
        out = src.with_name(f"{base}_lecture_{name}.wav")
        y = degrade(clean, **p)
        write_wav(out, y)
        if ref.exists():
            shutil.copy(ref, out.with_suffix(".txt"))
        print(f"{out.name:<30} RT60 {p['rt60']}s · SNR {p['snr_db']:>4}dB · "
              f"레벨 {20*np.log10(active_rms(y)):.1f} dBFS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
