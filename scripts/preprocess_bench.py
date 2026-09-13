"""전처리 체인 × VAD 설정 조합별 CER 비교 — "무엇이 강의 녹음에 실제로 효과가 있나".

증폭/잡음제거는 직관적으로는 좋아 보이지만, Whisper 는 원래 잡음에 강하게 학습돼 있어서
과한 잡음제거가 오히려 정확도를 떨어뜨리는 경우가 흔하다. 그래서 추측하지 않고 잰다.

사용법
    python -X utf8 -m scripts.preprocess_bench                       # 기본 평가셋
    python -X utf8 -m scripts.preprocess_bench --files 강의1.m4a      # 내 강의 녹음 (+ 강의1.txt 정답)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import audio, config, postprocess                # noqa: E402
from server.transcriber import Transcriber                  # noqa: E402
from scripts.benchmark import cer                           # noqa: E402

RNN_DIR = config.MODEL_DIR / "rnnoise"
LOUD = "loudnorm=I=-16:TP=-1.5:LRA=11"

CHAINS = {
    "현재(standard)":       f"highpass=f=80,{LOUD},afftdn=nf=-25",
    "원본(무처리)":          "anull",
    "speechnorm":           f"highpass=f=80,{LOUD},speechnorm=e=12.5:r=0.0001:l=1,alimiter=limit=0.95",
    "dynaudnorm":           f"highpass=f=80,{LOUD},dynaudnorm=f=250:g=15:m=30:p=0.95,alimiter=limit=0.95",
    "RNNoise+dynaudnorm":   f"highpass=f=80,{LOUD},arnndn=m=sh.rnnn,dynaudnorm=f=250:g=15:m=30:p=0.95,alimiter=limit=0.95",
    "RNNoise(70%)+dynaud":  f"highpass=f=80,{LOUD},arnndn=m=sh.rnnn:mix=0.7,dynaudnorm=f=250:g=15:m=30:p=0.95,alimiter=limit=0.95",
}


def run_chain(src: Path, dst: Path, chain: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-af", chain, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    # arnndn 모델 경로의 'C:' 콜론이 필터 문법과 충돌하므로 모델 폴더에서 실행
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RNN_DIR))
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:300])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[
        "long_ko_16k.wav", "long_ko_lecture_mild.wav", "long_ko_lecture_hard.wav"])
    ap.add_argument("--chains", default=",".join(CHAINS))
    ap.add_argument("--vad", default="0.5", help="쉼표 구분 VAD threshold 목록. 예: 0.5,0.3")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()

    tr = Transcriber()
    tr.load(args.model)
    vads = [float(v) for v in args.vad.split(",")]
    chains = [c for c in args.chains.split(",") if c in CHAINS]
    rows = []

    for fname in args.files:
        src = Path(fname) if Path(fname).is_absolute() else config.EVAL_DIR / fname
        ref_path = src.with_suffix(".txt")
        if not ref_path.exists():
            ref_path = src.with_name(src.stem.replace("_16k", "") + ".txt")
        ref = ref_path.read_text(encoding="utf-8-sig")
        print(f"\n■ {src.name}")
        for cname in chains:
            tmp = config.EVAL_DIR / f"_pp_{src.stem}.wav"
            try:
                run_chain(src, tmp, CHAINS[cname])
            except Exception as e:
                print(f"  {cname:<22} 전처리 실패: {e}")
                continue
            for v in vads:
                opts_backup = dict(config.TRANSCRIBE_OPTS["vad_parameters"])
                config.TRANSCRIBE_OPTS["vad_parameters"]["threshold"] = v
                try:
                    t0 = time.time()
                    segs, dur, _ = tr.transcribe(tmp, initial_prompt=args.prompt)
                    el = time.time() - t0
                    hyp = " ".join(s["text"] for s in postprocess.clean_transcript(segs))
                    e = cer(ref, hyp)
                    rows.append((src.stem, cname, v, e, el))
                    print(f"  {cname:<22} VAD {v:.2f}  CER {e*100:5.1f}%  ({el:4.0f}s)")
                finally:
                    config.TRANSCRIBE_OPTS["vad_parameters"].update(opts_backup)
            tmp.unlink(missing_ok=True)

    # ── 요약표: 행=처리, 열=파일 ──
    files = list(dict.fromkeys(r[0] for r in rows))
    print("\n" + "=" * (32 + 12 * len(files)))
    print(f"{'처리 / VAD':<30}" + "".join(f"{f[-12:]:>12}" for f in files))
    print("-" * (32 + 12 * len(files)))
    for cname in chains:
        for v in vads:
            cells = []
            for f in files:
                m = [r for r in rows if r[0] == f and r[1] == cname and r[2] == v]
                cells.append(f"{m[0][3]*100:11.1f}%" if m else f"{'-':>12}")
            print(f"{cname + f' / {v}':<30}" + "".join(cells))
    print("=" * (32 + 12 * len(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
