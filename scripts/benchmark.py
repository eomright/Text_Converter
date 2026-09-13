"""모델별 한국어 정확도(CER) + 속도 측정 — DESIGN.md §5.5 / Phase 3.

"잘 되는 것 같다"로 끝내지 않고 숫자로 모델을 고르기 위한 스크립트.

사용법
------
1) 평가셋 준비: data/eval/ 에 오디오와 정답 텍스트를 같은 이름으로 둔다.
       data/eval/quiet.m4a   +  data/eval/quiet.txt
       data/eval/noisy.m4a   +  data/eval/noisy.txt
       data/eval/lecture.m4a +  data/eval/lecture.txt
2) 실행:
       .venv\\Scripts\\python.exe -m scripts.benchmark
       .venv\\Scripts\\python.exe -m scripts.benchmark --models small,large-v3-turbo
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import audio, config, postprocess          # noqa: E402
from server.transcriber import Transcriber             # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}


def normalize(s: str) -> str:
    """CER 비교용 정규화: 공백·문장부호를 제거해 표기 차이의 영향을 줄인다."""
    s = re.sub(r"[\s]+", "", s)
    s = re.sub(r"[.,!?~\"'`\-—…·:;()\[\]{}]", "", s)
    return s


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


def find_pairs(eval_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for a in sorted(eval_dir.iterdir()):
        if a.suffix.lower() in AUDIO_EXT:
            ref = a.with_suffix(".txt")
            if ref.exists():
                pairs.append((a, ref))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small,medium,large-v3-turbo",
                    help="쉼표로 구분. 예: small,large-v3-turbo")
    ap.add_argument("--eval-dir", default=str(config.EVAL_DIR))
    ap.add_argument("--prompt", default="", help="initial_prompt (용어집) 효과 측정용")
    ap.add_argument("--no-denoise", action="store_true")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    pairs = find_pairs(eval_dir)
    if not pairs:
        print(f"평가셋이 없습니다: {eval_dir}")
        print("오디오 파일과 같은 이름의 .txt(정답)를 넣어주세요.")
        print("  예) quiet.m4a + quiet.txt")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"평가셋 {len(pairs)}개 · 모델 {len(models)}개\n")

    results: list[tuple[str, str, float, float, float]] = []
    tr = Transcriber()

    for name in models:
        print(f"── {name} 로딩…")
        try:
            t0 = time.time()
            tr.load(name)
            print(f"   로드 {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"   실패: {e}")
            continue

        for wav_src, ref_path in pairs:
            ref = ref_path.read_text(encoding="utf-8")
            tmp = wav_src.with_suffix(".bench.wav")
            try:
                audio.to_wav(wav_src, tmp, denoise=not args.no_denoise)
                segs, dur, elapsed = tr.transcribe(tmp, initial_prompt=args.prompt)
                segs = postprocess.clean_transcript(segs)
                hyp = " ".join(s["text"] for s in segs)
                e = cer(ref, hyp)
                speed = dur / elapsed if elapsed else 0
                results.append((name, wav_src.stem, e, elapsed, speed))
                print(f"   {wav_src.stem:<14} CER {e*100:5.1f}%  "
                      f"{elapsed:5.1f}s  ({speed:.2f}x 실시간)")
                (eval_dir / f"{wav_src.stem}.{name}.hyp.txt").write_text(
                    hyp, encoding="utf-8")
            except Exception as ex:
                print(f"   {wav_src.stem}: 실패 {ex}")
            finally:
                tmp.unlink(missing_ok=True)
        print()

    # ── 요약 ──
    print("=" * 58)
    print(f"{'모델':<18}{'평균 CER':>12}{'평균 시간':>12}{'실시간 배속':>14}")
    print("-" * 58)
    for name in models:
        rows = [r for r in results if r[0] == name]
        if not rows:
            continue
        avg_cer = sum(r[2] for r in rows) / len(rows)
        avg_t = sum(r[3] for r in rows) / len(rows)
        avg_s = sum(r[4] for r in rows) / len(rows)
        print(f"{name:<18}{avg_cer*100:11.1f}%{avg_t:11.1f}s{avg_s:13.2f}x")
    print("=" * 58)
    print("\nCER이 낮을수록 정확. 배속이 높을수록 빠름(1x = 실시간).")
    print("이 표를 DESIGN.md §5.5 에 채우고 기본 모델을 확정하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
