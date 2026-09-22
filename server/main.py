"""FastAPI 엔트리포인트. 정적 프론트엔드까지 같이 서빙하므로 서버는 이거 하나만 띄우면 된다."""
from __future__ import annotations

import asyncio
import json
import math
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

from fastapi import (
    FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import audio, config, db, live, postprocess, review
from .transcriber import transcriber


# ── 작업 상태 (메모리) ────────────────────────────────────────────────
class JobState:
    """진행 중 작업 1건. 이벤트를 버퍼링해서 늦게 붙은 WebSocket도 처음부터 볼 수 있다."""

    def __init__(self) -> None:
        self.status = "queued"
        self.percent = 0.0
        self.segments: list[dict] = []
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()

    def emit(self, msg: dict) -> None:
        """반드시 이벤트 루프 스레드에서 호출할 것."""
        self.events.append(msg)
        if msg.get("type") == "progress":
            self.percent = msg.get("percent", self.percent)
        for q in list(self.subscribers):
            q.put_nowait(msg)


JOBS: dict[str, JobState] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS)
# 실시간 초안은 별도 워커에서 — 정밀 변환 큐에 막히면 실시간이 아니게 된다
LIVE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")


def _safe_load(name: str) -> None:
    try:
        transcriber.load(name)
        print(f"[OK] 모델 로드 완료: {name}")
    except Exception as e:
        print(f"[FAIL] 모델 로드 실패: {e}")


def _warmup() -> None:
    # 정밀 변환용 먼저, 그다음 실시간용. 순서대로 올려야 첫 요청이 안 밀린다.
    _safe_load(config.DEFAULT_MODEL)
    try:
        transcriber.preload_live()
        print(f"[OK] 실시간 모델 로드 완료: {config.LIVE_MODEL}")
    except Exception as e:
        print(f"[FAIL] 실시간 모델 로드 실패: {e}")


def _salvage_recording(rec: dict) -> None:
    """녹음 도중 서버가 꺼진 기록: 디스크에 쌓아둔 .pcm 을 WAV 로 되살린다."""
    pcm = config.UPLOAD_DIR / f"{rec['id']}.pcm"
    wav = Path(rec["audio_path"])
    try:
        if pcm.exists() and pcm.stat().st_size >= live.SR:   # 0.5초 이상
            audio.pcm_to_wav(pcm, wav)
            pcm.unlink(missing_ok=True)
        if not wav.exists():
            db.set_status(rec["id"], "error", "녹음 파일을 되살리지 못했습니다.")
            return
        db.set_recorded(rec["id"], title=None, duration_sec=audio.probe_duration(wav),
                        status="interrupted",
                        error="녹음 도중 서버가 꺼졌습니다. 그때까지의 녹음은 저장됐고, 이어서 변환할 수 있습니다.")
    except Exception as e:
        traceback.print_exc()
        db.set_status(rec["id"], "error", f"녹음 복구 실패: {e}")


def _recover_on_start() -> list[dict]:
    """지난번 서버가 꺼질 때 하던 작업을 '중단됨'으로 정리한다 (변환은 저장된 곳까지 남아 있음)."""
    rows = db.mark_interrupted()
    for rec in rows:
        if rec["status"] == "recording":
            _salvage_recording(rec)
    # 끊긴 작업이 남긴 전처리 임시 파일 (강의 하나에 100MB 넘게 남는다)
    for pattern in ("*.16k.wav", "*.compare.wav"):
        for f in config.UPLOAD_DIR.glob(pattern):
            f.unlink(missing_ok=True)
    return rows


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    if not audio.ffmpeg_available():
        print("[WARN] ffmpeg 를 PATH 에서 찾지 못했습니다. 전처리가 실패합니다.")
    stopped = _recover_on_start()
    if stopped:
        print(f"[INFO] 지난번에 끊긴 작업 {len(stopped)}개를 '중단됨'으로 표시했습니다.")
        if config.AUTO_RESUME_ON_START:
            loop = asyncio.get_running_loop()
            for rec in stopped:
                try:
                    _resume(db.get(rec["id"]), loop)
                except Exception as e:
                    print(f"[WARN] 자동 이어서 변환 실패 {rec['id']}: {e}")
    # 모델 로딩(수십 초)이 서버 기동을 막지 않도록 백그라운드 스레드에서 예열
    threading.Thread(target=_warmup, daemon=True).start()
    print(f"[READY] http://{config.HOST}:{config.PORT}  (모델 예열 중)")
    yield


app = FastAPI(title="한국어 받아쓰기", lifespan=lifespan)


# ── 변환 파이프라인 ───────────────────────────────────────────────────
def _run_job(job_id: str, src: Path, loop: asyncio.AbstractEventLoop,
             denoise: bool, prompt_override: str | None,
             preset: str = config.DEFAULT_PRESET) -> None:
    """워커 스레드에서 실행. UI 갱신은 loop.call_soon_threadsafe 로 넘긴다.

    몇 문장마다 결과를 DB 에 저장하고(progress_sec), 다시 실행되면 거기서부터 이어간다.
    50~80분 강의는 변환에 1~2시간 걸려서, 끝날 때 한 번만 저장하면 도중에 꺼질 때 전부 사라진다.
    """
    state = JOBS[job_id]

    def push(msg: dict) -> None:
        loop.call_soon_threadsafe(state.emit, msg)

    rec = db.get(job_id)
    offset = float(rec.get("progress_sec") or 0.0)
    prior = [s for s in rec["segments"] if s["end"] <= offset + 0.01] if offset > 0 else []
    elapsed_before = float(rec.get("elapsed_sec") or 0.0) if offset > 0 else 0.0
    total = float(rec.get("duration_sec") or 0.0) or audio.probe_duration(src)
    state.segments = list(prior)

    wav = src.with_suffix(".16k.wav")
    try:
        db.set_status(job_id, "running")
        resume_note = f" — {int(offset // 60)}분 {int(offset % 60)}초부터 이어서" if offset > 0 else ""
        push({"type": "status", "status": "running", "stage": "전처리" + resume_note})

        audio.to_wav(src, wav, preset=preset, denoise=denoise, start_sec=offset)
        vad_threshold = config.PRESETS[preset]["vad_threshold"]
        model_name = config.PRESETS[preset].get("model") or transcriber.model_name
        if model_name not in transcriber.loaded_names():
            push({"type": "status", "status": "running",
                  "stage": f"모델 불러오는 중 ({model_name}, 최초 1회 30초 정도)"})

        glossary = postprocess.load_glossary()
        prompt = prompt_override if prompt_override is not None else glossary["prompt"]
        replacements = glossary["replacements"]

        push({"type": "status", "status": "running",
              "stage": f"변환 ({model_name}){resume_note}"})

        saved = list(prior)          # 중간 저장용 (정리된 문장)
        progress = offset            # 여기(초)까지 처리함 — 걸러진 문장도 진행으로 친다
        last_save, unsaved = time.time(), 0

        def on_segment(seg: dict, pct: float) -> None:
            nonlocal progress, last_save, unsaved
            seg = {**seg, "start": round(seg["start"] + offset, 2),
                   "end": round(seg["end"] + offset, 2)}
            progress = seg["end"]
            cleaned = postprocess.clean_segment(seg["text"], replacements)
            if cleaned:
                item = {**seg, "text": cleaned}
                state.segments.append(item)
                saved.append(item)
                push({"type": "segment", **item})
            push({"type": "progress",
                  "percent": round(min(99.0, progress / total * 100) if total else pct, 1)})
            unsaved += 1
            if unsaved >= config.SAVE_EVERY_SEGMENTS or time.time() - last_save >= config.SAVE_EVERY_SEC:
                db.save_partial(job_id, saved, progress)
                last_save, unsaved = time.time(), 0

        raw, _, elapsed = transcriber.transcribe(
            wav, initial_prompt=prompt, vad_threshold=vad_threshold,
            model_name=model_name,
            on_segment=on_segment
        )
        shifted = [{**s, "start": round(s["start"] + offset, 2),
                    "end": round(s["end"] + offset, 2)} for s in raw]

        # 스트리밍 중에는 세그먼트 단위로만 정리했으니, 끝나고 전체 맥락으로 한 번 더
        final = postprocess.clean_transcript(prior + shifted, replacements)
        text = postprocess.to_paragraphs(final)
        elapsed_total = elapsed_before + elapsed
        db.finish(job_id, final, text, elapsed_total)

        push({
            "type": "done",
            "segments": final,
            "text": text,
            "duration_sec": round(total, 1),
            "elapsed_sec": round(elapsed_total, 1),
            "speed": round(total / elapsed_total, 2) if elapsed_total > 0 else 0,
        })
    except Exception as e:
        traceback.print_exc()
        db.set_status(job_id, "error", str(e))
        push({"type": "error", "message": str(e)})
    finally:
        wav.unlink(missing_ok=True)


# ── API ───────────────────────────────────────────────────────────────
def _model_for(preset: str) -> str:
    return config.PRESETS[preset].get("model") or transcriber.model_name


def _enqueue(job_id: str, src: Path, loop: asyncio.AbstractEventLoop,
             denoise: bool = True, prompt: str | None = None,
             preset: str = config.DEFAULT_PRESET) -> None:
    JOBS[job_id] = JobState()
    db.set_status(job_id, "queued")
    EXECUTOR.submit(_run_job, job_id, src, loop, denoise, prompt, preset)


def _start_job(job_id: str, src: Path, title: str, denoise: bool,
               prompt: str | None, loop: asyncio.AbstractEventLoop,
               preset: str = config.DEFAULT_PRESET) -> dict:
    """저장된 오디오 파일 하나를 정밀 변환 큐에 넣는다."""
    duration = audio.probe_duration(src)
    db.create_job(job_id, title, str(src), _model_for(preset), duration, preset)
    _enqueue(job_id, src, loop, denoise, prompt, preset)
    return {"job_id": job_id, "title": title, "duration_sec": round(duration, 1)}


def _preset_of(rec: dict) -> str:
    """기록의 녹음 환경. 이 컬럼이 생기기 전 기록은 쓴 모델로 추정한다."""
    if rec.get("preset") in config.PRESETS:
        return rec["preset"]
    for name, p in config.PRESETS.items():
        if p.get("model") and p["model"] == rec.get("model"):
            return name
    return config.DEFAULT_PRESET


def _resume(rec: dict, loop: asyncio.AbstractEventLoop) -> dict:
    src = Path(rec.get("audio_path") or "")
    if not src.exists():
        raise HTTPException(404, "원본 오디오가 없어 이어서 변환할 수 없습니다.")
    preset = _preset_of(rec)
    _enqueue(rec["id"], src, loop, preset=preset)
    return {"job_id": rec["id"], "from_sec": rec["progress_sec"],
            "duration_sec": rec["duration_sec"], "model": _model_for(preset)}


@app.post("/api/transcripts/{job_id}/resume")
async def resume(job_id: str):
    """끊긴 변환을 저장된 지점부터 이어서 한다."""
    rec = db.get(job_id)
    if not rec:
        raise HTTPException(404, "없는 기록입니다.")
    if rec["status"] not in ("interrupted", "error"):
        raise HTTPException(409, "이어서 변환할 수 있는 상태가 아닙니다.")
    return _resume(rec, asyncio.get_running_loop())


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    title: str = Form(""),
    denoise: bool = Form(True),
    prompt: str | None = Form(None),
    preset: str = Form(config.DEFAULT_PRESET),
):
    if preset not in config.PRESETS:
        raise HTTPException(400, f"알 수 없는 녹음 환경: {preset}")
    suffix = Path(file.filename or "rec.webm").suffix or ".webm"
    job_id = uuid.uuid4().hex[:12]
    src = config.UPLOAD_DIR / f"{job_id}{suffix}"

    data = await file.read()
    if not data:
        raise HTTPException(400, "빈 파일입니다.")
    src.write_bytes(data)

    name = title.strip() or Path(file.filename or "녹음").stem or "녹음"
    return _start_job(job_id, src, name, denoise, prompt,
                      asyncio.get_running_loop(), preset)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """실시간(하이브리드) 모드.

    수신: 16kHz mono int16 PCM 바이너리 프레임, 그리고 {"type":"stop"} 텍스트
    송신: draft (small 초안) → finalizing (job_id) → 이후는 /ws/progress 가 이어받음

    녹음을 시작하는 순간 기록을 만들고, 소리는 .pcm 파일에, 초안은 DB 에 계속 저장한다.
    탭이 닫히거나 서버가 꺼져도 그때까지의 녹음과 초안이 남고 "이어서 변환"으로 마무리할 수 있다.
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    preset = ws.query_params.get("preset", config.DEFAULT_PRESET)
    if preset not in config.PRESETS:
        preset = config.DEFAULT_PRESET

    job_id = uuid.uuid4().hex[:12]
    wav_path = config.UPLOAD_DIR / f"{job_id}.wav"
    pcm_path = config.UPLOAD_DIR / f"{job_id}.pcm"
    default_title = "실시간_" + datetime.now().strftime("%y%m%d%H%M")
    db.create_job(job_id, default_title, str(wav_path), _model_for(preset), 0.0, preset,
                  status="recording")
    sess = live.LiveSession(live_gain=config.PRESETS[preset]["live_gain"], pcm_path=pcm_path)
    glossary = postprocess.load_glossary()
    prompt, replacements = glossary["prompt"], glossary["replacements"]
    drafts: list[dict] = []
    draining = False
    finished = False

    async def drain(final: bool = False) -> None:
        """확정된 발화를 꺼내 초안 변환 후 밀어준다."""
        nonlocal draining
        if draining and not final:
            return
        while draining:                       # final 은 진행 중인 drain 을 기다린다
            await asyncio.sleep(0.05)
        draining = True
        try:
            while True:
                u = await loop.run_in_executor(
                    LIVE_POOL, partial(sess.next_utterance, final=final))
                if u is None:
                    break
                start, end, chunk = u
                text = await loop.run_in_executor(
                    LIVE_POOL,
                    partial(transcriber.transcribe_array, chunk, initial_prompt=prompt))
                text = postprocess.clean_segment(text, replacements)
                if text:
                    seg = {"start": round(start / live.SR, 2),
                           "end": round(end / live.SR, 2), "text": text,
                           "gain_db": round(20 * math.log10(sess.gain), 1)}
                    drafts.append(seg)
                    db.set_drafts(job_id, drafts)      # 서버가 꺼져도 초안은 남게
                    if not finished:
                        await ws.send_json({"type": "draft", **seg})
                if not final:
                    break                     # 한 번에 한 발화씩만
        finally:
            draining = False

    async def close_recording() -> bool:
        """녹음 파일을 확정한다. 너무 짧으면 기록째 지우고 False."""
        sess.close_file()
        if sess.total_samples < live.SR * 0.5:
            pcm_path.unlink(missing_ok=True)
            db.delete(job_id)
            return False
        await loop.run_in_executor(LIVE_POOL, partial(sess.save_wav, wav_path))
        pcm_path.unlink(missing_ok=True)
        return True

    try:
        await ws.send_json({"type": "ready", "model": config.LIVE_MODEL,
                            "preset": preset, "job_id": job_id})
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return

            chunk = msg.get("bytes")
            if chunk:
                sess.feed(chunk)
                if not draining:
                    asyncio.create_task(drain())
                continue

            raw = msg.get("text")
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("type") != "stop":
                continue

            # ── 종료: 남은 발화를 비우고 전체를 정밀 변환으로 넘긴다 ──
            await drain(final=True)
            finished = True
            if not await close_recording():
                await ws.send_json({"type": "error", "message": "녹음이 너무 짧습니다."})
                return

            title = (data.get("title") or "").strip() or default_title
            db.set_recorded(job_id, title=title, duration_sec=sess.duration_sec, status="queued")
            _enqueue(job_id, wav_path, loop, bool(data.get("denoise", True)), None, preset)
            await ws.send_json({"type": "finalizing", "job_id": job_id, "title": title,
                                "duration_sec": round(sess.duration_sec, 1),
                                "model": _model_for(preset)})
            return
    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if not finished:
            # 정지 버튼 없이 끊김 (탭 닫힘, 새로고침, 네트워크) — 녹음은 살려서 "중단됨"으로
            finished = True
            try:
                if await close_recording():
                    db.set_recorded(job_id, title=None, duration_sec=sess.duration_sec,
                                    status="interrupted",
                                    error="녹음 중 연결이 끊겼습니다 (탭 닫힘 등). "
                                          "그때까지의 녹음은 저장됐고, 이어서 변환할 수 있습니다.")
            except Exception:
                traceback.print_exc()


@app.websocket("/ws/progress/{job_id}")
async def ws_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    state = JOBS.get(job_id)
    if state is None:
        await ws.send_json({"type": "error", "message": "알 수 없는 작업입니다."})
        await ws.close()
        return

    q: asyncio.Queue = asyncio.Queue()
    # 이미 지나간 이벤트를 먼저 재생 → 연결이 늦어도 앞부분을 놓치지 않는다
    for msg in list(state.events):
        q.put_nowait(msg)
    state.subscribers.add(q)
    try:
        while True:
            msg = await q.get()
            await ws.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(q)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    rec = db.get(job_id)
    if not rec:
        raise HTTPException(404, "없는 작업입니다.")
    state = JOBS.get(job_id)
    if state and rec["status"] == "running":
        rec["segments"] = state.segments
        rec["percent"] = state.percent
    if rec["status"] == "done":
        rec["drafts"] = []
    else:
        covered = max([rec["progress_sec"]] + [s["end"] for s in rec["segments"]])
        rec["drafts"] = [d for d in rec["drafts"] if d["start"] >= covered - 0.5]
    return rec


@app.get("/api/transcripts")
def list_transcripts(limit: int = 50):
    return db.list_recent(limit)


def _listening_wav_path(job_id: str) -> Path:
    return config.UPLOAD_DIR / f"{job_id}.listen.wav"


@app.delete("/api/transcripts/{job_id}")
def delete_transcript(job_id: str):
    rec = db.get(job_id)
    if rec and rec.get("audio_path"):
        Path(rec["audio_path"]).unlink(missing_ok=True)
    _listening_wav_path(job_id).unlink(missing_ok=True)
    (config.UPLOAD_DIR / f"{job_id}.pcm").unlink(missing_ok=True)
    db.delete(job_id)
    JOBS.pop(job_id, None)
    JOBS.pop(f"cmp-{job_id}", None)
    return {"ok": True}


def _require_done(job_id: str) -> dict:
    rec = db.get(job_id)
    if not rec:
        raise HTTPException(404, "없는 기록입니다.")
    if rec["status"] != "done":
        # 변환이 끝나면 결과가 통째로 덮어써지므로, 그 전에 고친 내용은 사라진다
        raise HTTPException(409, "변환이 끝난 뒤에 수정할 수 있습니다.")
    return rec


@app.put("/api/transcripts/{job_id}/segments")
def save_segments(job_id: str, payload: dict):
    """화면에서 고친 구간 저장. 전문(txt 내보내기용)도 같이 다시 만든다."""
    _require_done(job_id)
    try:
        segs = review.sanitize_segments(payload.get("segments"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.update_segments(job_id, segs, postprocess.to_paragraphs(segs))
    return {"ok": True, "count": len(segs)}


# ── 교정 모드: 모델 비교 분석 ───────────────────────────────────────────
def _run_compare(key: str, job_id: str, models: list[str],
                 loop: asyncio.AbstractEventLoop) -> None:
    """다른 모델로 한 번 더 변환해, 원래 결과와 다르게 들은 구간에 ⚠ 를 붙인다."""
    state = JOBS[key]

    def push(msg: dict) -> None:
        loop.call_soon_threadsafe(state.emit, msg)

    rec = db.get(job_id)
    wav = config.UPLOAD_DIR / f"{job_id}.compare.wav"
    try:
        push({"type": "status", "stage": "전처리"})
        audio.to_wav(Path(rec["audio_path"]), wav, preset="standard")

        words_by_model: dict[str, list[dict]] = {}
        for i, name in enumerate(models):
            step = f"{i + 1}/{len(models)} {name}"
            if name not in transcriber.loaded_names():
                push({"type": "status", "stage": f"{step} 불러오는 중 (최초 30초 정도)"})
                transcriber.get_model(name)
            push({"type": "status", "stage": f"{step} 변환 중"})

            def on_progress(pct: float, i=i) -> None:
                push({"type": "progress",
                      "percent": round((i + pct / 100) / len(models) * 100, 1)})

            words_by_model[name] = transcriber.transcribe_words(
                wav, name, on_progress=on_progress)

        # 분석하는 동안 사용자가 고쳤을 수 있으니 마지막에 최신 저장본을 다시 읽어 합친다
        latest = db.get(job_id)
        segs = latest["segments"]
        alts = {m: review.align_words(segs, w) for m, w in words_by_model.items()}
        flagged = review.apply_comparison(segs, alts, set(config.COMPARE_REFERENCE_ONLY))
        db.update_segments(job_id, segs, postprocess.to_paragraphs(segs))
        push({"type": "done", "flagged": flagged, "models": models,
              "segments": [{"alts": s.get("alts", {}), "flag": s.get("flag")}
                           for s in segs]})
    except Exception as e:
        traceback.print_exc()
        push({"type": "error", "message": str(e)})
    finally:
        wav.unlink(missing_ok=True)


@app.post("/api/transcripts/{job_id}/compare")
async def compare(job_id: str, payload: dict | None = None):
    rec = _require_done(job_id)
    if not rec.get("audio_path") or not Path(rec["audio_path"]).exists():
        raise HTTPException(404, "원본 오디오가 없습니다.")

    key = f"cmp-{job_id}"
    prev = JOBS.get(key)
    if prev and not any(e.get("type") in ("done", "error") for e in prev.events):
        raise HTTPException(409, "이미 분석 중입니다.")

    models = [m for m in config.COMPARE_MODELS if m != rec.get("model")]
    if (payload or {}).get("include_small"):
        models += [m for m in config.COMPARE_REFERENCE_ONLY if m != rec.get("model")]

    JOBS[key] = JobState()
    EXECUTOR.submit(_run_compare, key, job_id, models, asyncio.get_running_loop())
    return {"job_id": key, "models": models}


# ── 교정 모드: 평가용 정답으로 저장 ─────────────────────────────────────
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}


@app.post("/api/transcripts/{job_id}/save-eval")
def save_eval(job_id: str, payload: dict):
    """오디오와 고친 받아쓰기를 같은 이름으로 평가 폴더에 저장 (scripts/benchmark.py 가 짝지어 읽음)."""
    rec = _require_done(job_id)
    src = Path(rec.get("audio_path") or "")
    if not src.exists():
        raise HTTPException(404, "원본 오디오가 없습니다.")

    # 기본 이름은 기록 제목. 제목에도 허용 안 되는 문자가 있을 수 있어 한 번 더 거른다
    name = review.safe_eval_name(
        review.safe_eval_name(str(payload.get("name", "")), fallback=rec["title"]))
    # 정확히 같은 이름만 대상 — "강의1.large-v3.hyp.txt" 같은 벤치마크 결과는 건드리지 않는다
    existing = [p for p in config.EVAL_DIR.glob(f"{name}.*")
                if p.stem == name and p.suffix.lower() in AUDIO_EXTS | {".txt"}]
    if existing and not payload.get("overwrite"):
        raise HTTPException(409, f"같은 이름이 이미 있습니다: {name}")
    for p in existing:            # 확장자가 다른 옛 오디오가 남아 짝이 둘이 되지 않게
        p.unlink(missing_ok=True)

    audio_dst = config.EVAL_DIR / f"{name}{src.suffix.lower()}"
    txt_dst = config.EVAL_DIR / f"{name}.txt"
    shutil.copyfile(src, audio_dst)
    txt_dst.write_text("\n".join(s["text"] for s in rec["segments"] if s.get("text")) + "\n",
                       encoding="utf-8")
    return {"ok": True, "dir": str(config.EVAL_DIR),
            "audio": audio_dst.name, "txt": txt_dst.name,
            "unchecked": sum(1 for s in rec["segments"] if not s.get("checked"))}


@app.get("/api/transcripts/{job_id}/export")
def export(job_id: str, fmt: str = "txt"):
    rec = db.get(job_id)
    if not rec:
        raise HTTPException(404, "없는 작업입니다.")
    segs = rec["segments"]
    if fmt == "srt":
        body, ext = postprocess.to_srt(segs), "srt"
    elif fmt == "md":
        body, ext = postprocess.to_markdown(segs, rec["title"]), "md"
    else:
        body, ext = (rec["text"] or postprocess.to_paragraphs(segs)), "txt"

    quoted = urllib.parse.quote(f"{rec['title']}.{ext}")
    disposition = "attachment; filename*=UTF-8''" + quoted
    return PlainTextResponse(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str, format: str = ""):
    rec = db.get(job_id)
    if not rec or not rec.get("audio_path"):
        raise HTTPException(404, "오디오 없음")
    p = Path(rec["audio_path"])
    if not p.exists():
        raise HTTPException(404, "파일이 삭제되었습니다.")
    if format == "wav":
        # 교정 모드는 구간 단위로 정확히 이동해야 해서 WAV 로 한 번 변환해 캐시해둔다
        listen = _listening_wav_path(job_id)
        if not listen.exists():
            try:
                audio.to_listening_wav(p, listen)
            except audio.AudioError as e:
                raise HTTPException(500, str(e))
        return FileResponse(listen, media_type="audio/wav")
    return FileResponse(p)


@app.get("/api/models")
def models():
    return {
        "available": config.AVAILABLE_MODELS,
        "current": transcriber.model_name,
        "loaded": transcriber.loaded,
        "default": config.DEFAULT_MODEL,
        "device": config.DEVICE,
        "compute_type": config.COMPUTE_TYPE,
        "live_model": config.LIVE_MODEL,
        "live_ready": config.LIVE_MODEL in transcriber.loaded_names(),
    }


@app.post("/api/models/load")
def load_model(payload: dict):
    name = payload.get("name") or config.DEFAULT_MODEL
    if name not in config.AVAILABLE_MODELS:
        raise HTTPException(400, f"알 수 없는 모델: {name}")
    threading.Thread(target=_safe_load, args=(name,), daemon=True).start()
    return {"loading": name}


@app.get("/api/presets")
def presets():
    return {
        "default": config.DEFAULT_PRESET,
        "presets": {k: {"label": v["label"]} for k, v in config.PRESETS.items()},
    }


@app.get("/api/glossary")
def get_glossary():
    return postprocess.load_glossary()


@app.put("/api/glossary")
def put_glossary(payload: dict):
    return postprocess.save_glossary(payload)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "ffmpeg": audio.ffmpeg_available(),
        "model_loaded": transcriber.loaded,
        "model": transcriber.model_name,
        "live_ready": config.LIVE_MODEL in transcriber.loaded_names(),
        "live_model": config.LIVE_MODEL,
    }


# 정적 프론트엔드는 맨 마지막에 마운트 (API 경로를 가리지 않도록)
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
