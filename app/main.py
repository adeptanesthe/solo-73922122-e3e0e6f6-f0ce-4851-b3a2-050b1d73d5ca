"""FastAPI application: a streaming CTC decoding workbench.

The server keeps sessions in memory.  CPU-bound work runs on a per-session
single-worker executor so the event loop never blocks.

Concurrency guarantees
----------------------
* All mutations of one session are queued on that session's single worker and
  applied in submission order.  Advancing or undoing while a confirmation's
  replay is still running is therefore safe: the later operation is applied to
  the post-replay state, and a slow earlier result can never overwrite it.
* Every response carries a monotonically increasing ``revision``.  The UI drops
  any response whose revision is older than the state it already shows.
* Validation errors do not mutate the session; the UI keeps its previous
  subtitles and frame progress and merely shows the error.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .attribution import (
    MAX_REFERENCE_CHARS,
    MAX_WINDOW_FRAMES,
    AttributionError,
    speaker_attribution,
    validate_speaker_rows,
)
from .decoder import DecodeError, finite_or_none
from . import state as state_module
from .state import SessionError, SessionManager

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_DIR = BASE_DIR.parent / "samples"

app = FastAPI(title="流式 CTC 解码工作台", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
manager = SessionManager()


# ---------------------------------------------------------------------------
# Request schemas


class CreateBody(BaseModel):
    vocab: list[str]
    beam_width: int = 8
    frames: list[list[float]]


class AdvanceBody(BaseModel):
    session_id: str
    to_frame: int


class PeekBody(BaseModel):
    session_id: str
    frame: int


class ConfirmBody(BaseModel):
    session_id: str
    prefix: str


class UndoBody(BaseModel):
    session_id: str


class AttributionBody(BaseModel):
    session_id: str
    start_frame: int
    end_frame: int
    reference: str
    # One [p_甲, p_乙] row per selected frame; each row sums to 1.
    speaker_probs: list[list[float]]
    analysis_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers


def _session_or_404(session_id: str):
    try:
        return manager.get(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _frame_info(session, frame: int) -> dict:
    index = frame if frame < session.frame_count else session.frame_count - 1
    index = max(index, 0)
    return {
        "frame": frame,
        "vocab": list(session.vocab),
        "probs": [float(v) for v in session.probs[index]],
        "log_probs": [
            None if v == float("-inf") else float(v) for v in session.frames[index]
        ],
    }


async def _run_op(session, func, *args) -> dict:
    """Queue a mutating operation on the session worker and await its result.

    Operations of one session run on its single worker in submission order, so
    an advance or undo issued while an undo replay is still running is applied
    afterwards and can never be overwritten by the earlier result.
    """

    def apply(sess, inner, inner_args):
        inner(sess, *inner_args)
        return _op_state(sess)

    future = manager.submit(session, apply, func, args)
    try:
        return await asyncio.wrap_future(future)
    except SessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _with_meta(session, payload: dict) -> dict:
    payload = dict(payload)
    payload["frame_info"] = _frame_info(session, payload["next_frame"])
    return payload


def _op_state(session) -> dict:
    with session.lock:
        return _with_meta(session, session.state())


# ---------------------------------------------------------------------------
# Pages and health


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Session creation


@app.post("/api/sessions")
async def create_session(body: CreateBody) -> dict:
    loop = asyncio.get_running_loop()

    def build():
        session = manager.create(body.vocab, body.beam_width, body.frames)
        with session.lock:
            return _with_meta(session, session.state())

    try:
        return await loop.run_in_executor(None, build)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/upload")
async def create_session_upload(
    vocab_file: UploadFile = File(..., description="词表 JSON"),
    frames_file: UploadFile = File(..., description="帧概率 JSON"),
    beam_width: int = Form(8),
) -> dict:
    """Create a session from an uploaded vocabulary and frame-probability file.

    The vocabulary may be a JSON array, ``{"vocab": [...]}`` or plain text with
    one token per line.  The frames file may be a JSON array of rows,
    ``{"frames": [...]}`` or ``{"probs": [...]}``.
    """
    try:
        vocab_text = (await vocab_file.read()).decode("utf-8-sig")
        frames_text = (await frames_file.read()).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文件必须是 UTF-8 编码") from exc

    try:
        vocab_raw = json.loads(vocab_text)
    except json.JSONDecodeError:
        vocab_raw = [line.strip() for line in vocab_text.splitlines() if line.strip()]
    try:
        frames_raw = json.loads(frames_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"帧文件 JSON 解析失败：{exc}") from exc

    if isinstance(vocab_raw, dict):
        vocab_raw = vocab_raw.get("vocab")
    if isinstance(frames_raw, dict):
        frames_raw = frames_raw.get("frames", frames_raw.get("probs"))

    def build():
        session = manager.create(vocab_raw, beam_width, frames_raw)
        with session.lock:
            return _with_meta(session, session.state())

    try:
        return await asyncio.get_running_loop().run_in_executor(None, build)
    except DecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Session inspection


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = _session_or_404(session_id)

    def read():
        with session.lock:
            return _with_meta(session, session.state())

    return await asyncio.get_running_loop().run_in_executor(None, read)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    _session_or_404(session_id)
    manager.delete(session_id)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Streaming operations


@app.post("/api/advance")
async def advance_endpoint(body: AdvanceBody) -> dict:
    session = _session_or_404(body.session_id)
    return await _run_op(session, state_module.advance, body.to_frame)


@app.post("/api/peek")
async def peek_endpoint(body: PeekBody) -> dict:
    session = _session_or_404(body.session_id)

    def run(sess):
        if body.frame < 0 or body.frame > sess.next_frame:
            raise SessionError("只能查看已经推进过的帧")
        return {
            "frame": body.frame,
            "revision": sess.revision,
            "prefix": sess.prefix,
            "beams": [state_module.beam_to_dict(b) for b in sess.beams_at(body.frame)],
            "frame_info": _frame_info(sess, body.frame),
        }

    future = manager.submit(session, run)
    try:
        return await asyncio.wrap_future(future)
    except SessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/confirm")
async def confirm_endpoint(body: ConfirmBody) -> dict:
    session = _session_or_404(body.session_id)
    return await _run_op(session, state_module.confirm, body.prefix)


@app.post("/api/undo")
async def undo_endpoint(body: UndoBody) -> dict:
    session = _session_or_404(body.session_id)
    return await _run_op(session, state_module.undo)


# ---------------------------------------------------------------------------
# Speaker attribution
#
# Attribution is a read-only analysis: it never mutates the decoding session,
# so it does not bump ``revision``.  It still runs on the session's single
# worker, which gives two guarantees the UI relies on:
#   * an advance issued while an analysis is running is applied afterwards, so
#     the response's echoed ``revision`` can never describe a newer state than
#     the frames the analysis actually used;
#   * analyses of one session execute in submission order.
# The UI tags every request with its own ``analysis_id`` and marks a result
# stale when its inputs or the session revision no longer match, so a late
# response can never overwrite a newer conclusion.


@app.post("/api/attribution")
async def attribution_endpoint(body: AttributionBody) -> dict:
    session = _session_or_404(body.session_id)

    def run(sess):
        with sess.lock:
            revision = sess.revision
            consumed = sess.next_frame

        # --- frame window --------------------------------------------------
        if isinstance(body.start_frame, bool) or isinstance(body.end_frame, bool):
            raise AttributionError("帧号必须是整数")
        if not isinstance(body.start_frame, int) or not isinstance(body.end_frame, int):
            raise AttributionError("帧号必须是整数")
        if body.start_frame < 0 or body.end_frame < body.start_frame:
            raise AttributionError("帧区间非法：起始帧不能为负，结束帧不能早于起始帧")
        window = body.end_frame - body.start_frame + 1
        if window > MAX_WINDOW_FRAMES:
            raise AttributionError(f"一次最多选择 {MAX_WINDOW_FRAMES} 帧")
        if body.end_frame >= consumed:
            raise SessionError("只能分析已经推进过的帧")

        # --- reference text ------------------------------------------------
        reference = body.reference
        if not isinstance(reference, str) or not reference:
            raise AttributionError("参考字幕不能为空")
        if len(reference) > MAX_REFERENCE_CHARS:
            raise AttributionError(f"参考字幕最多 {MAX_REFERENCE_CHARS} 个字")
        token_index = sess.decoder._token_index
        unknown = [ch for ch in reference if ch not in token_index]
        if unknown:
            raise AttributionError(f"参考字幕包含词表外字符：{''.join(unknown)}")
        label_ids = [token_index[ch] for ch in reference]

        # --- speaker probabilities ----------------------------------------
        speaker = validate_speaker_rows(body.speaker_probs, window)

        result = speaker_attribution(
            sess.frames[body.start_frame : body.end_frame + 1],
            label_ids,
            speaker,
            body.start_frame,
        )
        if not result.reachable:
            # Unreachable text is reported as a 422 so the UI keeps its prior
            # result instead of overwriting it, exactly like invalid inputs.
            raise AttributionError(
                f"参考字幕「{reference}」在所选 {window} 帧中不可达："
                "没有任何 CTC 路径能折叠为该字幕"
            )
        result.reference = reference
        for char, label_id in zip(result.characters, label_ids):
            char.char = sess.vocab[label_id]
        return _attribution_to_dict(result, revision, consumed, body.analysis_id)

    future = manager.submit(session, run)
    try:
        return await asyncio.wrap_future(future)
    except SessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AttributionError as exc:
        # 422: the request was understood but the analysis cannot be produced
        # (invalid probabilities, unreachable text).  The UI keeps its prior
        # result and surfaces the message instead of overwriting anything.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _attribution_to_dict(result, revision: int, consumed: int, analysis_id) -> dict:
    return {
        "analysis_id": analysis_id,
        "revision": revision,
        "consumed_frames": consumed,
        "reachable": result.reachable,
        "reference": result.reference,
        "start_frame": result.start_frame,
        "end_frame": result.end_frame,
        "log_total": finite_or_none(result.log_total),
        "log_acoustic_total": finite_or_none(result.log_acoustic_total),
        "path_count": result.path_count,
        "characters": [
            {
                "char": ch.char,
                "label_id": ch.label_id,
                "speaker": ch.speaker,
                "tied": ch.tied,
                "reachable": ch.reachable,
                "posterior_a": ch.posterior_a,
                "posterior_b": ch.posterior_b,
                "logp_a": finite_or_none(ch.logp_a),
                "logp_b": finite_or_none(ch.logp_b),
                "first_frame": ch.first_frame,
                "last_frame": ch.last_frame,
            }
            for ch in result.characters
        ],
        "frame_evidence": [
            {
                "frame": ev.frame,
                "blank_prob": ev.blank_prob,
                "label_a": ev.label_a,
                "label_b": ev.label_b,
            }
            for ev in result.frame_evidence
        ],
    }


# ---------------------------------------------------------------------------
# Bundled samples


@app.get("/api/samples")
async def list_samples() -> dict:
    items = []
    if SAMPLE_DIR.is_dir():
        for path in sorted(SAMPLE_DIR.glob("*.json")):
            items.append({"name": path.name})
    return {"samples": items}


@app.get("/api/samples/{name}")
async def get_sample(name: str) -> JSONResponse:
    path = (SAMPLE_DIR / name).resolve()
    if path.parent != SAMPLE_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="样例不存在")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
