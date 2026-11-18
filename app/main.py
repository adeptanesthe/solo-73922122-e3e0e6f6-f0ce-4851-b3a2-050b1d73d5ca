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

from .decoder import DecodeError
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
