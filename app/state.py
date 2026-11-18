"""Session state: streaming advance, prefix confirmation and exact undo.

Rules implemented here
----------------------
1. ``advance`` is a loop of single-frame beam steps.  Because
   :meth:`CTCBeamDecoder.advance` is exactly a loop over :meth:`step`, advancing
   to frame N in one call and advancing one frame at a time produce the same
   beam, bit for bit.
2. ``confirm`` fixes a prefix at the current frame.  From then on incompatible
   hypotheses are dropped *inside* the beam step, before pruning, so the beam
   fills with compatible candidates instead of losing slots to hypotheses that
   are about to be discarded.  No score is renormalised: surviving beams keep
   their absolute log probabilities, and the blank / non-blank path mass of the
   confirmed last character is recorded from an exact CTC forward pass.
3. ``undo`` restores the beam that would exist at the **current** frame had the
   confirmation never happened.  Merely lifting the text filter is incorrect:
   the restriction changed which hypotheses survived pruning.  The correct
   state is obtained by replaying the frames consumed since the confirmation
   from the stored pre-confirmation beam, with the earlier prefix restriction.
4. Operations for one session are submitted to a single-worker executor, so
   they are applied in arrival order: advance / undo issued while a replay is
   running are queued and see the post-replay state.  Responses carry a
   monotonically increasing ``revision`` so the UI can discard late replies.
"""

from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .decoder import (
    Beam,
    CTCBeamDecoder,
    DecodeError,
    compute_prefix_alpha,
    finite_or_none,
    prefix_alpha_scores,
)


class SessionError(RuntimeError):
    """The requested operation is not valid for the session's current state."""


MAX_FRAMES = 2000
MAX_EXTRA_TOKENS = 8


@dataclass
class Confirmation:
    """Everything needed to undo one confirmation exactly."""

    seq: int
    prefix_before: str
    prefix_after: str
    frame: int
    # Beam at ``frame`` with only the *earlier* confirmations applied.  Undo
    # replays the frames consumed since from this state.
    beams_before: List[Beam]
    last_char: str
    # Exact forward probabilities of the prefix before / after confirmation,
    # split by blank-ending and label-ending paths.  Never renormalised.
    prefix_logp: float
    prefix_logp_blank: float
    prefix_logp_nonblank: float
    confirmed_logp: float
    confirmed_logp_blank: float
    confirmed_logp_nonblank: float


@dataclass
class Session:
    id: str
    decoder: CTCBeamDecoder
    frames: np.ndarray  # (T, V) natural-log probabilities, -inf at zero
    probs: np.ndarray   # (T, V) original probabilities, for display
    beams: List[Beam] = field(default_factory=list)
    next_frame: int = 0
    prefix: str = ""
    confirmations: List[Confirmation] = field(default_factory=list)
    seq: int = 0
    revision: int = 0
    error: Optional[str] = None
    # Single worker => operations apply in submission order.
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctc"),
        repr=False,
    )
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not self.beams:
            self.beams = self.decoder.initial_beams()

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def vocab(self) -> List[str]:
        return self.decoder.vocab

    @property
    def beam_width(self) -> int:
        return self.decoder.beam_width

    # -- derived views -----------------------------------------------------

    def best_text(self) -> str:
        return self.beams[0].text if self.beams else self.prefix

    def label_ids(self, text: str) -> List[int]:
        try:
            return [self.decoder.vocab.index(ch) for ch in text]
        except ValueError as exc:
            raise SessionError(f"文本包含词表外字符：{exc}") from exc

    def prefix_scores(self, text: str) -> tuple[float, float, float]:
        """Exact CTC forward score of ``text`` at the current frame.

        Returns ``(total, blank-ending, non-blank-ending)`` log probabilities.
        This is the prefix's own probability, not a renormalised beam score.
        """
        alpha = compute_prefix_alpha(self.frames[: self.next_frame], self.label_ids(text))
        return prefix_alpha_scores(alpha)

    def state(self) -> dict:
        """JSON-safe snapshot of the session."""
        return {
            "session_id": self.id,
            "vocab": list(self.vocab),
            "beam_width": self.beam_width,
            "frame_count": self.frame_count,
            "next_frame": self.next_frame,
            "progress": self.next_frame / self.frame_count if self.frame_count else 0.0,
            "prefix": self.prefix,
            "prefix_scores": self.prefix_score_view(),
            "revision": self.revision,
            "seq": self.seq,
            "confirmed_count": len(self.confirmations),
            "best_text": self.best_text(),
            "beams": [beam_to_dict(b) for b in self.beams],
            "confirmations": [confirmation_to_dict(c) for c in self.confirmations],
            "error": self.error,
        }

    def prefix_score_view(self) -> Optional[dict]:
        """Retained blank / non-blank path mass of the confirmed last character.

        Captured at confirmation time from an exact CTC forward pass over the
        frames consumed so far, and then kept as-is: extending the prefix or
        advancing frames never rescales them.  Showing the retained value is
        what lets you compare confirmations on the same absolute scale.
        """
        if not self.prefix:
            return {
                "text": "",
                "last_char": "",
                "logp": 0.0,
                "logp_blank": 0.0,
                "logp_nonblank": None,
            }
        record = self.confirmations[-1]
        # Scores for the last character of the confirmed text, both ending
        # states, retained exactly as computed at confirmation time.
        return {
            "text": self.prefix,
            "last_char": self.prefix[-1],
            "confirmed_at": record.frame,
            "logp": finite_or_none(record.confirmed_logp),
            "logp_blank": finite_or_none(record.confirmed_logp_blank),
            "logp_nonblank": finite_or_none(record.confirmed_logp_nonblank),
            "previous_logp": finite_or_none(record.prefix_logp),
            "previous_logp_blank": finite_or_none(record.prefix_logp_blank),
            "previous_logp_nonblank": finite_or_none(record.prefix_logp_nonblank),
        }

    def beams_at(self, frame: int) -> List[Beam]:
        """Reproduce the beams displayed at boundary ``frame``.

        The replay applies confirmations at the boundaries where they were
        made, so the result matches what the stream actually showed, rather
        than "unrestricted decode plus a filter".
        """
        if frame < 0 or frame > self.next_frame:
            raise SessionError("只能查看已经推进过的帧")
        schedule = {record.frame: record.prefix_after for record in self.confirmations}
        prefix = ""
        beams = self.decoder.initial_beams()
        if 0 in schedule:
            prefix = schedule[0]
            beams = self.decoder.filter_compatible(beams, prefix)
        for index in range(frame):
            beams = self.decoder.step(beams, self.frames[index], prefix=prefix)
            boundary = index + 1
            if boundary in schedule:
                prefix = schedule[boundary]
                beams = self.decoder.filter_compatible(beams, prefix)
        return beams


def beam_to_dict(beam: Beam) -> dict:
    return {
        "text": beam.text,
        "logp": float(beam.logp),
        "logp_blank": finite_or_none(beam.logp_blank),
        "logp_nonblank": finite_or_none(beam.logp_nonblank),
        "prob": float(np.exp(beam.logp)) if np.isfinite(beam.logp) else 0.0,
    }


def confirmation_to_dict(record: Confirmation) -> dict:
    return {
        "seq": record.seq,
        "prefix_before": record.prefix_before,
        "prefix_after": record.prefix_after,
        "frame": record.frame,
        "last_char": record.last_char,
        "confirmed_logp": finite_or_none(record.confirmed_logp),
        "confirmed_logp_blank": finite_or_none(record.confirmed_logp_blank),
        "confirmed_logp_nonblank": finite_or_none(record.confirmed_logp_nonblank),
    }


def _copy_beams(beams: Sequence[Beam]) -> List[Beam]:
    return [copy.copy(b) for b in beams]


# ---------------------------------------------------------------------------
# Input validation


def validate_vocab(vocab) -> List[str]:
    """Normalise a vocabulary: index 0 blank, at most 8 single characters."""
    if not isinstance(vocab, list) or not vocab:
        raise DecodeError("词表必须是非空数组")
    if len(vocab) > MAX_EXTRA_TOKENS + 1:
        raise DecodeError(f"词表最多包含 blank 及 {MAX_EXTRA_TOKENS} 个单字")
    normalised: List[str] = []
    for index, token in enumerate(vocab):
        if not isinstance(token, str):
            raise DecodeError("词表项必须是字符串")
        if index == 0:
            if token.strip().lower() not in ("<blank>", "<b>", "blank", "_"):
                raise DecodeError("第 0 项必须是 blank（可写 <blank>、<b>、blank 或 _）")
            normalised.append("<blank>")
            continue
        if token.strip().lower() in ("<blank>", "<b>", "blank", "_") or token == "":
            raise DecodeError("blank 只能出现在第 0 项，其余项不能为空")
        if any(ch.isspace() for ch in token):
            raise DecodeError(f"词表项不能含空白字符：{token!r}")
        if len(token) != 1:
            raise DecodeError(f"词表第 {index} 项必须是单个字符：{token!r}")
        normalised.append(token)
    if len(normalised) < 2:
        raise DecodeError("词表至少需要一个非 blank 字符")
    if len(set(normalised)) != len(normalised):
        raise DecodeError("词表项不能重复")
    return normalised


def validate_frames(frames, vocab_size: int) -> np.ndarray:
    """Validate frame probabilities and return a (T, V) float64 array."""
    if not isinstance(frames, list) or not frames:
        raise DecodeError("帧概率必须是非空数组")
    if len(frames) > MAX_FRAMES:
        raise DecodeError(f"帧数不能超过 {MAX_FRAMES}")
    array = np.empty((len(frames), vocab_size), dtype=np.float64)
    for t, row in enumerate(frames):
        if not isinstance(row, (list, tuple)) or len(row) != vocab_size:
            raise DecodeError(f"第 {t} 帧应有 {vocab_size} 个概率")
        for k, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DecodeError(f"第 {t} 帧第 {k} 个值不是数字")
            number = float(value)
            if not np.isfinite(number):
                raise DecodeError(f"第 {t} 帧第 {k} 个值不是有限数")
            if number < 0.0 or number > 1.0:
                raise DecodeError(f"第 {t} 帧第 {k} 个值不在 [0,1] 内")
            array[t, k] = number
    return array


# ---------------------------------------------------------------------------
# Session manager


class SessionManager:
    """In-memory store of decoding sessions."""

    def __init__(self, max_sessions: int = 64) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def create(self, vocab, beam_width: int, frames) -> Session:
        normalised = validate_vocab(vocab)
        probs = validate_frames(frames, len(normalised))
        with np.errstate(divide="ignore"):
            logp = np.log(probs)
        decoder = CTCBeamDecoder(normalised, beam_width)
        session = Session(id=uuid.uuid4().hex, decoder=decoder, frames=logp, probs=probs)
        with self._lock:
            while len(self._sessions) >= self.max_sessions:
                oldest = next(iter(self._sessions))
                evicted = self._sessions.pop(oldest)
                evicted.executor.shutdown(wait=False, cancel_futures=True)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("会话不存在或已过期")
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            # Acquire the lock so a request that already fetched this session
            # cannot start mutating it after it has been removed.
            with session.lock:
                session.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, session: Session, func: Callable, *args) -> Future:
        """Queue an operation on the session's ordered worker.

        The session lock is taken inside the worker so that a concurrent read
        (``GET /api/sessions/{id}``) can never observe a half-applied change.
        """

        def run():
            with session.lock:
                return func(session, *args)

        return session.executor.submit(run)


# ---------------------------------------------------------------------------
# Mutating operations.  They run on the session's single worker, in order.


def advance(session: Session, to_frame: int) -> None:
    """Consume frames up to boundary ``to_frame`` without going backwards."""
    if isinstance(to_frame, bool) or not isinstance(to_frame, int):
        raise SessionError("目标帧必须是整数")
    if to_frame < session.next_frame:
        raise SessionError("不能后退到更早的帧；撤销确认请使用 undo")
    if to_frame > session.frame_count:
        raise SessionError("目标帧超出帧数")
    if to_frame == session.next_frame:
        return
    block = session.frames[session.next_frame : to_frame]
    session.beams = session.decoder.advance(session.beams, block, prefix=session.prefix)
    session.next_frame = to_frame
    session.revision += 1
    session.error = None


def confirm(session: Session, prefix: str) -> None:
    """Extend the confirmed prefix to ``prefix`` at the current frame.

    ``prefix`` must extend the current confirmed prefix and must be present in
    at least one live hypothesis.  Scores are not renormalised.
    """
    if not isinstance(prefix, str):
        raise SessionError("确认文本必须是字符串")
    if not prefix:
        raise SessionError("确认文本不能为空")
    if not prefix.startswith(session.prefix):
        raise SessionError("确认文本必须保留已确认前缀")
    if prefix == session.prefix:
        raise SessionError("确认文本没有延长前缀")
    if any(ch not in session.decoder._token_index for ch in prefix):
        raise SessionError("确认文本包含词表外字符")
    compatible = [b for b in session.beams if b.text.startswith(prefix)]
    if not compatible:
        raise SessionError("当前候选中没有以该文本开头的路径")

    before_total, before_blank, before_nonblank = session.prefix_scores(session.prefix)
    after_total, after_blank, after_nonblank = session.prefix_scores(prefix)

    session.seq += 1
    session.confirmations.append(
        Confirmation(
            seq=session.seq,
            prefix_before=session.prefix,
            prefix_after=prefix,
            frame=session.next_frame,
            beams_before=_copy_beams(session.beams),
            last_char=prefix[-1],
            prefix_logp=before_total,
            prefix_logp_blank=before_blank,
            prefix_logp_nonblank=before_nonblank,
            confirmed_logp=after_total,
            confirmed_logp_blank=after_blank,
            confirmed_logp_nonblank=after_nonblank,
        )
    )
    session.prefix = prefix
    session.beams = compatible
    session.revision += 1
    session.error = None


def undo(session: Session) -> None:
    """Undo the latest confirmation at the current frame.

    Restoring the beam is *not* just removing the text restriction: the
    restriction changed which hypotheses survived pruning, so the frames
    consumed since the confirmation are replayed from the stored
    pre-confirmation beam, under the earlier prefix restriction.  The result is
    exactly the beam that would exist now had the confirmation never happened.
    """
    if not session.confirmations:
        raise SessionError("没有可撤销的确认")
    record = session.confirmations[-1]
    if session.next_frame < record.frame:  # pragma: no cover - no rewind path
        raise SessionError("当前帧早于确认帧，无法撤销")

    beams = _copy_beams(record.beams_before)
    if session.next_frame > record.frame:
        block = session.frames[record.frame : session.next_frame]
        beams = session.decoder.advance(beams, block, prefix=record.prefix_before)

    session.confirmations.pop()
    session.prefix = record.prefix_before
    session.beams = beams
    session.revision += 1
    session.error = None
