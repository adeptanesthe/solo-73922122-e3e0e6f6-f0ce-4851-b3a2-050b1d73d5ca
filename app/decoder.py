"""Numerical core: CTC prefix beam search and exact prefix scoring.

All scores are natural-log probabilities.  A frame probability vector has the
CTC blank at index 0.  The decoder is intentionally small and dependency-free
apart from NumPy so that its invariants are easy to test:

* paths that collapse to the same text are merged with log-sum-exp **before**
  pruning, so a text can never occupy two beam slots;
* equal scores are ordered by Unicode code point order of the text;
* :meth:`CTCBeamDecoder.advance` is only a loop over :meth:`step`, therefore
  advancing any chunk is identical to advancing one frame at a time;
* every beam retains the probability mass of paths ending in blank and in a
  non-blank label, which the UI exposes for the last character;
* ``compute_prefix_alpha`` computes the exact CTC forward probability of a
  fixed target text, split by whether it ends in blank, without renormalising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

BLANK_INDEX = 0
NEG_INF = float("-inf")


class DecodeError(ValueError):
    """Invalid vocabulary, frame data or decoder parameter."""


def is_compatible(text: str, prefix: str) -> bool:
    """Whether a hypothesis is compatible with the confirmed prefix.

    Once a prefix is confirmed it is treated as settled output: only
    hypotheses that already contain it survive.  This keeps every displayed
    beam a valid continuation of the confirmed text and makes the restriction
    easy to reason about when replaying frames for undo.
    """
    if not prefix:
        return True
    return text.startswith(prefix)


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a)+exp(b)) with exact -inf handling."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp over a small sequence of floats."""
    vals = [v for v in values if v != NEG_INF]
    if not vals:
        return NEG_INF
    top = max(vals)
    if top == NEG_INF:
        return NEG_INF
    if len(vals) == 1:
        return top
    total = 0.0
    for value in vals:
        total += math.exp(value - top)
    return top + math.log(total)


@dataclass
class Beam:
    """One distinct decoded text and its path probability bookkeeping."""

    text: str
    logp: float
    logp_blank: float
    logp_nonblank: float

    def clone(self) -> "Beam":
        return Beam(self.text, self.logp, self.logp_blank, self.logp_nonblank)


class CTCBeamDecoder:
    """CTC prefix beam search over a tiny vocabulary (blank + single chars)."""

    def __init__(self, vocab: Sequence[str], beam_width: int) -> None:
        if not 1 <= int(beam_width) <= 16:
            raise DecodeError("束宽必须在 1～16 之间")
        if not vocab:
            raise DecodeError("词表不能为空")
        if len(vocab) > 9:
            raise DecodeError("词表最多为 blank 加 8 个单字")
        self.vocab = list(vocab)
        self.beam_width = int(beam_width)
        self.vocab_size = len(self.vocab)
        self._token_index: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        if len(self._token_index) != self.vocab_size:
            raise DecodeError("词表项不能重复")

    def initial_beams(self) -> list[Beam]:
        """State before the first frame: empty text, conceptually ended in blank."""
        return [Beam(text="", logp=0.0, logp_blank=0.0, logp_nonblank=NEG_INF)]

    def step(
        self, beams: Sequence[Beam], frame_logp: np.ndarray, prefix: str = ""
    ) -> list[Beam]:
        """Consume exactly one frame and prune to the beam width."""
        frame = np.asarray(frame_logp, dtype=np.float64)
        if frame.ndim != 1 or frame.shape[0] != self.vocab_size:
            raise DecodeError(f"每帧应有 {self.vocab_size} 个概率")
        if np.isnan(frame).any() or np.isposinf(frame).any():
            raise DecodeError("帧概率包含 NaN 或 +inf")

        vocab = self.vocab
        v_size = self.vocab_size
        blank_logp = float(frame[0])
        token_index = self._token_index

        # text -> [blank-ending, non-blank-ending] log mass.  Merging here is
        # the CTC rule "paths collapsing to the same text are summed"; it
        # happens before pruning so a text never occupies two beam slots.
        acc: Dict[str, list[float]] = {}

        def add(text: str, blank: float, nonblank: float) -> None:
            slot = acc.get(text)
            if slot is None:
                acc[text] = [blank, nonblank]
            else:
                slot[0] = _logaddexp(slot[0], blank)
                slot[1] = _logaddexp(slot[1], nonblank)

        for beam in beams:
            p_blank = beam.logp_blank
            p_nonblank = beam.logp_nonblank
            if p_blank == NEG_INF and p_nonblank == NEG_INF:
                continue
            total = _logaddexp(p_blank, p_nonblank)
            text = beam.text

            # 1. blank: decoded text unchanged.
            if total != NEG_INF:
                add(text, total + blank_logp, NEG_INF)

            # 2. a label after blank (or before the first frame) always
            #    appends, including the same character again.
            if p_blank != NEG_INF:
                for k in range(1, v_size):
                    add(text + vocab[k], NEG_INF, p_blank + float(frame[k]))

            # 3. a label directly after a non-blank: the same label collapses
            #    into the current text, any other label appends.
            if p_nonblank != NEG_INF:
                last_idx = token_index.get(text[-1]) if text else None
                for k in range(1, v_size):
                    if k == last_idx:
                        add(text, NEG_INF, p_nonblank + float(frame[k]))
                    else:
                        add(text + vocab[k], NEG_INF, p_nonblank + float(frame[k]))

        # A confirmed prefix is applied before pruning, so a restricted search
        # fills the beam with compatible candidates instead of losing slots to
        # hypotheses that are about to be discarded.
        ranked = [
            Beam(text=text, logp=_logaddexp(pair[0], pair[1]), logp_blank=pair[0], logp_nonblank=pair[1])
            for text, pair in acc.items()
            if (pair[0] != NEG_INF or pair[1] != NEG_INF) and is_compatible(text, prefix)
        ]
        # Highest score first; exact ties by Unicode code point order.
        ranked.sort(key=lambda beam: (-beam.logp, beam.text))
        return ranked[: self.beam_width]

    def advance(
        self, beams: Sequence[Beam], frames: np.ndarray, prefix: str = ""
    ) -> list[Beam]:
        """Consume a block of frames; identical to calling :meth:`step` per frame."""
        current = list(beams)
        for row in np.asarray(frames, dtype=np.float64):
            current = self.step(current, row, prefix=prefix)
        return current

    @staticmethod
    def filter_compatible(beams: Sequence[Beam], prefix: str) -> list[Beam]:
        return [beam for beam in beams if is_compatible(beam.text, prefix)]


# ---------------------------------------------------------------------------
# Exact CTC forward probability for a fixed target text
#
# State layout for a target with L labels:
#   2j     : blank state after j emitted labels (j = 0..L)
#   2j - 1 : label j state (j = 1..L)
#
# This is the usual CTC forward graph.  It gives the *exact* probability that
# the whole path decodes to the target, including paths whose final symbol is
# blank.  The split between even (blank-ending) and odd (label-ending) states is
# the blank / non-blank score needed to continue decoding without renormalising.


def prefix_initial_alpha(label_count: int) -> np.ndarray:
    alpha = np.full(2 * label_count + 1, NEG_INF, dtype=np.float64)
    alpha[0] = 0.0
    return alpha


def prefix_step_alpha(
    alpha: np.ndarray, frame_logp: np.ndarray, label_ids: Sequence[int]
) -> np.ndarray:
    """Advance the exact CTC forward state by one frame."""
    labels = np.asarray(label_ids, dtype=np.int64)
    label_count = labels.size
    if alpha.size != 2 * label_count + 1:
        raise DecodeError("prefix 前向状态长度不匹配")
    frame = np.asarray(frame_logp, dtype=np.float64)
    if frame.ndim != 1 or np.any(labels < 1) or np.any(labels >= frame.size):
        raise DecodeError("prefix 标签索引越界")

    even = alpha[0::2]  # blank-after-j states, length L+1
    odd = alpha[1::2]   # label-j states, length L
    frame_blank = frame[BLANK_INDEX]
    frame_labels = frame[labels]

    new_even = np.full(label_count + 1, NEG_INF, dtype=np.float64)
    new_even[0] = even[0] + frame_blank
    if label_count:
        # Stay after j labels in blank, or emit blank from label j.
        new_even[1:] = np.logaddexp(even[1:] + frame_blank, odd + frame_blank)

    new_odd = np.full(label_count, NEG_INF, dtype=np.float64)
    if label_count:
        # Emit label j from the blank state before it, or repeat label j.
        new_odd = np.logaddexp(even[:-1] + frame_labels, odd + frame_labels)
        if label_count > 1:
            different = labels[1:] != labels[:-1]
            direct = np.where(different, odd[:-1] + frame_labels[1:], NEG_INF)
            new_odd[1:] = np.logaddexp(new_odd[1:], direct)

    result = np.empty_like(alpha)
    result[0::2] = new_even
    result[1::2] = new_odd
    return result


def prefix_alpha_scores(alpha: np.ndarray) -> tuple[float, float, float]:
    """Return the exact (total, blank-ending, non-blank-ending) log probability.

    Only the final two forward states count: the blank state after all labels
    (``alpha[-1]``, an even state) and the last-label state (``alpha[-2]``, an
    odd state).  Intermediate blank states represent shorter texts and must not
    be added.
    """
    if alpha.size <= 1:
        return float(alpha[0]) if alpha.size else NEG_INF, (
            float(alpha[0]) if alpha.size else NEG_INF
        ), NEG_INF
    blank = float(alpha[-1])
    nonblank = float(alpha[-2])
    return logsumexp((blank, nonblank)), blank, nonblank


def compute_prefix_alpha(frames_logp: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    alpha = prefix_initial_alpha(len(label_ids))
    for row in np.asarray(frames_logp, dtype=np.float64):
        alpha = prefix_step_alpha(alpha, row, label_ids)
    return alpha


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None
