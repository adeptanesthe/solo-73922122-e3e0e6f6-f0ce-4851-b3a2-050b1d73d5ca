"""Speaker attribution over a window of already-consumed CTC frames.

The annotation workbench knows two probability streams:

* the acoustic model's per-frame CTC distribution (blank + single characters);
* a per-frame diarisation distribution over two speakers 甲/A and 乙/B, with
  ``p_a + p_b == 1`` on every frame.

Given a reference subtitle of at most 16 characters and a continuous window of
1～64 advanced frames, :func:`speaker_attribution` sums **every** CTC label path
that collapses to the reference text (the standard CTC forward/backward graph,
so the sum is exact, not beam-pruned) and weights each path by the diarisation
probability of the speaker assigned to each of its non-blank emissions.

Invariants enforced here
------------------------
* One character may occupy several consecutive non-blank frames (CTC label
  repeats), and every such frame must carry the *same* speaker.  A speaker
  switch is only allowed between two different characters, i.e. across a blank
  or a direct different-label transition.
* Blank frames carry no speaker evidence: they are weighted by the acoustic
  blank probability alone, never by ``p_a`` / ``p_b``.
* Two identical characters must be separated by a blank, exactly as in CTC;
  the speaker may change between them because the blank boundary is a new
  emission.
* The posterior of speaker s for character j is the share of all valid path
  mass (acoustic probability x diarisation weight) whose j-th character is
  spoken by s.  When the two shares are equal the speaker is ``undecided``.

State layout
------------
For a reference of L characters the flat forward/backward vector holds, for
``j = 0..L`` and ``s in {0=甲, 1=乙}``:

* blank state after j labels:  index ``3j``
* label-j speaker-s state:      index ``3j + 1 + s``   (j = 0..L-1)

so the vector has ``3L + 1`` entries and blank / label indices never collide.

All arithmetic is in natural-log space; hard zeros become ``-inf`` and never
produce NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .decoder import BLANK_INDEX, NEG_INF, DecodeError

# Public limits, mirrored by the API validation.
MAX_WINDOW_FRAMES = 64
MAX_REFERENCE_CHARS = 16
# Two speaker probabilities are considered equal within this relative gap.
TIE_REL_TOL = 1e-9
# Mass below this is treated as "no evidence" when building frame details.
EVIDENCE_FLOOR = 1e-12


class AttributionError(DecodeError):
    """The attribution request is invalid or the text is unreachable."""


@dataclass
class CharacterAttribution:
    """Posterior speaker split for one reference character."""

    char: str
    label_id: int
    # Marginal log mass of every valid path in which this character is 甲 / 乙.
    logp_a: float
    logp_b: float
    # Normalised posteriors; both 0.0 when the character is unreachable.
    posterior_a: float
    posterior_b: float
    speaker: str  # "甲" | "乙" | "未定"
    tied: bool
    reachable: bool
    # Absolute frame range (inclusive) that carries non-blank evidence.
    first_frame: int = -1
    last_frame: int = -1


@dataclass
class FrameEvidence:
    """Per-frame diagnostic for the result table."""

    frame: int
    blank_prob: float
    # Per-character non-blank occupancy gamma split by speaker (0.0 if none).
    label_a: List[float] = field(default_factory=list)
    label_b: List[float] = field(default_factory=list)


@dataclass
class AttributionResult:
    reference: str
    start_frame: int
    end_frame: int
    reachable: bool
    # Exact log total of every valid, diarisation-weighted path.
    log_total: float
    # Exact log total of the same paths using acoustic probability only.
    log_acoustic_total: float
    # Number of distinct CTC label paths collapsing to the reference (exact
    # integer; paths differing only in speaker assignment share one label path).
    path_count: int
    characters: List[CharacterAttribution]
    frame_evidence: List[FrameEvidence]


def _logaddexp_scalar(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def validate_speaker_rows(rows, expected: int) -> np.ndarray:
    """Validate the per-frame 甲/乙 probability rows.

    Returns an ``(n, 2)`` float64 array.  Each row must hold two finite numbers
    in ``[0, 1]`` that sum to 1 (a small rounding drift is accepted).
    """
    if not isinstance(rows, list) or not rows:
        raise AttributionError("说话人概率必须是非空数组，每帧一行 [甲, 乙]")
    if len(rows) != expected:
        raise AttributionError(
            f"说话人概率应有 {expected} 行（与所选帧数相同），实际为 {len(rows)} 行"
        )
    array = np.empty((len(rows), 2), dtype=np.float64)
    for t, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise AttributionError(f"第 {t} 行说话人概率必须是长度为 2 的数组 [甲, 乙]")
        for k, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AttributionError(f"第 {t} 行第 {k} 个说话人概率不是数字")
            number = float(value)
            if not math.isfinite(number):
                raise AttributionError(f"第 {t} 行第 {k} 个说话人概率不是有限数")
            if number < 0.0 or number > 1.0:
                raise AttributionError(f"第 {t} 行第 {k} 个说话人概率不在 [0,1] 内")
            array[t, k] = number
        if abs(float(array[t, 0]) + float(array[t, 1]) - 1.0) > 1e-6:
            raise AttributionError(f"第 {t} 行甲、乙概率之和必须为 1")
    return array


def speaker_attribution(
    frame_logp: np.ndarray,
    label_ids: Sequence[int],
    speaker_probs: np.ndarray,
    start_frame: int,
) -> AttributionResult:
    """Exact CTC forward/backward with speaker evidence."""
    frames = np.asarray(frame_logp, dtype=np.float64)
    speakers = np.asarray(speaker_probs, dtype=np.float64)
    labels = np.asarray(label_ids, dtype=np.int64)
    t_count = frames.shape[0]
    char_count = int(labels.size)

    if t_count == 0 or char_count == 0:
        raise AttributionError("帧窗口与参考字幕均不能为空")
    if speakers.shape != (t_count, 2):
        raise AttributionError("说话人概率行数必须等于帧数，且每行两列")
    if t_count > MAX_WINDOW_FRAMES or char_count > MAX_REFERENCE_CHARS:
        raise AttributionError(
            f"最多 {MAX_WINDOW_FRAMES} 帧、{MAX_REFERENCE_CHARS} 个字"
        )
    if np.any(labels < 1) or np.any(labels >= frames.shape[1]):
        raise AttributionError("参考字幕包含词表外字符")

    with np.errstate(divide="ignore"):
        log_sp = np.log(speakers)  # (T, 2), -inf on hard zeros
    log_blank = frames[:, BLANK_INDEX]
    log_label_emission = frames[:, labels]  # (T, L)

    def b_idx(j: int) -> int:
        return 3 * j

    def l_idx(j: int, speaker: int) -> int:
        return 3 * j + 1 + speaker

    state_count = 3 * char_count + 1

    # ---------------------------------------------------------------- forward
    alpha = np.full(state_count, NEG_INF, dtype=np.float64)
    alpha[b_idx(0)] = 0.0

    # Forward alphas at every frame are retained so the backward pass can form
    # forward-backward marginals.
    alpha_history: List[np.ndarray] = []

    for t in range(t_count):
        new = np.full(state_count, NEG_INF, dtype=np.float64)

        # Blank states B_j: blank self-loop, or blank emitted from the two
        # speaker copies of label j-1.  Blank adds no speaker evidence.
        new[b_idx(0)] = alpha[b_idx(0)] + log_blank[t]
        for j in range(1, char_count + 1):
            stay = alpha[b_idx(j)] + log_blank[t]
            from_label = (
                _logaddexp_scalar(alpha[l_idx(j - 1, 0)], alpha[l_idx(j - 1, 1)])
                + log_blank[t]
            )
            new[b_idx(j)] = _logaddexp_scalar(stay, from_label)

        # Label states L_{j,s}.
        for j in range(char_count):
            emission = log_label_emission[t, j]
            if emission == NEG_INF:
                continue
            for s in range(2):
                idx = l_idx(j, s)
                weighted = emission + log_sp[t, s]
                # (a) fresh emission from the blank state before label j:
                # acoustic label probability x this frame's speaker evidence.
                enter = alpha[b_idx(j)] + weighted
                # (b) repeat the same label with the same speaker: the run
                # keeps one uniform speaker, so this frame contributes both
                # the acoustic probability and the same speaker's evidence.
                repeat = alpha[idx] + weighted
                best = _logaddexp_scalar(enter, repeat)
                # (c) direct transition from a different previous label.
                # Either speaker is allowed (the boundary may keep the same
                # speaker or switch); equal labels forbid the branch because
                # repeats must go through a blank.
                if j > 0 and labels[j] != labels[j - 1]:
                    direct = (
                        _logaddexp_scalar(
                            alpha[l_idx(j - 1, 0)], alpha[l_idx(j - 1, 1)]
                        )
                        + weighted
                    )
                    best = _logaddexp_scalar(best, direct)
                new[idx] = best
        alpha = new
        alpha_history.append(alpha.copy())

    log_total = _logaddexp_scalar(
        float(alpha[b_idx(char_count)]),
        _logaddexp_scalar(
            float(alpha[l_idx(char_count - 1, 0)]),
            float(alpha[l_idx(char_count - 1, 1)]),
        ),
    )
    if log_total == NEG_INF:
        return _unreachable_result(labels, start_frame, frames, t_count, char_count)

    acoustic_total = _acoustic_forward(frames, labels, log_blank, log_label_emission)

    # ---------------------------------------------------------------- backward
    beta = np.full(state_count, NEG_INF, dtype=np.float64)
    beta[b_idx(char_count)] = 0.0
    beta[l_idx(char_count - 1, 0)] = 0.0
    beta[l_idx(char_count - 1, 1)] = 0.0

    gamma = np.zeros((t_count, char_count, 2), dtype=np.float64)
    # Entry marginal per (frame, character, speaker): probability mass of the
    # transition that *enters* label state (j, s) at this frame.  A valid path
    # enters every character exactly once, so the entry masses of one character
    # partition the total path mass with no run-length weighting.
    entry_marg = np.full((t_count, char_count, 2), NEG_INF, dtype=np.float64)
    # Prefix mass at the boundary *before* each frame (alpha at frame t-1);
    # entering transitions are weighted from this, not from alpha_t, which
    # already contains frame t's emission.
    alpha_before: List[np.ndarray] = []
    prefix = np.full(state_count, NEG_INF, dtype=np.float64)
    prefix[b_idx(0)] = 0.0
    for t in range(t_count):
        alpha_before.append(prefix.copy())
        prefix = alpha_history[t]

    for t in range(t_count - 1, -1, -1):
        alpha_t = alpha_history[t]
        before = alpha_before[t]
        if t < t_count - 1:
            nxt = t + 1
            new_beta = np.full(state_count, NEG_INF, dtype=np.float64)

            # Blank B_j at t: blank self-loop at t+1, or fresh emission of
            # label j (either speaker) at t+1.
            for j in range(char_count + 1):
                cont = beta[b_idx(j)] + log_blank[nxt]
                if j < char_count:
                    emission = log_label_emission[nxt, j]
                    if emission != NEG_INF:
                        for s in range(2):
                            branch = beta[l_idx(j, s)] + emission + log_sp[nxt, s]
                            cont = _logaddexp_scalar(cont, branch)
                new_beta[b_idx(j)] = cont

            # Label L_{j,s} at t: blank after the label, same-label repeat, or
            # a direct different-label transition (either next speaker).
            for j in range(char_count):
                for s in range(2):
                    idx = l_idx(j, s)
                    via_blank = beta[b_idx(j + 1)] + log_blank[nxt]
                    best = via_blank
                    emission_same = log_label_emission[nxt, j]
                    if emission_same != NEG_INF:
                        # Repeat keeps the same speaker and re-applies this
                        # frame's acoustic + speaker evidence.
                        repeat = beta[idx] + emission_same + log_sp[nxt, s]
                        best = _logaddexp_scalar(best, repeat)
                    if j + 1 < char_count and labels[j + 1] != labels[j]:
                        next_emission = log_label_emission[nxt, j + 1]
                        if next_emission != NEG_INF:
                            # The next character may take either speaker.
                            direct = _logaddexp_scalar(
                                beta[l_idx(j + 1, 0)]
                                + next_emission
                                + log_sp[nxt, 0],
                                beta[l_idx(j + 1, 1)]
                                + next_emission
                                + log_sp[nxt, 1],
                            )
                            best = _logaddexp_scalar(best, direct)
                    new_beta[idx] = best
            beta = new_beta

        # Frame occupancy gamma and first-entry marginals, in joint mass space.
        for j in range(char_count):
            emission = log_label_emission[t, j]
            if emission == NEG_INF:
                continue
            for s in range(2):
                idx = l_idx(j, s)
                # gamma: joint probability of occupying label state (j, s)
                # on this frame.
                if alpha_t[idx] != NEG_INF and beta[idx] != NEG_INF:
                    value = math.exp(alpha_t[idx] + beta[idx] - log_total)
                    if value > EVIDENCE_FLOOR:
                        gamma[t, j, s] = value
                # First-entry mass: only transitions that *start* the run of
                # character j count, i.e. a fresh emission from blank state
                # B_j, or a direct different-label transition (which permits
                # either speaker).  Same-label repeats are deliberately
                # excluded so each path contributes to character j exactly once.
                weighted = emission + log_sp[t, s]
                start_mass = before[b_idx(j)] + weighted
                if j > 0 and labels[j] != labels[j - 1]:
                    start_mass = _logaddexp_scalar(
                        start_mass,
                        _logaddexp_scalar(
                            before[l_idx(j - 1, 0)], before[l_idx(j - 1, 1)]
                        )
                        + weighted,
                    )
                if start_mass != NEG_INF and beta[idx] != NEG_INF:
                    entry_marg[t, j, s] = start_mass + beta[idx]

    # Per-character marginals from the first-entry masses.  Because every valid
    # path starts each character's run on exactly one frame with one speaker,
    # the two speakers' entry masses for a character together equal the total
    # path mass Z.  The run's last frame is read from the occupancy gamma.
    characters: List[CharacterAttribution] = []
    for j in range(char_count):
        logp_a = NEG_INF
        logp_b = NEG_INF
        first = -1
        last = -1
        for t in range(t_count):
            if entry_marg[t, j, 0] != NEG_INF or entry_marg[t, j, 1] != NEG_INF:
                if first < 0:
                    first = t
            if gamma[t, j, 0] > 0.0 or gamma[t, j, 1] > 0.0:
                last = t
            logp_a = _logaddexp_scalar(logp_a, entry_marg[t, j, 0])
            logp_b = _logaddexp_scalar(logp_b, entry_marg[t, j, 1])
        joint = _logaddexp_scalar(logp_a, logp_b)
        if joint == NEG_INF:
            posterior_a = posterior_b = 0.0
            speaker = "未定"
            tied = True
            reachable = False
        else:
            posterior_a = 0.0 if logp_a == NEG_INF else math.exp(logp_a - joint)
            posterior_b = 0.0 if logp_b == NEG_INF else math.exp(logp_b - joint)
            drift = 1.0 - (posterior_a + posterior_b)
            if abs(drift) <= 1e-12:
                if posterior_a >= posterior_b:
                    posterior_a += drift
                else:
                    posterior_b += drift
            tied = _is_tie(posterior_a, posterior_b)
            speaker = "未定" if tied else ("甲" if posterior_a > posterior_b else "乙")
            reachable = True
        characters.append(
            CharacterAttribution(
                char="",
                label_id=int(labels[j]),
                logp_a=logp_a,
                logp_b=logp_b,
                posterior_a=posterior_a,
                posterior_b=posterior_b,
                speaker=speaker,
                tied=tied,
                reachable=reachable,
                first_frame=start_frame + first if first >= 0 else -1,
                last_frame=start_frame + last if last >= 0 else -1,
            )
        )

    blank_probs = np.exp(log_blank)
    evidence = [
        FrameEvidence(
            frame=start_frame + t,
            blank_prob=float(blank_probs[t]),
            label_a=[float(gamma[t, j, 0]) for j in range(char_count)],
            label_b=[float(gamma[t, j, 1]) for j in range(char_count)],
        )
        for t in range(t_count)
    ]

    return AttributionResult(
        reference="",
        start_frame=start_frame,
        end_frame=start_frame + t_count - 1,
        reachable=True,
        log_total=log_total,
        log_acoustic_total=acoustic_total,
        path_count=_count_label_paths(frames, labels),
        characters=characters,
        frame_evidence=evidence,
    )


def _is_tie(posterior_a: float, posterior_b: float) -> bool:
    """Whether the two speaker posteriors count as an exact tie."""
    gap = abs(posterior_a - posterior_b)
    scale = max(posterior_a, posterior_b, 1e-300)
    return gap <= TIE_REL_TOL * scale


def _acoustic_forward(
    frames: np.ndarray,
    labels: np.ndarray,
    log_blank: np.ndarray,
    log_label_emission: np.ndarray,
) -> float:
    """Same CTC graph as the main forward pass, ignoring speaker weights."""
    t_count = frames.shape[0]
    char_count = int(labels.size)
    blank = np.full(char_count + 1, NEG_INF, dtype=np.float64)
    label = np.full(char_count, NEG_INF, dtype=np.float64)
    blank[0] = 0.0
    for t in range(t_count):
        new_blank = np.full_like(blank, NEG_INF)
        new_label = np.full_like(label, NEG_INF)
        new_blank[0] = blank[0] + log_blank[t]
        if char_count:
            new_blank[1:] = np.logaddexp(blank[1:] + log_blank[t], label + log_blank[t])
            fresh = blank[:-1] + log_label_emission[t]
            repeat = label + log_label_emission[t]
            new_label = np.logaddexp(fresh, repeat)
            if char_count > 1:
                different = labels[1:] != labels[:-1]
                direct = np.where(
                    different, label[:-1] + log_label_emission[t, 1:], NEG_INF
                )
                new_label[1:] = np.logaddexp(new_label[1:], direct)
        blank, label = new_blank, new_label
    return _logaddexp_scalar(float(blank[-1]), float(label[-1]))


def _count_label_paths(frames: np.ndarray, labels: np.ndarray) -> int:
    """Count distinct CTC label paths collapsing to the reference.

    A frame is usable for a label / blank when its acoustic probability is
    positive.  The count follows the same forward graph without speaker copies
    or weights; integer arithmetic keeps it exact within the 64-frame limit.
    """
    t_count = frames.shape[0]
    char_count = int(labels.size)
    blank_ok = frames[:, BLANK_INDEX] != NEG_INF
    label_ok = frames[:, labels] != NEG_INF  # (T, L)
    blank_count = [0] * (char_count + 1)
    label_count = [0] * char_count
    blank_count[0] = 1
    for t in range(t_count):
        new_blank = [0] * (char_count + 1)
        new_label = [0] * char_count
        if blank_ok[t]:
            new_blank[0] = blank_count[0]
            for j in range(1, char_count + 1):
                new_blank[j] = blank_count[j] + label_count[j - 1]
        for j in range(char_count):
            if not label_ok[t, j]:
                continue
            total = blank_count[j] + label_count[j]
            if j > 0 and labels[j] != labels[j - 1]:
                total += label_count[j - 1]
            new_label[j] = total
        blank_count, label_count = new_blank, new_label
    return int(blank_count[char_count] + label_count[char_count - 1])


def _unreachable_result(
    labels: np.ndarray,
    start_frame: int,
    frames: np.ndarray,
    t_count: int,
    char_count: int,
) -> AttributionResult:
    characters = [
        CharacterAttribution(
            char="",
            label_id=int(labels[j]),
            logp_a=NEG_INF,
            logp_b=NEG_INF,
            posterior_a=0.0,
            posterior_b=0.0,
            speaker="未定",
            tied=True,
            reachable=False,
        )
        for j in range(char_count)
    ]
    with np.errstate(divide="ignore"):
        blank_probs = np.exp(frames[:, BLANK_INDEX])
    evidence = [
        FrameEvidence(
            frame=start_frame + t,
            blank_prob=float(blank_probs[t]),
            label_a=[0.0] * char_count,
            label_b=[0.0] * char_count,
        )
        for t in range(t_count)
    ]
    return AttributionResult(
        reference="",
        start_frame=start_frame,
        end_frame=start_frame + t_count - 1,
        reachable=False,
        log_total=NEG_INF,
        log_acoustic_total=NEG_INF,
        path_count=0,
        characters=characters,
        frame_evidence=evidence,
    )
