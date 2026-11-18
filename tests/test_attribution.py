"""Unit tests for the speaker-attribution forward/backward engine.

The engine is checked against an exhaustive enumeration of every CTC label
path (windows are tiny), so every posterior is verified from first principles
rather than against another implementation of the same DP.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from app.attribution import (
    MAX_REFERENCE_CHARS,
    MAX_WINDOW_FRAMES,
    AttributionError,
    speaker_attribution,
    validate_speaker_rows,
)
from app.decoder import NEG_INF

VOCAB = ["<blank>", "a", "b", "c"]


def logp(rows):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(rows, dtype=np.float64))


def brute_force(frames, speakers, text):
    """Enumerate every CTC path and weight it with speaker evidence.

    Returns (log_total, per-character logp_a, per-character logp_b, path_count).

    The joint model used by the engine: one speaker is chosen for each
    character and held constant across its non-blank run; each non-blank frame
    contributes that speaker's diarisation probability and blank frames
    contribute nothing.  A CTC label path therefore appears once per speaker
    assignment of its characters.
    """
    v_size = len(frames[0])
    t_count = len(frames)
    total_a = [NEG_INF] * len(text)
    total_b = [NEG_INF] * len(text)
    grand = NEG_INF
    path_count = 0

    for path in itertools.product(range(v_size), repeat=t_count):
        # CTC collapse.
        collapsed = []
        previous = None
        for k in path:
            if k != 0 and k != previous:
                collapsed.append(k)
            previous = k
        if "".join(VOCAB[k] for k in collapsed) != text:
            continue
        # Frames belonging to each character's non-blank run.
        char_frames = [[] for _ in text]
        pos = -1
        previous = None
        for t, k in enumerate(path):
            if k == 0:
                previous = k
                continue
            if k != previous:
                pos += 1
            char_frames[pos].append(t)
            previous = k
        if pos != len(text) - 1:  # pragma: no cover - collapse already checked
            continue

        acoustic = 0.0
        for t, k in enumerate(path):
            p = frames[t][k]
            if p <= 0.0:
                acoustic = NEG_INF
                break
            acoustic += math.log(p)
        if acoustic == NEG_INF:
            continue
        path_count += 1

        # One uniform speaker per character.  The diarisation row is a per-frame
        # posterior over speakers, so every non-blank frame of the run
        # contributes that speaker's probability; blank frames contribute none.
        run_log = [[0.0, 0.0] for _ in text]
        feasible = [[True, True] for _ in text]
        for j, frame_list in enumerate(char_frames):
            for s in (0, 1):
                weight = 0.0
                for t in frame_list:
                    p = speakers[t][s]
                    if p <= 0.0:
                        feasible[j][s] = False
                        break
                    weight += math.log(p)
                run_log[j][s] = weight

        for assignment in itertools.product((0, 1), repeat=len(text)):
            if any(not feasible[j][assignment[j]] for j in range(len(text))):
                continue
            score = acoustic
            for j, speaker in enumerate(assignment):
                score += run_log[j][speaker]
            grand = _lse(grand, score)
            for j, speaker in enumerate(assignment):
                if speaker == 0:
                    total_a[j] = _lse(total_a[j], score)
                else:
                    total_b[j] = _lse(total_b[j], score)
    return grand, total_a, total_b, path_count


def _lse(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def run(frames, speakers, text, start=0):
    ids = [VOCAB.index(ch) for ch in text]
    return speaker_attribution(logp(frames), ids, np.asarray(speakers, dtype=np.float64), start)


def close(a, b, tol=1e-9):
    if a == NEG_INF or b == NEG_INF:
        return a == b
    return abs(a - b) <= tol


# --------------------------------------------------------------------- basics


def test_single_frame_single_character_posteriors():
    frames = [[0.4, 0.6, 0.0, 0.0]]
    speakers = [[0.7, 0.3]]
    result = run(frames, speakers, "a")
    assert result.reachable
    ch = result.characters[0]
    assert ch.posterior_a == pytest.approx(0.7)
    assert ch.posterior_b == pytest.approx(0.3)
    assert ch.speaker == "甲"
    # Only one label path: a at frame 0.
    assert result.path_count == 1
    # Joint total = sum over the two speaker states: 0.6*0.7 + 0.6*0.3 = 0.78.
    assert result.log_total == pytest.approx(math.log(0.6 * 0.7 + 0.6 * 0.3))
    # The acoustic-only marginal drops every speaker weight (0.7+0.3 == 1).
    assert result.log_acoustic_total == pytest.approx(math.log(0.6))
    # Frame occupancy in joint path mass (normalised by the joint total 0.6):
    # gamma_甲 = 0.42 / 0.6 = 0.7, gamma_乙 = 0.18 / 0.6 = 0.3.
    ev = result.frame_evidence[0]
    assert ev.blank_prob == pytest.approx(0.4)
    assert ev.label_a[0] == pytest.approx(0.7)
    assert ev.label_b[0] == pytest.approx(0.3)


def test_tie_is_reported_as_undecided():
    frames = [[0.2, 0.8, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
    speakers = [[0.5, 0.5], [0.5, 0.5]]
    result = run(frames, speakers, "a")
    ch = result.characters[0]
    assert ch.tied
    assert ch.speaker == "未定"
    assert ch.posterior_a == pytest.approx(0.5)
    assert ch.posterior_b == pytest.approx(0.5)


def test_blank_frames_carry_no_speaker_evidence():
    # A frame that is blank on *every surviving* path contributes no speaker
    # evidence.  Here frame 1 still has a non-blank "a" path, so its speaker
    # row does matter; instead use a frame forced to blank (blank prob 1).
    frames = [[0.1, 0.9, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    speakers_a = [[0.9, 0.1], [0.99, 0.01]]
    speakers_b = [[0.9, 0.1], [0.01, 0.99]]
    r1 = run(frames, speakers_a, "a")
    r2 = run(frames, speakers_b, "a")
    # Frame 1 is blank with probability 1: no path emits a label there, so its
    # speaker row cannot move the posterior.
    assert r1.characters[0].posterior_a == r2.characters[0].posterior_a
    assert r1.log_total == pytest.approx(r2.log_total)
    assert r1.characters[0].posterior_a == pytest.approx(0.9)
    # And its frame evidence shows no label occupancy on the blank frame.
    assert r1.frame_evidence[1].label_a == [0.0]
    assert r1.frame_evidence[1].label_b == [0.0]


def test_unreachable_text_returns_unreachable_result():
    # "a" is impossible: its probability is zero on every frame.
    frames = [[0.5, 0.0, 0.5, 0.0], [0.6, 0.0, 0.4, 0.0]]
    speakers = [[0.5, 0.5], [0.5, 0.5]]
    result = run(frames, speakers, "a")
    assert not result.reachable
    assert result.path_count == 0
    assert result.log_total == NEG_INF
    assert all(not ch.reachable and ch.speaker == "未定" for ch in result.characters)


def test_repeated_character_requires_blank_and_may_switch_speaker():
    # a, blank, a -> "aa"; the two a's may take different speakers.
    frames = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    speakers = [[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]]
    result = run(frames, speakers, "aa")
    assert result.reachable
    assert result.path_count == 1
    first, second = result.characters
    assert first.posterior_a == pytest.approx(0.8)
    assert second.posterior_a == pytest.approx(0.3)
    assert first.speaker == "甲"
    assert second.speaker == "乙"
    # Blank frame shows no label occupancy.
    assert result.frame_evidence[1].label_a == [0.0, 0.0]


def test_same_label_without_blank_keeps_one_speaker():
    # Three a-frames collapse to one character; the speaker must be uniform.
    frames = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    speakers = [[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]
    result = run(frames, speakers, "a")
    assert result.reachable
    ch = result.characters[0]
    # Uniform speaker over the whole run: 甲 weight 0.6*0.2*0.9,
    # 乙 weight 0.4*0.8*0.1 (acoustic mass is 1 on every frame).
    pa = 0.6 * 0.2 * 0.9
    pb = 0.4 * 0.8 * 0.1
    assert ch.posterior_a == pytest.approx(pa / (pa + pb))
    assert ch.posterior_b == pytest.approx(pb / (pa + pb))
    assert ch.speaker == "甲"
    # All three frames carry label occupancy for the same character.
    assert all(ev.label_a[0] > 0 or ev.label_b[0] > 0 for ev in result.frame_evidence)


def test_speaker_switch_between_different_labels_direct_transition():
    # "ab" with no blank: a-frame then b-frame, speakers may differ.
    frames = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    speakers = [[0.7, 0.3], [0.2, 0.8]]
    result = run(frames, speakers, "ab")
    assert result.reachable
    assert result.path_count == 1
    a, b = result.characters
    assert a.posterior_a == pytest.approx(0.7)
    assert b.posterior_b == pytest.approx(0.8)
    assert a.speaker == "甲" and b.speaker == "乙"


def test_direct_transition_forbidden_for_same_label():
    # Two a frames with zero blank cannot form "aa".
    frames = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    speakers = [[0.5, 0.5], [0.5, 0.5]]
    assert not run(frames, speakers, "aa").reachable
    # But "a" is reachable with a single (uniform-speaker) run.
    assert run(frames, speakers, "a").reachable


# ------------------------------------------------------------- brute force


@pytest.mark.parametrize(
    "frames,speakers,text",
    [
        (
            [[0.6, 0.4, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.5, 0.0, 0.5, 0.0]],
            [[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]],
            "a",
        ),
        (
            [[0.6, 0.4, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.5, 0.0, 0.5, 0.0]],
            [[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]],
            "b",
        ),
        (
            [[0.3, 0.5, 0.2, 0.0], [0.2, 0.3, 0.5, 0.0], [0.7, 0.2, 0.1, 0.0]],
            [[0.6, 0.4], [0.55, 0.45], [0.2, 0.8]],
            "ab",
        ),
        (
            [[0.1, 0.9, 0.0, 0.0], [0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]],
            [[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]],
            "aa",
        ),
        (
            [[0.2, 0.7, 0.1, 0.0], [0.2, 0.7, 0.1, 0.0], [0.6, 0.2, 0.2, 0.0]],
            [[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]],
            "a",
        ),
        (
            [[0.25, 0.25, 0.25, 0.25]] * 4,
            [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.35, 0.65]],
            "abc",
        ),
    ],
)
def test_posteriors_match_brute_force(frames, speakers, text):
    result = run(frames, speakers, text)
    grand, brute_a, brute_b, path_count = brute_force(frames, speakers, text)
    if grand == NEG_INF:
        assert not result.reachable
        assert result.path_count == 0
        return
    assert result.reachable
    assert close(result.log_total, grand), (result.log_total, grand)
    assert result.path_count == path_count
    for j, ch in enumerate(result.characters):
        joint = _lse(brute_a[j], brute_b[j])
        expected_a = 0.0 if brute_a[j] == NEG_INF else math.exp(brute_a[j] - joint)
        expected_b = 0.0 if brute_b[j] == NEG_INF else math.exp(brute_b[j] - joint)
        assert ch.posterior_a == pytest.approx(expected_a, abs=1e-12), (text, j)
        assert ch.posterior_b == pytest.approx(expected_b, abs=1e-12), (text, j)
        assert close(ch.logp_a, brute_a[j])
        assert close(ch.logp_b, brute_b[j])


def test_entry_masses_partition_total_path_mass():
    # Every valid path enters each character once: per-character entries sum to Z.
    frames = [
        [0.3, 0.5, 0.2, 0.0],
        [0.2, 0.3, 0.5, 0.0],
        [0.7, 0.2, 0.1, 0.0],
        [0.4, 0.4, 0.2, 0.0],
    ]
    speakers = [[0.6, 0.4], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]
    result = run(frames, speakers, "ab")
    for ch in result.characters:
        joint = _lse(ch.logp_a, ch.logp_b)
        assert close(joint, result.log_total)
        assert ch.posterior_a + ch.posterior_b == pytest.approx(1.0)


def test_hard_zero_speaker_eliminates_that_speaker():
    frames = [[0.1, 0.9, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
    speakers = [[1.0, 0.0], [1.0, 0.0]]  # 乙 impossible on both frames
    result = run(frames, speakers, "a")
    ch = result.characters[0]
    assert ch.posterior_a == pytest.approx(1.0)
    assert ch.posterior_b == 0.0
    assert ch.logp_b == NEG_INF
    assert ch.speaker == "甲"


def test_zero_blank_frame_does_not_break_graph():
    # Frame 1 has blank probability 0: blank paths must simply skip it.
    frames = [
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.5, 0.3, 0.2, 0.0],
    ]
    speakers = [[0.7, 0.3], [0.7, 0.3], [0.2, 0.8]]
    result = run(frames, speakers, "a")
    assert result.reachable
    grand, brute_a, brute_b, _ = brute_force(frames, speakers, "a")
    assert close(result.log_total, grand)


def test_frame_numbering_uses_absolute_start():
    frames = [[0.4, 0.6, 0.0, 0.0]]
    result = run(frames, [[0.5, 0.5]], "a", start=10)
    assert result.start_frame == 10
    assert result.end_frame == 10
    assert result.frame_evidence[0].frame == 10
    assert result.characters[0].first_frame == 10


# ---------------------------------------------------------------- validation


def test_speaker_row_validation():
    rows = validate_speaker_rows([[0.5, 0.5], [0.2, 0.8]], 2)
    assert rows.shape == (2, 2)
    for bad in (
        [],
        [[0.5, 0.5]],
        [[0.5], [0.5]],
        [[0.5, 0.6], [0.5, 0.5]],
        [[-0.1, 1.1], [0.5, 0.5]],
        [[float("nan"), 1.0]],
        [[True, False]],
        [["0.5", "0.5"]],
    ):
        with pytest.raises(AttributionError):
            validate_speaker_rows(bad, 2 if isinstance(bad, list) and bad else 1)


def test_limits_are_enforced():
    big_frames = np.full((MAX_WINDOW_FRAMES + 1, 4), 0.25)
    speakers = np.full((MAX_WINDOW_FRAMES + 1, 2), 0.5)
    with pytest.raises(AttributionError):
        speaker_attribution(np.log(big_frames), [1], speakers, 0)
    frames = np.full((4, 4), 0.25)
    with pytest.raises(AttributionError):
        speaker_attribution(
            np.log(frames), [1] * (MAX_REFERENCE_CHARS + 1), np.full((4, 2), 0.5), 0
        )


def test_label_index_out_of_vocab_rejected():
    frames = np.full((2, 4), 0.25)
    with pytest.raises(AttributionError):
        speaker_attribution(np.log(frames), [9], np.full((2, 2), 0.5), 0)
