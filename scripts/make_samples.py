#!/usr/bin/env python3
"""Generate the bundled sample files.

Run from the repository root::

    python3 scripts/make_samples.py

The samples are deliberately tiny and fully offline.  They exercise the three
interesting CTC situations the workbench cares about:

``sample_repeat.json``
    a long run of the same character separated by blanks, plus consecutive
    identical emissions that must collapse into one character;
``sample_blank_heavy.json``
    many frames that are almost pure blank, with a couple of short words;
``sample_sparse_zero.json``
    hard zeros (``log 0 = -inf``) for some labels, including a frame whose blank
    probability is exactly 0, so the UI can show ``-∞`` without NaNs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def normalise(rows):
    """Round to 4 decimals and make each frame sum to 1 where possible."""
    out = []
    for row in rows:
        total = sum(row)
        if total > 0:
            row = [round(v / total, 4) for v in row]
            # fix rounding drift on the largest entry
            drift = round(1.0 - sum(row), 4)
            if drift:
                idx = max(range(len(row)), key=lambda i: row[i])
                row[idx] = round(row[idx] + drift, 4)
        out.append(row)
    return out


def repeat_sample():
    vocab = ["<blank>", "a", "b", "c"]
    # index: <blank>, a, b, c
    frames = [
        # "a" sustained across several frames: only the first should survive
        [0.05, 0.90, 0.03, 0.02],
        [0.05, 0.90, 0.03, 0.02],
        [0.05, 0.90, 0.03, 0.02],
        [0.80, 0.10, 0.05, 0.05],
        # "aa" needs an explicit blank between the two a's
        [0.60, 0.30, 0.05, 0.05],
        [0.05, 0.90, 0.03, 0.02],
        [0.05, 0.90, 0.03, 0.02],
        [0.85, 0.05, 0.05, 0.05],
        # "b"
        [0.05, 0.05, 0.88, 0.02],
        [0.05, 0.05, 0.88, 0.02],
        [0.75, 0.08, 0.12, 0.05],
        # "c"
        [0.10, 0.05, 0.05, 0.80],
        [0.80, 0.05, 0.05, 0.10],
        # trailing blanks
        [0.95, 0.02, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
    ]
    return {
        "name": "连续同字 + blank 分隔",
        "description": "同一字符连续多帧只计一次；真正的重复字符需要中间 blank。",
        "vocab": vocab,
        "frames": normalise(frames),
    }


def blank_heavy_sample():
    vocab = ["<blank>", "啊", "阿", "波", "播", "佛"]
    rows = []
    for _ in range(8):
        rows.append([0.97, 0.006, 0.008, 0.006, 0.005, 0.005])
    rows += [
        [0.30, 0.46, 0.12, 0.05, 0.04, 0.03],
        [0.25, 0.55, 0.10, 0.05, 0.03, 0.02],
        [0.88, 0.05, 0.03, 0.02, 0.01, 0.01],
        [0.20, 0.10, 0.08, 0.42, 0.15, 0.05],
        [0.30, 0.05, 0.05, 0.40, 0.15, 0.05],
        [0.90, 0.03, 0.03, 0.02, 0.01, 0.01],
        [0.92, 0.02, 0.02, 0.02, 0.01, 0.01],
    ]
    for _ in range(6):
        rows.append([0.98, 0.005, 0.005, 0.004, 0.003, 0.003])
    return {
        "name": "空白居多",
        "description": "大量纯 blank 帧，中间夹两个短音，便于观察空白对分数的影响。",
        "vocab": vocab,
        "frames": normalise(rows),
    }


def sparse_zero_sample():
    vocab = ["<blank>", "x", "y", "z"]
    rows = [
        [0.55, 0.40, 0.05, 0.00],   # z has probability exactly 0 -> log p = -inf
        [0.10, 0.85, 0.05, 0.00],
        [0.00, 0.45, 0.55, 0.00],   # blank and z are impossible here
        [0.60, 0.10, 0.30, 0.00],
        [0.90, 0.03, 0.07, 0.00],
        [0.20, 0.05, 0.75, 0.00],
        [0.95, 0.02, 0.03, 0.00],
    ]
    return {
        "name": "含零概率帧",
        "description": "包含 log(0) = -∞（某些字符或 blank 概率为 0），用于检查界面不出现 NaN 且分数正确。",
        "vocab": vocab,
        "frames": rows,
    }


def attribution_sample():
    vocab = ["<blank>", "a", "b", "c"]
    # Two speakers: 甲 dominates the opening "a" region, 乙 the "b"/"c" region.
    # Blank-heavy frames carry neutral (0.5, 0.5) speaker rows.
    frames = [
        [0.08, 0.86, 0.04, 0.02],
        [0.08, 0.86, 0.04, 0.02],
        [0.82, 0.08, 0.06, 0.04],
        [0.10, 0.05, 0.80, 0.05],
        [0.10, 0.05, 0.80, 0.05],
        [0.78, 0.08, 0.10, 0.04],
        [0.12, 0.06, 0.06, 0.76],
        [0.85, 0.05, 0.05, 0.05],
        [0.92, 0.03, 0.03, 0.02],
    ]
    speakers = [
        [0.92, 0.08],
        [0.88, 0.12],
        [0.50, 0.50],
        [0.15, 0.85],
        [0.10, 0.90],
        [0.50, 0.50],
        [0.20, 0.80],
        [0.50, 0.50],
        [0.50, 0.50],
    ]
    return {
        "name": "双人归属示例",
        "description": "在 CTC 帧概率之外附带逐帧甲/乙说话人概率（每行和为 1），用于说话人归属分析。",
        "vocab": vocab,
        "frames": normalise(frames),
        "speakers": speakers,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("sample_repeat.json", repeat_sample()),
        ("sample_blank_heavy.json", blank_heavy_sample()),
        ("sample_sparse_zero.json", sparse_zero_sample()),
        ("sample_attribution.json", attribution_sample()),
    ):
        path = SAMPLES / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(payload['frames'])} frames)")


if __name__ == "__main__":
    main()
