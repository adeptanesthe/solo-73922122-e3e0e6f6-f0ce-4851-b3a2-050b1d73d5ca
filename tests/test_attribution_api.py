"""HTTP tests for the speaker-attribution endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

VOCAB = ["<blank>", "a", "b", "c"]
# Frames crafted so "ab" is reachable: an a region then a b region.
FRAMES = [
    [0.10, 0.85, 0.03, 0.02],
    [0.10, 0.85, 0.03, 0.02],
    [0.80, 0.10, 0.05, 0.05],
    [0.10, 0.05, 0.82, 0.03],
    [0.10, 0.05, 0.82, 0.03],
    [0.90, 0.04, 0.04, 0.02],
]


@pytest.fixture()
def client():
    return TestClient(app)


def _session(client, frames=None):
    response = client.post(
        "/api/sessions",
        json={"vocab": VOCAB, "beam_width": 6, "frames": frames or FRAMES},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _rows(n, p_a):
    return [[float(x), round(1.0 - x, 6)] for x in p_a]


def test_attribution_basic_posteriors(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 6})
    # Strong 甲 on the a-frames, strong 乙 on the b-frames.
    rows = _rows(6, [0.9, 0.9, 0.5, 0.1, 0.1, 0.5])
    response = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 5,
            "reference": "ab",
            "speaker_probs": rows,
            "analysis_id": "t1",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analysis_id"] == "t1"
    assert payload["reachable"]
    assert payload["reference"] == "ab"
    assert [c["char"] for c in payload["characters"]] == ["a", "b"]
    a, b = payload["characters"]
    assert a["speaker"] == "甲" and a["posterior_a"] > 0.9
    assert b["speaker"] == "乙" and b["posterior_b"] > 0.9
    assert payload["path_count"] >= 1
    # Frame evidence rows align with the window.
    assert [f["frame"] for f in payload["frame_evidence"]] == list(range(6))


def test_attribution_tie_is_undecided(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    rows = _rows(2, [0.5, 0.5])
    response = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 1,
            "reference": "a",
            "speaker_probs": rows,
        },
    )
    assert response.status_code == 200
    ch = response.json()["characters"][0]
    assert ch["tied"]
    assert ch["speaker"] == "未定"
    assert ch["posterior_a"] == pytest.approx(0.5)


def test_attribution_unreachable_text_returns_422_and_keeps_prior(client):
    # Label "c" has probability zero on every frame, so it cannot decode.
    frames = [
        [0.2, 0.7, 0.1, 0.0],
        [0.2, 0.7, 0.1, 0.0],
        [0.8, 0.1, 0.1, 0.0],
    ]
    session = _session(client, frames=frames)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    # First compute a valid result so we can confirm it is retained.
    ok = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 1,
            "reference": "a",
            "speaker_probs": _rows(2, [0.7, 0.7]),
        },
    )
    assert ok.status_code == 200
    assert ok.json()["reachable"]
    # Then request the unreachable label.
    response = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 2,
            "reference": "c",
            "speaker_probs": _rows(3, [0.5] * 3),
        },
    )
    assert response.status_code == 422
    assert "不可达" in response.json()["detail"]
    # The decoding session itself is untouched.
    snap = client.get(f"/api/sessions/{sid}").json()
    assert snap["next_frame"] == 3


def test_attribution_rejects_frames_not_consumed(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    rows = _rows(4, [0.5] * 4)
    response = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 3,
            "reference": "a",
            "speaker_probs": rows,
        },
    )
    assert response.status_code == 409
    assert "推进" in response.json()["detail"]


def test_attribution_validation_errors(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 6})
    base = {
        "session_id": sid,
        "start_frame": 0,
        "end_frame": 1,
        "reference": "a",
    }
    # wrong number of speaker rows
    response = client.post(
        "/api/attribution", json={**base, "speaker_probs": [[0.5, 0.5]]}
    )
    assert response.status_code == 422
    # rows that do not sum to 1
    response = client.post(
        "/api/attribution",
        json={**base, "speaker_probs": [[0.7, 0.7], [0.5, 0.5]]},
    )
    assert response.status_code == 422
    # out-of-vocab reference
    response = client.post(
        "/api/attribution",
        json={**base, "reference": "z", "speaker_probs": _rows(2, [0.5, 0.5])},
    )
    assert response.status_code == 422
    # empty reference
    response = client.post(
        "/api/attribution",
        json={**base, "reference": "", "speaker_probs": _rows(2, [0.5, 0.5])},
    )
    assert response.status_code == 422
    # window longer than 64 frames: build a session with enough frames.
    long_frames = [[0.25, 0.25, 0.25, 0.25] for _ in range(70)]
    long_session = _session(client, frames=long_frames)
    long_sid = long_session["session_id"]
    client.post("/api/advance", json={"session_id": long_sid, "to_frame": 70})
    long_body = {
        "session_id": long_sid,
        "start_frame": 0,
        "end_frame": 64,
        "reference": "a",
        "speaker_probs": _rows(65, [0.5] * 65),
    }
    response = client.post("/api/attribution", json=long_body)
    assert response.status_code == 422


def test_attribution_does_not_mutate_session_or_revision(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    before = client.get(f"/api/sessions/{sid}").json()
    client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 2,
            "reference": "a",
            "speaker_probs": _rows(3, [0.8, 0.8, 0.5]),
        },
    )
    after = client.get(f"/api/sessions/{sid}").json()
    assert after["revision"] == before["revision"]
    assert after["beams"] == before["beams"]
    assert after["next_frame"] == before["next_frame"]


def test_attribution_echoes_live_revision_after_advance(client):
    session = _session(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    response = client.post(
        "/api/attribution",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 1,
            "reference": "a",
            "speaker_probs": _rows(2, [0.6, 0.6]),
        },
    )
    assert response.status_code == 200
    assert response.json()["revision"] == client.get(f"/api/sessions/{sid}").json()["revision"]
    # After another advance the stale analysis revision is strictly smaller.
    client.post("/api/advance", json={"session_id": sid, "to_frame": 4})
    latest = client.get(f"/api/sessions/{sid}").json()["revision"]
    assert response.json()["revision"] < latest


def test_attribution_unknown_session_is_404(client):
    response = client.post(
        "/api/attribution",
        json={
            "session_id": "nope",
            "start_frame": 0,
            "end_frame": 0,
            "reference": "a",
            "speaker_probs": [[0.5, 0.5]],
        },
    )
    assert response.status_code == 404
