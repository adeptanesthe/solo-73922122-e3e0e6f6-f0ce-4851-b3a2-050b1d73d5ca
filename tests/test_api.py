"""HTTP-level tests via FastAPI's TestClient.

These check the parts the browser relies on: uploads, revision handling,
error responses that leave the session untouched, and the peek endpoint.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app

VOCAB = ["<blank>", "a", "b"]
FRAMES = [
    [0.6, 0.4, 0.0],
    [0.1, 0.8, 0.1],
    [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5],
    [0.7, 0.2, 0.1],
]


@pytest.fixture()
def client():
    return TestClient(app)


def create(client, vocab=None, frames=None, width=4):
    response = client.post(
        "/api/sessions",
        json={"vocab": vocab or VOCAB, "beam_width": width, "frames": frames or FRAMES},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_index_and_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "流式 CTC 解码工作台" in page.text


def test_create_and_advance(client):
    session = create(client)
    assert session["frame_count"] == len(FRAMES)
    assert session["next_frame"] == 0
    assert session["beams"][0]["text"] == ""
    sid = session["session_id"]

    response = client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    assert response.status_code == 200
    state = response.json()
    assert state["next_frame"] == 5
    assert state["beams"]
    assert state["frame_info"]["frame"] == 5


def test_advance_chunked_matches_http_state(client):
    one = create(client)
    client.post("/api/advance", json={"session_id": one["session_id"], "to_frame": 5})
    full = client.get(f"/api/sessions/{one['session_id']}").json()

    other = create(client)
    for frame in range(1, 6):
        client.post("/api/advance", json={"session_id": other["session_id"], "to_frame": frame})
    stepwise = client.get(f"/api/sessions/{other['session_id']}").json()
    assert full["beams"] == stepwise["beams"]


def test_confirm_then_undo_over_http(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    state = client.get(f"/api/sessions/{sid}").json()
    target = state["beams"][0]["text"]
    assert target

    response = client.post("/api/confirm", json={"session_id": sid, "prefix": target})
    assert response.status_code == 200
    confirmed = response.json()
    assert confirmed["prefix"] == target
    assert all(beam["text"].startswith(target) for beam in confirmed["beams"])
    assert confirmed["prefix_scores"]["last_char"] == target[-1]
    assert confirmed["prefix_scores"]["logp"] is not None

    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    response = client.post("/api/undo", json={"session_id": sid})
    assert response.status_code == 200
    restored = response.json()
    assert restored["prefix"] == ""
    assert restored["confirmed_count"] == 0


def test_failed_confirm_keeps_state(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    before = client.get(f"/api/sessions/{sid}").json()

    response = client.post("/api/confirm", json={"session_id": sid, "prefix": "zzz"})
    assert response.status_code == 409
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before

    response = client.post("/api/undo", json={"session_id": sid})
    assert response.status_code == 409
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before


def test_invalid_inputs_return_400_without_session(client):
    for payload in (
        {"vocab": ["<blank>", "ab"], "beam_width": 4, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 0, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 17, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 4, "frames": [[0.5, 0.5]]},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 4, "frames": [[1.2, -0.2, 0.0]]},
    ):
        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 400, (payload, response.text)


def test_upload_endpoint(client):
    vocab_file = ("vocab.json", json.dumps(["<blank>", "a", "b"]).encode(), "application/json")
    frames_file = ("frames.json", json.dumps({"frames": FRAMES}).encode(), "application/json")
    response = client.post(
        "/api/sessions/upload",
        files={"vocab_file": vocab_file, "frames_file": frames_file},
        data={"beam_width": "4"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["frame_count"] == len(FRAMES)


def test_upload_bad_frames_returns_400(client):
    vocab_file = ("vocab.json", json.dumps(["<blank>", "a", "b"]).encode(), "application/json")
    frames_file = ("frames.json", b"{not json", "application/json")
    response = client.post(
        "/api/sessions/upload",
        files={"vocab_file": vocab_file, "frames_file": frames_file},
        data={"beam_width": "4"},
    )
    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_peek_only_for_consumed_frames(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})

    response = client.post("/api/peek", json={"session_id": sid, "frame": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["frame"] == 2
    assert payload["beams"]

    response = client.post("/api/peek", json={"session_id": sid, "frame": 4})
    assert response.status_code == 409


def test_samples_are_served(client):
    response = client.get("/api/samples")
    names = [item["name"] for item in response.json()["samples"]]
    assert "sample_repeat.json" in names
    sample = client.get("/api/samples/sample_sparse_zero.json").json()
    assert sample["vocab"][0] == "<blank>"
    assert any(0.0 in row for row in sample["frames"])
    assert client.get("/api/samples/..%2Fapp%2Fmain.py").status_code == 404


def test_sample_creates_working_session(client):
    sample = client.get("/api/samples/sample_repeat.json").json()
    response = client.post(
        "/api/sessions",
        json={"vocab": sample["vocab"], "beam_width": 6, "frames": sample["frames"]},
    )
    assert response.status_code == 200
    state = response.json()
    client.post("/api/advance", json={"session_id": state["session_id"], "to_frame": len(sample["frames"])})
    final = client.get(f"/api/sessions/{state['session_id']}").json()
    assert final["beams"][0]["text"].startswith("a")


def test_unknown_session_is_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/advance", json={"session_id": "nope", "to_frame": 1}).status_code == 404


def test_concurrent_ops_apply_in_order(client):
    """Advance and undo issued together must both apply, in submission order."""
    session = create(client)
    sid = session["session_id"]
    state = client.get(f"/api/sessions/{sid}").json()
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    state = client.get(f"/api/sessions/{sid}").json()
    target = state["beams"][0]["text"]
    client.post("/api/confirm", json={"session_id": sid, "prefix": target})

    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    undo = client.post("/api/undo", json={"session_id": sid})
    assert undo.status_code == 200
    final = undo.json()
    assert final["next_frame"] == 5
    assert final["prefix"] == ""


def test_manager_cleanup_after_delete(client):
    session = create(client)
    sid = session["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_slow_recompute_does_not_swallow_later_actions(monkeypatch):
    """A slow undo replay must not lose an advance issued while it runs.

    Undo and advance are queued on the session's single worker, so the advance
    is applied *after* the replay and the late undo response cannot revert it.
    """
    import asyncio
    import time

    import httpx

    from app import state as state_module

    original = state_module.undo

    def slow_undo(session):
        time.sleep(0.4)
        return original(session)

    monkeypatch.setattr(state_module, "undo", slow_undo)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = (
                await client.post(
                    "/api/sessions",
                    json={"vocab": VOCAB, "beam_width": 4, "frames": FRAMES},
                )
            ).json()
            sid = created["session_id"]
            await client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
            state = (await client.get(f"/api/sessions/{sid}")).json()
            target = state["beams"][0]["text"]
            await client.post("/api/confirm", json={"session_id": sid, "prefix": target})

            undo_task = asyncio.create_task(client.post("/api/undo", json={"session_id": sid}))
            await asyncio.sleep(0.05)
            advance_task = asyncio.create_task(
                client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
            )
            undo_response, advance_response = await asyncio.gather(undo_task, advance_task)
            final = (await client.get(f"/api/sessions/{sid}")).json()
            return undo_response, advance_response, final

    undo_response, advance_response, final = asyncio.run(scenario())
    assert undo_response.status_code == 200
    assert advance_response.status_code == 200
    # The later advance wins: prefix undone, frames advanced.
    assert final["next_frame"] == 5
    assert final["prefix"] == ""
    assert undo_response.json()["revision"] < final["revision"]
