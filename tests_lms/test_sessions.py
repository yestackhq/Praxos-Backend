from __future__ import annotations

import io

from tests_lms.test_indexing import _minimal_pdf


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def _upload(client, name="policy.pdf", text="Report security incidents to the IT team within 24 hours."):
    return client.post(
        "/api/documents/upload",
        files={"file": (name, io.BytesIO(_minimal_pdf(text)), "application/pdf")},
    ).json()


def test_start_requires_openai(client):
    """Without an OpenAI key, starting a voice session returns a clear 503."""
    try:
        _as({"sub": "sess_owner"})
        client.post("/api/bootstrap", json={"name": "Sess Owner", "email": "so@x.dev"})
        doc = _upload(client)
        r = client.post("/api/sessions/start", json={"documentId": doc["id"]})
        assert r.status_code == 503
        assert "OpenAI" in r.json()["detail"]
    finally:
        _clear()


def test_start_404_for_foreign_document(client):
    try:
        _as({"sub": "sess_a"})
        client.post("/api/bootstrap", json={"name": "A", "email": "a@sess.dev"})
        doc = _upload(client, name="a.pdf")
        _as({"sub": "sess_b"})
        client.post("/api/bootstrap", json={"name": "B", "email": "b@sess.dev"})
        r = client.post("/api/sessions/start", json={"documentId": doc["id"]})
        assert r.status_code == 404  # B can't start a session on A's document
    finally:
        _clear()


def test_score_writes_understanding_and_records_session(client, monkeypatch):
    from lms_app.routers import sessions

    monkeypatch.setattr(
        sessions.ai,
        "score_understanding",
        lambda doc_name, transcript: {
            "score": 80,
            "summary": "Solid grasp of incident reporting.",
            "topics": [{"name": "Reporting", "score": 80}],
            "strengths": ["Knew the 24h window"],
            "gaps": [],
        },
    )
    try:
        _as({"sub": "score_owner"})
        client.post("/api/bootstrap", json={"name": "Score Owner", "email": "score@x.dev"})
        doc = _upload(client)
        r = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "tutor", "text": "When must you report an incident?"},
                    {"role": "learner", "text": "Within 24 hours, to the IT team."},
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["score"] == 80
        assert body["understanding"] == 80  # first session → equals the score
        assert body["summary"]
        # A second, lower-scoring session blends understanding downward. (Substantive
        # answer so it clears the strict gate and reaches the scorer stub.)
        monkeypatch.setattr(sessions.ai, "score_understanding", lambda d, t: {"score": 40, "topics": []})
        r2 = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "learner", "text": "You report the incident to the IT security team within twenty four hours."}
                ],
            },
        ).json()
        assert r2["understanding"] == 60  # round((80 + 40) / 2)
    finally:
        _clear()


def test_score_gates_thin_or_garbage_answers(client, monkeypatch):
    """A near-silent / filler answer is scored low WITHOUT calling the LLM, so
    misrecognised noise can't produce a random high score."""
    from lms_app.routers import sessions

    def _boom(doc_name, transcript):  # must not be reached for a thin transcript
        raise AssertionError("score_understanding should not run for a thin transcript")

    monkeypatch.setattr(sessions.ai, "score_understanding", _boom)
    try:
        _as({"sub": "gate_owner"})
        client.post("/api/bootstrap", json={"name": "Gate Owner", "email": "gate@x.dev"})
        doc = _upload(client)
        r = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "tutor", "text": "Explain how to report an incident."},
                    {"role": "learner", "text": "yeah um ok right"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["score"] <= 15  # gated low, not a random high score
    finally:
        _clear()
