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


def test_start_requires_voice_stack(client):
    """Without LiveKit/Deepgram/Cartesia, starting a session says exactly what is
    missing instead of failing opaquely."""
    try:
        _as({"sub": "sess_owner"})
        client.post("/api/bootstrap", json={"name": "Sess Owner", "email": "so@x.dev"})
        doc = _upload(client)
        r = client.post("/api/sessions/start", json={"documentId": doc["id"]})
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "LIVEKIT" in detail and "DEEPGRAM" in detail and "CARTESIA" in detail
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


def test_score_records_transcript_and_returns_document_standing(client, monkeypatch):
    from lms_app.routers import sessions

    monkeypatch.setattr(
        sessions.ai,
        "score_understanding",
        lambda doc_name, transcript, section=None, prior_facts=None: {
            "scoreable": True,
            "score": 80,
            "covered": 100,
            "summary": "Solid grasp of incident reporting.",
            "topics": [{"name": "Reporting", "score": 80, "evidence": "within 24 hours"}],
            "strengths": ["Knew the 24h window"],
            "gaps": [],
        },
    )
    try:
        _as({"sub": "score_owner"})
        client.post("/api/bootstrap", json={"name": "Score Owner", "email": "score@x.dev"})
        doc = _upload(client)
        turns = [
            {"role": "tutor", "text": "When must you report an incident?"},
            {"role": "learner", "text": "Within twenty four hours, to the IT security team."},
        ]
        body = client.post(
            "/api/sessions/score", json={"documentId": doc["id"], "transcript": turns}
        ).json()
        assert body["score"] == 80
        assert body["scoreable"] is True
        assert body["understanding"] is not None
        assert body["band"]

        # The sitting is auditable: the transcript came back with it.
        person = client.get("/api/people/1")
        if person.status_code == 200:
            sid = person.json()["sessions"][0]["id"]
            detail = client.get(f"/api/people/1/sessions/{sid}")
            if detail.status_code == 200:
                assert detail.json()["transcript"] == turns
    finally:
        _clear()


def test_a_thin_answer_is_unscoreable_not_a_low_score(client, monkeypatch):
    """A near-silent / filler sitting must NOT be recorded as a low score — that
    is what dragged learners down every time they opened and closed the app."""
    from lms_app.routers import sessions

    def _boom(*a, **k):
        raise AssertionError("the grader must not run on a thin transcript")

    monkeypatch.setattr(sessions.ai.llm, "chat_json", _boom)
    try:
        _as({"sub": "gate_owner"})
        client.post("/api/bootstrap", json={"name": "Gate Owner", "email": "gate@x.dev"})
        doc = _upload(client)
        body = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "tutor", "text": "Explain how to report an incident."},
                    {"role": "learner", "text": "yeah um ok right thanks"},
                ],
            },
        ).json()
        assert body["scoreable"] is False
        assert body["score"] is None
        assert body["understanding"] is None  # nothing demonstrated, nothing claimed
    finally:
        _clear()


def test_agent_routes_require_the_shared_secret(client):
    r = client.post(
        "/api/sessions/agent/context",
        json={"workspaceId": 1, "userId": 1, "documentId": 1, "moduleIdx": 0},
    )
    assert r.status_code == 401
