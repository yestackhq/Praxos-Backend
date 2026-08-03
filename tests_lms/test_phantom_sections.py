from __future__ import annotations

"""Phantom sections: a grade aimed past the end of the plan.

The production failure: the session flow walked a learner past the last real
section, the tutor had nothing to teach, and the resulting near-empty sitting
was CLAMPED onto the final real section and graded — one junk grade there sank
the whole document, flipped it to "completed", and forced a full retake.

These pin the three guards: a past-the-end grade touches nothing, a grader
reply without a usable number is unscoreable rather than a zero, and covered=0
survives instead of being rewritten as "fully covered"."""

import io

from sqlalchemy import select

from lms_app import models, scoring
from lms_app.db import SessionLocal
from tests_lms.test_indexing import _minimal_pdf


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def _upload(client, name="phantom.pdf"):
    return client.post(
        "/api/documents/upload",
        files={
            "file": (
                name,
                io.BytesIO(_minimal_pdf("Report incidents to IT within 24 hours.")),
                "application/pdf",
            )
        },
    ).json()


def _add_modules(document_id: int, n: int) -> None:
    with SessionLocal() as db:
        for i in range(n):
            db.add(
                models.Module(
                    document_id=document_id,
                    idx=i,
                    title=f"S{i}",
                    key_points=[f"point {i}"],
                    check_questions=[f"question {i}?"],
                )
            )
        db.commit()


def test_score_past_the_end_of_the_plan_grades_nothing(client, monkeypatch):
    """moduleIdx == total means the course is over — the sitting must not be
    graded, recorded, or clamped onto the final real section."""
    from lms_app.routers import sessions

    def _boom(*a, **k):
        raise AssertionError("the grader must not run for a section past the end")

    monkeypatch.setattr(sessions.ai, "score_understanding", _boom)
    try:
        _as({"sub": "phantom_owner"})
        client.post("/api/bootstrap", json={"name": "Phantom Owner", "email": "ph@x.dev"})
        doc = _upload(client)
        _add_modules(doc["id"], 2)

        body = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "moduleIdx": 2,  # sections are 0 and 1 — this one does not exist
                "transcript": [
                    {"role": "tutor", "text": "That was the last section."},
                    {"role": "learner", "text": "There is nothing here, what am I meant to say?"},
                ],
            },
        ).json()

        assert body["scoreable"] is False
        assert body["score"] is None
        assert body["totalModules"] == 2

        with SessionLocal() as db:
            rows = db.scalars(
                select(models.LearningSession).where(
                    models.LearningSession.document_id == doc["id"]
                )
            ).all()
            assert rows == []  # nothing recorded at ANY index — no clamping
            progress = db.scalars(
                select(models.SectionProgress).where(
                    models.SectionProgress.document_id == doc["id"]
                )
            ).all()
            assert progress == []
    finally:
        _clear()


def test_agent_score_past_the_end_is_refused(client, monkeypatch):
    from lms_app.config import settings

    monkeypatch.setattr(settings, "AGENT_SHARED_SECRET", "s3cret")
    try:
        _as({"sub": "phantom_agent"})
        client.post("/api/bootstrap", json={"name": "Phantom Agent", "email": "pa@x.dev"})
        doc = _upload(client, name="agent-phantom.pdf")
        _add_modules(doc["id"], 2)
        with SessionLocal() as db:
            user_id = db.scalar(
                select(models.User.id).where(models.User.clerk_id == "phantom_agent")
            )
            ws_id = db.scalar(
                select(models.User.workspace_id).where(models.User.id == user_id)
            )

        r = client.post(
            "/api/sessions/agent/score",
            headers={"X-Agent-Secret": "s3cret"},
            json={
                "workspaceId": ws_id,
                "userId": user_id,
                "documentId": doc["id"],
                "moduleIdx": 2,
                "transcript": [{"role": "learner", "text": "some words that would grade"}],
            },
        ).json()
        assert r["recorded"] is False

        with SessionLocal() as db:
            rows = db.scalars(
                select(models.LearningSession).where(
                    models.LearningSession.document_id == doc["id"]
                )
            ).all()
            assert rows == []
    finally:
        _clear()


def test_non_numeric_grader_score_is_unscoreable_not_zero(monkeypatch):
    """A grader answering "N/A" was coerced to a recorded 0 by _as_int."""
    from lms_app import ai

    monkeypatch.setattr(
        ai.llm,
        "chat_json",
        lambda *a, **k: {"score": "N/A", "summary": "Nothing was taught."},
    )
    result = ai.score_understanding(
        "Doc",
        [
            {"role": "tutor", "text": "..."},
            {"role": "learner", "text": "there is nothing here what am i meant to say"},
        ],
    )
    assert result["scoreable"] is False
    assert result["score"] is None


def test_covered_zero_is_recorded_not_rewritten_as_full(client):
    """covered=0 (an outage sitting, or a section never taught) must be stored
    faithfully — `or 100` made empty sittings look like full assessments."""
    with SessionLocal() as db:
        ws = models.Workspace(name="CoveredCo", plan="Admin workspace")
        db.add(ws)
        db.flush()
        u = models.User(
            clerk_id="covered_u", workspace_id=ws.id, name="C", email="c@t.dev", role="Learner"
        )
        doc = models.Document(workspace_id=ws.id, name="Covered Doc", chunk_count=1)
        db.add_all([u, doc])
        db.flush()
        db.add(models.Module(document_id=doc.id, idx=0, title="S0"))
        db.flush()

        row = scoring.apply_session(
            db,
            user=u,
            document=doc,
            module_idx=0,
            transcript=[{"role": "learner", "text": "a real answer with enough words"}],
            result={"scoreable": True, "score": 50, "covered": 0, "summary": ""},
            paused=False,
            total_sections=1,
        )
        db.commit()
        assert row.covered == 0
