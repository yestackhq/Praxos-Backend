from __future__ import annotations

"""Reconnect and resume.

The reported failure: a laptop sleeping mid-sitting silently killed the worker;
the browser's reconnect dispatched a REPLACEMENT worker that greeted the learner
and started the section over, with no memory of the conversation. Three parts:
the dying worker must save (its safety net was permanently disarmed after the
first advance), the replacement must find where the learner actually is (the
room metadata carries the sitting's STARTING section), and the tutor must get
the interrupted conversation back, not six distilled facts.
"""

import asyncio
import io
from datetime import timedelta

from tests_lms.test_section_advance import _ctx, _harness, _ok


def test_advance_rearms_the_shutdown_safety_net():
    """client_scored covers the turns the browser graded when it advanced. It
    used to stay True forever, so everything said AFTER the first advance was
    discarded if the sitting ended without a clean goodbye."""
    ctx = _ctx(module_idx=0)
    _, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert ctx.client_scored is False, "the safety net must protect the new section"


def test_next_section_instructions_carry_the_previous_sections_words():
    """Cross-section carry-over must not depend on the memory service having
    ingested a grade that is still being computed — the worker holds the turns
    and hands them over directly."""
    ctx = _ctx(module_idx=0, turns=4)
    ctx.transcript = [
        {"role": "tutor", "text": "What is a mechanism?"},
        {"role": "learner", "text": "My example is the ten step onboarding flow."},
    ]
    _, agent, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    swapped = agent.instructions[-1]
    assert "teach section 1" in swapped
    assert "PREVIOUS SECTION" in swapped
    assert "ten step onboarding flow" in swapped
    # Learner turns ONLY: carrying the tutor's side put its closing line at the
    # end of the new instructions and the model opened the next section by
    # repeating "you have finished this section, tap the button".
    assert "What is a mechanism?" not in swapped


def _seeded_doc(client, monkeypatch, clerk_id: str):
    from lms_app import models
    from lms_app.auth import optional_claims
    from lms_app.config import settings
    from lms_app.db import SessionLocal
    from lms_app.main import app
    from tests_lms.test_indexing import _minimal_pdf

    monkeypatch.setattr(settings, "AGENT_SHARED_SECRET", "test-agent-secret", raising=False)
    app.dependency_overrides[optional_claims] = lambda: {"sub": clerk_id}
    client.post("/api/bootstrap", json={"name": "Re Connect", "email": f"{clerk_id}@x.dev"})
    doc = client.post(
        "/api/documents/upload",
        files={"file": ("rc.pdf", io.BytesIO(_minimal_pdf("Alpha. Beta. Gamma.")), "application/pdf")},
    ).json()
    with SessionLocal() as db:
        user = db.query(models.User).filter_by(clerk_id=clerk_id).one()
        for i in range(3):
            db.add(
                models.Module(
                    document_id=doc["id"], idx=i, title=f"S{i}",
                    key_points=[f"point {i}"], check_questions=[f"q {i}"],
                    minutes=5, chunk_start=0, chunk_end=1,
                )
            )
        db.commit()
        return doc["id"], user.id, user.workspace_id


def test_locate_resolves_a_recently_paused_section(client, monkeypatch):
    """A replacement worker boots with the section the sitting STARTED on. With
    locate, the API answers with the recently paused section instead."""
    from lms_app import models
    from lms_app.auth import optional_claims
    from lms_app.db import SessionLocal
    from lms_app.main import app

    did, uid, wsid = _seeded_doc(client, monkeypatch, "rc_locate")
    try:
        with SessionLocal() as db:
            db.add(
                models.SectionProgress(
                    user_id=uid, document_id=did, module_idx=2, status="paused",
                    updated_at=models.utcnow(),
                )
            )
            db.add(
                models.LearningSession(
                    user_id=uid, document_id=did, module_idx=2,
                    transcript=[
                        {"role": "tutor", "text": "Where were we — what does gamma mean?"},
                        {"role": "learner", "text": "Gamma is the third greek letter."},
                    ],
                )
            )
            db.commit()

        headers = {"X-Agent-Secret": "test-agent-secret"}
        payload = {"workspaceId": wsid, "userId": uid, "documentId": did}

        # Stamped with section 0, locate resolves the paused section 2 — and the
        # instructions replay the interrupted conversation.
        resp = client.post(
            "/api/sessions/agent/context",
            json={**payload, "moduleIdx": 0, "locate": True},
            headers=headers,
        ).json()
        assert resp["moduleIdx"] == 2
        assert "INTERRUPTED CONVERSATION" in resp["instructions"]
        assert "third greek letter" in resp["instructions"]

        # Without locate the stamp wins — a normal start is untouched.
        plain = client.post(
            "/api/sessions/agent/context", json={**payload, "moduleIdx": 0}, headers=headers
        ).json()
        assert plain["moduleIdx"] == 0
    finally:
        app.dependency_overrides.pop(optional_claims, None)


def test_locate_ignores_an_old_pause(client, monkeypatch):
    """The locate window is a reconnect affordance. A pause from hours ago goes
    through start_session's own resume logic, not a metadata override."""
    from lms_app import models
    from lms_app.auth import optional_claims
    from lms_app.db import SessionLocal
    from lms_app.main import app

    did, uid, wsid = _seeded_doc(client, monkeypatch, "rc_stale")
    try:
        with SessionLocal() as db:
            db.add(
                models.SectionProgress(
                    user_id=uid, document_id=did, module_idx=2, status="paused",
                    updated_at=models.utcnow() - timedelta(hours=2),
                )
            )
            db.commit()

        resp = client.post(
            "/api/sessions/agent/context",
            json={"workspaceId": wsid, "userId": uid, "documentId": did, "moduleIdx": 0, "locate": True},
            headers={"X-Agent-Secret": "test-agent-secret"},
        ).json()
        assert resp["moduleIdx"] == 0
    finally:
        app.dependency_overrides.pop(optional_claims, None)


def test_advance_keeps_the_leaving_turns_until_the_browser_confirms():
    """The browser's advance-time grade post is best-effort; twice a silent
    failure lost a whole section's conversation. The worker now keeps its copy
    of the leaving segment until a "scored" confirmation arrives."""
    ctx = _ctx(module_idx=0)
    ctx.transcript = [
        {"role": "tutor", "text": "What is the canvas?"},
        {"role": "learner", "text": "A review sheet with the three lenses."},
    ]
    _, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert len(ctx.unconfirmed) == 1
    seg = ctx.unconfirmed[0]
    assert seg["module_idx"] == 0
    assert seg["turns"][-1]["text"] == "A review sheet with the three lenses."


def test_safety_net_grades_unconfirmed_segments_even_when_the_end_was_claimed():
    """"ending" means the browser grades the FINAL section itself. It must not
    also bury an earlier section whose grade post never landed."""
    from voice_agent import agent as agent_mod

    ctx = _ctx(module_idx=1)
    ctx.client_scored = True  # browser claimed the final section
    ctx.unconfirmed = [{"module_idx": 0, "turns": [{"role": "learner", "text": "lens talk"}]}]
    posted: list[tuple[int, bool]] = []

    async def fake_post_turns(_ctx, *, module_idx, turns, paused):
        posted.append((module_idx, paused))

    orig = agent_mod.post_turns
    agent_mod.post_turns = fake_post_turns
    try:
        asyncio.run(agent_mod.post_score(ctx, paused=True))
    finally:
        agent_mod.post_turns = orig
    assert posted == [(0, False)], "the orphaned section is graded; the claimed final is not"
    assert ctx.unconfirmed == []


def test_section_opener_cannot_be_a_silent_tool_call():
    """The first reply of a new section must be speech: a model that spent it on
    a silent mark_section_understood left the learner in a dead room."""
    ctx = _ctx(module_idx=0)
    session, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert session.reply_kwargs.get("tool_choice") == "none"


def test_recording_a_sitting_moves_the_resume_point_before_the_grade_lands(client, monkeypatch):
    """Grading takes the assessor minutes; the resume point must not wait for
    it. A substantive sitting counts as sat the moment it is recorded — a
    filler sitting still does not (phantom-sitting rules unchanged)."""
    from lms_app import models, scoring
    from lms_app.auth import optional_claims
    from lms_app.db import SessionLocal
    from lms_app.main import app
    from lms_app.routers.sessions import _record_sitting

    did, uid, wsid = _seeded_doc(client, monkeypatch, "rc_defer")
    try:
        with SessionLocal() as db:
            user = db.get(models.User, uid)
            doc = db.get(models.Document, did)
            row = _record_sitting(
                db, user=user, doc=doc, module_idx=0,
                transcript=[
                    {"role": "tutor", "text": "What is alpha?"},
                    {"role": "learner", "text": "Alpha is the first letter and it starts every sequence."},
                ],
                paused=False, total_sections=3,
            )
            db.commit()
            assert row.score is None, "no grade yet — the assessor has not run"
            assert scoring.next_section_idx(db, uid, did, 3) == 1, (
                "the learner sat section 0; resume must move on immediately"
            )

            filler = _record_sitting(
                db, user=user, doc=doc, module_idx=1,
                transcript=[{"role": "tutor", "text": "Only the tutor spoke here."}],
                paused=False, total_sections=3,
            )
            db.commit()
            assert filler.score is None
            assert scoring.next_section_idx(db, uid, did, 3) == 1, (
                "a tutor-only sitting still does not count as sitting the section"
            )
    finally:
        app.dependency_overrides.pop(optional_claims, None)
