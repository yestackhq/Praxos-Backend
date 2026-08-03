from __future__ import annotations

"""Section switching.

The reported failure: finishing a section and tapping "Next" sometimes left the
learner on the section they had just completed. It was intermittent because it
depended on whether the tutor happened to be mid-sentence when the button was
tapped — ``update_instructions`` only affects the NEXT inference, so a turn
already in flight kept speaking about the old section and the new turn queued
behind it.

These drive ``advance_section`` directly with fakes, which is the unit the bug
lived in, and cover the three ways an advance could previously get stuck:
ordering, a failed context fetch, and running off the end of the plan.
"""

import asyncio

import pytest

from voice_agent.agent import SessionContext, advance_section


class FakeSession:
    """Records the order of the calls that matter."""

    def __init__(self):
        self.calls: list[str] = []
        self.interrupt_fails = False

    async def interrupt(self):
        if self.interrupt_fails:
            raise RuntimeError("nothing to interrupt")
        self.calls.append("interrupt")

    def generate_reply(self):
        self.calls.append("generate_reply")


class FakeAgent:
    def __init__(self, session: FakeSession):
        self.instructions: list[str] = []
        self._session = session

    async def update_instructions(self, text: str):
        self._session.calls.append("update_instructions")
        self.instructions.append(text)


def _ctx(*, module_idx=0, total=4, turns=6) -> SessionContext:
    ctx = SessionContext(
        {"workspaceId": 1, "userId": 9, "documentId": 4, "moduleIdx": module_idx}
    )
    ctx.total_modules = total
    ctx.transcript = [{"role": "learner", "text": f"turn {i}"} for i in range(turns)]
    return ctx


def _harness(ctx, fetch):
    session = FakeSession()
    agent = FakeAgent(session)
    published: list[dict] = []

    async def publish(_room, payload):
        published.append(payload)

    return session, agent, published, lambda target: advance_section(
        target, sctx=ctx, session=session, agent=agent, room=None, fetch=fetch, publish=publish
    )


def _ok(idx: int, *, total=4):
    async def fetch(_ctx, advancing=False):
        return {
            "complete": False,
            "instructions": f"teach section {idx}",
            "moduleIdx": idx,
            "moduleTitle": f"S{idx}",
            "totalModules": total,
            "isLast": idx >= total - 1,
        }

    return fetch


# ---- the ordering fix --------------------------------------------------------


def test_current_turn_is_stopped_before_the_swap():
    """The regression that caused the bug: instructions were swapped and a reply
    generated while the previous turn was still being spoken."""
    ctx = _ctx(module_idx=0)
    session, agent, published, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))

    assert session.calls == ["interrupt", "update_instructions", "generate_reply"], session.calls
    # The fetched instructions come first; the previous section's hand-off
    # digest (see test_reconnect_resume) rides behind them.
    assert agent.instructions[0].startswith("teach section 1")
    assert ctx.module_idx == 1
    assert published[-1]["type"] == "section_changed"
    assert published[-1]["moduleIdx"] == 1


def test_advance_still_completes_when_there_is_nothing_to_interrupt():
    """A silent tutor must not block the swap."""
    ctx = _ctx(module_idx=0)
    session, agent, published, advance = _harness(ctx, _ok(1))
    session.interrupt_fails = True
    asyncio.run(advance(1))

    assert ctx.module_idx == 1
    assert "generate_reply" in session.calls
    assert published[-1]["type"] == "section_changed"


def test_section_boundary_moves_so_the_next_grade_covers_only_the_new_section():
    ctx = _ctx(module_idx=0, turns=6)
    _, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert ctx.scored_upto == 6
    assert ctx.section_turns() == []


# ---- failure no longer dead-ends --------------------------------------------


def test_a_failed_fetch_rolls_back_and_tells_the_browser():
    """Previously this returned silently: the UI had already hidden the button
    and was waiting for a section change that never came."""
    ctx = _ctx(module_idx=2)

    async def fetch(_ctx, advancing=False):
        return None

    _, agent, published, advance = _harness(ctx, fetch)
    asyncio.run(advance(3))

    assert ctx.module_idx == 2, "must not claim to have moved"
    assert agent.instructions == []
    assert published == [{"type": "advance_failed", "moduleIdx": 2}]


def test_running_past_the_last_section_reports_completion():
    """The backend used to CLAMP a past-the-end request back to the final
    section, so the worker re-taught the section just finished — which is
    exactly 'it says the lesson is already completed'."""
    ctx = _ctx(module_idx=3, total=4)

    async def fetch(_ctx, advancing=False):
        return {"complete": True, "moduleIdx": 3, "totalModules": 4}

    _, agent, published, advance = _harness(ctx, fetch)
    # total_modules is known here, so it short-circuits before even fetching.
    asyncio.run(advance(4))

    assert ctx.module_idx == 3
    assert agent.instructions == []
    assert published == [{"type": "course_complete", "moduleIdx": 3}]


def test_backend_reported_completion_is_honoured_when_the_count_is_stale():
    """If the worker's section count is out of date it still must not re-teach:
    the backend's `complete` flag settles it."""
    ctx = _ctx(module_idx=3, total=0)  # unknown count

    async def fetch(_ctx, advancing=False):
        return {"complete": True, "moduleIdx": 3, "totalModules": 4}

    _, agent, published, advance = _harness(ctx, fetch)
    asyncio.run(advance(4))

    assert ctx.module_idx == 3
    assert agent.instructions == []
    assert published[-1]["type"] == "course_complete"


# ---- duplicate / out-of-order taps ------------------------------------------


def test_a_repeated_tap_for_the_same_section_is_ignored():
    ctx = _ctx(module_idx=1)
    session, agent, published, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))  # already there
    assert session.calls == []
    assert published == []


def test_two_taps_in_flight_do_not_interleave():
    """Both advances are serialised, so the worker can never end up teaching one
    section while the UI has been told about another."""
    ctx = _ctx(module_idx=0, total=4)
    order: list[str] = []

    async def slow_fetch(c, advancing=False):
        order.append(f"fetch-start:{c.module_idx}")
        await asyncio.sleep(0.01)
        order.append(f"fetch-end:{c.module_idx}")
        return {
            "complete": False,
            "instructions": f"teach {c.module_idx}",
            "moduleIdx": c.module_idx,
            "totalModules": 4,
            "isLast": False,
        }

    session, agent, published, advance = _harness(ctx, slow_fetch)

    async def both():
        await asyncio.gather(advance(1), advance(2))

    asyncio.run(both())

    # Serialised: the first fetch completes before the second begins.
    assert order == ["fetch-start:1", "fetch-end:1", "fetch-start:2", "fetch-end:2"], order
    assert ctx.module_idx == 2
    assert [p["moduleIdx"] for p in published if p["type"] == "section_changed"] == [1, 2]


# ---- the browser and the worker must not both grade -------------------------


def test_advancing_marks_the_section_as_graded_by_the_browser():
    """The browser grades the section it is leaving (it holds the learner's token,
    so the spend is attributed to them as verified). The worker's safety net must
    not grade those turns again — after the swap they sit BEHIND the section
    boundary, so section_turns() has nothing left to re-grade. The flag itself is
    re-armed for the new section (test_reconnect_resume), because leaving it set
    discarded everything said after the first advance on an unclean exit."""
    ctx = _ctx(module_idx=0)
    assert ctx.client_scored is False
    _, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert ctx.scored_upto == len(ctx.transcript)
    assert ctx.section_turns() == [], "nothing left for the safety net to re-grade"


# ---- the backend half of the same bug ---------------------------------------


def test_agent_context_reports_completion_instead_of_reserving_the_last_section(
    client, monkeypatch
):
    """/agent/context used to clamp moduleIdx to the last section, so asking for
    'the section after the last one' returned the last one again."""
    import io

    from lms_app import models
    from lms_app.config import settings
    from lms_app.db import SessionLocal
    from tests_lms.test_indexing import _minimal_pdf

    monkeypatch.setattr(settings, "AGENT_SHARED_SECRET", "test-agent-secret", raising=False)

    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: {"sub": "adv_owner"}
    try:
        me = client.post(
            "/api/bootstrap", json={"name": "Adv Owner", "email": "adv@x.dev"}
        ).json()
        doc = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "adv.pdf",
                    io.BytesIO(_minimal_pdf("Section one text. Section two text.")),
                    "application/pdf",
                )
            },
        ).json()

        with SessionLocal() as db:
            user = db.query(models.User).filter_by(clerk_id="adv_owner").one()
            for i in range(2):
                db.add(
                    models.Module(
                        document_id=doc["id"], idx=i, title=f"S{i}", minutes=5,
                        chunk_start=0, chunk_end=1,
                    )
                )
            db.commit()
            uid, wsid = user.id, user.workspace_id

        headers = {"X-Agent-Secret": "test-agent-secret"}
        payload = {"workspaceId": wsid, "userId": uid, "documentId": doc["id"]}

        last = client.post(
            "/api/sessions/agent/context", json={**payload, "moduleIdx": 1}, headers=headers
        ).json()
        assert last["complete"] is False
        assert last["moduleIdx"] == 1
        assert last["isLast"] is True

        beyond = client.post(
            "/api/sessions/agent/context", json={**payload, "moduleIdx": 2}, headers=headers
        ).json()
        assert beyond["complete"] is True, "must not silently re-serve the last section"
        assert "instructions" not in beyond
        assert me["role"] == "Admin"
    finally:
        app.dependency_overrides.pop(optional_claims, None)


def test_join_token_dispatches_the_named_agent(monkeypatch):
    """The worker registers under an agent_name, which disables LiveKit's
    automatic dispatch. Without an explicit dispatch request on the token the
    learner joins a room no agent ever enters — they hear silence and nothing
    logs an error."""
    import base64
    import json as _json

    from lms_app import voice
    from lms_app.config import settings

    monkeypatch.setattr(settings, "LIVEKIT_URL", "wss://x.livekit.cloud", raising=False)
    monkeypatch.setattr(settings, "LIVEKIT_API_KEY", "APItest", raising=False)
    monkeypatch.setattr(settings, "LIVEKIT_API_SECRET", "s" * 40, raising=False)

    jwt = voice.mint_join_token(
        room="praxos-u9-d4-s0-abcd",
        identity="learner-9",
        name="Kiran Varma",
        metadata=_json.dumps({"userId": 9, "documentId": 4}),
    )
    assert jwt

    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    claims = _json.loads(base64.urlsafe_b64decode(payload))

    cfg = claims.get("roomConfig") or claims.get("room_config") or {}
    agents = cfg.get("agents") or []
    assert agents, f"token carries no agent dispatch: {claims}"
    assert agents[0].get("agentName") == voice.AGENT_IDENTITY
    # The dispatch carries the ids, so the worker can resolve the lesson before
    # room/participant metadata has replicated.
    assert '"documentId": 4' in agents[0].get("metadata", "")


def test_confidence_accepts_the_scale_the_model_actually_uses():
    """A live session failed here. The tool declared `confidence: int` for a 0-100
    value; the model sent 0.95 — the natural reading of "confidence" — pydantic
    rejected the call, and the learner never got a Next-section button while the
    tutor told them to move on. Both scales must work."""
    import inspect

    from voice_agent.agent import TutorAgent

    fn = TutorAgent.mark_section_understood
    raw = getattr(fn, "__wrapped__", None) or getattr(fn, "func", None) or fn
    sig = inspect.signature(raw)
    assert sig.parameters["confidence"].annotation in (float, "float"), (
        "confidence must be a float; an int type rejects 0.95 outright"
    )


def test_advance_tool_schema_advertises_a_number():
    from lms_app.tutor import ADVANCE_TOOL

    prop = ADVANCE_TOOL["parameters"]["properties"]["confidence"]
    assert prop["type"] == "number", "integer makes a 0-1 confidence fail validation"


# ---- the tutor is speaking, not writing --------------------------------------


def test_prompt_forbids_written_formatting():
    """Everything the tutor produces is read aloud by Cartesia. A live session
    produced arrows, colons-as-labels and quoted blocks, which are spoken
    literally and are gibberish to the ear."""
    from lms_app import tutor

    text = tutor.build_instructions(
        doc_name="D",
        sections=[{"idx": 0, "title": "S", "description": "d", "topics": [],
                   "key_points": ["k"], "check_questions": ["q"]}],
        idx=0,
        material="m",
    )
    # Collapse whitespace: the prompt is hand-wrapped prose, so a phrase can be
    # split across lines. Asserting on raw text would fail on a reflow.
    lowered = " ".join(text.lower().split())
    assert "converted straight to speech" in lowered
    for rule in ("no markdown", "no bullet points", "no numbered lists"):
        assert rule in lowered, rule
    # Brevity has to be stated as a hard limit, not a preference.
    assert "two or three sentences" in lowered
    assert "hard limit" in lowered


def test_prompt_biases_towards_advancing():
    """The reported failure was being pinned to section one while the tutor kept
    asking questions."""
    from lms_app import tutor

    text = tutor.build_instructions(
        doc_name="D",
        sections=[{"idx": 0, "title": "S", "description": "d", "topics": [],
                   "key_points": ["k"], "check_questions": ["q"]}],
        idx=0,
        material="m",
    )
    lowered = " ".join(text.lower().split())
    assert "when in doubt, advance" in lowered
    assert "one good answer is enough" in lowered
    # Asking to move on must be honoured immediately.
    assert "ask to move on" in lowered


def test_advance_tool_floor_is_low_enough_to_be_reachable():
    """This guard exists to stop the tool firing on 'yeah, ok' — not to
    second-guess the tutor. Set high, it became the thing keeping learners stuck."""
    import inspect

    from voice_agent import agent as agent_mod

    src = inspect.getsource(agent_mod.TutorAgent.mark_section_understood)
    assert "< 4" in src and "< 40" in src, src[-400:]


def test_a_sitting_with_no_learner_speech_is_not_recorded_as_attempting_a_section(client):
    """The phantom sittings. Every 'Section N+1' row in production held exactly one
    tutor line — "You have finished this section, tap the button" — and zero
    learner turns. It must not count as having sat that section, or the resume
    point skips past a section the learner never saw."""
    from lms_app import models, scoring
    from lms_app.db import SessionLocal

    with SessionLocal() as db:
        ws = models.Workspace(name="Phantom", plan="x")
        db.add(ws)
        db.flush()
        u = models.User(clerk_id="ph_u", workspace_id=ws.id, name="P", email="p@x.dev", role="Learner")
        doc = models.Document(workspace_id=ws.id, name="Phantom Doc", chunk_count=3)
        db.add_all([u, doc])
        db.flush()
        for i in range(3):
            db.add(models.Module(document_id=doc.id, idx=i, title=f"S{i}", minutes=5))
        db.commit()

        # Section 1 sat properly; section 2 gets only the trailing tutor line.
        scoring.apply_session(
            db, user=u, document=doc, module_idx=0,
            transcript=[{"role": "learner", "text": "A mechanism is how the product creates the change."}],
            result={"scoreable": True, "score": 80, "covered": 100, "topics": []},
            paused=False, total_sections=3,
        )
        scoring.apply_session(
            db, user=u, document=doc, module_idx=1,
            transcript=[{"role": "tutor", "text": "You have finished this section. Tap the button."}],
            result={"scoreable": False, "score": None, "topics": []},
            paused=False, total_sections=3,
        )
        db.commit()

        assert scoring.attempted_sections(db, u.id, doc.id) == {0}, "section 2 was never really sat"
        assert scoring.next_section_idx(db, u.id, doc.id, 3) == 1, "must resume at section 2, not skip it"
