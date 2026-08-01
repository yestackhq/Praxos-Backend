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
    assert agent.instructions == ["teach section 1"]
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
    so the spend is attributed to them as verified). Without this flag the
    worker's disconnect safety net graded the same turns again — two sittings
    recorded and two billed model calls for one conversation."""
    ctx = _ctx(module_idx=0)
    assert ctx.client_scored is False
    _, _, _, advance = _harness(ctx, _ok(1))
    asyncio.run(advance(1))
    assert ctx.client_scored is True


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
