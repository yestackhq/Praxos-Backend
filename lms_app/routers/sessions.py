from __future__ import annotations

"""Voice teaching sessions.

  start   → creates a LiveKit room carrying the learner/document/section ids and
            returns a short-lived join token. The agent worker (voice_agent/)
            is dispatched into that room and runs Deepgram → LLM → Cartesia.
  section → instructions for advancing mid-session, without reconnecting.
  score   → grades a sitting against the section's key points and source text,
            records the transcript, and folds the result into the learner's
            standing (see scoring.py).

The agent worker authenticates with the shared ``AGENT_SHARED_SECRET`` on the
``/agent/*`` routes; learners authenticate with their Clerk session as usual.
"""

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import ai, llm, memory, meldos, models, plan as plan_service, poke, scoring, tutor, voice
from ..auth import active_membership, bearer_token
from ..config import settings
from ..db import get_db

logger = logging.getLogger("praxos.sessions")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class StartIn(BaseModel):
    documentId: int
    moduleIdx: Optional[int] = None  # explicit section; else resume at the saved point
    restart: bool = False  # re-learn from the first section


class Turn(BaseModel):
    role: str  # "learner" | "tutor"
    text: str


class ScoreIn(BaseModel):
    documentId: int
    transcript: list[Turn] = []
    moduleIdx: Optional[int] = None
    paused: bool = False  # learner paused mid-section (resume later) vs finished it


def _meldos_http_error(exc: meldos.MeldOSError) -> HTTPException:
    """Map a MeldOS failure onto a response.

    401/403 are the SERVER's credentials being wrong, not the learner's, so they
    surface as 503 rather than being reflected back as an auth failure the
    learner could act on. 429 is passed through with Retry-After so a client can
    back off. Nothing here carries the application key or a user token — the
    exception detail is built in meldos.py precisely so it cannot.
    """
    if exc.status == 429:
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.detail, headers=headers
        )
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail)


def _doc_in_workspace(db: Session, document_id: int, ws_id: int) -> models.Document:
    doc = db.get(models.Document, document_id)
    if doc is None or doc.workspace_id != ws_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


def _instructions_for(
    db: Session,
    *,
    user: models.User,
    doc: models.Document,
    idx: int,
    resumed: bool = False,
    advancing: bool = False,
    with_recap: bool = True,
) -> tuple[str, list[models.Module]]:
    modules = plan_service.get_modules(db, doc.id)
    cur = modules[idx] if modules and 0 <= idx < len(modules) else None
    recap = ""
    if with_recap and not advancing:
        recap = memory.recap_for_tutor(
            workspace_id=user.workspace_id,
            user_id=user.id,
            document_id=doc.id,
            doc_name=doc.name,
        )
    text = tutor.build_instructions(
        doc_name=doc.name,
        sections=[plan_service.module_payload(m) for m in modules],
        idx=idx,
        material=tutor.section_material(plan_service.section_chunks(doc, cur)),
        recap=recap,
        resumed=resumed,
        advancing=advancing,
    )
    return text, modules


@router.post("/start")
def start_session(
    body: StartIn,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    # Validate the request before reporting server state, so a request for someone
    # else's document is a 404 whether or not voice happens to be configured.
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    if not settings.voice_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Voice teaching needs LIVEKIT_URL/API_KEY/API_SECRET, DEEPGRAM_API_KEY and "
                "CARTESIA_API_KEY on the server."
            ),
        )
    modules = plan_service.get_modules(db, doc.id)
    total = len(modules)

    if body.restart:
        # Re-learn from the start. Past bests are DELIBERATELY kept — a learner
        # revisiting a document should be able to raise their score, not be forced
        # to re-earn ground they already demonstrated.
        for p in db.scalars(
            select(models.SectionProgress).where(
                models.SectionProgress.user_id == user.id,
                models.SectionProgress.document_id == doc.id,
            )
        ).all():
            p.status = "in_progress"
        item = db.scalar(
            select(models.LearningPathItem).where(
                models.LearningPathItem.user_id == user.id,
                models.LearningPathItem.document_id == doc.id,
            )
        )
        if item is not None and item.status == "mastered":
            item.status = "in_progress"
        db.flush()
        idx = 0
    elif body.moduleIdx is not None:
        idx = body.moduleIdx
    else:
        idx = scoring.next_section_idx(db, user.id, doc.id, total)
    idx = max(0, min(idx, max(0, total - 1)))

    prog = db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user.id,
            models.SectionProgress.document_id == doc.id,
            models.SectionProgress.module_idx == idx,
        )
    )
    resumed = bool(prog is not None and prog.status == "paused")

    nonce = secrets.token_hex(4)
    room = voice.room_name(
        user_id=user.id, document_id=doc.id, module_idx=idx, session_nonce=nonce
    )
    token = voice.mint_join_token(
        room=room,
        identity=f"learner-{user.id}",
        name=user.name,
        metadata=voice.room_metadata(
            workspace_id=user.workspace_id,
            user_id=user.id,
            document_id=doc.id,
            module_idx=idx,
        ),
    )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LiveKit is not configured."
        )

    if prog is None:
        db.add(
            models.SectionProgress(
                user_id=user.id, document_id=doc.id, module_idx=idx, status="in_progress"
            )
        )
    else:
        prog.status = "in_progress"
    db.commit()

    return {
        "document": {"id": doc.id, "name": doc.name},
        "livekitUrl": settings.LIVEKIT_URL,
        "room": room,
        "token": token,
        "moduleIdx": idx,
        "moduleTitle": modules[idx].title if modules else None,
        "totalModules": total,
        "isLast": idx >= total - 1 if total else True,
        "resumed": resumed,
        "returning": bool(scoring.section_bests(db, user.id, doc.id)),
    }


class SectionIn(BaseModel):
    documentId: int
    moduleIdx: int


@router.post("/section")
def section_instructions(
    body: SectionIn,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    """Instructions for advancing to a section MID-session — sent to the agent so
    it moves on without reconnecting (keeps context, no re-introduction)."""
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    modules = plan_service.get_modules(db, doc.id)
    idx = max(0, min(body.moduleIdx, max(0, len(modules) - 1)))
    text, _ = _instructions_for(db, user=user, doc=doc, idx=idx, advancing=True, with_recap=False)
    cur = modules[idx] if modules else None
    return {
        "moduleIdx": idx,
        "moduleTitle": cur.title if cur else None,
        "totalModules": len(modules),
        "isLast": idx >= len(modules) - 1 if modules else True,
        "instructions": text,
    }


def _grade(
    db: Session,
    *,
    user: models.User,
    doc: models.Document,
    module_idx: int,
    transcript: list[dict],
    end_user: Optional[llm.EndUser] = None,
) -> Optional[dict]:
    """Run the assessor with the section's plan and source text as ground truth.
    Grading a transcript without them — which is what used to happen — leaves the
    model guessing whether an answer is right, and produces the clustered 41/65/70
    scores seen in the history."""
    modules = plan_service.get_modules(db, doc.id)
    cur = modules[module_idx] if modules and 0 <= module_idx < len(modules) else None
    section = None
    if cur is not None:
        section = plan_service.module_payload(cur)
        section["material"] = tutor.section_material(plan_service.section_chunks(doc, cur))
    prior = memory.prior_understanding(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        topic=(cur.title if cur else doc.name),
    )
    return ai.score_understanding(
        doc.name,
        transcript,
        section=section,
        prior_facts=prior,
        end_user=end_user,
        # One sitting = one MeldOS session, so a learner's grading calls for a
        # section group together in the gateway's ledger.
        session_id=f"praxos-u{user.id}-d{doc.id}-s{module_idx}",
    )


@router.post("/score")
def score_session(
    body: ScoreIn,
    user: models.User = Depends(active_membership),
    token: Optional[str] = Depends(bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    transcript = [{"role": t.role, "text": t.text} for t in body.transcript if t.text.strip()]
    modules = plan_service.get_modules(db, doc.id)
    total = len(modules)

    prog_idx = db.scalar(
        select(models.SectionProgress.module_idx)
        .where(
            models.SectionProgress.user_id == user.id,
            models.SectionProgress.document_id == doc.id,
        )
        .order_by(models.SectionProgress.updated_at.desc())
    )
    module_idx = body.moduleIdx if body.moduleIdx is not None else int(prog_idx or 0)
    module_idx = max(0, min(module_idx, max(0, total - 1)))

    # The learner is signed in and this is their own request, so forward their
    # token for VERIFIED attribution. meldos.py refuses to attach it to anything
    # other than the MeldOS host.
    graded = True
    try:
        result = _grade(
            db,
            user=user,
            doc=doc,
            module_idx=module_idx,
            transcript=transcript,
            end_user=llm.EndUser.verified(token),
        )
    except meldos.MeldOSError as exc:
        # Do NOT throw the conversation away. Returning an error here meant a
        # grader timeout destroyed a finished session outright: the learner spoke
        # for ten minutes and nothing was recorded, with no way to recover it.
        # The transcript is the valuable part — store it unscored so the sitting
        # can be re-graded (scripts/regrade.py) once the provider recovers.
        logger.warning("grading failed (MeldOS %s); recording the sitting unscored", exc.status)
        graded = False
        result = {
            "scoreable": False,
            "score": None,
            "covered": 0,
            "summary": "Not graded yet — the assessor was unavailable. This session is saved and will be re-graded.",
            "topics": [],
            "strengths": [],
            "gaps": [],
        }
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring needs an AI provider on the server (set LLM_API_KEY).",
        )

    session_row = scoring.apply_session(
        db,
        user=user,
        document=doc,
        module_idx=module_idx,
        transcript=transcript,
        result=result,
        paused=body.paused,
        total_sections=total,
    )
    db.commit()
    db.refresh(session_row)

    # Persist to memory so the next sitting can recall it. Best-effort.
    summary_bits = [str(result.get("summary") or "")]
    if result.get("strengths"):
        summary_bits.append("Demonstrated: " + "; ".join(map(str, result["strengths"])))
    if result.get("gaps"):
        summary_bits.append("Still to revisit: " + "; ".join(map(str, result["gaps"])))
    memory.write_session(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        session_id=session_row.id,
        module_idx=module_idx,
        transcript=transcript,
        summary=" ".join(b for b in summary_bits if b.strip()) or None,
    )

    doc_score = scoring.document_understanding(db, user.id, doc.id)
    completion = scoring.document_completion(db, user.id, doc.id)
    _maybe_notify(db, user=user, doc=doc, doc_score=doc_score, completion=completion)

    return {
        # This sitting.
        "score": session_row.score,
        "scoreable": session_row.score is not None,
        # False = we could not reach the assessor, NOT "you said nothing".
        # The UI must not tell a learner who spoke at length that there was
        # nothing to assess.
        "graded": graded,
        "summary": result.get("summary", ""),
        "topics": result.get("topics", []),
        "strengths": result.get("strengths", []),
        "gaps": result.get("gaps", []),
        # Where the learner now stands — the number the admin sees, not the last sitting.
        "understanding": doc_score,
        "band": scoring.band(doc_score),
        "documentUnderstanding": doc_score,
        "completion": completion,
        "sectionBests": scoring.section_bests(db, user.id, doc.id),
        "moduleIdx": module_idx,
        "totalModules": total,
        "courseComplete": completion >= 100,
        "paused": body.paused,
    }


def _maybe_notify(
    db: Session,
    *,
    user: models.User,
    doc: models.Document,
    doc_score: Optional[int],
    completion: int,
) -> None:
    """Opt-in Poke nudges for the admin. Poke can't run the lesson, but it is a
    fine place to land 'this learner needs help'."""
    if not (settings.POKE_NOTIFY_AT_RISK and poke.enabled() and doc_score is not None):
        return
    ws = db.get(models.Workspace, user.workspace_id)
    ws_name = ws.name if ws else f"workspace {user.workspace_id}"
    if completion >= 100:
        poke.notify_document_complete(
            learner=user.name, document=doc.name, score=doc_score, workspace=ws_name
        )
    elif doc_score < settings.AT_RISK_THRESHOLD:
        poke.notify_at_risk(
            learner=user.name, document=doc.name, score=doc_score, workspace=ws_name
        )


# ---------------------------------------------------------------------------
# Agent-worker routes. Authenticated with the shared secret, not a learner JWT.
# ---------------------------------------------------------------------------


def _agent_auth(x_agent_secret: str = Header(default="")) -> None:
    expected = settings.AGENT_SHARED_SECRET
    if not expected or not secrets.compare_digest(x_agent_secret, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad agent secret")


class AgentContextIn(BaseModel):
    workspaceId: int
    userId: int
    documentId: int
    moduleIdx: int = 0
    advancing: bool = False


@router.post("/agent/context", dependencies=[Depends(_agent_auth)])
def agent_context(body: AgentContextIn, db: Session = Depends(get_db)) -> dict:
    """Everything the worker needs to teach one section: grounded instructions,
    the section plan, and the advancement tool schema."""
    user = db.get(models.User, body.userId)
    doc = db.get(models.Document, body.documentId)
    if user is None or doc is None or doc.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session")
    modules = plan_service.get_modules(db, doc.id)
    total = len(modules)
    # Advancing PAST the last section must be reported, not clamped. Clamping
    # silently returned the final section again, so the worker re-taught the
    # section the learner had just finished and the UI looked stuck on it.
    if total and body.moduleIdx >= total:
        return {"complete": True, "moduleIdx": total - 1, "totalModules": total}
    idx = max(0, min(body.moduleIdx, max(0, total - 1)))
    prog = db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user.id,
            models.SectionProgress.document_id == doc.id,
            models.SectionProgress.module_idx == idx,
        )
    )
    text, _ = _instructions_for(
        db,
        user=user,
        doc=doc,
        idx=idx,
        resumed=bool(prog is not None and prog.status == "paused"),
        advancing=body.advancing,
    )
    return {
        "complete": False,
        "instructions": text,
        "learnerName": user.name,
        "document": {"id": doc.id, "name": doc.name},
        "moduleIdx": idx,
        "moduleTitle": modules[idx].title if modules else None,
        "totalModules": len(modules),
        "isLast": idx >= len(modules) - 1 if modules else True,
        "tool": tutor.ADVANCE_TOOL,
        "tts": {"model": settings.CARTESIA_MODEL, "voice": settings.CARTESIA_VOICE},
        "stt": {"model": settings.DEEPGRAM_MODEL, "language": settings.DEEPGRAM_LANGUAGE},
    }


class AgentScoreIn(BaseModel):
    workspaceId: int
    userId: int
    documentId: int
    moduleIdx: int = 0
    transcript: list[Turn] = []
    paused: bool = False


@router.post("/agent/score", dependencies=[Depends(_agent_auth)])
def agent_score(body: AgentScoreIn, db: Session = Depends(get_db)) -> dict:
    """The worker posts the finished sitting here if the browser never did (tab
    closed, network dropped) — so a real conversation is never lost unscored."""
    user = db.get(models.User, body.userId)
    doc = db.get(models.Document, body.documentId)
    if user is None or doc is None or doc.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session")
    transcript = [{"role": t.role, "text": t.text} for t in body.transcript if t.text.strip()]
    if not transcript:
        return {"recorded": False}

    modules = plan_service.get_modules(db, doc.id)
    idx = max(0, min(body.moduleIdx, max(0, len(modules) - 1)))
    # The agent worker authenticates with a service secret and never holds the
    # learner's token, so attribution here is CLAIMED, by name.
    try:
        result = _grade(
            db,
            user=user,
            doc=doc,
            module_idx=idx,
            transcript=transcript,
            end_user=llm.EndUser.claimed(user.name),
        )
    except meldos.MeldOSError as exc:
        raise _meldos_http_error(exc) from None
    if result is None:
        return {"recorded": False, "reason": "no AI provider configured"}
    row = scoring.apply_session(
        db,
        user=user,
        document=doc,
        module_idx=idx,
        transcript=transcript,
        result=result,
        paused=body.paused,
        total_sections=len(modules),
    )
    db.commit()
    db.refresh(row)
    memory.write_session(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        session_id=row.id,
        module_idx=idx,
        transcript=transcript,
        summary=str(result.get("summary") or "") or None,
    )
    return {"recorded": True, "sessionId": row.id, "score": row.score}
