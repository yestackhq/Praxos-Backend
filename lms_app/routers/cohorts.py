from __future__ import annotations

"""Cohort management + teaching-plan endpoints (admin only).

A cohort is a group of learners assigned an ordered curriculum of documents.
Creating a cohort drafts an AI teaching plan per document; the admin can edit it,
then PUBLISH to push the plan + document context into every member's memory so the
voice tutor knows what and how to teach, section by section."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .. import llm, memory, meldos, models, plan as plan_service, scoring, workspace
from ..auth import active_membership, bearer_token
from ..db import get_db

router = APIRouter(prefix="/api", tags=["cohorts"])


def _admin(user: models.User = Depends(active_membership)) -> models.User:
    """Dependency: the caller's active-workspace membership, requiring admin there."""
    if not workspace.is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only admins can manage cohorts")
    return user


class CohortIn(BaseModel):
    name: str
    documentIds: list[int] = []
    memberUserIds: list[int] = []


class CohortPatch(BaseModel):
    name: Optional[str] = None
    documentIds: Optional[list[int]] = None
    memberUserIds: Optional[list[int]] = None


def _get_cohort(db: Session, cid: int, ws_id: int) -> models.Cohort:
    c = db.get(models.Cohort, cid)
    if c is None or c.workspace_id != ws_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cohort not found")
    return c


def _valid_doc_ids(db: Session, ws_id: int, ids: list[int]) -> list[int]:
    if not ids:
        return []
    valid = set(
        db.scalars(
            select(models.Document.id).where(
                models.Document.workspace_id == ws_id, models.Document.id.in_(ids)
            )
        ).all()
    )
    return [i for i in ids if i in valid]  # preserve caller order


def _valid_member_ids(db: Session, ws_id: int, ids: list[int]) -> list[int]:
    if not ids:
        return []
    valid = set(
        db.scalars(
            select(models.User.id).where(
                models.User.workspace_id == ws_id, models.User.id.in_(ids)
            )
        ).all()
    )
    return [i for i in ids if i in valid]


def _set_documents(db: Session, cohort: models.Cohort, doc_ids: list[int]) -> None:
    db.execute(delete(models.CohortDocument).where(models.CohortDocument.cohort_id == cohort.id))
    for idx, did in enumerate(doc_ids):
        db.add(models.CohortDocument(cohort_id=cohort.id, document_id=did, idx=idx))


def _set_members(db: Session, cohort: models.Cohort, member_ids: list[int], ws_id: int) -> None:
    db.execute(delete(models.CohortMember).where(models.CohortMember.cohort_id == cohort.id))
    for uid in member_ids:
        db.add(models.CohortMember(cohort_id=cohort.id, user_id=uid))


def _draft_plans(db: Session, doc_ids: list[int], end_user=None) -> None:
    for did in doc_ids:
        try:
            plan_service.ensure_plan(db, did, end_user=end_user)
        except Exception:  # plan generation is best-effort; never block cohort ops
            pass


@router.post("/cohorts", status_code=status.HTTP_201_CREATED)
def create_cohort(
    body: CohortIn,
    user: models.User = Depends(_admin),
    token: Optional[str] = Depends(bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    name = (body.name or "").strip() or "Untitled cohort"
    c = models.Cohort(workspace_id=user.workspace_id, name=name, published=False)
    db.add(c)
    db.flush()
    doc_ids = _valid_doc_ids(db, user.workspace_id, body.documentIds)
    member_ids = _valid_member_ids(db, user.workspace_id, body.memberUserIds)
    _set_documents(db, c, doc_ids)
    _set_members(db, c, member_ids, user.workspace_id)
    db.commit()
    _draft_plans(db, doc_ids, llm.EndUser.verified(token))
    db.refresh(c)
    return workspace.cohort_detail(db, c)


@router.patch("/cohorts/{cid}")
def edit_cohort(
    cid: int,
    body: CohortPatch,
    user: models.User = Depends(_admin),
    token: Optional[str] = Depends(bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    c = _get_cohort(db, cid, user.workspace_id)
    if body.name is not None and body.name.strip():
        c.name = body.name.strip()
    new_docs: list[int] = []
    if body.documentIds is not None:
        new_docs = _valid_doc_ids(db, user.workspace_id, body.documentIds)
        _set_documents(db, c, new_docs)
    if body.memberUserIds is not None:
        member_ids = _valid_member_ids(db, user.workspace_id, body.memberUserIds)
        _set_members(db, c, member_ids, user.workspace_id)
    db.commit()
    _draft_plans(db, new_docs, llm.EndUser.verified(token))
    db.refresh(c)
    return workspace.cohort_detail(db, c)


@router.delete("/cohorts/{cid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cohort(cid: int, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> None:
    c = _get_cohort(db, cid, user.workspace_id)
    db.execute(delete(models.CohortDocument).where(models.CohortDocument.cohort_id == c.id))
    db.execute(delete(models.CohortMember).where(models.CohortMember.cohort_id == c.id))
    db.delete(c)
    db.commit()
    db.commit()


@router.post("/cohorts/{cid}/publish")
def publish_cohort(
    cid: int,
    user: models.User = Depends(_admin),
    token: Optional[str] = Depends(bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    """Push each document's teaching plan + a learning-path seed into every
    member's memory, so the tutor opens already knowing what and how to teach."""
    c = _get_cohort(db, cid, user.workspace_id)
    doc_ids = workspace._cohort_doc_ids(db, c.id)
    member_ids = workspace._cohort_member_ids(db, c.id)
    if not doc_ids or not member_ids:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Add at least one document and one learner before publishing.",
        )
    # A cohort's FIRST publish (draft → published) is a fresh assignment: documents a learner has
    # already COMPLETED are re-opened so they re-learn them in this cohort. Re-publishing an
    # already-published cohort (e.g. after editing the plan or adding a doc) stays non-destructive,
    # so completed learners aren't forced to redo everything.
    fresh = not c.published
    for did in doc_ids:
        mods = plan_service.ensure_plan(db, did, end_user=llm.EndUser.verified(token))
        doc = db.get(models.Document, did)
        if doc is None:
            continue
        mod_payload = [plan_service.module_payload(m) for m in mods]
        for uid in member_ids:
            try:
                memory.write_lesson_plan(
                    workspace_id=user.workspace_id,
                    user_id=uid,
                    document_id=did,
                    doc_name=doc.name,
                    modules=mod_payload,
                )
            except Exception:  # memory push is best-effort
                pass
            item = db.scalar(
                select(models.LearningPathItem).where(
                    models.LearningPathItem.user_id == uid,
                    models.LearningPathItem.document_id == doc.id,
                )
            )
            # Re-open only a COMPLETED document, and only on a fresh publish — so in-progress work
            # and routine re-publishes are never wiped.
            reopen = fresh and item is not None and item.status == "mastered"
            _seed_path(db, uid, doc, reopen=reopen)
            _seed_progress(db, uid, did, reopen=reopen)
    c.published = True
    db.commit()
    db.refresh(c)
    return workspace.cohort_detail(db, c)


def _seed_path(db: Session, user_id: int, doc: models.Document, reopen: bool = False) -> None:
    """Add the document to a learner's learning path (idempotent by document id —
    it used to key off the document NAME, so a rename orphaned everyone's path)."""
    existing = db.scalar(
        select(models.LearningPathItem).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.document_id == doc.id,
        )
    )
    if existing is not None:
        if reopen:
            # Fresh publish re-opens a completed document for another pass. Past
            # section bests are kept, so the learner can only raise their score.
            existing.status = "up_next"
        # Otherwise keep their status — this is what makes re-publishing (after
        # adding a new doc) NON-destructive to what they've already learnt.
        return
    count = (
        db.scalar(
            select(func.count())
            .select_from(models.LearningPathItem)
            .where(models.LearningPathItem.user_id == user_id)
        )
        or 0
    )
    # A newly-added document is immediately learnable when the learner has nothing
    # active; otherwise it's queued (locked) and unlocks when the current document
    # is mastered — so it's an EXPANSION, never a reset of in-flight progress.
    has_active = db.scalar(
        select(models.LearningPathItem.id).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.status.in_(["in_progress", "up_next"]),
        )
    )
    item_status = "up_next" if (count == 0 or has_active is None) else "locked"
    db.add(
        models.LearningPathItem(
            user_id=user_id, document_id=doc.id, idx=count, status=item_status
        )
    )


def _seed_progress(db: Session, user_id: int, document_id: int, reopen: bool = False) -> None:
    """Open the first section (idempotent). On a fresh publish of a completed
    document, re-open section 0 — WITHOUT clearing best scores, so re-learning can
    only raise a learner's standing, never reset it to zero."""
    row = db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user_id,
            models.SectionProgress.document_id == document_id,
            models.SectionProgress.module_idx == 0,
        )
    )
    if row is None:
        db.add(
            models.SectionProgress(
                user_id=user_id, document_id=document_id, module_idx=0, status="in_progress"
            )
        )
    elif reopen:
        row.status = "in_progress"


# ---------------------------------------------------------------------------
# Teaching-plan endpoints (per document)
# ---------------------------------------------------------------------------


def _mod_out(m: models.Module) -> dict:
    return {
        "id": m.id,
        "idx": m.idx,
        "title": m.title,
        "description": m.description,
        "topics": list(m.topics or []),
        # What the learner must be able to state, and how the tutor checks it.
        # The grader marks answers against key_points, so an admin editing these
        # is editing the rubric too.
        "keyPoints": list(m.key_points or []),
        "checkQuestions": list(m.check_questions or []),
        "minutes": m.minutes,
        "chunkStart": m.chunk_start,
        "chunkEnd": m.chunk_end,
    }


def _own_doc(db: Session, did: int, ws_id: int) -> models.Document:
    doc = db.get(models.Document, did)
    if doc is None or doc.workspace_id != ws_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.get("/documents/{did}/plan")
def get_plan(did: int, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    doc = _own_doc(db, did, user.workspace_id)
    mods = plan_service.get_modules(db, did)
    return {
        "document": {"id": doc.id, "name": doc.name},
        "modules": [_mod_out(m) for m in mods],
        "coverage": plan_service.plan_coverage(db, did),
    }


@router.post("/documents/{did}/plan/generate")
def regenerate_plan(
    did: int,
    user: models.User = Depends(_admin),
    token: Optional[str] = Depends(bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    doc = _own_doc(db, did, user.workspace_id)
    try:
        mods = plan_service.generate_plan(db, did, end_user=llm.EndUser.verified(token))
    except meldos.MeldOSError as exc:
        from .sessions import _meldos_http_error

        raise _meldos_http_error(exc) from None
    return {
        "document": {"id": doc.id, "name": doc.name},
        "modules": [_mod_out(m) for m in mods],
        "coverage": plan_service.plan_coverage(db, did),
    }


class ModuleIn(BaseModel):
    title: str
    description: str = ""
    topics: list[str] = []
    keyPoints: list[str] = []
    checkQuestions: list[str] = []
    minutes: int = 5


class PlanPatch(BaseModel):
    modules: list[ModuleIn]


@router.patch("/documents/{did}/plan")
def save_plan(did: int, body: PlanPatch, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    """Save the admin's edited plan. Chunk ranges are carried over by position so
    section-by-section grounding survives a title/topic edit."""
    doc = _own_doc(db, did, user.workspace_id)
    old = plan_service.get_modules(db, did)
    db.execute(delete(models.Module).where(models.Module.document_id == did))
    for i, m in enumerate(body.modules):
        cs = old[i].chunk_start if i < len(old) else 0
        ce = old[i].chunk_end if i < len(old) else 0
        db.add(
            models.Module(
                document_id=did,
                idx=i,
                title=m.title[:160],
                description=m.description[:2000],
                topics=[str(t)[:80] for t in m.topics][:6],
                key_points=[str(t)[:400] for t in m.keyPoints][:6]
                or (list(old[i].key_points or []) if i < len(old) else []),
                check_questions=[str(t)[:300] for t in m.checkQuestions][:5]
                or (list(old[i].check_questions or []) if i < len(old) else []),
                minutes=max(2, min(20, m.minutes or 5)),
                chunk_start=cs,
                chunk_end=ce,
            )
        )
    db.commit()
    mods = plan_service.get_modules(db, did)
    return {
        "document": {"id": doc.id, "name": doc.name},
        "modules": [_mod_out(m) for m in mods],
        "coverage": plan_service.plan_coverage(db, did),
    }


@router.get("/people/{uid}")
def person_detail(uid: int, admin: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    """Full detail for one learner — drives the Understanding row sidebar."""
    u = db.get(models.User, uid)
    if u is None or u.workspace_id != admin.workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Person not found")
    understanding = scoring.user_understanding(db, u.id)
    sessions = db.scalars(
        select(models.LearningSession)
        .where(models.LearningSession.user_id == u.id)
        .order_by(models.LearningSession.id.desc())
        .limit(12)
    ).all()
    path = db.scalars(
        select(models.LearningPathItem)
        .where(models.LearningPathItem.user_id == u.id)
        .order_by(models.LearningPathItem.idx)
    ).all()

    def _doc_name(did: int) -> str:
        d = db.get(models.Document, did)
        return workspace.clean_name(d.name) if d else ""

    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "cohort": workspace._user_cohort_map(db, admin.workspace_id).get(u.id, ""),
        "team": workspace._user_team_map(db, admin.workspace_id).get(u.id, ""),
        "understanding": understanding,
        "band": scoring.band(understanding),
        "documents": len(scoring.started_document_ids(db, u.id)),
        "sessions": [
            {
                "id": s.id,
                "doc": _doc_name(s.document_id),
                "section": s.module_idx + 1,
                "date": s.started_at.date().isoformat() if s.started_at else "",
                "score": s.score,
                "summary": s.summary,
                "turns": s.learner_turns,
            }
            for s in sessions
        ],
        "path": [
            {
                "docId": i.document_id,
                "title": _doc_name(i.document_id),
                "status": i.status,
                "progress": scoring.document_completion(db, u.id, i.document_id),
                "understanding": scoring.document_understanding(db, u.id, i.document_id),
            }
            for i in path
        ],
    }


@router.get("/people/{uid}/sessions/{sid}")
def session_transcript(
    uid: int, sid: int, admin: models.User = Depends(_admin), db: Session = Depends(get_db)
) -> dict:
    """The full transcript and per-key-point evidence behind one score.

    Sittings used to be graded and thrown away, so neither an admin nor an
    engineer could see WHY a learner was given a number. They can now."""
    u = db.get(models.User, uid)
    row = db.get(models.LearningSession, sid)
    if u is None or u.workspace_id != admin.workspace_id or row is None or row.user_id != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    doc = db.get(models.Document, row.document_id)
    return {
        "id": row.id,
        "doc": workspace.clean_name(doc.name) if doc else "",
        "section": row.module_idx + 1,
        "date": row.started_at.isoformat() if row.started_at else "",
        "score": row.score,
        "band": scoring.band(row.score),
        "covered": row.covered,
        "summary": row.summary,
        "topics": row.topics,
        "strengths": row.strengths,
        "gaps": row.gaps,
        "transcript": row.transcript,
    }
