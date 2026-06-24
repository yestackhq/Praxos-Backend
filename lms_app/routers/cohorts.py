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

from .. import memory, models, plan as plan_service, workspace
from ..auth import active_membership
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
    cohort.members = len(member_ids)


def _resync_member_labels(db: Session, ws_id: int) -> None:
    """Recompute each user's denormalized ``cohort`` label (shown in the People
    table) from their current memberships."""
    db.flush()  # autoflush is off — make pending membership changes visible
    rows = db.execute(
        select(models.CohortMember.user_id, models.Cohort.name)
        .join(models.Cohort, models.Cohort.id == models.CohortMember.cohort_id)
        .where(models.Cohort.workspace_id == ws_id)
        .order_by(models.Cohort.id)
    ).all()
    label: dict[int, str] = {}
    for uid, name in rows:
        label.setdefault(uid, name)
    for u in db.scalars(select(models.User).where(models.User.workspace_id == ws_id)).all():
        u.cohort = label.get(u.id, "—")


def _draft_plans(db: Session, doc_ids: list[int]) -> None:
    for did in doc_ids:
        try:
            plan_service.ensure_plan(db, did)
        except Exception:  # plan generation is best-effort; never block cohort ops
            pass


@router.post("/cohorts", status_code=status.HTTP_201_CREATED)
def create_cohort(body: CohortIn, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    name = (body.name or "").strip() or "Untitled cohort"
    c = models.Cohort(workspace_id=user.workspace_id, name=name, status="Draft", published=False)
    db.add(c)
    db.flush()
    doc_ids = _valid_doc_ids(db, user.workspace_id, body.documentIds)
    member_ids = _valid_member_ids(db, user.workspace_id, body.memberUserIds)
    _set_documents(db, c, doc_ids)
    _set_members(db, c, member_ids, user.workspace_id)
    _resync_member_labels(db, user.workspace_id)
    db.commit()
    _draft_plans(db, doc_ids)
    db.refresh(c)
    return workspace.cohort_detail(db, c)


@router.patch("/cohorts/{cid}")
def edit_cohort(cid: int, body: CohortPatch, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
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
    _resync_member_labels(db, user.workspace_id)
    db.commit()
    _draft_plans(db, new_docs)
    db.refresh(c)
    return workspace.cohort_detail(db, c)


@router.delete("/cohorts/{cid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cohort(cid: int, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> None:
    c = _get_cohort(db, cid, user.workspace_id)
    db.execute(delete(models.CohortDocument).where(models.CohortDocument.cohort_id == c.id))
    db.execute(delete(models.CohortMember).where(models.CohortMember.cohort_id == c.id))
    db.delete(c)
    db.commit()
    _resync_member_labels(db, user.workspace_id)
    db.commit()


@router.post("/cohorts/{cid}/publish")
def publish_cohort(cid: int, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
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
        mods = plan_service.ensure_plan(db, did)
        doc = db.get(models.Document, did)
        if doc is None:
            continue
        mod_payload = [
            {"idx": m.idx, "title": m.title, "description": m.description, "topics": m.topics}
            for m in mods
        ]
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
                    models.LearningPathItem.title == doc.name,
                )
            )
            # Re-open only a COMPLETED document, and only on a fresh publish — so in-progress work
            # and routine re-publishes are never wiped.
            reopen = fresh and item is not None and item.status == "mastered"
            _seed_path(db, uid, doc, len(mods), reopen=reopen)
            _seed_progress(db, uid, did, reopen=reopen)
        doc.assigned = len(member_ids)
    c.published = True
    c.status = "On track"
    db.commit()
    db.refresh(c)
    return workspace.cohort_detail(db, c)


def _seed_path(db: Session, user_id: int, doc: models.Document, sections: int, reopen: bool = False) -> None:
    """Add the document to a learner's learning path (idempotent by title)."""
    existing = db.scalar(
        select(models.LearningPathItem).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.title == doc.name,
        )
    )
    if existing is not None:
        existing.sections = sections
        if reopen:
            # Fresh publish re-opens a completed document for another pass (past scores are kept).
            existing.status = "up_next"
            existing.progress = 0
        # Otherwise keep their status/progress — this is what makes re-publishing (after adding a
        # new doc) NON-destructive to what they've already learnt.
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
    # active (e.g. finished their current docs); otherwise it's queued (locked) and
    # unlocks when the current document is mastered — so it's an EXPANSION, never a
    # reset of in-flight progress.
    has_active = db.scalar(
        select(models.LearningPathItem.id).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.status.in_(["in_progress", "up_next"]),
        )
    )
    status = "up_next" if (count == 0 or has_active is None) else "locked"
    db.add(
        models.LearningPathItem(
            user_id=user_id,
            idx=count,
            title=doc.name,
            sections=sections,
            status=status,
            progress=0,
        )
    )


def _seed_progress(db: Session, user_id: int, document_id: int, reopen: bool = False) -> None:
    """Seed section 0 as the resume point (idempotent). On a fresh publish of a completed
    document (``reopen``), reset the resume point back to the first section."""
    existing = db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user_id,
            models.SectionProgress.document_id == document_id,
        )
    )
    if existing is None:
        db.add(
            models.SectionProgress(
                user_id=user_id, document_id=document_id, module_idx=0, status="in_progress"
            )
        )
    elif reopen:
        existing.module_idx = 0
        existing.status = "in_progress"
        existing.score = None


# ---------------------------------------------------------------------------
# Teaching-plan endpoints (per document)
# ---------------------------------------------------------------------------


def _mod_out(m: models.Module) -> dict:
    return {
        "id": m.id,
        "idx": m.idx,
        "title": m.title,
        "description": m.description,
        "topics": m.topics,
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
    return {"document": {"id": doc.id, "name": doc.name}, "modules": [_mod_out(m) for m in mods]}


@router.post("/documents/{did}/plan/generate")
def regenerate_plan(did: int, user: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    doc = _own_doc(db, did, user.workspace_id)
    mods = plan_service.generate_plan(db, did)
    return {"document": {"id": doc.id, "name": doc.name}, "modules": [_mod_out(m) for m in mods]}


class ModuleIn(BaseModel):
    title: str
    description: str = ""
    topics: list[str] = []
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
                minutes=max(2, min(20, m.minutes or 5)),
                chunk_start=cs,
                chunk_end=ce,
                source=f"Section {i + 1} · taught by voice",
            )
        )
    db.commit()
    mods = plan_service.get_modules(db, did)
    return {"document": {"id": doc.id, "name": doc.name}, "modules": [_mod_out(m) for m in mods]}


@router.get("/people/{uid}")
def person_detail(uid: int, admin: models.User = Depends(_admin), db: Session = Depends(get_db)) -> dict:
    """Full detail for one learner — drives the Understanding row sidebar."""
    u = db.get(models.User, uid)
    if u is None or u.workspace_id != admin.workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Person not found")
    team = workspace._user_team_map(db, admin.workspace_id).get(u.id, "")
    sessions = db.scalars(
        select(models.LearningSession)
        .where(models.LearningSession.user_id == u.id)
        .order_by(models.LearningSession.id.desc())
    ).all()
    path = db.scalars(
        select(models.LearningPathItem)
        .where(models.LearningPathItem.user_id == u.id)
        .order_by(models.LearningPathItem.idx)
    ).all()
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "cohort": u.cohort if u.cohort and u.cohort != "—" else "",
        "team": team,
        "understanding": u.understanding,
        "documents": u.documents,
        "sessions": [
            {"doc": workspace.clean_name(s.doc), "date": s.date, "score": s.score, "duration": s.duration}
            for s in sessions[:12]
        ],
        "path": [
            {"title": workspace.clean_name(i.title), "status": i.status, "progress": i.progress}
            for i in path
        ],
    }
