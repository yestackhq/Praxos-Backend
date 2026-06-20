from __future__ import annotations

"""Team management (admin only). A team is a group of learners (with an optional
lead) assigned a curriculum of documents — published into members' memory exactly
like a cohort. Reuses the cohort helpers so the two stay in lockstep."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .. import memory, models, plan as plan_service, workspace
from ..auth import optional_claims
from ..db import get_db
from .cohorts import _admin, _draft_plans, _seed_path, _seed_progress, _valid_doc_ids, _valid_member_ids

router = APIRouter(prefix="/api", tags=["teams"])


class TeamIn(BaseModel):
    name: str
    lead: str = ""
    documentIds: list[int] = []
    memberUserIds: list[int] = []


class TeamPatch(BaseModel):
    name: Optional[str] = None
    lead: Optional[str] = None
    documentIds: Optional[list[int]] = None
    memberUserIds: Optional[list[int]] = None


def _get_team(db: Session, tid: int, ws_id: int) -> models.Team:
    t = db.get(models.Team, tid)
    if t is None or t.workspace_id != ws_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Team not found")
    return t


def _set_documents(db: Session, team: models.Team, doc_ids: list[int]) -> None:
    db.execute(delete(models.TeamDocument).where(models.TeamDocument.team_id == team.id))
    for idx, did in enumerate(doc_ids):
        db.add(models.TeamDocument(team_id=team.id, document_id=did, idx=idx))
    team.paths = len(doc_ids)


def _set_members(db: Session, team: models.Team, member_ids: list[int]) -> None:
    db.execute(delete(models.TeamMember).where(models.TeamMember.team_id == team.id))
    for uid in member_ids:
        db.add(models.TeamMember(team_id=team.id, user_id=uid))
    team.members = len(member_ids)


@router.post("/teams", status_code=status.HTTP_201_CREATED)
def create_team(body: TeamIn, claims=Depends(optional_claims), db: Session = Depends(get_db)) -> dict:
    user = _admin(claims, db)
    t = models.Team(
        workspace_id=user.workspace_id,
        name=(body.name or "").strip() or "Untitled team",
        lead=(body.lead or "").strip(),
        published=False,
    )
    db.add(t)
    db.flush()
    doc_ids = _valid_doc_ids(db, user.workspace_id, body.documentIds)
    _set_documents(db, t, doc_ids)
    _set_members(db, t, _valid_member_ids(db, user.workspace_id, body.memberUserIds))
    db.commit()
    _draft_plans(db, doc_ids)
    db.refresh(t)
    return workspace.team_detail(db, t)


@router.patch("/teams/{tid}")
def edit_team(tid: int, body: TeamPatch, claims=Depends(optional_claims), db: Session = Depends(get_db)) -> dict:
    user = _admin(claims, db)
    t = _get_team(db, tid, user.workspace_id)
    if body.name is not None and body.name.strip():
        t.name = body.name.strip()
    if body.lead is not None:
        t.lead = body.lead.strip()
    new_docs: list[int] = []
    if body.documentIds is not None:
        new_docs = _valid_doc_ids(db, user.workspace_id, body.documentIds)
        _set_documents(db, t, new_docs)
    if body.memberUserIds is not None:
        _set_members(db, t, _valid_member_ids(db, user.workspace_id, body.memberUserIds))
    db.commit()
    _draft_plans(db, new_docs)
    db.refresh(t)
    return workspace.team_detail(db, t)


@router.delete("/teams/{tid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team(tid: int, claims=Depends(optional_claims), db: Session = Depends(get_db)) -> None:
    user = _admin(claims, db)
    t = _get_team(db, tid, user.workspace_id)
    db.execute(delete(models.TeamDocument).where(models.TeamDocument.team_id == t.id))
    db.execute(delete(models.TeamMember).where(models.TeamMember.team_id == t.id))
    db.delete(t)
    db.commit()


@router.post("/teams/{tid}/publish")
def publish_team(tid: int, claims=Depends(optional_claims), db: Session = Depends(get_db)) -> dict:
    user = _admin(claims, db)
    t = _get_team(db, tid, user.workspace_id)
    doc_ids = workspace._team_doc_ids(db, t.id)
    member_ids = workspace._team_member_ids(db, t.id)
    if not doc_ids or not member_ids:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Add at least one document and one learner before publishing."
        )
    for did in doc_ids:
        mods = plan_service.ensure_plan(db, did)
        doc = db.get(models.Document, did)
        if doc is None:
            continue
        payload = [
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
                    modules=payload,
                )
            except Exception:
                pass
            _seed_path(db, uid, doc, len(mods))
            _seed_progress(db, uid, did)
        doc.assigned = max(doc.assigned, len(member_ids))
    t.published = True
    db.commit()
    db.refresh(t)
    return workspace.team_detail(db, t)
