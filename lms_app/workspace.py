from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models

ZERO_KPIS = [
    {"label": "Avg understanding", "value": "0", "hint": "no sessions yet"},
    {"label": "Active learners", "value": "1", "hint": "just you"},
    {"label": "Completion", "value": "0%", "hint": "no documents yet"},
    {"label": "At risk", "value": "0", "hint": "all clear"},
    {"label": "Sessions today", "value": "0", "hint": "none yet"},
]


def first_name(name: Optional[str]) -> str:
    return (name or "there").strip().split(" ")[0] or "there"


def resolve_user(db: Session, sub: str, name: Optional[str], email: Optional[str]) -> models.User:
    """Find the user by Clerk id, or create them on first login.

    On first login we check for a pending invite matching their email: if found,
    they JOIN that workspace with the invited role; otherwise they get a fresh
    personal workspace as its Admin.
    """
    user = db.scalar(select(models.User).where(models.User.clerk_id == sub))
    if user is not None:
        if name and user.name != name:
            user.name = name
        if email and user.email != email:
            user.email = email
        db.commit()
        return user

    invite = None
    if email:
        invite = db.scalar(
            select(models.Invite).where(
                func.lower(models.Invite.email) == email.lower(),
                models.Invite.status == "pending",
            )
        )

    if invite is not None:
        ws_id = invite.workspace_id
        role = invite.role
        invite.status = "accepted"
    else:
        ws = models.Workspace(name=f"{first_name(name)}'s workspace", plan="Personal workspace")
        db.add(ws)
        db.flush()
        ws_id = ws.id
        role = "Admin"

    user = models.User(
        clerk_id=sub,
        workspace_id=ws_id,
        name=name or "New user",
        email=email or f"{sub}@clerk.local",
        role=role,
        cohort="-",
        documents=0,
        understanding=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def is_admin(user: models.User) -> bool:
    return user.role in ("Admin", "Owner")


def _people(db: Session, ws_id: int) -> list[dict]:
    users = db.scalars(
        select(models.User).where(models.User.workspace_id == ws_id).order_by(models.User.id)
    ).all()
    return [
        {
            "name": u.name,
            "email": u.email,
            "cohort": u.cohort,
            "documents": u.documents,
            "understanding": u.understanding,
            "role": u.role,
            "id": u.id,
        }
        for u in users
    ]


def _pending(db: Session, ws_id: int) -> list[dict]:
    invites = db.scalars(
        select(models.Invite)
        .where(models.Invite.workspace_id == ws_id, models.Invite.status == "pending")
        .order_by(models.Invite.id)
    ).all()
    return [{"id": i.id, "email": i.email, "role": i.role} for i in invites]


def _cohort_member_ids(db: Session, cohort_id: int) -> list[int]:
    return list(
        db.scalars(
            select(models.CohortMember.user_id).where(models.CohortMember.cohort_id == cohort_id)
        ).all()
    )


def _cohort_doc_ids(db: Session, cohort_id: int) -> list[int]:
    return list(
        db.scalars(
            select(models.CohortDocument.document_id)
            .where(models.CohortDocument.cohort_id == cohort_id)
            .order_by(models.CohortDocument.idx)
        ).all()
    )


def cohort_detail(db: Session, c: models.Cohort) -> dict:
    """Full cohort shape for the admin UI (members + ordered documents + status)."""
    member_ids = _cohort_member_ids(db, c.id)
    doc_ids = _cohort_doc_ids(db, c.id)
    docs = [
        {"id": d.id, "name": d.name}
        for d in (db.get(models.Document, did) for did in doc_ids)
        if d is not None
    ]
    return {
        "id": c.id,
        "name": c.name,
        "members": len(member_ids),
        "memberIds": member_ids,
        "documentIds": doc_ids,
        "documents": docs,
        "status": c.status,
        "published": c.published,
        "avg": c.avg,
        "completion": c.completion,
    }


def _cohorts(db: Session, ws_id: int) -> list[dict]:
    rows = db.scalars(
        select(models.Cohort).where(models.Cohort.workspace_id == ws_id).order_by(models.Cohort.id)
    ).all()
    return [cohort_detail(db, c) for c in rows]


def _documents(db: Session, ws_id: int) -> list[dict]:
    docs = db.scalars(
        select(models.Document).where(models.Document.workspace_id == ws_id).order_by(models.Document.id.desc())
    ).all()
    return [
        {"id": d.id, "name": d.name, "sections": d.sections, "assigned": d.assigned, "status": d.status}
        for d in docs
    ]


def is_owner(db: Session, user: models.User) -> bool:
    """The owner is the earliest member of the workspace (its creator)."""
    first_id = db.scalar(
        select(func.min(models.User.id)).where(models.User.workspace_id == user.workspace_id)
    )
    return user.id == first_id


def build_bundle(db: Session, user: models.User, display_name: str) -> dict:
    ws = db.get(models.Workspace, user.workspace_id)
    needs_onboarding = (not ws.onboarded) and is_owner(db, user)
    return {
        "mode": "user",
        "needsOnboarding": needs_onboarding,
        "workspace": {"name": ws.name, "plan": ws.plan},
        "account": {"name": display_name, "email": user.email, "role": "Workspace owner" if is_admin(user) else user.role},
        "role": user.role,
        "learner": {
            "name": display_name,
            "firstName": first_name(display_name),
            "understanding": user.understanding,
            "pathProgress": "0 / 0",
            "practisedThisWeek": "0m",
            "sessions": 0,
            "streak": 0,
        },
        "continueLearning": None,
        "learningPath": [],
        "pastSessions": [],
        "myDocuments": [],
        "admin": {
            "kpis": ZERO_KPIS,
            "understandingTrend": [],
            "cohortHealth": [],
            "needsAttention": [],
            "recentActivity": [],
            "cohorts": _cohorts(db, user.workspace_id),
            "people": _people(db, user.workspace_id),
            "pendingInvites": _pending(db, user.workspace_id),
            "teams": [],
            "documents": _documents(db, user.workspace_id),
        },
    }
