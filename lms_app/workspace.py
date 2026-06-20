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


# ---- Learner-side data (real assignments + progress, not mock) ----


def _path_items(db: Session, user_id: int) -> list[models.LearningPathItem]:
    return list(
        db.scalars(
            select(models.LearningPathItem)
            .where(models.LearningPathItem.user_id == user_id)
            .order_by(models.LearningPathItem.idx)
        ).all()
    )


def _doc_by_name(db: Session, ws_id: int, name: str) -> Optional[models.Document]:
    return db.scalar(
        select(models.Document).where(
            models.Document.workspace_id == ws_id, models.Document.name == name
        )
    )


def _learning_path(db: Session, user: models.User) -> list[dict]:
    return [
        {"title": i.title, "sections": i.sections, "status": i.status, "progress": i.progress}
        for i in _path_items(db, user.id)
    ]


def _my_documents(db: Session, user: models.User) -> list[dict]:
    out: list[dict] = []
    for i in _path_items(db, user.id):
        doc = _doc_by_name(db, user.workspace_id, i.title)
        status = (
            "Mastered" if i.status == "mastered" else "Locked" if i.status == "locked" else "Assigned"
        )
        out.append(
            {
                "name": i.title,
                "pages": doc.sections if doc else i.sections,
                "status": status,
                "added": "",
                "docId": doc.id if doc else None,
            }
        )
    return out


def _past_sessions(db: Session, user: models.User) -> list[dict]:
    rows = db.scalars(
        select(models.LearningSession)
        .where(models.LearningSession.user_id == user.id)
        .order_by(models.LearningSession.id.desc())
    ).all()
    return [
        {"doc": s.doc, "date": s.date, "score": s.score, "duration": s.duration, "topics": s.topics}
        for s in rows[:10]
    ]


def _continue_learning(db: Session, user: models.User) -> Optional[dict]:
    item = db.scalar(
        select(models.LearningPathItem)
        .where(
            models.LearningPathItem.user_id == user.id,
            models.LearningPathItem.status.in_(["in_progress", "up_next"]),
        )
        .order_by(models.LearningPathItem.idx)
    )
    if item is None:
        return None
    doc = _doc_by_name(db, user.workspace_id, item.title)
    total = item.sections
    cur = 1
    if doc is not None:
        total = (
            db.scalar(
                select(func.count()).select_from(models.Module).where(models.Module.document_id == doc.id)
            )
            or item.sections
        )
        prog = db.scalar(
            select(models.SectionProgress).where(
                models.SectionProgress.user_id == user.id,
                models.SectionProgress.document_id == doc.id,
            )
        )
        cur = (prog.module_idx + 1) if prog else 1
    started = (item.progress or 0) > 0
    return {
        "doc": item.title,
        "position": f"Section {min(cur, total)} of {total}" if total else "Ready to start",
        "remaining": "Pick up where you left off." if started else "Start your first section.",
        "understanding": item.progress if item.progress is not None else user.understanding,
        "docId": doc.id if doc else None,
    }


def _learner_stats(db: Session, user: models.User) -> dict:
    items = _path_items(db, user.id)
    mastered = sum(1 for i in items if i.status == "mastered")
    sessions = (
        db.scalar(
            select(func.count())
            .select_from(models.LearningSession)
            .where(models.LearningSession.user_id == user.id)
        )
        or 0
    )
    return {"pathProgress": f"{mastered} / {len(items)}", "sessions": int(sessions)}


def build_bundle(db: Session, user: models.User, display_name: str) -> dict:
    ws = db.get(models.Workspace, user.workspace_id)
    needs_onboarding = (not ws.onboarded) and is_owner(db, user)
    stats = _learner_stats(db, user)
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
            "pathProgress": stats["pathProgress"],
            "practisedThisWeek": "0m",
            "sessions": stats["sessions"],
            "streak": 0,
        },
        "continueLearning": _continue_learning(db, user),
        "learningPath": _learning_path(db, user),
        "pastSessions": _past_sessions(db, user),
        "myDocuments": _my_documents(db, user),
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
