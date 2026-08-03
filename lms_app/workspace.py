from __future__ import annotations

"""Workspace membership + the single read-model the app renders from.

Every understanding number here comes from ``scoring.py``. Nothing is cached on
a row any more: the People table, cohort health, the trend chart and the
learner's own dashboard all recompute from ``section_progress``, so the admin
and the learner can never see two different numbers for the same person.
"""

import re
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models, scoring
from .config import settings


def clean_name(name: str) -> str:
    """Readable document title from a raw upload filename: drop the extension and
    the underscores/dashes uploaders leave behind."""
    base = re.sub(r"\.pdf$", "", name or "", flags=re.IGNORECASE)
    base = re.sub(r"[_-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base or name


def slugify(value: str) -> str:
    """URL-safe workspace slug from a name/link, e.g. 'Acme Inc.' → 'acme-inc'."""
    s = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return s[:60] or "workspace"


def first_name(name: Optional[str]) -> str:
    return (name or "there").strip().split(" ")[0] or "there"


def _display_name(email: Optional[str]) -> str:
    """A human-ish name from an email local-part, used only when no real name was
    provided — so a new user is never labelled 'New user' / 'there'."""
    if email and "@" in email:
        local = email.split("@", 1)[0]
        parts = [p for p in re.split(r"[._+-]+", local) if p]
        derived = " ".join(p.capitalize() for p in parts)
        if derived:
            return derived
    return "New user"


# ---- membership --------------------------------------------------------------


def _ensure_membership(
    db: Session, sub: str, ws_id: int, email: Optional[str], name: Optional[str], role: str
) -> bool:
    """Ensure a (clerk_id, workspace_id) membership row exists. Claims a pre-seeded
    member row matched by email (clerk_id still NULL) rather than duplicating it.
    Returns True if a row was created or claimed."""
    member = db.scalar(
        select(models.User).where(
            models.User.clerk_id == sub, models.User.workspace_id == ws_id
        )
    )
    if member is not None:
        return False
    orphan = db.scalar(
        select(models.User).where(
            models.User.workspace_id == ws_id,
            func.lower(models.User.email) == (email or "").lower(),
            models.User.clerk_id.is_(None),
        )
    )
    if orphan is not None:
        orphan.clerk_id = sub
        if role and orphan.role != role:
            orphan.role = role
        return True
    db.add(
        models.User(
            clerk_id=sub,
            workspace_id=ws_id,
            name=name or "New user",
            email=email or f"{sub}@clerk.local",
            role=role,
        )
    )
    return True


def apply_pending_invites(
    db: Session, sub: str, email: Optional[str], name: Optional[str] = None
) -> bool:
    """Join this person to every workspace that has a pending invite for their email,
    marking those invites accepted. Runs on every bootstrap (not one-shot), so an
    ALREADY-registered user gains the new membership instead of the invite being
    ignored. Idempotent. Returns True if anything changed."""
    if not email:
        return False
    invites = db.scalars(
        select(models.Invite).where(
            func.lower(models.Invite.email) == email.lower(),
            models.Invite.status == "pending",
        )
    ).all()
    changed = False
    for inv in invites:
        _ensure_membership(db, sub, inv.workspace_id, email, name, inv.role)
        inv.status = "accepted"
        changed = True
    if changed:
        db.commit()
    return changed


def reconcile_memberships(db: Session) -> int:
    """Idempotent backfill: for every pending invite whose email belongs to an
    already-signed-up person, create the missing membership and mark the invite
    accepted. Returns the number of invites resolved."""
    invites = db.scalars(select(models.Invite).where(models.Invite.status == "pending")).all()
    resolved = 0
    for inv in invites:
        rows = db.scalars(
            select(models.User).where(
                func.lower(models.User.email) == inv.email.lower(),
                models.User.clerk_id.is_not(None),
            )
        ).all()
        if not rows:
            continue
        seen: set[str] = set()
        for r in rows:
            if r.clerk_id in seen:
                continue
            seen.add(r.clerk_id)
            _ensure_membership(db, r.clerk_id, inv.workspace_id, inv.email, r.name, inv.role)
        inv.status = "accepted"
        resolved += 1
    if resolved:
        db.commit()
    return resolved


def resolve_active_membership(
    db: Session,
    sub: str,
    active_ws_id: Optional[int] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
) -> models.User:
    """Resolve the signed-in person's ACTIVE workspace membership.

    1. Apply any pending invites matching their email (creating memberships).
    2. Ensure at least one membership — first-ever login with no invite gets a
       fresh personal workspace as its Admin.
    3. Return the membership for ``active_ws_id`` when the person belongs to it,
       else their default (earliest) membership. A stale/forged id can never
       select a workspace they aren't a member of.
    """

    def _memberships() -> list[models.User]:
        return list(
            db.scalars(
                select(models.User).where(models.User.clerk_id == sub).order_by(models.User.id)
            ).all()
        )

    memberships = _memberships()
    eff_email = email or (memberships[0].email if memberships else None)
    eff_name = name or (memberships[0].name if memberships else None) or _display_name(eff_email)

    if apply_pending_invites(db, sub, eff_email, eff_name):
        memberships = _memberships()

    if not memberships:
        if not eff_email:
            # Refuse to fabricate a personal workspace before Clerk has an email —
            # that would strand an INVITED user (their invite matches by email).
            raise ValueError("account not ready: email required")
        ws = models.Workspace(name=f"{first_name(eff_name)}'s workspace", plan="Personal workspace")
        db.add(ws)
        db.flush()
        user = models.User(
            clerk_id=sub,
            workspace_id=ws.id,
            name=eff_name or "New user",
            email=eff_email or f"{sub}@clerk.local",
            role="Admin",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        memberships = [user]

    active = None
    if active_ws_id is not None:
        active = next((m for m in memberships if m.workspace_id == active_ws_id), None)
    if active is None:
        active = memberships[0]

    changed = False
    if name and active.name != name:
        active.name = name
        changed = True
    if email and active.email != email:
        active.email = email
        changed = True
    if changed:
        db.commit()
    return active


def resolve_user(db: Session, sub: str, name: Optional[str], email: Optional[str]) -> models.User:
    """Back-compat shim: resolve the person's default active membership."""
    return resolve_active_membership(db, sub, None, name, email)


def is_admin(user: models.User) -> bool:
    return user.role in ("Admin", "Owner")


def is_owner(db: Session, user: models.User) -> bool:
    """The owner is the earliest member of the workspace (its creator)."""
    first_id = db.scalar(
        select(func.min(models.User.id)).where(models.User.workspace_id == user.workspace_id)
    )
    return user.id == first_id


# ---- lookups -----------------------------------------------------------------


def _user_team_map(db: Session, ws_id: int) -> dict[int, str]:
    rows = db.execute(
        select(models.TeamMember.user_id, models.Team.name)
        .join(models.Team, models.Team.id == models.TeamMember.team_id)
        .where(models.Team.workspace_id == ws_id)
        .order_by(models.Team.id)
    ).all()
    out: dict[int, str] = {}
    for uid, name in rows:
        out.setdefault(uid, name)
    return out


def _user_cohort_map(db: Session, ws_id: int) -> dict[int, str]:
    """Cohort label per learner, derived from membership. This used to be a
    ``users.cohort`` string kept in sync by hand on every cohort edit."""
    rows = db.execute(
        select(models.CohortMember.user_id, models.Cohort.name)
        .join(models.Cohort, models.Cohort.id == models.CohortMember.cohort_id)
        .where(models.Cohort.workspace_id == ws_id)
        .order_by(models.Cohort.id)
    ).all()
    out: dict[int, str] = {}
    for uid, name in rows:
        out.setdefault(uid, name)
    return out


def _assigned_count(db: Session, document_id: int) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(models.LearningPathItem)
            .where(models.LearningPathItem.document_id == document_id)
        )
        or 0
    )


def _section_counts(db: Session, document_ids: list[int]) -> dict[int, int]:
    """Sections per document, in one query rather than one per document."""
    if not document_ids:
        return {}
    rows = db.execute(
        select(models.Module.document_id, func.count())
        .where(models.Module.document_id.in_(document_ids))
        .group_by(models.Module.document_id)
    ).all()
    return {int(d): int(n) for d, n in rows}


def _assigned_counts(db: Session, document_ids: list[int]) -> dict[int, int]:
    if not document_ids:
        return {}
    rows = db.execute(
        select(models.LearningPathItem.document_id, func.count())
        .where(models.LearningPathItem.document_id.in_(document_ids))
        .group_by(models.LearningPathItem.document_id)
    ).all()
    return {int(d): int(n) for d, n in rows}


def document_out(
    db: Session,
    d: models.Document,
    sections: Optional[int] = None,
    assigned: Optional[int] = None,
) -> dict:
    """Counts may be supplied by the caller when rendering a whole list, so a
    13-document workspace costs two queries rather than twenty-six."""
    if sections is None:
        sections = int(
            db.scalar(
                select(func.count()).select_from(models.Module).where(models.Module.document_id == d.id)
            )
            or 0
        )
    return {
        "id": d.id,
        "name": d.name,
        "title": clean_name(d.name),
        # `sections` is the number of TEACHING sections; `chunks` is the retrieval
        # split. The old API returned the chunk count under the name "sections".
        "sections": sections,
        "chunks": d.chunk_count,
        "assigned": _assigned_count(db, d.id) if assigned is None else assigned,
        "status": d.status,
    }


def _documents(db: Session, ws_id: int) -> list[dict]:
    docs = db.scalars(
        select(models.Document)
        .where(models.Document.workspace_id == ws_id)
        .order_by(models.Document.id.desc())
    ).all()
    ids = [d.id for d in docs]
    sections = _section_counts(db, ids)
    assigned = _assigned_counts(db, ids)
    return [document_out(db, d, sections.get(d.id, 0), assigned.get(d.id, 0)) for d in docs]


def _people(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    users = db.scalars(
        select(models.User).where(models.User.workspace_id == ws_id).order_by(models.User.id)
    ).all()
    team_of = _user_team_map(db, ws_id)
    cohort_of = _user_cohort_map(db, ws_id)
    out: list[dict] = []
    for u in users:
        score = idx.user_understanding(u.id)
        out.append(
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "role": u.role,
                "cohort": cohort_of.get(u.id, "—"),
                "team": team_of.get(u.id, ""),
                "documents": len(idx.started_document_ids(u.id)),
                # Demonstrated across every document they have started — not the
                # score of whichever section happened to be graded most recently.
                "understanding": score,
                "band": scoring.band(score),
            }
        )
    return out


def _pending(db: Session, ws_id: int) -> list[dict]:
    invites = db.scalars(
        select(models.Invite)
        .where(models.Invite.workspace_id == ws_id, models.Invite.status == "pending")
        .order_by(models.Invite.id)
    ).all()
    return [{"id": i.id, "email": i.email, "role": i.role} for i in invites]


# ---- cohorts / teams ---------------------------------------------------------


def _cohort_member_ids(db: Session, cohort_id: int) -> list[int]:
    return scoring.cohort_member_ids(db, cohort_id)


def _cohort_doc_ids(db: Session, cohort_id: int) -> list[int]:
    return scoring.cohort_document_ids(db, cohort_id)


def cohort_detail(db: Session, c: models.Cohort, idx=None) -> dict:
    member_ids = _cohort_member_ids(db, c.id)
    doc_ids = _cohort_doc_ids(db, c.id)
    docs = [
        {"id": d.id, "name": d.name, "title": clean_name(d.name)}
        for d in (db.get(models.Document, did) for did in doc_ids)
        if d is not None
    ]
    if idx is None:
        idx = scoring.ScoreIndex(db, c.workspace_id)
    cu = idx.group_understanding(member_ids, doc_ids) if doc_ids else None
    completion = idx.group_completion(member_ids, doc_ids)
    return {
        "id": c.id,
        "name": c.name,
        "members": len(member_ids),
        "memberIds": member_ids,
        "documentIds": doc_ids,
        "documents": docs,
        "published": c.published,
        "understanding": cu,
        "avg": cu or 0,  # legacy key the admin UI still reads
        "band": scoring.band(cu),
        "completion": completion,
        "status": _cohort_status(cu, completion, c.published),
    }


def _cohort_status(understanding: Optional[int], completion: int, published: bool) -> str:
    if not published:
        return "Draft"
    if understanding is None:
        return "Not started"
    if understanding < settings.AT_RISK_THRESHOLD:
        return "At risk"
    if completion < 50:
        return "Behind"
    return "On track"


def _cohorts(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    rows = db.scalars(
        select(models.Cohort).where(models.Cohort.workspace_id == ws_id).order_by(models.Cohort.id)
    ).all()
    return [cohort_detail(db, c, idx) for c in rows]


def _team_member_ids(db: Session, team_id: int) -> list[int]:
    return [
        int(u)
        for u in db.scalars(
            select(models.TeamMember.user_id).where(models.TeamMember.team_id == team_id)
        ).all()
    ]


def _team_doc_ids(db: Session, team_id: int) -> list[int]:
    return [
        int(d)
        for d in db.scalars(
            select(models.TeamDocument.document_id)
            .where(models.TeamDocument.team_id == team_id)
            .order_by(models.TeamDocument.idx)
        ).all()
    ]


def team_detail(db: Session, t: models.Team, idx=None) -> dict:
    member_ids = _team_member_ids(db, t.id)
    doc_ids = _team_doc_ids(db, t.id)
    docs = [
        {"id": d.id, "name": d.name, "title": clean_name(d.name)}
        for d in (db.get(models.Document, did) for did in doc_ids)
        if d is not None
    ]
    if idx is None:
        idx = scoring.ScoreIndex(db, t.workspace_id)
    tu = idx.group_understanding(member_ids, doc_ids or None)
    return {
        "id": t.id,
        "name": t.name,
        "lead": t.lead,
        "members": len(member_ids),
        "memberIds": member_ids,
        "documentIds": doc_ids,
        "documents": docs,
        "published": t.published,
        "understanding": tu,
        "avg": tu or 0,
        "band": scoring.band(tu),
        "paths": len(doc_ids),
    }


def _teams(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    rows = db.scalars(
        select(models.Team).where(models.Team.workspace_id == ws_id).order_by(models.Team.id)
    ).all()
    return [team_detail(db, t, idx) for t in rows]


# ---- learner-side ------------------------------------------------------------


def _path_items(db: Session, user_id: int) -> list[models.LearningPathItem]:
    return list(
        db.scalars(
            select(models.LearningPathItem)
            .where(models.LearningPathItem.user_id == user_id)
            .order_by(models.LearningPathItem.idx, models.LearningPathItem.id)
        ).all()
    )


def _section_count(db: Session, document_id: int) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(models.Module)
            .where(models.Module.document_id == document_id)
        )
        or 0
    )


def _learning_path(db: Session, user: models.User, idx: "scoring.ScoreIndex") -> list[dict]:
    out: list[dict] = []
    items = _path_items(db, user.id)
    sections = _section_counts(db, [i.document_id for i in items])
    for i in items:
        doc = db.get(models.Document, i.document_id)
        out.append(
            {
                "docId": i.document_id,
                "title": clean_name(doc.name) if doc else "",
                "sections": sections.get(i.document_id, 0),
                "status": i.status,
                # % of the document taken to mastery, and how much is demonstrated.
                # How far THROUGH the document, not how much is mastered — see
                # scoring.document_progress.
                "progress": idx.document_progress(user.id, i.document_id),
                "mastered": idx.document_completion(user.id, i.document_id),
                "understanding": idx.document_understanding(user.id, i.document_id),
            }
        )
    return out


def _my_documents(db: Session, user: models.User) -> list[dict]:
    out: list[dict] = []
    items = _path_items(db, user.id)
    sections = _section_counts(db, [i.document_id for i in items])
    for i in items:
        doc = db.get(models.Document, i.document_id)
        if doc is None:
            continue
        out.append(
            {
                "docId": doc.id,
                "name": clean_name(doc.name),
                "pages": doc.chunk_count,
                "sections": sections.get(doc.id, 0),
                "status": (
                    "Mastered"
                    if i.status == "mastered"
                    else "Completed"
                    if i.status == "completed"
                    else "Locked"
                    if i.status == "locked"
                    else "Assigned"
                ),
                "added": "",
            }
        )
    return out


def _past_sessions(db: Session, user: models.User, limit: int = 10) -> list[dict]:
    rows = db.scalars(
        select(models.LearningSession)
        .where(models.LearningSession.user_id == user.id)
        .order_by(models.LearningSession.id.desc())
        .limit(limit)
    ).all()
    out: list[dict] = []
    for s in rows:
        doc = db.get(models.Document, s.document_id)
        out.append(
            {
                "id": s.id,
                "doc": clean_name(doc.name) if doc else "",
                "docId": s.document_id,
                "section": s.module_idx + 1,
                "date": s.started_at.date().isoformat() if s.started_at else "",
                "score": s.score,  # null = the sitting had nothing to assess
                "band": scoring.band(s.score),
                "summary": s.summary,
                "turns": s.learner_turns,
                "topics": s.topics,
            }
        )
    return out


def _continue_learning(db: Session, user: models.User, idx: "scoring.ScoreIndex") -> Optional[dict]:
    item = db.scalar(
        select(models.LearningPathItem)
        .where(
            models.LearningPathItem.user_id == user.id,
            models.LearningPathItem.status.in_(["in_progress", "up_next"]),
        )
        .order_by(models.LearningPathItem.idx, models.LearningPathItem.id)
    )
    if item is None:
        return None
    doc = db.get(models.Document, item.document_id)
    if doc is None:
        return None
    total = _section_count(db, doc.id)
    cur = scoring.next_section_idx(db, user.id, doc.id, total) + 1
    completion = idx.document_completion(user.id, doc.id)
    progress = idx.document_progress(user.id, doc.id)
    return {
        "docId": doc.id,
        "doc": clean_name(doc.name),
        "position": f"Section {min(cur, total)} of {total}" if total else "Ready to start",
        "remaining": "Pick up where you left off." if progress else "Start your first section.",
        "understanding": idx.document_understanding(user.id, doc.id),
        "progress": progress,
        "mastered": completion,
    }


def _learner_stats(db: Session, user: models.User) -> dict:
    items = _path_items(db, user.id)
    # How far along their path they are — documents finished, not documents
    # scored above the bar. Mastery is reported separately, per document.
    mastered = sum(1 for i in items if i.status in scoring.DONE_STATUSES)
    sessions = (
        db.scalar(
            select(func.count())
            .select_from(models.LearningSession)
            .where(models.LearningSession.user_id == user.id)
        )
        or 0
    )
    return {"pathProgress": f"{mastered} / {len(items)}", "sessions": int(sessions)}


# ---- admin analytics ---------------------------------------------------------


def _understanding_series(db: Session, ws_id: int) -> list[dict]:
    """Raw scored sittings (ISO date + score), oldest → newest, so the Overview
    chart can re-bucket by week / month / quarter. Unscoreable sittings are
    excluded — plotting them as zeros is what made the trend look like noise."""
    rows = db.execute(
        select(models.LearningSession.started_at, models.LearningSession.score)
        .join(models.User, models.User.id == models.LearningSession.user_id)
        .where(models.User.workspace_id == ws_id, models.LearningSession.score.is_not(None))
        .order_by(models.LearningSession.id)
    ).all()
    return [{"date": d.date().isoformat() if d else "", "score": int(s)} for d, s in rows]


def _understanding_trend(db: Session, ws_id: int) -> list[dict]:
    series = _understanding_series(db, ws_id)[-12:]
    return [
        {"m": (p["date"] or "")[5:] if i % 2 == 0 else "", "v": p["score"]}
        for i, p in enumerate(series)
    ]


def _workspace_learners(db: Session, ws_id: int) -> list[models.User]:
    return list(db.scalars(select(models.User).where(models.User.workspace_id == ws_id)).all())


def _kpis(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    users = _workspace_learners(db, ws_id)
    measured = [v for v in (idx.user_understanding(u.id) for u in users) if v is not None]
    avg = round(sum(measured) / len(measured)) if measured else 0
    at_risk = sum(1 for v in measured if v < settings.AT_RISK_THRESHOLD)
    items = (
        db.scalars(
            select(models.LearningPathItem).where(
                models.LearningPathItem.user_id.in_([u.id for u in users])
            )
        ).all()
        if users
        else []
    )
    # Completion counts documents FINISHED; "Mastery rate" below counts the
    # stricter thing. Reporting one number for both hid which was which.
    finished = sum(1 for i in items if i.status in scoring.DONE_STATUSES)
    completion = round(100 * finished / len(items)) if items else 0
    return [
        {"label": "Avg understanding", "value": str(avg), "hint": "demonstrated, not guessed"},
        {"label": "Active learners", "value": str(len(measured)), "hint": f"of {len(users)} in workspace"},
        {"label": "Completion", "value": f"{completion}%", "hint": "documents completed"},
        {"label": "At risk", "value": str(at_risk), "hint": f"below {settings.AT_RISK_THRESHOLD}"},
        {"label": "Sessions today", "value": str(scoring.sessions_today(db, ws_id)), "hint": ""},
    ]


def _understanding_kpis(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    users = _workspace_learners(db, ws_id)
    measured = [v for v in (idx.user_understanding(u.id) for u in users) if v is not None]
    avg = round(sum(measured) / len(measured)) if measured else 0
    docs = (
        db.scalar(
            select(func.count())
            .select_from(models.Document)
            .where(models.Document.workspace_id == ws_id)
        )
        or 0
    )
    topics = (
        db.scalar(
            select(func.count())
            .select_from(models.Module)
            .join(models.Document, models.Document.id == models.Module.document_id)
            .where(models.Document.workspace_id == ws_id)
        )
        or 0
    )
    items = (
        db.scalars(
            select(models.LearningPathItem).where(
                models.LearningPathItem.user_id.in_([u.id for u in users])
            )
        ).all()
        if users
        else []
    )
    mastery = round(100 * sum(1 for i in items if i.status == "mastered") / len(items)) if items else 0
    return [
        {"label": "Average understanding", "value": str(avg), "hint": "demonstrated, not guessed"},
        {"label": "Learners measured", "value": str(len(measured)), "hint": "in this workspace"},
        {
            "label": "Sections tracked",
            "value": str(int(topics)),
            "hint": f"from {int(docs)} document{'' if docs == 1 else 's'}",
        },
        {"label": "Mastery rate", "value": f"{mastery}%", "hint": "documents mastered"},
    ]


def _cohort_health(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    out: list[dict] = []
    for c in db.scalars(
        select(models.Cohort).where(models.Cohort.workspace_id == ws_id).order_by(models.Cohort.id)
    ).all():
        members = scoring.cohort_member_ids(db, c.id)
        docs = scoring.cohort_document_ids(db, c.id)
        cu = idx.group_understanding(members, docs) if docs else None
        out.append(
            {
                "name": c.name,
                "value": cu or 0,
                "band": scoring.band(cu),
                "pct": idx.group_completion(members, docs),
            }
        )
    return out


def _team_health(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    return [
        {
            "name": t.name,
            "value": idx.group_understanding(
                _team_member_ids(db, t.id), _team_doc_ids(db, t.id) or None
            )
            or 0,
        }
        for t in db.scalars(
            select(models.Team).where(models.Team.workspace_id == ws_id).order_by(models.Team.id)
        ).all()
    ]


def _falling_behind(db: Session, ws_id: int, idx: "scoring.ScoreIndex") -> list[dict]:
    cohort_of = _user_cohort_map(db, ws_id)
    out: list[dict] = []
    for u in _workspace_learners(db, ws_id):
        score = idx.user_understanding(u.id)
        if score is not None and score < settings.AT_RISK_THRESHOLD:
            out.append(
                {
                    "name": u.name,
                    "cohort": cohort_of.get(u.id, "No cohort"),
                    "score": score,
                    "band": scoring.band(score),
                }
            )
    return sorted(out, key=lambda x: x["score"])[:8]


def _user_workspaces(db: Session, user: models.User) -> list[dict]:
    if not user.clerk_id:
        rows = [user]
    else:
        rows = list(
            db.scalars(
                select(models.User)
                .where(models.User.clerk_id == user.clerk_id)
                .order_by(models.User.id)
            ).all()
        )
    out: list[dict] = []
    for m in rows:
        ws = db.get(models.Workspace, m.workspace_id)
        if ws is None:
            continue
        out.append({"id": ws.id, "name": ws.name, "slug": ws.slug or slugify(ws.name), "role": m.role})
    return out


def build_bundle(db: Session, user: models.User, display_name: str) -> dict:
    ws = db.get(models.Workspace, user.workspace_id)
    needs_onboarding = (not ws.onboarded) and is_owner(db, user)
    stats = _learner_stats(db, user)
    # One batched read of every score input for this workspace, shared by every
    # section below. They previously each recomputed the same figures — the
    # People table, both KPI rows, the at-risk list and every cohort all asked
    # for the same learner's understanding independently.
    idx = scoring.ScoreIndex(db, user.workspace_id)
    understanding = idx.user_understanding(user.id)
    return {
        "mode": "user",
        "needsOnboarding": needs_onboarding,
        "workspace": {"name": ws.name, "plan": ws.plan, "slug": ws.slug or slugify(ws.name)},
        "account": {
            "name": display_name,
            "email": user.email,
            "role": "Workspace owner" if is_admin(user) else user.role,
        },
        "role": user.role,
        "workspaces": _user_workspaces(db, user),
        "activeWorkspaceId": user.workspace_id,
        "learner": {
            "name": display_name,
            "firstName": first_name(display_name),
            "understanding": understanding,
            "band": scoring.band(understanding),
            "pathProgress": stats["pathProgress"],
            "practisedThisWeek": "0m",
            "sessions": stats["sessions"],
            "streak": 0,
        },
        "continueLearning": _continue_learning(db, user, idx),
        "learningPath": _learning_path(db, user, idx),
        "pastSessions": _past_sessions(db, user),
        "myDocuments": _my_documents(db, user),
        "admin": {
            "kpis": _kpis(db, user.workspace_id, idx),
            "understandingKpis": _understanding_kpis(db, user.workspace_id, idx),
            "understandingTrend": _understanding_trend(db, user.workspace_id),
            "understandingSeries": _understanding_series(db, user.workspace_id),
            "cohortHealth": _cohort_health(db, user.workspace_id, idx),
            "teamHealth": _team_health(db, user.workspace_id, idx),
            "needsAttention": _falling_behind(db, user.workspace_id, idx),
            "recentActivity": [],
            "cohorts": _cohorts(db, user.workspace_id, idx),
            "people": _people(db, user.workspace_id, idx),
            "pendingInvites": _pending(db, user.workspace_id),
            "teams": _teams(db, user.workspace_id, idx),
            "documents": _documents(db, user.workspace_id),
        },
    }
