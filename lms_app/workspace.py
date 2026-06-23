from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


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

ZERO_KPIS = [
    {"label": "Avg understanding", "value": "0", "hint": "no sessions yet"},
    {"label": "Active learners", "value": "1", "hint": "just you"},
    {"label": "Completion", "value": "0%", "hint": "no documents yet"},
    {"label": "At risk", "value": "0", "hint": "all clear"},
    {"label": "Sessions today", "value": "0", "hint": "none yet"},
]


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
            cohort="—",
            documents=0,
            understanding=0,
        )
    )
    return True


def apply_pending_invites(db: Session, sub: str, email: Optional[str], name: Optional[str] = None) -> bool:
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
    """One-time/idempotent backfill across ALL existing data: for every pending
    invite whose email belongs to an already-signed-up person (a user row with a
    clerk_id), create the missing membership and mark the invite accepted. This
    "moves" existing data into the multi-membership model and stops people who have
    accepted from showing as a pending invite. Returns the number of invites resolved."""
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

    1. Apply any pending invites matching their email (creating memberships) so an
       invited workspace shows up without re-signup.
    2. Ensure at least one membership — first-ever login with no invite gets a fresh
       personal workspace as its Admin.
    3. Return the membership for ``active_ws_id`` when the person belongs to it, else
       their default (earliest) membership. A stale/forged id can never select a
       workspace they aren't a member of.
    """
    def _memberships() -> list[models.User]:
        return list(
            db.scalars(
                select(models.User)
                .where(models.User.clerk_id == sub)
                .order_by(models.User.id)
            ).all()
        )

    memberships = _memberships()
    eff_email = email or (memberships[0].email if memberships else None)
    # Real name if given, else the existing one, else a sensible name from the email
    # (never "New user"/"there").
    eff_name = name or (memberships[0].name if memberships else None) or _display_name(eff_email)

    if apply_pending_invites(db, sub, eff_email, eff_name):
        memberships = _memberships()

    if not memberships:
        if not eff_email:
            # No membership and no email yet (Clerk hasn't populated it) — refuse to
            # fabricate a personal workspace. Doing so would strand an INVITED user on
            # the create-workspace/onboarding screen (their invite is matched by email).
            # The client retries once the email is available.
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
            cohort="—",
            documents=0,
            understanding=0,
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


def _user_team_map(db: Session, ws_id: int) -> dict[int, str]:
    """Map each user to their (first) team name in the workspace."""
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


def _people(db: Session, ws_id: int) -> list[dict]:
    users = db.scalars(
        select(models.User).where(models.User.workspace_id == ws_id).order_by(models.User.id)
    ).all()
    team_of = _user_team_map(db, ws_id)
    return [
        {
            "name": u.name,
            "email": u.email,
            "cohort": u.cohort,
            "team": team_of.get(u.id, ""),
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


def _team_member_ids(db: Session, team_id: int) -> list[int]:
    return list(
        db.scalars(select(models.TeamMember.user_id).where(models.TeamMember.team_id == team_id)).all()
    )


def _team_doc_ids(db: Session, team_id: int) -> list[int]:
    return list(
        db.scalars(
            select(models.TeamDocument.document_id)
            .where(models.TeamDocument.team_id == team_id)
            .order_by(models.TeamDocument.idx)
        ).all()
    )


def team_detail(db: Session, t: models.Team) -> dict:
    member_ids = _team_member_ids(db, t.id)
    doc_ids = _team_doc_ids(db, t.id)
    docs = [
        {"id": d.id, "name": d.name}
        for d in (db.get(models.Document, did) for did in doc_ids)
        if d is not None
    ]
    return {
        "id": t.id,
        "name": t.name,
        "lead": t.lead,
        "members": len(member_ids),
        "memberIds": member_ids,
        "documentIds": doc_ids,
        "documents": docs,
        "published": t.published,
        "avg": t.avg,
        "paths": len(doc_ids),
    }


def _teams(db: Session, ws_id: int) -> list[dict]:
    rows = db.scalars(
        select(models.Team).where(models.Team.workspace_id == ws_id).order_by(models.Team.id)
    ).all()
    return [team_detail(db, t) for t in rows]


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
    out: list[dict] = []
    for i in _path_items(db, user.id):
        doc = _doc_by_name(db, user.workspace_id, i.title)
        out.append(
            {
                "title": clean_name(i.title),
                "sections": i.sections,
                "status": i.status,
                "progress": i.progress,
                "docId": doc.id if doc else None,
            }
        )
    return out


def _my_documents(db: Session, user: models.User) -> list[dict]:
    out: list[dict] = []
    for i in _path_items(db, user.id):
        doc = _doc_by_name(db, user.workspace_id, i.title)
        status = (
            "Mastered" if i.status == "mastered" else "Locked" if i.status == "locked" else "Assigned"
        )
        out.append(
            {
                "name": clean_name(i.title),
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
        {"doc": clean_name(s.doc), "date": s.date, "score": s.score, "duration": s.duration, "topics": s.topics}
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
        "doc": clean_name(item.title),
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


def _understanding_trend(db: Session, ws_id: int) -> list[dict]:
    """Workspace understanding over time: one point per session (last 12),
    oldest → newest, for the admin trend chart. Empty until sessions exist."""
    rows = db.execute(
        select(models.LearningSession.date, models.LearningSession.score)
        .join(models.User, models.User.id == models.LearningSession.user_id)
        .where(models.User.workspace_id == ws_id)
        .order_by(models.LearningSession.id)
    ).all()
    rows = rows[-12:]
    out: list[dict] = []
    for idx, (d, s) in enumerate(rows):
        # Label alternate points to keep the x-axis readable (ISO date → MM-DD).
        label = (d or "")[5:] if idx % 2 == 0 else ""
        out.append({"m": label, "v": int(s or 0)})
    return out


def _understanding_series(db: Session, ws_id: int) -> list[dict]:
    """Raw workspace session points (ISO date + score), oldest → newest, so the
    Overview chart can re-bucket by week / month / quarter."""
    rows = db.execute(
        select(models.LearningSession.date, models.LearningSession.score)
        .join(models.User, models.User.id == models.LearningSession.user_id)
        .where(models.User.workspace_id == ws_id)
        .order_by(models.LearningSession.id)
    ).all()
    return [{"date": d or "", "score": int(s or 0)} for d, s in rows]


def _avg_understanding(db: Session, user_ids: list[int]) -> int:
    """Average understanding over the MEASURED members of a group (those with a
    score > 0). 0 when nobody in the group has been measured yet."""
    if not user_ids:
        return 0
    scores = [
        u.understanding
        for u in db.scalars(select(models.User).where(models.User.id.in_(user_ids))).all()
        if u.understanding and u.understanding > 0
    ]
    return round(sum(scores) / len(scores)) if scores else 0


def _understanding_kpis(db: Session, ws_id: int) -> list[dict]:
    """Real, workspace-scoped headline numbers for the Understanding page."""
    users = list(db.scalars(select(models.User).where(models.User.workspace_id == ws_id)).all())
    measured = [u.understanding for u in users if u.understanding and u.understanding > 0]
    avg = round(sum(measured) / len(measured)) if measured else 0
    docs = (
        db.scalar(select(func.count()).select_from(models.Document).where(models.Document.workspace_id == ws_id))
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
    user_ids = [u.id for u in users]
    total = mastered = 0
    if user_ids:
        items = list(
            db.scalars(
                select(models.LearningPathItem).where(models.LearningPathItem.user_id.in_(user_ids))
            ).all()
        )
        total = len(items)
        mastered = sum(1 for i in items if i.status == "mastered")
    mastery = round(100 * mastered / total) if total else 0
    return [
        {"label": "Average understanding", "value": str(avg), "hint": "demonstrated, not guessed"},
        {"label": "Learners measured", "value": str(len(measured)), "hint": "in this workspace"},
        {"label": "Topics tracked", "value": str(int(topics)), "hint": f"from {int(docs)} document{'' if docs == 1 else 's'}"},
        {"label": "Mastery rate", "value": f"{mastery}%", "hint": "sections mastered"},
    ]


def _cohort_health(db: Session, ws_id: int) -> list[dict]:
    return [
        {"name": c.name, "value": _avg_understanding(db, _cohort_member_ids(db, c.id)), "pct": c.completion}
        for c in db.scalars(
            select(models.Cohort).where(models.Cohort.workspace_id == ws_id).order_by(models.Cohort.id)
        ).all()
    ]


def _team_health(db: Session, ws_id: int) -> list[dict]:
    return [
        {"name": t.name, "value": _avg_understanding(db, _team_member_ids(db, t.id))}
        for t in db.scalars(
            select(models.Team).where(models.Team.workspace_id == ws_id).order_by(models.Team.id)
        ).all()
    ]


def _falling_behind(db: Session, ws_id: int) -> list[dict]:
    users = db.scalars(
        select(models.User).where(models.User.workspace_id == ws_id).order_by(models.User.understanding)
    ).all()
    return [
        {
            "name": u.name,
            "cohort": u.cohort if u.cohort and u.cohort != "—" else "No cohort",
            "score": u.understanding,
        }
        for u in users
        if u.understanding and 0 < u.understanding < 55
    ][:8]


def _user_workspaces(db: Session, user: models.User) -> list[dict]:
    """Every workspace this person belongs to (drives the switcher), with their role
    in each. Falls back to just the active membership when there's no clerk_id
    (auth-disabled dev mode)."""
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
    return {
        "mode": "user",
        "needsOnboarding": needs_onboarding,
        "workspace": {"name": ws.name, "plan": ws.plan, "slug": ws.slug or slugify(ws.name)},
        "account": {"name": display_name, "email": user.email, "role": "Workspace owner" if is_admin(user) else user.role},
        "role": user.role,
        "workspaces": _user_workspaces(db, user),
        "activeWorkspaceId": user.workspace_id,
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
            "understandingKpis": _understanding_kpis(db, user.workspace_id),
            "understandingTrend": _understanding_trend(db, user.workspace_id),
            "understandingSeries": _understanding_series(db, user.workspace_id),
            "cohortHealth": _cohort_health(db, user.workspace_id),
            "teamHealth": _team_health(db, user.workspace_id),
            "needsAttention": _falling_behind(db, user.workspace_id),
            "recentActivity": [],
            "cohorts": _cohorts(db, user.workspace_id),
            "people": _people(db, user.workspace_id),
            "pendingInvites": _pending(db, user.workspace_id),
            "teams": _teams(db, user.workspace_id),
            "documents": _documents(db, user.workspace_id),
        },
    }
