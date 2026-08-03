from __future__ import annotations

"""The single source of truth for every understanding number in the product.

How a number is built, bottom-up:

  sitting   one section, one attempt  → LearningSession.score (nullable)
  section   best score ever achieved  → SectionProgress.best_score
  document  minutes-weighted mean of section bests, over ALL planned sections
  learner   mean of their document scores, over documents actually started
  cohort    mean of member scores, scoped to the cohort's documents

Three properties this fixes, in order of how much damage they were doing:

1. A sitting where the learner said nothing is UNSCOREABLE, not a 10. It is
   recorded with score=NULL and never enters an average.
2. A section keeps its BEST demonstrated score. Opening a document and closing
   it can no longer wipe out understanding the learner already proved.
3. A document counts its UNTAUGHT sections as 0. A learner who explained
   section 1 brilliantly and never opened sections 2-6 scores ~17, not 90 —
   because the document score answers "how much of this do they know", and the
   admin was previously shown whichever section happened to be graded last.
"""

from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import ai, models
from .config import settings


def band(score: Optional[int]) -> str:
    """Proficiency band for an understanding score (corporate-competency style)."""
    if score is None:
        return "Not started"
    if score >= 90:
        return "Mastered"
    if score >= settings.MASTERY_THRESHOLD:
        return "Proficient"
    if score >= 40:
        return "Progressing"
    return "Developing"


# ---- plan shape --------------------------------------------------------------


def _plan_weights(db: Session, document_id: int) -> dict[int, int]:
    """module_idx -> minutes. A document with no plan yet is treated as one
    section so a score is still expressible."""
    rows = db.execute(
        select(models.Module.idx, models.Module.minutes).where(
            models.Module.document_id == document_id
        )
    ).all()
    return {int(idx): max(1, int(mins or 1)) for idx, mins in rows} or {0: 1}


def section_bests(db: Session, user_id: int, document_id: int) -> dict[int, int]:
    """module_idx -> the learner's best demonstrated score on that section."""
    rows = db.execute(
        select(models.SectionProgress.module_idx, models.SectionProgress.best_score).where(
            models.SectionProgress.user_id == user_id,
            models.SectionProgress.document_id == document_id,
            models.SectionProgress.best_score.is_not(None),
        )
    ).all()
    return {int(idx): int(score) for idx, score in rows}


# ---- rollups -----------------------------------------------------------------


def document_understanding(db: Session, user_id: int, document_id: int) -> Optional[int]:
    """How much of one document this learner has demonstrated (0-100).
    None when they have never been scored on it."""
    bests = section_bests(db, user_id, document_id)
    if not bests:
        return None
    return ai.document_score(bests, _plan_weights(db, document_id))


def document_completion(db: Session, user_id: int, document_id: int) -> int:
    """Share of the document's sections the learner has taken to mastery (0-100)."""
    weights = _plan_weights(db, document_id)
    if not weights:
        return 0
    bests = section_bests(db, user_id, document_id)
    done = sum(1 for idx in weights if bests.get(idx, 0) >= settings.MASTERY_THRESHOLD)
    return round(100 * done / len(weights))


def started_document_ids(db: Session, user_id: int) -> list[int]:
    """Documents the learner has at least one scored section on."""
    return [
        int(d)
        for d in db.scalars(
            select(models.SectionProgress.document_id)
            .where(
                models.SectionProgress.user_id == user_id,
                models.SectionProgress.best_score.is_not(None),
            )
            .distinct()
        ).all()
    ]


def user_understanding(
    db: Session, user_id: int, document_ids: Optional[Iterable[int]] = None
) -> Optional[int]:
    """A learner's understanding, optionally scoped to a set of documents (a
    cohort's curriculum). The mean of their per-document scores over the
    documents they have actually started. None when nothing is measured."""
    started = set(started_document_ids(db, user_id))
    if document_ids is not None:
        started &= set(document_ids)
    scores = [s for s in (document_understanding(db, user_id, d) for d in sorted(started)) if s is not None]
    return round(sum(scores) / len(scores)) if scores else None


def cohort_document_ids(db: Session, cohort_id: int) -> list[int]:
    return [
        int(d)
        for d in db.scalars(
            select(models.CohortDocument.document_id)
            .where(models.CohortDocument.cohort_id == cohort_id)
            .order_by(models.CohortDocument.idx)
        ).all()
    ]


def cohort_member_ids(db: Session, cohort_id: int) -> list[int]:
    return [
        int(u)
        for u in db.scalars(
            select(models.CohortMember.user_id).where(models.CohortMember.cohort_id == cohort_id)
        ).all()
    ]


def cohort_understanding(db: Session, cohort_id: int) -> Optional[int]:
    docs = cohort_document_ids(db, cohort_id)
    if not docs:
        return None
    vals = [
        v
        for v in (user_understanding(db, uid, docs) for uid in cohort_member_ids(db, cohort_id))
        if v is not None
    ]
    return round(sum(vals) / len(vals)) if vals else None


def cohort_completion(db: Session, cohort_id: int) -> int:
    docs = cohort_document_ids(db, cohort_id)
    members = cohort_member_ids(db, cohort_id)
    if not docs or not members:
        return 0
    per = [
        sum(document_completion(db, uid, d) for d in docs) / len(docs) for uid in members
    ]
    return round(sum(per) / len(per))


def team_understanding(db: Session, team_id: int) -> Optional[int]:
    docs = [
        int(d)
        for d in db.scalars(
            select(models.TeamDocument.document_id).where(models.TeamDocument.team_id == team_id)
        ).all()
    ]
    members = [
        int(u)
        for u in db.scalars(
            select(models.TeamMember.user_id).where(models.TeamMember.team_id == team_id)
        ).all()
    ]
    vals = [
        v
        for v in (user_understanding(db, uid, docs or None) for uid in members)
        if v is not None
    ]
    return round(sum(vals) / len(vals)) if vals else None


# ---- writes ------------------------------------------------------------------


def apply_session(
    db: Session,
    *,
    user: models.User,
    document: models.Document,
    module_idx: int,
    transcript: list[dict],
    result: dict,
    paused: bool,
    total_sections: int,
) -> models.LearningSession:
    """Persist one sitting and fold it into the learner's standing.

    ``result`` is what ``ai.score_understanding`` returned; an unscoreable
    sitting (``scoreable`` False) is still recorded — with score NULL — so the
    transcript is auditable, but it does not move any number."""
    scoreable = bool(result.get("scoreable"))
    score = result.get("score") if scoreable else None

    session_row = models.LearningSession(
        user_id=user.id,
        document_id=document.id,
        module_idx=module_idx,
        score=score,
        covered=int(result.get("covered", 100) or 100),
        summary=str(result.get("summary", ""))[:2000],
        topics=result.get("topics") or [],
        strengths=[str(s) for s in (result.get("strengths") or [])],
        gaps=[str(g) for g in (result.get("gaps") or [])],
        transcript=transcript,
        learner_turns=sum(1 for t in transcript if t.get("role") == "learner"),
        paused=paused,
        ended_at=models.utcnow(),
    )
    db.add(session_row)
    db.flush()

    prog = db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user.id,
            models.SectionProgress.document_id == document.id,
            models.SectionProgress.module_idx == module_idx,
        )
    )
    if prog is None:
        prog = models.SectionProgress(
            user_id=user.id, document_id=document.id, module_idx=module_idx
        )
        db.add(prog)

    if score is not None:
        prog.attempts = (prog.attempts or 0) + 1
        prog.last_score = score
        # Best-ever: demonstrated understanding is not forfeited by a later
        # sitting that was cut short.
        prog.best_score = score if prog.best_score is None else max(prog.best_score, score)

    if paused:
        prog.status = "paused"
    elif score is not None and score >= settings.MASTERY_THRESHOLD:
        prog.status = "completed"
    else:
        prog.status = "in_progress"
    prog.updated_at = models.utcnow()

    refresh_path_item(db, user_id=user.id, document_id=document.id, total_sections=total_sections)
    return session_row


def refresh_path_item(db: Session, *, user_id: int, document_id: int, total_sections: int) -> None:
    """Recompute this document's standing on the learner's path, and unlock the
    next document once it is genuinely mastered."""
    item = db.scalar(
        select(models.LearningPathItem).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.document_id == document_id,
        )
    )
    if item is None:
        return
    completion = document_completion(db, user_id, document_id)
    understanding = document_understanding(db, user_id, document_id)

    if (
        total_sections
        and completion >= 100
        and understanding is not None
        and understanding >= settings.MASTERY_THRESHOLD
    ):
        item.status = "mastered"
        nxt = db.scalar(
            select(models.LearningPathItem)
            .where(
                models.LearningPathItem.user_id == user_id,
                models.LearningPathItem.status == "locked",
            )
            .order_by(models.LearningPathItem.idx)
        )
        if nxt is not None:
            nxt.status = "up_next"
    elif item.status != "mastered":
        item.status = "in_progress"


def next_section_idx(db: Session, user_id: int, document_id: int, total_sections: int) -> int:
    """Where the learner should resume: the first section not yet taken to
    mastery, else the last one."""
    if total_sections <= 0:
        return 0
    bests = section_bests(db, user_id, document_id)
    for idx in range(total_sections):
        if bests.get(idx, 0) < settings.MASTERY_THRESHOLD:
            return idx
    return total_sections - 1


def sessions_today(db: Session, workspace_id: int) -> int:
    today = models.utcnow().date()
    return int(
        db.scalar(
            select(func.count())
            .select_from(models.LearningSession)
            .join(models.User, models.User.id == models.LearningSession.user_id)
            .where(
                models.User.workspace_id == workspace_id,
                func.date(models.LearningSession.started_at) == today,
            )
        )
        or 0
    )
