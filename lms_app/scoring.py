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


def _completion_from(bests: dict[int, int], weights: dict[int, int]) -> int:
    """Share of a document's sections taken to mastery. Pure."""
    if not weights:
        return 0
    done = sum(1 for idx in weights if bests.get(idx, 0) >= settings.MASTERY_THRESHOLD)
    return round(100 * done / len(weights))


def _mean(values: list[int]) -> Optional[int]:
    return round(sum(values) / len(values)) if values else None


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
    return _completion_from(section_bests(db, user_id, document_id), _plan_weights(db, document_id))


def document_progress(db: Session, user_id: int, document_id: int) -> int:
    """How far THROUGH the document the learner is: sections sat / total (0-100).

    Deliberately distinct from ``document_completion``, which counts only
    sections taken to mastery. Showing the mastery figure as a learner's
    "progress" told someone who had sat two of three sections that they were 0%
    through — which reads as "none of that counted" and is the opposite of what
    a progress bar is for. Mastery still gates advancement; progress just says
    where they are."""
    weights = _plan_weights(db, document_id)
    if not weights:
        return 0
    sat = len(attempted_sections(db, user_id, document_id) & set(weights))
    return round(100 * sat / len(weights))


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
    return _mean(scores)


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


#: Path statuses that mean "the learner is finished with this document" — they
#: have sat every section. ``mastered`` additionally means they scored well.
DONE_STATUSES = ("completed", "mastered")


def refresh_path_item(db: Session, *, user_id: int, document_id: int, total_sections: int) -> None:
    """Recompute this document's standing on the learner's path, and unlock the
    next document once the learner has finished this one.

    FINISHED means every section has been sat. That — and only that — is what
    opens the next document. Score does not gate the path: a learner who has
    worked through all of a document is entitled to move on to the next one,
    whatever the grader made of their answers.

    MASTERED is the stricter reading — finished, and the document score (the
    minutes-weighted mean across all sections) clears the threshold. It is a
    label on how well they did, reported to them and to their admin. It is not a
    turnstile.

    Both of those used to be the same thing, and the gate compounded badly. The
    grader has repeatedly landed in the low 60s on long, engaged conversations,
    so learners who had genuinely worked through a document were held at it
    indefinitely by a threshold they had no way to see or argue with. A learner
    can still deliberately redo a section (``restart``) to raise their score,
    and the weak sections stay visible per-section to an admin — but nothing
    about a score can now strand someone on a document they have finished."""
    item = db.scalar(
        select(models.LearningPathItem).where(
            models.LearningPathItem.user_id == user_id,
            models.LearningPathItem.document_id == document_id,
        )
    )
    if item is None:
        return
    understanding = document_understanding(db, user_id, document_id)
    sat = attempted_sections(db, user_id, document_id)
    all_sections_sat = total_sections > 0 and len(sat) >= total_sections

    if all_sections_sat:
        item.status = (
            "mastered"
            if understanding is not None and understanding >= settings.MASTERY_THRESHOLD
            else "completed"
        )
        nxt = db.scalar(
            select(models.LearningPathItem)
            .where(
                models.LearningPathItem.user_id == user_id,
                models.LearningPathItem.status == "locked",
            )
            # id breaks ties: legacy paths were seeded with duplicate idx values,
            # and rows were inserted in curriculum order, so id is the honest
            # fallback for "which document comes next".
            .order_by(models.LearningPathItem.idx, models.LearningPathItem.id)
        )
        if nxt is not None:
            nxt.status = "up_next"
    elif sat and item.status not in DONE_STATUSES:
        # ``sat`` matters: this branch used to fire unconditionally, so calling
        # this for a document the learner has never touched marked it in_progress
        # — turning a locked document into an open one. Nothing hit that in the
        # live path (it is only ever called for the section just sat), but it
        # made recomputing a whole path unsafe, which is exactly what a rule
        # change needs to do.
        item.status = "in_progress"


def attempted_sections(db: Session, user_id: int, document_id: int) -> set[int]:
    """Sections the learner has actually sat, scored or not.

    A section counts as sat if it has an attempt OR a score. Those cannot
    disagree in data written by apply_session, but they can in rows written any
    other way — the schema migration rebuilt progress rows with attempts=0, and
    a seeded row need not set both. Keying mastery on attempts alone would then
    quietly refuse to unlock the next document for a learner whose scores are
    plainly there."""
    return {
        int(i)
        for i in db.scalars(
            select(models.SectionProgress.module_idx).where(
                models.SectionProgress.user_id == user_id,
                models.SectionProgress.document_id == document_id,
                (models.SectionProgress.attempts > 0)
                | (models.SectionProgress.best_score.is_not(None)),
            )
        ).all()
    }


def next_section_idx(db: Session, user_id: int, document_id: int, total_sections: int) -> int:
    """Where the learner should resume: the first section they have not sat yet,
    else the last one.

    This used to return the first section below the MASTERY threshold, which made
    the resume point a gate rather than a bookmark. A learner whose best on a
    section was 65 — five points short — was returned to it every single time and
    could never reach the rest of the document, however many times they sat it.
    One learner did so four times, for 33, 19, 20 and 5 answers.

    Progress and understanding are different things and the schema already keeps
    them apart: the document score still reflects that weak section honestly and
    still counts it against mastery, so nothing is being waved through. The
    learner is simply allowed to carry on, and can redo a section deliberately
    (``restart``) to raise it.
    """
    if total_sections <= 0:
        return 0
    attempted = attempted_sections(db, user_id, document_id)
    for idx in range(total_sections):
        if idx not in attempted:
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


# ---- batched read model ------------------------------------------------------


class ScoreIndex:
    """Every understanding number for one workspace, loaded in two queries.

    ``build_bundle`` asks for the same figures over and over — each learner's
    understanding is recomputed by the People table, the KPI row, the
    at-risk list and every cohort. Answering those one at a time issued ~1700
    queries per /api/bootstrap; against a remote database that is tens of seconds
    of round trips, and the page failed rather than loaded.

    The arithmetic is identical to the per-call functions above — both sides call
    the same pure helpers — so this is purely a change in HOW the inputs are
    fetched, not in what the numbers mean.
    """

    def __init__(self, db: Session, workspace_id: int):
        self._weights: dict[int, dict[int, int]] = {}
        self._bests: dict[tuple[int, int], dict[int, int]] = {}
        self._started: dict[int, set[int]] = {}
        # Sat, whether or not it produced a score — progress is not mastery.
        self._attempted: dict[tuple[int, int], set[int]] = {}

        doc_ids = [
            int(d)
            for d in db.scalars(
                select(models.Document.id).where(models.Document.workspace_id == workspace_id)
            ).all()
        ]
        user_ids = [
            int(u)
            for u in db.scalars(
                select(models.User.id).where(models.User.workspace_id == workspace_id)
            ).all()
        ]

        if doc_ids:
            for did, idx, mins in db.execute(
                select(models.Module.document_id, models.Module.idx, models.Module.minutes).where(
                    models.Module.document_id.in_(doc_ids)
                )
            ).all():
                self._weights.setdefault(int(did), {})[int(idx)] = max(1, int(mins or 1))

        if user_ids:
            for uid, did, idx, best, attempts in db.execute(
                select(
                    models.SectionProgress.user_id,
                    models.SectionProgress.document_id,
                    models.SectionProgress.module_idx,
                    models.SectionProgress.best_score,
                    models.SectionProgress.attempts,
                ).where(models.SectionProgress.user_id.in_(user_ids))
            ).all():
                if (attempts or 0) > 0 or best is not None:
                    self._attempted.setdefault((int(uid), int(did)), set()).add(int(idx))
                if best is None:
                    continue
                self._bests.setdefault((int(uid), int(did)), {})[int(idx)] = int(best)
                self._started.setdefault(int(uid), set()).add(int(did))

    # A document with no plan is treated as one section, matching _plan_weights.
    def plan_weights(self, document_id: int) -> dict[int, int]:
        return self._weights.get(document_id) or {0: 1}

    def section_bests(self, user_id: int, document_id: int) -> dict[int, int]:
        return self._bests.get((user_id, document_id), {})

    def started_document_ids(self, user_id: int) -> list[int]:
        return sorted(self._started.get(user_id, set()))

    def document_understanding(self, user_id: int, document_id: int) -> Optional[int]:
        bests = self.section_bests(user_id, document_id)
        if not bests:
            return None
        return ai.document_score(bests, self.plan_weights(document_id))

    def document_completion(self, user_id: int, document_id: int) -> int:
        return _completion_from(
            self.section_bests(user_id, document_id), self.plan_weights(document_id)
        )

    def document_progress(self, user_id: int, document_id: int) -> int:
        """Sections sat / total. See scoring.document_progress."""
        weights = self.plan_weights(document_id)
        if not weights:
            return 0
        sat = len(self._attempted.get((user_id, document_id), set()) & set(weights))
        return round(100 * sat / len(weights))

    def user_understanding(
        self, user_id: int, document_ids: Optional[Iterable[int]] = None
    ) -> Optional[int]:
        started = set(self.started_document_ids(user_id))
        if document_ids is not None:
            started &= set(document_ids)
        return _mean(
            [
                s
                for s in (self.document_understanding(user_id, d) for d in sorted(started))
                if s is not None
            ]
        )

    def group_understanding(
        self, user_ids: Iterable[int], document_ids: Optional[Iterable[int]] = None
    ) -> Optional[int]:
        docs = list(document_ids) if document_ids is not None else None
        return _mean(
            [v for v in (self.user_understanding(u, docs) for u in user_ids) if v is not None]
        )

    def group_completion(self, user_ids: Iterable[int], document_ids: Iterable[int]) -> int:
        docs = list(document_ids)
        members = list(user_ids)
        if not docs or not members:
            return 0
        per = [
            sum(self.document_completion(u, d) for d in docs) / len(docs) for u in members
        ]
        return round(sum(per) / len(per))
