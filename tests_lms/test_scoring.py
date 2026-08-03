from __future__ import annotations

"""Understanding scoring: section bests → weighted document score → learner/cohort.

These pin down the three behaviours that were wrong in production:
  • a sitting with nothing to assess must not become a low score,
  • a later weak sitting must not erase a section already demonstrated,
  • a document score must account for the sections never attempted.
"""

from lms_app import models, scoring
from lms_app.config import settings
from lms_app.db import SessionLocal


def _fixture(db, *, name: str, minutes: list[int]):
    ws = models.Workspace(name=f"{name}Co", plan="Admin workspace")
    db.add(ws)
    db.flush()
    u = models.User(
        clerk_id=f"{name}_u", workspace_id=ws.id, name=name, email=f"{name}@t.dev", role="Learner"
    )
    doc = models.Document(workspace_id=ws.id, name=f"{name} Doc", chunk_count=len(minutes))
    db.add_all([u, doc])
    db.flush()
    for i, m in enumerate(minutes):
        db.add(models.Module(document_id=doc.id, idx=i, title=f"S{i}", minutes=m))
    db.flush()
    return ws, u, doc


def _progress(db, u, doc, idx, best):
    db.add(
        models.SectionProgress(
            user_id=u.id, document_id=doc.id, module_idx=idx, best_score=best, last_score=best
        )
    )


def test_document_score_weights_sections_and_counts_untaught_as_zero(client):
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Weighted", minutes=[5, 5, 10])
        # Only the first two of three sections attempted.
        _progress(db, u, doc, 0, 80)
        _progress(db, u, doc, 1, 60)
        db.commit()

        # (5*80 + 5*60 + 10*0) / 20 = 35 — a learner who has done a third of the
        # document does not read as "understands the document".
        assert scoring.document_understanding(db, u.id, doc.id) == 35
        # One section at/above the mastery threshold, out of three.
        assert scoring.document_completion(db, u.id, doc.id) == 33


def test_full_document_scores_on_merit(client):
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Full", minutes=[5, 5])
        _progress(db, u, doc, 0, 90)
        _progress(db, u, doc, 1, 80)
        db.commit()
        assert scoring.document_understanding(db, u.id, doc.id) == 85
        assert scoring.document_completion(db, u.id, doc.id) == 100
        assert scoring.band(85) == "Proficient"
        assert scoring.band(95) == "Mastered"
        assert scoring.band(None) == "Not started"


def test_a_weak_later_sitting_cannot_erase_a_demonstrated_section(client):
    """The production bug: opening a document and saying nothing dropped a
    learner's understanding to whatever that sitting scored."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Best", minutes=[5])
        db.commit()

        strong = {"scoreable": True, "score": 85, "covered": 100, "topics": []}
        weak = {"scoreable": True, "score": 20, "covered": 40, "topics": []}
        thin = {"scoreable": False, "score": None, "topics": []}

        for result in (strong, weak, thin):
            scoring.apply_session(
                db,
                user=u,
                document=doc,
                module_idx=0,
                transcript=[{"role": "learner", "text": "..."}],
                result=result,
                paused=False,
                total_sections=1,
            )
        db.commit()

        assert scoring.section_bests(db, u.id, doc.id) == {0: 85}
        assert scoring.document_understanding(db, u.id, doc.id) == 85

        rows = (
            db.query(models.SectionProgress)
            .filter_by(user_id=u.id, document_id=doc.id, module_idx=0)
            .one()
        )
        assert rows.last_score == 20  # the most recent SCOREABLE attempt
        assert rows.attempts == 2  # the unscoreable sitting is not an attempt


def test_unscoreable_sitting_is_recorded_but_scores_null(client):
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Null", minutes=[5])
        db.commit()
        row = scoring.apply_session(
            db,
            user=u,
            document=doc,
            module_idx=0,
            transcript=[{"role": "learner", "text": "yeah ok"}],
            result={"scoreable": False, "score": None, "topics": []},
            paused=False,
            total_sections=1,
        )
        db.commit()
        assert row.score is None
        assert row.transcript  # the transcript is kept for audit
        assert scoring.document_understanding(db, u.id, doc.id) is None
        assert scoring.user_understanding(db, u.id) is None


def test_cohort_scoping(client):
    with SessionLocal() as db:
        ws, u, doc_a = _fixture(db, name="Scope", minutes=[5])
        doc_b = models.Document(workspace_id=ws.id, name="Scope Doc B", chunk_count=1)
        db.add(doc_b)
        db.flush()
        db.add(models.Module(document_id=doc_b.id, idx=0, title="B0", minutes=5))
        _progress(db, u, doc_a, 0, 60)
        _progress(db, u, doc_b, 0, 90)

        c_all = models.Cohort(workspace_id=ws.id, name="All")
        c_a = models.Cohort(workspace_id=ws.id, name="OnlyA")
        db.add_all([c_all, c_a])
        db.flush()
        db.add_all(
            [
                models.CohortDocument(cohort_id=c_all.id, document_id=doc_a.id, idx=0),
                models.CohortDocument(cohort_id=c_all.id, document_id=doc_b.id, idx=1),
                models.CohortMember(cohort_id=c_all.id, user_id=u.id),
                models.CohortDocument(cohort_id=c_a.id, document_id=doc_a.id, idx=0),
                models.CohortMember(cohort_id=c_a.id, user_id=u.id),
            ]
        )
        db.commit()

        assert scoring.user_understanding(db, u.id) == 75  # mean(60, 90)
        assert scoring.cohort_understanding(db, c_all.id) == 75
        assert scoring.cohort_understanding(db, c_a.id) == 60


def test_next_section_is_the_first_one_not_yet_sat(client):
    """Resume follows PROGRESS, not mastery. It used to return the first section
    below the mastery threshold, which permanently pinned anyone who scored just
    under it — see test_a_weak_section_does_not_lock_a_learner_out_of_the_document."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Next", minutes=[5, 5, 5])
        db.add_all([
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=0,
                                   best_score=85, last_score=85, attempts=1),
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=1,
                                   best_score=40, last_score=40, attempts=1),
        ])
        db.commit()
        # Section 2 was weak but it WAS sat, so the learner moves on to section 3.
        assert scoring.next_section_idx(db, u.id, doc.id, 3) == 2


def test_completing_the_last_section_unlocks_the_next_document(client):
    """Whether a score arrives live or from a re-grade, finishing a document has
    to open the next one — otherwise the learner is stuck on a document that
    reads 100% complete."""
    from lms_app.config import settings

    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Unlock", minutes=[5, 5])
        nxt = models.Document(workspace_id=ws.id, name="Unlock Next", chunk_count=1)
        db.add(nxt)
        db.flush()
        db.add(models.Module(document_id=nxt.id, idx=0, title="N0", minutes=5))
        db.add_all([
            models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="in_progress"),
            models.LearningPathItem(user_id=u.id, document_id=nxt.id, idx=1, status="locked"),
        ])
        _progress(db, u, doc, 0, 90)
        _progress(db, u, doc, 1, 88)
        db.commit()

        scoring.refresh_path_item(db, user_id=u.id, document_id=doc.id, total_sections=2)
        db.commit()

        items = {i.document_id: i.status for i in db.query(models.LearningPathItem).all()}
        assert items[doc.id] == "mastered"
        assert items[nxt.id] == "up_next", "the next document must open"
        assert scoring.document_completion(db, u.id, doc.id) == 100
        assert scoring.document_understanding(db, u.id, doc.id) >= settings.MASTERY_THRESHOLD


def test_score_index_agrees_exactly_with_the_per_call_functions(client):
    """The batched read model must be a change in HOW inputs are fetched, never
    in what the numbers mean. Any drift here silently rewrites everyone's score."""
    with SessionLocal() as db:
        ws, u1, doc_a = _fixture(db, name="Agree", minutes=[5, 6, 4])
        u2 = models.User(clerk_id="agree_u2", workspace_id=ws.id, name="Second",
                         email="s@agree.dev", role="Learner")
        doc_b = models.Document(workspace_id=ws.id, name="Agree B", chunk_count=2)
        db.add_all([u2, doc_b])
        db.flush()
        for i, m in enumerate([7, 3]):
            db.add(models.Module(document_id=doc_b.id, idx=i, title=f"B{i}", minutes=m))
        # A deliberately uneven spread: partial, complete, untouched, zero.
        _progress(db, u1, doc_a, 0, 90)
        _progress(db, u1, doc_a, 2, 55)
        _progress(db, u1, doc_b, 0, 71)
        _progress(db, u2, doc_a, 1, 100)
        db.commit()

        idx = scoring.ScoreIndex(db, ws.id)
        for u in (u1, u2):
            assert idx.started_document_ids(u.id) == scoring.started_document_ids(db, u.id)
            assert idx.user_understanding(u.id) == scoring.user_understanding(db, u.id)
            for doc in (doc_a, doc_b):
                assert idx.section_bests(u.id, doc.id) == scoring.section_bests(db, u.id, doc.id)
                assert idx.document_understanding(u.id, doc.id) == scoring.document_understanding(
                    db, u.id, doc.id
                )
                assert idx.document_completion(u.id, doc.id) == scoring.document_completion(
                    db, u.id, doc.id
                )
            # Scoped to a subset of documents, as cohort figures are.
            assert idx.user_understanding(u.id, [doc_a.id]) == scoring.user_understanding(
                db, u.id, [doc_a.id]
            )


def test_score_index_matches_cohort_and_team_rollups(client):
    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Roll", minutes=[5, 5])
        _progress(db, u, doc, 0, 80)
        c = models.Cohort(workspace_id=ws.id, name="RollCohort")
        t = models.Team(workspace_id=ws.id, name="RollTeam")
        db.add_all([c, t])
        db.flush()
        db.add_all([
            models.CohortDocument(cohort_id=c.id, document_id=doc.id, idx=0),
            models.CohortMember(cohort_id=c.id, user_id=u.id),
            models.TeamDocument(team_id=t.id, document_id=doc.id, idx=0),
            models.TeamMember(team_id=t.id, user_id=u.id),
        ])
        db.commit()

        idx = scoring.ScoreIndex(db, ws.id)
        assert idx.group_understanding(
            scoring.cohort_member_ids(db, c.id), scoring.cohort_document_ids(db, c.id)
        ) == scoring.cohort_understanding(db, c.id)
        assert idx.group_completion(
            scoring.cohort_member_ids(db, c.id), scoring.cohort_document_ids(db, c.id)
        ) == scoring.cohort_completion(db, c.id)
        assert idx.group_understanding([u.id], [doc.id]) == scoring.team_understanding(db, t.id)


def test_a_weak_section_does_not_lock_a_learner_out_of_the_document(client):
    """Resume is a bookmark, not a gate. Requiring mastery to move on meant a
    learner stuck at 65 on section 2 was returned there forever and could never
    reach section 3 — observed in production across four sittings."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="NotStuck", minutes=[5, 5, 5])
        db.add_all([
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=0,
                                   best_score=92, last_score=92, attempts=1),
            # Sat repeatedly, never reached the mastery threshold.
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=1,
                                   best_score=65, last_score=50, attempts=4),
        ])
        db.commit()

        assert scoring.next_section_idx(db, u.id, doc.id, 3) == 2, "must move on to section 3"
        # The score still tells the truth: that section is NOT mastered.
        assert scoring.document_completion(db, u.id, doc.id) == 33
        assert scoring.document_understanding(db, u.id, doc.id) < settings.MASTERY_THRESHOLD


def test_resume_returns_to_the_first_unsat_section(client):
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Bookmark", minutes=[5, 5, 5])
        db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=0,
                                      best_score=80, last_score=80, attempts=1))
        db.commit()
        assert scoring.next_section_idx(db, u.id, doc.id, 3) == 1


def test_progress_reflects_sections_sat_not_sections_mastered(client):
    """A learner who had sat two of three sections was shown 0% progress, because
    the bar was driven by the MASTERY count. That reads as 'none of that
    counted'. Progress says where you are; mastery still gates advancement."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Progress", minutes=[5, 6, 4])
        db.add_all([
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=0,
                                   best_score=68, last_score=68, attempts=2),
            models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=1,
                                   best_score=62, last_score=62, attempts=1),
        ])
        db.commit()

        assert scoring.document_progress(db, u.id, doc.id) == 67, "two of three sections sat"
        assert scoring.document_completion(db, u.id, doc.id) == 0, "neither reached mastery"
        # The understanding figure is unchanged and still weighted by minutes.
        assert scoring.document_understanding(db, u.id, doc.id) == 47

        idx = scoring.ScoreIndex(db, doc.workspace_id)
        assert idx.document_progress(u.id, doc.id) == scoring.document_progress(db, u.id, doc.id)
        assert idx.document_completion(u.id, doc.id) == scoring.document_completion(db, u.id, doc.id)


def test_a_sat_but_unscored_section_still_counts_as_progress(client):
    """Sitting a section and saying too little to be graded is still progress
    through the document, even though it earns no score."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Unscored", minutes=[5, 5])
        db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=0,
                                      best_score=None, last_score=None, attempts=1))
        db.commit()
        assert scoring.document_progress(db, u.id, doc.id) == 50
        assert scoring.document_understanding(db, u.id, doc.id) is None


def test_one_weak_section_does_not_veto_a_document(client):
    """Mastery gates on the document score, not on every section clearing the bar
    individually. A learner scoring 92, 65 and 75 averages 77 — they have plainly
    understood it, and one section five points short used to keep them off the
    next document forever."""
    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Veto", minutes=[5, 5, 5])
        nxt = models.Document(workspace_id=ws.id, name="Veto Next", chunk_count=1)
        db.add(nxt)
        db.flush()
        db.add(models.Module(document_id=nxt.id, idx=0, title="N", minutes=5))
        db.add_all([
            models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="in_progress"),
            models.LearningPathItem(user_id=u.id, document_id=nxt.id, idx=1, status="locked"),
        ])
        for idx, score in ((0, 92), (1, 65), (2, 75)):
            db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=idx,
                                          best_score=score, last_score=score, attempts=1))
        db.commit()

        assert scoring.document_understanding(db, u.id, doc.id) == 77
        assert scoring.document_completion(db, u.id, doc.id) == 67, "one section below the bar"

        scoring.refresh_path_item(db, user_id=u.id, document_id=doc.id, total_sections=3)
        db.commit()
        items = {i.document_id: i.status for i in db.query(models.LearningPathItem).all()}
        assert items[doc.id] == "mastered"
        assert items[nxt.id] == "up_next"


def test_finishing_every_section_opens_the_next_document_whatever_the_score(client):
    """Working through a document is what unlocks the next one. The score says how
    well it went; it does not decide whether the learner may move on.

    Scores of 62, 55 and 48 average 55 — well short of mastery — but the learner
    has sat every section of the document. Holding them there means their only
    route forward is a grader they cannot see or argue with, on a threshold they
    were never shown."""
    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Onward", minutes=[5, 5, 5])
        nxt = models.Document(workspace_id=ws.id, name="Onward Next", chunk_count=1)
        db.add(nxt)
        db.flush()
        db.add(models.Module(document_id=nxt.id, idx=0, title="N", minutes=5))
        db.add_all([
            models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="in_progress"),
            models.LearningPathItem(user_id=u.id, document_id=nxt.id, idx=1, status="locked"),
        ])
        for idx, score in ((0, 62), (1, 55), (2, 48)):
            db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=idx,
                                          best_score=score, last_score=score, attempts=1))
        db.commit()

        assert scoring.document_understanding(db, u.id, doc.id) == 55
        scoring.refresh_path_item(db, user_id=u.id, document_id=doc.id, total_sections=3)
        db.commit()

        items = {i.document_id: i.status for i in db.query(models.LearningPathItem).all()}
        assert items[doc.id] == "completed", "finished, but not mastered — and that is said plainly"
        assert items[nxt.id] == "up_next", "the next document must open regardless of score"


def test_a_finished_document_still_reports_mastery_honestly(client):
    """Unlocking the next document must not quietly relabel a weak pass as mastery.
    The learner moves on; the number still tells the truth about how it went."""
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Honest", minutes=[5, 5])
        db.add(models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="in_progress"))
        for idx, score in ((0, 50), (1, 40)):
            db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=idx,
                                          best_score=score, last_score=score, attempts=1))
        db.commit()
        scoring.refresh_path_item(db, user_id=u.id, document_id=doc.id, total_sections=2)
        db.commit()

        item = db.query(models.LearningPathItem).filter_by(user_id=u.id, document_id=doc.id).one()
        assert item.status == "completed"
        assert scoring.document_understanding(db, u.id, doc.id) == 45
        assert scoring.band(45) == "Progressing", "not dressed up as a pass"
        assert scoring.document_completion(db, u.id, doc.id) == 0, "no section reached mastery"


def test_recomputing_a_path_does_not_open_documents_the_learner_never_touched(client):
    """refresh_path_item must be safe to call for any document, not just the one
    the learner just sat. It used to mark every non-finished document in_progress,
    so recomputing a whole path unlocked everything at once."""
    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Untouched", minutes=[5])
        later = models.Document(workspace_id=ws.id, name="Untouched Later", chunk_count=1)
        db.add(later)
        db.flush()
        db.add(models.Module(document_id=later.id, idx=0, title="L", minutes=5))
        db.add_all([
            models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="up_next"),
            models.LearningPathItem(user_id=u.id, document_id=later.id, idx=1, status="locked"),
        ])
        db.commit()

        for d in (doc, later):
            scoring.refresh_path_item(db, user_id=u.id, document_id=d.id, total_sections=1)
        db.commit()

        items = {i.document_id: i.status for i in db.query(models.LearningPathItem).all()}
        assert items[doc.id] == "up_next", "not started is not in progress"
        assert items[later.id] == "locked", "a locked document must stay locked"


def test_a_document_is_not_mastered_with_a_section_never_sat(client):
    """The other half: strong scores on two of three sections must NOT unlock the
    next document while a section is untouched."""
    with SessionLocal() as db:
        ws, u, doc = _fixture(db, name="Unsat", minutes=[5, 5, 5])
        db.add(models.LearningPathItem(user_id=u.id, document_id=doc.id, idx=0, status="in_progress"))
        for idx, score in ((0, 95), (1, 95)):
            db.add(models.SectionProgress(user_id=u.id, document_id=doc.id, module_idx=idx,
                                          best_score=score, last_score=score, attempts=1))
        db.commit()
        scoring.refresh_path_item(db, user_id=u.id, document_id=doc.id, total_sections=3)
        db.commit()
        item = db.query(models.LearningPathItem).filter_by(user_id=u.id, document_id=doc.id).one()
        assert item.status == "in_progress"
