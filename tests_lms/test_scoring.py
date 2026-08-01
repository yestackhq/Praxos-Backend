from __future__ import annotations

"""Understanding scoring: section bests → weighted document score → learner/cohort.

These pin down the three behaviours that were wrong in production:
  • a sitting with nothing to assess must not become a low score,
  • a later weak sitting must not erase a section already demonstrated,
  • a document score must account for the sections never attempted.
"""

from lms_app import models, scoring
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


def test_next_section_is_the_first_not_yet_mastered(client):
    with SessionLocal() as db:
        _, u, doc = _fixture(db, name="Next", minutes=[5, 5, 5])
        _progress(db, u, doc, 0, 85)
        _progress(db, u, doc, 1, 40)  # below mastery → resume here
        db.commit()
        assert scoring.next_section_idx(db, u.id, doc.id, 3) == 1
