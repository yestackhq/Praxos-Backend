from __future__ import annotations

"""Cohort-scoped, recency-weighted (EMA) understanding scoring."""

from lms_app.db import SessionLocal
from lms_app import models, workspace


def test_ema_recency_cohort_scoping_and_bands(client):  # client fixture creates the schema
    with SessionLocal() as db:
        ws = models.Workspace(name="ScoreCo", plan="Admin workspace")
        db.add(ws)
        db.flush()
        u = models.User(clerk_id="score_u", workspace_id=ws.id, name="S", email="s@score.dev", role="Learner")
        db.add(u)
        db.flush()
        d1 = models.Document(workspace_id=ws.id, name="Doc A", sections=1)
        d2 = models.Document(workspace_id=ws.id, name="Doc B", sections=1)
        db.add_all([d1, d2])
        db.flush()
        # Doc A attempts in order: 40, 0 (ignored), 80 → EMA = 0.5*80 + 0.5*40 = 60
        for sc in [40, 0, 80]:
            db.add(models.LearningSession(user_id=u.id, doc="Doc A", date="2026-06-23", score=sc))
        db.add(models.LearningSession(user_id=u.id, doc="Doc B", date="2026-06-23", score=90))
        db.commit()

        assert workspace._doc_ema(db, u.id, "Doc A") == 60  # recency-weighted, 0 skipped
        assert workspace._doc_ema(db, u.id, "Doc B") == 90
        assert workspace._user_understanding(db, u.id) == 75  # avg(60, 90)
        assert workspace.understanding_band(60) == "Progressing"
        assert workspace.understanding_band(90) == "Mastered"
        assert workspace.understanding_band(None) == "Not started"

        # Cohort over BOTH docs → 75; cohort scoped to Doc A only → 60 (cohort scoping works).
        c_all = models.Cohort(workspace_id=ws.id, name="All", status="On track")
        c_a = models.Cohort(workspace_id=ws.id, name="OnlyA", status="On track")
        db.add_all([c_all, c_a])
        db.flush()
        db.add_all(
            [
                models.CohortDocument(cohort_id=c_all.id, document_id=d1.id, idx=0),
                models.CohortDocument(cohort_id=c_all.id, document_id=d2.id, idx=1),
                models.CohortMember(cohort_id=c_all.id, user_id=u.id),
                models.CohortDocument(cohort_id=c_a.id, document_id=d1.id, idx=0),
                models.CohortMember(cohort_id=c_a.id, user_id=u.id),
            ]
        )
        db.commit()

        assert workspace._cohort_understanding(db, c_all.id) == 75
        assert workspace._cohort_understanding(db, c_a.id) == 60
