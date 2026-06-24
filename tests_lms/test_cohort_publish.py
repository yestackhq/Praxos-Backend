"""Publishing a cohort gives its learners a fresh path for documents they've already
completed (first publish re-opens them), while routine re-publishing stays non-destructive."""

from lms_app.db import SessionLocal
from lms_app import models
from lms_app.routers import cohorts


def test_fresh_publish_reopens_completed_doc(client):
    db = SessionLocal()
    try:
        ws = models.Workspace(name="Pub WS", slug="pub-ws")
        db.add(ws)
        db.flush()
        admin = models.User(workspace_id=ws.id, name="Admin", email="a@pub.test", role="Admin", clerk_id="ck_pub_a")
        learner = models.User(workspace_id=ws.id, name="Learner", email="l@pub.test", role="Learner", clerk_id="ck_pub_l")
        db.add_all([admin, learner])
        db.flush()
        doc = models.Document(workspace_id=ws.id, name="Pub Doc", sections=3, status="Indexed")
        db.add(doc)
        db.flush()
        for i in range(3):
            db.add(
                models.Module(
                    document_id=doc.id, idx=i, title=f"S{i}", description="", topics=[],
                    minutes=5, chunk_start=i, chunk_end=i + 1,
                )
            )
        # The learner already COMPLETED this document (mastered + section progress at the end).
        item = models.LearningPathItem(
            user_id=learner.id, idx=0, title=doc.name, sections=3, status="mastered", progress=100
        )
        prog = models.SectionProgress(
            user_id=learner.id, document_id=doc.id, module_idx=2, status="completed", score=90
        )
        db.add_all([item, prog])
        # A fresh (draft) cohort containing that document + learner.
        c = models.Cohort(workspace_id=ws.id, name="Pub Cohort", published=False)
        db.add(c)
        db.flush()
        db.add(models.CohortDocument(cohort_id=c.id, document_id=doc.id, idx=0))
        db.add(models.CohortMember(cohort_id=c.id, user_id=learner.id))
        db.commit()

        # First publish → re-opens the completed document as a fresh path.
        cohorts.publish_cohort(c.id, user=admin, db=db)
        db.refresh(item)
        db.refresh(prog)
        db.refresh(c)
        assert c.published is True
        assert item.status == "up_next", item.status
        assert item.progress == 0
        assert prog.module_idx == 0
        assert prog.status == "in_progress"
        assert prog.score is None

        # Learner finishes again; a routine RE-publish must NOT wipe their completion.
        item.status = "mastered"
        item.progress = 100
        prog.module_idx = 2
        prog.status = "completed"
        prog.score = 88
        db.commit()
        cohorts.publish_cohort(c.id, user=admin, db=db)
        db.refresh(item)
        db.refresh(prog)
        assert item.status == "mastered", "re-publish must be non-destructive"
        assert item.progress == 100
        assert prog.module_idx == 2
        assert prog.status == "completed"
    finally:
        db.close()
