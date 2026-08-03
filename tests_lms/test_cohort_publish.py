"""Publishing a cohort gives its learners a fresh path for documents they've already
completed (first publish re-opens them), while routine re-publishing stays
non-destructive — and neither ever discards a demonstrated section score."""

from lms_app import models
from lms_app.db import SessionLocal
from lms_app.routers import cohorts


def test_fresh_publish_reopens_completed_doc(client):
    db = SessionLocal()
    try:
        ws = models.Workspace(name="Pub WS", slug="pub-ws")
        db.add(ws)
        db.flush()
        admin = models.User(
            workspace_id=ws.id, name="Admin", email="a@pub.test", role="Admin", clerk_id="ck_pub_a"
        )
        learner = models.User(
            workspace_id=ws.id, name="Learner", email="l@pub.test", role="Learner", clerk_id="ck_pub_l"
        )
        db.add_all([admin, learner])
        db.flush()
        doc = models.Document(workspace_id=ws.id, name="Pub Doc", chunk_count=3, status="Indexed")
        db.add(doc)
        db.flush()
        for i in range(3):
            db.add(
                models.Module(
                    document_id=doc.id, idx=i, title=f"S{i}", description="", topics=[],
                    minutes=5, chunk_start=i, chunk_end=i + 1,
                )
            )
        # The learner already COMPLETED this document.
        item = models.LearningPathItem(
            user_id=learner.id, document_id=doc.id, idx=0, status="mastered"
        )
        db.add(item)
        for i in range(3):
            db.add(
                models.SectionProgress(
                    user_id=learner.id, document_id=doc.id, module_idx=i,
                    status="completed", best_score=90, last_score=90, attempts=1,
                )
            )
        c = models.Cohort(workspace_id=ws.id, name="Pub Cohort", published=False)
        db.add(c)
        db.flush()
        db.add(models.CohortDocument(cohort_id=c.id, document_id=doc.id, idx=0))
        db.add(models.CohortMember(cohort_id=c.id, user_id=learner.id))
        db.commit()

        # First publish → re-opens the completed document as a fresh path...
        cohorts.publish_cohort(c.id, user=admin, db=db)
        db.refresh(item)
        db.refresh(c)
        assert c.published is True
        assert item.status == "up_next"

        # ...but WITHOUT discarding what the learner already demonstrated. Wiping
        # scores on re-publish would mean an admin re-publishing a cohort silently
        # zeroed everyone's record.
        first = (
            db.query(models.SectionProgress)
            .filter_by(user_id=learner.id, document_id=doc.id, module_idx=0)
            .one()
        )
        assert first.best_score == 90
        assert first.status == "in_progress"

        # A routine RE-publish must not touch a learner who has finished again.
        item.status = "mastered"
        db.commit()
        cohorts.publish_cohort(c.id, user=admin, db=db)
        db.refresh(item)
        assert item.status == "mastered", "re-publish must be non-destructive"
    finally:
        db.close()


def test_publishing_a_multi_document_cohort_orders_the_path(client):
    """Each document must land at its own position on the path, in curriculum
    order. They all used to be seeded with the same idx, so "which document
    unlocks next" came down to whatever order the database returned."""
    db = SessionLocal()
    try:
        ws = models.Workspace(name="Order WS", slug="order-ws")
        db.add(ws)
        db.flush()
        admin = models.User(
            workspace_id=ws.id, name="Admin", email="a@ord.test", role="Admin", clerk_id="ck_ord_a"
        )
        learner = models.User(
            workspace_id=ws.id, name="Learner", email="l@ord.test", role="Learner", clerk_id="ck_ord_l"
        )
        db.add_all([admin, learner])
        db.flush()

        c = models.Cohort(workspace_id=ws.id, name="Ordered Cohort", published=False)
        db.add(c)
        db.flush()
        db.add(models.CohortMember(cohort_id=c.id, user_id=learner.id))
        doc_ids = []
        for n in range(3):
            doc = models.Document(
                workspace_id=ws.id, name=f"Chapter {n + 1}", chunk_count=1, status="Indexed"
            )
            db.add(doc)
            db.flush()
            db.add(
                models.Module(
                    document_id=doc.id, idx=0, title="S0", description="", topics=[],
                    minutes=5, chunk_start=0, chunk_end=1,
                )
            )
            db.add(models.CohortDocument(cohort_id=c.id, document_id=doc.id, idx=n))
            doc_ids.append(doc.id)
        db.commit()

        cohorts.publish_cohort(c.id, user=admin, db=db)

        items = (
            db.query(models.LearningPathItem)
            .filter_by(user_id=learner.id)
            .order_by(models.LearningPathItem.idx)
            .all()
        )
        assert [i.idx for i in items] == [0, 1, 2], "every document gets its own position"
        assert [i.document_id for i in items] == doc_ids, "in curriculum order"
        assert [i.status for i in items] == ["up_next", "locked", "locked"], (
            "only the first is open; the rest unlock as the learner finishes"
        )
    finally:
        db.close()
