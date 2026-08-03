"""Recompute every learner's path standing against the current unlock rule.

Path status is written when a sitting ends, so a rule change does not reach the
learners already sitting under the old one. Anyone who had worked through every
section of a document but scored below the mastery threshold is still marked
in_progress with the next document locked — which is exactly the state the rule
change exists to end.

    python -m scripts.refresh_paths            # report what would change
    python -m scripts.refresh_paths --apply

It also renumbers each learner's path. Paths seeded before the publish fix gave
every document of a cohort the same ``idx``, so "the next document" was decided
by whatever order Postgres returned — renumbering pins the order that (idx, id)
already resolves to, which is the order the documents were published in.

Statuses only ever come from scoring.refresh_path_item, so this can produce no
state a live session could not. Running it twice changes nothing the second time.
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from lms_app import models, plan as plan_service, scoring
from lms_app.db import SessionLocal


def main(apply: bool) -> int:
    with SessionLocal() as db:
        items = db.scalars(
            select(models.LearningPathItem).order_by(
                models.LearningPathItem.user_id,
                models.LearningPathItem.idx,
                models.LearningPathItem.id,
            )
        ).all()
        before = {i.id: i.status for i in items}
        before_idx = {i.id: i.idx for i in items}

        # Renumber in the order the app already reads them, so nothing moves.
        seen: dict[int, int] = {}
        for i in items:
            i.idx = seen.get(i.user_id, 0)
            seen[i.user_id] = i.idx + 1
        renumbered = sum(1 for i in items if i.idx != before_idx[i.id])

        sections: dict[int, int] = {}
        for item in items:
            # Skip documents already marked done. Their status cannot change (a
            # best score never falls), and re-running refresh_path_item on them
            # would fire their unlock a SECOND time — granting a learner an extra
            # open document for a completion that was already credited.
            if before[item.id] in scoring.DONE_STATUSES:
                continue
            if item.document_id not in sections:
                sections[item.document_id] = len(plan_service.get_modules(db, item.document_id))
            scoring.refresh_path_item(
                db,
                user_id=item.user_id,
                document_id=item.document_id,
                total_sections=sections[item.document_id],
            )

        # Unlocks land on OTHER rows than the one being refreshed, so read the
        # whole set back rather than trusting the loop's own view.
        db.flush()

        # Nobody may be left with nothing to study while documents sit locked —
        # the one case where a missed unlock has to be repaired rather than
        # replayed. Idempotent: it does nothing once a learner has an open document.
        by_user: dict[int, list[models.LearningPathItem]] = {}
        for i in items:
            by_user.setdefault(i.user_id, []).append(i)
        for rows in by_user.values():
            if any(r.status in ("in_progress", "up_next") for r in rows):
                continue
            nxt = next((r for r in sorted(rows, key=lambda r: r.idx) if r.status == "locked"), None)
            if nxt is not None:
                nxt.status = "up_next"

        changed = [i for i in items if i.status != before[i.id]]

        names = {
            u.id: u.name for u in db.scalars(select(models.User)).all()
        }
        docs = {
            d.id: d.name for d in db.scalars(select(models.Document)).all()
        }
        for i in changed:
            print(
                f"  {names.get(i.user_id, i.user_id)}: {docs.get(i.document_id, i.document_id)[:50]}"
                f"  {before[i.id]} -> {i.status}"
            )
        print(f"\n{len(changed)} of {len(items)} path item(s) change status")
        print(f"{renumbered} of {len(items)} path item(s) renumbered")

        if not apply:
            db.rollback()
            print("(dry run — pass --apply)")
            return 0
        db.commit()
        print("applied")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv))
