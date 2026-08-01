"""Seed demonstration progress for named learners on one document.

For demos and screenshots: makes a learner read as having fully understood a
document, with the Continue-learning card still pointing at it so a session can
be started from there.

    python -m scripts.seed_demo_progress --doc 4 --user 9 --user 11
    python -m scripts.seed_demo_progress --doc 4 --user 9 --user 11 --apply
    python -m scripts.seed_demo_progress --doc 4 --user 9 --user 11 --apply --mastered
    python -m scripts.seed_demo_progress --doc 4 --user 9 --user 11 --clear --apply

``--mastered`` marks the document COMPLETE on the learner's path instead of
leaving it active. It also unlocks the next document, exactly as finishing a
document for real does (see scoring._refresh_path_item) — otherwise the learner
is left with a completed document and nothing to continue to.

THESE ARE NOT REAL ASSESSMENTS. Every row this writes is tagged with
``SEED_MARKER`` in its summary and carries a transcript saying so, because the
whole point of this system is that a score means someone demonstrated something.
An untagged fake score is indistinguishable from an earned one in the admin UI,
and someone would eventually read it as evidence about a real person. ``--clear``
removes exactly the rows this wrote, and nothing else.
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from lms_app import models, plan as plan_service, scoring
from lms_app.db import SessionLocal

SEED_MARKER = "[DEMO SEED — not a real assessment]"

# Per-section scores, cycled. High enough to read as mastery, varied enough not
# to look like a fill-down.
SCORES = [92, 90, 88, 91, 89, 93, 87, 90]


def _clear(db, user_id: int, document_id: int) -> int:
    rows = db.scalars(
        select(models.LearningSession).where(
            models.LearningSession.user_id == user_id,
            models.LearningSession.document_id == document_id,
        )
    ).all()
    removed = 0
    for r in rows:
        if r.summary and r.summary.startswith(SEED_MARKER):
            db.delete(r)
            removed += 1
    # Only reset progress rows if this document has no real sessions left.
    real = [r for r in rows if not (r.summary or "").startswith(SEED_MARKER)]
    if not real:
        for p in db.scalars(
            select(models.SectionProgress).where(
                models.SectionProgress.user_id == user_id,
                models.SectionProgress.document_id == document_id,
            )
        ).all():
            p.best_score = None
            p.last_score = None
            p.attempts = 0
            p.status = "in_progress"
    return removed


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    clear = "--clear" in argv
    mastered = "--mastered" in argv
    doc_ids = [int(argv[i + 1]) for i, a in enumerate(argv) if a == "--doc" and i + 1 < len(argv)]
    user_ids = [int(argv[i + 1]) for i, a in enumerate(argv) if a == "--user" and i + 1 < len(argv)]
    if not doc_ids or not user_ids:
        print("need --doc <id> and at least one --user <id>")
        return 2
    document_id = doc_ids[0]

    with SessionLocal() as db:
        doc = db.get(models.Document, document_id)
        if doc is None:
            print(f"no document {document_id}")
            return 2
        mods = plan_service.get_modules(db, document_id)
        if not mods:
            print(f"document {document_id} has no teaching plan — generate one first")
            return 2

        print(f"document {document_id}: {doc.name}  ({len(mods)} sections)")
        print(f"mode: {'CLEAR' if clear else 'SEED'}{'' if apply else '  (dry run)'}\n")

        for uid in user_ids:
            user = db.get(models.User, uid)
            if user is None:
                print(f"  user {uid}: not found — skipped")
                continue

            if clear:
                if apply:
                    n = _clear(db, uid, document_id)
                    db.commit()
                    print(f"  {user.name}: removed {n} seeded session(s), progress reset")
                else:
                    print(f"  {user.name}: would remove seeded sessions and reset progress")
                continue

            if not apply:
                print(f"  {user.name}: would mark {len(mods)} section(s) demonstrated")
                continue

            _clear(db, uid, document_id)  # idempotent: replace previous seed

            for m in mods:
                score = SCORES[m.idx % len(SCORES)]
                db.add(
                    models.LearningSession(
                        user_id=uid,
                        document_id=document_id,
                        module_idx=m.idx,
                        score=score,
                        covered=100,
                        summary=f"{SEED_MARKER} Section {m.idx + 1}: {m.title}",
                        topics=[{"name": kp[:80], "score": score, "evidence": SEED_MARKER}
                                for kp in (m.key_points or [])[:3]],
                        strengths=[f"{SEED_MARKER} seeded for demonstration"],
                        gaps=[],
                        # Deliberately NOT a plausible fake conversation. Anyone
                        # auditing this score must find a marker, not a transcript
                        # of words the learner never said.
                        transcript=[{"role": "tutor", "text": SEED_MARKER}],
                        learner_turns=0,
                        paused=False,
                        ended_at=models.utcnow(),
                    )
                )
                prog = db.scalar(
                    select(models.SectionProgress).where(
                        models.SectionProgress.user_id == uid,
                        models.SectionProgress.document_id == document_id,
                        models.SectionProgress.module_idx == m.idx,
                    )
                )
                if prog is None:
                    prog = models.SectionProgress(
                        user_id=uid, document_id=document_id, module_idx=m.idx
                    )
                    db.add(prog)
                prog.best_score = score
                prog.last_score = score
                prog.attempts = 1
                prog.status = "completed"
                prog.updated_at = models.utcnow()

            # --mastered  -> the document reads as COMPLETE on the path, and the
            #                next one opens, mirroring a real completion.
            # default     -> the document stays ACTIVE, so the Continue-learning
            #                card keeps pointing at it.
            item = db.scalar(
                select(models.LearningPathItem).where(
                    models.LearningPathItem.user_id == uid,
                    models.LearningPathItem.document_id == document_id,
                )
            )
            if item is None:
                item = models.LearningPathItem(
                    user_id=uid, document_id=document_id, idx=0, status="in_progress"
                )
                db.add(item)
                db.flush()
            item.status = "mastered" if mastered else "in_progress"
            item.idx = 0

            if mastered:
                # Open the next document. Without this the learner has a completed
                # document and nothing to continue to, because a previous seed run
                # locked everything else so it would not compete for the card.
                nxt = db.scalar(
                    select(models.LearningPathItem)
                    .where(
                        models.LearningPathItem.user_id == uid,
                        models.LearningPathItem.document_id != document_id,
                        models.LearningPathItem.status == "locked",
                    )
                    .order_by(models.LearningPathItem.idx, models.LearningPathItem.document_id)
                )
                if nxt is not None:
                    nxt.status = "up_next"
            else:
                # Anything else up_next would compete for the card.
                for other in db.scalars(
                    select(models.LearningPathItem).where(
                        models.LearningPathItem.user_id == uid,
                        models.LearningPathItem.document_id != document_id,
                        models.LearningPathItem.status == "up_next",
                    )
                ).all():
                    other.status = "locked"
            db.commit()

            u = scoring.document_understanding(db, uid, document_id)
            c = scoring.document_completion(db, uid, document_id)
            nxt_doc = db.scalar(
                select(models.LearningPathItem).where(
                    models.LearningPathItem.user_id == uid,
                    models.LearningPathItem.status.in_(["in_progress", "up_next"]),
                ).order_by(models.LearningPathItem.idx)
            )
            nxt_name = ""
            if nxt_doc is not None:
                nd = db.get(models.Document, nxt_doc.document_id)
                nxt_name = nd.name[:38] if nd else str(nxt_doc.document_id)
            print(
                f"  {user.name}: {len(mods)} sections demonstrated — "
                f"understanding {u}/100 ({scoring.band(u)}), completion {c}%, "
                f"path status '{item.status}', next up: {nxt_name or 'nothing'}"
            )

        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
