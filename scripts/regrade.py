"""Grade sittings that were recorded but never scored.

A session is stored the moment it ends, transcript and all, even when the
assessor could not be reached — losing a learner's ten-minute conversation
because a provider timed out is not an acceptable failure mode. This picks those
sittings up afterwards.

    python -m scripts.regrade                # list what is ungraded
    python -m scripts.regrade --apply        # grade them
    python -m scripts.regrade --apply --all  # re-grade everything with a transcript

Only sittings with a transcript are eligible: a session where the learner truly
said nothing is unscoreable by design and stays that way.
"""

from __future__ import annotations

import sys
import time

from sqlalchemy import select

from lms_app import ai, memory, models, plan as plan_service, scoring, tutor
from lms_app.db import SessionLocal
from lms_app.meldos import MeldOSError

MAX_ATTEMPTS = 3


def _section(db, doc: models.Document, module_idx: int):
    mods = plan_service.get_modules(db, doc.id)
    cur = mods[module_idx] if mods and 0 <= module_idx < len(mods) else None
    if cur is None:
        return None
    payload = plan_service.module_payload(cur)
    payload["material"] = tutor.section_material(plan_service.section_chunks(doc, cur))
    return payload


def main(apply: bool, redo_all: bool) -> int:
    with SessionLocal() as db:
        q = select(models.LearningSession).order_by(models.LearningSession.id)
        if not redo_all:
            q = q.where(models.LearningSession.score.is_(None))
        rows = [s for s in db.scalars(q).all() if s.transcript]
        print(f"{len(rows)} sitting(s) with a transcript and no score\n")
        if not rows:
            return 0
        if not apply:
            for s in rows:
                learner = sum(1 for t in s.transcript if t.get("role") == "learner")
                print(f"  session {s.id}: user {s.user_id}, doc {s.document_id}, "
                      f"section {s.module_idx + 1}, {learner} learner turn(s)")
            print("\n(dry run — pass --apply)")
            return 0

        ok = failed = skipped = 0
        for n, row in enumerate(rows, 1):
            doc = db.get(models.Document, row.document_id)
            user = db.get(models.User, row.user_id)
            if doc is None or user is None:
                skipped += 1
                continue
            label = f"[{n}/{len(rows)}] session {row.id} (user {user.id}, doc {doc.id}, s{row.module_idx})"

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    result = ai.score_understanding(
                        doc.name,
                        row.transcript,
                        section=_section(db, doc, row.module_idx),
                        prior_facts=memory.prior_understanding(
                            workspace_id=user.workspace_id,
                            user_id=user.id,
                            document_id=doc.id,
                            topic=doc.name,
                        ),
                        # Attribute the re-grade to the learner it belongs to.
                        end_user=ai.llm.EndUser.claimed(user.name),
                        session_id=f"praxos-regrade-{row.id}",
                    )
                    break
                except MeldOSError as exc:
                    if attempt == MAX_ATTEMPTS:
                        print(f"{label}: gave up — MeldOS {exc.status}")
                        result = None
                        break
                    wait = 5 * attempt
                    print(f"{label}: MeldOS {exc.status}, retrying in {wait}s")
                    time.sleep(wait)
            else:
                result = None

            if result is None:
                failed += 1
                continue
            if not result.get("scoreable"):
                print(f"{label}: nothing substantive said — left unscored")
                skipped += 1
                continue

            row.score = result["score"]
            row.covered = int(result.get("covered", 100) or 100)
            row.summary = str(result.get("summary", ""))[:2000]
            row.topics = result.get("topics") or []
            row.strengths = [str(x) for x in (result.get("strengths") or [])]
            row.gaps = [str(x) for x in (result.get("gaps") or [])]

            # Fold it into the learner's standing exactly as a live sitting would.
            prog = db.scalar(
                select(models.SectionProgress).where(
                    models.SectionProgress.user_id == user.id,
                    models.SectionProgress.document_id == doc.id,
                    models.SectionProgress.module_idx == row.module_idx,
                )
            )
            if prog is None:
                prog = models.SectionProgress(
                    user_id=user.id, document_id=doc.id, module_idx=row.module_idx
                )
                db.add(prog)
            prog.attempts = (prog.attempts or 0) + 1
            prog.last_score = row.score
            prog.best_score = row.score if prog.best_score is None else max(prog.best_score, row.score)
            prog.updated_at = models.utcnow()
            # Fold it into the learner's PATH too, not just their score. Without
            # this a re-grade that completed someone's final section left the
            # document at 100% while its path item stayed in_progress, so the
            # next document never unlocked and the learner was stuck.
            scoring.refresh_path_item(
                db,
                user_id=user.id,
                document_id=doc.id,
                total_sections=len(plan_service.get_modules(db, doc.id)),
            )
            db.commit()

            print(f"{label}: {row.score}/100 — {row.summary[:70]}")
            ok += 1

        print(f"\ngraded {ok}, failed {failed}, skipped {skipped}")
        return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv, redo_all="--all" in sys.argv))
