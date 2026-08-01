"""Regenerate teaching plans so every section has key_points and check_questions.

Plans created before the planner rewrite carry neither. Without them the tutor
cannot enforce per-point checks before advancing, and the grader falls back to
marking answers against raw section text instead of named claims — which is the
gap between the scoring fix being deployed and it actually working.

    python -m scripts.regenerate_plans                 # report what is stale
    python -m scripts.regenerate_plans --apply         # regenerate stale plans
    python -m scripts.regenerate_plans --apply --all   # regenerate every plan
    python -m scripts.regenerate_plans --apply --doc 5 --doc 6

Regenerating REPLACES a document's sections, so section indices change. That is
safe only while no learner holds a score against the old indices; the script
refuses to run if any exist unless --force is given.
"""

from __future__ import annotations

import sys
import time

from sqlalchemy import func, select

from lms_app import models, plan as plan_service
from lms_app.db import SessionLocal
from lms_app.meldos import MeldOSError

MAX_ATTEMPTS = 3


def _stale(db, doc_id: int) -> bool:
    mods = plan_service.get_modules(db, doc_id)
    return not mods or not any(m.key_points for m in mods)


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    redo_all = "--all" in argv
    force = "--force" in argv
    only = {int(argv[i + 1]) for i, a in enumerate(argv) if a == "--doc" and i + 1 < len(argv)}

    with SessionLocal() as db:
        scored = (
            db.scalar(
                select(func.count())
                .select_from(models.SectionProgress)
                .where(models.SectionProgress.best_score.is_not(None))
            )
            or 0
        )
        if scored and not force:
            print(
                f"REFUSING: {scored} section_progress row(s) hold a score against the current\n"
                "section indices. Regenerating would re-number the sections those scores\n"
                "belong to. Re-run with --force if you accept that."
            )
            return 2

        docs = db.scalars(select(models.Document).order_by(models.Document.id)).all()
        targets = [
            d
            for d in docs
            if (d.id in only if only else (redo_all or _stale(db, d.id)))
        ]
        print(f"{len(targets)} document(s) to regenerate\n")
        if not apply:
            for d in targets:
                print(f"  would regenerate doc {d.id}: {d.name[:56]}")
            print("\n(dry run — pass --apply)")
            return 0

        ok = failed = 0
        for n, doc in enumerate(targets, 1):
            label = f"[{n}/{len(targets)}] doc {doc.id} {doc.name[:44]}"
            for attempt in range(1, MAX_ATTEMPTS + 1):
                started = time.monotonic()
                try:
                    mods = plan_service.generate_plan(db, doc.id)
                except MeldOSError as exc:
                    wait = 5 * attempt
                    print(f"{label}: MeldOS {exc.status} (attempt {attempt}) — retrying in {wait}s")
                    if attempt == MAX_ATTEMPTS:
                        failed += 1
                        break
                    time.sleep(wait)
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
                    failed += 1
                    break

                elapsed = time.monotonic() - started
                cov = plan_service.plan_coverage(db, doc.id)
                kp = sum(len(m.key_points or []) for m in mods)
                cq = sum(len(m.check_questions or []) for m in mods)
                flag = "" if (cov["complete"] and kp) else "   <-- CHECK"
                print(
                    f"{label}: {len(mods)} sections, {kp} key points, {cq} checks, "
                    f"coverage {cov['covered']}/{cov['chunks']}, {elapsed:.0f}s{flag}"
                )
                ok += 1
                break

        print(f"\nregenerated {ok}, failed {failed}")
        return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
