"""Repair teaching plans that do not cover their document.

Plans generated before the planner fix routinely skipped chunks: the model
emitted inclusive end indices (dropping one chunk at every section boundary) and
the prompt was truncated, so long documents were planned only over the prefix
the model was shown. One live document had 21 of its 46 chunks in no section at
all — that text was never taught to anyone.

This rewrites the CHUNK RANGES of existing modules so the plan tiles [0, n)
exactly once, keeping each section's title, description, topics and order. It
does not call a model, so it works with no provider configured — but it also
cannot re-balance the split: a section that absorbs a large untaught tail stays
large. Regenerating the plan (POST /api/documents/{id}/plan/generate) gives a
better division once an LLM is configured; this makes sure nothing is missed in
the meantime.

    python -m scripts.repair_plan_coverage            # report only
    python -m scripts.repair_plan_coverage --apply    # write the repaired ranges
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from lms_app import models, plan as plan_service
from lms_app.ai import _normalise_coverage
from lms_app.db import SessionLocal


def main(apply: bool) -> int:
    repaired = 0
    with SessionLocal() as db:
        for doc in db.scalars(select(models.Document).order_by(models.Document.id)).all():
            before = plan_service.plan_coverage(db, doc.id)
            if before["complete"]:
                continue
            mods = plan_service.get_modules(db, doc.id)
            if not mods:
                print(f"doc {doc.id}: no plan at all — regenerate it")
                continue

            proposed = [
                {"chunk_start": m.chunk_start, "chunk_end": m.chunk_end, "_id": m.id} for m in mods
            ]
            fixed = _normalise_coverage(proposed, before["chunks"])
            by_id = {f["_id"]: f for f in fixed}

            print(
                f"doc {doc.id} {doc.name[:48]}: "
                f"{before['covered']}/{before['chunks']} chunks, gaps={before['gaps']}"
            )
            for m in mods:
                f = by_id.get(m.id)
                if f is None:
                    # _normalise_coverage drops sections beyond the chunk count
                    # (a 5-section plan over 4 chunks cannot tile).
                    print(f"    section {m.idx} '{m.title[:40]}' — dropped (more sections than chunks)")
                    if apply:
                        db.delete(m)
                    continue
                if (m.chunk_start, m.chunk_end) != (f["chunk_start"], f["chunk_end"]):
                    print(
                        f"    section {m.idx} '{m.title[:40]}': "
                        f"[{m.chunk_start},{m.chunk_end}) -> [{f['chunk_start']},{f['chunk_end']})"
                    )
                    if apply:
                        m.chunk_start = f["chunk_start"]
                        m.chunk_end = f["chunk_end"]
            repaired += 1
        if apply:
            db.commit()
            for doc in db.scalars(select(models.Document)).all():
                after = plan_service.plan_coverage(db, doc.id)
                if not after["complete"]:
                    print(f"STILL INCOMPLETE: doc {doc.id} {after}")
    print(f"\n{'repaired' if apply else 'would repair'} {repaired} document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv))
