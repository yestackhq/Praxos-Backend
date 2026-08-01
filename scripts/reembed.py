"""Re-embed every document chunk with the currently configured model.

Run after changing EMBED_MODEL / EMBED_DIM (and after the matching migration).
Vectors from different models are not comparable, so a model change means every
chunk has to be embedded again — there is no conversion.

    python -m scripts.reembed              # report what would change
    python -m scripts.reembed --apply      # write the vectors
    python -m scripts.reembed --apply --all   # redo chunks that already have one

Resumable and idempotent: only chunks missing a vector (or of the wrong width)
are processed unless --all is passed, so an interrupted run can simply be
re-run. Free-tier endpoints rate-limit aggressively, so batches are small and a
429 backs off rather than aborting the run.
"""

from __future__ import annotations

import sys
import time

from sqlalchemy import select

from lms_app import llm, models
from lms_app.config import settings
from lms_app.db import SessionLocal

BATCH = 16
MAX_ATTEMPTS = 5


def _embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Embed with backoff. ``llm.embed_texts`` swallows failures and returns
    None, so a retry here is what distinguishes 'rate limited, try again' from
    'genuinely unavailable'."""
    delay = 2.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        vectors = llm.embed_texts(texts)
        if vectors:
            return vectors
        if attempt == MAX_ATTEMPTS:
            return None
        print(f"    batch failed (attempt {attempt}/{MAX_ATTEMPTS}); retrying in {delay:.0f}s")
        time.sleep(delay)
        delay *= 2
    return None


def main(apply: bool, redo_all: bool) -> int:
    dim = settings.embed_dim
    print(f"model  : {settings.embed_model}")
    print(f"base   : {settings.embed_base_url}")
    print(f"dim    : {dim}")
    if not llm.embed_enabled():
        print("\nNo embedding provider configured (EMBED_API_KEY). Nothing to do.")
        return 1

    with SessionLocal() as db:
        chunks = list(
            db.scalars(
                select(models.DocumentChunk).order_by(
                    models.DocumentChunk.document_id, models.DocumentChunk.idx
                )
            ).all()
        )
        stale = [
            c
            for c in chunks
            if redo_all or c.embedding is None or len(c.embedding) != dim
        ]
        print(f"\nchunks : {len(chunks)} total, {len(stale)} to embed")
        if not stale:
            print("nothing to do")
            return 0
        if not apply:
            print("\n(dry run — pass --apply to write)")
            return 0

        done = failed = 0
        for start in range(0, len(stale), BATCH):
            batch = stale[start : start + BATCH]
            print(f"  [{start + 1}-{start + len(batch)}/{len(stale)}] embedding…")
            vectors = _embed_batch([c.content for c in batch])
            if not vectors or len(vectors) != len(batch):
                print("    giving up on this batch; re-run to retry it")
                failed += len(batch)
                continue
            bad = next((v for v in vectors if len(v) != dim), None)
            if bad is not None:
                # Writing a mismatched width would be rejected by Postgres anyway;
                # saying so is more useful than a driver error.
                print(
                    f"    ABORT: provider returned {len(bad)} dims, EMBED_DIM is {dim}. "
                    "Set EMBED_DIM to match and re-run the migration."
                )
                return 2
            for chunk, vector in zip(batch, vectors):
                chunk.embedding = vector
            db.commit()  # commit per batch so an interrupted run keeps its progress
            done += len(batch)

        print(f"\nembedded {done} chunk(s); {failed} still missing")
        return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(apply="--apply" in sys.argv, redo_all="--all" in sys.argv))
