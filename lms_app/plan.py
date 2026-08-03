from __future__ import annotations

"""Teaching-plan service: turn an indexed document into an ordered set of
``Module`` rows (sections) the voice tutor teaches one at a time. The plan is
generated once per document (idempotent) and can be regenerated/edited by an
admin before a cohort is published."""

from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import ai, llm, models


def get_modules(db: Session, document_id: int) -> list[models.Module]:
    return list(
        db.scalars(
            select(models.Module)
            .where(models.Module.document_id == document_id)
            .order_by(models.Module.idx)
        ).all()
    )


def module_payload(m: models.Module) -> dict:
    """The plan shape the tutor prompt and the grader both consume."""
    return {
        "idx": m.idx,
        "title": m.title,
        "description": m.description,
        "topics": list(m.topics or []),
        "key_points": list(m.key_points or []),
        "check_questions": list(m.check_questions or []),
        "minutes": m.minutes,
    }


def section_chunks(doc: models.Document, mod: Optional[models.Module]) -> list[str]:
    """The text a section is taught from. Falls back to the whole document only
    when there is no plan at all — never silently to "the first 8 chunks", which
    used to make every unplanned section teach the same opening pages."""
    if mod is not None and mod.chunk_end > mod.chunk_start:
        return [c.content for c in doc.chunks if mod.chunk_start <= c.idx < mod.chunk_end]
    return [c.content for c in doc.chunks]


def ensure_plan(
    db: Session, document_id: int, *, end_user: Optional[llm.EndUser] = None
) -> list[models.Module]:
    """Return the document's teaching plan, generating it the first time."""
    existing = get_modules(db, document_id)
    return existing if existing else generate_plan(db, document_id, end_user=end_user)


class PlanGenerationError(RuntimeError):
    """A model was available but produced no usable plan."""


def generate_plan(
    db: Session,
    document_id: int,
    *,
    end_user: Optional[llm.EndUser] = None,
    allow_fallback: Optional[bool] = None,
) -> list[models.Module]:
    """(Re)generate a document's plan from its chunks, replacing any existing
    modules.

    The even-split fallback exists for ONE case: no model is configured, and the
    document still needs some section structure to be teachable. It must not
    cover for a model that answered with nothing usable — a fallback plan has no
    key_points, so the tutor cannot check understanding and the grader has no
    ground truth, and it looks identical to a real plan in the admin UI. That is
    how a document silently ended up with four keyless sections during a bulk
    regeneration.

    So when a model IS available and returns nothing usable, this raises rather
    than quietly downgrading the document. Pass ``allow_fallback=True`` to opt
    back into the old behaviour.
    """
    doc = db.get(models.Document, document_id)
    if doc is None:
        return []
    chunks = [c.content for c in doc.chunks]
    sections = (
        ai.generate_lesson_plan(
            doc.name, chunks, end_user=end_user, session_id=f"praxos-plan-d{document_id}"
        )
        if chunks
        else None
    )
    if not sections:
        model_available = llm.chat_enabled() and bool(chunks)
        if model_available and not (allow_fallback is True):
            raise PlanGenerationError(
                f"The model returned no usable plan for '{doc.name}'. The existing plan was "
                "left untouched; retry rather than teaching from an unchecked one."
            )
        sections = _fallback_sections(doc.name, len(chunks))
    else:
        # A model plan may also be PARTIALLY unusable: a section with no
        # key_points/check_questions gives the tutor nothing to check and the
        # grader no ground truth — the learner meets a section with nothing to
        # teach. Refuse the whole plan rather than persisting a hollow section.
        hollow = [
            str(s.get("title") or f"section {i + 1}")
            for i, s in enumerate(sections)
            if not (s.get("key_points") and s.get("check_questions"))
        ]
        if hollow:
            raise PlanGenerationError(
                f"The model's plan for '{doc.name}' left {', '.join(hollow)} without key points "
                "or check questions, so it cannot be taught or graded. The existing plan was "
                "left untouched; retry."
            )

    db.execute(delete(models.Module).where(models.Module.document_id == document_id))
    db.flush()
    mods: list[models.Module] = []
    for i, s in enumerate(sections):
        m = models.Module(
            document_id=document_id,
            idx=i,
            title=s["title"],
            description=s["description"],
            topics=s.get("topics") or [],
            key_points=s.get("key_points") or [],
            check_questions=s.get("check_questions") or [],
            minutes=s.get("minutes", 5),
            chunk_start=s.get("chunk_start", 0),
            chunk_end=s.get("chunk_end", 0),
        )
        db.add(m)
        mods.append(m)
    db.commit()
    for m in mods:
        db.refresh(m)
    return mods


def plan_coverage(db: Session, document_id: int) -> dict:
    """Diagnostic: does the plan actually tile the document? Surfaced to admins so
    a plan that skips a third of the source is visible instead of silent."""
    doc = db.get(models.Document, document_id)
    if doc is None:
        return {"chunks": 0, "covered": 0, "gaps": [], "complete": True}
    n = len(doc.chunks)
    mods = get_modules(db, document_id)
    seen: set[int] = set()
    for m in mods:
        seen.update(range(max(0, m.chunk_start), min(n, m.chunk_end)))
    missing = sorted(set(range(n)) - seen)
    gaps: list[list[int]] = []
    for i in missing:
        if gaps and gaps[-1][1] == i - 1:
            gaps[-1][1] = i
        else:
            gaps.append([i, i])
    return {
        "chunks": n,
        "covered": len(seen),
        "gaps": [{"from": a, "to": b} for a, b in gaps],
        "complete": not missing,
    }


def _fallback_sections(doc_name: str, n_chunks: int) -> list[dict]:
    """No model → split the document into a few even sections so section-by-section
    teaching still works."""
    if n_chunks <= 0:
        return [
            {
                "title": doc_name,
                "description": "Overview of the document.",
                "topics": [],
                "key_points": [],
                "check_questions": [],
                "minutes": 5,
                "chunk_start": 0,
                "chunk_end": 0,
            }
        ]
    k = min(4, n_chunks)
    size = max(1, (n_chunks + k - 1) // k)
    out: list[dict] = []
    for start in range(0, n_chunks, size):
        out.append(
            {
                "title": f"{doc_name} — part {len(out) + 1}",
                "description": "Key points from this part of the document.",
                "topics": [],
                "key_points": [],
                "check_questions": [],
                "minutes": 5,
                "chunk_start": start,
                "chunk_end": min(n_chunks, start + size),
            }
        )
    return out
