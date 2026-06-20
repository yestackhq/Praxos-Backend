from __future__ import annotations

"""Teaching-plan service: turn an indexed document into an ordered set of
``Module`` rows (sections) the voice tutor teaches one at a time. The plan is
generated once per document (idempotent) and can be regenerated/edited by an
admin before a cohort is published."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import ai, models


def get_modules(db: Session, document_id: int) -> list[models.Module]:
    return list(
        db.scalars(
            select(models.Module)
            .where(models.Module.document_id == document_id)
            .order_by(models.Module.idx)
        ).all()
    )


def ensure_plan(db: Session, document_id: int) -> list[models.Module]:
    """Return the document's teaching plan, generating it the first time."""
    existing = get_modules(db, document_id)
    return existing if existing else generate_plan(db, document_id)


def generate_plan(db: Session, document_id: int) -> list[models.Module]:
    """(Re)generate a document's plan from its chunks, replacing any existing
    modules. Falls back to evenly-sized sections when the LLM is unavailable so
    the section structure always exists."""
    doc = db.get(models.Document, document_id)
    if doc is None:
        return []
    chunks = [c.content for c in doc.chunks]
    sections = ai.generate_lesson_plan(doc.name, chunks) if chunks else None
    if not sections:
        sections = _fallback_sections(doc.name, len(chunks))

    db.execute(delete(models.Module).where(models.Module.document_id == document_id))
    mods: list[models.Module] = []
    for i, s in enumerate(sections):
        m = models.Module(
            document_id=document_id,
            idx=i,
            title=s["title"],
            description=s["description"],
            topics=s.get("topics") or [],
            minutes=s.get("minutes", 5),
            chunk_start=s.get("chunk_start", 0),
            chunk_end=s.get("chunk_end", 0),
            source=f"Section {i + 1} · taught by voice, checked with questions",
        )
        db.add(m)
        mods.append(m)
    db.commit()
    for m in mods:
        db.refresh(m)
    return mods


def _fallback_sections(doc_name: str, n_chunks: int) -> list[dict]:
    """No LLM → split the document into a few even sections so section-by-section
    teaching still works."""
    if n_chunks <= 0:
        return [
            {
                "title": doc_name,
                "description": "Overview of the document.",
                "topics": [],
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
                "minutes": 5,
                "chunk_start": start,
                "chunk_end": min(n_chunks, start + size),
            }
        )
    return out
