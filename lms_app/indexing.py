from __future__ import annotations

"""Document indexing: extract text from a PDF, split into chunks, embed them,
and store them for retrieval. Also a small cosine-similarity retriever used by
the voice teaching session.

Indexing is resilient: if OpenAI isn't configured we still extract and store the
text chunks (without embeddings) so the document becomes usable; retrieval then
falls back to keyword overlap.
"""

import io
import math
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import ai, models

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150


def extract_text(data: bytes) -> str:
    """Pull text out of a PDF byte stream. Returns "" if it can't be read."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def chunk_text(text: str) -> list[str]:
    """Split on paragraph boundaries, packing into ~CHUNK_CHARS windows with a
    small overlap so a sentence split across a boundary is still retrievable."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= CHUNK_CHARS:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= CHUNK_CHARS:
                buf = para
            else:  # a single huge paragraph — hard-split it
                for i in range(0, len(para), CHUNK_CHARS - CHUNK_OVERLAP):
                    chunks.append(para[i : i + CHUNK_CHARS])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def index_document(db: Session, doc: models.Document, data: bytes) -> int:
    """Extract → chunk → embed → store. Sets the document status and section
    count. Returns the number of chunks indexed. Replaces any prior chunks."""
    for old in list(doc.chunks):
        db.delete(old)
    db.flush()

    text = extract_text(data)
    chunks = chunk_text(text)
    if not chunks:
        doc.status = "Needs review"  # couldn't read the file (scanned/encrypted)
        doc.sections = 0
        db.commit()
        return 0

    vectors = ai.embed_texts(chunks)  # None when OpenAI unconfigured
    for i, content in enumerate(chunks):
        db.add(
            models.DocumentChunk(
                document_id=doc.id,
                idx=i,
                content=content,
                embedding=vectors[i] if vectors else None,
            )
        )
    doc.sections = len(chunks)
    doc.status = "Indexed"
    db.commit()
    return len(chunks)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _keyword_score(query: str, content: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    c = set(re.findall(r"\w+", content.lower()))
    return len(q & c) / len(q) if q else 0.0


def retrieve(db: Session, document_id: int, query: str, k: int = 4) -> list[str]:
    """Return the k most relevant chunks for a query. Uses embedding cosine
    similarity when vectors exist, else keyword overlap. Scoped to one document,
    so a Python scan is plenty fast at this scale."""
    rows = db.scalars(
        select(models.DocumentChunk).where(models.DocumentChunk.document_id == document_id)
    ).all()
    if not rows:
        return []
    qvec = ai.embed_one(query)
    if qvec and all(r.embedding for r in rows):
        ranked = sorted(rows, key=lambda r: _cosine(qvec, r.embedding), reverse=True)
    else:
        ranked = sorted(rows, key=lambda r: _keyword_score(query, r.content), reverse=True)
    return [r.content for r in ranked[:k]]
