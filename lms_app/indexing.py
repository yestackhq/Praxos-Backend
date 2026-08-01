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


def _is_letter_spaced(line: str) -> bool:
    """True when a line is mostly single characters, i.e. the extractor emitted a
    space between every glyph."""
    tokens = line.split()
    if len(tokens) < 3:
        return False
    singles = sum(1 for t in tokens if len(t) == 1)
    return singles / len(tokens) > 0.6


def repair_letter_spacing(text: str) -> str:
    """Undo per-glyph spacing from PDFs that position each character separately.

    Such a PDF extracts as::

        E n g i n e e r i n g 's  o b j e c t  i s  t h e

    — letters separated by ONE space, words by TWO. That distinction is the only
    record of where words begin, and it must be used before any whitespace
    normalisation: collapsing runs of spaces first (which ``chunk_text`` does)
    destroys it irrecoverably, turning the page into one unreadable string.

    That is what happened to a live document: 18 of its 19 chunks were stored as
    letter salad, so the tutor taught from it and the grader marked answers
    against it. Lines that are not letter-spaced are returned untouched.
    """
    out: list[str] = []
    for line in text.split("\n"):
        if not _is_letter_spaced(line):
            out.append(line)
            continue
        stripped = line.strip()
        if "  " in stripped:
            # Double spaces mark the word boundaries — use them.
            words = [w.replace(" ", "") for w in re.split(r" {2,}", stripped)]
            out.append(" ".join(w for w in words if w))
        else:
            # A short fragment with no double space carries no boundary
            # information ("s p e c ."), so it is a single word. Note the
            # trade-off: a genuine run of initials ("A B C") joins to "ABC".
            # In this corpus that is far rarer, and far less damaging, than
            # leaving a fragment as letter salad in the text the tutor teaches
            # from and the grader marks against.
            out.append(stripped.replace(" ", ""))
    return "\n".join(out)


def extract_text(data: bytes) -> str:
    """Pull text out of a PDF byte stream. Returns "" if it can't be read."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        # Repair BEFORE joining/normalising — the word boundaries only exist as
        # double spaces at this point.
        return repair_letter_spacing("\n\n".join(pages)).strip()
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
    """Extract -> chunk -> embed -> store. Sets the document status and chunk
    count. Returns the number of chunks indexed. Replaces any prior chunks.

    Any existing teaching plan is dropped: its chunk ranges point into the OLD
    chunk list, so keeping it would leave sections grounded in the wrong text."""
    for old in list(doc.chunks):
        db.delete(old)
    for old_module in list(doc.modules):
        db.delete(old_module)
    db.flush()

    text = extract_text(data)
    chunks = chunk_text(text)
    if not chunks:
        doc.status = "Needs review"  # couldn't read the file (scanned/encrypted)
        doc.chunk_count = 0
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
    doc.chunk_count = len(chunks)
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
    # Every stored vector must come from the SAME model as the query vector, or
    # the comparison is meaningless. Width is the observable proxy: _cosine zips
    # its inputs, so a 1536-dim leftover scored against a 2048-dim query would
    # silently rank on the first 1536 components instead of failing. Falling back
    # to keyword overlap is the honest answer until the chunks are re-embedded.
    usable = bool(qvec) and all(
        r.embedding is not None and len(r.embedding) == len(qvec) for r in rows
    )
    if usable:
        ranked = sorted(rows, key=lambda r: _cosine(qvec, r.embedding), reverse=True)
    else:
        ranked = sorted(rows, key=lambda r: _keyword_score(query, r.content), reverse=True)
    return [r.content for r in ranked[:k]]
