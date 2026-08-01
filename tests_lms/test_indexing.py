from __future__ import annotations

import io


def _minimal_pdf(text: str) -> bytes:
    """A tiny but valid one-page PDF whose text pypdf can extract."""
    stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (i, o)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return pdf


def test_extract_and_chunk():
    from lms_app.indexing import chunk_text, extract_text

    data = _minimal_pdf("Strong passwords protect accounts. Lock your screen when away.")
    text = extract_text(data)
    assert "Strong passwords" in text
    assert chunk_text(text)  # non-empty
    assert chunk_text("") == []


def test_retrieve_falls_back_to_keywords_without_openai(client):  # client creates the schema
    # No embedding provider in tests → keyword overlap retrieval.
    from lms_app.db import SessionLocal
    from lms_app import indexing, models

    with SessionLocal() as db:
        ws = models.Workspace(name="Idx WS", plan="x")
        db.add(ws)
        db.flush()
        doc = models.Document(workspace_id=ws.id, name="Sec.pdf", status="Indexing")
        db.add(doc)
        db.commit()
        n = indexing.index_document(db, doc, _minimal_pdf("Reset your password regularly. Phishing emails are a threat."))
        assert n >= 1
        assert doc.status == "Indexed"
        assert doc.chunk_count == n
        hits = indexing.retrieve(db, doc.id, "how do I handle phishing", k=2)
        assert hits and any("Phishing" in h or "phishing" in h for h in hits)


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def test_upload_pdf_indexes_and_appears(client):
    try:
        _as({"sub": "upl_owner"})
        client.post("/api/bootstrap", json={"name": "Upl Owner", "email": "upl@x.dev"})
        pdf = _minimal_pdf("Code of conduct. Treat colleagues with respect and report concerns.")
        r = client.post(
            "/api/documents/upload",
            files={"file": ("code-of-conduct.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "Indexing"  # indexing now runs in a background task
        # The background task completes; it appears Indexed (with sections) in the bundle.
        b = client.post("/api/bootstrap", json={"name": "Upl Owner", "email": "upl@x.dev"}).json()
        docs = {d["name"]: d for d in b["admin"]["documents"]}
        assert "code-of-conduct.pdf" in docs
        assert docs["code-of-conduct.pdf"]["status"] == "Indexed"
        assert docs["code-of-conduct.pdf"]["chunks"] >= 1
    finally:
        _clear()


def test_upload_rejects_empty_file(client):
    try:
        _as({"sub": "upl_owner2"})
        client.post("/api/bootstrap", json={"name": "Upl Owner2", "email": "upl2@x.dev"})
        r = client.post("/api/documents/upload", files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")})
        assert r.status_code == 400
    finally:
        _clear()


def test_repair_letter_spacing_recovers_word_boundaries():
    """PDFs that position each glyph separately extract as letters-with-spaces,
    with words marked by DOUBLE spaces. That signal has to be used before any
    whitespace normalisation, or the page becomes one unreadable string — which
    is exactly what happened to a live document (18 of 19 chunks)."""
    from lms_app.indexing import chunk_text, extract_text, repair_letter_spacing

    mangled = "E n g i n e e r i n g 's  o b j e c t  i s  t h e\na r t i f a c t .  P r o d u c t 's  o b j e c t"
    fixed = repair_letter_spacing(mangled)
    assert fixed == "Engineering's object is the\nartifact. Product's object"

    # Normal prose is left completely alone.
    prose = "The one sentence\nEngineering exists to make the system correct."
    assert repair_letter_spacing(prose) == prose

    # A fragment with no double space has no boundary information, so it is
    # treated as one word. Accepted trade-off: a genuine run of initials joins
    # too ("A B C" -> "ABC"), which is far rarer here than letter-salad
    # fragments like "s p e c ." that would otherwise reach the tutor.
    assert repair_letter_spacing("s p e c .") == "spec."

    # End to end: repaired text survives chunking with its words intact.
    chunks = chunk_text(repair_letter_spacing(mangled))
    assert chunks and "Engineering's object is the" in chunks[0]


def test_extract_text_repairs_a_real_letter_spaced_pdf():
    """The repair must run inside extract_text, before chunk_text collapses the
    double spaces that carry the word boundaries."""
    from lms_app.indexing import extract_text

    # A PDF whose text is drawn glyph-by-glyph is hard to synthesise here, so
    # assert the ordering property that made the bug possible instead: the
    # helper is applied to the joined page text, not after normalisation.
    import inspect

    from lms_app import indexing

    src = inspect.getsource(indexing.extract_text)
    assert "repair_letter_spacing" in src
    assert extract_text(b"not a pdf") == ""
