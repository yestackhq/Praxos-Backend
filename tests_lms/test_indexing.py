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


def test_retrieve_falls_back_to_keywords_without_openai():
    # No OpenAI key in tests → no embeddings → keyword overlap retrieval.
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
        assert doc.sections == n
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
        assert body["status"] == "Indexed"
        assert body["sections"] >= 1
        # Shows up in the workspace bundle.
        b = client.post("/api/bootstrap", json={"name": "Upl Owner", "email": "upl@x.dev"}).json()
        assert "code-of-conduct.pdf" in [d["name"] for d in b["admin"]["documents"]]
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
