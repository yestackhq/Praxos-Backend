from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, plan as plan_service, workspace
from ..auth import active_membership, current_user
from ..db import get_db

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(current_user)])


@router.get("")
def list_documents(
    user: models.User = Depends(active_membership), db: Session = Depends(get_db)
) -> list[dict]:
    rows = db.scalars(
        select(models.Document)
        .where(models.Document.workspace_id == user.workspace_id)
        .order_by(models.Document.id)
    ).all()
    return [workspace.document_out(db, d) for d in rows]


@router.get("/{document_id}/coverage")
def plan_coverage(
    document_id: int,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    """Does this document's teaching plan actually cover the whole document?

    Surfaced because it silently did not: a 46-chunk document was planned from a
    truncated prompt and the last 14 chunks were never taught to anyone."""
    doc = db.get(models.Document, document_id)
    if doc is None or doc.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return plan_service.plan_coverage(db, document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> None:
    """Remove a document (and its indexed chunks + modules, via cascade). Admins
    only, and only within their own workspace."""
    doc = db.get(models.Document, document_id)
    if doc is None or doc.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can delete documents")
    db.delete(doc)  # cascades to document_chunks + modules
    db.commit()
