from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas, workspace
from ..auth import current_user
from ..db import get_db

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(current_user)])


@router.get("", response_model=list[schemas.DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[schemas.DocumentOut]:
    rows = db.scalars(select(models.Document).order_by(models.Document.id)).all()
    return [schemas.DocumentOut.model_validate(r) for r in rows]


@router.get("/{document_id}/plan", response_model=schemas.TeachingPlanOut)
def teaching_plan(document_id: int, db: Session = Depends(get_db)) -> schemas.TeachingPlanOut:
    doc = db.get(models.Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return schemas.TeachingPlanOut(
        doc=doc.name,
        modules=[schemas.ModuleOut.model_validate(m) for m in doc.modules],
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int, claims: dict = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Remove a document (and its indexed chunks + modules, via cascade). Admins
    only, and only within their own workspace."""
    sub = claims.get("sub") if claims else None
    user = workspace.resolve_user(db, sub, None, None) if sub else None
    doc = db.get(models.Document, document_id)
    if user is None or doc is None or doc.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can delete documents")
    db.delete(doc)  # cascades to document_chunks + modules
    db.commit()
