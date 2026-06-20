from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
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
