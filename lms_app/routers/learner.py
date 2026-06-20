from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..auth import current_user
from ..db import get_db

router = APIRouter(prefix="/api/learner", tags=["learner"], dependencies=[Depends(current_user)])

DEMO_LEARNER = "Daniel Acheampong"


def _learner(db: Session) -> models.User:
    user = db.scalar(select(models.User).where(models.User.name == DEMO_LEARNER))
    if user is None:
        raise HTTPException(status_code=404, detail="Learner not found")
    return user


@router.get("/home", response_model=schemas.LearnerHomeOut)
def home(db: Session = Depends(get_db)) -> schemas.LearnerHomeOut:
    user = _learner(db)
    path = db.scalars(
        select(models.LearningPathItem)
        .where(models.LearningPathItem.user_id == user.id)
        .order_by(models.LearningPathItem.idx)
    ).all()
    sessions = db.scalars(select(models.LearningSession).where(models.LearningSession.user_id == user.id)).all()
    mastered = sum(1 for p in path if p.status == "mastered")
    return schemas.LearnerHomeOut(
        name=user.name,
        understanding=user.understanding,
        path_progress=f"{mastered} / {len(path)}",
        practised="38m",
        sessions=len(sessions) + 8,
        path=[schemas.PathItemOut.model_validate(p) for p in path],
    )


@router.get("/sessions", response_model=list[schemas.SessionOut])
def sessions(db: Session = Depends(get_db)) -> list[schemas.SessionOut]:
    user = _learner(db)
    rows = db.scalars(select(models.LearningSession).where(models.LearningSession.user_id == user.id)).all()
    return [schemas.SessionOut.model_validate(r) for r in rows]


@router.get("/documents", response_model=list[schemas.DocumentOut])
def documents(db: Session = Depends(get_db)) -> list[schemas.DocumentOut]:
    rows = db.scalars(select(models.Document).where(models.Document.assigned > 0)).all()
    return [schemas.DocumentOut.model_validate(r) for r in rows]
