from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..auth import current_user
from ..db import get_db

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(current_user)])

KPIS = [
    schemas.KpiOut(label="Avg understanding", value="74", hint="↗ +6 this month"),
    schemas.KpiOut(label="Active learners", value="128", hint="↗ of 142 invited"),
    schemas.KpiOut(label="Completion", value="86%", hint="↗ +4 this week"),
    schemas.KpiOut(label="At risk", value="11", hint="↘ needs follow-up"),
    schemas.KpiOut(label="Sessions today", value="34", hint="↗ +12 vs yesterday"),
]


@router.get("/overview", response_model=schemas.AdminOverviewOut)
def overview(db: Session = Depends(get_db)) -> schemas.AdminOverviewOut:
    cohorts = db.scalars(select(models.Cohort)).all()
    at_risk = db.scalars(
        select(models.User)
        .where(models.User.role == "Learner", models.User.understanding < 55)
        .order_by(models.User.understanding)
    ).all()
    return schemas.AdminOverviewOut(
        kpis=KPIS,
        cohort_health=[schemas.CohortOut.model_validate(c) for c in cohorts],
        people_at_risk=[schemas.PersonOut.model_validate(p) for p in at_risk],
    )


@router.get("/people", response_model=list[schemas.PersonOut])
def people(db: Session = Depends(get_db)) -> list[schemas.PersonOut]:
    rows = db.scalars(select(models.User).order_by(models.User.id)).all()
    return [schemas.PersonOut.model_validate(r) for r in rows]


@router.get("/teams", response_model=list[schemas.TeamOut])
def teams(db: Session = Depends(get_db)) -> list[schemas.TeamOut]:
    rows = db.scalars(select(models.Team).order_by(models.Team.id)).all()
    return [schemas.TeamOut.model_validate(r) for r in rows]


@router.get("/cohorts", response_model=list[schemas.CohortOut])
def cohorts(db: Session = Depends(get_db)) -> list[schemas.CohortOut]:
    rows = db.scalars(select(models.Cohort).order_by(models.Cohort.id)).all()
    return [schemas.CohortOut.model_validate(r) for r in rows]
