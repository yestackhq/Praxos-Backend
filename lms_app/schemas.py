from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PersonOut(ORM):
    name: str
    email: str
    cohort: str
    documents: int
    understanding: int
    role: str


class TeamOut(ORM):
    name: str
    lead: str
    members: int
    paths: int
    avg: int


class CohortOut(ORM):
    name: str
    members: int
    avg: int
    completion: int
    status: str


class DocumentOut(ORM):
    id: int
    name: str
    sections: int
    assigned: int
    status: str


class ModuleOut(ORM):
    title: str
    description: str
    topics: list[str]
    minutes: int
    source: str


class SessionOut(ORM):
    doc: str
    date: str
    score: int
    duration: str
    topics: str


class PathItemOut(ORM):
    title: str
    sections: int
    status: str
    progress: Optional[int] = None


class TeachingPlanOut(BaseModel):
    doc: str
    modules: list[ModuleOut]


class LearnerHomeOut(BaseModel):
    name: str
    understanding: int
    path_progress: str
    practised: str
    sessions: int
    path: list[PathItemOut]


class KpiOut(BaseModel):
    label: str
    value: str
    hint: str


class AdminOverviewOut(BaseModel):
    kpis: list[KpiOut]
    cohort_health: list[CohortOut]
    people_at_risk: list[PersonOut]
