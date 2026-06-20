from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models

# Mirrors web/src/lib/mock.ts so the API and the UI tell the same story.

PEOPLE = [
    ("Aisha Bello", "aisha.bello@meridian.health", "March new hires", 3, 71, "Learner"),
    ("Tomás Herrera", "tomas.h@meridian.health", "March new hires", 3, 63, "Learner"),
    ("Marcus Lindqvist", "marcus.l@meridian.health", "Q1 Engineering", 2, 78, "Learner"),
    ("Grace Mwangi", "grace.m@meridian.health", "Sales onboarding", 4, 49, "Learner"),
    ("Kenji Watanabe", "kenji.w@meridian.health", "Q1 Engineering", 2, 88, "Learner"),
    ("Priya Nair", "priya.n@meridian.health", "Compliance refresh", 5, 100, "Manager"),
    ("Daniel Acheampong", "daniel.a@meridian.health", "March new hires", 2, 74, "Learner"),
    ("Sofia Okonkwo", "sofia.o@meridian.health", "—", 24, 0, "Admin"),
]

TEAMS = [
    ("Engineering", "Marcus Lindqvist", 18, 3, 82),
    ("Sales", "Grace Mwangi", 31, 2, 64),
    ("Operations", "Tomás Herrera", 24, 4, 71),
    ("People & Culture", "Sofia Okonkwo", 9, 5, 88),
    ("Finance", "Priya Nair", 12, 3, 76),
    ("Customer Success", "Kenji Watanabe", 16, 3, 79),
]

COHORTS = [
    ("March new hires", 24, 71, 58, "On track"),
    ("Q1 Engineering", 31, 82, 74, "On track"),
    ("Sales onboarding", 18, 64, 41, "At risk"),
    ("Compliance refresh", 42, 77, 66, "On track"),
]

DOCUMENTS = [
    ("Code of conduct", 5, 142, "Indexed"),
    ("Data protection & GDPR", 6, 128, "Indexed"),
    ("Information security basics", 4, 96, "Indexed"),
    ("Expense & travel policy", 3, 54, "Indexed"),
    ("Anti-bribery & corruption", 5, 31, "Indexed"),
    ("Health & safety handbook", 8, 0, "Indexing"),
]

GDPR_MODULES = [
    ("What personal data means", "Identify personal and special-category data with real examples.",
     ["Definitions", "Special categories", "Worked examples", "Pseudonymisation"], 8, "From section 1"),
    ("Lawful bases for processing", "Name the six lawful bases and pick the right one for a scenario.",
     ["Consent", "Contract", "Legal obligation", "Legitimate interest", "Vital interests", "Public task"], 11, "From section 2"),
    ("Data subject rights", "Explain the key rights and handle a subject access request.",
     ["Right of access", "Rectification", "Erasure", "Subject access request", "The 30-day rule"], 12, "From sections 3–4"),
    ("Retention & data minimisation", "Apply retention schedules and justify minimisation.",
     ["Retention windows", "Minimisation"], 8, "From section 5"),
    ("Breach reporting", "Know the 72-hour rule and who to notify.",
     ["72-hour rule", "Who to notify"], 9, "From section 6"),
]

PATH = [
    ("Code of conduct", 5, "mastered", None),
    ("Data protection & GDPR", 6, "in_progress", 62),
    ("Information security basics", 4, "up_next", None),
    ("Expense & travel policy", 3, "locked", None),
    ("Anti-harassment training", 4, "locked", None),
]

SESSIONS = [
    ("Code of conduct", "18 Jun", 88, "9m", "5 / 5"),
    ("Data protection & GDPR", "17 Jun", 62, "14m", "4 / 6"),
    ("Information security basics", "15 Jun", 71, "11m", "3 / 4"),
    ("Code of conduct", "12 Jun", 54, "8m", "3 / 5"),
]


def seed(db: Session) -> None:
    """Idempotently populate demo data. No-op if a workspace already exists."""
    if db.scalar(select(models.Workspace).limit(1)) is not None:
        return

    ws = models.Workspace(name="Meridian Health", plan="Admin workspace")
    db.add(ws)
    db.flush()

    learner = None
    for name, email, cohort, docs, und, role in PEOPLE:
        u = models.User(workspace_id=ws.id, name=name, email=email, cohort=cohort,
                        documents=docs, understanding=und, role=role)
        db.add(u)
        if name == "Daniel Acheampong":
            learner = u

    for name, lead, members, paths, avg in TEAMS:
        db.add(models.Team(workspace_id=ws.id, name=name, lead=lead, members=members, paths=paths, avg=avg))

    for name, members, avg, completion, status in COHORTS:
        db.add(models.Cohort(workspace_id=ws.id, name=name, members=members, avg=avg,
                             completion=completion, status=status))

    gdpr = None
    for name, sections, assigned, status in DOCUMENTS:
        d = models.Document(workspace_id=ws.id, name=name, sections=sections, assigned=assigned, status=status)
        db.add(d)
        if name == "Data protection & GDPR":
            gdpr = d
    db.flush()

    if gdpr is not None:
        for i, (title, desc, topics, minutes, source) in enumerate(GDPR_MODULES):
            db.add(models.Module(document_id=gdpr.id, idx=i, title=title, description=desc,
                                 topics=topics, minutes=minutes, source=source))

    db.flush()
    if learner is not None:
        for i, (title, sections, status, progress) in enumerate(PATH):
            db.add(models.LearningPathItem(user_id=learner.id, idx=i, title=title,
                                           sections=sections, status=status, progress=progress))
        for doc, date, score, duration, topics in SESSIONS:
            db.add(models.LearningSession(user_id=learner.id, doc=doc, date=date,
                                          score=score, duration=duration, topics=topics))

    db.commit()
