from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .config import settings
from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Embedding(TypeDecorator):
    """An embedding vector stored as a native pgvector ``vector`` column on
    Postgres (so it's queryable with pgvector operators / ANN indexes), and as
    portable JSON on SQLite so the test suite runs with no extension. Either way
    the Python value is a plain ``list[float]``."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(settings.embed_dim))
        return dialect.type_descriptor(JSON())

    def process_result_value(self, value, dialect):
        return None if value is None else list(value)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    plan: Mapped[str] = mapped_column(String(60), default="Admin workspace")
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    slug: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)


class User(Base):
    """A person's membership in ONE workspace. A person (Clerk ``clerk_id``) can
    belong to several workspaces — one row per (clerk_id, workspace_id) — so neither
    ``clerk_id`` nor ``email`` is globally unique; the pair is. Per-workspace learner
    data (path items, sessions, progress) keys off this row's ``id``.

    Deliberately carries NO cached learner metrics. ``understanding``, ``cohort``
    and ``documents`` used to live here and were written on every scoring call,
    which is why the admin saw whatever the learner's last few seconds happened to
    produce. Cohort membership lives in ``cohort_members``; understanding is
    derived from ``section_progress`` (see ``scoring.py``)."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("clerk_id", "workspace_id", name="uq_user_clerk_workspace"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    clerk_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), index=True)
    role: Mapped[str] = mapped_column(String(40), default="Learner")  # Learner | Manager | Admin


class Team(Base):
    """A team. Member and document counts are derived from ``team_members`` /
    ``team_documents``; the average understanding from ``section_progress``."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(120))
    lead: Mapped[str] = mapped_column(String(120), default="")
    published: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class TeamDocument(Base):
    """Ordered curriculum assigned to a team."""

    __tablename__ = "team_documents"
    __table_args__ = (UniqueConstraint("team_id", "document_id", name="uq_team_document"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)


class TeamMember(Base):
    """Which learners belong to a team."""

    __tablename__ = "team_members"
    __table_args__ = (UniqueConstraint("team_id", "user_id", name="uq_team_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Cohort(Base):
    """A cohort. Like Team, it stores only what an admin sets — membership,
    curriculum, understanding and completion are all derived."""

    __tablename__ = "cohorts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(120))
    # Draft until the admin reviews the AI plan and submits — publishing pushes
    # the lesson plan + document context into each member's memory.
    published: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class CohortDocument(Base):
    """Ordered curriculum: the documents a cohort learns, in path order."""

    __tablename__ = "cohort_documents"
    __table_args__ = (UniqueConstraint("cohort_id", "document_id", name="uq_cohort_document"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)


class CohortMember(Base):
    """Which learners belong to a cohort."""

    __tablename__ = "cohort_members"
    __table_args__ = (UniqueConstraint("cohort_id", "user_id", name="uq_cohort_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(160))
    # How many retrieval chunks the text was split into. NOT the number of
    # teaching sections — those are ``modules`` (the old column was called
    # `sections`, which made a 46-chunk document look like a 46-section course).
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(40), default="Indexed")
    storage_path: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)

    modules: Mapped[list["Module"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Module.idx"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.idx"
    )


class DocumentChunk(Base):
    """A chunk of a document's extracted text plus its embedding. Powers
    retrieval during a voice teaching session."""

    __tablename__ = "document_chunks"
    __table_args__ = (Index("ix_document_chunks_document_id", "document_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[Optional[list]] = mapped_column(Embedding, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Module(Base):
    """One teaching section of a document's lesson plan.

    ``key_points`` is what the learner must be able to state and is the ground
    truth the grader checks answers against; ``check_questions`` are the probes
    the tutor must get through before it may advance. Without these the grader
    was scoring answers with no idea what a correct answer looked like."""

    __tablename__ = "modules"
    __table_args__ = (UniqueConstraint("document_id", "idx", name="uq_module_document_idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)
    key_points: Mapped[list] = mapped_column(JSON, default=list)
    check_questions: Mapped[list] = mapped_column(JSON, default=list)
    minutes: Mapped[int] = mapped_column(Integer, default=5)
    # The chunk range this section is taught from, so the tutor is grounded in
    # just this section's text (inclusive start, exclusive end). The plan is
    # normalised to tile [0, chunk_count) exactly once — see ai.generate_lesson_plan.
    chunk_start: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunk_end: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    document: Mapped[Document] = relationship(back_populates="modules")


class LearningPathItem(Base):
    """A document on a learner's path, in order. Keyed by ``document_id`` — it
    used to key off the document NAME, so renaming a document orphaned every
    learner's progress and every session."""

    __tablename__ = "learning_path_items"
    __table_args__ = (UniqueConstraint("user_id", "document_id", name="uq_path_user_document"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    # mastered|completed|in_progress|up_next|locked
    # "completed" = every section sat; "mastered" = that, and the document score
    # cleared the threshold. Both open the next document — see scoring.refresh_path_item.
    status: Mapped[str] = mapped_column(String(40), default="locked")


class Invite(Base):
    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    email: Mapped[str] = mapped_column(String(200), index=True)
    role: Mapped[str] = mapped_column(String(40), default="Learner")  # Learner | Manager | Admin
    invited_by: Mapped[str] = mapped_column(String(120), default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | accepted
    clerk_invite_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)


class LearningSession(Base):
    """One sitting: a learner working through one section of one document.

    The transcript is stored. Previously it was sent to an external memory
    service and nothing else, so there was no way to audit why a learner was
    given a score — or to re-grade when the rubric improved.

    ``score`` is nullable on purpose: a sitting where the learner never said
    anything substantive is UNSCOREABLE, not a zero. Recording those as 10 is
    what pulled everyone's numbers down."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user_document", "user_id", "document_id"),
        Index("ix_sessions_started_at", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    module_idx: Mapped[int] = mapped_column(Integer, default=0)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # How much of the section was actually taught in this sitting (0-100), so a
    # short sitting is not mistaken for a full assessment of the section.
    covered: Mapped[int] = mapped_column(Integer, default=100, server_default="100")
    summary: Mapped[str] = mapped_column(Text, default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)  # [{name, score, evidence}]
    strengths: Mapped[list] = mapped_column(JSON, default=list)
    gaps: Mapped[list] = mapped_column(JSON, default=list)
    transcript: Mapped[list] = mapped_column(JSON, default=list)  # [{role, text}]
    learner_turns: Mapped[int] = mapped_column(Integer, default=0)
    paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class SectionProgress(Base):
    """A learner's standing on one section of a document.

    ``best_score`` is the learner's highest demonstrated score for the section
    and is what rolls up into the document score — a learner who explains a
    section well keeps that credit even if a later sitting is cut short.
    ``last_score`` is kept for "how did the most recent attempt go"."""

    __tablename__ = "section_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "document_id", "module_idx", name="uq_progress_user_doc_module"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    module_idx: Mapped[int] = mapped_column(Integer, default=0)
    # in_progress | paused | completed
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    best_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
