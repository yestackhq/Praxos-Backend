from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from .config import settings
from .db import Base


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

            return dialect.type_descriptor(Vector(settings.OPENAI_EMBED_DIM))
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
    data (path items, sessions, progress) keys off this row's ``id``."""

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
    cohort: Mapped[str] = mapped_column(String(120), default="—")
    documents: Mapped[int] = mapped_column(Integer, default=0)
    understanding: Mapped[int] = mapped_column(Integer, default=0)


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(120))
    lead: Mapped[str] = mapped_column(String(120), default="")
    members: Mapped[int] = mapped_column(Integer, default=0)
    paths: Mapped[int] = mapped_column(Integer, default=0)
    avg: Mapped[int] = mapped_column(Integer, default=0)
    published: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class TeamDocument(Base):
    """Ordered curriculum assigned to a team."""

    __tablename__ = "team_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)


class TeamMember(Base):
    """Which learners belong to a team."""

    __tablename__ = "team_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Cohort(Base):
    __tablename__ = "cohorts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(120))
    members: Mapped[int] = mapped_column(Integer, default=0)
    avg: Mapped[int] = mapped_column(Integer, default=0)
    completion: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="On track")
    # Draft until the admin reviews the AI plan and submits — publishing pushes
    # the lesson plan + document context into each member's memory.
    published: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class CohortDocument(Base):
    """Ordered curriculum: the documents a cohort learns, in path order."""

    __tablename__ = "cohort_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)


class CohortMember(Base):
    """Which learners belong to a cohort."""

    __tablename__ = "cohort_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(160))
    sections: Mapped[int] = mapped_column(Integer, default=0)
    assigned: Mapped[int] = mapped_column(Integer, default=0)
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[Optional[list]] = mapped_column(Embedding, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Module(Base):
    __tablename__ = "modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)
    minutes: Mapped[int] = mapped_column(Integer, default=5)
    source: Mapped[str] = mapped_column(String(200), default="")
    # The chunk range this section is taught from, so the tutor is grounded in
    # just this section's text (inclusive start, exclusive end).
    chunk_start: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunk_end: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    document: Mapped[Document] = relationship(back_populates="modules")


class LearningPathItem(Base):
    __tablename__ = "learning_path_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(160))
    sections: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(40), default="locked")  # mastered|in_progress|up_next|locked
    progress: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


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
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    doc: Mapped[str] = mapped_column(String(160))
    date: Mapped[str] = mapped_column(String(40))
    score: Mapped[int] = mapped_column(Integer, default=0)
    duration: Mapped[str] = mapped_column(String(20), default="")
    topics: Mapped[str] = mapped_column(String(20), default="")


class SectionProgress(Base):
    """A learner's progress through one section (module) of a document. Lets a
    paused section resume where it left off and drives the next-up section."""

    __tablename__ = "section_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    module_idx: Mapped[int] = mapped_column(Integer, default=0)
    # in_progress | paused | completed
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[str] = mapped_column(String(40), default="")
