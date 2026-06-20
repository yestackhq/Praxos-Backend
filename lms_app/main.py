from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from sqlalchemy import text

from .db import Base, SessionLocal, engine, ensure_schema
from .routers import admin, bootstrap, cohorts, documents, learner, sessions, teams
from .seed import seed


def _ensure_pgvector() -> None:
    """pgvector must exist before create_all emits the ``vector`` column DDL."""
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:
        pass  # may need superuser; on Supabase it's typically pre-installed


def _ensure_columns() -> None:
    """Self-heal additive columns on existing Postgres tables (create_all only
    adds missing tables, not columns). No-op on SQLite (create_all covers it)."""
    if engine.dialect.name != "postgresql":
        return
    schema = settings.db_schema
    q = lambda t: f'"{schema}".{t}' if schema else t  # noqa: E731
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {q('workspaces')} ADD COLUMN IF NOT EXISTS onboarded boolean NOT NULL DEFAULT false"))
        conn.execute(text(f"ALTER TABLE {q('documents')} ADD COLUMN IF NOT EXISTS storage_path varchar(400)"))
        conn.execute(text(f"ALTER TABLE {q('invites')} ADD COLUMN IF NOT EXISTS clerk_invite_id varchar(120)"))
        conn.execute(text(f"ALTER TABLE {q('cohorts')} ADD COLUMN IF NOT EXISTS published boolean NOT NULL DEFAULT false"))
        conn.execute(text(f"ALTER TABLE {q('teams')} ADD COLUMN IF NOT EXISTS published boolean NOT NULL DEFAULT false"))
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS chunk_start integer NOT NULL DEFAULT 0"))
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS chunk_end integer NOT NULL DEFAULT 0"))


def init_db() -> None:
    """Create tables and seed demo data (idempotent). Alembic owns prod schema."""
    ensure_schema()
    _ensure_pgvector()
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    if settings.SEED_ON_STARTUP:
        with SessionLocal() as db:
            seed(db)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "auth_enabled": settings.auth_enabled,
        "openai_enabled": settings.openai_enabled,
    }


@app.get("/api/workspace", tags=["meta"])
def workspace() -> dict:
    from sqlalchemy import select

    from . import models

    with SessionLocal() as db:
        ws = db.scalar(select(models.Workspace).limit(1))
        if ws is None:
            return {}
        return {"name": ws.name, "plan": ws.plan}


app.include_router(bootstrap.router)
app.include_router(learner.router)
app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(cohorts.router)
app.include_router(teams.router)
