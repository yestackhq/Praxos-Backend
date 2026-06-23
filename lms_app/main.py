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
        conn.execute(text(f"ALTER TABLE {q('workspaces')} ADD COLUMN IF NOT EXISTS slug varchar(80)"))
        conn.execute(text(f"ALTER TABLE {q('documents')} ADD COLUMN IF NOT EXISTS storage_path varchar(400)"))
        conn.execute(text(f"ALTER TABLE {q('invites')} ADD COLUMN IF NOT EXISTS clerk_invite_id varchar(120)"))
        conn.execute(text(f"ALTER TABLE {q('cohorts')} ADD COLUMN IF NOT EXISTS published boolean NOT NULL DEFAULT false"))
        conn.execute(text(f"ALTER TABLE {q('teams')} ADD COLUMN IF NOT EXISTS published boolean NOT NULL DEFAULT false"))
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS chunk_start integer NOT NULL DEFAULT 0"))
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS chunk_end integer NOT NULL DEFAULT 0"))


def _ensure_membership_schema() -> None:
    """Self-heal the users table for multi-workspace membership on existing Postgres:
    drop the old global-unique constraints on email/clerk_id and add the composite
    (clerk_id, workspace_id) unique + lookup indexes. create_all/_ensure_columns only
    ADD tables/columns — they never alter constraints, so this closes the gap. Each
    statement is isolated and idempotent. No-op on SQLite (create_all builds it)."""
    if engine.dialect.name != "postgresql":
        return
    schema = settings.db_schema
    q = lambda t: f'"{schema}".{t}' if schema else t  # noqa: E731
    stmts = [
        f"ALTER TABLE {q('users')} DROP CONSTRAINT IF EXISTS users_email_key",
        f"ALTER TABLE {q('users')} DROP CONSTRAINT IF EXISTS users_clerk_id_key",
        f"CREATE INDEX IF NOT EXISTS ix_users_clerk_id ON {q('users')} (clerk_id)",
        f"CREATE INDEX IF NOT EXISTS ix_users_email ON {q('users')} (email)",
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_user_clerk_workspace') THEN "
        f"ALTER TABLE {q('users')} ADD CONSTRAINT uq_user_clerk_workspace UNIQUE (clerk_id, workspace_id); "
        "END IF; END $$;",
    ]
    for sql in stmts:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
        except Exception:
            pass  # best-effort, idempotent self-heal


def init_db() -> None:
    """Create tables and seed demo data (idempotent). Alembic owns prod schema."""
    ensure_schema()
    _ensure_pgvector()
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    _ensure_membership_schema()
    # Backfill multi-workspace memberships from accepted invites (idempotent): people
    # who accepted before this change stop showing as pending and gain their membership.
    from . import workspace as workspace_svc

    with SessionLocal() as db:
        try:
            workspace_svc.reconcile_memberships(db)
        except Exception:
            db.rollback()
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
