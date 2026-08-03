from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import llm, meldos, poke
from .config import settings
from sqlalchemy import text

from .db import Base, SessionLocal, engine, ensure_schema
from .routers import bootstrap, cohorts, documents, sessions, teams

logger = logging.getLogger("praxos.startup")


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
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS key_points json NOT NULL DEFAULT '[]'::json"))
        conn.execute(text(f"ALTER TABLE {q('modules')} ADD COLUMN IF NOT EXISTS check_questions json NOT NULL DEFAULT '[]'::json"))


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


def _bound_ddl_locks() -> None:
    """Never let startup DDL wait on a lock.

    Railway starts the new container while the old one is still serving, so the
    ALTER TABLEs below race live reads. That deadlocked a deploy — Postgres
    killed the DDL, the exception escaped the lifespan, and the container failed
    to start at all. A short lock_timeout turns that into a fast, catchable error
    instead of a deadlock, and the self-heal below is optional anyway."""
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("SET lock_timeout = '3s'"))
    except Exception:
        pass


def init_db() -> None:
    """Bring the schema up to date, best-effort. Alembic owns the production schema.

    Every step here is idempotent AND optional: on a deployed instance Alembic has
    already applied the schema, and these calls only exist so a fresh or
    pre-Alembic database still comes up. None of them may prevent the app from
    starting — a container that cannot serve is strictly worse than one running
    against a schema it did not manage to patch.
    """
    _bound_ddl_locks()
    for step in (ensure_schema, _ensure_pgvector, _ensure_columns, _ensure_membership_schema):
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            logger.warning("startup schema step %s skipped: %s", step.__name__, exc)
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_all skipped: %s", exc)

    # Backfill multi-workspace memberships from accepted invites (idempotent): people
    # who accepted before this change stop showing as pending and gain their membership.
    from . import workspace as workspace_svc

    try:
        with SessionLocal() as db:
            try:
                workspace_svc.reconcile_memberships(db)
            except Exception:
                db.rollback()
    except Exception as exc:  # noqa: BLE001
        logger.warning("membership reconcile skipped: %s", exc)


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
        # What the deployment can actually do, so a misconfigured env is visible
        # from /api/health instead of surfacing as a dead voice button.
        "llm": {
            # When MeldOS is configured it fronts the model, so report THAT as the
            # provider — otherwise health would name a vendor no request reaches.
            # Only the base URL and model alias are reported; the application key
            # is never included in any response.
            "provider": "meldos" if settings.meldos_enabled else settings.LLM_PROVIDER,
            "model": settings.MELDOS_MODEL if settings.meldos_enabled else settings.LLM_MODEL,
            "gateway": meldos.base_url() or None,
            "enabled": llm.chat_enabled(),
            "note": poke.inference_unavailable_reason(),
        },
        "embeddings_enabled": llm.embed_enabled(),
        "voice_enabled": settings.voice_enabled,
        "memory_enabled": settings.memory_enabled,
        "poke_enabled": poke.enabled(),
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
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(cohorts.router)
app.include_router(teams.router)
