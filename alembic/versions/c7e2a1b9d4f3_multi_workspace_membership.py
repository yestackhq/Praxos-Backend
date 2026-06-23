"""multi-workspace membership

Make ``users`` a true membership table: a person (Clerk ``clerk_id``) can belong to
many workspaces, one row per (clerk_id, workspace_id). Drops the old global-unique
constraints on ``email`` / ``clerk_id`` and adds the composite unique + lookup
indexes. Then backfills memberships from accepted invites so people who already
accepted stop showing as a pending invite.

Revision ID: c7e2a1b9d4f3
Revises: 4df8a7d1aa6c
Create Date: 2026-06-23 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy.orm import Session

from lms_app.config import settings

revision: str = "c7e2a1b9d4f3"
down_revision: Union[str, None] = "4df8a7d1aa6c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables live in a dedicated schema on Postgres (None on SQLite), matching the
# initial migration's convention.
_SCHEMA = settings.db_schema
_P = f"{_SCHEMA}." if _SCHEMA else ""


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # create_all already built the new schema on a fresh DB; these ALTERs heal an
        # EXISTING DB that still has the old single-membership constraints.
        op.execute(f"ALTER TABLE {_P}users DROP CONSTRAINT IF EXISTS users_email_key")
        op.execute(f"ALTER TABLE {_P}users DROP CONSTRAINT IF EXISTS users_clerk_id_key")
        op.execute(f"CREATE INDEX IF NOT EXISTS ix_users_clerk_id ON {_P}users (clerk_id)")
        op.execute(f"CREATE INDEX IF NOT EXISTS ix_users_email ON {_P}users (email)")
        op.execute(
            "DO $$ BEGIN "
            "IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_user_clerk_workspace') THEN "
            f"ALTER TABLE {_P}users ADD CONSTRAINT uq_user_clerk_workspace UNIQUE (clerk_id, workspace_id); "
            "END IF; END $$;"
        )

    # Move existing data into the multi-membership model + confirm accepted invites.
    # Idempotent; also runs on every app startup, so this is best-effort here.
    from lms_app import workspace as workspace_svc

    with Session(bind=bind) as db:
        try:
            workspace_svc.reconcile_memberships(db)
        except Exception:
            db.rollback()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(f"ALTER TABLE {_P}users DROP CONSTRAINT IF EXISTS uq_user_clerk_workspace")
    op.execute(f"DROP INDEX IF EXISTS {_P}ix_users_email")
    op.execute(f"DROP INDEX IF EXISTS {_P}ix_users_clerk_id")
    # Re-adding the global-unique constraints can fail if multi-memberships now exist
    # (duplicate email / clerk_id across workspaces) — best-effort.
    op.execute(f"ALTER TABLE {_P}users ADD CONSTRAINT users_clerk_id_key UNIQUE (clerk_id)")
    op.execute(f"ALTER TABLE {_P}users ADD CONSTRAINT users_email_key UNIQUE (email)")
