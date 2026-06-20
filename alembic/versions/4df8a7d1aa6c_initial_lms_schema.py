"""initial lms schema

Revision ID: 4df8a7d1aa6c
Revises:
Create Date: 2026-06-19 12:22:38.006442
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from lms_app.config import settings

revision: str = "4df8a7d1aa6c"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# LMS tables live in a dedicated schema on Postgres (None on SQLite). FK targets
# are schema-qualified accordingly. This migration only ever creates these tables,
# so it never touches pre-existing app tables.
_SCHEMA = settings.db_schema
_P = f"{_SCHEMA}." if _SCHEMA else ""


def upgrade() -> None:
    if _SCHEMA:
        op.execute(f'CREATE SCHEMA IF NOT EXISTS "{_SCHEMA}"')

    op.create_table(
        "workspaces",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("plan", sa.String(length=60), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "cohorts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("members", sa.Integer(), nullable=False),
        sa.Column("avg", sa.Integer(), nullable=False),
        sa.Column("completion", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], [f"{_P}workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("sections", sa.Integer(), nullable=False),
        sa.Column("assigned", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], [f"{_P}workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("lead", sa.String(length=120), nullable=False),
        sa.Column("members", sa.Integer(), nullable=False),
        sa.Column("paths", sa.Integer(), nullable=False),
        sa.Column("avg", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], [f"{_P}workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("clerk_id", sa.String(length=120), nullable=True),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=40), nullable=False),
        sa.Column("cohort", sa.String(length=120), nullable=False),
        sa.Column("documents", sa.Integer(), nullable=False),
        sa.Column("understanding", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], [f"{_P}workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("clerk_id"),
        sa.UniqueConstraint("email"),
        schema=_SCHEMA,
    )
    op.create_table(
        "learning_path_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("sections", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], [f"{_P}users.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "modules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("topics", sa.JSON(), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=200), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], [f"{_P}documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("doc", sa.String(length=160), nullable=False),
        sa.Column("date", sa.String(length=40), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("duration", sa.String(length=20), nullable=False),
        sa.Column("topics", sa.String(length=20), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], [f"{_P}users.id"]),
        sa.PrimaryKeyConstraint("id"),
        schema=_SCHEMA,
    )


def downgrade() -> None:
    for table in (
        "sessions",
        "modules",
        "learning_path_items",
        "users",
        "teams",
        "documents",
        "cohorts",
        "workspaces",
    ):
        op.drop_table(table, schema=_SCHEMA)
