"""Resize document_chunks.embedding to the configured embedding dimension.

The embedding provider moved from OpenAI ``text-embedding-3-small`` (1536 dims)
to OpenRouter ``nvidia/nemotron-3-embed-1b`` (2048 dims). Postgres rejects a
write whose width does not match the ``vector(n)`` column, so the column has to
be widened before anything can be indexed.

Existing vectors are DISCARDED rather than migrated. There is no conversion
between embedding spaces: a cosine similarity computed between a vector from one
model and a vector from another is meaningless, so keeping the old values would
silently degrade retrieval rather than preserve it. The chunk TEXT is untouched —
only the vectors go — and retrieval falls back to keyword overlap until the
chunks are re-embedded:

    python -m scripts.reembed --apply

The target width is read from ``settings.embed_dim`` (EMBED_DIM), the same
source ``models.Embedding`` uses to emit the DDL, so the column and the ORM can
never disagree. Set EMBED_DIM before running this.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from lms_app.config import settings

revision: str = "f5c20b8e91a7"
down_revision: Union[str, None] = "e3a91c7b40d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = settings.db_schema


def _t(name: str) -> str:
    return f'"{SCHEMA}".{name}' if SCHEMA else name


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite stores embeddings as JSON; there is no width to change.

    dim = settings.embed_dim
    # Drop the stale vectors first: they cannot be cast to the new width, and
    # they are not comparable to anything the new model produces.
    bind.execute(sa.text(f"UPDATE {_t('document_chunks')} SET embedding = NULL"))
    bind.execute(
        sa.text(
            f"ALTER TABLE {_t('document_chunks')} "
            f"ALTER COLUMN embedding TYPE vector({dim}) USING NULL"
        )
    )


def downgrade() -> None:
    raise NotImplementedError(
        "f5c20b8e91a7 is not reversible — the previous model's vectors were "
        "discarded, not converted. Re-embed with the old EMBED_MODEL/EMBED_DIM."
    )
