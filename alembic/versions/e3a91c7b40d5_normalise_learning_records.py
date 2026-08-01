"""Normalise learning records: FK-linked sessions with transcripts, per-section
progress, derived understanding.

What this removes and why
-------------------------
* ``users.understanding`` / ``users.cohort`` / ``users.documents`` — caches that
  were rewritten on every scoring call. ``understanding`` held whatever the last
  few seconds of speech produced, which is what the admin was being shown.
  All three are now derived (``scoring.py`` / ``cohort_members``).
* ``cohorts.members|avg|completion|status``, ``teams.members|paths|avg``,
  ``documents.assigned`` — the same class of hand-maintained counter.
* ``sessions.doc`` (document NAME) → ``document_id`` FK. Name-matching meant
  renaming a document orphaned every session and every path item.
* ``sessions.date|duration|topics`` — strings like "2 exchanges" / "1 topics".
  Replaced by real columns plus the stored transcript.
* ``modules.source`` — a display string ("Section 3 · taught by voice").

What it adds
------------
* ``sessions.transcript`` — sittings were graded and thrown away, so no one
  could see WHY a learner got a score, or re-grade when the rubric improved.
* ``sessions.module_idx`` — sessions were per-section but recorded no section.
* ``section_progress.best_score`` — a learner keeps their best demonstrated
  score for a section, so a cut-short sitting cannot erase it.
* ``modules.key_points`` / ``check_questions`` — the ground truth the grader
  marks against. It previously graded with only the document's filename.

Data migration
--------------
Existing sessions are mapped to documents by name within the learner's
workspace, and their ``score`` is carried over. ``section_progress`` gains a
best_score seeded from the previous ``score`` column.

Historical sessions carry no module_idx (it was never recorded) and no
transcript (never stored), so they are attributed to section 0 and left with an
empty transcript. They stay visible as history but, having no per-section
evidence, they are NOT what the new document rollup is computed from — that
comes from ``section_progress``, which this migration seeds.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from lms_app.config import settings

revision: str = "e3a91c7b40d5"
down_revision: Union[str, None] = "c7e2a1b9d4f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = settings.db_schema


def _t(name: str) -> str:
    return f'"{SCHEMA}".{name}' if SCHEMA else name


def _is_pg() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _reconstruct_section_history(conn) -> None:
    """Attribute historical sittings to sections, and seed per-section bests.

    Sessions were written one-per-section-sitting but recorded no section, and
    ``section_progress`` held a single resume pointer per document — so the only
    surviving per-section signal is the ORDER of a learner's sittings on a
    document. The old code advanced the pointer by one on every completed
    sitting, so the n-th sitting is attributed to section ``min(n, last)``.

    This is a reconstruction, not a recovery: where a learner restarted a
    document the pointer was reset and the attribution will be off. It is
    recorded so historical work is not thrown away, and it is strictly better
    than the previous state, in which a learner's understanding was whatever
    their most recent sitting scored.

    Two classes of sitting are excluded from the bests, because they were never
    measurements of understanding:
      • score 10 with "1 exchanges"/"0 topics" — the literal hard-coded fallback
        for "the learner said nothing" (23 rows in production);
      • score 0 — the grader's floor for a transcript with no answer in it.
    Both are now represented as an unscoreable sitting (score NULL).
    """
    sessions = conn.execute(
        sa.text(
            f"SELECT id, user_id, document_id, score, duration, topics "
            f"FROM {_t('sessions')} ORDER BY user_id, document_id, id"
        )
    ).mappings().all()
    if not sessions:
        return

    section_counts = {
        int(did): int(n)
        for did, n in conn.execute(
            sa.text(f"SELECT document_id, COUNT(*) FROM {_t('modules')} GROUP BY document_id")
        ).all()
    }

    def _is_noise(row) -> bool:
        if row["score"] is None or row["score"] <= 0:
            return True
        return (
            row["score"] == 10
            and (row["duration"] or "").startswith("1 exchange")
            and (row["topics"] or "").startswith("0 ")
        )

    ordinal: dict[tuple[int, int], int] = {}
    bests: dict[tuple[int, int, int], int] = {}
    for row in sessions:
        key = (int(row["user_id"]), int(row["document_id"]))
        n = ordinal.get(key, 0)
        ordinal[key] = n + 1
        last = max(0, section_counts.get(key[1], 1) - 1)
        idx = min(n, last)
        conn.execute(
            sa.text(f"UPDATE {_t('sessions')} SET module_idx = :i WHERE id = :sid"),
            {"i": idx, "sid": row["id"]},
        )
        if _is_noise(row):
            # An unscoreable sitting keeps its transcript-less record but stops
            # counting as a measurement.
            conn.execute(
                sa.text(f"UPDATE {_t('sessions')} SET score = NULL WHERE id = :sid"),
                {"sid": row["id"]},
            )
            continue
        bkey = (key[0], key[1], idx)
        bests[bkey] = max(bests.get(bkey, 0), int(row["score"]))

    # Replace the single resume-pointer row per document with one row per section.
    conn.execute(sa.text(f"DELETE FROM {_t('section_progress')}"))
    for (uid, did, idx), best in sorted(bests.items()):
        conn.execute(
            sa.text(
                f"INSERT INTO {_t('section_progress')} "
                "(user_id, document_id, module_idx, status, best_score, last_score, attempts, "
                # the legacy varchar updated_at is still NOT NULL at this point;
                # it is dropped a few statements later.
                " updated_at, updated_at_ts) "
                "VALUES (:u, :d, :i, :st, :b, :b, 1, '', "
                + ("now()" if _is_pg() else "CURRENT_TIMESTAMP")
                + ")"
            ),
            {
                "u": uid,
                "d": did,
                "i": idx,
                "st": "completed" if best >= 70 else "in_progress",
                "b": best,
            },
        )


def upgrade() -> None:
    conn = op.get_bind()
    json_type = sa.dialects.postgresql.JSONB if _is_pg() else sa.JSON

    # ---- documents ---------------------------------------------------------
    with op.batch_alter_table("documents", schema=SCHEMA) as b:
        b.add_column(sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"))
    conn.execute(sa.text(f"UPDATE {_t('documents')} SET chunk_count = sections"))
    with op.batch_alter_table("documents", schema=SCHEMA) as b:
        b.drop_column("sections")
        b.drop_column("assigned")

    # ---- modules -----------------------------------------------------------
    with op.batch_alter_table("modules", schema=SCHEMA) as b:
        b.add_column(sa.Column("key_points", json_type(), nullable=False, server_default="[]"))
        b.add_column(sa.Column("check_questions", json_type(), nullable=False, server_default="[]"))
        b.drop_column("source")

    # ---- learning_path_items: title -> document_id -------------------------
    with op.batch_alter_table("learning_path_items", schema=SCHEMA) as b:
        b.add_column(sa.Column("document_id", sa.Integer(), nullable=True))
    conn.execute(
        sa.text(
            f"""
            UPDATE {_t('learning_path_items')} li
               SET document_id = d.id
              FROM {_t('users')} u, {_t('documents')} d
             WHERE u.id = li.user_id
               AND d.workspace_id = u.workspace_id
               AND d.name = li.title
            """
        )
        if _is_pg()
        else sa.text(
            """
            UPDATE learning_path_items
               SET document_id = (
                   SELECT d.id FROM documents d
                     JOIN users u ON u.id = learning_path_items.user_id
                    WHERE d.workspace_id = u.workspace_id AND d.name = learning_path_items.title
                    LIMIT 1)
            """
        )
    )
    # Path items whose document no longer exists cannot be represented; drop them.
    conn.execute(sa.text(f"DELETE FROM {_t('learning_path_items')} WHERE document_id IS NULL"))
    with op.batch_alter_table("learning_path_items", schema=SCHEMA) as b:
        b.alter_column("document_id", nullable=False)
        b.create_foreign_key(
            "fk_path_document", "documents", ["document_id"], ["id"], referent_schema=SCHEMA
        )
        b.create_unique_constraint("uq_path_user_document", ["user_id", "document_id"])
        b.drop_column("title")
        b.drop_column("sections")
        b.drop_column("progress")

    # ---- sessions ----------------------------------------------------------
    with op.batch_alter_table("sessions", schema=SCHEMA) as b:
        b.add_column(sa.Column("document_id", sa.Integer(), nullable=True))
        b.add_column(sa.Column("module_idx", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        b.add_column(sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True))
        b.add_column(sa.Column("covered", sa.Integer(), nullable=False, server_default="100"))
        b.add_column(sa.Column("summary", sa.Text(), nullable=False, server_default=""))
        b.add_column(sa.Column("strengths", json_type(), nullable=False, server_default="[]"))
        b.add_column(sa.Column("gaps", json_type(), nullable=False, server_default="[]"))
        b.add_column(sa.Column("transcript", json_type(), nullable=False, server_default="[]"))
        b.add_column(sa.Column("learner_turns", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("paused", sa.Boolean(), nullable=False, server_default="false"))

    conn.execute(
        sa.text(
            f"""
            UPDATE {_t('sessions')} s
               SET document_id = d.id
              FROM {_t('users')} u, {_t('documents')} d
             WHERE u.id = s.user_id
               AND d.workspace_id = u.workspace_id
               AND d.name = s.doc
            """
        )
        if _is_pg()
        else sa.text(
            """
            UPDATE sessions
               SET document_id = (
                   SELECT d.id FROM documents d
                     JOIN users u ON u.id = sessions.user_id
                    WHERE d.workspace_id = u.workspace_id AND d.name = sessions.doc
                    LIMIT 1)
            """
        )
    )
    # Carry the old ISO date string onto the real timestamp column.
    if _is_pg():
        conn.execute(
            sa.text(
                f"UPDATE {_t('sessions')} SET started_at = "
                "COALESCE(NULLIF(date,'')::timestamptz, now()), "
                "ended_at = COALESCE(NULLIF(date,'')::timestamptz, now())"
            )
        )
    else:
        conn.execute(
            sa.text(
                "UPDATE sessions SET started_at = COALESCE(NULLIF(date,''), CURRENT_TIMESTAMP), "
                "ended_at = COALESCE(NULLIF(date,''), CURRENT_TIMESTAMP)"
            )
        )
    # A session whose document is gone has nothing left to point at.
    conn.execute(sa.text(f"DELETE FROM {_t('sessions')} WHERE document_id IS NULL"))

    # ---- section_progress --------------------------------------------------
    with op.batch_alter_table("section_progress", schema=SCHEMA) as b:
        b.add_column(sa.Column("best_score", sa.Integer(), nullable=True))
        b.add_column(sa.Column("last_score", sa.Integer(), nullable=True))
        b.add_column(sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
        b.add_column(sa.Column("updated_at_ts", sa.DateTime(timezone=True), nullable=True))

    # score must accept NULL before the reconstruction can mark the hard-coded
    # "learner said nothing" sittings as unscoreable.
    op.alter_column("sessions", "score", nullable=True, schema=SCHEMA)
    _reconstruct_section_history(conn)

    # `topics` was the string "N topics"; the column becomes structured evidence.
    with op.batch_alter_table("sessions", schema=SCHEMA) as b:
        b.drop_column("topics")
        b.drop_column("duration")
        b.drop_column("doc")
        b.drop_column("date")
        b.add_column(sa.Column("topics", json_type(), nullable=False, server_default="[]"))
        b.alter_column("document_id", nullable=False)
        b.alter_column("started_at", nullable=False)
        b.create_foreign_key(
            "fk_session_document", "documents", ["document_id"], ["id"], referent_schema=SCHEMA
        )
        b.create_index("ix_sessions_user_document", ["user_id", "document_id"])
        b.create_index("ix_sessions_started_at", ["started_at"])

    with op.batch_alter_table("section_progress", schema=SCHEMA) as b:
        b.drop_column("score")
        b.drop_column("updated_at")
    op.alter_column(
        "section_progress",
        "updated_at_ts",
        new_column_name="updated_at",
        nullable=False,
        schema=SCHEMA,
    )
    with op.batch_alter_table("section_progress", schema=SCHEMA) as b:
        b.create_unique_constraint(
            "uq_progress_user_doc_module", ["user_id", "document_id", "module_idx"]
        )

    # ---- drop the hand-maintained caches -----------------------------------
    with op.batch_alter_table("users", schema=SCHEMA) as b:
        b.drop_column("understanding")
        b.drop_column("cohort")
        b.drop_column("documents")
    with op.batch_alter_table("cohorts", schema=SCHEMA) as b:
        b.drop_column("members")
        b.drop_column("avg")
        b.drop_column("completion")
        b.drop_column("status")
    with op.batch_alter_table("teams", schema=SCHEMA) as b:
        b.drop_column("members")
        b.drop_column("paths")
        b.drop_column("avg")

    # ---- membership uniqueness (was duplicable) ----------------------------
    for table, name, cols in (
        ("cohort_members", "uq_cohort_member", ["cohort_id", "user_id"]),
        ("cohort_documents", "uq_cohort_document", ["cohort_id", "document_id"]),
        ("team_members", "uq_team_member", ["team_id", "user_id"]),
        ("team_documents", "uq_team_document", ["team_id", "document_id"]),
        ("modules", "uq_module_document_idx", ["document_id", "idx"]),
    ):
        try:
            op.create_unique_constraint(name, table, cols, schema=SCHEMA)
        except Exception:  # pre-existing duplicates: leave the constraint off
            pass


def downgrade() -> None:
    """Not supported: the dropped columns held derived values that were being
    overwritten continuously, and the pre-migration session rows cannot be
    reconstructed from FK-linked ones without inventing the strings they held."""
    raise NotImplementedError(
        "e3a91c7b40d5 is not reversible — restore from a database snapshot instead."
    )
