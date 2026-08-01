from __future__ import annotations

"""Conversation memory, backed by mem0 (``mem0ai``).

The voice tutor uses this to remember a learner across sessions and sections:

  • at session END   → ``write_session`` ingests the turns (+ the distilled
                        assessment) so what the learner grasped and missed
                        becomes a queryable fact.
  • at session START → ``recap_for_tutor`` retrieves those facts and returns a
                        short block injected into the tutor's instructions.
  • before scoring   → ``prior_understanding`` pulls what this learner has
                        already demonstrated, so the grader can tell a genuinely
                        new explanation from a repeat of a memorised phrase.

Entity partitioning
-------------------
Following mem0's entity-partitioning playbook, every write and every read is
tagged so one learner's memories can never leak into another's, and one
document's context can never bleed into another's:

  user_id  = "ws<workspace>-user<user>"   the learner, inside their workspace
  run_id   = "doc<document>-s<session>"   one episode (one section sitting)
  agent_id = "praxos-tutor"               the role writing the memory
  metadata = {workspace_id, document_id, module_idx, kind}

Retrieval uses AND filters over ``user_id`` + ``metadata.document_id`` for
cross-session recall of one document, and adds ``run_id`` to re-open a single
paused sitting. NOTE: ``agent_id`` is sent for forward-compatibility but is NOT
relied on for retrieval — it is not persisted on every mem0 plan, so filtering
on it can silently return nothing. The playbook's "use wildcards / avoid field
mismatches" warning is exactly this failure mode.

Everything degrades gracefully: with no ``MEM0_API_KEY`` the functions return
False / "" / None and the tutor simply teaches without recall. Nothing here ever
raises into a request handler.
"""

import logging
from functools import lru_cache
from typing import Optional

from .config import settings

logger = logging.getLogger("praxos.memory")

AGENT_ID = "praxos-tutor"


@lru_cache
def _client():
    if not settings.memory_enabled:
        return None
    try:
        from mem0 import MemoryClient

        return MemoryClient(api_key=settings.MEM0_API_KEY)
    except Exception as exc:  # missing package / bad key — degrade, never break
        logger.warning("mem0 client unavailable: %s", exc)
        return None


def _learner(workspace_id: int, user_id: int) -> str:
    return f"ws{workspace_id}-user{user_id}"


def _episode(document_id: int, session_id: int | str) -> str:
    return f"doc{document_id}-s{session_id}"


def _doc_filter(workspace_id: int, user_id: int, document_id: int) -> dict:
    return {
        "AND": [
            {"user_id": _learner(workspace_id, user_id)},
            {"metadata": {"document_id": document_id}},
        ]
    }


# mem0 roles are {"user","assistant"}; Praxos transcripts use {"learner","tutor"}.
def _to_mem_role(role: str) -> str:
    return "user" if role == "learner" else "assistant"


def _memories(resp) -> list[dict]:
    if isinstance(resp, dict):
        return list(resp.get("results") or [])
    return list(resp or [])


# ---- writes ------------------------------------------------------------------


def write_session(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    session_id: int | str,
    module_idx: int = 0,
    transcript: list[dict],
    summary: Optional[str] = None,
) -> bool:
    """Ingest a finished sitting. ``transcript`` is [{role: learner|tutor, text}].
    The optional ``summary`` (what they grasped / missed) is appended so the
    assessment itself becomes recallable. Returns True on success."""
    client = _client()
    if client is None:
        return False
    turns = [
        {"role": _to_mem_role(t.get("role", "")), "content": t.get("text", "").strip()}
        for t in transcript
        if t.get("text", "").strip()
    ]
    if not turns:
        return False
    if summary and summary.strip():
        turns.append({"role": "assistant", "content": f"[Assessment] {summary.strip()}"})
    try:
        client.add(
            turns,
            user_id=_learner(workspace_id, user_id),
            agent_id=AGENT_ID,
            run_id=_episode(document_id, session_id),
            metadata={
                "workspace_id": workspace_id,
                "document_id": document_id,
                "module_idx": module_idx,
                "kind": "session",
            },
            version="v2",
        )
        return True
    except Exception as exc:
        logger.warning("mem0 add(session) failed: %s", exc)
        return False


def write_lesson_plan(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    doc_name: str,
    modules: list[dict],
) -> bool:
    """Seed a learner's memory with the teaching plan, so the tutor recalls WHAT
    to teach and HOW from the very first session. ``modules`` is a list of
    {idx, title, description, topics}."""
    client = _client()
    if client is None or not modules:
        return False
    lines: list[str] = []
    for m in modules:
        topics = ", ".join(str(t) for t in (m.get("topics") or []))
        line = f"Section {int(m.get('idx', 0)) + 1}: {m.get('title', '')}. {m.get('description', '')}"
        if topics:
            line += f" Key topics: {topics}."
        lines.append(line)
    try:
        client.add(
            [
                {"role": "user", "content": f"What will I learn from '{doc_name}' and how will you teach it?"},
                {
                    "role": "assistant",
                    "content": f"[Lesson plan] Here is how I will teach '{doc_name}':\n" + "\n".join(lines),
                },
            ],
            user_id=_learner(workspace_id, user_id),
            agent_id=AGENT_ID,
            run_id=f"doc{document_id}-plan",
            metadata={
                "workspace_id": workspace_id,
                "document_id": document_id,
                "kind": "lesson_plan",
            },
            version="v2",
        )
        return True
    except Exception as exc:
        logger.warning("mem0 add(plan) failed: %s", exc)
        return False


# ---- reads -------------------------------------------------------------------


def recall(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    query: str,
    top_k: int = 10,
    session_id: Optional[int | str] = None,
) -> list[dict]:
    """Facts this learner has accumulated on this document. Pass ``session_id``
    to narrow to one sitting. Returns [] on any failure — never raises."""
    client = _client()
    if client is None:
        return []
    filters = _doc_filter(workspace_id, user_id, document_id)
    if session_id is not None:
        filters["AND"].append({"run_id": _episode(document_id, session_id)})
    try:
        return _memories(
            client.search(query, filters=filters, top_k=top_k, version="v2")
        )
    except Exception as exc:
        logger.warning("mem0 search failed: %s", exc)
        return []


def recap_for_tutor(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    doc_name: str,
) -> str:
    """A recap block for the tutor's instructions. "" when there is nothing to
    recall (first session, or memory unconfigured), so the caller teaches fresh."""
    facts = recall(
        workspace_id=workspace_id,
        user_id=user_id,
        document_id=document_id,
        query=(
            f"What has this learner already covered, understood and struggled with in "
            f"'{doc_name}'? Include their role/context, strengths and gaps."
        ),
        top_k=settings.MEMORY_RECAP_FACTS,
    )
    lines = [str(f.get("memory") or "").strip() for f in facts]
    lines = [ln for ln in lines if ln][: settings.MEMORY_RECAP_FACTS]
    if not lines:
        return ""
    return "\n".join(
        ["--- WHAT YOU ALREADY KNOW ABOUT THIS LEARNER (earlier sessions) ---", *(f"• {ln}" for ln in lines)]
    )


def prior_understanding(
    *, workspace_id: int, user_id: int, document_id: int, topic: str
) -> list[str]:
    """What this learner has previously demonstrated on a topic — given to the
    grader so a fresh explanation is credited and a parroted one is not."""
    facts = recall(
        workspace_id=workspace_id,
        user_id=user_id,
        document_id=document_id,
        query=f"What has this learner already demonstrated understanding of regarding {topic}?",
        top_k=6,
    )
    return [str(f.get("memory") or "").strip() for f in facts if f.get("memory")]


def forget_document(*, workspace_id: int, user_id: int, document_id: int) -> bool:
    """Drop a learner's memories for one document — used when an admin resets a
    document so a re-learn starts genuinely clean."""
    client = _client()
    if client is None:
        return False
    try:
        client.delete_all(
            user_id=_learner(workspace_id, user_id),
            metadata={"document_id": document_id},
        )
        return True
    except Exception as exc:
        logger.warning("mem0 delete_all failed: %s", exc)
        return False
