from __future__ import annotations

"""HyperMem (helix) conversation-memory client.

The voice tutor uses this to remember a learner across sessions and chapters of
a document:

  • at session END  → ``write_session`` ingests the transcript (+ a distilled
                       summary) so the learner's name, context, strengths and
                       gaps become queryable facts.
  • at session START → ``recap_for_tutor`` retrieves what the learner already
                       knows / struggled with and returns a short block that is
                       injected into the tutor's instructions, so the agent
                       greets them by name and continues confidently.

Scope mapping (so reads and writes line up):
  tenant_id  = "<HELIX_TENANT>-ws<workspace_id>"   (isolate each workspace)
  namespace  = "doc-<document_id>"                  (one bucket per document)
  user_id    = "user-<user_id>"                     (the learner)
  conversation_id = a per-session id (one episode per chapter/session)

Everything degrades gracefully: when HELIX is unconfigured or unreachable the
functions return ``False`` / ``""`` / ``None`` and the caller carries on without
memory. Nothing here ever raises into a request handler.
"""

import logging
from typing import Optional

import httpx

from .config import settings

logger = logging.getLogger("praxos.memory")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.HELIX_API_KEY}",
        "Content-Type": "application/json",
    }


def _scope(workspace_id: int, user_id: int, document_id: int) -> dict:
    return {
        "tenant_id": f"{settings.HELIX_TENANT}-ws{workspace_id}",
        "namespace": f"doc-{document_id}",
        "user_id": f"user-{user_id}",
    }


# HyperMem turn roles are {"user","assistant","system"}; Praxos transcripts use
# {"learner","tutor"}. The learner is the human ("user"), the tutor is the AI.
def _to_mem_role(role: str) -> str:
    return "user" if role == "learner" else "assistant"


def write_session(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    conversation_id: str,
    transcript: list[dict],
    summary: Optional[str] = None,
    sync: bool = False,
) -> bool:
    """Ingest a finished session's transcript into memory. ``transcript`` is a
    list of {role: "learner"|"tutor", text: str}. An optional ``summary`` (the
    distilled understanding/strengths/gaps) is appended as a final note so it's
    captured as facts. Async by default (returns fast; extraction happens in the
    background). Returns True on a 2xx, False otherwise — never raises."""
    if not settings.helix_enabled:
        return False
    turns = [
        {"role": _to_mem_role(t.get("role", "")), "content": t.get("text", "").strip()}
        for t in transcript
        if t.get("text", "").strip()
    ]
    if not turns:
        return False
    if summary and summary.strip():
        turns.append({"role": "assistant", "content": f"[Session summary] {summary.strip()}"})

    payload = {
        **_scope(workspace_id, user_id, document_id),
        "conversation_id": conversation_id,
        "sync": sync,
        "turns": turns,
    }
    try:
        resp = httpx.post(
            f"{settings.HELIX_URL}/v1/ingest",
            headers=_headers(),
            json=payload,
            timeout=settings.HELIX_WRITE_TIMEOUT,
        )
        ok = resp.status_code < 300
        if not ok:
            logger.warning("helix ingest failed: %s %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as exc:  # network/timeout — degrade, never break scoring
        logger.warning("helix ingest error: %s", exc)
        return False


def recall(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    query: str,
    top_k: int = 12,
    answer: bool = True,
) -> Optional[dict]:
    """Query the learner's memory for this document. Returns the raw HyperMem
    response ({answer, facts, ...}) or None on failure — never raises."""
    if not settings.helix_enabled:
        return None
    payload = {
        **_scope(workspace_id, user_id, document_id),
        "query": query,
        "fact_top_k": top_k,
        "answer": answer,
    }
    try:
        resp = httpx.post(
            f"{settings.HELIX_URL}/v1/retrieve",
            headers=_headers(),
            json=payload,
            timeout=settings.HELIX_READ_TIMEOUT,
        )
        if resp.status_code >= 300:
            logger.warning("helix retrieve failed: %s %s", resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("helix retrieve error: %s", exc)
        return None


def recap_for_tutor(
    *,
    workspace_id: int,
    user_id: int,
    document_id: int,
    doc_name: str,
) -> str:
    """Build a recap block for the tutor's instructions from prior memory.

    Returns "" when there's no usable memory (first-ever session, or memory off/
    slow) so the caller can teach from scratch. Otherwise returns a short block:
    a synthesized recap line plus a few specific facts the tutor can lean on.
    """
    data = recall(
        workspace_id=workspace_id,
        user_id=user_id,
        document_id=document_id,
        query=(
            f"Who is this learner and what have they already covered and struggled with in "
            f"'{doc_name}'? Include their name, role/context, strengths, and gaps."
        ),
        top_k=12,
        answer=True,
    )
    if not data:
        return ""
    answer = (data.get("answer") or "").strip()
    facts = [f.get("content", "").strip() for f in data.get("facts", []) if f.get("content")]
    facts = facts[:6]
    if not answer and not facts:
        return ""

    lines = ["--- WHAT YOU ALREADY KNOW ABOUT THIS LEARNER (from earlier sessions) ---"]
    if answer:
        lines.append(answer)
    if facts:
        lines.append("Specific recalled facts:")
        lines.extend(f"• {f}" for f in facts)
    return "\n".join(lines)
