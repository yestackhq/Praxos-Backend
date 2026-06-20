from __future__ import annotations

"""Voice teaching sessions.

  start  → builds tutor instructions grounded in the document's indexed text and
           mints an ephemeral OpenAI Realtime token the browser uses to open a
           WebRTC voice connection directly to OpenAI (the API key never reaches
           the client).
  score  → grades the session transcript with an LLM into an understanding
           score (0-100), writes it back to the learner, and records the session.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import ai, indexing, memory, models, workspace
from ..auth import optional_claims
from ..db import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

MAX_CONTEXT_CHARS = 6000


class StartIn(BaseModel):
    documentId: int


class Turn(BaseModel):
    role: str  # "learner" | "tutor"
    text: str


class ScoreIn(BaseModel):
    documentId: int
    transcript: list[Turn] = []


def _require_user(claims: Optional[dict], db: Session) -> models.User:
    sub = claims.get("sub") if claims else None
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    return workspace.resolve_user(db, sub, None, None)


def _doc_in_workspace(db: Session, document_id: int, ws_id: int) -> models.Document:
    doc = db.get(models.Document, document_id)
    if doc is None or doc.workspace_id != ws_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


def _build_instructions(db: Session, doc: models.Document, recap: str = "") -> str:
    chunks = [c.content for c in doc.chunks][:8]
    context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
    if recap:
        memory_block = (
            f"\n\n{recap}\n"
            "You have taught this learner before. Greet them by name, give a one-line recap of "
            "what you covered last time, and ask one or two quick questions to check they still "
            "remember before moving on to new material. Build on their known strengths and gently "
            "revisit their gaps.\n"
        )
    else:
        memory_block = (
            "\n\nThis is your first session with this learner on this document. Briefly introduce "
            "yourself, ask their name and what they already know, then begin teaching.\n"
        )
    return (
        f"You are Praxos, a warm, concise voice tutor teaching the document '{doc.name}'. "
        "Teach it conversationally in short turns: explain a key point, then ask the learner "
        "a question to check understanding. Listen to their answer, give brief feedback, and "
        "move on. Stay strictly within the material below — if asked something outside it, say "
        "so. Keep replies under three sentences so it feels like a real conversation."
        f"{memory_block}"
        f"\n--- DOCUMENT CONTENT ---\n{context}"
    )


@router.post("/start")
def start_session(body: StartIn, claims: Optional[dict] = Depends(optional_claims), db: Session = Depends(get_db)) -> dict:
    user = _require_user(claims, db)
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    # Restore what we already know about this learner so the tutor opens
    # confidently with a recap (empty string for a brand-new learner / if memory
    # is unconfigured or slow — teaching then starts from scratch).
    recap = memory.recap_for_tutor(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        doc_name=doc.name,
    )
    instructions = _build_instructions(db, doc, recap)
    realtime = ai.mint_realtime_session(instructions)
    if realtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice teaching needs an OpenAI API key on the server.",
        )
    # The ephemeral client secret is the top-level ``value`` (GA Realtime API).
    return {
        "document": {"id": doc.id, "name": doc.name},
        "clientSecret": realtime.get("value"),
        "expiresAt": realtime.get("expires_at"),
        "returning": bool(recap),  # frontend can show "Welcome back" vs "Let's begin"
    }


@router.post("/score")
def score_session(body: ScoreIn, claims: Optional[dict] = Depends(optional_claims), db: Session = Depends(get_db)) -> dict:
    user = _require_user(claims, db)
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)

    transcript = [{"role": t.role, "text": t.text} for t in body.transcript if t.text.strip()]
    result = ai.score_understanding(doc.name, transcript)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring needs an OpenAI API key on the server.",
        )

    score = result["score"]
    # Blend toward the latest result so understanding tracks the learner over time.
    user.understanding = score if user.understanding == 0 else round((user.understanding + score) / 2)
    learner_turns = sum(1 for t in transcript if t["role"] == "learner")
    session_row = models.LearningSession(
        user_id=user.id,
        doc=doc.name,
        date=date.today().isoformat(),
        score=score,
        duration=f"{max(1, learner_turns)} exchanges",
        topics=f"{len(result.get('topics', []))} topics",
    )
    db.add(session_row)
    db.commit()
    db.refresh(session_row)

    # Persist this session to conversation memory so the next session can recall
    # it. Async ingest (returns fast); a distilled summary rides along so the
    # learner's strengths/gaps become queryable facts. Best-effort — a memory
    # outage must never fail scoring.
    summary_bits: list[str] = []
    if result.get("summary"):
        summary_bits.append(str(result["summary"]))
    if result.get("strengths"):
        summary_bits.append("Strengths: " + "; ".join(map(str, result["strengths"])))
    if result.get("gaps"):
        summary_bits.append("Gaps to revisit next time: " + "; ".join(map(str, result["gaps"])))
    memory.write_session(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        conversation_id=f"praxos-sess-{session_row.id}",
        transcript=transcript,
        summary=" ".join(summary_bits) or None,
        sync=False,
    )
    return {
        "score": score,
        "understanding": user.understanding,
        "summary": result.get("summary", ""),
        "topics": result.get("topics", []),
        "strengths": result.get("strengths", []),
        "gaps": result.get("gaps", []),
    }
