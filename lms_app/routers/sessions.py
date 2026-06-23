from __future__ import annotations

"""Voice teaching sessions.

  start  → builds tutor instructions grounded in the document's indexed text and
           mints an ephemeral OpenAI Realtime token the browser uses to open a
           WebRTC voice connection directly to OpenAI (the API key never reaches
           the client).
  score  → grades the session transcript with an LLM into an understanding
           score (0-100), writes it back to the learner, and records the session.
"""

import re
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import ai, indexing, memory, models, plan as plan_service, workspace
from ..auth import active_membership
from ..db import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

MAX_CONTEXT_CHARS = 6000


class StartIn(BaseModel):
    documentId: int
    moduleIdx: Optional[int] = None  # explicit section; else resume at the saved point
    restart: bool = False  # re-learn from the first section (reset saved progress)


class Turn(BaseModel):
    role: str  # "learner" | "tutor"
    text: str


class ScoreIn(BaseModel):
    documentId: int
    transcript: list[Turn] = []
    moduleIdx: Optional[int] = None
    paused: bool = False  # learner paused mid-section (resume later) vs finished it


def _doc_in_workspace(db: Session, document_id: int, ws_id: int) -> models.Document:
    doc = db.get(models.Document, document_id)
    if doc is None or doc.workspace_id != ws_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


def _progress_row(db: Session, user_id: int, document_id: int) -> Optional[models.SectionProgress]:
    return db.scalar(
        select(models.SectionProgress).where(
            models.SectionProgress.user_id == user_id,
            models.SectionProgress.document_id == document_id,
        )
    )


def _section_chunks(doc: models.Document, mod: Optional[models.Module]) -> list[str]:
    """The text the tutor teaches from: just this section's chunk range when there
    is a plan, else the first few chunks (un-planned document)."""
    if mod is not None and mod.chunk_end > mod.chunk_start:
        return [c.content for c in doc.chunks if mod.chunk_start <= c.idx < mod.chunk_end]
    return [c.content for c in doc.chunks][:8]


def _build_instructions(
    doc: models.Document,
    modules: list[models.Module],
    idx: int,
    recap: str = "",
    resumed: bool = False,
    advancing: bool = False,
) -> str:
    cur = modules[idx] if modules and 0 <= idx < len(modules) else None
    context = "\n\n".join(_section_chunks(doc, cur))[:MAX_CONTEXT_CHARS]

    if advancing:
        memory_block = (
            "\n\nThe learner just finished the previous section. In ONE sentence recap what it "
            "covered, then immediately start teaching THIS section. Do not greet or re-introduce yourself.\n"
        )
    elif recap:
        memory_block = (
            f"\n\n{recap}\n"
            "You have taught this learner before. Do NOT introduce yourself. Give a ONE-line recap of "
            "where they left off, then continue teaching this section.\n"
        )
    else:
        memory_block = (
            "\n\nStart teaching THIS section directly — do NOT introduce yourself or summarise the "
            "document. Open with the first key point and a question.\n"
        )

    plan_block = ""
    section_block = ""
    if modules:
        outline = "\n".join(
            f"  {i + 1}. {m.title}" + ("  ← teaching now" if i == idx else "")
            for i, m in enumerate(modules)
        )
        plan_block = f"\n--- COURSE OUTLINE ({len(modules)} sections) ---\n{outline}\n"
    if cur is not None:
        topics = ", ".join(str(t) for t in (cur.topics or []))
        section_block = (
            f"\nYou are teaching SECTION {idx + 1} of {len(modules)}: \"{cur.title}\". "
            f"Aim: {cur.description} "
            + (f"Make sure to cover: {topics}. " if topics else "")
            + "Teach ONLY this section now. When the learner has correctly EXPLAINED this section's "
            "idea in their own words (not merely agreed or thanked you), CALL the "
            "`ready_for_next_section` tool AND, in that same turn, tell them out loud in one short "
            "sentence that they've completed this section and can tap the on-screen button to "
            "continue whenever they're ready. NEVER call the tool silently — always speak. Keep "
            "teaching and probing until they can explain it. You CANNOT switch sections yourself — "
            "only the learner's on-screen button advances. So if, after you've signalled readiness, "
            "the learner says they want to move on, simply and warmly tell them to tap the button "
            "(e.g. 'Tap Next section and we'll dive right in') — never say you 'can't', never "
            "apologise, and never promise to teach the next section in this turn. If they keep "
            "discussing THIS section, keep helping."
        )
        if modules and idx >= len(modules) - 1:
            section_block += (
                " This is the FINAL section: once they understand it, briefly wrap up the whole "
                "document in a sentence or two, then call `ready_for_next_section`."
            )
        if resumed:
            section_block += (
                " The learner PAUSED partway through this section last time — recall from your "
                "memory of the conversation where you left off, briefly recap, and CONTINUE from "
                "there. Do NOT restart the section from the beginning."
            )

    return (
        f"You are Praxos, a warm but RIGOROUS voice tutor teaching '{doc.name}'. Teach "
        "conversationally in short turns: explain a key point, then ask a question that makes the "
        "learner EXPLAIN the idea in their own words. Ground your teaching in the SECTION MATERIAL "
        "below; explain it in your own words and use general knowledge only to clarify or give "
        "everyday examples, never to add facts that contradict it. You ALWAYS have the material you "
        "need — NEVER claim a section's text is missing or unavailable, and NEVER ask the learner to "
        "paste, type, share, upload, or provide any document, section, or text (this is a voice "
        "conversation; they cannot send you text). If a section's material is brief, just teach the "
        "concept it states concisely. Keep replies under three sentences. Speak FIRST the moment the "
        "session begins — start teaching immediately; do NOT introduce yourself or greet at length; "
        "never sit silently.\n"
        "RIGOR (critical): acknowledgements and filler are NOT answers. If the learner only says "
        "things like 'thank you', 'ok', 'yeah', 'mm', 'right', 'got it', stays silent, gives a single "
        "word, or says something off-topic or that sounds like stray background speech, do NOT say "
        "'exactly/right/correct', do NOT give credit, do NOT call the tool, and do NOT move on. Ask "
        "them to explain it in their own words, or re-ask. If unsure you heard a real answer, ask "
        "them to repeat it."
        f"{memory_block}{plan_block}{section_block}"
        f"\n--- SECTION MATERIAL ---\n{context}"
    )


@router.post("/start")
def start_session(body: StartIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    modules = plan_service.get_modules(db, doc.id)
    prog = _progress_row(db, user.id, doc.id)

    # Re-learn: reset saved progress + re-open the path item, so a learner can redo a
    # mastered or low-scoring document from the first section to raise their understanding.
    if body.restart:
        if prog is not None:
            prog.module_idx = 0
            prog.status = "in_progress"
            prog.score = None
        item = db.scalar(
            select(models.LearningPathItem).where(
                models.LearningPathItem.user_id == user.id,
                models.LearningPathItem.title == doc.name,
            )
        )
        if item is not None and item.status == "mastered":
            item.status = "in_progress"
            item.progress = 0
        db.flush()

    # Which section to teach: restart → first; an explicit override; else the saved point.
    if body.restart:
        idx = 0
    elif body.moduleIdx is not None:
        idx = body.moduleIdx
    elif prog is not None:
        idx = prog.module_idx
    else:
        idx = 0
    idx = max(0, min(idx, max(0, len(modules) - 1)))
    resumed = bool(
        prog is not None
        and prog.status == "paused"
        and (body.moduleIdx is None or body.moduleIdx == prog.module_idx)
    )

    # Restore what we already know about this learner (incl. the published lesson
    # plan + any paused conversation) so the tutor opens confidently. Empty string
    # for a brand-new learner / if memory is unconfigured or slow.
    recap = memory.recap_for_tutor(
        workspace_id=user.workspace_id,
        user_id=user.id,
        document_id=doc.id,
        doc_name=doc.name,
    )
    instructions = _build_instructions(doc, modules, idx, recap, resumed)
    realtime = ai.mint_realtime_session(instructions)
    if realtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice teaching needs an OpenAI API key on the server.",
        )

    # Mark this section in-progress so closing + reopening resumes it.
    if prog is None:
        db.add(
            models.SectionProgress(
                user_id=user.id, document_id=doc.id, module_idx=idx, status="in_progress"
            )
        )
    else:
        prog.module_idx = idx
        prog.status = "in_progress"
    db.commit()

    # The ephemeral client secret is the top-level ``value`` (GA Realtime API).
    return {
        "document": {"id": doc.id, "name": doc.name},
        "clientSecret": realtime.get("value"),
        "expiresAt": realtime.get("expires_at"),
        "returning": bool(recap),  # frontend can show "Welcome back" vs "Let's begin"
        "moduleIdx": idx,
        "moduleTitle": modules[idx].title if modules else None,
        "totalModules": len(modules),
        "isLast": idx >= len(modules) - 1 if modules else True,
        "resumed": resumed,
    }


class SectionIn(BaseModel):
    documentId: int
    moduleIdx: int


@router.post("/section")
def section_instructions(body: SectionIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    """Instructions for advancing to a section MID-session — the client sends these via a
    realtime session.update so the tutor moves on WITHOUT reconnecting (keeps context, no
    re-intro). Recap framing is baked in (advancing=True)."""
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)
    modules = plan_service.get_modules(db, doc.id)
    idx = max(0, min(body.moduleIdx, max(0, len(modules) - 1)))
    cur = modules[idx] if modules else None
    return {
        "moduleIdx": idx,
        "moduleTitle": cur.title if cur else None,
        "totalModules": len(modules),
        "isLast": idx >= len(modules) - 1 if modules else True,
        "instructions": _build_instructions(doc, modules, idx, advancing=True),
    }


@router.post("/score")
def score_session(body: ScoreIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    doc = _doc_in_workspace(db, body.documentId, user.workspace_id)

    transcript = [{"role": t.role, "text": t.text} for t in body.transcript if t.text.strip()]

    # Noise-proof gate: if the learner said essentially NOTHING (silence/noise →
    # empty or pure filler), skip the LLM and score low. Real answers — even short or
    # numeric ones — go to the strict rubric in score_understanding, which judges them.
    _filler = {
        "yeah", "yes", "no", "ok", "okay", "um", "uh", "hmm", "mhm", "mm",
        "right", "sure", "nope", "yep", "what", "huh", "idk", "dunno", "the", "a", "i",
        "thank", "thanks", "you", "thankyou", "please",
    }
    learner_text = " ".join(t["text"] for t in transcript if t["role"] == "learner")
    substantive = [w for w in re.findall(r"[a-z0-9']+", learner_text.lower()) if w not in _filler]
    if not substantive:
        result = {
            "score": 10,
            "summary": "Not enough was said to show understanding — explain the material in your own words.",
            "topics": [],
            "strengths": [],
            "gaps": ["Give a full spoken explanation of the section."],
        }
    else:
        result = ai.score_understanding(doc.name, transcript)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scoring needs an OpenAI API key on the server.",
            )

    score = result["score"]
    # Understanding reflects the learner's LATEST attempt — so re-learning a document
    # immediately raises (or lowers) it, instead of being capped by an average.
    user.understanding = score
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

    # Advance section progress: a pause keeps the same section (resume later); a
    # finished section moves the resume point forward and bumps the path %.
    modules = plan_service.get_modules(db, doc.id)
    last = max(0, len(modules) - 1)
    prog = _progress_row(db, user.id, doc.id)
    cur_idx = body.moduleIdx if body.moduleIdx is not None else (prog.module_idx if prog else 0)
    if prog is None:
        prog = models.SectionProgress(user_id=user.id, document_id=doc.id, module_idx=cur_idx)
        db.add(prog)
    if body.paused:
        prog.module_idx = cur_idx
        prog.status = "paused"
    elif cur_idx < last:
        prog.module_idx = cur_idx + 1
        prog.status = "in_progress"
    else:
        prog.module_idx = last
        prog.status = "completed"
    prog.score = score

    if modules:
        done = cur_idx if body.paused else cur_idx + 1
        pct = min(100, round(100 * done / len(modules)))
        item = db.scalar(
            select(models.LearningPathItem).where(
                models.LearningPathItem.user_id == user.id,
                models.LearningPathItem.title == doc.name,
            )
        )
        if item is not None:
            item.progress = pct
            if pct >= 100:
                item.status = "mastered"
                # Unlock the next document in the path so a multi-document / EXPANDED
                # cohort flows forward — a doc added later becomes learnable in turn,
                # without disturbing anything the learner has already done.
                nxt = db.scalar(
                    select(models.LearningPathItem)
                    .where(
                        models.LearningPathItem.user_id == user.id,
                        models.LearningPathItem.status == "locked",
                    )
                    .order_by(models.LearningPathItem.idx)
                )
                if nxt is not None:
                    nxt.status = "up_next"
            else:
                item.status = "in_progress"
    db.commit()

    return {
        "score": score,
        "understanding": user.understanding,
        "summary": result.get("summary", ""),
        "topics": result.get("topics", []),
        "strengths": result.get("strengths", []),
        "gaps": result.get("gaps", []),
        "moduleIdx": prog.module_idx,
        "totalModules": len(modules),
        "courseComplete": prog.status == "completed",
        "paused": body.paused,
    }
