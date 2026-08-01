from __future__ import annotations

"""Poke (poke.com) integration.

What Poke's public API actually is, verified against
https://poke.com/docs/api and a live call:

    POST https://poke.com/api/v1/inbound/api-message
    Authorization: Bearer <V2 key>
    {"message": "..."}            ->  {"success": true, "message": "Message sent successfully"}

It hands a message to the account's Poke agent and returns an acknowledgement.
There is no completion in the response, no model parameter and no streaming, so
Poke cannot be the inference step between speech-to-text and text-to-speech —
nothing would come back to speak. ``lms_app/llm.py`` says so explicitly if
someone sets LLM_PROVIDER=poke.

What it IS good for, and what is wired here: pushing an event out to whoever
runs the workspace. Praxos uses it to nudge an admin when a learner's
understanding drops, and to confirm a cohort has finished a document.

Every call is best-effort: a Poke outage must never break a lesson.
"""

import logging
from typing import Optional

import httpx

from .config import settings

logger = logging.getLogger("praxos.poke")

API_URL = "https://poke.com/api/v1/inbound/api-message"


def enabled() -> bool:
    return bool(settings.POKE_API_KEY)


def send(message: str, **context: object) -> bool:
    """Hand a message to the account's Poke agent. Extra keyword arguments ride
    along in the payload — Poke forwards the whole body to the agent as context.
    Returns True on a 2xx. Never raises."""
    if not enabled() or not message.strip():
        return False
    payload: dict = {"message": message.strip()}
    payload.update({k: v for k, v in context.items() if v is not None})
    try:
        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {settings.POKE_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.POKE_TIMEOUT,
        )
        if resp.status_code >= 300:
            logger.warning("poke send failed: %s %s", resp.status_code, resp.text[:200])
            return False
        return bool(resp.json().get("success", True))
    except Exception as exc:
        logger.warning("poke send error: %s", exc)
        return False


def notify_at_risk(*, learner: str, document: str, score: int, workspace: str) -> bool:
    """Tell the admin a learner is not retaining a document."""
    return send(
        f"Praxos: {learner} scored {score}/100 on '{document}' in the {workspace} workspace "
        f"and is below the at-risk threshold. Consider following up.",
        source="praxos",
        event="learner_at_risk",
        learner=learner,
        document=document,
        score=score,
        workspace=workspace,
    )


def notify_document_complete(*, learner: str, document: str, score: int, workspace: str) -> bool:
    """Tell the admin a learner has finished a document, with the final score."""
    return send(
        f"Praxos: {learner} completed '{document}' in the {workspace} workspace with an "
        f"understanding score of {score}/100.",
        source="praxos",
        event="document_complete",
        learner=learner,
        document=document,
        score=score,
        workspace=workspace,
    )


def inference_unavailable_reason() -> Optional[str]:
    """Why Poke cannot back LLM_PROVIDER, for surfacing in /api/health."""
    from .llm import _POKE_INFERENCE_ERROR

    return _POKE_INFERENCE_ERROR if settings.LLM_PROVIDER.lower() == "poke" else None
