from __future__ import annotations

"""Thin OpenAI wrapper used by indexing (embeddings), scoring (chat), and the
realtime voice session (ephemeral token mint). Everything degrades gracefully
when ``OPENAI_API_KEY`` is unset so the app still runs in review/dev mode."""

import json
from functools import lru_cache
from typing import Optional

import httpx

from .config import settings


@lru_cache
def _client():
    if not settings.openai_enabled:
        return None
    from openai import OpenAI

    return OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """Embed a batch of texts. Returns None when OpenAI isn't configured."""
    client = _client()
    if client is None or not texts:
        return None
    resp = client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> Optional[list[float]]:
    out = embed_texts([text])
    return out[0] if out else None


def score_understanding(doc_name: str, transcript: list[dict]) -> Optional[dict]:
    """Grade a teaching-session transcript into an understanding score (0-100)
    plus a short summary and per-topic breakdown. Returns None when OpenAI is
    unconfigured. ``transcript`` is a list of {role, text} turns."""
    client = _client()
    if client is None or not transcript:
        return None
    convo = "\n".join(f"{t.get('role', '?')}: {t.get('text', '')}" for t in transcript)
    system = (
        "You are an assessor for a corporate learning platform. A learner was just "
        f"taught the document '{doc_name}' by a voice tutor and answered questions about it. "
        "From the transcript, judge how well the learner UNDERSTOOD the material — reward "
        "correct reasoning in their own words, penalise wrong or absent answers. Be strict "
        "and noise-proof: ignore filler, mishearings, and the tutor's own statements; score "
        "only the learner's demonstrated understanding. Respond ONLY as JSON: "
        '{"score": <int 0-100>, "summary": "<one sentence>", '
        '"topics": [{"name": "<topic>", "score": <int 0-100>}], '
        '"strengths": ["..."], "gaps": ["..."]}'
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Transcript:\n{convo}"},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    data["score"] = max(0, min(100, int(data.get("score", 0))))
    return data


def mint_realtime_session(instructions: str) -> Optional[dict]:
    """Create an ephemeral Realtime client secret the browser uses to open a
    WebRTC voice connection directly to OpenAI (the key never reaches the client).
    Session config (model, voice, instructions, input transcription) is baked into
    the secret server-side. Returns None when OpenAI isn't configured.

    Returns a dict with at least ``value`` (the ephemeral token) and ``expires_at``.
    """
    if not settings.openai_enabled:
        return None
    resp = httpx.post(
        "https://api.openai.com/v1/realtime/client_secrets",
        headers={
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "session": {
                "type": "realtime",
                "model": settings.OPENAI_REALTIME_MODEL,
                "instructions": instructions,
                "audio": {
                    "input": {
                        "transcription": {"model": "whisper-1"},
                        # Filter steady background noise so the mic doesn't pick up
                        # room hum / keyboard as "speech".
                        "noise_reduction": {"type": "near_field"},
                        # Server-side voice-activity detection. A higher threshold
                        # + longer trailing silence stops the tutor from thinking
                        # the learner is talking when it's just background noise.
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.72,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 900,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {"voice": settings.OPENAI_REALTIME_VOICE},
                },
            }
        },
        timeout=20,
    )
    if resp.status_code >= 400:
        # Surface OpenAI's reason (e.g. an unknown session field) instead of a bare 500.
        raise RuntimeError(f"Realtime mint failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()
