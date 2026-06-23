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
        "You are a STRICT assessor for a corporate learning platform. A learner was taught "
        f"the document '{doc_name}' by a voice tutor, then answered questions. Score ONLY the "
        "LEARNER's turns — completely ignore the tutor's words, and ignore filler, mishearings, "
        "and transcription noise. Give credit only for correct ideas the learner expresses IN "
        "THEIR OWN WORDS; never give credit just because the tutor explained something. "
        "Rubric (0-100): 0-15 = no real answer / filler / off-topic / unintelligible; "
        "16-40 = vague or mostly wrong; 41-69 = partially correct with clear gaps; "
        "70-89 = solid, mostly correct; 90-100 = thorough and precise. When in doubt, score LOW. "
        "Respond ONLY as JSON: "
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


def generate_lesson_plan(doc_name: str, chunks: list[str]) -> Optional[list[dict]]:
    """Design a section-by-section teaching plan for a document from its text.

    Returns an ordered list of section dicts — each a coherent unit a voice tutor
    teaches in one sitting:
      {title, description, topics: [str], minutes: int, chunk_start: int, chunk_end: int}
    chunk_start/chunk_end index into ``chunks`` (inclusive start, exclusive end) so
    the tutor is later grounded in just that section's text. None when OpenAI is
    unconfigured or the model returns nothing usable."""
    client = _client()
    if client is None or not chunks:
        return None
    n = len(chunks)
    numbered = "\n\n".join(f"[chunk {i}]\n{c[:900]}" for i, c in enumerate(chunks))[:24000]
    system = (
        "You are an expert curriculum designer for a voice-based microlearning tutor. "
        f"You are given the document '{doc_name}', split into numbered chunks. Design a teaching "
        "plan: break it into 3-8 ordered SECTIONS the tutor will teach one at a time, each a "
        "coherent unit of a few minutes. For each section give: a short title; a 1-2 sentence "
        "description of what the learner should come away understanding AND how to teach it (the "
        "approach / what to emphasise and check); 2-4 key topics; estimated minutes (3-8); and the "
        "contiguous chunk range it is taught from. Cover the whole document in order, no gaps or "
        f"overlaps; chunk indices run 0..{n - 1}. Respond ONLY as JSON: "
        '{"sections": [{"title": "...", "description": "...", "topics": ["..."], '
        '"minutes": <int>, "chunk_start": <int>, "chunk_end": <int>}]} '
        "where chunk_start is inclusive and chunk_end exclusive."
    )
    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    out: list[dict] = []
    for i, s in enumerate(data.get("sections") or []):
        try:
            cs = max(0, min(n - 1, int(s.get("chunk_start", 0))))
            ce = max(cs + 1, min(n, int(s.get("chunk_end", n))))
        except (TypeError, ValueError):
            cs, ce = 0, n
        out.append(
            {
                "title": str(s.get("title") or f"Section {i + 1}")[:160],
                "description": str(s.get("description") or "")[:2000],
                "topics": [str(t)[:80] for t in (s.get("topics") or [])][:6],
                "minutes": max(2, min(20, int(s.get("minutes", 5) or 5))),
                "chunk_start": cs,
                "chunk_end": ce,
            }
        )
    return out or None


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
                        "transcription": {"model": settings.OPENAI_TRANSCRIBE_MODEL},
                        # Filter steady background noise so the mic doesn't pick up
                        # room hum / keyboard as "speech".
                        "noise_reduction": {"type": "near_field"},
                        # Semantic VAD: a model decides when the learner has actually
                        # FINISHED a spoken turn (not a raw energy threshold), so background
                        # noise/silence doesn't trigger it. eagerness=low → it waits through
                        # the learner's mid-sentence PAUSES instead of cutting in early.
                        # interrupt_response=True → if the learner keeps talking over a reply
                        # it cleanly restarts the turn, instead of leaving an active response
                        # that blocks the next one (which deadlocked the session).
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "low",
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
