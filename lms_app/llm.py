from __future__ import annotations

"""Provider-agnostic LLM + embedding access.

Everything that needs a model (lesson-plan design, understanding scoring, the
voice tutor's turn-by-turn reasoning) goes through here instead of importing an
SDK directly, so the provider is a deployment choice rather than a code change.

Configuration (all env / .env):

    LLM_PROVIDER    openai | openai_compatible | poke | none   (default: openai)
    LLM_BASE_URL    OpenAI-compatible /v1 base URL
    LLM_API_KEY     key for that base URL (falls back to OPENAI_API_KEY)
    LLM_MODEL       model id, e.g. gpt-4o / llama-3.3-70b / claude-…

    EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL / EMBED_DIM   same, for vectors

``openai_compatible`` covers anything that speaks the /v1/chat/completions
contract — Together, Groq, Fireworks, vLLM, OpenRouter, LiteLLM, an in-house
gateway. Point LLM_BASE_URL at it and set LLM_MODEL; no code changes.

When MELDOS_API_BASE_URL + MELDOS_APPLICATION_KEY are set, chat completions are
routed through the MeldOS gateway instead (``lms_app/meldos.py``) so spend is
metered per application and per person. MeldOS takes precedence over LLM_* —
it is a gateway in front of a provider, not a competing one. Embeddings are not
part of the MeldOS contract and continue to use EMBED_*/LLM_*.

``poke`` is accepted so the provider name is selectable, but Poke's public API
(POST /api/v1/inbound/api-message) is a ONE-WAY message intake: it answers
``{"success": true}`` and never returns model output. It therefore cannot serve
a chat completion, and selecting it raises a clear error rather than failing
mysteriously mid-lesson. See ``lms_app/poke.py`` for what Poke IS wired to do.
"""

import json
import logging
import re
from functools import lru_cache
from typing import Optional

from . import meldos
from .config import settings

logger = logging.getLogger("praxos.llm")

# Re-exported so callers name the end user without importing the gateway module.
EndUser = meldos.EndUser
MeldOSError = meldos.MeldOSError


class LLMError(RuntimeError):
    """Raised when the configured provider cannot serve a request."""


_POKE_INFERENCE_ERROR = (
    "LLM_PROVIDER=poke cannot generate text. Poke's API "
    "(POST https://poke.com/api/v1/inbound/api-message) only ACCEPTS a message "
    "and replies {\"success\": true} — it returns no completion, exposes no model "
    "choice and does not stream, so it cannot sit in a speech-to-speech loop. "
    "Set LLM_PROVIDER=openai (or openai_compatible + LLM_BASE_URL) for inference; "
    "Poke stays wired up as a notifier (see lms_app/poke.py)."
)


# ---- clients ----------------------------------------------------------------


@lru_cache
def _chat_client():
    """An OpenAI-SDK client pointed at whichever provider is configured."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "none":
        return None
    if provider == "poke":
        raise LLMError(_POKE_INFERENCE_ERROR)
    if not settings.llm_api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=settings.llm_api_key, base_url=settings.LLM_BASE_URL)


@lru_cache
def _embed_client():
    if not settings.embed_api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=settings.embed_api_key, base_url=settings.embed_base_url)


def chat_enabled() -> bool:
    if meldos.enabled():
        return True
    try:
        return _chat_client() is not None
    except LLMError:
        return False


def embed_enabled() -> bool:
    return _embed_client() is not None


# ---- text / JSON completions -------------------------------------------------


def chat_text(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    end_user: Optional["meldos.EndUser"] = None,
    session_id: Optional[str] = None,
) -> Optional[str]:
    """A plain text completion. None when no provider is configured.

    ``end_user`` attributes the spend to a person when MeldOS is the provider;
    it is ignored otherwise (and the token is never sent off-MeldOS — see
    ``meldos.attribution_headers``)."""
    if meldos.enabled():
        body = meldos.chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            end_user=end_user,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return meldos.completion_text(body)
    client = _chat_client()
    if client is None:
        return None
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    end_user: Optional["meldos.EndUser"] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """A JSON-object completion, parsed. None when no provider is configured or
    the reply could not be parsed as an object.

    Uses response_format=json_object where supported and degrades to extracting
    the first ``{...}`` block, so an OpenAI-compatible endpoint that lacks JSON
    mode still works.

    ``end_user`` attributes the spend to a person when MeldOS is the provider.
    A MeldOS failure raises MeldOSError rather than returning None, so 401/403/
    429/5xx reach the caller as distinct, actionable conditions instead of
    collapsing into a generic "scoring unavailable".
    """
    if meldos.enabled():
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            body = meldos.chat_completion(
                messages,
                end_user=end_user,
                session_id=session_id,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
        except meldos.MeldOSError as exc:
            # A gateway that does not implement JSON mode is a shape problem, not
            # an outage — retry once in plain mode before giving up.
            if exc.status not in (400, 422):
                raise
            logger.info("meldos json mode unavailable (%s); retrying without it", exc.status)
            body = meldos.chat_completion(
                messages,
                end_user=end_user,
                session_id=session_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _parse_json_object(meldos.completion_text(body))

    client = _chat_client()
    if client is None:
        return None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=messages,
            **kwargs,
        )
    except Exception as exc:  # provider without JSON mode → retry plain
        logger.info("json mode unavailable (%s); retrying without it", exc)
        try:
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL, temperature=temperature, messages=messages, **kwargs
            )
        except Exception as exc2:
            logger.warning("chat completion failed: %s", exc2)
            return None
    return _parse_json_object(resp.choices[0].message.content)


def _parse_json_object(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    return data if isinstance(data, dict) else None


# ---- embeddings --------------------------------------------------------------


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    client = _embed_client()
    if client is None or not texts:
        return None
    resp = client.embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> Optional[list[float]]:
    out = embed_texts([text])
    return out[0] if out else None
