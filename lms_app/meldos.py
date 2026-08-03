from __future__ import annotations

"""MeldOS gateway client.

MeldOS fronts the model with an OpenAI-compatible endpoint:

    POST $MELDOS_API_BASE_URL/v1/chat/completions
    Authorization: Bearer $MELDOS_APPLICATION_KEY
    {"model": "company-chat-model", "messages": [...]}

Cost attribution
----------------
Requests may name the person they were made for, via exactly one of:

  X-End-User-Token   the signed-in user's own Clerk token -> recorded VERIFIED.
                     MeldOS validates it against issuer
                     https://able-dinosaur-26.clerk.accounts.dev, audience praxos.
  X-End-User-Id      a stable identifier -> recorded CLAIMED, no sign-in needed.

Praxos sends the token whenever the call is made while handling that person's own
authenticated request, and falls back to the claimed header for work that runs
outside a request — the voice agent worker and background indexing, which
authenticate with a service secret and never hold a user token.

MeldOS additionally REQUIRES ``X-Session-ID`` on every model request, to group
the calls belonging to one piece of work. Praxos passes the thing that genuinely
is the session — the teaching sitting, or the LiveKit room for a live lesson —
and falls back to a fresh id so a call can never fail validation for want of one.

Two hard rules, enforced in this module rather than left to callers:

  1. The user's token leaves this process ONLY to the MeldOS host. ``_is_meldos``
     checks the destination before any attribution header is attached, so
     repointing LLM_BASE_URL at another provider cannot leak it.
  2. Neither the application key nor a user token is ever logged. Nothing here
     logs headers, and ``_safe`` scrubs response text before it is logged.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .config import settings

logger = logging.getLogger("praxos.meldos")

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class MeldOSError(RuntimeError):
    """A MeldOS call failed. ``status`` is the upstream HTTP status (0 for a
    transport failure); ``detail`` is safe to show an operator — it never
    contains the application key or a user token."""

    def __init__(self, status: int, detail: str, retry_after: Optional[str] = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


@dataclass(frozen=True)
class EndUser:
    """Who a model call is on behalf of.

    ``token`` is the person's own signed-in credential and yields VERIFIED
    attribution. ``name`` is used for CLAIMED attribution when no token is in
    hand. Deliberately the person's display name, not their Clerk id.
    """

    token: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def verified(cls, token: Optional[str]) -> "EndUser":
        return cls(token=token or None)

    @classmethod
    def claimed(cls, name: Optional[str]) -> "EndUser":
        return cls(name=(name or "").strip() or None)

    @classmethod
    def for_user(cls, token: Optional[str], name: Optional[str]) -> "EndUser":
        """Prefer a verified attribution, but keep the name so a rejected token
        can degrade to a claimed one rather than failing the whole call."""
        return cls(token=token or None, name=(name or "").strip() or None)

    def without_token(self) -> "EndUser":
        return EndUser(token=None, name=self.name)

    @property
    def empty(self) -> bool:
        return not self.token and not self.name


def enabled() -> bool:
    return bool(settings.MELDOS_API_BASE_URL and settings.MELDOS_APPLICATION_KEY)


def base_url() -> str:
    """The configured base URL, normalised to an absolute https origin.

    The value is documented as a bare host (``control-api-testing.up.railway.app``),
    so a missing scheme is normal input, not an error.
    """
    raw = (settings.MELDOS_API_BASE_URL or "").strip().rstrip("/")
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    return raw


def chat_url() -> str:
    return f"{base_url()}{CHAT_COMPLETIONS_PATH}"


def _is_meldos(url: str) -> bool:
    """True only when ``url`` points at the configured MeldOS host. Every
    attribution header is gated on this — a user's token must never be attached
    to a request bound anywhere else."""
    target = base_url()
    if not target or not url:
        return False
    a, b = urlparse(url), urlparse(target)
    return bool(a.hostname) and a.hostname == b.hostname and a.scheme == b.scheme


def attribution_headers(end_user: Optional[EndUser], *, url: Optional[str] = None) -> dict[str, str]:
    """The one attribution header for this call, or {} when there is nothing to
    attribute or the destination is not MeldOS.

    Sends at most one header: the token (verified) wins over the name (claimed),
    because a verified attribution is strictly better and sending both would be
    ambiguous.
    """
    if end_user is None or end_user.empty:
        return {}
    if not _is_meldos(url or chat_url()):
        return {}
    if end_user.token:
        return {"X-End-User-Token": end_user.token}
    return {"X-End-User-Id": end_user.name or ""}


def _safe(text: str, limit: int = 300) -> str:
    """Response text made safe to log: truncated, and with the application key
    scrubbed in case an upstream error echoes the Authorization header back."""
    out = (text or "")[:limit]
    key = settings.MELDOS_APPLICATION_KEY
    if key:
        out = out.replace(key, "<redacted>")
    return out


def _raise_for_status(resp: httpx.Response) -> None:
    """Turn the documented failure modes into an actionable MeldOSError.

    Nothing here includes the request headers, so the application key and any
    user token stay out of the exception, the logs and any surfaced message.
    """
    status = resp.status_code
    if status < 400:
        return
    body = _safe(resp.text)
    if status in (401, 403):
        # Log the scrubbed body. Returning only a canned message meant a live
        # failure showed up as "MeldOS 401" with no way to tell an invalid
        # application key from a rejected end-user token — two very different
        # problems. _safe() removes the key, so this is safe to log.
        logger.error("meldos %s: %s", status, body)
        if status == 401:
            raise MeldOSError(401, f"MeldOS rejected the request (401). {body}")
        raise MeldOSError(403, f"MeldOS denied the request (403). {body}")
    if status == 429:
        retry_after = resp.headers.get("Retry-After")
        raise MeldOSError(
            429,
            "MeldOS rate limit reached (429)." + (f" Retry after {retry_after}s." if retry_after else ""),
            retry_after=retry_after,
        )
    if status >= 500:
        raise MeldOSError(status, f"MeldOS upstream error ({status}).")
    raise MeldOSError(status, f"MeldOS returned {status}: {body}")


def chat_completion(
    messages: list[dict],
    *,
    end_user: Optional[EndUser] = None,
    session_id: Optional[str] = None,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> dict:
    """One chat completion through MeldOS. Returns the parsed response body.

    Raises MeldOSError on any failure. Uses httpx directly rather than an SDK so
    the exact headers sent are visible in one place and no client library can log
    the Authorization header on our behalf.
    """
    if not enabled():
        raise MeldOSError(0, "MeldOS is not configured (MELDOS_API_BASE_URL / MELDOS_APPLICATION_KEY).")

    url = chat_url()
    headers = {
        "Authorization": f"Bearer {settings.MELDOS_APPLICATION_KEY}",
        "Content-Type": "application/json",
        # Required by MeldOS: groups the calls belonging to one piece of work.
        "X-Session-ID": session_id or f"praxos-{uuid.uuid4()}",
        **attribution_headers(end_user, url=url),
    }
    payload: dict[str, Any] = {"model": settings.MELDOS_MODEL, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        resp = httpx.post(
            url, headers=headers, json=payload, timeout=timeout or settings.MELDOS_TIMEOUT
        )
    except httpx.HTTPError as exc:
        # exc carries the URL only — never the headers.
        raise MeldOSError(0, f"Could not reach MeldOS: {type(exc).__name__}") from None

    # Attribution is metadata; the completion is the product. If MeldOS refuses
    # the END-USER token (wrong issuer/audience/shape — all deployment concerns
    # outside this request), retry once attributing by name instead of failing.
    # Without this, one misconfigured sign-in integration silently costs every
    # learner their grade, which is exactly what happened in production.
    if resp.status_code in (401, 403) and "X-End-User-Token" in headers:
        logger.warning(
            "meldos rejected the end-user token (%s); retrying with claimed attribution",
            resp.status_code,
        )
        return chat_completion(
            messages,
            end_user=(end_user.without_token() if end_user else None),
            session_id=session_id,
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    _raise_for_status(resp)
    try:
        return resp.json()
    except ValueError:
        raise MeldOSError(resp.status_code, "MeldOS returned a non-JSON body.") from None


def completion_text(body: dict) -> Optional[str]:
    """The assistant message from an OpenAI-shaped completion body."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
