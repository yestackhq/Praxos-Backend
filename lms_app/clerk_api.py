from __future__ import annotations

"""Clerk Backend API calls. Used to send real invitation emails when an admin
invites a teammate. Degrades to a no-op when CLERK_SECRET_KEY is unset — the
invite is still stored and the email-match join still works on sign-up."""

from typing import Optional

import httpx

from .config import settings

_BASE = "https://api.clerk.com/v1"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
        "Content-Type": "application/json",
    }


def create_invitation(email: str, role: str, redirect_url: Optional[str] = None) -> Optional[str]:
    """Send a Clerk invitation email. Returns the Clerk invitation id, or None
    when Clerk isn't configured (or already invited). Never raises — invite
    delivery is best-effort and must not block storing our own invite row."""
    if not settings.CLERK_SECRET_KEY:
        return None
    payload: dict = {
        "email_address": email,
        "public_metadata": {"role": role},
        "notify": True,
        "ignore_existing": True,
    }
    if redirect_url:
        payload["redirect_url"] = redirect_url
    try:
        resp = httpx.post(f"{_BASE}/invitations", headers=_headers(), json=payload, timeout=20)
        if resp.status_code in (200, 201):
            return resp.json().get("id")
    except httpx.HTTPError:
        return None
    return None


def revoke_invitation(clerk_invite_id: str) -> None:
    """Revoke a previously-sent Clerk invitation. Best-effort."""
    if not settings.CLERK_SECRET_KEY or not clerk_invite_id:
        return
    try:
        httpx.post(f"{_BASE}/invitations/{clerk_invite_id}/revoke", headers=_headers(), timeout=20)
    except httpx.HTTPError:
        pass
