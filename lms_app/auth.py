from __future__ import annotations

from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db

_bearer = HTTPBearer(auto_error=False)

# Cache the JWKS client so we don't refetch keys on every request.
_jwks_client: Optional["jwt.PyJWKClient"] = None


def _get_jwks_client() -> "jwt.PyJWKClient":
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(settings.CLERK_JWKS_URL)  # type: ignore[arg-type]
    return _jwks_client


def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk session JWT against the instance JWKS. Returns claims."""
    signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
    options = {"verify_aud": False}
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=settings.CLERK_ISSUER if settings.CLERK_ISSUER else None,
        options=options,
    )


async def optional_claims(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Like current_user but never raises: returns verified claims, or None when
    there is no/invalid token or auth is disabled. Used by endpoints that serve a
    public demo when signed out and the user's own data when signed in."""
    if not settings.auth_enabled or creds is None or not creds.credentials:
        return None
    try:
        return verify_clerk_token(creds.credentials)
    except Exception:  # noqa: BLE001
        return None


def active_membership(
    x_workspace_id: Optional[str] = Header(default=None, alias="X-Workspace-Id"),
    claims: Optional[dict] = Depends(optional_claims),
    db: Session = Depends(get_db),
):
    """Resolve the signed-in person's ACTIVE workspace membership, selected by the
    ``X-Workspace-Id`` header (falls back to their default workspace). The header is
    only a selector — resolution always verifies a real membership, so it can never
    reach a workspace the person doesn't belong to."""
    from . import workspace  # lazy import avoids a module-load cycle

    sub = claims.get("sub") if claims else None
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    ws_id = int(x_workspace_id) if (x_workspace_id and x_workspace_id.isdigit()) else None
    try:
        return workspace.resolve_active_membership(db, sub, ws_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Account not ready, retry")


async def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency.

    When Clerk is configured (CLERK_JWKS_URL set), require and verify a bearer
    token. Otherwise (review/dev mode) return a stub identity so the read API
    stays usable without auth.
    """
    if not settings.auth_enabled:
        return {"sub": "dev-user", "auth": "disabled"}

    if creds is None or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        claims = verify_clerk_token(creds.credentials)
    except Exception as exc:  # noqa: BLE001 — surface any verification failure as 401
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}") from exc
    return claims
