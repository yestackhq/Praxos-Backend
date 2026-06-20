# Sessions

Sessions are the boilerplate's default authentication mechanism. All built-in API routes use session auth.

## Protecting Routes

Import the session dependencies and add them to your routes:

```python
from typing import Annotated, Any
from fastapi import APIRouter, Depends

from ...infrastructure.auth.session.dependencies import get_current_user

router = APIRouter()


@router.get("/my-profile")
async def get_profile(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    return {"user_id": current_user["id"], "email": current_user["email"]}
```

If the request doesn't have a valid session, the boilerplate returns `401 Unauthorized`.

### Available Dependencies

All from `src/infrastructure/auth/session/dependencies.py`.

**`get_current_user`** — Returns the authenticated user dict. Raises 401 if not authenticated.

```python
@router.get("/dashboard")
async def dashboard(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    return {"welcome": current_user["username"]}
```

**`get_current_superuser`** — Same as `get_current_user`, plus checks `is_superuser=True`. Raises 403 if not a superuser.

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: Annotated[dict[str, Any], Depends(get_current_superuser)],
) -> None:
    # Only superusers reach this code
    ...
```

**`get_optional_user`** — Returns the user dict if authenticated, `None` otherwise. Never raises.

```python
@router.get("/products")
async def list_products(
    current_user: Annotated[dict[str, Any] | None, Depends(get_optional_user)],
) -> list[dict[str, Any]]:
    if current_user:
        # Personalize for logged-in users
        ...
```

**`get_current_session_data`** — Returns the full `SessionData` object (id, user_id, ip, device info, timestamps). Useful for endpoints like `/check-auth` that need to expose session metadata.

### Protecting Entire Routers

Apply auth to every route in a router:

```python
router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(get_current_superuser)],
)


@router.get("/stats")
async def stats() -> dict[str, Any]:
    # Already authenticated at the router level
    ...
```

Note: router-level dependencies don't inject values into handlers. If you need the user object inside the handler, also add `Depends(get_current_user)` to that specific route.

## How Sessions Work

When a user hits `POST /api/v1/auth/login`:

1. Login rate limiter checks IP+username (`LOGIN_MAX_ATTEMPTS` per `LOGIN_WINDOW_MINUTES`)
2. `authenticate_user(...)` validates the credentials
3. `SessionManager.create_session(...)` writes a record to the configured backend (Redis by default)
4. A new CSRF token is generated and bound to the session
5. Two cookies are set on the response:
    - `session_id` — HTTP-only, the session identifier
    - `csrf_token` — readable by JS, mirrors the CSRF token returned in the JSON body

On every subsequent request, the session dependency:

1. Reads `session_id` from cookies
2. Looks it up in the configured backend; rejects expired or missing sessions
3. For mutating requests (POST/PUT/DELETE/PATCH), validates the CSRF token if `CSRF_ENABLED=true`
4. Returns the user record (joined with the `Tier` relationship via `lazy="selectin"`)

Logout (`POST /api/v1/auth/logout`) terminates the session record and clears the cookies.

## CSRF Protection

Session auth ships with CSRF protection. For non-GET requests, send the CSRF token via either:

- The `csrf_token` cookie (browsers send it automatically), or
- The `X-CSRF-Token` header (typical for JS clients)

```javascript
const csrfToken = getCookie('csrf_token');

await fetch('/api/v1/users/', {
    method: 'POST',
    credentials: 'include',          // include cookies cross-origin
    headers: {
        'X-CSRF-Token': csrfToken,
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
});
```

Need a fresh token mid-session? Hit `POST /api/v1/auth/refresh-csrf` — it returns a new token and sets the cookie.

For dev/test environments where CSRF gets in the way, set `CSRF_ENABLED=false`.

## Device Tracking

Sessions capture the IP address and parsed User-Agent fields. Inspect via the session dep:

```python
from typing import Annotated, Any
from fastapi import Depends

from src.infrastructure.auth.session.dependencies import get_current_session_data
from src.infrastructure.auth.session.schemas import SessionData


@router.get("/my-current-session")
async def my_session(
    session_data: Annotated[SessionData, Depends(get_current_session_data)],
) -> dict[str, Any]:
    return {
        "ip": session_data.ip_address,
        "user_agent": session_data.user_agent,
        "device_info": session_data.device_info,   # browser, os, is_mobile, etc.
        "created_at": session_data.created_at,
        "last_activity": session_data.last_activity,
    }
```

This makes it straightforward to build "your active sessions" UIs or detect suspicious activity.

## Login Rate Limiting

Failed login attempts are tracked per IP+username. After `LOGIN_MAX_ATTEMPTS` failures within `LOGIN_WINDOW_MINUTES`, further attempts on `/api/v1/auth/login` are blocked.

This happens automatically in the login route — you don't need to wire it up. The defaults (5 attempts in 15 minutes) are conservative; tune per your threat model.

## Session Limits

Per-user concurrent session count is capped by `MAX_SESSIONS_PER_USER` (default 5). When a user logs in beyond this cap, the oldest session is terminated.

## Storage Backends

Sessions are stored server-side. Configure via `SESSION_BACKEND`:

| Value | When to use |
|-------|-------------|
| `redis` *(default)* | Production. Supports key expiration, pattern scans for cleanup, persists across restarts |
| `memcached` | Production alternative — choose based on what your infrastructure already runs |
| `memory` | Tests only. Cleared on restart, not safe for multi-process deploys |

Storage backends live in `src/infrastructure/auth/session/backends/`.

## Configuration

```env
# Backend
SESSION_BACKEND=redis

# Lifetime
SESSION_TIMEOUT_MINUTES=30           # inactive sessions expire
SESSION_CLEANUP_INTERVAL_MINUTES=15  # how often the storage backend sweeps expired entries
SESSION_COOKIE_MAX_AGE=86400         # 1 day — total cookie lifetime

# Per-user cap
MAX_SESSIONS_PER_USER=5

# Cookie security (HTTPS only)
SESSION_SECURE_COOKIES=true

# CSRF
CSRF_ENABLED=true

# Login rate limiting
LOGIN_MAX_ATTEMPTS=5
LOGIN_WINDOW_MINUTES=15
```

For development you'll typically set `SESSION_SECURE_COOKIES=false` and `CSRF_ENABLED=false` so cookies work over plain HTTP and curl/Postman aren't blocked. Re-enable both for staging and production.

## Login & Logout Flow

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your_admin_password" \
  -c cookies.txt
```

Response:

```json
{ "csrf_token": "..." }
```

The HTTP-only `session_id` cookie is now in `cookies.txt`. The CSRF token is also set as a cookie *and* returned in the body so JS clients can store it (browsers can't read HTTP-only cookies).

### Authenticated Request

```bash
curl http://localhost:8000/api/v1/users/me -b cookies.txt
```

For mutating requests, add the CSRF header:

```bash
curl -X POST http://localhost:8000/api/v1/users/ \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: <token-from-login-response>" \
  -d '{"name": "...", "username": "...", "email": "...", "password": "..."}'
```

### Refresh CSRF Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh-csrf -b cookies.txt
```

### Logout

```bash
curl -X POST http://localhost:8000/api/v1/auth/logout -b cookies.txt
```

Terminates the session and clears the cookies.

## Key Files

| Component | Location |
|-----------|----------|
| Dependencies | `backend/src/infrastructure/auth/session/dependencies.py` |
| Session manager | `backend/src/infrastructure/auth/session/manager.py` |
| Storage backends | `backend/src/infrastructure/auth/session/backends/` |
| Schemas | `backend/src/infrastructure/auth/session/schemas.py` |
| Login/logout routes | `backend/src/infrastructure/auth/routes.py` |
| Auth settings | `backend/src/infrastructure/config/settings.py` (`AuthSettings`) |

---

[← Authentication Overview](index.md){ .md-button } [User Management →](user-management.md){ .md-button .md-button--primary }
