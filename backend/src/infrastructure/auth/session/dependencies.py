from typing import Annotated, Any

from fastapi import Cookie, Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ....infrastructure.auth.http_exceptions import (
    CSRFException,
    ForbiddenException,
    UnauthorizedException,
)
from ....infrastructure.database.session import async_session
from ....modules.user.crud import crud_users
from ...config.settings import get_settings
from ...logging import get_logger
from ...rate_limit.provider import get_rate_limiter_backend
from ..utils import verify_password
from .manager import SessionManager
from .schemas import SessionData
from .storage import AbstractSessionStorage, get_session_storage

settings = get_settings()
logger = get_logger()

_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the session manager singleton (initialized once, reused across requests)."""
    global _session_manager  # noqa: PLW0603
    if _session_manager is not None:
        return _session_manager

    storage: AbstractSessionStorage[SessionData] = get_session_storage(
        backend=settings.SESSION_BACKEND,
        model_type=SessionData,
        prefix="session:",
        expiration=settings.SESSION_TIMEOUT_MINUTES * 60,
        host=settings.CACHE_REDIS_HOST,
        port=settings.CACHE_REDIS_PORT,
        db=settings.CACHE_REDIS_DB,
        password=settings.CACHE_REDIS_PASSWORD,
    )

    rate_limiter = None
    if settings.RATE_LIMITER_ENABLED:
        try:
            rate_limiter = get_rate_limiter_backend(settings.RATE_LIMITER_BACKEND)
            logger.info(f"Rate limiter initialized for login attempts using {settings.RATE_LIMITER_BACKEND} backend")
        except Exception as e:
            logger.warning(f"Failed to initialize rate limiter for login attempts: {e}")
            logger.warning("Login rate limiting will be disabled")

    _session_manager = SessionManager(
        session_storage=storage,
        max_sessions_per_user=settings.MAX_SESSIONS_PER_USER,
        session_timeout_minutes=settings.SESSION_TIMEOUT_MINUTES,
        cleanup_interval_minutes=settings.SESSION_CLEANUP_INTERVAL_MINUTES,
        rate_limiter=rate_limiter,
        login_max_attempts=settings.LOGIN_MAX_ATTEMPTS,
        login_window_minutes=settings.LOGIN_WINDOW_MINUTES,
    )
    return _session_manager


async def get_session_from_cookie(
    request: Request,
    session_id: str | None = Cookie(None),
    session_manager: SessionManager = Depends(get_session_manager),
) -> SessionData | None:
    """Get session data from cookie, validating it.

    Args:
        request: The request object
        session_id: The session ID from cookie
        session_manager: The session manager

    Returns:
        The session data or None if invalid
    """
    if not session_id:
        return None

    await session_manager.cleanup_expired_sessions()

    return await session_manager.validate_session(session_id)


async def verify_csrf_token(
    request: Request,
    session_data: Annotated[SessionData | None, Depends(get_session_from_cookie)],
    csrf_token: str | None = Cookie(None),
    x_csrf_token: str | None = Header(None, alias="X-CSRF-Token"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Verify CSRF token for mutation operations.

    This should be used for POST/PUT/DELETE operations.

    Args:
        request: The request object
        session_data: The session data
        csrf_token: The CSRF token from cookie
        x_csrf_token: The CSRF token from header
        session_manager: The session manager

    Raises:
        CSRFException: If CSRF validation fails
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None

    if not settings.CSRF_ENABLED:
        logger.debug("CSRF validation disabled by configuration")
        return None

    if not session_data:
        return None

    token = x_csrf_token or csrf_token
    if not token:
        raise CSRFException("Missing CSRF token")

    is_valid = await session_manager.validate_csrf_token(session_data.session_id, token)

    if not is_valid:
        raise CSRFException("Invalid CSRF token")


async def get_current_user(
    session_data: Annotated[SessionData | None, Depends(get_session_from_cookie)],
    db: Annotated[AsyncSession, Depends(async_session)],
    _: Annotated[None, Depends(verify_csrf_token)],
) -> dict[str, Any]:
    """Get the current authenticated user.

    Args:
        session_data: The session data
        db: The database session

    Returns:
        The user data

    Raises:
        UnauthorizedException: If not authenticated or user doesn't exist
    """
    credentials_exception = UnauthorizedException("Not authenticated")

    if not session_data:
        raise credentials_exception

    if not session_data.is_active:
        raise credentials_exception

    user = await crud_users.get(db=db, id=session_data.user_id, is_deleted=False)

    if user is None:
        raise credentials_exception

    return user


async def get_optional_user(
    session_data: Annotated[SessionData | None, Depends(get_session_from_cookie)],
    db: Annotated[AsyncSession, Depends(async_session)],
) -> dict[str, Any] | None:
    """Get the current user if authenticated, None otherwise.

    Args:
        session_data: The session data
        db: The database session

    Returns:
        The user data or None
    """
    if not session_data:
        return None

    if not session_data.is_active:
        return None

    user = await crud_users.get(db=db, id=session_data.user_id, is_deleted=False)

    return user


async def get_current_superuser(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Get the current user, requiring superuser privileges.

    Args:
        current_user: The current user

    Returns:
        The user data

    Raises:
        ForbiddenException: If not a superuser
    """
    if not current_user.get("is_superuser", False):
        raise ForbiddenException("Insufficient privileges")

    return current_user


async def get_session_id_from_cookie(request: Request) -> str | None:
    """Extract session ID from cookies.

    Args:
        request: The request object

    Returns:
        The session ID from cookies or None if not present
    """
    return request.cookies.get("session_id")


async def get_current_session_data(
    request: Request,
    session_id: Annotated[str | None, Depends(get_session_id_from_cookie)],
    session_manager: Annotated[SessionManager, Depends(get_session_manager)],
) -> SessionData:
    """Get the current session data from cookie.

    Args:
        request: The request object
        session_id: The session ID from cookie
        session_manager: The session manager

    Returns:
        The session data

    Raises:
        UnauthorizedException: If not authenticated or session is invalid
    """
    if not session_id:
        raise UnauthorizedException("Not authenticated")

    session_data = await session_manager.validate_session(session_id)
    if not session_data:
        raise UnauthorizedException("Invalid or expired session")

    return session_data


async def authenticate_user(username_or_email: str, password: str, db: AsyncSession) -> dict[str, Any] | None:
    """Authenticate a user by username/email and password.

    Args:
        username_or_email: The username or email
        password: The plaintext password
        db: The database session

    Returns:
        The user data dict if authenticated, None otherwise
    """
    if "@" in username_or_email:
        user = await crud_users.get(db=db, email=username_or_email, is_deleted=False)
    else:
        user = await crud_users.get(db=db, username=username_or_email, is_deleted=False)

    if not user:
        return None

    if not await verify_password(password, user["hashed_password"]):
        return None

    return user
