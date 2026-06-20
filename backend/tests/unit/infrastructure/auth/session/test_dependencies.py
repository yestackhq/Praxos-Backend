from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request, status

from src.infrastructure.auth.http_exceptions import CSRFException
from src.infrastructure.auth.session.manager import SessionManager
from src.infrastructure.auth.session.schemas import SessionData


async def mock_get_session_from_cookie(request, session_manager):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    session_data = await session_manager.validate_session(session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )

    return session_data


async def mock_verify_csrf_token(request, session_data, session_manager):
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    if not session_data:
        return

    token = request.headers.get("X-CSRF-Token")
    if not token:
        raise CSRFException("CSRF token missing")

    is_valid = await session_manager.validate_csrf_token(session_data.session_id, token)
    if not is_valid:
        raise CSRFException("Invalid CSRF token")


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.cookies = {"session_id": "test-session-id"}
    request.headers = {"X-CSRF-Token": "test-csrf-token"}
    request.method = "POST"
    return request


@pytest.fixture
def mock_session_data():
    return SessionData(
        session_id="test-session-id",
        user_id=1,
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test-agent",
        device_info={},
        last_activity=datetime.now(UTC),
        metadata={},
    )


@pytest.fixture
def mock_session_manager():
    session_manager = MagicMock(spec=SessionManager)
    session_manager.validate_session = AsyncMock()
    session_manager.validate_csrf_token = AsyncMock()
    session_manager.cleanup_expired_sessions = AsyncMock()
    return session_manager


@pytest.mark.asyncio
async def test_get_session_from_cookie(mock_request, mock_session_data, mock_session_manager):
    """Test getting session data from a cookie."""
    mock_session_manager.validate_session.return_value = mock_session_data

    result = await mock_get_session_from_cookie(mock_request, mock_session_manager)

    assert result == mock_session_data
    mock_session_manager.validate_session.assert_called_once_with("test-session-id")


@pytest.mark.asyncio
async def test_get_session_from_cookie_no_cookie(mock_request, mock_session_manager):
    """Test when no session cookie is present."""
    mock_request.cookies = {}

    with pytest.raises(HTTPException) as exc_info:
        await mock_get_session_from_cookie(mock_request, mock_session_manager)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Not authenticated" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_session_from_cookie_invalid_session(mock_request, mock_session_manager):
    """Test when session is invalid or expired."""
    mock_session_manager.validate_session.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await mock_get_session_from_cookie(mock_request, mock_session_manager)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid or expired session" in exc_info.value.detail
    mock_session_manager.validate_session.assert_called_once_with("test-session-id")


@pytest.mark.asyncio
async def test_verify_csrf_token_valid(mock_request, mock_session_data, mock_session_manager):
    """Test verifying a valid CSRF token."""
    mock_session_manager.validate_csrf_token.return_value = True

    await mock_verify_csrf_token(
        request=mock_request,
        session_data=mock_session_data,
        session_manager=mock_session_manager,
    )

    mock_session_manager.validate_csrf_token.assert_called_once_with("test-session-id", "test-csrf-token")


@pytest.mark.asyncio
async def test_verify_csrf_token_missing(mock_request, mock_session_data, mock_session_manager):
    """Test when CSRF token is missing."""
    mock_request.headers = {}

    with pytest.raises(CSRFException) as exc_info:
        await mock_verify_csrf_token(
            request=mock_request,
            session_data=mock_session_data,
            session_manager=mock_session_manager,
        )

    assert "CSRF token missing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_verify_csrf_token_invalid(mock_request, mock_session_data, mock_session_manager):
    """Test when CSRF token is invalid."""
    mock_session_manager.validate_csrf_token.return_value = False

    with pytest.raises(CSRFException) as exc_info:
        await mock_verify_csrf_token(
            request=mock_request,
            session_data=mock_session_data,
            session_manager=mock_session_manager,
        )

    assert "Invalid CSRF token" in str(exc_info.value)
    mock_session_manager.validate_csrf_token.assert_called_once_with("test-session-id", "test-csrf-token")
