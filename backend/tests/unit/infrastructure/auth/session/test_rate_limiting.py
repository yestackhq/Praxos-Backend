from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.auth.routes import login
from src.infrastructure.auth.session.manager import SessionManager


@pytest.mark.asyncio
async def test_track_login_attempt_with_rate_limiter():
    """Test tracking login attempts when a rate limiter is configured."""
    mock_rate_limiter = MagicMock()
    mock_rate_limiter.increment = AsyncMock(side_effect=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    mock_rate_limiter.delete = AsyncMock(return_value=True)

    mock_storage = MagicMock()

    session_manager = SessionManager(
        session_storage=mock_storage, rate_limiter=mock_rate_limiter, login_max_attempts=5, login_window_minutes=15
    )

    is_allowed, remaining = await session_manager.track_login_attempt(
        ip_address="192.168.1.1", username="testuser", success=True
    )

    assert is_allowed is True
    assert remaining is None
    assert mock_rate_limiter.delete.call_count == 2
    mock_rate_limiter.delete.assert_any_call("login:ip:192.168.1.1")
    mock_rate_limiter.delete.assert_any_call("login:user:testuser")

    mock_rate_limiter.delete.reset_mock()
    mock_rate_limiter.increment.reset_mock()

    for i in range(1, 5):
        is_allowed, remaining = await session_manager.track_login_attempt(
            ip_address="192.168.1.1", username="testuser", success=False
        )

        assert is_allowed is True
        assert remaining == (5 - i)
        assert mock_rate_limiter.increment.call_count == i * 2

    mock_rate_limiter.increment.side_effect = [6, 6]

    is_allowed, remaining = await session_manager.track_login_attempt(
        ip_address="192.168.1.1", username="testuser", success=False
    )

    assert is_allowed is False
    assert remaining == 0


@pytest.mark.asyncio
async def test_track_login_attempt_without_rate_limiter():
    """Test tracking login attempts when no rate limiter is configured."""
    mock_storage = MagicMock()

    session_manager = SessionManager(
        session_storage=mock_storage, rate_limiter=None, login_max_attempts=5, login_window_minutes=15
    )

    is_allowed, remaining = await session_manager.track_login_attempt(
        ip_address="192.168.1.1", username="testuser", success=True
    )

    assert is_allowed is True
    assert remaining is None

    is_allowed, remaining = await session_manager.track_login_attempt(
        ip_address="192.168.1.1", username="testuser", success=False
    )

    assert is_allowed is True
    assert remaining is None


@pytest.mark.asyncio
async def test_login_endpoint_rate_limiting():
    """Test that the login endpoint applies rate limiting."""
    mock_request = MagicMock(spec=Request)
    mock_request.client.host = "192.168.1.1"

    mock_response = MagicMock(spec=Response)

    mock_form_data = MagicMock(spec=OAuth2PasswordRequestForm)
    mock_form_data.username = "testuser"
    mock_form_data.password = "password123"

    mock_db = MagicMock(spec=AsyncSession)

    mock_session_manager = MagicMock()

    mock_session_manager.track_login_attempt = AsyncMock(return_value=(True, 4))
    mock_auth_user = AsyncMock(return_value={"id": 1, "username": "testuser"})
    mock_session_manager.create_session = AsyncMock(return_value=("session_id", "csrf_token"))

    with patch("src.infrastructure.auth.routes.authenticate_user", mock_auth_user):
        result = await login(
            request=mock_request,
            response=mock_response,
            form_data=mock_form_data,
            db=mock_db,
            session_manager=mock_session_manager,
        )

    assert result == {"csrf_token": "csrf_token"}
    assert mock_session_manager.track_login_attempt.call_count == 2
    mock_session_manager.track_login_attempt.assert_any_call(ip_address="192.168.1.1", username="testuser", success=False)
    mock_session_manager.track_login_attempt.assert_any_call(ip_address="192.168.1.1", username="testuser", success=True)

    mock_session_manager.reset_mock()
    mock_auth_user.reset_mock()

    mock_session_manager.track_login_attempt = AsyncMock(return_value=(False, 0))

    with patch("src.infrastructure.auth.routes.authenticate_user", mock_auth_user):
        with pytest.raises(Exception) as excinfo:
            await login(
                request=mock_request,
                response=mock_response,
                form_data=mock_form_data,
                db=mock_db,
                session_manager=mock_session_manager,
            )

    assert "Too many failed login attempts" in str(excinfo.value)
    assert mock_auth_user.call_count == 0


@pytest.mark.asyncio
async def test_cleanup_rate_limits():
    """Test cleanup_rate_limits method."""
    mock_rate_limiter = MagicMock()
    mock_rate_limiter.delete_pattern = AsyncMock(return_value=5)

    mock_storage = MagicMock()

    session_manager = SessionManager(session_storage=mock_storage, rate_limiter=mock_rate_limiter)

    await session_manager.cleanup_rate_limits()

    mock_rate_limiter.delete_pattern.assert_called_once_with("login:*")
