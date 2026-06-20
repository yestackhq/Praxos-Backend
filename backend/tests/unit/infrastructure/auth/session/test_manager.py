from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response

from src.infrastructure.auth.session.manager import SessionManager
from src.infrastructure.auth.session.schemas import CSRFToken, SessionData, UserAgentInfo


@pytest.fixture
def mock_storage():
    """Create a mock session storage."""
    storage = AsyncMock()
    storage.create = AsyncMock()
    storage.get = AsyncMock()
    storage.update = AsyncMock()
    storage.delete = AsyncMock()
    storage.extend = AsyncMock()
    storage.exists = AsyncMock()
    storage.get_user_sessions = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_csrf_storage():
    """Create a mock CSRF token storage."""
    storage = AsyncMock()
    storage.create = AsyncMock()
    storage.get = AsyncMock()
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def session_manager(mock_storage, mock_csrf_storage):
    """Create a session manager with mock storage."""
    with (
        patch("src.infrastructure.auth.session.manager.get_session_storage", return_value=mock_storage),
        patch("src.infrastructure.auth.session.storage.get_session_storage", return_value=mock_csrf_storage),
    ):
        manager = SessionManager(session_storage=mock_storage)
        manager.csrf_storage = mock_csrf_storage
        return manager


@pytest.fixture
def mock_request():
    """Create a mock request for session creation."""
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36",
        "x-forwarded-for": "192.168.1.1",
    }
    return request


@pytest.fixture
def mock_response():
    """Create a mock response for cookie handling."""
    response = MagicMock(spec=Response)
    response.set_cookie = MagicMock()
    response.delete_cookie = MagicMock()
    return response


@pytest.mark.asyncio
async def test_create_session(session_manager, mock_storage, mock_csrf_storage, mock_request):
    """Test creating a new session with CSRF token."""
    user_id = 1
    session_id = "test-session-id"
    csrf_token = "test-csrf-token"

    mock_storage.create.return_value = session_id

    with patch.object(session_manager, "_generate_csrf_token", return_value=csrf_token) as mock_gen:
        result_session_id, result_csrf_token = await session_manager.create_session(
            request=mock_request,
            user_id=user_id,
        )

        assert result_session_id == session_id
        assert result_csrf_token == csrf_token

        mock_storage.create.assert_called_once()
        mock_gen.assert_called_once()

        create_args = mock_storage.create.call_args[0][0]
        assert create_args.user_id == user_id
        assert create_args.ip_address == "192.168.1.1"


@pytest.mark.asyncio
async def test_validate_session_valid(session_manager, mock_storage):
    """Test validating a valid, non-expired session."""
    session_id = "test-session-id"

    current_time = datetime.now(UTC)
    session_data = SessionData(
        session_id=session_id,
        user_id=1,
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test_agent",
        device_info={},
        last_activity=current_time - timedelta(minutes=5),
        metadata={},
    )

    mock_storage.get.return_value = session_data
    mock_storage.update.return_value = True

    result = await session_manager.validate_session(session_id)

    assert result is not None
    assert result.session_id == session_id
    assert result.user_id == 1

    mock_storage.get.assert_called_once_with(session_id, SessionData)
    mock_storage.update.assert_called_once()


@pytest.mark.asyncio
async def test_validate_session_expired(session_manager, mock_storage):
    """Test validating an expired session."""
    session_id = "test-session-id"

    current_time = datetime.now(UTC)
    session_data = SessionData(
        session_id=session_id,
        user_id=1,
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test_agent",
        device_info={},
        last_activity=current_time - timedelta(minutes=45),
        metadata={},
    )

    mock_storage.get.return_value = session_data

    result = await session_manager.validate_session(session_id)

    assert result is None

    assert mock_storage.get.call_count >= 1
    assert mock_storage.get.call_args_list[0][0][0] == session_id
    assert mock_storage.get.call_args_list[0][0][1] == SessionData

    assert mock_storage.update.call_count == 1
    update_args = mock_storage.update.call_args[0]
    assert update_args[0] == session_id
    assert update_args[1].is_active is False
    assert "terminated_at" in update_args[1].metadata


@pytest.mark.asyncio
async def test_validate_session_inactive(session_manager, mock_storage):
    """Test validating an inactive session."""
    session_id = "test-session-id"

    current_time = datetime.now(UTC)
    session_data = SessionData(
        session_id=session_id,
        user_id=1,
        is_active=False,
        ip_address="127.0.0.1",
        user_agent="test_agent",
        device_info={},
        last_activity=current_time - timedelta(minutes=5),
        metadata={},
    )

    mock_storage.get.return_value = session_data

    result = await session_manager.validate_session(session_id)

    assert result is None

    mock_storage.get.assert_called_once_with(session_id, SessionData)
    mock_storage.update.assert_not_called()


@pytest.mark.asyncio
async def test_validate_csrf_token_valid(session_manager, mock_csrf_storage):
    """Test validating a valid CSRF token."""
    session_id = "test-session-id"
    csrf_token = "test-csrf-token"

    current_time = datetime.now(UTC)
    token_data = CSRFToken(
        token=csrf_token,
        user_id=1,
        session_id=session_id,
        expiry=current_time + timedelta(minutes=30),
    )

    mock_csrf_storage.get.return_value = token_data

    result = await session_manager.validate_csrf_token(session_id, csrf_token)

    assert result is True

    mock_csrf_storage.get.assert_called_once_with(csrf_token, CSRFToken)


@pytest.mark.asyncio
async def test_validate_csrf_token_expired(session_manager, mock_csrf_storage):
    """Test validating an expired CSRF token."""
    session_id = "test-session-id"
    csrf_token = "test-csrf-token"

    current_time = datetime.now(UTC)
    token_data = CSRFToken(
        token=csrf_token,
        user_id=1,
        session_id=session_id,
        expiry=current_time - timedelta(minutes=5),
    )

    mock_csrf_storage.get.return_value = token_data

    result = await session_manager.validate_csrf_token(session_id, csrf_token)

    assert result is False

    mock_csrf_storage.get.assert_called_once_with(csrf_token, CSRFToken)
    mock_csrf_storage.delete.assert_called_once_with(csrf_token)


@pytest.mark.asyncio
async def test_validate_csrf_token_mismatched_session(session_manager, mock_csrf_storage):
    """Test validating a CSRF token with wrong session ID."""
    session_id = "test-session-id"
    wrong_session_id = "wrong-session-id"
    csrf_token = "test-csrf-token"

    current_time = datetime.now(UTC)
    token_data = CSRFToken(
        token=csrf_token,
        user_id=1,
        session_id=wrong_session_id,
        expiry=current_time + timedelta(minutes=30),
    )

    mock_csrf_storage.get.return_value = token_data

    result = await session_manager.validate_csrf_token(session_id, csrf_token)

    assert result is False

    mock_csrf_storage.get.assert_called_once_with(csrf_token, CSRFToken)


@pytest.mark.asyncio
async def test_regenerate_csrf_token(session_manager, mock_csrf_storage):
    """Test regenerating a CSRF token for an existing session."""
    user_id = 1
    session_id = "test-session-id"
    new_csrf_token = "new-csrf-token"

    with patch.object(session_manager, "_generate_csrf_token", return_value=new_csrf_token) as mock_generate:
        result = await session_manager.regenerate_csrf_token(user_id, session_id)

        assert result == new_csrf_token

        mock_generate.assert_called_once_with(user_id, session_id)


@pytest.mark.asyncio
async def test_terminate_session(session_manager, mock_storage):
    """Test terminating a session."""
    session_id = "test-session-id"

    session_data = SessionData(
        session_id=session_id,
        user_id=1,
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test_agent",
        device_info={},
        last_activity=datetime.now(UTC),
        metadata={},
    )

    mock_storage.get.return_value = session_data
    mock_storage.update.return_value = True

    result = await session_manager.terminate_session(session_id)

    assert result is True

    mock_storage.get.assert_called_once_with(session_id, SessionData)
    mock_storage.update.assert_called_once()

    update_args = mock_storage.update.call_args[0]
    assert update_args[0] == session_id
    assert not update_args[1].is_active
    assert "terminated_at" in update_args[1].metadata
    assert "termination_reason" in update_args[1].metadata


@pytest.mark.asyncio
async def test_set_session_cookies(session_manager, mock_response):
    """Test setting session cookies."""
    session_id = "test-session-id"
    csrf_token = "test-csrf-token"

    session_manager.set_session_cookies(
        response=mock_response,
        session_id=session_id,
        csrf_token=csrf_token,
    )

    assert mock_response.set_cookie.call_count == 2

    session_cookie_args = mock_response.set_cookie.call_args_list[0][1]
    assert session_cookie_args["key"] == "session_id"
    assert session_cookie_args["value"] == session_id
    assert session_cookie_args["httponly"] is True

    csrf_cookie_args = mock_response.set_cookie.call_args_list[1][1]
    assert csrf_cookie_args["key"] == "csrf_token"
    assert csrf_cookie_args["value"] == csrf_token
    assert csrf_cookie_args["httponly"] is False


@pytest.mark.asyncio
async def test_clear_session_cookies(session_manager, mock_response):
    """Test clearing session cookies."""
    session_manager.clear_session_cookies(response=mock_response)

    assert mock_response.delete_cookie.call_count == 2

    assert mock_response.delete_cookie.call_args_list[0][1]["key"] == "session_id"
    assert mock_response.delete_cookie.call_args_list[1][1]["key"] == "csrf_token"


@pytest.mark.asyncio
async def test_parse_user_agent(session_manager):
    """Test parsing a user agent string."""
    ua_string = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    result = session_manager.parse_user_agent(ua_string)

    assert isinstance(result, UserAgentInfo)
    assert result.browser == "Chrome"
    assert "91.0" in result.browser_version
    assert result.os == "Windows"
    assert result.is_pc is True
    assert result.is_mobile is False

    ua_string = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    )

    result = session_manager.parse_user_agent(ua_string)

    assert result.browser == "Mobile Safari"
    assert result.os == "iOS"
    assert result.device == "iPhone"
    assert result.is_mobile is True
    assert result.is_pc is False


@pytest.mark.asyncio
async def test_enforce_session_limit(session_manager, mock_storage):
    """Test enforcing maximum sessions per user."""
    user_id = 1
    active_sessions = []

    for i in range(6):
        session_data = SessionData(
            session_id=f"session-{i}",
            user_id=user_id,
            is_active=True,
            ip_address="127.0.0.1",
            user_agent="test-agent",
            device_info={},
            last_activity=datetime.now(UTC) - timedelta(minutes=i),
            metadata={},
        )
        active_sessions.append(session_data)

    mock_storage.get_user_sessions.return_value = [s.session_id for s in active_sessions]
    mock_storage.get.side_effect = lambda sid, cls: next((s for s in active_sessions if s.session_id == sid), None)

    await session_manager._enforce_session_limit(user_id)

    assert mock_storage.update.call_count == 2

    terminated_sessions = [args[0][0] for args in mock_storage.update.call_args_list]
    assert "session-5" in terminated_sessions
    assert "session-4" in terminated_sessions
