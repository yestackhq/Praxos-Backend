from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from src.infrastructure.auth.session.dependencies import (
    get_current_session_data,
    get_current_user,
    get_session_from_cookie,
    get_session_manager,
    verify_csrf_token,
)
from src.infrastructure.auth.session.manager import SessionManager
from src.infrastructure.auth.session.schemas import SessionData
from src.infrastructure.database.session import async_session
from src.interfaces.main import app


@pytest.mark.asyncio
async def test_login_endpoint_success(client, test_user, db_session, monkeypatch):
    """Test successful login with session authentication."""
    mock_manager = MagicMock(spec=SessionManager)
    mock_manager.create_session = AsyncMock(return_value=("test-session-id", "test-csrf-token"))
    mock_manager.set_session_cookies = MagicMock()
    mock_manager.track_login_attempt = AsyncMock(return_value=(True, 5))

    app.dependency_overrides[get_session_manager] = lambda: mock_manager

    try:
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": "Password123!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert data["csrf_token"] == "test-csrf-token"

        mock_manager.create_session.assert_called_once()
        mock_manager.set_session_cookies.assert_called_once()

        create_args = mock_manager.create_session.call_args
        assert create_args[1]["user_id"] == test_user["id"]
    finally:
        if get_session_manager in app.dependency_overrides:
            del app.dependency_overrides[get_session_manager]


@pytest.mark.asyncio
async def test_login_endpoint_invalid_credentials(client, test_user, db_session, monkeypatch):
    """Test login with invalid credentials."""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user["username"],
            "password": "WrongPassword!",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert "Incorrect username or password" in data["detail"]


@pytest.mark.asyncio
async def test_logout_endpoint(client, test_user, db_session, monkeypatch):
    """Test logout endpoint."""
    session_id = "test-session-id"
    session_data = SessionData(
        session_id=session_id,
        user_id=test_user["id"],
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test-agent",
        device_info={},
        last_activity=datetime.now(UTC),
        metadata={},
    )

    mock_manager = MagicMock(spec=SessionManager)
    mock_manager.terminate_session = AsyncMock(return_value=True)
    mock_manager.clear_session_cookies = MagicMock()

    async def mock_get_current_session_data(request: Request, session_id=None, session_manager=None):
        return session_data

    app.dependency_overrides[get_session_manager] = lambda: mock_manager
    app.dependency_overrides[get_current_session_data] = mock_get_current_session_data

    try:
        response = await client.post("/api/v1/auth/logout")

        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Logged out successfully"

        mock_manager.terminate_session.assert_called_once_with(session_id)
        mock_manager.clear_session_cookies.assert_called_once()
    finally:
        if get_session_manager in app.dependency_overrides:
            del app.dependency_overrides[get_session_manager]
        if get_current_session_data in app.dependency_overrides:
            del app.dependency_overrides[get_current_session_data]


@pytest.mark.asyncio
async def test_refresh_csrf_token_endpoint(client, test_user, db_session, monkeypatch):
    """Test refreshing the CSRF token."""
    session_id = "test-session-id"
    session_data = SessionData(
        session_id=session_id,
        user_id=test_user["id"],
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test-agent",
        device_info={},
        last_activity=datetime.now(UTC),
        metadata={},
    )

    mock_manager = MagicMock(spec=SessionManager)
    mock_manager.regenerate_csrf_token = AsyncMock(return_value="new-csrf-token")
    mock_manager.session_timeout = timedelta(minutes=30)  # Add session_timeout attribute

    async def mock_get_current_session_data(request: Request, session_id=None, session_manager=None):
        return session_data

    app.dependency_overrides[get_session_manager] = lambda: mock_manager
    app.dependency_overrides[get_current_session_data] = mock_get_current_session_data

    try:
        response = await client.post("/api/v1/auth/refresh-csrf")

        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert data["csrf_token"] == "new-csrf-token"

        mock_manager.regenerate_csrf_token.assert_called_once_with(
            user_id=test_user["id"],
            session_id=session_id,
        )
    finally:
        if get_session_manager in app.dependency_overrides:
            del app.dependency_overrides[get_session_manager]
        if get_current_session_data in app.dependency_overrides:
            del app.dependency_overrides[get_current_session_data]


@pytest.mark.asyncio
async def test_unauthorized_access(client, test_user, db_session):
    """Test accessing protected endpoint without authentication."""
    response = await client.get("/api/v1/users/me")

    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert "Not authenticated" in data["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_access(client, test_user, db_session, monkeypatch):
    """Test accessing protected endpoint with valid session."""
    test_user_with_image = test_user.copy()
    test_user_with_image["profile_image_url"] = "https://example.com/default-avatar.png"

    app.dependency_overrides[get_current_user] = lambda: test_user_with_image
    app.dependency_overrides[async_session] = lambda: db_session
    app.dependency_overrides[get_session_from_cookie] = lambda request: None
    app.dependency_overrides[verify_csrf_token] = lambda request, session_data=None: None

    try:
        response = await client.get("/api/v1/users/me")

        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        assert response.status_code == 200
        user_data = response.json()
        assert user_data["id"] == test_user["id"]
        assert user_data["username"] == test_user["username"]
        assert user_data["profile_image_url"] == test_user_with_image["profile_image_url"]
    finally:
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]
        if async_session in app.dependency_overrides:
            del app.dependency_overrides[async_session]
        if get_session_from_cookie in app.dependency_overrides:
            del app.dependency_overrides[get_session_from_cookie]
        if verify_csrf_token in app.dependency_overrides:
            del app.dependency_overrides[verify_csrf_token]


@pytest.mark.asyncio
async def test_csrf_protection(client, test_user, db_session):
    """Test CSRF protection for mutation operations."""
    session_data = SessionData(
        session_id="test-session-id",
        user_id=test_user["id"],
        is_active=True,
        ip_address="127.0.0.1",
        user_agent="test-agent",
        device_info={},
        last_activity=datetime.now(UTC),
        metadata={},
    )

    with (
        patch("src.infrastructure.auth.session.dependencies.get_session_from_cookie", return_value=session_data),
        patch(
            "src.infrastructure.auth.session.dependencies.verify_csrf_token", side_effect=Exception("CSRF validation failed")
        ),
    ):
        response = await client.patch(
            f"/api/v1/users/{test_user['username']}",
            json={"name": "Updated Name"},
        )

        assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_login_endpoint_with_rate_limiting(client, test_user, db_session, monkeypatch):
    """Test that the login endpoint correctly applies rate limiting."""
    mock_manager = MagicMock(spec=SessionManager)

    mock_manager.track_login_attempt = AsyncMock(return_value=(True, 4))
    mock_manager.create_session = AsyncMock(return_value=("test-session-id", "test-csrf-token"))
    mock_manager.set_session_cookies = MagicMock()

    app.dependency_overrides[get_session_manager] = lambda: mock_manager

    try:
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": "Password123!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert data["csrf_token"] == "test-csrf-token"

        mock_manager.track_login_attempt.assert_any_call(ip_address="127.0.0.1", username=test_user["username"], success=False)
        mock_manager.track_login_attempt.assert_any_call(ip_address="127.0.0.1", username=test_user["username"], success=True)

        mock_manager.track_login_attempt.reset_mock()
        mock_manager.track_login_attempt = AsyncMock(return_value=(False, 0))

        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user["username"],
                "password": "Password123!",
            },
        )

        assert response.status_code == 401

        mock_manager.track_login_attempt.assert_called_once_with(
            ip_address="127.0.0.1", username=test_user["username"], success=False
        )

        assert mock_manager.create_session.call_count == 1

    finally:
        if get_session_manager in app.dependency_overrides:
            del app.dependency_overrides[get_session_manager]
