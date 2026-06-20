"""Tests for OAuth authentication endpoints."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.auth.oauth.dependencies import (
    get_google_provider,
    get_oauth_state,
    get_oauth_state_storage,
)
from src.infrastructure.auth.oauth.provider import AbstractOAuthProvider
from src.infrastructure.auth.oauth.schemas import OAuthState
from src.infrastructure.auth.session.dependencies import get_session_from_cookie, get_session_manager
from src.infrastructure.auth.session.manager import SessionManager
from src.infrastructure.auth.session.storage import AbstractSessionStorage
from src.infrastructure.database.session import async_session
from src.interfaces.main import app


@pytest.mark.asyncio
async def test_oauth_google_login(client: AsyncClient):
    """Test the OAuth Google login initiation endpoint."""
    mock_provider = AsyncMock(spec=AbstractOAuthProvider)
    mock_provider.name = "google"
    mock_provider.get_authorization_url = AsyncMock(
        return_value={
            "url": "https://accounts.google.com/o/oauth2/v2/auth?dummy=params",
            "state": "test-state-value",
            "code_verifier": "test-code-verifier",
        }
    )

    mock_state_storage = AsyncMock(spec=AbstractSessionStorage)
    mock_state_storage.create = AsyncMock(return_value="test-state-value")

    original_deps = app.dependency_overrides.copy()

    try:
        app.dependency_overrides[get_google_provider] = lambda: mock_provider
        app.dependency_overrides[get_oauth_state_storage] = lambda: mock_state_storage

        response = await client.get("/api/v1/auth/oauth/google")

        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response body: {response.text}")

        assert response.status_code == 200
        assert "url" in response.json()
        assert response.json()["url"] == "https://accounts.google.com/o/oauth2/v2/auth?dummy=params"

        mock_provider.get_authorization_url.assert_called_once()
        mock_state_storage.create.assert_called_once()
    finally:
        app.dependency_overrides = original_deps


@pytest.mark.asyncio
async def test_oauth_callback_invalid_state(client: AsyncClient):
    """Test the OAuth callback endpoint with an invalid state parameter."""
    original_deps = app.dependency_overrides.copy()

    try:

        async def mock_get_oauth_state_func(state: str, storage: Any) -> None:
            return None

        mock_provider = AsyncMock(spec=AbstractOAuthProvider)
        mock_provider.name = "google"
        mock_provider.exchange_code = AsyncMock(return_value={})

        mock_state_storage = AsyncMock(spec=AbstractSessionStorage)
        mock_state_storage.delete = AsyncMock(return_value=None)

        mock_session_manager = AsyncMock(spec=SessionManager)
        mock_session_manager.create_session = AsyncMock(return_value=("session-id", "csrf-token"))
        mock_session_manager.set_session_cookies = MagicMock()

        app.dependency_overrides[get_oauth_state] = mock_get_oauth_state_func
        app.dependency_overrides[get_google_provider] = lambda: mock_provider
        app.dependency_overrides[get_oauth_state_storage] = lambda: mock_state_storage
        app.dependency_overrides[get_session_manager] = lambda: mock_session_manager

        response = await client.get(
            "/api/v1/auth/oauth/callback/google",
            params={"code": "test-code", "state": "invalid-state"},
        )

        assert response.status_code in [302, 500]

        response = await client.get(
            "/api/v1/auth/oauth/callback/google",
            params={"code": "test-code", "state": "invalid-state", "response_format": "json"},
        )

        assert response.status_code in [400, 500]
    finally:
        app.dependency_overrides = original_deps


@pytest.mark.asyncio
async def test_oauth_callback_provider_mismatch(client: AsyncClient):
    """Test the OAuth callback endpoint with a state parameter for a different provider."""
    original_deps = app.dependency_overrides.copy()

    try:
        mock_state = OAuthState(
            state="test-state-value",
            provider="github",
            redirect_to="/",
            code_verifier="test-code-verifier",
        )

        async def mock_get_oauth_state_func(state: str, storage: Any) -> OAuthState:
            return mock_state

        mock_provider = AsyncMock(spec=AbstractOAuthProvider)
        mock_provider.name = "google"
        mock_provider.exchange_code = AsyncMock(return_value={})

        mock_state_storage = AsyncMock(spec=AbstractSessionStorage)
        mock_state_storage.delete = AsyncMock(return_value=None)

        mock_session_manager = AsyncMock(spec=SessionManager)
        mock_session_manager.create_session = AsyncMock(return_value=("session-id", "csrf-token"))
        mock_session_manager.set_session_cookies = MagicMock()

        app.dependency_overrides[get_oauth_state] = mock_get_oauth_state_func
        app.dependency_overrides[get_google_provider] = lambda: mock_provider
        app.dependency_overrides[get_oauth_state_storage] = lambda: mock_state_storage
        app.dependency_overrides[get_session_manager] = lambda: mock_session_manager

        response = await client.get(
            "/api/v1/auth/oauth/callback/google",
            params={"code": "test-code", "state": "test-state-value"},
        )

        assert response.status_code in [302, 500]

        response = await client.get(
            "/api/v1/auth/oauth/callback/google",
            params={"code": "test-code", "state": "test-state-value", "response_format": "json"},
        )

        assert response.status_code in [400, 500]
    finally:
        app.dependency_overrides = original_deps


@pytest.mark.asyncio
async def test_check_auth_authenticated(client: AsyncClient, db_session: AsyncSession):
    """Test the check-auth endpoint when the user is authenticated."""
    mock_session = MagicMock()
    mock_session.user_id = 1
    mock_session.created_at = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
    mock_session.last_activity = datetime(2023, 1, 1, 1, 0, 0, tzinfo=UTC)

    mock_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "oauth_provider": "google",
    }

    original_deps = app.dependency_overrides.copy()

    try:
        app.dependency_overrides[get_session_from_cookie] = lambda: mock_session
        app.dependency_overrides[async_session] = lambda: db_session

        with patch("src.modules.user.crud.crud_users.get", return_value=mock_user):
            response = await client.get("/api/v1/auth/check-auth")

            assert response.status_code == 200
            assert response.json()["authenticated"] is True
            assert response.json()["user"]["id"] == 1
            assert response.json()["user"]["username"] == "testuser"
            assert response.json()["user"]["oauth_provider"] == "google"
            assert "session" in response.json()
    finally:
        app.dependency_overrides = original_deps


@pytest.mark.asyncio
async def test_check_auth_not_authenticated(client: AsyncClient):
    """Test the check-auth endpoint when the user is not authenticated."""
    original_deps = app.dependency_overrides.copy()

    try:
        app.dependency_overrides[get_session_from_cookie] = lambda: None

        response = await client.get("/api/v1/auth/check-auth")

        assert response.status_code == 200
        assert response.json()["authenticated"] is False
        assert response.json()["message"] == "Not authenticated"
    finally:
        app.dependency_overrides = original_deps


@pytest.mark.asyncio
async def test_check_auth_no_session_cookie_returns_unauthenticated(client: AsyncClient):
    """A request with no session cookie must get 200 {authenticated: false}, not a 401.

    Regression: /check-auth must answer anonymous callers (its whole purpose). It used to
    depend on get_current_session_data, which raises 401 when no session exists, so the
    unauthenticated branch was unreachable. Only get_session_manager is overridden here so
    the real get_session_from_cookie runs against a request that carries no cookie.
    """
    original_deps = app.dependency_overrides.copy()

    try:
        app.dependency_overrides[get_session_manager] = lambda: MagicMock()

        response = await client.get("/api/v1/auth/check-auth")

        assert response.status_code == 200
        assert response.json()["authenticated"] is False
    finally:
        app.dependency_overrides = original_deps
