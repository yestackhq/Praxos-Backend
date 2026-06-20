"""Helper functions for auth API tests."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from src.infrastructure.auth.oauth.provider import AbstractOAuthProvider
from src.infrastructure.auth.oauth.schemas import OAuthState
from src.infrastructure.auth.session.manager import SessionManager
from src.infrastructure.auth.session.storage import AbstractSessionStorage


def create_mock_oauth_provider(name: str = "google") -> AsyncMock:
    """Create a mock OAuth provider for testing."""
    mock_provider = AsyncMock(spec=AbstractOAuthProvider)
    mock_provider.name = name
    mock_provider.get_authorization_url = AsyncMock(
        return_value={
            "url": f"https://accounts.{name}.com/o/oauth2/v2/auth?dummy=params",
            "state": "test-state-value",
            "code_verifier": "test-code-verifier",
        }
    )
    mock_provider.exchange_code = AsyncMock(return_value={})
    return mock_provider


def create_mock_oauth_state_storage() -> AsyncMock:
    """Create a mock OAuth state storage for testing."""
    mock_storage = AsyncMock(spec=AbstractSessionStorage)
    mock_storage.create = AsyncMock(return_value="test-state-value")
    mock_storage.delete = AsyncMock(return_value=None)
    return mock_storage


def create_mock_session_manager() -> AsyncMock:
    """Create a mock session manager for testing."""
    mock_manager = AsyncMock(spec=SessionManager)
    mock_manager.create_session = AsyncMock(return_value=("session-id", "csrf-token"))
    mock_manager.set_session_cookies = MagicMock()
    return mock_manager


def create_oauth_state(provider: str = "google", state: str = "test-state-value") -> OAuthState:
    """Create an OAuth state object for testing."""
    return OAuthState(
        state=state,
        provider=provider,
        redirect_to="/",
        code_verifier="test-code-verifier",
    )


def create_mock_session_data(user_id: int = 1):
    """Create mock session data for testing."""
    mock_session = MagicMock()
    mock_session.user_id = user_id
    mock_session.created_at = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
    mock_session.last_activity = datetime(2023, 1, 1, 1, 0, 0, tzinfo=UTC)
    return mock_session


def create_mock_user_data(
    user_id: int = 1, username: str = "testuser", email: str = "test@example.com", provider: str = "google"
) -> dict:
    """Create mock user data for testing."""
    return {
        "id": user_id,
        "username": username,
        "email": email,
        "oauth_provider": provider,
    }
