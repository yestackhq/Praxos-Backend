"""Tests for the AbstractOAuthProvider class."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.auth.oauth.provider import AbstractOAuthProvider
from src.infrastructure.auth.oauth.schemas import OAuthUserInfo


class ConcreteOAuthProvider(AbstractOAuthProvider):
    """Concrete implementation of AbstractOAuthProvider for testing."""

    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """Process user info from the provider."""
        return OAuthUserInfo(
            provider="test",
            provider_user_id=str(user_info.get("id", "")),
            email=user_info.get("email"),
            email_verified=user_info.get("email_verified", False),
            name=user_info.get("name"),
            given_name=user_info.get("given_name"),
            family_name=user_info.get("family_name"),
            username=user_info.get("username"),
            picture=user_info.get("picture"),
            raw_data=user_info,
        )


@pytest.fixture
def oauth_provider():
    """Create a test provider instance."""
    return ConcreteOAuthProvider(
        client_id="test-client-id",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
        scopes=["openid", "email", "profile"],
        authorize_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        userinfo_endpoint="https://auth.example.com/userinfo",
        provider_name="test",
    )


def test_provider_initialization(oauth_provider):
    """Test provider initialization with correct attributes."""
    assert oauth_provider.client_id == "test-client-id"
    assert oauth_provider.client_secret == "test-client-secret"
    assert oauth_provider.redirect_uri == "https://example.com/callback"
    assert oauth_provider.scopes == ["openid", "email", "profile"]
    assert oauth_provider.authorize_endpoint == "https://auth.example.com/authorize"
    assert oauth_provider.token_endpoint == "https://auth.example.com/token"
    assert oauth_provider.userinfo_endpoint == "https://auth.example.com/userinfo"
    assert oauth_provider.name == "test"


def test_generate_state(oauth_provider):
    """Test that generate_state returns a random string of expected length."""
    state = oauth_provider.generate_state()
    assert isinstance(state, str)
    assert len(state) > 32


def test_generate_pkce_codes(oauth_provider):
    """Test that PKCE code generation returns both verifier and challenge."""
    pkce_codes = oauth_provider.generate_pkce_codes()
    assert "code_verifier" in pkce_codes
    assert "code_challenge" in pkce_codes
    assert isinstance(pkce_codes["code_verifier"], str)
    assert isinstance(pkce_codes["code_challenge"], str)
    assert len(pkce_codes["code_verifier"]) >= 43
    assert len(pkce_codes["code_challenge"]) >= 43


@pytest.mark.asyncio
async def test_get_authorization_url_with_pkce(oauth_provider):
    """Test authorization URL generation with PKCE."""
    state = "test-state-value"
    result = await oauth_provider.get_authorization_url(state=state, pkce=True)

    assert "url" in result
    assert "state" in result
    assert "code_verifier" in result
    assert result["state"] == state
    assert "client_id=test-client-id" in result["url"]
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in result["url"]
    assert "code_challenge" in result["url"]
    assert "code_challenge_method=S256" in result["url"]


@pytest.mark.asyncio
async def test_get_authorization_url_without_pkce(oauth_provider):
    """Test authorization URL generation without PKCE."""
    state = "test-state-value"
    result = await oauth_provider.get_authorization_url(state=state, pkce=False)

    assert "url" in result
    assert "state" in result
    assert "code_verifier" not in result
    assert result["state"] == state
    assert "client_id=test-client-id" in result["url"]
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in result["url"]
    assert "code_challenge" not in result["url"]
    assert "code_challenge_method=S256" not in result["url"]


@pytest.mark.asyncio
async def test_get_authorization_url_with_extra_params(oauth_provider):
    """Test authorization URL generation with extra parameters."""
    extra_params = {"prompt": "consent", "access_type": "offline"}
    result = await oauth_provider.get_authorization_url(extra_params=extra_params)

    assert "url" in result
    assert "prompt=consent" in result["url"]
    assert "access_type=offline" in result["url"]


@pytest.mark.asyncio
async def test_exchange_code_successful(oauth_provider):
    """Test successful code exchange for access token."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "access_token": "test-access-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "test-refresh-token",
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_context):
        result = await oauth_provider.exchange_code("test-code", "test-verifier")

        mock_client.post.assert_called_once()

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://auth.example.com/token"

        post_data = call_args[1]["data"]
        assert post_data["client_id"] == "test-client-id"
        assert post_data["client_secret"] == "test-client-secret"
        assert post_data["code"] == "test-code"
        assert post_data["redirect_uri"] == "https://example.com/callback"
        assert post_data["code_verifier"] == "test-verifier"

        assert result["access_token"] == "test-access-token"
        assert result["refresh_token"] == "test-refresh-token"


@pytest.mark.asyncio
async def test_get_user_info_successful(oauth_provider):
    """Test successful user info retrieval."""
    test_user_info = {"id": "12345", "email": "test@example.com", "name": "Test User", "picture": "https://example.com/pic.jpg"}

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = test_user_info

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_context):
        result = await oauth_provider.get_user_info("test-access-token")

        mock_client.get.assert_called_once()

        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://auth.example.com/userinfo"

        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-access-token"

        assert result == test_user_info


@pytest.mark.asyncio
async def test_process_user_info(oauth_provider):
    """Test processing of user info into standardized format."""
    test_user_info = {
        "id": "12345",
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "username": "testuser",
        "picture": "https://example.com/pic.jpg",
    }

    result = await oauth_provider.process_user_info(test_user_info)

    assert isinstance(result, OAuthUserInfo)
    assert result.provider == "test"
    assert result.provider_user_id == "12345"
    assert result.email == "test@example.com"
    assert result.email_verified is True
    assert result.name == "Test User"
    assert result.given_name == "Test"
    assert result.family_name == "User"
    assert result.username == "testuser"
    assert result.picture == "https://example.com/pic.jpg"
    assert result.raw_data == test_user_info


@pytest.mark.asyncio
async def test_validate_token_valid(oauth_provider):
    """Test token validation with valid token."""
    with patch.object(oauth_provider, "get_user_info", new_callable=AsyncMock) as mock_get_user_info:
        mock_get_user_info.return_value = {"id": "12345"}

        result = await oauth_provider.validate_token("valid-token")

        assert result is True
        mock_get_user_info.assert_called_once_with("valid-token")


@pytest.mark.asyncio
async def test_validate_token_invalid(oauth_provider):
    """Test token validation with invalid token."""
    with patch.object(oauth_provider, "get_user_info", new_callable=AsyncMock) as mock_get_user_info:
        mock_get_user_info.side_effect = Exception("Invalid token")

        result = await oauth_provider.validate_token("invalid-token")

        assert result is False
        mock_get_user_info.assert_called_once_with("invalid-token")
