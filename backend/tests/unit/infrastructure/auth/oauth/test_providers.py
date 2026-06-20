"""Tests for the specific OAuth provider implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.auth.oauth.providers.github import GitHubOAuthProvider
from src.infrastructure.auth.oauth.providers.google import GoogleOAuthProvider
from src.infrastructure.auth.oauth.schemas import OAuthUserInfo


@pytest.fixture
def google_provider():
    """Create a Google OAuth provider for testing."""
    return GoogleOAuthProvider(
        client_id="google-client-id",
        client_secret="google-client-secret",
        redirect_uri="https://example.com/oauth/callback/google",
    )


@pytest.fixture
def github_provider():
    """Create a GitHub OAuth provider for testing."""
    return GitHubOAuthProvider(
        client_id="github-client-id",
        client_secret="github-client-secret",
        redirect_uri="https://example.com/oauth/callback/github",
    )


def test_google_provider_initialization(google_provider):
    """Test Google provider initialization with default scopes."""
    assert google_provider.client_id == "google-client-id"
    assert google_provider.client_secret == "google-client-secret"
    assert google_provider.redirect_uri == "https://example.com/oauth/callback/google"
    assert "openid" in google_provider.scopes
    assert "https://www.googleapis.com/auth/userinfo.email" in google_provider.scopes
    assert "https://www.googleapis.com/auth/userinfo.profile" in google_provider.scopes
    assert google_provider.authorize_endpoint == "https://accounts.google.com/o/oauth2/v2/auth"
    assert google_provider.token_endpoint == "https://oauth2.googleapis.com/token"
    assert google_provider.userinfo_endpoint == "https://www.googleapis.com/oauth2/v3/userinfo"
    assert google_provider.name == "google"


def test_github_provider_initialization(github_provider):
    """Test GitHub provider initialization with default scopes."""
    assert github_provider.client_id == "github-client-id"
    assert github_provider.client_secret == "github-client-secret"
    assert github_provider.redirect_uri == "https://example.com/oauth/callback/github"
    assert "read:user" in github_provider.scopes
    assert "user:email" in github_provider.scopes
    assert github_provider.authorize_endpoint == "https://github.com/login/oauth/authorize"
    assert github_provider.token_endpoint == "https://github.com/login/oauth/access_token"
    assert github_provider.userinfo_endpoint == "https://api.github.com/user"
    assert github_provider.name == "github"


@pytest.mark.asyncio
async def test_google_authorization_url(google_provider):
    """Test Google-specific authorization URL generation."""
    auth_data = await google_provider.get_authorization_url()

    assert "url" in auth_data
    assert "state" in auth_data
    assert "code_verifier" in auth_data
    assert "access_type=offline" in auth_data["url"]
    assert "prompt=consent" in auth_data["url"]


@pytest.mark.asyncio
async def test_google_process_user_info(google_provider):
    """Test Google-specific user info processing."""
    google_user_data = {
        "sub": "123456789",
        "email": "user@example.com",
        "email_verified": True,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "picture": "https://example.com/photo.jpg",
    }

    result = await google_provider.process_user_info(google_user_data)

    assert isinstance(result, OAuthUserInfo)
    assert result.provider == "google"
    assert result.provider_user_id == "123456789"
    assert result.email == "user@example.com"
    assert result.email_verified is True
    assert result.name == "Test User"
    assert result.given_name == "Test"
    assert result.family_name == "User"
    assert result.username is None
    assert result.picture == "https://example.com/photo.jpg"
    assert result.raw_data == google_user_data


@pytest.mark.asyncio
async def test_github_exchange_code(github_provider):
    """Test GitHub-specific code exchange with Accept header."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "access_token": "github-token",
        "token_type": "bearer",
        "scope": "read:user,user:email",
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_context):
        result = await github_provider.exchange_code("github-code")

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Accept"] == "application/json"

        assert result["access_token"] == "github-token"
        assert result["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_github_get_user_info_with_emails(github_provider):
    """Test GitHub user info retrieval with separate emails endpoint call."""
    profile_data = {
        "id": 12345,
        "login": "testuser",
        "name": "Test User",
        "email": None,
        "avatar_url": "https://github.com/avatars/user.jpg",
    }

    emails_data = [
        {"email": "private@example.com", "primary": True, "verified": True},
        {"email": "public@example.com", "primary": False, "verified": True},
    ]

    mock_profile_response = MagicMock()
    mock_profile_response.raise_for_status = MagicMock()
    mock_profile_response.json.return_value = profile_data

    mock_emails_response = MagicMock()
    mock_emails_response.status_code = 200
    mock_emails_response.json.return_value = emails_data

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_profile_response, mock_emails_response]

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_context):
        result = await github_provider.get_user_info("github-token")

        assert mock_client.get.call_count == 2

        assert mock_client.get.call_args_list[0][0][0] == "https://api.github.com/user"

        assert mock_client.get.call_args_list[1][0][0] == "https://api.github.com/user/emails"

        assert result["id"] == 12345
        assert result["login"] == "testuser"
        assert "emails" in result
        assert result["emails"] == emails_data


@pytest.mark.asyncio
async def test_github_process_user_info_with_primary_email(github_provider):
    """Test GitHub-specific user info processing with primary email from emails array."""
    github_user_data = {
        "id": 12345,
        "login": "testuser",
        "name": "Test User",
        "email": "public@example.com",
        "avatar_url": "https://github.com/avatars/user.jpg",
        "emails": [
            {"email": "private@example.com", "primary": True, "verified": True},
            {"email": "public@example.com", "primary": False, "verified": False},
        ],
    }

    result = await github_provider.process_user_info(github_user_data)

    assert isinstance(result, OAuthUserInfo)
    assert result.provider == "github"
    assert result.provider_user_id == "12345"
    assert result.email == "private@example.com"
    assert result.email_verified is True
    assert result.name == "Test User"
    assert result.username == "testuser"
    assert result.picture == "https://github.com/avatars/user.jpg"
    assert result.raw_data == github_user_data


@pytest.mark.asyncio
async def test_google_create_class_method():
    """Test Google provider's create class method."""
    provider = GoogleOAuthProvider.create(
        client_id="test-id",
        client_secret="test-secret",
        redirect_uri="https://example.com/callback",
    )

    assert isinstance(provider, GoogleOAuthProvider)
    assert provider.client_id == "test-id"
    assert provider.client_secret == "test-secret"
    assert provider.redirect_uri == "https://example.com/callback"
    assert provider.name == "google"
    assert "openid" in provider.scopes


@pytest.mark.asyncio
async def test_github_create_class_method():
    """Test GitHub provider's create class method."""
    provider = GitHubOAuthProvider.create(
        client_id="test-id",
        client_secret="test-secret",
        redirect_uri="https://example.com/callback",
    )

    assert isinstance(provider, GitHubOAuthProvider)
    assert provider.client_id == "test-id"
    assert provider.client_secret == "test-secret"
    assert provider.redirect_uri == "https://example.com/callback"
    assert provider.name == "github"
    assert "read:user" in provider.scopes
