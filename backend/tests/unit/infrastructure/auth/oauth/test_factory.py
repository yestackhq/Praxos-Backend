"""Tests for the OAuthProviderFactory class."""

from typing import Any

import pytest

from src.infrastructure.auth.oauth.factory import OAuthProviderFactory
from src.infrastructure.auth.oauth.provider import AbstractOAuthProvider
from src.infrastructure.auth.oauth.schemas import OAuthUserInfo


class MockOAuthProvider(AbstractOAuthProvider):
    """Mock OAuth provider for testing factory patterns."""

    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """Process user info from the provider."""
        return OAuthUserInfo(
            provider="mock",
            provider_user_id=str(user_info.get("id", "")),
            email=user_info.get("email"),
            email_verified=False,
            name=user_info.get("name"),
            given_name=None,
            family_name=None,
            username=None,
            picture=None,
            raw_data=user_info,
        )

    @classmethod
    def create(cls, client_id: str, client_secret: str, redirect_uri: str) -> "MockOAuthProvider":
        """Factory method to create a new instance."""
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=["email", "profile"],
            authorize_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            userinfo_endpoint="https://example.com/userinfo",
            provider_name="mock",
        )


class CreatelessMockProvider(AbstractOAuthProvider):
    """Mock provider without a create method to test fallback instantiation."""

    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """Process user info from the provider."""
        return OAuthUserInfo(
            provider="createless",
            provider_user_id=str(user_info.get("id", "")),
            email=user_info.get("email"),
            email_verified=False,
            name=user_info.get("name"),
            given_name=None,
            family_name=None,
            username=None,
            picture=None,
            raw_data=user_info,
        )


@pytest.fixture
def setup_factory():
    """Setup the factory with test providers and clean up after tests."""
    original_providers = OAuthProviderFactory._providers.copy()

    OAuthProviderFactory.register_provider("mock", MockOAuthProvider)
    OAuthProviderFactory.register_provider("createless", CreatelessMockProvider)

    yield

    OAuthProviderFactory._providers = original_providers


def test_register_provider():
    """Test registering a provider class."""
    original_providers = OAuthProviderFactory._providers.copy()

    OAuthProviderFactory.register_provider("test", MockOAuthProvider)

    assert "test" in OAuthProviderFactory._providers
    assert OAuthProviderFactory._providers["test"] == MockOAuthProvider

    OAuthProviderFactory._providers = original_providers


def test_get_provider_class_registered(setup_factory):
    """Test getting a registered provider class."""
    provider_class = OAuthProviderFactory.get_provider_class("mock")
    assert provider_class == MockOAuthProvider


def test_get_provider_class_not_registered(setup_factory):
    """Test getting a non-registered provider class."""
    provider_class = OAuthProviderFactory.get_provider_class("nonexistent")
    assert provider_class is None


def test_create_provider_with_create_method(setup_factory):
    """Test creating a provider that has a create class method."""
    provider = OAuthProviderFactory.create_provider(
        provider_name="mock",
        client_id="test-id",
        client_secret="test-secret",
        redirect_uri="https://test.com/callback",
    )

    assert isinstance(provider, MockOAuthProvider)
    assert provider.client_id == "test-id"
    assert provider.client_secret == "test-secret"
    assert provider.redirect_uri == "https://test.com/callback"
    assert provider.name == "mock"


def test_create_provider_without_create_method(setup_factory):
    """Test creating a provider without a create class method."""
    provider = OAuthProviderFactory.create_provider(
        provider_name="createless",
        client_id="test-id",
        client_secret="test-secret",
        redirect_uri="https://test.com/callback",
    )

    assert isinstance(provider, CreatelessMockProvider)
    assert provider.client_id == "test-id"
    assert provider.client_secret == "test-secret"
    assert provider.redirect_uri == "https://test.com/callback"
    assert provider.name == "createless"


def test_create_provider_not_registered():
    """Test creating a provider that is not registered."""
    with pytest.raises(ValueError):
        OAuthProviderFactory.create_provider(
            provider_name="nonexistent",
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="https://test.com/callback",
        )
