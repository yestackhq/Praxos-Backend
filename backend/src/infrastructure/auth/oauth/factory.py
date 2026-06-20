from typing import cast

from .provider import AbstractOAuthProvider


class OAuthProviderFactory:
    """Factory class for creating OAuth provider instances."""

    _providers: dict[str, type[AbstractOAuthProvider]] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type[AbstractOAuthProvider]) -> None:
        """
        Register an OAuth provider class.

        Args:
            provider_name: Name identifier for the provider
            provider_class: The provider class to register
        """
        cls._providers[provider_name] = provider_class

    @classmethod
    def get_provider_class(cls, provider_name: str) -> type[AbstractOAuthProvider] | None:
        """
        Get an OAuth provider class by name.

        Args:
            provider_name: Name identifier for the provider

        Returns:
            The provider class if registered, None otherwise
        """
        return cls._providers.get(provider_name)

    @classmethod
    def create_provider(
        cls, provider_name: str, client_id: str, client_secret: str, redirect_uri: str
    ) -> AbstractOAuthProvider:
        """
        Create an instance of the requested provider with the given credentials.

        Args:
            provider_name: Name of the provider to create
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: Callback URL for OAuth flow

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider not registered
        """
        provider_class = cls.get_provider_class(provider_name)
        if not provider_class:
            raise ValueError(f"OAuth provider {provider_name} not registered")

        if hasattr(provider_class, "create"):
            return cast(
                AbstractOAuthProvider,
                provider_class.create(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri),
            )

        return provider_class(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=[],
            authorize_endpoint="",
            token_endpoint="",
            userinfo_endpoint="",
            provider_name=provider_name,
        )
