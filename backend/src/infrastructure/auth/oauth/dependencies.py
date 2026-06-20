from fastapi import Depends, HTTPException, status

from ....infrastructure.config.settings import get_settings
from ....modules.user.enums import OAuthProvider
from ...logging import get_logger
from ..session.storage import AbstractSessionStorage, get_session_storage
from .factory import OAuthProviderFactory
from .provider import AbstractOAuthProvider
from .providers.github import GitHubOAuthProvider
from .providers.google import GoogleOAuthProvider
from .schemas import OAuthState

logger = get_logger()
settings = get_settings()

OAuthProviderFactory.register_provider(OAuthProvider.GOOGLE.value, GoogleOAuthProvider)
OAuthProviderFactory.register_provider(OAuthProvider.GITHUB.value, GitHubOAuthProvider)


def get_oauth_state_storage() -> AbstractSessionStorage[OAuthState]:
    """Get a storage backend for OAuth state objects."""
    return get_session_storage(
        backend=settings.SESSION_BACKEND,
        model_type=OAuthState,
        prefix="oauth_state:",
        expiration=1800,
        host=settings.CACHE_REDIS_HOST,
        port=settings.CACHE_REDIS_PORT,
        db=settings.CACHE_REDIS_DB,
        password=settings.CACHE_REDIS_PASSWORD,
    )


def get_google_provider() -> AbstractOAuthProvider:
    """
    Get the configured Google OAuth provider instance.

    Returns:
        Configured Google OAuth provider

    Raises:
        HTTPException: If provider is not configured properly
    """
    if not settings.OAUTH_GOOGLE_CLIENT_ID or not settings.OAUTH_GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google OAuth credentials not configured")

    try:
        return OAuthProviderFactory.create_provider(
            provider_name=OAuthProvider.GOOGLE.value,
            client_id=settings.OAUTH_GOOGLE_CLIENT_ID,
            client_secret=settings.OAUTH_GOOGLE_CLIENT_SECRET,
            redirect_uri=f"{settings.OAUTH_REDIRECT_BASE_URL}/api/v1/auth/oauth/callback/{OAuthProvider.GOOGLE.value}",
        )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google OAuth provider not configured")


def get_github_provider() -> AbstractOAuthProvider:
    """
    Get the configured GitHub OAuth provider instance.

    Returns:
        Configured GitHub OAuth provider

    Raises:
        HTTPException: If provider is not configured properly
    """
    if not settings.OAUTH_GITHUB_CLIENT_ID or not settings.OAUTH_GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GitHub OAuth credentials not configured")

    try:
        return OAuthProviderFactory.create_provider(
            provider_name=OAuthProvider.GITHUB.value,
            client_id=settings.OAUTH_GITHUB_CLIENT_ID,
            client_secret=settings.OAUTH_GITHUB_CLIENT_SECRET,
            redirect_uri=f"{settings.OAUTH_REDIRECT_BASE_URL}/api/v1/auth/oauth/callback/{OAuthProvider.GITHUB.value}",
        )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GitHub OAuth provider not configured")


async def get_oauth_state(
    state: str, state_storage: AbstractSessionStorage[OAuthState] = Depends(get_oauth_state_storage)
) -> OAuthState | None:
    """
    Get and validate the OAuth state from storage.

    Args:
        state: State parameter from OAuth callback
        state_storage: Storage backend for OAuth state

    Returns:
        OAuthState if found and valid, None otherwise
    """
    return await state_storage.get(state, OAuthState)
