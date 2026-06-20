"""OAuth authentication integration."""

from .factory import OAuthProviderFactory
from .providers.github import GitHubOAuthProvider
from .providers.google import GoogleOAuthProvider
from .schemas import OAuthState, OAuthToken, OAuthUserInfo
from .services import oauth_account_service

__all__ = [
    "GoogleOAuthProvider",
    "GitHubOAuthProvider",
    "OAuthProviderFactory",
    "OAuthState",
    "OAuthUserInfo",
    "OAuthToken",
    "oauth_account_service",
]
