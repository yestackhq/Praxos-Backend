from typing import Any

from ..provider import AbstractOAuthProvider
from ..schemas import OAuthUserInfo


class GoogleOAuthProvider(AbstractOAuthProvider):
    """
    OAuth authentication provider for Google Sign-In.

    This provider implements Google's OAuth 2.0 authentication flow,
    allowing users to sign in with their Google accounts. It handles
    the OAuth flow and standardizes the user information format.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str] | None = None,
    ):
        """
        Initialize the Google OAuth provider.

        Args:
            client_id: Google OAuth client ID from Google Cloud Console
            client_secret: Google OAuth client secret
            redirect_uri: Callback URL for OAuth flow completion
            scopes: Optional list of Google OAuth scopes to request.
                   If not provided, uses default scopes for basic profile
                   and email access.
        """
        default_scopes = [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ]

        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=scopes or default_scopes,
            authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            userinfo_endpoint="https://www.googleapis.com/oauth2/v3/userinfo",
            provider_name="google",
        )

    async def get_authorization_url(
        self, state: str | None = None, pkce: bool = True, extra_params: dict[str, str] | None = None
    ) -> dict[str, str]:
        """
        Get Google authorization URL with additional parameters.

        Adds Google-specific parameters like access_type=offline to
        request a refresh token.

        Args:
            state: Optional state parameter for CSRF protection
            pkce: Whether to use PKCE for enhanced security
            extra_params: Additional query parameters to include

        Returns:
            Dict with authorization URL and state/PKCE parameters
        """
        if extra_params is None:
            extra_params = {}

        extra_params["access_type"] = "offline"
        extra_params["prompt"] = "consent"

        return await super().get_authorization_url(state, pkce, extra_params)

    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """
        Process Google user info into standardized format.

        Args:
            user_info: Raw user info from Google containing fields like
                      sub, email, name, picture, etc.

        Returns:
            Standardized user info
        """
        return OAuthUserInfo(
            provider="google",
            provider_user_id=str(user_info.get("sub", "")),
            email=user_info.get("email"),
            email_verified=user_info.get("email_verified", False),
            name=user_info.get("name"),
            given_name=user_info.get("given_name"),
            family_name=user_info.get("family_name"),
            username=None,
            picture=user_info.get("picture"),
            raw_data=user_info,
        )

    @classmethod
    def create(cls, client_id: str, client_secret: str, redirect_uri: str) -> "GoogleOAuthProvider":
        """
        Factory method to create an instance with default settings.

        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: Callback URL for OAuth flow completion

        Returns:
            Configured GoogleOAuthProvider instance
        """
        return cls(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
