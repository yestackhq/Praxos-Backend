from typing import Any

import httpx

from ..provider import AbstractOAuthProvider
from ..schemas import OAuthUserInfo


class GitHubOAuthProvider(AbstractOAuthProvider):
    """
    OAuth authentication provider for GitHub Sign-In.

    This provider implements GitHub's OAuth 2.0 authentication flow,
    allowing users to sign in with their GitHub accounts. It handles
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
        Initialize the GitHub OAuth provider.

        Args:
            client_id: GitHub OAuth client ID from GitHub Developer Settings
            client_secret: GitHub OAuth client secret
            redirect_uri: Callback URL for OAuth flow completion
            scopes: Optional list of GitHub OAuth scopes to request.
                   If not provided, uses default scopes for basic profile
                   and email access.
        """
        default_scopes = ["read:user", "user:email"]

        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=scopes or default_scopes,
            authorize_endpoint="https://github.com/login/oauth/authorize",
            token_endpoint="https://github.com/login/oauth/access_token",
            userinfo_endpoint="https://api.github.com/user",
            provider_name="github",
        )

    async def exchange_code(
        self, code: str, code_verifier: str | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Override to handle GitHub-specific token response.

        GitHub requires the 'Accept: application/json' header to receive
        the response in JSON format instead of the default
        application/x-www-form-urlencoded.

        Args:
            code: The authorization code received from GitHub
            code_verifier: PKCE code verifier if PKCE was used
            headers: Optional additional headers for the token request

        Returns:
            Dict[str, Any]: The token response containing:
                - access_token: OAuth access token
                - token_type: Token type (usually "bearer")
                - scope: Granted scopes as a comma-separated string
        """
        if headers is None:
            headers = {}

        headers["Accept"] = "application/json"
        return await super().exchange_code(code, code_verifier, headers)

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        """
        Get both user profile and email information from GitHub.

        Makes two API calls:
        1. Fetches the user's profile from the user endpoint
        2. Fetches the user's email addresses from the emails endpoint

        GitHub requires separate API calls to get email information,
        especially for users with private email addresses.

        Args:
            access_token: Valid GitHub OAuth access token

        Returns:
            Dict[str, Any]: Combined user profile and email data
        """
        profile = await super().get_user_info(access_token)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.github.com/user/emails", headers=headers)

            if response.status_code == 200:
                emails_data = response.json()
                profile["emails"] = emails_data

        return profile

    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """
        Process GitHub user info into standardized format.

        Transforms the raw user info from GitHub's API into a consistent
        format. Handles the extraction of primary email and its verification
        status from the emails array.

        Args:
            user_info: Raw user info from GitHub containing fields like
                      id, login, name, emails array, etc.

        Returns:
            Standardized user info
        """
        email = None
        email_verified = False

        if emails := user_info.get("emails", []):
            for e in emails:
                if e.get("primary"):
                    email = e.get("email")
                    email_verified = e.get("verified", False)
                    break

        return OAuthUserInfo(
            provider="github",
            provider_user_id=str(user_info.get("id")),
            email=email,
            email_verified=email_verified,
            name=user_info.get("name"),
            given_name=None,
            family_name=None,
            username=user_info.get("login"),
            picture=user_info.get("avatar_url"),
            raw_data=user_info,
        )

    @classmethod
    def create(cls, client_id: str, client_secret: str, redirect_uri: str) -> "GitHubOAuthProvider":
        """
        Factory method to create an instance with default settings.

        Args:
            client_id: GitHub OAuth client ID
            client_secret: GitHub OAuth client secret
            redirect_uri: Callback URL for OAuth flow completion

        Returns:
            Configured GitHubOAuthProvider instance
        """
        return cls(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
