import base64
import hashlib
import secrets
from abc import ABC, abstractmethod
from typing import Any, cast
from urllib.parse import urlencode

import httpx

from ...logging import get_logger
from .schemas import OAuthUserInfo

logger = get_logger()


class AbstractOAuthProvider(ABC):
    """
    Abstract base class for OAuth 2.0 authentication providers.

    This class defines the interface that all OAuth providers must implement
    and provides common functionality for the OAuth authentication flow.

    Attributes:
        client_id: OAuth client ID from provider
        client_secret: OAuth client secret
        redirect_uri: URI to redirect after authentication
        scopes: List of OAuth scopes to request
        authorize_endpoint: Provider's authorization endpoint
        token_endpoint: Provider's token endpoint
        userinfo_endpoint: Provider's user info endpoint
        name: Provider identifier (e.g. "google", "github")
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str],
        authorize_endpoint: str,
        token_endpoint: str,
        userinfo_endpoint: str,
        provider_name: str,
    ):
        """Initialize the OAuth provider with required configuration."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes
        self.authorize_endpoint = authorize_endpoint
        self.token_endpoint = token_endpoint
        self.userinfo_endpoint = userinfo_endpoint
        self._name = provider_name

    @property
    def name(self) -> str:
        """Get the provider name."""
        return self._name

    def generate_state(self) -> str:
        """Generate a random state parameter for CSRF protection."""
        return secrets.token_urlsafe(32)

    def generate_pkce_codes(self) -> dict[str, str]:
        """Generate PKCE code challenge and verifier for auth flow."""
        code_verifier = secrets.token_urlsafe(64)
        code_verifier_bytes = code_verifier.encode("ascii")
        code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier_bytes).digest()).decode("ascii").rstrip("=")

        return {"code_verifier": code_verifier, "code_challenge": code_challenge}

    async def get_authorization_url(
        self, state: str | None = None, pkce: bool = True, extra_params: dict[str, str] | None = None
    ) -> dict[str, str]:
        """
        Get the authorization URL for redirecting users to the provider.

        Args:
            state: Optional state parameter for CSRF protection. If not provided,
                  a random state will be generated.
            pkce: Whether to use PKCE extension for enhanced security
            extra_params: Additional query parameters to include in the URL

        Returns:
            Dict containing the authorization URL and state/pkce parameters
        """
        if state is None:
            state = self.generate_state()

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": " ".join(self.scopes),
        }

        result = {"url": "", "state": state}

        if pkce:
            pkce_codes = self.generate_pkce_codes()
            params["code_challenge"] = pkce_codes["code_challenge"]
            params["code_challenge_method"] = "S256"
            result["code_verifier"] = pkce_codes["code_verifier"]

        if extra_params:
            params.update(extra_params)

        result["url"] = f"{self.authorize_endpoint}?{urlencode(params)}"
        return result

    async def exchange_code(
        self, code: str, code_verifier: str | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code received from provider
            code_verifier: PKCE code verifier if PKCE was used
            headers: Additional headers for the token request

        Returns:
            Dict containing access_token and other provider response
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.token_endpoint, data=data, headers=request_headers)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
        except Exception as e:
            logger.error(f"Error exchanging code for {self.name}: {str(e)}")
            raise

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        """
        Get user information from the provider using an access token.

        Args:
            access_token: OAuth access token

        Returns:
            Dict containing user profile information from provider
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.userinfo_endpoint, headers=headers)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
        except Exception as e:
            logger.error(f"Error fetching user info for {self.name}: {str(e)}")
            raise

    async def validate_token(self, access_token: str) -> bool:
        """
        Validate that an access token is still valid.

        Default implementation checks if we can fetch user info.
        Override for providers with specific token validation endpoints.

        Args:
            access_token: OAuth access token to validate

        Returns:
            True if token is valid, False otherwise
        """
        try:
            await self.get_user_info(access_token)
            return True
        except Exception:
            return False

    @abstractmethod
    async def process_user_info(self, user_info: dict[str, Any]) -> OAuthUserInfo:
        """
        Process provider-specific user info into a standardized format.

        Must be implemented by each provider to normalize user data.

        Args:
            user_info: Raw user info from provider

        Returns:
            Standardized user info
        """
        pass
