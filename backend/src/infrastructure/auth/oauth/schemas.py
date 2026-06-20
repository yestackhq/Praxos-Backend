from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class OAuthState(BaseModel):
    """
    Store data needed for OAuth state validation.

    Used to maintain state between authorization request and callback.
    """

    state: str = Field(description="State parameter for CSRF protection")
    provider: str = Field(description="OAuth provider name")
    code_verifier: str | None = Field(None, description="PKCE code verifier")
    redirect_to: str | None = Field(None, description="Where to redirect after authentication")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OAuthUserInfo(BaseModel):
    """
    Standardized user information from OAuth providers.

    Each provider's raw user information is normalized to this format.
    """

    provider: str = Field(description="OAuth provider name")
    provider_user_id: str = Field(description="User ID from the provider")
    email: str | None = Field(None, description="User's email address")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    name: str | None = Field(None, description="User's full name")
    given_name: str | None = Field(None, description="User's given/first name")
    family_name: str | None = Field(None, description="User's family/last name")
    username: str | None = Field(None, description="Username if available")
    picture: str | None = Field(None, description="URL to user's profile picture")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Raw provider data")


class OAuthToken(BaseModel):
    """
    OAuth token information.

    Stores token data received from OAuth providers.
    """

    access_token: str
    token_type: str = "Bearer"
    id_token: str | None = None
    refresh_token: str | None = None
    expires_in: int | None = None
    scope: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        """
        Check if the token is expired.

        Returns:
            True if expired, False if still valid or no expiration set
        """
        if not self.expires_in:
            return False

        expiry = self.created_at.timestamp() + self.expires_in
        return datetime.now(UTC).timestamp() > expiry
