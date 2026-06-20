from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionData(BaseModel):
    """Common base data for any user session."""

    user_id: int
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    ip_address: str
    user_agent: str
    device_info: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(SessionData):
    """Schema for creating a new session."""

    pass


class SessionUpdate(BaseModel):
    """Schema for updating a session."""

    last_activity: datetime | None = None
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class UserAgentInfo(BaseModel):
    """Parsed User-Agent information."""

    browser: str
    browser_version: str
    os: str
    device: str
    is_mobile: bool
    is_tablet: bool
    is_pc: bool


class CSRFToken(BaseModel):
    """CSRF token schema."""

    token: str
    user_id: int
    session_id: str
    expiry: datetime
