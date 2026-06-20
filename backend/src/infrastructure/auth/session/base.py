"""Abstract base for session storage backends.

Lives in its own module so that the concrete backend implementations
under ``backends/`` can subclass it without participating in the
``storage.py`` factory's import cycle. The factory (``storage.py``)
imports both this base and the concrete backends; concrete backends
only import this base.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AbstractSessionStorage(Generic[T], ABC):
    """Abstract base class for session storage implementations."""

    def __init__(
        self,
        prefix: str = "session:",
        expiration: int = 1800,
    ):
        """Initialize the session storage.

        Args:
            prefix: Prefix for all session keys
            expiration: Default session expiration in seconds
        """
        self.prefix = prefix
        self.expiration = expiration

    def generate_session_id(self) -> str:
        """Generate a unique session ID.

        Returns:
            A unique session ID string
        """
        return str(uuid4())

    def get_key(self, session_id: str) -> str:
        """Generate the full key for a session ID.

        Args:
            session_id: The session ID

        Returns:
            The full storage key
        """
        return f"{self.prefix}{session_id}"

    @abstractmethod
    async def create(self, data: T, session_id: str | None = None, expiration: int | None = None) -> str:
        """Create a new session.

        Args:
            data: Session data (must be a Pydantic model)
            session_id: Optional session ID. If not provided, one will be generated
            expiration: Optional custom expiration in seconds

        Returns:
            The session ID
        """
        pass

    @abstractmethod
    async def get(self, session_id: str, model_class: type[T]) -> T | None:
        """Get session data.

        Args:
            session_id: The session ID
            model_class: The Pydantic model class to decode the data into

        Returns:
            The session data or None if session doesn't exist
        """
        pass

    @abstractmethod
    async def update(self, session_id: str, data: T, reset_expiration: bool = True, expiration: int | None = None) -> bool:
        """Update session data.

        Args:
            session_id: The session ID
            data: New session data
            reset_expiration: Whether to reset the expiration
            expiration: Optional custom expiration in seconds

        Returns:
            True if the session was updated, False if it didn't exist
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID

        Returns:
            True if the session was deleted, False if it didn't exist
        """
        pass

    @abstractmethod
    async def extend(self, session_id: str, expiration: int | None = None) -> bool:
        """Extend the expiration of a session.

        Args:
            session_id: The session ID
            expiration: Optional custom expiration in seconds

        Returns:
            True if the session was extended, False if it didn't exist
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: The session ID

        Returns:
            True if the session exists, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection."""
        pass
