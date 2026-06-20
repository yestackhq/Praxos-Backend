"""Session storage factory + backwards-compatible re-export of the abstract base.

The abstract base (``AbstractSessionStorage``) lives in ``base.py`` so
that the concrete ``backends/`` implementations don't form a cycle with
this module. Existing callers that import ``AbstractSessionStorage``
from ``.storage`` keep working via the re-export below.
"""

from typing import Generic, TypeVar, cast

from pydantic import BaseModel

from ...config import SessionBackend
from ...logging import get_logger
from .backends.memcached import MemcachedSessionStorage
from .backends.memory import MemorySessionStorage
from .backends.redis import RedisSessionStorage
from .base import AbstractSessionStorage

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)

__all__ = ["AbstractSessionStorage", "SessionStorage", "get_session_storage"]


class SessionStorage(AbstractSessionStorage[T], Generic[T]):
    def __new__(cls, backend: str = "memory", **kwargs) -> "SessionStorage[T]":
        """Factory method to create the appropriate session storage backend.

        Args:
            backend: The backend to use ("redis", "memcached", "memory")
            **kwargs: Additional arguments to pass to the backend

        Returns:
            An initialized storage backend
        """
        storage: AbstractSessionStorage[T] = get_session_storage(backend, cast(type[T], BaseModel), **kwargs)
        return cast("SessionStorage[T]", storage)


def get_session_storage(backend: str, model_type: type[BaseModel], **kwargs) -> AbstractSessionStorage[T]:
    """Get the appropriate session storage backend.

    Args:
        backend: The backend to use ("redis", "memcached", "memory")
        model_type: The pydantic model type for type checking
        **kwargs: Additional arguments to pass to the backend

    Returns:
        An initialized storage backend
    """
    if backend == SessionBackend.REDIS.value:
        return RedisSessionStorage(**kwargs)
    elif backend == SessionBackend.MEMCACHED.value:
        return MemcachedSessionStorage(**kwargs)
    elif backend == SessionBackend.MEMORY.value:
        return MemorySessionStorage(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
