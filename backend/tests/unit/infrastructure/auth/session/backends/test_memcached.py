import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from src.infrastructure.auth.session.backends.memcached import MemcachedSessionStorage


class MemcachedTestSessionData(BaseModel):
    """Test data model for session testing."""

    user_id: int
    session_id: str
    is_active: bool = True
    metadata: dict = {}


@pytest.fixture
def mock_memcached():
    """Create a mock Memcached client."""
    memcached_mock = AsyncMock()
    memcached_mock.get = AsyncMock()
    memcached_mock.set = AsyncMock()
    memcached_mock.delete = AsyncMock()
    return memcached_mock


@pytest.fixture
def memcached_storage(mock_memcached):
    """Create a Memcached session storage instance with a mock client."""
    with patch("src.infrastructure.auth.session.backends.memcached.aiomcache.Client", return_value=mock_memcached):
        storage: MemcachedSessionStorage[MemcachedTestSessionData] = MemcachedSessionStorage(
            prefix="test_session:", expiration=1800
        )
        storage.client = mock_memcached
        return storage


def encode_key(key):
    """Helper function to encode a key the same way the storage class does."""
    if len(key) > 240:
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        key = f"{key[:200]}:{key_hash}"
    return key.encode("utf-8")


@pytest.mark.asyncio
async def test_create_session(memcached_storage, mock_memcached):
    """Test creating a new session."""
    session_id = "test-session-id"
    test_data = MemcachedTestSessionData(user_id=1, session_id=session_id)

    mock_memcached.set.return_value = True
    mock_memcached.get.return_value = None

    result = await memcached_storage.create(test_data, session_id=session_id)

    assert result == session_id
    assert mock_memcached.set.call_count == 2

    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)
    assert mock_memcached.set.call_args_list[0][0][0] == encoded_key


@pytest.mark.asyncio
async def test_get_session(memcached_storage, mock_memcached):
    """Test retrieving a session by ID."""
    session_id = "test-session-id"
    test_data = MemcachedTestSessionData(user_id=1, session_id=session_id)

    encoded_data = test_data.model_dump_json().encode("utf-8")
    mock_memcached.get.return_value = encoded_data

    result = await memcached_storage.get(session_id, MemcachedTestSessionData)

    assert result is not None
    assert result.user_id == test_data.user_id
    assert result.session_id == test_data.session_id

    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)
    mock_memcached.get.assert_called_once_with(encoded_key)


@pytest.mark.asyncio
async def test_get_session_not_found(memcached_storage, mock_memcached):
    """Test retrieving a non-existent session."""
    session_id = "nonexistent-session-id"

    mock_memcached.get.return_value = None
    result = await memcached_storage.get(session_id, MemcachedTestSessionData)
    assert result is None

    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)
    mock_memcached.get.assert_called_once_with(encoded_key)


@pytest.mark.asyncio
async def test_update_session(memcached_storage, mock_memcached):
    """Test updating an existing session."""
    session_id = "test-session-id"
    test_data = MemcachedTestSessionData(user_id=1, session_id=session_id)

    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)

    user_sessions_key = memcached_storage.get_user_sessions_key(1)
    encoded_user_key = encode_key(user_sessions_key)

    def mock_get_side_effect(key):
        if key == encoded_key:
            return b"existing_data"
        elif key == encoded_user_key:
            return json.dumps(["session1", session_id]).encode("utf-8")
        return None

    mock_memcached.get.side_effect = mock_get_side_effect
    mock_memcached.set.return_value = True

    result = await memcached_storage.update(session_id, test_data)

    assert result is True
    assert mock_memcached.get.call_count == 2
    assert encoded_key in [call_args[0][0] for call_args in mock_memcached.get.call_args_list]
    assert encoded_user_key in [call_args[0][0] for call_args in mock_memcached.get.call_args_list]


@pytest.mark.asyncio
async def test_delete_session(memcached_storage, mock_memcached):
    """Test deleting a session."""
    session_id = "test-session-id"
    test_data = MemcachedTestSessionData(user_id=1, session_id=session_id)

    encoded_data = test_data.model_dump_json().encode("utf-8")
    mock_memcached.get.return_value = encoded_data

    user_sessions = [session_id, "other-session"]
    encoded_user_sessions = json.dumps(user_sessions).encode("utf-8")

    mock_memcached.get.side_effect = lambda key: (
        encoded_data if encode_key(memcached_storage.get_key(session_id)) == key else encoded_user_sessions
    )

    result = await memcached_storage.delete(session_id)
    assert result is True

    assert mock_memcached.delete.call_count == 1
    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)
    mock_memcached.delete.assert_called_once_with(encoded_key)


@pytest.mark.asyncio
async def test_extend_session(memcached_storage, mock_memcached):
    """Test extending the expiration of a session."""
    session_id = "test-session-id"
    test_data = MemcachedTestSessionData(user_id=1, session_id=session_id)
    encoded_data = test_data.model_dump_json().encode("utf-8")

    session_key = memcached_storage.get_key(session_id)
    encoded_key = encode_key(session_key)
    user_sessions_key = memcached_storage.get_user_sessions_key(1)
    encoded_user_key = encode_key(user_sessions_key)

    def mock_get_side_effect(key):
        if key == encoded_key:
            return encoded_data
        elif key == encoded_user_key:
            return json.dumps(["session1", session_id]).encode("utf-8")
        return None

    mock_memcached.get.side_effect = mock_get_side_effect
    mock_memcached.set.return_value = True

    result = await memcached_storage.extend(session_id)
    assert result is True

    assert mock_memcached.get.call_count == 2
    assert encoded_key in [call_args[0][0] for call_args in mock_memcached.get.call_args_list]
    assert encoded_user_key in [call_args[0][0] for call_args in mock_memcached.get.call_args_list]

    assert mock_memcached.set.call_count >= 2
    session_set_calls = [c for c in mock_memcached.set.call_args_list if c[0][0] == encoded_key]
    assert len(session_set_calls) == 1
    assert session_set_calls[0][1]["exptime"] == 1800  # Default expiration


@pytest.mark.asyncio
async def test_exists_session(memcached_storage, mock_memcached):
    """Test checking if a session exists."""
    session_id = "test-session-id"

    mock_memcached.get.return_value = b"some_data"
    result = await memcached_storage.exists(session_id)
    assert result is True

    mock_memcached.get.return_value = None
    result = await memcached_storage.exists(session_id)
    assert result is False


@pytest.mark.asyncio
async def test_get_user_sessions(memcached_storage, mock_memcached):
    """Test retrieving all sessions for a user."""
    user_id = 1
    session_ids = ["session1", "session2", "session3"]

    encoded_data = json.dumps(session_ids).encode("utf-8")
    mock_memcached.get.return_value = encoded_data

    result = await memcached_storage.get_user_sessions(user_id)

    assert result == session_ids

    user_sessions_key = memcached_storage.get_user_sessions_key(user_id)
    encoded_key = encode_key(user_sessions_key)
    mock_memcached.get.assert_called_once_with(encoded_key)
