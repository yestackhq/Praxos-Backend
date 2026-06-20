from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.infrastructure.auth.session.backends.redis import RedisSessionStorage


class RedisTestSessionData(BaseModel):
    """Test data model for session testing."""

    user_id: int
    session_id: str
    is_active: bool = True
    metadata: dict = {}


@pytest.fixture
def mock_pipeline():
    """Create a mock Redis pipeline."""
    pipeline = MagicMock()
    pipeline.set = MagicMock(return_value=pipeline)
    pipeline.delete = MagicMock(return_value=pipeline)
    pipeline.sadd = MagicMock(return_value=pipeline)
    pipeline.srem = MagicMock(return_value=pipeline)
    pipeline.expire = MagicMock(return_value=pipeline)
    pipeline.execute = AsyncMock(return_value=[True, True])
    return pipeline


@pytest.fixture
def mock_redis(mock_pipeline):
    """Create a mock Redis client with a non-coroutine pipeline."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock()
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock()
    redis_mock.sadd = AsyncMock()
    redis_mock.srem = AsyncMock()
    redis_mock.smembers = AsyncMock()
    redis_mock.expire = AsyncMock()
    redis_mock.exists = AsyncMock()
    redis_mock.ttl = AsyncMock(return_value=1000)
    redis_mock.pipeline = MagicMock(return_value=mock_pipeline)

    return redis_mock


@pytest.fixture
def redis_storage(mock_redis):
    """Create a Redis session storage instance with a mock Redis client."""
    with patch("src.infrastructure.auth.session.backends.redis.AsyncRedis", return_value=mock_redis):
        storage: RedisSessionStorage[RedisTestSessionData] = RedisSessionStorage(prefix="test_session:", expiration=1800)
        storage.client = mock_redis
        return storage


@pytest.fixture
def fake_redis():
    """Create a Redis storage with a more complete mock for the delete_pattern test."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value="{}")
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.scan_iter = AsyncMock()
    redis_mock.keys = AsyncMock(
        return_value=["test_session:1", "test_session:2", "test_session:3", "test_session:4", "test_session:5"]
    )

    # Set up pipeline for create
    pipeline_mock = MagicMock()
    pipeline_mock.set = MagicMock(return_value=pipeline_mock)
    pipeline_mock.sadd = MagicMock(return_value=pipeline_mock)
    pipeline_mock.expire = MagicMock(return_value=pipeline_mock)
    pipeline_mock.delete = MagicMock(return_value=pipeline_mock)
    pipeline_mock.execute = AsyncMock(return_value=[True, True, True])

    redis_mock.pipeline = MagicMock(return_value=pipeline_mock)

    # Configure scan_iter to yield login keys
    async def mock_scan_iter(**kwargs):
        if kwargs.get("match") == "login:*":
            for i in range(3):
                yield f"login:user:test{i}"
        # In async functions we can't use 'yield from'

    redis_mock.scan_iter.side_effect = mock_scan_iter

    # Create storage with the mock
    with patch("src.infrastructure.auth.session.backends.redis.Redis", return_value=redis_mock):
        storage: RedisSessionStorage = RedisSessionStorage()
        storage.client = redis_mock
        return storage


@pytest.mark.asyncio
async def test_create_session(redis_storage, mock_redis, mock_pipeline):
    """Test creating a new session."""
    session_id = "test-session-id"
    test_data = RedisTestSessionData(user_id=1, session_id=session_id)

    mock_pipeline.execute.return_value = [True, True, True]

    result = await redis_storage.create(test_data, session_id=session_id)

    assert result == session_id

    mock_pipeline.set.assert_called()
    mock_pipeline.sadd.assert_called()
    mock_pipeline.expire.assert_called()
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_session(redis_storage, mock_redis):
    """Test retrieving a session by ID."""
    session_id = "test-session-id"
    test_data = RedisTestSessionData(user_id=1, session_id=session_id)

    mock_redis.get.return_value = test_data.model_dump_json()
    result = await redis_storage.get(session_id, RedisTestSessionData)

    assert result is not None
    assert result.user_id == test_data.user_id
    assert result.session_id == test_data.session_id

    mock_redis.get.assert_called_once_with(f"test_session:{session_id}")


@pytest.mark.asyncio
async def test_get_session_not_found(redis_storage, mock_redis):
    """Test retrieving a non-existent session."""
    session_id = "nonexistent-session-id"

    mock_redis.get.return_value = None

    result = await redis_storage.get(session_id, RedisTestSessionData)
    assert result is None

    mock_redis.get.assert_called_once_with(f"test_session:{session_id}")


@pytest.mark.asyncio
async def test_update_session(redis_storage, mock_redis, mock_pipeline):
    """Test updating an existing session."""
    session_id = "test-session-id"
    test_data = RedisTestSessionData(user_id=1, session_id=session_id)

    mock_redis.exists.return_value = True
    mock_pipeline.execute.return_value = [True, True]

    result = await redis_storage.update(session_id, test_data)
    assert result is True

    mock_redis.exists.assert_called_once()
    mock_pipeline.set.assert_called()
    mock_pipeline.expire.assert_called()
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_session(redis_storage, mock_redis, mock_pipeline):
    """Test deleting a session."""
    session_id = "test-session-id"
    test_data = RedisTestSessionData(user_id=1, session_id=session_id)

    mock_redis.get.return_value = test_data.model_dump_json()
    mock_pipeline.execute.return_value = [1, 1]

    result = await redis_storage.delete(session_id)
    assert result is True

    mock_redis.get.assert_called_once()
    mock_pipeline.delete.assert_called()
    mock_pipeline.srem.assert_called()
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_extend_session(redis_storage, mock_redis, mock_pipeline):
    """Test extending the expiration of a session."""
    session_id = "test-session-id"
    test_data = RedisTestSessionData(user_id=1, session_id=session_id)

    mock_redis.get.return_value = test_data.model_dump_json()
    mock_pipeline.execute.return_value = [True, True]

    result = await redis_storage.extend(session_id)
    assert result is True

    mock_redis.get.assert_called_once()
    mock_pipeline.expire.assert_called()
    mock_pipeline.execute.assert_called_once()


@pytest.mark.asyncio
async def test_exists_session(redis_storage, mock_redis):
    """Test checking if a session exists."""
    session_id = "test-session-id"

    mock_redis.exists.return_value = True

    result = await redis_storage.exists(session_id)
    assert result is True

    mock_redis.exists.assert_called_once_with(f"test_session:{session_id}")


@pytest.mark.asyncio
async def test_get_user_sessions(redis_storage, mock_redis):
    """Test retrieving all sessions for a user."""
    user_id = 1
    session_ids = ["session1", "session2", "session3"]

    mock_redis.smembers.return_value = session_ids

    result = await redis_storage.get_user_sessions(user_id)
    assert result == session_ids

    mock_redis.smembers.assert_called_once_with(f"{redis_storage.user_sessions_prefix}{user_id}")


@pytest.mark.asyncio
async def test_delete_pattern(mock_redis):
    """Test deleting keys matching a pattern from Redis."""
    storage: RedisSessionStorage = RedisSessionStorage()
    storage.client = mock_redis

    login_keys = [f"login:user:test{i}".encode() for i in range(3)]

    class AsyncIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_redis.scan_iter = MagicMock(return_value=AsyncIterator(login_keys))

    mock_pipeline = MagicMock()
    mock_pipeline.delete = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = AsyncMock(return_value=[1, 1, 1])
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

    deleted_count = await storage.delete_pattern("login:*")

    mock_redis.scan_iter.assert_called_once_with(match="login:*")

    mock_redis.pipeline.assert_called_once()
    assert mock_pipeline.delete.call_count == 3
    mock_pipeline.execute.assert_called_once()

    assert deleted_count == 3
