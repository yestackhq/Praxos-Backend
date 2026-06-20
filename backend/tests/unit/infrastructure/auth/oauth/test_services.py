"""Tests for the OAuthAccountService class."""

from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.auth.oauth.schemas import OAuthUserInfo
from src.infrastructure.auth.oauth.services import OAuthAccountService
from src.modules.user.crud import crud_users
from src.modules.user.schemas import UserCreateInternal


@pytest.fixture
def oauth_service():
    """Create an instance of OAuthAccountService for testing."""
    return OAuthAccountService()


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def google_user_info():
    """Create a sample Google user info for testing."""
    return OAuthUserInfo(
        provider="google",
        provider_user_id="123456789",
        email="user@example.com",
        email_verified=True,
        name="Test User",
        given_name="Test",
        family_name="User",
        username=None,
        picture="https://example.com/photo.jpg",
        raw_data={
            "sub": "123456789",
            "email": "user@example.com",
            "email_verified": True,
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
            "picture": "https://example.com/photo.jpg",
        },
    )


@pytest.fixture
def github_user_info():
    """Create a sample GitHub user info for testing."""
    return OAuthUserInfo(
        provider="github",
        provider_user_id="987654321",
        email="user@example.com",
        email_verified=True,
        name="Test User",
        given_name=None,
        family_name=None,
        username="testuser",
        picture="https://github.com/avatars/user.jpg",
        raw_data={
            "id": 987654321,
            "login": "testuser",
            "name": "Test User",
            "email": "user@example.com",
            "avatar_url": "https://github.com/avatars/user.jpg",
            "emails": [{"email": "user@example.com", "primary": True, "verified": True}],
        },
    )


@pytest.mark.asyncio
async def test_get_or_create_user_existing_by_provider_id(oauth_service, mock_db, google_user_info):
    """Test getting a user that already exists with the provider ID."""
    existing_user = {
        "id": 1,
        "username": "existinguser",
        "email": "user@example.com",
        "google_id": "123456789",
    }

    mock_db.execute = AsyncMock()

    with patch.object(crud_users, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = existing_user

        user, created = await oauth_service.get_or_create_user(google_user_info, mock_db)

        assert user == existing_user
        assert created is False

        mock_get.assert_called_once()
        assert mock_get.call_args[1]["filter_by"] == {"google_id": "123456789"}


@pytest.mark.asyncio
async def test_get_or_create_user_existing_by_email(oauth_service, mock_db, google_user_info):
    """Test getting a user that exists by email but not provider ID."""
    existing_user = {
        "id": 1,
        "username": "existinguser",
        "email": "user@example.com",
        "google_id": None,
    }

    mock_db.execute = AsyncMock()

    with (
        patch.object(crud_users, "get", new_callable=AsyncMock) as mock_get,
        patch.object(crud_users, "update", new_callable=AsyncMock) as mock_update,
    ):
        mock_get.side_effect = [None, existing_user]
        mock_update.return_value = {**existing_user, "google_id": "123456789"}

        user, created = await oauth_service.get_or_create_user(google_user_info, mock_db)

        assert user["id"] == 1
        assert user["google_id"] == "123456789"
        assert created is False

        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0][1]["filter_by"] == {"google_id": "123456789"}
        assert mock_get.call_args_list[1][1]["filter_by"] == {"email": "user@example.com"}

        mock_update.assert_called_once()
        assert mock_update.call_args[1]["object_id"] == 1
        assert mock_update.call_args[1]["object"]["google_id"] == "123456789"
        assert "oauth_updated_at" in mock_update.call_args[1]["object"]


@pytest.mark.asyncio
async def test_get_or_create_user_new_user(oauth_service, mock_db, google_user_info):
    """Test creating a new user when none exists."""
    new_user = {
        "id": 1,
        "username": "testuser",
        "email": "user@example.com",
        "google_id": "123456789",
        "oauth_provider": "google",
    }

    with (
        patch.object(crud_users, "get", new_callable=AsyncMock) as mock_get,
        patch.object(crud_users, "exists", new_callable=AsyncMock) as mock_exists,
        patch.object(crud_users, "create", new_callable=AsyncMock) as mock_create,
        patch("secrets.token_urlsafe", return_value="random_password"),
    ):
        mock_get.return_value = None
        mock_exists.return_value = False
        mock_create.return_value = new_user

        user, created = await oauth_service.get_or_create_user(google_user_info, mock_db)

        assert user == new_user
        assert created is True

        assert mock_get.call_count == 2

        mock_exists.assert_called_once()

        mock_create.assert_called_once()
        create_args = mock_create.call_args[1]["object"]
        assert create_args.email == "user@example.com"
        assert create_args.name == "Test User"
        assert create_args.username.startswith("test")
        assert create_args.email_verified is True
        assert create_args.google_id == "123456789"
        assert create_args.oauth_provider == "google"


@pytest.mark.asyncio
async def test_get_or_create_user_username_conflict(oauth_service, mock_db, google_user_info):
    """Test username generation with conflicts."""
    new_user = {
        "id": 1,
        "username": "test1",
        "email": "user@example.com",
        "google_id": "123456789",
    }

    with (
        patch.object(crud_users, "get", new_callable=AsyncMock) as mock_get,
        patch.object(crud_users, "exists", new_callable=AsyncMock) as mock_exists,
        patch.object(crud_users, "create", new_callable=AsyncMock) as mock_create,
        patch("secrets.token_urlsafe", return_value="random_password"),
    ):
        mock_get.return_value = None

        mock_exists.side_effect = [True, False]

        mock_create.return_value = new_user

        user, created = await oauth_service.get_or_create_user(google_user_info, mock_db)

        assert user == new_user
        assert created is True

        assert mock_exists.call_count == 2

        mock_create.assert_called_once()
        create_args = mock_create.call_args[1]["object"]
        assert create_args.username == "test1"


@pytest.mark.asyncio
async def test_create_user_from_oauth_no_email(oauth_service, mock_db, google_user_info):
    """Test that creating a user without email raises ValueError."""
    user_info_no_email = OAuthUserInfo(
        provider="google",
        provider_user_id="123456789",
        email=None,
        email_verified=False,
        name="Test User",
        given_name="Test",
        family_name="User",
        username=None,
        picture="https://example.com/photo.jpg",
        raw_data={},
    )

    with pytest.raises(ValueError, match="Email is required for user creation"):
        await oauth_service._create_user_from_oauth(user_info_no_email, mock_db)


@pytest.mark.asyncio
async def test_create_user_from_oauth_with_username(oauth_service, mock_db):
    """Test creating a user with a pre-existing username."""
    user_info = OAuthUserInfo(
        provider="github",
        provider_user_id="987654321",
        email="user@example.com",
        email_verified=True,
        name="Test User",
        given_name=None,
        family_name=None,
        username="testuser",
        picture="https://example.com/photo.jpg",
        raw_data={},
    )

    new_user = {
        "id": 1,
        "username": "testuser",
        "email": "user@example.com",
        "github_id": "987654321",
    }

    with (
        patch.object(crud_users, "exists", new_callable=AsyncMock) as mock_exists,
        patch.object(crud_users, "create", new_callable=AsyncMock) as mock_create,
        patch("secrets.token_urlsafe", return_value="random_password"),
    ):
        mock_exists.return_value = False
        mock_create.return_value = new_user

        user, created = await oauth_service._create_user_from_oauth(user_info, mock_db)

        assert user == new_user
        assert created is True

        mock_exists.assert_called_once()
        assert mock_exists.call_args[1]["filter_by"] == {"username": "testuser"}

        mock_create.assert_called_once()
        create_args = mock_create.call_args[1]["object"]
        assert create_args.username == "testuser"
        assert create_args.email == "user@example.com"
        assert create_args.github_id == "987654321"
        assert create_args.oauth_provider == "github"


@pytest.mark.asyncio
async def test_oauth_user_creation_uses_hashed_password(oauth_service, mock_db, google_user_info):
    """OAuth user creation must use UserCreateInternal with a bcrypt-hashed password."""
    new_user = {"id": 1, "username": "test", "email": "user@example.com", "google_id": "123456789"}

    with (
        patch.object(crud_users, "get", new_callable=AsyncMock) as mock_get,
        patch.object(crud_users, "exists", new_callable=AsyncMock) as mock_exists,
        patch.object(crud_users, "create", new_callable=AsyncMock) as mock_create,
    ):
        mock_get.return_value = None
        mock_exists.return_value = False
        mock_create.return_value = new_user

        await oauth_service.get_or_create_user(google_user_info, mock_db)

        create_args = mock_create.call_args[1]["object"]

        # Must use UserCreateInternal, not UserCreate
        assert isinstance(create_args, UserCreateInternal), f"Expected UserCreateInternal, got {type(create_args).__name__}"

        # Must have hashed_password, not password
        assert hasattr(create_args, "hashed_password"), "Missing hashed_password field"
        assert not hasattr(create_args, "password"), "Should not have plaintext password field"

        # Must be a bcrypt hash (starts with $2b$)
        assert create_args.hashed_password.startswith("$2b$"), (
            f"Expected bcrypt hash, got: {create_args.hashed_password[:20]}..."
        )
