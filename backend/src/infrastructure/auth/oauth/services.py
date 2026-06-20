import secrets
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from ....modules.user.crud import crud_users
from ....modules.user.enums import OAuthProvider
from ....modules.user.schemas import UserCreateInternal, UserRead
from ...auth.utils import get_password_hash
from ...logging import get_logger
from .schemas import OAuthUserInfo

logger = get_logger()


class OAuthAccountService:
    """
    Service for handling OAuth account creation and linking.

    This service is responsible for:
    - Linking OAuth accounts to existing users
    - Creating new users from OAuth account information
    - Handling the user lookup during OAuth authentication
    """

    async def get_or_create_user(self, oauth_user_info: OAuthUserInfo, db: AsyncSession) -> tuple[dict[str, Any], bool]:
        """
        Get existing user or create a new one from OAuth information.

        Args:
            oauth_user_info: Standardized OAuth user info
            db: SQLAlchemy async session

        Returns:
            Tuple of (user_dict, created) where created is True if a new user was created

        Raises:
            ValueError: If required user data is missing
        """
        provider_field = f"{oauth_user_info.provider}_id"
        provider_id_filter = {provider_field: oauth_user_info.provider_user_id}

        user = await crud_users.get(db=db, filter_by=provider_id_filter)

        if user:
            logger.info(f"Found existing user by {provider_field}")
            return user, False

        if oauth_user_info.email:
            user = await crud_users.get(db=db, filter_by={"email": oauth_user_info.email})

            if user:
                logger.info(f"Found existing user by email {oauth_user_info.email}")

                update_data = {provider_field: oauth_user_info.provider_user_id, "oauth_updated_at": datetime.now(UTC)}

                user = await crud_users.update(db=db, object_id=user["id"], object=update_data)
                return cast(dict[str, Any], user), False

        logger.info("Creating new user from OAuth information")
        return await self._create_user_from_oauth(oauth_user_info, db)

    async def _create_user_from_oauth(self, oauth_user_info: OAuthUserInfo, db: AsyncSession) -> tuple[dict[str, Any], bool]:
        """
        Create a new user from OAuth user information.

        Args:
            oauth_user_info: Standardized OAuth user info
            db: SQLAlchemy async session

        Returns:
            Tuple of (user_dict, True) indicating a new user was created

        Raises:
            ValueError: If required user data is missing for account creation
        """
        if not oauth_user_info.email:
            logger.warning("Cannot create user without email")
            raise ValueError("Email is required for user creation")

        username = oauth_user_info.username
        if not username:
            username_base = oauth_user_info.given_name or oauth_user_info.name or oauth_user_info.email.split("@")[0]

            username_base = username_base.lower().replace(" ", "_")
            username = username_base

            i = 1
            while await crud_users.exists(db=db, filter_by={"username": username}):
                username = f"{username_base}{i}"
                i += 1
        else:
            if await crud_users.exists(db=db, filter_by={"username": username}):
                username_base = username
                i = 1
                while await crud_users.exists(db=db, filter_by={"username": username}):
                    username = f"{username_base}{i}"
                    i += 1

        name = oauth_user_info.name or f"{oauth_user_info.given_name or ''} {oauth_user_info.family_name or ''}".strip()
        if not name and oauth_user_info.email:
            name = oauth_user_info.email.split("@")[0]

        random_password = secrets.token_urlsafe(16)

        user_data = UserCreateInternal(
            username=username,
            email=oauth_user_info.email,
            name=name,
            hashed_password=get_password_hash(random_password),
            email_verified=oauth_user_info.email_verified,
            google_id=oauth_user_info.provider_user_id if oauth_user_info.provider == OAuthProvider.GOOGLE.value else None,
            github_id=oauth_user_info.provider_user_id if oauth_user_info.provider == OAuthProvider.GITHUB.value else None,
            oauth_provider=oauth_user_info.provider,
            oauth_created_at=datetime.now(UTC),
            oauth_updated_at=datetime.now(UTC),
        )

        user = await crud_users.create(db=db, object=user_data, schema_to_select=UserRead)

        return user, True


oauth_account_service = OAuthAccountService()
