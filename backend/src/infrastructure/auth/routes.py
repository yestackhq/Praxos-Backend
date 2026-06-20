import inspect
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse

from ...modules.user.crud import crud_users
from ...modules.user.enums import OAuthProvider
from ..config.settings import get_settings
from ..dependencies import (
    AsyncSessionDep,
    CurrentSessionDataDep,
    GoogleOAuthProviderDep,
    OAuth2FormDep,
    OAuthStateStorageDep,
    OptionalSessionDataDep,
    SessionManagerDep,
)
from ..logging import get_logger
from .http_exceptions import UnauthorizedException
from .oauth.dependencies import get_oauth_state
from .oauth.schemas import OAuthState, OAuthToken
from .oauth.services import oauth_account_service
from .session.dependencies import authenticate_user

settings = get_settings()
logger = get_logger()

router = APIRouter(tags=["Authentication"])


@router.post(
    "/login",
    summary="User Login",
    description="""
            Authenticates a user and creates a new session.

            This endpoint accepts username/email and password credentials and verifies them.
            On successful authentication:
            - A new session is created
            - A session ID is set as an HTTP-only cookie
            - A CSRF token is generated for protection against CSRF attacks

            The endpoint is protected by rate limiting to prevent brute force attacks.
            After multiple failed attempts, further login attempts will be temporarily blocked.
            """,
    responses={
        200: {"description": "Login successful, session created"},
        401: {"description": "Authentication failed or rate limit exceeded"},
        429: {"description": "Too many login attempts, try again later"},
    },
    response_description="CSRF token for use in subsequent requests",
)
async def login(
    request: Request,
    response: Response,
    form_data: OAuth2FormDep,
    db: AsyncSessionDep,
    session_manager: SessionManagerDep,
) -> dict[str, str]:
    """Login endpoint to get session cookies.

    The session ID is set as an HTTP-only cookie.
    The CSRF token is set as a regular cookie and returned in the response.
    This endpoint is protected by rate limiting to prevent brute force attacks.
    """
    ip_address = request.client.host if request.client and hasattr(request.client, "host") else "unknown"

    is_allowed, attempts_remaining = await session_manager.track_login_attempt(
        ip_address=ip_address, username=form_data.username, success=False
    )

    if not is_allowed:
        logger.warning(f"Login rate limit exceeded for {form_data.username} from IP {ip_address}")
        raise UnauthorizedException("Too many failed login attempts. Please try again later.")

    user = await authenticate_user(username_or_email=form_data.username, password=form_data.password, db=db)

    if user is None:
        logger.warning(f"Failed login attempt for {form_data.username} from IP {ip_address}")
        raise UnauthorizedException("Incorrect username or password")

    try:
        await session_manager.track_login_attempt(ip_address=ip_address, username=form_data.username, success=True)

        session_id, csrf_token = await session_manager.create_session(
            request=request,
            user_id=user["id"],
            metadata={
                "login_type": "password",
                "username": user["username"],
            },
        )

        session_manager.set_session_cookies(
            response=response,
            session_id=session_id,
            csrf_token=csrf_token,
            secure=settings.SESSION_SECURE_COOKIES,
            path="/",
        )

        return {"csrf_token": csrf_token}

    except Exception as e:
        logger.error(f"Error during login: {str(e)}", exc_info=True)
        raise UnauthorizedException("An error occurred during login")


@router.post(
    "/logout",
    summary="User Logout",
    description="""
            Terminates the current user session.

            This endpoint:
            - Invalidates the active session in the storage backend
            - Clears all session-related cookies from the client

            After logout, the user will need to authenticate again to access
            protected resources. Any existing session tokens will no longer be valid.
            """,
    responses={200: {"description": "Logout successful, session terminated"}, 401: {"description": "Not authenticated"}},
    response_description="Confirmation of successful logout",
)
async def logout(
    request: Request,
    response: Response,
    session_data: CurrentSessionDataDep,
    session_manager: SessionManagerDep,
) -> dict[str, str]:
    """Logout endpoint to terminate the session and clear cookies."""
    await session_manager.terminate_session(session_data.session_id)
    session_manager.clear_session_cookies(response)

    return {"message": "Logged out successfully"}


@router.post(
    "/refresh-csrf",
    summary="Refresh CSRF Token",
    description="""
            Generates a new CSRF token for the current session.

            This endpoint should be called to obtain a fresh CSRF token when:
            - The current token is about to expire
            - After a certain period of inactivity
            - When increased security is needed for sensitive operations

            The new token is returned in the response and also set as a cookie.
            """,
    responses={200: {"description": "New CSRF token generated successfully"}, 401: {"description": "Not authenticated"}},
    response_description="The new CSRF token for the session",
)
async def refresh_csrf_token(
    request: Request,
    response: Response,
    session_data: CurrentSessionDataDep,
    session_manager: SessionManagerDep,
) -> dict[str, str]:
    """Generate a new CSRF token for the current session."""
    csrf_token = await session_manager.regenerate_csrf_token(
        user_id=session_data.user_id,
        session_id=session_data.session_id,
    )

    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        max_age=int(session_manager.session_timeout.total_seconds()),
        path="/",
        httponly=False,
        secure=settings.SESSION_SECURE_COOKIES,
        samesite="lax",
    )

    return {"csrf_token": csrf_token}


@router.get(
    "/oauth/google",
    summary="Initiate Google OAuth Login",
    description="""
            Starts the OAuth 2.0 authentication flow with Google.

            This endpoint generates the authorization URL that the user should be
            redirected to in order to authenticate with Google. The flow includes:
            - Creation of a state parameter for CSRF protection
            - Generation of PKCE code challenge (for enhanced security)
            - Setting appropriate OAuth scopes for profile access

            After successful authentication with Google, the user will be redirected
            back to this application's callback endpoint.

            An optional redirect_uri can be specified to control where the user
            is sent after the entire authentication process completes.
            """,
    responses={
        200: {"description": "Authorization URL generated successfully"},
        500: {"description": "Failed to initiate Google login"},
    },
    response_description="The Google authorization URL to redirect the user to",
)
async def oauth_google_login(
    request: Request,
    oauth_provider: GoogleOAuthProviderDep,
    state_storage: OAuthStateStorageDep,
    redirect_uri: str | None = Query(None),
) -> dict[str, str]:
    """
    Initiate OAuth login flow for Google.

    Args:
        request: The request object
        redirect_uri: Optional URI to redirect after successful authentication

    Returns:
        Dict with authorization URL to redirect the user to Google
    """
    try:
        auth_data = await oauth_provider.get_authorization_url()

        state_obj = OAuthState(
            state=auth_data["state"],
            provider=OAuthProvider.GOOGLE.value,
            redirect_to=redirect_uri,
            code_verifier=auth_data.get("code_verifier"),
        )

        await state_storage.create(data=state_obj, session_id=auth_data["state"])

        return {"url": auth_data["url"]}

    except Exception as e:
        logger.error(f"Error initiating Google OAuth: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initiate Google login")


def _is_provider_valid(provider_value: Any, expected_provider: str) -> bool:
    """Check if a provider value matches the expected provider name.

    This handles different types of values (strings, objects, mocks) safely.

    Args:
        provider_value: The provider value to check (could be a string, object, or mock)
        expected_provider: The expected provider name (e.g., "google" or "github")

    Returns:
        bool: True if the provider is valid, False otherwise
    """
    if provider_value is None:
        return False

    if isinstance(provider_value, str):
        return provider_value.lower() == expected_provider.lower()

    if hasattr(provider_value, "name") and isinstance(getattr(provider_value, "name", None), str):
        name_value: str = getattr(provider_value, "name")
        return name_value.lower() == expected_provider.lower()

    if inspect.iscoroutine(provider_value) or inspect.isawaitable(provider_value):
        return expected_provider.lower() in str(provider_value).lower()

    try:
        return expected_provider.lower() in str(provider_value).lower()
    except Exception:
        return False


@router.get(
    "/oauth/callback/google",
    summary="Google OAuth Callback Handler",
    description="""
            Processes the authentication callback from Google OAuth.

            This endpoint handles the authorization code returned by Google after
            the user has successfully authenticated. The process includes:
            - Validating the state parameter to prevent CSRF attacks
            - Exchanging the authorization code for access/refresh tokens
            - Fetching the user profile from Google
            - Creating or updating the user account in the system
            - Establishing a new session for the authenticated user

            Two response formats are supported:
            - redirect: Redirects to the frontend with success/error parameters (default)
            - json: Returns user information and tokens as a JSON response

            The json format is useful for mobile apps or single-page applications that
            handle the OAuth flow programmatically.
            """,
    responses={
        200: {"description": "Authentication successful (JSON response)"},
        302: {"description": "Authentication successful (redirect response)"},
        400: {"description": "Invalid OAuth state or other parameter"},
        401: {"description": "Authentication failed"},
        500: {"description": "Server error during authentication"},
    },
    response_description="Authentication result with session cookies set",
)
async def oauth_google_callback(
    request: Request,
    response: Response,
    oauth_provider: GoogleOAuthProviderDep,
    state_storage: OAuthStateStorageDep,
    db: AsyncSessionDep,
    session_manager: SessionManagerDep,
    code: str = Query(...),
    state: str = Query(...),
    response_format: str = Query("redirect", description="Response format, either 'redirect' or 'json'"),
):
    """
    Handle OAuth callback from Google.

    Args:
        request: The request object
        response: The response object
        code: Authorization code from Google
        state: State parameter for CSRF protection
        response_format: Format of the response, either 'redirect' (default) or 'json'

    Returns:
        Redirect to frontend with success/error indication or JSON response with user info
    """
    state_data = await get_oauth_state(state, state_storage)

    if not state_data:
        logger.warning(f"Invalid OAuth state in callback: {state}")
        if response_format == "json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")
        return RedirectResponse(
            url=f"/login?error=oauth_error&provider={OAuthProvider.GOOGLE.value}&reason=invalid_state",
            status_code=status.HTTP_302_FOUND,
        )

    provider_valid = False
    try:
        provider_valid = _is_provider_valid(state_data.provider, OAuthProvider.GOOGLE.value)
    except Exception as e:
        logger.warning(f"Error checking provider type: {e}")
        provider_valid = False

    if not provider_valid:
        expected = OAuthProvider.GOOGLE.value
        actual = getattr(state_data, "provider", "unknown")
        logger.warning(f"Provider mismatch in OAuth callback: expected {expected}, got {actual}")
        if response_format == "json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provider mismatch")
        return RedirectResponse(
            url=f"/login?error=oauth_error&provider={OAuthProvider.GOOGLE.value}&reason=provider_mismatch",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        token_data = await oauth_provider.exchange_code(code, code_verifier=state_data.code_verifier)

        token = OAuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            id_token=token_data.get("id_token"),
            refresh_token=token_data.get("refresh_token"),
            expires_in=token_data.get("expires_in"),
            scope=token_data.get("scope"),
        )

        user_info_raw = await oauth_provider.get_user_info(token.access_token)
        user_info = await oauth_provider.process_user_info(user_info_raw)

        user, is_new_user = await oauth_account_service.get_or_create_user(user_info, db)

        session_id, csrf_token = await session_manager.create_session(
            request=request,
            user_id=user["id"],
            metadata={
                "login_type": "oauth",
                "oauth_provider": OAuthProvider.GOOGLE.value,
                "username": user["username"],
                "is_new_user": is_new_user,
            },
        )

        session_manager.set_session_cookies(
            response=response,
            session_id=session_id,
            csrf_token=csrf_token,
            secure=settings.SESSION_SECURE_COOKIES,
            path="/",
        )

        await state_storage.delete(state)

        if response_format == "json":
            return {
                "success": True,
                "user": {"id": user["id"], "username": user["username"], "email": user["email"], "is_new_user": is_new_user},
                "csrf_token": csrf_token,
            }

        redirect_to = "/"
        try:
            if state_data.redirect_to:
                redirect_to = str(state_data.redirect_to)
        except Exception as e:
            logger.warning(f"Error getting redirect_to value: {e}, using default")

        return RedirectResponse(
            url=redirect_to,
            status_code=status.HTTP_302_FOUND,
        )

    except Exception as e:
        logger.error(f"Error in Google OAuth callback: {str(e)}", exc_info=True)

        if response_format == "json":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OAuth authentication failed: {str(e)}"
            )

        return RedirectResponse(
            url=f"/login?error=oauth_error&provider={OAuthProvider.GOOGLE.value}",
            status_code=status.HTTP_302_FOUND,
        )


@router.get("/check-auth")
async def check_auth(
    session_data: OptionalSessionDataDep,
    db: AsyncSessionDep,
) -> dict[str, Any]:
    """
    Check if the user is authenticated and return basic user information.

    This is useful for clients to verify authentication status and can be used
    with both cookie-based and API-based authentication.

    Args:
        session_data: The session data if the user is authenticated

    Returns:
        Authentication status and user information if authenticated
    """
    if not session_data:
        return {"authenticated": False, "message": "Not authenticated"}

    try:
        user = await crud_users.get(db=db, id=session_data.user_id)

        if not user:
            return {"authenticated": False, "message": "User not found"}

        return {
            "authenticated": True,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "oauth_provider": user.get("oauth_provider"),
            },
            "session": {
                "created_at": session_data.created_at.isoformat() if session_data.created_at else None,
                "last_activity": session_data.last_activity.isoformat() if session_data.last_activity else None,
            },
        }
    except Exception as e:
        logger.error(f"Error checking authentication: {str(e)}", exc_info=True)
        return {"authenticated": False, "message": "Error checking authentication status"}
