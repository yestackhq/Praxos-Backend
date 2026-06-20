import bcrypt


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash.

    Performs secure password verification using bcrypt's built-in comparison
    function, which includes protection against timing attacks through
    constant-time comparison.

    Args:
        plain_password: The plaintext password to verify.
        hashed_password: The bcrypt hash to compare against.

    Returns:
        True if the password matches the hash, False otherwise.

    Note:
        This function uses bcrypt.checkpw() which:
        - Automatically handles salt extraction from the hash
        - Performs constant-time comparison to prevent timing attacks
        - Works with any valid bcrypt hash format
        - Is designed to be computationally expensive to prevent brute force

    Example:
        ```python
        # During user authentication
        stored_hash = user.password_hash
        entered_password = "user_entered_password"

        if await verify_password(entered_password, stored_hash):
            # Password is correct - authenticate user
            return authenticate_user(user)
        else:
            # Password is incorrect - deny access
            raise AuthenticationError("Invalid password")
        ```
    """
    verified: bool = bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    return verified


def get_password_hash(password: str) -> str:
    """Generate a secure bcrypt hash for a plaintext password.

    Creates a bcrypt hash using a randomly generated salt, providing
    strong password protection against rainbow table attacks and
    ensuring each password has a unique hash.

    Args:
        password: The plaintext password to hash.

    Returns:
        The bcrypt hash as a string, including the salt.

    Note:
        This function uses bcrypt.hashpw() with bcrypt.gensalt() which:
        - Generates a random salt for each password
        - Uses a default cost factor (rounds) appropriate for security
        - Produces hashes that are compatible with standard bcrypt libraries
        - Creates hashes that include the salt and cost parameters

    Example:
        ```python
        # During user registration
        plain_password = "user_new_password"
        hashed_password = get_password_hash(plain_password)

        # Store the hash in the database
        user = User(
            email="user@example.com",
            password_hash=hashed_password
        )
        await session.add(user)
        await session.commit()
        ```
    """
    hashed_password: bytes = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    decoded_password: str = hashed_password.decode()
    return decoded_password
