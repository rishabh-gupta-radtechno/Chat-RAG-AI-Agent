"""
FastAPI dependency injection functions.
"""

import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.utils.exceptions import UnauthorizedException
from app.utils.helpers import extract_user_id_from_token
from app.core.logging import get_logger  # Add this import

logger = get_logger(__name__)  # Add this line
security = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> uuid.UUID:
    """Get current user ID from JWT token."""
    if not credentials or not credentials.credentials:
        logger.warning("No credentials provided in Authorization header")  # Add log
        raise UnauthorizedException()

    token = credentials.credentials
    logger.info(f"Received token: {token[:20]}...")  # Log first 20 chars for security

    user_id = extract_user_id_from_token(token)
    if not user_id:
        logger.error(f"Token validation failed for token: {token[:20]}...")  # Add log
        raise UnauthorizedException()

    logger.info(f"Authenticated user ID: {user_id}")  # Add log
    return user_id