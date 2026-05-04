"""
Utility functions and helpers.
"""

import uuid
from typing import Optional

from jose import JWTError

from app.core.logging import get_logger
from app.core.security import decode_token

logger = get_logger(__name__)


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID."""
    return uuid.uuid4()


def validate_uuid(value: str) -> Optional[uuid.UUID]:
    """Validate and convert string to UUID."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return None


def extract_user_id_from_token(token: str) -> Optional[uuid.UUID]:
    """Extract user ID from JWT token."""
    try:
        payload = decode_token(token)
        user_id_str = payload.get("sub")
        if user_id_str:
            return validate_uuid(user_id_str)
        return None
    except JWTError as e:
        logger.error(f"Error decoding token: {e}")
        return None


def is_valid_email(email: str) -> bool:
    """Validate email format."""
    import re

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"
