"""
User repository for database access.
"""

from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """User repository with additional queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def email_exists(self, email: str) -> bool:
        """Check if email exists."""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first() is not None

    async def get_active_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get active users."""
        stmt = select(User).where(User.is_active == True).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
