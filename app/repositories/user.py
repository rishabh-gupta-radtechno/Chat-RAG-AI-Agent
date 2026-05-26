"""
User repository for database access.
"""
from datetime import datetime, time, timezone

from typing import Optional, List
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """User repository with additional queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID, excluding deleted ones."""
        stmt = select(User).where(User.id == user_id, User.is_deleted == False)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        # Ensure that only non-deleted users are returned by email lookup
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_mobile(self, mobile: int) -> Optional[User]:
        """Get user by mobile number."""
        stmt = select(User).where(User.mobile == mobile)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_all(self, skip: int = 0, limit: int = 100, name: Optional[str] = None, email: Optional[str] = None, department: Optional[str] = None, mobile: Optional[int] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[User]:
        """Get all users with optional filtering by name, email, and date, excluding deleted ones."""
        query = select(User).where(User.is_deleted == False)

        if name:
            query = query.where(User.name.ilike(f"%{name}%"))
        if email:
            query = query.where(User.email.ilike(f"%{email}%"))
        if department:
            query = query.where(User.department.ilike(f"%{department}%"))
        if mobile:
            query = query.where(User.mobile == mobile)
        if start_date:
            # Ensure timezone awareness and start of day
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            start_date = datetime.combine(start_date.date(), time.min, tzinfo=start_date.tzinfo)
            query = query.where(User.created_at >= start_date)
        if end_date:
            # Ensure timezone awareness and end of day
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            end_date = datetime.combine(end_date.date(), time.max, tzinfo=end_date.tzinfo)
            query = query.where(User.created_at <= end_date)

        query = query.offset(skip).limit(limit)

        stmt = query
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def email_exists(self, email: str) -> bool:
        """Check if email exists."""
        stmt = select(User).where(User.email == email, User.is_deleted == False)
        result = await self.session.execute(stmt)
        return result.scalars().first() is not None

    async def get_active_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get active users."""
        stmt = select(User).where(User.is_active == True, User.is_deleted == False).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
