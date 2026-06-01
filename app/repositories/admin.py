from typing import Optional
from sqlalchemy import select
from app.models import Admin
from app.repositories.base import BaseRepository
from sqlalchemy.ext.asyncio import AsyncSession

class AdminRepository(BaseRepository[Admin]):
    """Repository for Admin database operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Admin)

    async def get_by_email(self, email: str) -> Optional[Admin]:
        """Get admin by email."""
        query = select(Admin).where(Admin.email == email)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_id(self, admin_id) -> Optional[Admin]:
        """Get admin by ID."""
        query = select(Admin).where(Admin.id == admin_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()