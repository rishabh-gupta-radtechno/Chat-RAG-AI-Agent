"""
File repository for database access.
"""

from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import File
from app.repositories.base import BaseRepository


class FileRepository(BaseRepository[File]):
    """File repository with additional queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, File)

    async def get_by_user(self, user_id: uuid.UUID, skip: int = 0, limit: int = 100) -> list[File]:
        """Get files uploaded by a user."""
        stmt = (
            select(File)
            .where(File.uploaded_by == user_id)
            .where(File.is_active == True)
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_active_files(self, skip: int = 0, limit: int = 100) -> list[File]:
        """Get all active files."""
        stmt = (
            select(File)
            .where(File.is_active == True)
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_non_embedded_files(self) -> list[File]:
        """Get files that haven't been embedded yet."""
        stmt = select(File).where(
            (File.is_active == True) & (File.is_embedded == False)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def mark_as_embedded(self, file_id: uuid.UUID) -> Optional[File]:
        """Mark file as embedded."""
        return await self.update(file_id, is_embedded=True)
