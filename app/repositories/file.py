"""
File repository for database access.
"""
from datetime import datetime, time, timezone

from typing import Optional, List
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import File
from app.repositories.base import BaseRepository


class FileRepository(BaseRepository[File]):
    """File repository with additional queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, File)

    async def count_total(self) -> int:
        """Count total active knowledge files."""
        stmt = select(func.count(File.id)).where(File.is_active == True)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def count_synced(self) -> int:
        """Count total active files that are synced (embedded)."""
        stmt = select(func.count(File.id)).where(
            (File.is_active == True) & (File.is_embedded == True)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_by_user(
        self, 
        user_id: uuid.UUID, 
        skip: int = 0, 
        limit: int = 100,
        filename: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[File]:
        """Get files uploaded by a user with optional filtering."""
        query = select(File).where(File.uploaded_by == user_id, File.is_active == True)

        if filename:
            query = query.where(File.filename.ilike(f"%{filename}%"))
        
        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            start_date = datetime.combine(start_date.date(), time.min, tzinfo=start_date.tzinfo)
            query = query.where(File.created_at >= start_date)
            
        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            end_date = datetime.combine(end_date.date(), time.max, tzinfo=end_date.tzinfo)
            query = query.where(File.created_at <= end_date)

        stmt = query.order_by(File.created_at.desc()).offset(skip).limit(limit)
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

    async def update_file_status(self, file_id: uuid.UUID, file_status: bool) -> Optional[File]:
        """Update the file status flag."""
        return await self.update(file_id, file_status=file_status)

    async def get_disabled_file_ids(self) -> set[str]:
        """Return the ids of files whose status flag is disabled (file_status = False).

        Chunks belonging to these files must be completely ignored during chat
        retrieval, so callers use this set to drop matched chunks before they reach
        consolidation. Ids are returned as strings to compare directly against the
        ``file_id`` stored in each vector's payload.
        """
        stmt = select(File.id).where(File.file_status == False)  # noqa: E712
        result = await self.session.execute(stmt)
        return {str(file_id) for file_id in result.scalars().all()}
