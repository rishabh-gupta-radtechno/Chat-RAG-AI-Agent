"""
File upload and management service.
"""

import os
from typing import Optional
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import File
from app.repositories.file import FileRepository
from app.schemas import FileListResponse, FileUploadResponse

logger = get_logger(__name__)
settings = get_settings()


class FileService:
    """File management service."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.file_repo = FileRepository(session)

    async def upload_file(
        self,
        user_id: uuid.UUID,
        filename: str,
        file_content: bytes,
        file_type: str,
    ) -> FileUploadResponse:
        """Upload and save a file."""
        # Create upload directory if not exists
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = upload_dir / unique_filename

        # Save file
        file_size = len(file_content)
        with open(filepath, "wb") as f:
            f.write(file_content)

        logger.info(f"File saved: {filepath}")

        # Create database record
        file = await self.file_repo.create(
            filename=filename,
            filepath=str(filepath),
            file_size=file_size,
            file_type=file_type,
            uploaded_by=user_id,
        )
        await self.file_repo.commit()

        logger.info(f"File record created: {file.id}")

        return FileUploadResponse.model_validate(file)

    async def get_user_files(
        self, 
        user_id: uuid.UUID, 
        skip: int = 0, 
        limit: int = 100,
        filename: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[FileListResponse]:
        """Get files uploaded by user with optional filtering."""
        files = await self.file_repo.get_by_user(user_id, skip, limit, filename, start_date, end_date)
        return [FileListResponse.model_validate(f) for f in files]

    async def get_file(self, file_id: uuid.UUID) -> File | None:
        """Get file by ID."""
        return await self.file_repo.get_by_id(file_id)

    async def delete_file(self, file_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete file (soft delete)."""
        file = await self.file_repo.get_by_id(file_id)
        if not file or file.uploaded_by != user_id:
            return False

        # Soft delete
        updated_file = await self.file_repo.update(file_id, is_active=False)
        await self.file_repo.commit()

        logger.info(f"File deleted: {file_id}")

        return updated_file is not None

    async def read_file_content(self, filepath: str) -> str:
        """Read file content."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise

    async def get_non_embedded_files(self) -> list[File]:
        """Get files that need embedding."""
        return await self.file_repo.get_non_embedded_files()

    async def mark_as_embedded(self, file_id: uuid.UUID) -> bool:
        """Mark file as embedded."""
        updated_file = await self.file_repo.mark_as_embedded(file_id)
        await self.file_repo.commit()
        return updated_file is not None
