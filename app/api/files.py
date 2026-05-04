"""
File upload and management API routes.
"""

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id
from app.db.database import get_db
from app.schemas import FileListResponse, FileUploadResponse, SyncEmbeddingsResponse
from app.services.file import FileService
from app.ai.rag import RAGPipeline
from app.ai.vector_db import VectorDBClient
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter(prefix="/files", tags=["Files"])
logger = get_logger(__name__)
settings = get_settings()


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Upload a file for RAG processing."""
    try:
        # Validate file size
        content = await file.read()
        if len(content) > settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed ({settings.max_upload_size} bytes)",
            )

        # Validate file type
        allowed_types = {".pdf", ".txt", ".docx"}
        file_extension = "." + file.filename.split(".")[-1].lower()
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_types)}",
            )

        # Save file
        file_service = FileService(session)
        file_response = await file_service.upload_file(
            user_id=user_id,
            filename=file.filename,
            file_content=content,
            file_type=file_extension.lstrip("."),
        )

        return file_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error uploading file",
        )


@router.get("/list", response_model=List[FileListResponse])
async def list_user_files(
    skip: int = 0,
    limit: int = 100,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """List files uploaded by current user."""
    try:
        file_service = FileService(session)
        files = await file_service.get_user_files(user_id, skip, limit)
        return files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing files",
        )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Delete a file."""
    try:
        from app.utils.helpers import validate_uuid

        file_uuid = validate_uuid(file_id)
        if not file_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file ID")

        file_service = FileService(session)
        file_obj = await file_service.get_file(file_uuid)
        if not file_obj or file_obj.uploaded_by != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        vector_db = VectorDBClient()
        await vector_db.delete_by_file_id(str(file_uuid))

        success = await file_service.delete_file(file_uuid, user_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        return {
            "status": "deleted",
            "file_id": file_id,
            "embeddings_deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting file",
        )


@router.post("/sync-embeddings/{file_id}", response_model=SyncEmbeddingsResponse)
async def sync_embeddings(
    file_id: str,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Sync embeddings for a file."""
    try:
        from app.utils.helpers import validate_uuid

        file_uuid = validate_uuid(file_id)
        if not file_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file ID")

        file_service = FileService(session)
        file_obj = await file_service.get_file(file_uuid)
        logger.info(f"file UUID: {file_uuid}, file object: {file_obj}") 
        if not file_obj or file_obj.uploaded_by != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        # Process embeddings
        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        await rag_pipeline.vector_db.delete_by_file_id(str(file_uuid))
        chunks_created = await rag_pipeline.process_document(
            filepath=file_obj.filepath,
            file_id=file_uuid,
            filename=file_obj.filename,
        )

        # Mark file as embedded
        await file_service.mark_as_embedded(file_uuid)

        return SyncEmbeddingsResponse(
            file_id=file_uuid,
            chunks_created=chunks_created,
            status="completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error syncing embeddings",
        )
