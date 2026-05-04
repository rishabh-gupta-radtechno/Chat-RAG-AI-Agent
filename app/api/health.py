"""
Health check and system status API routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.ai.rag import RAGPipeline
from app.schemas import HealthCheckResponse, DetailedHealthResponse
from app.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger(__name__)


@router.get("/", response_model=HealthCheckResponse)
async def health_check(session: AsyncSession = Depends(get_db)):
    """Check system health."""
    try:
        rag_pipeline = RAGPipeline()
        health = await rag_pipeline.health_check()

        status = "healthy" if all(health.values()) else "degraded"

        return HealthCheckResponse(
            status=status,
            database="healthy",
            vector_db="healthy" if health.get("vector_db") else "unhealthy",
            llm="healthy" if health.get("llm") else "unhealthy",
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            database="unhealthy",
            vector_db="unhealthy",
            llm="unhealthy",
        )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(session: AsyncSession = Depends(get_db)):
    """Get detailed health information."""
    try:
        errors = []
        rag_pipeline = RAGPipeline()
        health = await rag_pipeline.health_check()

        if not health.get("vector_db"):
            errors.append("Vector database is unavailable")
        if not health.get("llm"):
            errors.append("LLM service is unavailable")

        status = "healthy" if not errors else "degraded"

        return DetailedHealthResponse(
            status=status,
            database="healthy",
            vector_db="healthy" if health.get("vector_db") else "unhealthy",
            llm="healthy" if health.get("llm") else "unhealthy",
            errors=errors,
        )

    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        return DetailedHealthResponse(
            status="unhealthy",
            errors=[str(e)],
        )
