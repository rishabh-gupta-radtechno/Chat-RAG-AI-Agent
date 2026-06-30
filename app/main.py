"""
Main FastAPI application.
"""

# Cap native thread pools BEFORE numpy / torch / OpenCV / Paddle are imported
# anywhere — those libraries read these env vars at import time, so setting them
# later has no effect. Without this, a single document ingestion fans out across
# every core (each lib defaults to all cores) and pegs the VPS, blocking health
# checks until the container is restarted. Values mirror
# settings.ingestion_max_threads; override the whole set via INGESTION_MAX_THREADS.
import os as _os

_INGESTION_THREADS = _os.environ.get("INGESTION_MAX_THREADS", "2")
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_NUM_THREADS",
):
    _os.environ.setdefault(_thread_var, _INGESTION_THREADS)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import admin, auth, chat, files, health, dashboard
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.db.database import create_all_tables
from app.ai.rag import RAGPipeline

# Setup logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting application...")

    try:
        # Create database tables
        await create_all_tables()
        logger.info("Database tables created")

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        logger.info("RAG pipeline initialized")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    import os
    if not os.path.exists(settings.upload_dir):
        os.makedirs(settings.upload_dir)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Include routers
    app.include_router(admin.router)
    app.include_router(auth.router)
    app.include_router(files.router)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(dashboard.router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Chat RAG AI Agent API",
            "version": settings.api_version,
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
