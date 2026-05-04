"""
Middleware for request/response handling.
"""

from typing import Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware:
    """Middleware for handling errors."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next: Callable):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"},
            )


class LoggingMiddleware:
    """Middleware for logging requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next: Callable):
        logger.info(f"{request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
