"""
Logging configuration for the application.
"""

import logging
import logging.config
from app.core.config import get_settings


def setup_logging():
    """Configure logging for the application."""
    settings = get_settings()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "detailed": {
                "formatter": "detailed",
                "class": "logging.FileHandler",
                "filename": "app.log",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default", "detailed"],
        },
        "loggers": {
            "sqlalchemy.engine": {
                "level": "WARNING",
            },
            "uvicorn": {
                "level": "INFO",
            },
        },
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
