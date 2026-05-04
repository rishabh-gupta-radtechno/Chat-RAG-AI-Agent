"""
Configuration management for the application.
Loads settings from environment variables and provides typed access.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    api_title: str = "Chat RAG AI Agent"
    api_version: str = "1.0.0"
    debug: bool = False
    env: str = "development"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chat_rag_db"
    sync_database_url: str = "postgresql://postgres:postgres@localhost:5432/chat_rag_db"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # JWT
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embeddings_path: str = "/api/embeddings"
   

    # File Upload
    upload_dir: str = "static/uploads"
    max_upload_size: int = 52428800  # 50MB

    # Embedding
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # RAG
    vector_search_top_k: int = 5
    similarity_threshold: float = 0.5

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
