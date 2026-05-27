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
    ollama_chat_model: str = "qwen2.5:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embeddings_path: str = "/api/embeddings"
    ollama_timeout_seconds: int = 120
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 512

    # File Upload
    upload_dir: str = "static/uploads"
    max_upload_size: int = 52428800  # 50MB

    # Embedding
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    chunk_size: int = 1024
    chunk_overlap: int = 128
    pdf_chunk_size: int = 450
    pdf_chunk_overlap: int = 80

    # BGE-M3 multilingual embeddings (dense + sparse, 1024-dim, cross-lingual)
    # Set use_bge_m3_embeddings=True to enable; requires FlagEmbedding installed.
    # When enabled, embedding_dimension is automatically treated as 1024.
    # A new Qdrant collection with sparse vector support will be created on first run.
    use_bge_m3_embeddings: bool = False
    bge_m3_model: str = "BAAI/bge-m3"
    bge_m3_device: str = "cpu"          # "cuda" for GPU acceleration
    bge_m3_batch_size: int = 16

    # Sparse vector / hybrid search (Qdrant RRF fusion)
    enable_sparse_vectors: bool = False  # Auto-enabled when use_bge_m3_embeddings=True
    hybrid_search_prefetch_k: int = 50  # Candidates per leg before RRF fusion

    # PDF Processing
    use_docling: bool = True
    ocr_engine: str = "paddleocr"
    # Comma-separated PaddleOCR language codes. "hi" covers Devanagari (Hindi/Marathi/Nepali)
    # and also recognises Latin/English text in the same pass.
    ocr_languages: str = "hi,en"
    enable_multilingual_ocr: bool = True
    enable_diagram_captioning: bool = False
    ocr_confidence_threshold: float = 0.6
    ocr_full_page: bool = True
    ocr_full_page_min_text_chars: int = 80
    ocr_full_page_dpi: int = 300
    enable_bm25_search: bool = False
    enable_reranking: bool = False
    rerank_top_k: int = 10
    # Upgraded to multilingual reranker (covers Hindi + English cross-lingual reranking)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_local_embeddings: bool = False

    # RAG
    vector_search_top_k: int = 5
    similarity_threshold: float = 0.5
    retrieval_neighbor_pages: int = 1
    rag_context_docs: int = 6
    rag_context_max_chars: int = 8000
    # Minimum OCR confidence for a chunk to be returned in retrieval results
    min_ocr_confidence_for_retrieval: float = 0.55

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
