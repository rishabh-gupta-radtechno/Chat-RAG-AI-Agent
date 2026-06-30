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

    # Hardware acceleration. "auto" uses CUDA when it is actually available, else
    # CPU; "cpu" forces CPU even on a GPU host; "cuda" prefers GPU but falls back
    # to CPU (with a warning) if none is found. Default keeps CPU-only hosts working.
    device: str = "auto"  # auto | cpu | cuda

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chat_rag_db"
    sync_database_url: str = "postgresql://postgres:postgres@localhost:5432/chat_rag_db"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # JWT
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    refresh_token_expire_days: int = 1

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:8b"
    ollama_embedding_model: str = "bge-m3"
    ollama_embeddings_path: str = "/api/embeddings"
    ollama_timeout_seconds: int = 120
    ollama_num_ctx: int = 4096  # must be larger than the largest prompt sent
    # Cap response length. -1 = unbounded, which on CPU lets a reasoning model run
    # for thousands of tokens and blow past the request timeout; bound it instead.
    ollama_num_predict: int = 1024
    # qwen3 / deepseek-r1 etc. emit a hidden <think> reasoning block before the
    # answer — slow on CPU. False disables it for much faster RAG answers (the
    # visible answer is unchanged). Only set True for a thinking model when you
    # actually want the reasoning, and never for a non-thinking model.
    ollama_think: bool = False

    # File Upload
    upload_dir: str = "static/uploads"
    max_upload_size: int = 314572800  # 300MB

    # Embedding
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    chunk_size: int = 1024
    chunk_overlap: int = 128
    # Mid-size, section-scoped chunks: small enough for precise retrieval,
    # large enough to keep a procedure/step coherent.
    pdf_chunk_size: int = 250
    pdf_chunk_overlap: int = 20
    # Max table rows per chunk; large tables split into batches, header repeated.
    table_rows_per_chunk: int = 15
    # Reject an "extracted table" whose text is >= this % similar to the page's
    # prose — it's a false-positive table (paragraphs reformatted into cells).
    table_vs_text_similarity_threshold: int = 85

    # Chunk deduplication (run at ingest, before/after embedding)
    dedup_enabled: bool = True
    dedup_fuzzy_threshold: float = 0.90        # token-Jaccard near-duplicate
    dedup_diagram_suppression_threshold: float = 0.85  # diagram text covered by text/table
    dedup_embedding_threshold: float = 0.95    # cosine similarity of embeddings

    # PDF Processing
    use_docling: bool = True  # Use Docling for advanced PDF processing
    # Let Docling OCR pages itself (needed to recover tables from SCANNED PDFs
    # via TableFormer). Off by default — it adds models/memory; enable only with
    # enough RAM (see the OCR out-of-memory notes).
    docling_do_ocr: bool = True
    # Docling page-render scale for TableFormer (1.0=72dpi). Raising it (e.g. 2.0)
    # can sharpen cell geometry on cleaner scans but costs memory; it does NOT fix
    # scans whose column labels are vertically offset (verified — no change there).
    docling_images_scale: float = 2.0
    # Heavier ACCURATE TableFormer model — better on complex tables, more memory.
    # Left off by default on this memory-constrained host; enable if RAM allows.
    docling_table_accurate: bool = True
    ocr_engine: str = "paddleocr"
    ocr_lang: str = "en"  # PaddleOCR language code; mapped to the Tesseract equivalent
    enable_diagram_captioning: bool = False
    # An embedded image covering >= this fraction of the page is treated as a real
    # diagram/engineering drawing (OCR'd + described); smaller ones are logos/icons.
    diagram_min_coverage: float = 0.15

    # Chart extraction (pie/bar/line) via a LOCAL Ollama vision model. Off by
    # default — needs the vision model pulled and adds memory; keeps data on-prem.
    enable_chart_extraction: bool = True
    # Local Ollama vision model used for BOTH charts and the table fallback. 32B
    # reads dense labels/cells far better than 7B but needs much more memory
    # (~20GB+); pull it first (`ollama pull qwen2.5vl:32b`). Drop to 7B if RAM-bound.
    chart_vision_model: str = "qwen2.5vl:7b"  # any Ollama vision model

    # On-prem vision fallback for tables: re-read ONLY low-confidence tables
    # (ragged/merged digital tables, or scanned pages) with the vision model.
    # Verifiable rules-based tables are kept as-is. Off-load stays on the host.
    enable_vision_table_fallback: bool = True
    # A rules-extracted table is "low confidence" if this fraction of cells are
    # empty, or this fraction of rows have an inconsistent column count.
    table_low_conf_empty_frac: float = 0.25
    table_low_conf_ragged_frac: float = 0.30
    # A page with at least this many vector drawing ops is a chart/figure candidate.
    # Kept low because a simple pie chart can be only ~2 drawing ops; other signals
    # (large raster, chart-like %/number text) also flag candidates.
    chart_candidate_min_drawings: int = 6
    # Render scale for chart images sent to the vision model (1.0 = 72 DPI).
    # Higher sharpens small pie/bar labels (better/more complete extraction) but
    # costs memory per page; raise to 3.0 if the host has RAM and labels are dense.
    chart_render_scale: float = 2.0
    ocr_confidence_threshold: float = 0.6
    ocr_full_page: bool = True
    ocr_full_page_min_text_chars: int = 80
    # 200 DPI keeps manual text/tables OCR-readable while roughly halving the
    # render's peak memory vs 300 DPI (raise it if the host has ample RAM).
    ocr_full_page_dpi: int = 300
    enable_bm25_search: bool = False
    enable_reranking: bool = False
    rerank_top_k: int = 10
    embedding_model_local: str = "paraphrase-multilingual-mpnet-base-v2"
    use_local_embeddings: bool = False

    # RAG
    vector_search_top_k: int = 5
    similarity_threshold: float = 0.5
    retrieval_neighbor_pages: int = 1
    rag_context_docs: int = 6
    rag_context_max_chars: int = 8000

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:4200",
        "http://radtech-001-site62.ntempurl.com",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
