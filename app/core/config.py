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
    # Upsert is sub-batched to this many points per request. A whole document (or
    # even one page-window) can be hundreds of points, each carrying full
    # chunk_text in its payload; sending them all in one request can overrun
    # Qdrant's payload/time limits and drop the connection (ResponseHandling
    # ReadError). Small, bounded requests avoid that and let a transient failure
    # retry cheaply.
    qdrant_upsert_batch_size: int = 64
    qdrant_timeout_seconds: float = 60.0    # per-request client timeout
    qdrant_upsert_max_retries: int = 3      # retry a dropped upsert this many times

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
    # Tighter cap for concise factual/lookup answers (single value, count, or one
    # table cell). Keeps a local model from mirroring the whole retrieved table
    # back when the question only wants one field.
    ollama_num_predict_concise: int = 256
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
    # Max table rows per chunk; large tables split into batches, header repeated
    # (no rows are dropped — a longer table just spans more chunks). Kept small so
    # a big reference/lookup table becomes several focused chunks instead of one
    # ~5KB block: this keeps each row's embedding signal from being diluted AND
    # keeps a chunk within the cross-encoder reranker's ~512-token window, so a
    # query for one specific row (e.g. a single wagon/publication) can actually
    # surface it. The token budget below is still the hard cap.
    table_rows_per_chunk: int = 6
    # Reject an "extracted table" whose text is >= this % similar to the page's
    # prose — it's a false-positive table (paragraphs reformatted into cells).
    table_vs_text_similarity_threshold: int = 85

    # Embedding context handling. bge-m3 accepts 8192 tokens; earlier the embed
    # request set no num_ctx, so Ollama applied its 2048 default and 500'd on any
    # longer chunk (the chunk was then silently skipped). We now (a) tell Ollama
    # the real window + truncate=true, (b) estimate tokens and split oversized
    # chunks before embedding, and (c) recursively split + retry if a chunk still
    # overflows — so no page content is ever dropped.
    embed_num_ctx: int = 8192               # embedding model context window (tokens)
    embed_safety_margin_tokens: int = 256   # reserve; split before the hard limit
    embed_min_chunk_tokens: int = 48        # stop recursive splitting below this
    embed_chars_per_token: float = 4.0      # heuristic divisor for token estimation
    embed_max_chars: int = 32000            # last-resort hard char cap before embed
    embed_split_max_depth: int = 12         # recursion guard for split-and-retry
    # Incremental upsert: after building + dedup'ing all chunks document-wide, we
    # embed and write them to the vector DB in page-windows of this many pages,
    # flushing after each window. Bounds peak memory (only one window's embeddings
    # are held at once instead of the whole document's) and persists progress
    # incrementally, so a crash part-way through embedding keeps the windows
    # already saved instead of discarding the entire document. Chunk relationships
    # are resolved document-wide before this, so windowing never splits them.
    embed_page_batch_size: int = 25
    # HF tokenizer id used for *exact* token counts when estimating chunk size.
    # Loaded lazily and cached; if it can't be fetched (offline / not installed)
    # we fall back to the char/word heuristic above. Empty string = heuristic only.
    embed_tokenizer_model: str = "BAAI/bge-m3"

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
    # Convert equations to LaTeX with Docling's formula-understanding (CodeFormula)
    # model so formulas reach retrieval as text instead of being flattened/garbled.
    # Downloads an extra model and adds ingest time + memory; disable with
    # DOCLING_FORMULA_ENRICHMENT=false on a RAM-constrained host.
    docling_formula_enrichment: bool = True
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
    # Circuit breaker: after this many consecutive vision-model failures within one
    # document, stop calling the vision model for the rest of that document (it's
    # down / OOM). Prevents a broken model from wasting minutes per page.
    vision_max_consecutive_failures: int = 2
    # Per-call timeout (seconds) for a single vision request, so one hung call can't
    # block ingestion. Generous — a large model on CPU is slow; a GPU host is fast.
    vision_timeout_seconds: int = 300
    # A rules-extracted table is "low confidence" if this fraction of cells are
    # empty, or this fraction of rows have an inconsistent column count.
    table_low_conf_empty_frac: float = 0.25
    table_low_conf_ragged_frac: float = 0.30
    # A page with at least this many vector drawing ops is a chart/figure candidate.
    # Kept low because a simple pie chart can be only ~2 drawing ops; other signals
    # (large raster, chart-like %/number text) also flag candidates.
    chart_candidate_min_drawings: int = 6
    # Render scale for chart/table images sent to the vision model (1.0 = 72 DPI).
    # Higher sharpens small pie/bar labels but produces MORE image tokens and costs
    # memory. At 2.0 a full page tokenized to ~4428 tokens and 400'd against the
    # 4096 vision context ("exceeds the available context size"); 1.5 keeps a page
    # to ~2500 tokens (fits with margin) and lowers per-call memory. Raise only if
    # the host has RAM AND the vision context (OLLAMA_NUM_CTX) is raised to match.
    chart_render_scale: float = 1.5
    ocr_confidence_threshold: float = 0.6
    ocr_full_page: bool = True
    ocr_full_page_min_text_chars: int = 80
    # 200 DPI keeps manual text/tables OCR-readable while roughly halving the
    # render's peak memory vs 300 DPI (raise it if the host has ample RAM).
    ocr_full_page_dpi: int = 300
    # Hybrid retrieval. Dense-only search misses exact-token lookups (a specific
    # part/wagon/code that is one row inside a big multi-row table, whose averaged
    # embedding ranks low). BM25 recalls those by exact tokens; the cross-encoder
    # reranker then reorders the merged candidates by true query relevance.
    enable_bm25_search: bool = True
    enable_reranking: bool = True
    # Retrieval funnel: cast a wide net, rerank a bounded pool, send a focused set
    # to the LLM. rerank_candidate_pool bounds how many of the best fused candidates
    # the cross-encoder actually scores (predictable latency on the heavier
    # bge-reranker, and matches "retrieve ~15-20, then rerank"); rerank_top_k is how
    # many it keeps; rag_context_docs (below) is how many finally reach the LLM.
    rerank_candidate_pool: int = 18
    rerank_top_k: int = 12
    # Cross-encoder reranker model. bge-reranker-v2-m3 is a strong MULTILINGUAL
    # reranker from the same BGE family as the bge-m3 embedder, so a Hindi query and
    # an English manual chunk are scored in one shared space — the right match for
    # this on-prem Hindi/English corpus. It is heavier than the previous
    # mmarco-mMiniLMv2 (XLM-RoBERTa-large, ~560M params), so on a CPU host it costs
    # more time/memory per query; drop to 'BAAI/bge-reranker-base' if RAM-bound.
    # Loaded via sentence-transformers CrossEncoder and downloaded once from HF on
    # first use (pre-pull on an offline host).
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Max query+chunk tokens the reranker scores at once. Kept at 512 to bound CPU
    # cost and keep parity with table_rows_per_chunk (chunks already fit this
    # window); bge-reranker-v2-m3 itself supports far longer input, so raise this on
    # a host with spare RAM if you also grow the chunk size.
    reranker_max_length: int = 512
    # Reciprocal Rank Fusion damping. Larger flattens the head (rank 1 vs 2 matter
    # less) and leans on cross-channel agreement; 60 is the standard published value.
    rrf_k: int = 60
    # How much the cross-encoder overrides rank fusion when reranking is enabled.
    # 0 = ignore the reranker, 1 = trust it alone. 0.5 keeps fusion as a prior so one
    # confidently-wrong rerank cannot bury a chunk every channel agreed on.
    rerank_weight: float = 0.5
    # BM25 index is built from the whole collection; cache it this long instead of
    # rebuilding per query. Invalidated immediately when a document is embedded, so
    # freshly-synced content is searchable right away (see invalidate_bm25_cache).
    bm25_cache_ttl_seconds: int = 600
    embedding_model_local: str = "paraphrase-multilingual-mpnet-base-v2"
    use_local_embeddings: bool = False

    # RAG
    # Dense candidate pool. Wider than before so the reranker (and BM25 merge) have
    # enough true candidates to surface a specific row from a big table.
    vector_search_top_k: int = 12
    similarity_threshold: float = 0.5
    retrieval_neighbor_pages: int = 1
    # How many same-page chunks ride along with each retrieved chunk. They are
    # emitted directly after the chunk they support, so they compete for the
    # rag_context_docs budget: 1 lets every result bring the page-mate holding the
    # value it references (a formula's table) while keeping several distinct sources
    # in context; raise it to favour depth on one page over breadth across manuals.
    retrieval_neighbors_per_source: int = 1
    # How many retrieved chunks reach the LLM (the final cut in _format_context).
    # These slots are shared with interleaved page-neighbours, and the total is still
    # bounded by rag_context_max_chars, so raising it lets more SMALL chunks in
    # without growing the token budget. 10 sits in the "best 8-12 to the LLM" band.
    rag_context_docs: int = 10
    rag_context_max_chars: int = 8000
    # After generation, verify the answer's words actually overlap the retrieved
    # context; if not, replace it with a "not found" response instead of returning
    # an ungrounded/hallucinated answer. Toggle off if it rejects good answers
    # (e.g. heavily numeric/tabular answers with little word overlap).
    enable_grounding_check: bool = True

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:4200",
        "http://radtech-001-site62.ntempurl.com",
        "http://10.143.8.189:8000",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
