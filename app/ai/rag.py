"""
RAG (Retrieval-Augmented Generation) pipeline.
"""

import math
import os
import re
import time
import uuid
from typing import Optional

from app.ai.llm import OllamaClient
from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor
from app.ai.vector_db import VectorDBClient
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# RAGPipeline is constructed per request, so these are shared at module scope: the
# cross-encoder loads ONCE (not per query) and the BM25 index is built once and
# reused. _BM25_CACHE maps a user key -> {built_at, bm25, docs}; it is TTL-bounded
# and invalidated the moment a document is embedded (invalidate_bm25_cache).
_RERANKER_SINGLETON = None
_RERANKER_FAILED = False
_BM25_CACHE: dict = {}

# Format/versioning noise carried by uploaded filenames; these say nothing about what
# a document is about, and every file shares them, so they must not count as the
# question naming a particular document (see _filename_affinity).
_FILENAME_NOISE_TOKENS = frozenset({
    "pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx", "final", "rev", "ver",
    "version", "draft", "copy", "new", "old", "updated", "revised",
})


class RAGPipeline:
    """RAG pipeline for document retrieval and processing."""

    def __init__(self):
        self.vector_db = VectorDBClient()
        self.llm_client = OllamaClient()
        self.text_processor = TextProcessor()
        self.pdf_processor = PDFProcessor()
        self._bm25_corpus = []
        self._bm25_index = None
        self._reranker = None
        self._local_embed_model = None
        self._initialized = False

    async def initialize(self):
        """Initialize RAG pipeline."""
        if not self._initialized:
            await self.vector_db.initialize()
            if settings.enable_bm25_search:
                self._initialize_bm25()
            if settings.enable_reranking:
                self._initialize_reranker()
            self._initialized = True
            logger.info("RAG pipeline initialized")

    def _initialize_bm25(self):
        """Verify BM25 is available; the index itself is built lazily + cached."""
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
            logger.info("BM25 search enabled")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            settings.enable_bm25_search = False

    def _initialize_reranker(self):
        """Load the cross-encoder ONCE and share it across per-request pipelines."""
        global _RERANKER_SINGLETON, _RERANKER_FAILED
        if _RERANKER_FAILED:
            settings.enable_reranking = False
            return
        if _RERANKER_SINGLETON is not None:
            self._reranker = _RERANKER_SINGLETON
            return
        try:
            from sentence_transformers import CrossEncoder
            from app.core.device import torch_device
            _RERANKER_SINGLETON = CrossEncoder(
                'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device=torch_device()
            )
            self._reranker = _RERANKER_SINGLETON
            logger.info("Reranking enabled (cross-encoder loaded once, shared)")
        except ImportError:
            _RERANKER_FAILED = True
            logger.warning("sentence-transformers not installed, reranking disabled")
            settings.enable_reranking = False
        except Exception as exc:
            _RERANKER_FAILED = True
            logger.warning("Reranker failed to load (%s); reranking disabled", exc)
            settings.enable_reranking = False

    @staticmethod
    def invalidate_bm25_cache() -> None:
        """Drop the cached BM25 index so freshly-embedded content is searchable."""
        if _BM25_CACHE:
            _BM25_CACHE.clear()
            logger.info("BM25 cache invalidated (documents changed)")

    async def process_document(
        self,
        filepath: str,
        file_id: uuid.UUID,
        filename: str,
        user_id: Optional[uuid.UUID] = None,
    ) -> int:
        """Process and embed a document."""
        # Per-document progress, so the end-of-document summary (success OR
        # failure) reports exactly how far ingestion got and where it stopped.
        started_at = time.perf_counter()
        progress = {
            "stage": "starting",
            "pages": 0,
            "chunks": 0,
            "embed_failures": 0,
            "vectors": 0,
        }
        try:
            try:
                file_size = os.path.getsize(filepath)
            except OSError:
                file_size = -1
            logger.info(
                "Begin embedding doc=%s file_id=%s size_bytes=%s",
                filename, file_id, file_size,
            )
            if filepath.endswith(".pdf"):
                progress["stage"] = "pdf_extraction"
                page_documents = self.pdf_processor.extract_page_documents(filepath, file_id=str(file_id))
                progress["pages"] = len(page_documents)
                # Vision passes share one circuit breaker per document: once the
                # model fails repeatedly, both the table fallback and the figure
                # pass stop calling it for the rest of this document.
                self.llm_client.reset_vision_breaker()
                if settings.enable_vision_table_fallback:
                    progress["stage"] = "vision_table_fallback"
                    await self._refine_low_confidence_tables(filepath, page_documents, filename)
                if settings.enable_chart_extraction:
                    progress["stage"] = "figure_extraction"
                    await self._extract_figures(filepath, page_documents, file_id, filename)
                progress["stage"] = "chunking"
                chunks = self.text_processor.build_pdf_chunks(page_documents, filename)
                # Stages 1 & 2: exact + diagram/fuzzy dedup before embedding.
                chunks = self.text_processor.deduplicate_chunks(chunks)
                progress["chunks"] = len(chunks)
                logger.info(
                    "Extracted PDF page data for %s pages and created %s chunks",
                    len(page_documents),
                    len(chunks),
                )
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()
                text = self.text_processor.clean_text(raw)
                logger.info(f"Extracted {len(text)} characters from {filename}")
                chunks = [
                    {
                        "text": chunk,
                        "metadata": {
                            "file_name": filename,
                            "page_number": 0,
                            "content_type": "text",
                            "chunk_id": f"{filename}|text|{i:03d}",
                        },
                    }
                    for i, chunk in enumerate(self.text_processor.chunk_by_sentences(text))
                ]
                progress["chunks"] = len(chunks)
                logger.info(f"Created {len(chunks)} chunks")

            # Embed + upsert in page-windows, flushing after every N pages. All
            # chunk relationships (chart/table/diagram links, cross-content dedup,
            # boilerplate, sections) were already resolved document-wide above, so
            # windowing only controls WHEN vectors are written — never HOW chunks
            # relate. Benefits: peak memory is bounded to one window's embeddings
            # (not the whole document's), and each flush persists progress so a
            # crash part-way through embedding keeps the windows already saved.
            progress["stage"] = "embedding"
            # A chunk's several embedding parts (oversized text split by
            # _embed_text_with_splitting) share these across windows so chunk_index
            # stays globally monotonic and no chunk_id is emitted twice.
            seen_chunk_ids: set[str] = set()
            embed_state = {"next_index": 0}
            total_vectors = 0

            for win_start, win_end, window_chunks in self._iter_page_windows(chunks):
                if not window_chunks:
                    continue
                vectors = await self._embed_window(
                    window_chunks, file_id, filename, filepath, user_id,
                    seen_chunk_ids, embed_state, progress,
                )
                # Stage 3 dedup runs per window (near-identical chunks are almost
                # always on the same/adjacent pages, so they fall in one window).
                vectors = self._dedup_by_embedding(vectors)
                if vectors:
                    progress["stage"] = f"upsert(pages {win_start}-{win_end})"
                    await self.vector_db.upsert_vectors(vectors)
                    total_vectors += len(vectors)
                    progress["vectors"] = total_vectors
                    logger.info(
                        "Flushed page-window pages=%s-%s doc=%s window_vectors=%s total_vectors=%s",
                        win_start, win_end, filename, len(vectors), total_vectors,
                    )
                progress["stage"] = "embedding"

            # New content is now in the collection — drop the BM25 index so the
            # next query rebuilds it and can find this document.
            if total_vectors:
                self.invalidate_bm25_cache()

            progress["stage"] = "done"
            self._log_document_summary(
                filename, file_id, progress, started_at, error=None,
            )
            return total_vectors

        except Exception as e:
            logger.exception(
                "Error processing document doc=%s file_id=%s filepath=%s: %s",
                filename, file_id, filepath, e,
            )
            # End-of-document summary on the failure path: names the doc, the
            # stage it died in, and how much work was completed before it stopped
            # — so a stalled/failed sync is diagnosable from one line.
            self._log_document_summary(
                filename, file_id, progress, started_at, error=e,
            )
            raise

    @staticmethod
    def _iter_page_windows(chunks: list[dict]):
        """Yield ``(win_start, win_end, window_chunks)`` grouped by page-window.

        Windows are half-open page ranges of ``embed_page_batch_size`` pages
        ([1..N], [N+1..2N], …). Every chunk of a given page stays together in one
        window (whole pages are never split across windows), which keeps all
        same-page relationships — chart data/summary/image, table chunks, a
        diagram and its linked text — inside a single flush. Non-PDF chunks
        (page_number 0) fall into the first window. Chunk ORDER is preserved
        within each window so downstream indexing is stable.
        """
        batch_pages = max(1, settings.embed_page_batch_size)

        def page_of(chunk: dict) -> int:
            try:
                return int(chunk.get("metadata", {}).get("page_number", 0) or 0)
            except (TypeError, ValueError):
                return 0

        max_page = max((page_of(c) for c in chunks), default=0)
        for win_start in range(0, max_page + 1, batch_pages):
            win_end = win_start + batch_pages - 1  # inclusive, for logging
            window_chunks = [
                c for c in chunks
                if win_start <= page_of(c) <= win_end
            ]
            yield win_start, win_end, window_chunks

    async def _embed_window(
        self,
        chunks: list[dict],
        file_id,
        filename: str,
        filepath: str,
        user_id,
        seen_chunk_ids: set,
        embed_state: dict,
        progress: dict,
    ) -> list[dict]:
        """Embed one page-window's chunks into vector records (pre-upsert).

        A single source chunk may embed as several child vectors when it exceeds
        the embedding context window (see _embed_text_with_splitting) — each gets
        a suffixed chunk_id. ``seen_chunk_ids`` and ``embed_state['next_index']``
        are carried across windows so chunk ids stay unique and chunk_index stays
        globally monotonic for the whole document.
        """
        vectors: list[dict] = []
        for i, chunk in enumerate(chunks):
            base_meta = chunk.get("metadata", {}) or {}
            chunk_type = base_meta.get("content_type", "text")
            page_number = base_meta.get("page_number", 0)
            parent_chunk_id = base_meta.get("chunk_id") or (
                f"{filename}|page{page_number}|{chunk_type}|{i:03d}"
            )

            try:
                embedded = await self._embed_text_with_splitting(
                    chunk["text"],
                    doc_name=filename,
                    page_number=page_number,
                    chunk_type=chunk_type,
                )
            except Exception as e:
                # A genuine embedding failure (service down, etc.) — logged loudly
                # with full context. Context-length overflow never reaches here;
                # it is split and retried instead of skipped.
                logger.error(
                    "Failed to embed chunk %s (doc=%s page=%s type=%s id=%s): %s",
                    i, filename, page_number, chunk_type, parent_chunk_id, e,
                )
                progress["embed_failures"] += 1
                continue

            multi = len(embedded) > 1
            for part_no, (part_text, embedding) in enumerate(embedded, start=1):
                metadata = dict(base_meta)
                metadata["file_id"] = str(file_id)
                if user_id:
                    metadata["user_id"] = str(user_id)
                metadata["filename"] = filename
                metadata["filepath"] = filepath
                metadata["chunk_index"] = embed_state["next_index"]
                metadata["chunk_size"] = len(part_text)
                metadata["chunk_text"] = part_text

                chunk_id = parent_chunk_id if not multi else f"{parent_chunk_id}#p{part_no:02d}"
                metadata["chunk_id"] = chunk_id
                if multi:
                    metadata["parent_chunk_id"] = parent_chunk_id
                    metadata["split_part"] = part_no
                    metadata["split_total"] = len(embedded)

                if chunk_id in seen_chunk_ids:
                    logger.warning(f"Skipping duplicate chunk id while embedding: {chunk_id}")
                    continue
                seen_chunk_ids.add(chunk_id)
                embed_state["next_index"] += 1

                vectors.append({
                    "id": str(uuid.uuid5(file_id, chunk_id)),
                    "embedding": embedding,
                    "metadata": metadata,
                })
        return vectors

    def _warn_if_vision_just_tripped(
        self, was_disabled: bool, doc_name: str, page_number, context: str
    ) -> bool:
        """Emit a live WARNING the instant the vision circuit breaker trips.

        Returns the breaker's current disabled state so the caller can pass it back
        in as ``was_disabled`` next time, making this fire exactly once per document
        (at the moment it flips from enabled to disabled) instead of only surfacing
        in the end-of-document summary. ``context`` names the pass (table/figure).
        """
        is_disabled = getattr(self.llm_client, "_vision_disabled", False)
        if is_disabled and not was_disabled:
            failures = getattr(self.llm_client, "_vision_failures", 0)
            logger.warning(
                "Vision model DISABLED mid-document at doc=%s page=%s during %s after "
                "%s consecutive failures (model likely down/OOM, e.g. 500 'unexpected "
                "EOF'); remaining vision calls for this document are skipped.",
                doc_name, page_number, context, failures,
            )
        return is_disabled

    def _log_document_summary(
        self,
        filename: str,
        file_id,
        progress: dict,
        started_at: float,
        error: Optional[BaseException],
    ) -> None:
        """Emit one end-of-document line summarizing the whole embedding run.

        Logged on both success and failure. On failure it records the stage that
        was in progress when it stopped, so a sync that fails or stalls can be
        diagnosed from a single line (which PDF, how far it got, why it stopped).
        Also surfaces whether the per-document vision circuit breaker tripped
        (repeated 500 "unexpected EOF" = the local vision model OOM'd), which is
        the usual reason a large scanned manual degrades or stresses the host.
        """
        elapsed = time.perf_counter() - started_at
        vision_disabled = getattr(self.llm_client, "_vision_disabled", False)
        vision_failures = getattr(self.llm_client, "_vision_failures", 0)
        status = "FAILED" if error is not None else "OK"
        message = (
            "Embedding summary doc=%s file_id=%s status=%s stage=%s "
            "pages=%s chunks=%s vectors=%s embed_failures=%s "
            "vision_failures=%s vision_disabled=%s duration_seconds=%.1f"
        )
        args = (
            filename, file_id, status, progress.get("stage"),
            progress.get("pages"), progress.get("chunks"), progress.get("vectors"),
            progress.get("embed_failures"), vision_failures, vision_disabled,
            elapsed,
        )
        if error is not None:
            logger.error(
                message + " error=%s: %s",
                *args, type(error).__name__, error,
            )
        else:
            logger.info(message, *args)

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: Optional[uuid.UUID] = None,
        embed_queries: Optional[list[str]] = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query with hybrid search and reranking.

        embed_queries: the text(s) to embed for dense/vector search, independent of
        the lexical ``query``. Pass one string for an English question; pass BOTH the
        English translation and the original Hindi for a Devanagari question, so each
        is embedded and searched and the results merged (bge-m3 aligns the two
        languages, so the Hindi vector recovers what a lossy translation drops). When
        omitted, the lexical ``query`` is embedded.
        """
        try:
            if not self._initialized:
                await self.initialize()

            top_k = top_k or settings.vector_search_top_k

            # Embed each retrieval text (English translation and/or original Hindi),
            # de-duplicated so an English question — where translation is a no-op —
            # is not embedded twice.
            embed_texts = [t.strip() for t in (embed_queries or [query]) if t and t.strip()]
            if not embed_texts:
                embed_texts = [query]
            seen_embed_texts: set[str] = set()
            unique_embed_texts: list[str] = []
            for text in embed_texts:
                if text.lower() not in seen_embed_texts:
                    seen_embed_texts.add(text.lower())
                    unique_embed_texts.append(text)

            query_embeddings: list[list[float]] = []
            for text in unique_embed_texts:
                if settings.use_local_embeddings:
                    query_embeddings.append(self._embed_locally(text))
                else:
                    query_embeddings.append(await self.llm_client.embed(text))
            logger.info("Generated %d query embedding(s) for: %s", len(query_embeddings), unique_embed_texts)

            semantic_documents, bm25_documents, keyword_documents, retrieval_user_id = (
                await self._retrieve_candidates(
                    query=query,
                    query_embeddings=query_embeddings,
                    limit=top_k * 2,
                    user_id=user_id,
                )
            )

            # Section-scoped retrieval: when the query names a section (e.g.
            # "section 3.6"), pull that whole section by exact metadata match so
            # it is returned even if the bare reference has little to embed.
            section_documents = []
            section_ref = self._section_reference(query)
            if section_ref:
                section_documents = await self.vector_db.search_by_section(
                    section_ref,
                    limit=top_k * 2,
                    user_id=str(retrieval_user_id) if retrieval_user_id else None,
                )
                for document in section_documents:
                    document["section_match"] = True
                logger.info("Section '%s' filter matched %d chunk(s)", section_ref, len(section_documents))

            # Heading-scoped retrieval: when the query quotes an ALL-CAPS heading
            # ("MUST CHANGE ITEMS") but not a numbered/lettered section, pull the
            # chunks that carry that heading in uppercase — i.e. the slide/table it
            # names — so a value living in a check-sheet cell is retrieved even though
            # general manuals out-score it on the shared vocabulary. Section
            # references take precedence (they are more specific).
            heading_documents = []
            heading_ref = None if section_ref else self._heading_reference(query)
            if heading_ref:
                heading_documents = await self.vector_db.search_by_heading(
                    heading_ref,
                    limit=top_k * 2,
                    user_id=str(retrieval_user_id) if retrieval_user_id else None,
                )
                for document in heading_documents:
                    document["heading_match"] = True
                logger.info("Heading '%s' matched %d chunk(s)", heading_ref, len(heading_documents))

            # Put every channel on a common 0..1 scale before combining. Semantic
            # scores are cosine (0..1) but BM25 returns unbounded term-frequency
            # scores (tens to hundreds); feeding those raw into
            # _combined_retrieval_score let a single channel's magnitude dominate,
            # which pinned relevance to raw BM25 and made the reranker and every
            # bonus (content/section) numerically irrelevant. Scaling each channel to
            # its own max keeps within-channel ranking but makes cross-channel
            # combination and the bonuses meaningful.
            fused = self._fuse_channels(
                [semantic_documents, bm25_documents, keyword_documents]
            )

            # Combine and deduplicate. A chunk may surface in several channels and
            # also be a section/heading rescue; merge those views into ONE record
            # that keeps every signal, instead of keeping whichever copy happened to
            # score highest and discarding the rest.
            query_terms = self.vector_db._keyword_terms(query)
            documents_by_id = {}
            for document in (
                semantic_documents + bm25_documents + keyword_documents
                + section_documents + heading_documents
            ):
                # Diagram chunks (incl. scanned-page image pointers) are surfaced
                # via the dedicated diagram path, not as text context — keep them
                # out of text retrieval so they never occupy a context slot.
                if document.get("content_type") == "diagram":
                    continue
                # A table of contents / index page is navigation, never an answer.
                if self._is_index_chunk(document):
                    continue
                document_id = str(document.get("id"))
                # Rank-fusion score is a property of the chunk, not of the copy of it
                # that a given channel returned. It is kept in its own field so the
                # ranking score can be derived from it repeatedly without compounding:
                # relevance_score is an OUTPUT (written once at the end of retrieval),
                # never an input to the scoring.
                document["fusion_score"] = fused.get(document_id, 0.0)
                # Still published for downstream consumers (evidence gating); it is
                # deliberately NOT part of the ranking score — the keyword channel
                # already contributes its lexical judgement through the fusion, and
                # adding it again double-counted lexical overlap and let a chunk that
                # merely repeats the query's words outrank the one holding the answer.
                document["lexical_score"] = self.vector_db._keyword_score(
                    query_terms, document.get("chunk_text", "")
                )
                document["filename_affinity"] = self._filename_affinity(
                    document.get("filename", ""), query_terms
                )
                existing = documents_by_id.get(document_id)
                if existing is None:
                    documents_by_id[document_id] = document
                    continue
                for flag in ("section_match", "heading_match"):
                    if document.get(flag):
                        existing[flag] = True

            candidates = list(documents_by_id.values())

            # Rerank if enabled
            if settings.enable_reranking and self._reranker:
                candidates = self._rerank_documents(query, candidates, settings.rerank_top_k)
            else:
                candidates.sort(key=self._combined_retrieval_score, reverse=True)

            documents = await self._add_neighbor_context(
                candidates[:top_k],
                user_id=retrieval_user_id,
                max_documents=settings.rag_context_docs + max(top_k, 6),
                query=query,
            )

            # Publish the ranking score once retrieval has settled. Callers (source
            # ordering, evidence gating) read relevance_score, so it must reflect the
            # order documents are actually returned in — a rescued chunk is highly
            # relevant to the question even though rank fusion never saw it. Writing
            # it here rather than inside the scorer keeps the scorer a pure function
            # of fusion_score, so computing it twice can never compound.
            for document in documents:
                document["relevance_score"] = self._combined_retrieval_score(document)

            logger.info(f"Retrieved {len(documents)} documents after reranking")
            logger.info("Retrieval scores for query: %s", query[:120])
            for i, doc in enumerate(documents[:5], 1):
                logger.info(
                    "  #%d  score=%.3f  %s  page %s  (%s)",
                    i,
                    doc.get("relevance_score", 0.0),
                    doc.get("filename", "unknown"),
                    doc.get("page_number", "?"),
                    doc.get("content_type", "?"),
                )

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    async def _retrieve_candidates(
        self,
        query: str,
        query_embeddings: list[list[float]],
        limit: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> tuple[list[dict], list[dict], list[dict], Optional[uuid.UUID]]:
        """Retrieve scoped docs first, then fall back to shared/admin-ingested docs.

        Server manuals are commonly uploaded by an admin account, while end users
        chat from their own accounts. If the user-scoped Qdrant filter finds no
        points, an unscoped fallback lets those centrally uploaded manuals answer.

        ``query_embeddings`` may hold more than one vector (e.g. an English + Hindi
        pair); each is searched and the hits are merged into one semantic channel.
        """
        semantic_documents = await self._semantic_search(
            query_embeddings, limit, user_id=str(user_id) if user_id else None,
        )

        bm25_documents = []
        if settings.enable_bm25_search:
            bm25_documents = await self._bm25_search(query, limit, user_id=user_id)

        keyword_documents = await self.vector_db.keyword_search(
            query,
            limit,
            user_id=str(user_id) if user_id else None,
        )

        if not user_id or semantic_documents or bm25_documents or keyword_documents:
            return semantic_documents, bm25_documents, keyword_documents, user_id

        logger.info(
            "No user-scoped documents found for user %s; retrying retrieval without user filter",
            user_id,
        )
        semantic_documents = await self._semantic_search(
            query_embeddings, limit, user_id=None,
        )

        bm25_documents = []
        if settings.enable_bm25_search:
            bm25_documents = await self._bm25_search(query, limit, user_id=None)

        keyword_documents = await self.vector_db.keyword_search(
            query,
            limit,
            user_id=None,
        )

        for document in semantic_documents + bm25_documents + keyword_documents:
            document["retrieval_scope"] = "global_fallback"

        return semantic_documents, bm25_documents, keyword_documents, None

    async def _semantic_search(
        self,
        query_embeddings: list[list[float]],
        limit: int,
        user_id: Optional[str],
    ) -> list[dict]:
        """Dense search over one or more query vectors, merged into one ranking.

        With a single vector this is a plain Qdrant search. With several (the
        English + Hindi pair for a Devanagari question), each vector is searched
        independently and the hits are merged by document id, keeping the highest
        cosine any vector gave a chunk, then re-sorted so the merged list is a
        proper best-of ranking. Rank — not raw score — is what the downstream RRF
        fusion consumes, so this simply gives each chunk its best rank across the
        languages before fusion sees the channel.
        """
        if len(query_embeddings) == 1:
            return await self.vector_db.search(
                query_embedding=query_embeddings[0],
                top_k=limit,
                threshold=settings.similarity_threshold,
                user_id=user_id,
            )

        best_by_id: dict[str, dict] = {}
        for embedding in query_embeddings:
            documents = await self.vector_db.search(
                query_embedding=embedding,
                top_k=limit,
                threshold=settings.similarity_threshold,
                user_id=user_id,
            )
            for document in documents:
                document_id = str(document.get("id"))
                score = float(document.get("relevance_score") or 0.0)
                existing = best_by_id.get(document_id)
                if existing is None or score > float(existing.get("relevance_score") or 0.0):
                    best_by_id[document_id] = document

        merged = sorted(
            best_by_id.values(),
            key=lambda document: float(document.get("relevance_score") or 0.0),
            reverse=True,
        )
        return merged[:limit]

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        """Lowercased alphanumeric tokens for BM25.

        Better than str.split() for lookups: case-insensitive ('BOXNEL' matches
        'boxnel') and it breaks glued tokens like '70/85' into '70','85' so a
        query for a specific value hits the row that contains it.
        """
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    async def _get_bm25_index(self, user_id: Optional[uuid.UUID]):
        """Return a cached ``(bm25, docs)`` for the corpus, rebuilding only when the
        cache is stale (TTL) or was invalidated by a new document embedding."""
        from rank_bm25 import BM25Okapi

        key = str(user_id) if user_id else "__all__"
        entry = _BM25_CACHE.get(key)
        if entry and (time.time() - entry["built_at"]) < settings.bm25_cache_ttl_seconds:
            return entry["bm25"], entry["docs"]

        docs = await self.vector_db._get_all_documents(
            user_id=str(user_id) if user_id else None
        )
        # Diagram/chart-image pointers carry no lexical text worth indexing.
        docs = [d for d in docs if d.get("content_type") not in ("diagram", "chart_image")]
        tokenized = [self._bm25_tokenize(d.get("chunk_text", "")) for d in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        _BM25_CACHE[key] = {"built_at": time.time(), "bm25": bm25, "docs": docs}
        logger.info(
            "BM25 index built key=%s docs=%s (cached %ss)",
            key, len(docs), settings.bm25_cache_ttl_seconds,
        )
        return bm25, docs

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        """Perform BM25 search over the cached corpus index."""
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError:
            return []
        try:
            bm25, docs = await self._get_bm25_index(user_id)
            if bm25 is None or not docs:
                return []
            scores = bm25.get_scores(self._bm25_tokenize(query))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            return [
                {"id": docs[i]["id"], "relevance_score": float(scores[i]), **docs[i]}
                for i in top_indices if scores[i] > 0
            ]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    async def _add_neighbor_context(
        self,
        documents: list[dict],
        user_id: Optional[uuid.UUID],
        max_documents: int,
        query: str = "",
    ) -> list[dict]:
        """Interleave each retrieved chunk with the best chunks from its own page.

        A chunk frequently answers only half a question: a slide states the formula
        ("X = 79 + Y") while the value it needs sits in the table beside it on the
        same page. Such page-mates carry none of the query's wording, so they never
        rank on their own and must be pulled in by adjacency (same file, within
        +/-retrieval_neighbor_pages).

        Two rules make them actually usable:
          * They are emitted DIRECTLY AFTER the document they support, rather than
            appended after every candidate. Only the first rag_context_docs documents
            reach the LLM, and rag_context_docs (6) is smaller than the candidate
            count (top_k), so tail-appended neighbours were always truncated away —
            the pass scrolled the whole collection and then changed nothing.
          * Within a page they are ranked by lexical relevance to the query and capped
            at retrieval_neighbors_per_source, so the page-mate carrying the asked-for
            value wins the slot. They were previously ordered by file_id — an
            arbitrary UUID — which let an unrelated document's page-mates crowd out
            those of the top-ranked hit purely by alphabetical luck.

        Image-pointer chunks are excluded: they are surfaced via the diagram path and
        would only consume budget here.
        """
        if not documents or settings.retrieval_neighbor_pages <= 0:
            return documents

        per_source = max(0, settings.retrieval_neighbors_per_source)
        if per_source == 0:
            return documents

        all_docs = await self.vector_db._get_all_documents(
            user_id=str(user_id) if user_id else None
        )
        seen_ids = {str(doc.get("id")) for doc in documents if doc.get("id")}
        radius = settings.retrieval_neighbor_pages
        query_terms = self.vector_db._keyword_terms(query or "")

        # Each neighbour is claimed by the HIGHEST-RANKED document whose page it
        # adjoins, so it rides with the best match instead of being repeated.
        claimed: set[str] = set()
        neighbors_by_source: dict[int, list[dict]] = {}
        for index, source in enumerate(documents):
            file_id = str(source.get("file_id") or "")
            page = int(source.get("page_number") or 0)
            if not file_id or not page:
                continue

            bucket = []
            for candidate in all_docs:
                candidate_id = str(candidate.get("id"))
                if candidate_id in seen_ids or candidate_id in claimed:
                    continue
                if candidate.get("content_type") in ("diagram", "chart_image"):
                    continue
                if str(candidate.get("file_id") or "") != file_id:
                    continue
                candidate_page = int(candidate.get("page_number") or 0)
                if not candidate_page or abs(candidate_page - page) > radius:
                    continue
                if self._is_index_chunk(candidate):
                    continue
                bucket.append(candidate)

            if not bucket:
                continue
            bucket.sort(
                key=lambda doc: (
                    -self.vector_db._keyword_score(query_terms, doc.get("chunk_text", "")),
                    int(doc.get("page_number") or 0),
                    int(doc.get("chunk_index") or 0),
                )
            )
            selected = bucket[:per_source]
            for neighbor in selected:
                neighbor["neighbor_context"] = True
                neighbor.setdefault("relevance_score", 0.0)
                claimed.add(str(neighbor.get("id")))
            neighbors_by_source[index] = selected

        result: list[dict] = []
        for index, source in enumerate(documents):
            if len(result) >= max_documents:
                break
            result.append(source)
            for neighbor in neighbors_by_source.get(index, []):
                if len(result) >= max_documents:
                    break
                result.append(neighbor)

        return result

    def _rerank_documents(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Rerank documents using cross-encoder."""
        try:
            if not self._reranker or not documents:
                return documents

            pairs = [[query, doc.get("chunk_text", "")] for doc in documents]
            scores = self._reranker.predict(pairs)
            for i, score in enumerate(scores):
                raw_score = float(score)
                documents[i]["rerank_raw_score"] = raw_score
                documents[i]["rerank_score"] = 1.0 / (1.0 + math.exp(-raw_score))

            # relevance_score stays the fused 0..1 relevance of the chunk; the ranking
            # score (which adds the rescue bonuses) is derived, not stored over it, so
            # "how relevant is this chunk" and "where does it sort" remain separable —
            # overwriting the former with the latter made a rescued chunk look like a
            # 5.0-relevance match to every downstream consumer.
            documents.sort(key=self._combined_retrieval_score, reverse=True)
            return documents[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return documents

    @staticmethod
    def _is_index_chunk(document: dict) -> bool:
        """True for a table-of-contents / index / table-of-figures chunk.

        Such a chunk only lists section titles and the pages they live on — a
        navigation aid, never an answer — so it is excluded from retrieval context
        and sources. A ToC extracted as a TABLE is tagged page_type='content' at
        ingest (the "a page with a table is real content" rule), so the retrieval
        toc penalty misses it; this catches it by its 'PAGE' column instead.

        Detection is deliberately precise (an explicit contents/figures heading, or
        a PAGE column beside a description/section column) so genuine data tables —
        wear-limit and part-list tables that also carry section numbers — are kept.
        """
        if str(document.get("page_type") or "").lower() == "toc":
            return True
        text = document.get("chunk_text") or ""
        if re.search(r"table\s+of\s+contents|table\s+of\s+figures|list\s+of\s+(?:contents|figures)", text, re.IGNORECASE):
            return True
        if re.search(r"\bcontents\b[\s\W]{1,3}\bdescription\b", text, re.IGNORECASE):
            return True
        header = " ".join(str(h) for h in (document.get("table_header") or [])).lower()
        if re.search(r"\bpages?\b", header) and re.search(
            r"\b(description|contents|title|section|chapter|sr\.?\s*no|s\.?\s*no|sl\.?\s*no|particulars)\b",
            header,
        ):
            return True
        return False

    @staticmethod
    def _fuse_channels(channels: list[list[dict]]) -> dict[str, float]:
        """Reciprocal Rank Fusion of the retrieval channels -> ``{doc_id: 0..1}``.

        The channels score on incompatible scales — cosine is 0..1, BM25 is an
        unbounded term-frequency score, keyword is an ad-hoc 0..4 — so no arithmetic
        on the raw values is meaningful. Rescaling each channel to its own maximum
        does not fix it either: it pins every channel's best hit to exactly 1.0, so a
        spurious top keyword match ties a genuine top semantic match and some
        tie-breaker decides the answer.

        RRF uses only each channel's RANK, which is the one thing the channels agree
        on the meaning of, and sums 1/(k + rank). Its useful property is that
        agreement compounds: a chunk placed well by several channels beats a chunk
        that one channel loves and the others have never heard of — which is exactly
        the signal wanted from a hybrid search. ``k`` damps the head so ranks 1 and 2
        are not wildly far apart (60 is the standard value from the original paper).

        The result is normalised by the best score attainable given how many channels
        actually returned anything, so it stays absolute and comparable: 1.0 means
        "ranked first in every channel that ran", ~0.33 means "ranked first in one of
        three". Every other term in _combined_retrieval_score is anchored to this
        0..1 scale.
        """
        k = max(1, settings.rrf_k)
        active = [channel for channel in channels if channel]
        if not active:
            return {}

        scores: dict[str, float] = {}
        for channel in active:
            for rank, document in enumerate(channel, start=1):
                document_id = str(document.get("id"))
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)

        best_possible = len(active) * (1.0 / (k + 1))
        return {
            document_id: score / best_possible
            for document_id, score in scores.items()
        }

    @staticmethod
    def _filename_affinity(filename: str, query_terms: list[str]) -> float:
        """0..1 — how much of the document's own NAME the question names back.

        Manuals are near-duplicates of one another: five retrofitment decks carry a
        byte-identical "Section A" slide, and every BMBS manual repeats the same
        prose. When the text cannot separate them, the document the question actually
        asked for usually can ("the Twin Pipe modification" -> PPT_ON_Twin_Pipe).
        Chunk text never contains the filename, so no text-based channel can see this.

        Deliberately a weak term: it breaks ties between otherwise equal chunks and
        nudges the right manual up, but cannot by itself promote an irrelevant chunk.
        """
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", (filename or "").lower())
            if len(token) > 2 and token not in _FILENAME_NOISE_TOKENS
        }
        if not tokens:
            return 0.0
        return len(tokens & set(query_terms)) / len(tokens)

    @staticmethod
    def _combined_retrieval_score(document: dict) -> float:
        """Final ranking score. Every term lives on one documented 0..1-anchored
        scale, so the weights mean something and stay correct as channels change.

            relevance  0..1    rank-fusion, optionally blended with the reranker
          + content    0..0.2  mild preference between content types
          + filename   0..0.15 the question naming the document itself
          + rescue     0/1.5/2.0  an exact section/heading the user named
          - penalty    0..0.3  navigation pages

        Anything that is not a rescue therefore tops out at 1.35, which is what keeps
        the rescue bonuses decisive by construction rather than by a magnitude picked
        to win one example.
        """
        relevance = float(document.get("fusion_score") or 0.0)
        rerank_score = document.get("rerank_score")

        # The cross-encoder reads the query and the chunk together, so where it has
        # an opinion it is the better relevance estimate; fusion stays in the blend
        # as a prior, since the reranker sees only a truncated chunk and can be
        # confidently wrong. Previously this was max(fusion, rerank), which silently
        # discarded the reranker whenever fusion scored higher — and when raw BM25
        # values were still in this field, that was always.
        if rerank_score is not None:
            weight = min(max(settings.rerank_weight, 0.0), 1.0)
            relevance = (1.0 - weight) * relevance + weight * float(rerank_score)

        # A table is first-class evidence in these manuals, not a degraded copy of
        # prose: limits, ratings and part dimensions exist ONLY in tables, and a
        # retrieved table chunk is rendered to the model as its complete markdown
        # (see ReActAgent._format_context) rather than as the partial row text that
        # was embedded. Scoring it below prose contradicted that — it handed the slot
        # to a manual that merely discusses the topic over the table holding the
        # asked-for value. OCR text stays lower: it is the same content, read less
        # reliably.
        content_bonus = {
            "text": 0.2,
            "table": 0.2,
            "ocr": 0.1,
            "diagram": 0.0,
        }.get(document.get("content_type"), 0.0)

        page_type_penalty = {
            "toc": 0.3,
        }.get(document.get("page_type", "content"), 0.0)

        filename_bonus = 0.15 * float(document.get("filename_affinity") or 0.0)

        # A section/heading the user typed verbatim is an authoritative signal: the
        # answer lives on that exact slide/section, which often has almost nothing to
        # embed and cannot compete on relevance alone (a bare "Section A" slide, a
        # check-sheet cell). These rescues fire ONLY when the query names a section
        # ("Section A") or quotes an ALL-CAPS heading ("MUST CHANGE ITEMS"), so making
        # them decisive is safe — queries that reference neither never see them.
        # Both exceed the 1.2 non-rescue ceiling by construction, so they are
        # dominant by design rather than by a tuned magnitude; section (an exact
        # label) outranks heading (a phrase that may appear in several documents).
        section_bonus = 2.0 if document.get("section_match") else 0.0
        heading_bonus = 1.5 if document.get("heading_match") else 0.0

        return (
            relevance + content_bonus + filename_bonus
            + section_bonus + heading_bonus - page_type_penalty
        )

    async def _refine_low_confidence_tables(
        self, filepath: str, page_documents: list, filename: str = ""
    ) -> None:
        """Re-read only low-confidence tables with the local vision model (on-prem).

        Rules-based extraction (Docling + pdfplumber/camelot) handles most tables,
        but ragged/merged digital tables and scanned tables can come out mis-mapped.
        For those — and only those — render the page and ask the local vision model
        to extract the table as structured rows. Confidential pages never leave the
        host. Best-effort and gated by enable_vision_table_fallback.
        """
        try:
            from app.ai.table_extractor import TableExtractor
        except Exception as exc:
            logger.warning("Table extractor unavailable: %s", exc)
            return

        extractor = TableExtractor(self.llm_client)
        vision_was_disabled = getattr(self.llm_client, "_vision_disabled", False)
        for page in page_documents:
            tables = page.get("tables") or []
            if not tables:
                continue
            page_number = page.get("page_number")
            # Scanned pages (text came from OCR, not a native layer) are exactly the
            # case rules-based table mapping struggles with — treat their tables as
            # low-confidence too.
            page_is_ocr = (
                not (page.get("native_text") or "").strip()
                and bool((page.get("ocr_text") or "").strip())
            )
            # Only vision-refine tables that are (a) a REAL table — validate_table
            # rejects repeating header/footer boxes and prose-in-a-grid — AND (b)
            # low confidence. This stops the fallback firing on every page's header
            # box (which validate_table would reject downstream anyway).
            page_prose = page.get("native_text") or page.get("ocr_text") or ""
            flags = [
                self.text_processor.validate_table(t, page_prose)
                and (page_is_ocr or self.pdf_processor._table_is_low_confidence(t))
                for t in tables
            ]
            n_low = sum(1 for f in flags if f)
            if not n_low:
                continue
            logger.info("page=%s: %d low-confidence table(s); rendering for vision fallback", page_number, n_low)
            image = self.pdf_processor.render_page_png(filepath, page_number)
            if not image:
                logger.warning(
                    "doc=%s page=%s: render failed; vision table fallback skipped",
                    filename or filepath, page_number,
                )
                continue
            vision_tables = await extractor.analyze(
                image, doc_name=filename or filepath, page_number=page_number
            )
            vision_was_disabled = self._warn_if_vision_just_tripped(
                vision_was_disabled, filename or filepath, page_number, "table fallback"
            )
            logger.info("page=%s: vision table fallback returned %d table(s)", page_number, len(vision_tables))
            if not vision_tables:
                continue
            page["tables"] = self.pdf_processor._merge_vision_tables(tables, flags, vision_tables)
            logger.info("page=%s: low-confidence tables refined via vision", page_number)

    async def _extract_figures(
        self, filepath: str, page_documents: list, file_id, filename: str = ""
    ) -> None:
        """Vision-analyse figure regions on candidate pages (charts AND diagrams).

        One local vision call per region classifies and reads it: a data chart ->
        structured chart records; an engineering diagram/schematic (often a VECTOR
        drawing that get_images() can't see) -> a saved image + a written
        description. A region is a chart XOR a diagram. Best-effort and gated by
        enable_chart_extraction; any failure is logged and skipped so it never
        blocks ingestion. Fully on-prem.
        """
        try:
            from app.ai.chart_extractor import ChartExtractor
        except Exception as exc:
            logger.warning("Figure extractor unavailable: %s", exc)
            return

        # Geometry pre-pass over the PDF (vector drawings / raster / native text).
        candidates = self.pdf_processor.chart_candidate_pages(filepath)
        extractor = ChartExtractor(self.llm_client)
        vision_was_disabled = getattr(self.llm_client, "_vision_disabled", False)
        for page in page_documents:
            page_number = page.get("page_number")
            # A flattened slide deck has no text layer (pages arrive as OCR) and a
            # pie chart is only ~2 vector drawings, so the geometry pass misses it.
            # Fall back to the text the pipeline already OCR'd: chart slides are
            # dominated by %/number labels.
            page_text = f"{page.get('native_text') or ''} {page.get('ocr_text') or ''}"
            text_is_chart_like = self.pdf_processor._looks_like_chart_text(page_text)
            if page_number not in candidates and not text_is_chart_like:
                continue
            if page_number not in candidates:
                logger.info("page=%s figure-candidate (ocr_chart_like_text)", page_number)
            # If a raster diagram was already captured for this page, don't also save
            # a vision diagram for the same figure (avoid duplicate images).
            had_raster_diagram = bool(page.get("diagrams"))
            # One image per region (a page may hold several figures side-by-side).
            region_images = self.pdf_processor.chart_region_images(filepath, page_number)
            for region_index, image in enumerate(region_images, start=1):
                if not image:
                    continue
                result = await extractor.analyze_region(image)
                vision_was_disabled = self._warn_if_vision_just_tripped(
                    vision_was_disabled, filename or filepath, page_number, "figure extraction"
                )
                charts = result.get("charts") or []
                if charts:
                    image_url = self.pdf_processor._save_diagram_image(
                        file_id=str(file_id), page_number=page_number,
                        image_index=f"chart{region_index}", image_bytes=image,
                    )
                    for chart in charts:
                        chart["image_url"] = image_url
                        page.setdefault("charts", []).append(chart)
                        logger.info(
                            "page=%s region=%s chart extracted type=%s points=%s",
                            page_number, region_index, chart.get("chart_type"),
                            len(chart.get("structured_data", [])),
                        )
                    continue
                diagram = result.get("diagram")
                if diagram and not had_raster_diagram:
                    image_url = self.pdf_processor._save_diagram_image(
                        file_id=str(file_id), page_number=page_number,
                        image_index=f"figure{region_index}", image_bytes=image,
                    )
                    try:
                        labels = (self.pdf_processor._run_ocr(image) or "").strip()
                    except Exception:
                        labels = ""
                    page.setdefault("diagrams", []).append({
                        "image_index": f"figure{region_index}",
                        "description": diagram.get("description", ""),
                        "ocr_text": labels,
                        "image_url": image_url,
                    })
                    logger.info(
                        "page=%s region=%s diagram captured via vision description",
                        page_number, region_index,
                    )

    @staticmethod
    def _dedup_by_embedding(vectors: list[dict]) -> list[dict]:
        """Stage 3 dedup: drop chunks whose embeddings are near-identical.

        For any pair with cosine similarity >= threshold, the lower-priority
        content_type is dropped (text/table kept over ocr/diagram), so the same
        information embedded two ways is stored once.
        """
        if not settings.dedup_enabled or len(vectors) < 2:
            return vectors
        try:
            import numpy as np
        except ImportError:
            return vectors

        priority = TextProcessor._DEDUP_PRIORITY
        threshold = settings.dedup_embedding_threshold
        normed = []
        for v in vectors:
            arr = np.asarray(v["embedding"], dtype=float)
            norm = np.linalg.norm(arr)
            normed.append(arr / norm if norm else arr)

        keep = [True] * len(vectors)
        for i in range(len(vectors)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(vectors)):
                if not keep[j]:
                    continue
                if float(np.dot(normed[i], normed[j])) < threshold:
                    continue
                pi = priority.get(vectors[i]["metadata"].get("content_type"), 0)
                pj = priority.get(vectors[j]["metadata"].get("content_type"), 0)
                if pj <= pi:
                    keep[j] = False
                    RAGPipeline._link_dropped_image(vectors[i], vectors[j])
                else:
                    keep[i] = False
                    RAGPipeline._link_dropped_image(vectors[j], vectors[i])
                    break

        result = [v for v, k in zip(vectors, keep) if k]
        if len(result) != len(vectors):
            logger.info("Embedding dedup: %d -> %d vectors", len(vectors), len(result))
        return result

    @staticmethod
    def _link_dropped_image(keeper: dict, dropped: dict) -> None:
        """When an embedding-duplicate diagram vector is dropped in favor of a
        text/table vector, keep its image by linking it to the survivor."""
        dmeta = dropped.get("metadata", {})
        kmeta = keeper.get("metadata", {})
        if dmeta.get("content_type") != "diagram" or not dmeta.get("image_url"):
            return
        if kmeta.get("content_type") not in ("text", "table", "ocr"):
            return
        related = kmeta.setdefault("related_images", [])
        if any(r.get("chunk_id") == dmeta.get("chunk_id") for r in related):
            return
        related.append(
            {
                "chunk_id": dmeta.get("chunk_id"),
                "image_url": dmeta.get("image_url"),
                "image_index": dmeta.get("image_index"),
                "page_number": dmeta.get("page_number"),
            }
        )

    @staticmethod
    def _section_reference(query: str) -> Optional[str]:
        """Extract an explicit section reference from a query.

        e.g. "what is in section 3.6?" / "sec 3.6" / "clause 3.6" -> "3.6", and
        "the Section A slide" / "Section B scope" -> "A" / "B". Requires a section
        keyword so plain numbers/measurements aren't treated as section references.

        Alpha labels are matched only for a capital single letter directly after
        "section" (case-sensitive on the letter), so ordinary prose like
        "section a valve is fitted" is not mistaken for a reference to section "A".
        """
        if not query:
            return None
        match = re.search(
            r"\b(?:section|sec|clause|para(?:graph)?|article|point)\s*\.?\s*(\d+(?:\.\d+){0,3})\b",
            query,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        alpha = re.search(r"\b[Ss]ection\s+([A-Z])\b", query)
        return alpha.group(1) if alpha else None

    @staticmethod
    def _heading_reference(query: str) -> Optional[str]:
        """Extract an ALL-CAPS heading phrase the user quotes from a slide/table.

        e.g. "the MUST CHANGE ITEMS section of the presentation" -> "MUST CHANGE
        ITEMS". Presentations carry section/table headings in capitals, whereas
        manuals mention the same words in lowercase prose — so matching the UPPERCASE
        form of this phrase later selects the heading-bearing slide/table chunk that
        actually holds the answer (a value in a check-sheet cell) over the manuals
        that merely discuss the topic. Returns the longest run of >=2 consecutive
        fully-uppercase alphabetic tokens (each >=2 chars), or None. Requiring >=2
        tokens keeps lone acronyms (BMBS, POH) and ordinary words from triggering it.
        """
        if not query:
            return None
        best: list[str] = []
        current: list[str] = []
        for token in query.split():
            core = re.sub(r"[^A-Za-z]", "", token)
            if len(core) >= 2 and core.isupper():
                current.append(core)
            else:
                if len(current) > len(best):
                    best = current
                current = []
        if len(current) > len(best):
            best = current
        return " ".join(best) if len(best) >= 2 else None

    async def _embed(self, text: str) -> list[float]:
        """Embed one string via the configured backend (local ST model or Ollama).

        If local embeddings are requested but sentence-transformers is missing, we
        fall back to Ollama here (awaited correctly) rather than in the sync
        ``_embed_locally``, which cannot await the coroutine.
        """
        if settings.use_local_embeddings:
            try:
                return self._embed_locally(text)
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to Ollama")
        return await self.llm_client.embed(text)

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """True when an embedding failure is caused by the input exceeding context."""
        message = str(exc).lower()
        return "context length" in message or "exceeds the context" in message

    async def _embed_text_with_splitting(
        self,
        text: str,
        *,
        doc_name: str,
        page_number,
        chunk_type: str,
        depth: int = 0,
    ) -> list[tuple[str, list[float]]]:
        """Embed ``text``, splitting oversized content so nothing is ever dropped.

        Guarantees (for non-empty text):
          1. If the estimated token count exceeds the embedding budget, the text
             is split into sub-budget pieces *before* the first embed call.
          2. If an embed call still fails with a context-length error, the piece
             is split further and each part retried recursively, down to a minimum
             chunk size (below which we let Ollama's truncate=true store what fits).
          3. A non-context error is re-raised for the caller to log — it is never
             silently swallowed.

        Returns a list of ``(text, embedding)`` pairs (one per surviving child).
        Every split/retry is logged with doc/page/type/token metadata.
        """
        text = (text or "").strip()
        if not text:
            return []

        budget = TextProcessor.embed_token_budget()
        est = TextProcessor.estimate_tokens(text)

        # (1) Proactive split: too big by estimate — break it up before embedding.
        if est > budget and depth < settings.embed_split_max_depth:
            parts = TextProcessor.split_text_by_tokens(text, budget)
            if len(parts) > 1:
                logger.info(
                    "embed proactive-split doc=%s page=%s type=%s est_tokens=%s "
                    "budget=%s depth=%s -> %s child chunks",
                    doc_name, page_number, chunk_type, est, budget, depth, len(parts),
                )
                results: list[tuple[str, list[float]]] = []
                for part in parts:
                    results.extend(
                        await self._embed_text_with_splitting(
                            part, doc_name=doc_name, page_number=page_number,
                            chunk_type=chunk_type, depth=depth + 1,
                        )
                    )
                return results

        # Last-resort hard char cap (only bites a single pathological token/run).
        if len(text) > settings.embed_max_chars:
            logger.warning(
                "embed char-cap doc=%s page=%s type=%s chars=%s -> truncated to %s",
                doc_name, page_number, chunk_type, len(text), settings.embed_max_chars,
            )
            text = text[: settings.embed_max_chars]

        # (2) Try to embed; on a context-length error, split further and retry.
        try:
            embedding = await self._embed(text)
            return [(text, embedding)]
        except Exception as e:
            can_split = (
                self._is_context_length_error(e)
                and depth < settings.embed_split_max_depth
                and est > settings.embed_min_chunk_tokens
            )
            if not can_split:
                raise
            # Halve the *current* text's estimated size each retry (not the global
            # budget) so an under-estimate that slipped past the budget still
            # converges: a chunk that failed at N tokens is re-split toward N/2.
            smaller = max(TextProcessor.estimate_tokens(text) // 2, settings.embed_min_chunk_tokens)
            parts = TextProcessor.split_text_by_tokens(text, smaller)
            if len(parts) <= 1:
                # Cannot reduce further (single huge token); re-raise to be logged.
                raise
            logger.warning(
                "embed retry-split doc=%s page=%s type=%s est_tokens=%s "
                "new_budget=%s depth=%s err=%s -> %s child chunks",
                doc_name, page_number, chunk_type, est, smaller, depth, e, len(parts),
            )
            results = []
            for part in parts:
                results.extend(
                    await self._embed_text_with_splitting(
                        part, doc_name=doc_name, page_number=page_number,
                        chunk_type=chunk_type, depth=depth + 1,
                    )
                )
            return results

    def _embed_locally(self, text: str) -> list[float]:
        """Generate embeddings using the local sentence-transformers model.

        The model is loaded once and cached on the instance (it was previously
        rebuilt on every chunk, which is very slow). Raises ImportError when
        sentence-transformers is unavailable so the caller can fall back to Ollama.
        """
        if self._local_embed_model is None:
            from sentence_transformers import SentenceTransformer
            from app.core.device import torch_device
            self._local_embed_model = SentenceTransformer(
                settings.embedding_model_local, device=torch_device()
            )
        embedding = self._local_embed_model.encode(text)
        return embedding.tolist()

    async def health_check(self) -> dict:
        """Check health of RAG components."""
        return {
            "vector_db": await self.vector_db.health_check(),
            "llm": await self.llm_client.health_check(),
        }
