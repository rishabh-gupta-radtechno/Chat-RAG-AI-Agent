"""
RAG (Retrieval-Augmented Generation) pipeline.
"""

import asyncio
import math
import re
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
        """Initialize BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            # For now, BM25 will be built on retrieval if needed
            logger.info("BM25 search enabled")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            settings.enable_bm25_search = False

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            logger.info("Reranking enabled")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            settings.enable_reranking = False

    async def process_document(
        self,
        filepath: str,
        file_id: uuid.UUID,
        filename: str,
        user_id: Optional[uuid.UUID] = None,
    ) -> int:
        """Process and embed a document."""
        try:
            if filepath.endswith(".pdf"):
                # The extraction pipeline (Docling, PaddleOCR, OpenCV, PyMuPDF) is
                # synchronous and CPU-bound. Run it in a worker thread so it never
                # blocks the API event loop — these native libraries release the
                # GIL during their heavy work, so health checks and chat stay
                # responsive while a document ingests. Caller serializes ingestion
                # with a semaphore so this can't run for many docs at once.
                page_documents = await asyncio.to_thread(
                    self.pdf_processor.extract_page_documents, filepath, file_id=str(file_id)
                )
                if settings.enable_vision_table_fallback:
                    await self._refine_low_confidence_tables(filepath, page_documents)
                if settings.enable_chart_extraction:
                    await self._extract_figures(filepath, page_documents, file_id)
                chunks = await asyncio.to_thread(
                    self.text_processor.build_pdf_chunks, page_documents, filename
                )
                # Stages 1 & 2: exact + diagram/fuzzy dedup before embedding.
                chunks = await asyncio.to_thread(self.text_processor.deduplicate_chunks, chunks)
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
                logger.info(f"Created {len(chunks)} chunks")

            # Generate embeddings and upsert
            vectors = []
            seen_chunk_ids: set[str] = set()
            for i, chunk in enumerate(chunks):
                try:
                    if settings.use_local_embeddings:
                        embedding = self._embed_locally(chunk["text"])
                    else:
                        embedding = await self.llm_client.embed(chunk["text"])
                    metadata = chunk.get("metadata", {})
                    metadata["file_id"] = str(file_id)
                    if user_id:
                        metadata["user_id"] = str(user_id)
                    metadata["filename"] = filename
                    metadata["filepath"] = filepath
                    metadata["chunk_index"] = i
                    metadata["chunk_size"] = len(chunk["text"])
                    metadata["chunk_text"] = chunk["text"]

                    chunk_id = metadata.get("chunk_id")
                    if not chunk_id:
                        chunk_id = f"{filename}|page{metadata.get('page_number', 0)}|{metadata.get('content_type', 'text')}|{i:03d}"
                        metadata["chunk_id"] = chunk_id

                    if chunk_id in seen_chunk_ids:
                        logger.warning(f"Skipping duplicate chunk id while embedding: {chunk_id}")
                        continue
                    seen_chunk_ids.add(chunk_id)

                    vectors.append({
                        "id": str(uuid.uuid5(file_id, chunk_id)),
                        "embedding": embedding,
                        "metadata": metadata,
                    })
                except Exception as e:
                    logger.warning(f"Error embedding chunk {i}: {e}")
                    continue

            # Stage 3: drop near-identical chunks by embedding cosine similarity.
            # O(n^2) numpy pass — run off the event loop.
            vectors = await asyncio.to_thread(self._dedup_by_embedding, vectors)

            if vectors:
                await self.vector_db.upsert_vectors(vectors)
                logger.info(f"Upserted {len(vectors)} vectors")

            return len(vectors)

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: Optional[uuid.UUID] = None,
        embed_query: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query with hybrid search and reranking.

        embed_query: if provided, use this text for the vector embedding instead of
        the full query string. Pass the English translation for Hindi questions so the
        embedding is not diluted by Devanagari tokens.
        """
        try:
            if not self._initialized:
                await self.initialize()

            top_k = top_k or settings.vector_search_top_k

            # Use embed_query for vector embedding (English-only for Hindi input)
            embed_text = embed_query or query
            if settings.use_local_embeddings:
                query_embedding = self._embed_locally(embed_text)
            else:
                query_embedding = await self.llm_client.embed(embed_text)
            logger.info(f"Generated embedding for query: {embed_text}")

            semantic_documents, bm25_documents, keyword_documents, retrieval_user_id = (
                await self._retrieve_candidates(
                    query=query,
                    query_embedding=query_embedding,
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

            # Combine and deduplicate
            documents_by_id = {}
            for document in semantic_documents + bm25_documents + keyword_documents + section_documents:
                # Diagram chunks (incl. scanned-page image pointers) are surfaced
                # via the dedicated diagram path, not as text context — keep them
                # out of text retrieval so they never occupy a context slot.
                if document.get("content_type") == "diagram":
                    continue
                document_id = document.get("id")
                document["lexical_score"] = self.vector_db._keyword_score(
                    self.vector_db._keyword_terms(query),
                    document.get("chunk_text", ""),
                )
                existing = documents_by_id.get(document_id)
                if (
                    existing is None
                    or self._combined_retrieval_score(document) > self._combined_retrieval_score(existing)
                ):
                    documents_by_id[document_id] = document

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
            )

            logger.info(f"Retrieved {len(documents)} documents after reranking")
            logger.info("Retrieval scores for query: %s", embed_text[:120])
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
        query_embedding: list[float],
        limit: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> tuple[list[dict], list[dict], list[dict], Optional[uuid.UUID]]:
        """Retrieve scoped docs first, then fall back to shared/admin-ingested docs.

        Server manuals are commonly uploaded by an admin account, while end users
        chat from their own accounts. If the user-scoped Qdrant filter finds no
        points, an unscoped fallback lets those centrally uploaded manuals answer.
        """
        semantic_documents = await self.vector_db.search(
            query_embedding=query_embedding,
            top_k=limit,
            threshold=settings.similarity_threshold,
            user_id=str(user_id) if user_id else None,
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
        semantic_documents = await self.vector_db.search(
            query_embedding=query_embedding,
            top_k=limit,
            threshold=settings.similarity_threshold,
            user_id=None,
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

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        """Perform BM25 search on stored documents."""
        try:
            from rank_bm25 import BM25Okapi
            # For simplicity, build BM25 on all documents (in production, cache this)
            all_docs = await self.vector_db._get_all_documents(
                user_id=str(user_id) if user_id else None
            )
            corpus = [doc.get("chunk_text", "") for doc in all_docs]
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            return [
                {
                    "id": all_docs[i]["id"],
                    "relevance_score": scores[i],
                    **all_docs[i],
                }
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
    ) -> list[dict]:
        """Add nearby chunks from the same file/page to reduce missing-context answers."""
        if not documents or settings.retrieval_neighbor_pages <= 0:
            return documents

        all_docs = await self.vector_db._get_all_documents(
            user_id=str(user_id) if user_id else None
        )
        by_id = {str(doc.get("id")): doc for doc in documents if doc.get("id")}
        result = list(documents)

        source_keys = {
            (str(doc.get("file_id")), int(doc.get("page_number") or 0))
            for doc in documents
            if doc.get("file_id") and doc.get("page_number")
        }
        if not source_keys:
            return documents

        neighbor_radius = settings.retrieval_neighbor_pages
        neighbors = []
        for candidate in all_docs:
            candidate_id = str(candidate.get("id"))
            if candidate_id in by_id:
                continue
            candidate_file = str(candidate.get("file_id"))
            candidate_page = int(candidate.get("page_number") or 0)
            if not candidate_file or not candidate_page:
                continue
            if any(
                candidate_file == file_id and abs(candidate_page - page_number) <= neighbor_radius
                for file_id, page_number in source_keys
            ):
                candidate["neighbor_context"] = True
                candidate.setdefault("relevance_score", 0.0)
                neighbors.append(candidate)

        neighbors.sort(
            key=lambda doc: (
                str(doc.get("file_id")),
                int(doc.get("page_number") or 0),
                int(doc.get("chunk_index") or 0),
            )
        )

        for neighbor in neighbors:
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
                documents[i]["relevance_score"] = self._combined_retrieval_score(documents[i])

            documents.sort(key=lambda x: x["relevance_score"], reverse=True)
            return documents[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return documents

    @staticmethod
    def _combined_retrieval_score(document: dict) -> float:
        lexical_score = float(document.get("lexical_score") or 0.0)
        semantic_score = float(document.get("relevance_score") or 0.0)
        rerank_score = document.get("rerank_score")

        if rerank_score is not None:
            semantic_score = max(semantic_score, float(rerank_score))

        content_bonus = {
            "text": 0.2,
            "ocr": 0.1,
            "table": 0.05,
            "diagram": 0.0,
        }.get(document.get("content_type"), 0.0)

        page_type_penalty = {
            "toc": 0.3,
        }.get(document.get("page_type", "content"), 0.0)

        # An exact section-number match is the strongest signal for a query that
        # explicitly asks for a section, so float it to the top.
        section_bonus = 1.0 if document.get("section_match") else 0.0

        return (semantic_score * 2.0) + lexical_score + content_bonus + section_bonus - page_type_penalty

    async def _refine_low_confidence_tables(self, filepath: str, page_documents: list) -> None:
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
            flags = [
                page_is_ocr or self.pdf_processor._table_is_low_confidence(t)
                for t in tables
            ]
            n_low = sum(1 for f in flags if f)
            if not n_low:
                continue
            logger.info("page=%s: %d low-confidence table(s); rendering for vision fallback", page_number, n_low)
            image = await asyncio.to_thread(self.pdf_processor.render_page_png, filepath, page_number)
            if not image:
                logger.warning("page=%s: render failed; vision table fallback skipped", page_number)
                continue
            vision_tables = await extractor.analyze(image)
            logger.info("page=%s: vision table fallback returned %d table(s)", page_number, len(vision_tables))
            if not vision_tables:
                continue
            page["tables"] = self.pdf_processor._merge_vision_tables(tables, flags, vision_tables)
            logger.info("page=%s: low-confidence tables refined via vision", page_number)

    async def _extract_figures(self, filepath: str, page_documents: list, file_id) -> None:
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
        candidates = await asyncio.to_thread(self.pdf_processor.chart_candidate_pages, filepath)
        extractor = ChartExtractor(self.llm_client)
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
            region_images = await asyncio.to_thread(
                self.pdf_processor.chart_region_images, filepath, page_number
            )
            for region_index, image in enumerate(region_images, start=1):
                if not image:
                    continue
                result = await extractor.analyze_region(image)
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
                        labels = (await asyncio.to_thread(self.pdf_processor._run_ocr, image) or "").strip()
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
        """Extract an explicit section number from a query.

        e.g. "what is in section 3.6?" / "sec 3.6" / "clause 3.6" -> "3.6".
        Requires a section keyword so plain numbers/measurements aren't treated
        as section references.
        """
        if not query:
            return None
        match = re.search(
            r"\b(?:section|sec|clause|para(?:graph)?|article|point)\s*\.?\s*(\d+(?:\.\d+){0,3})\b",
            query,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _embed_locally(self, text: str) -> list[float]:
        """Generate embeddings using local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.embedding_model_local)
            embedding = model.encode(text)
            return embedding.tolist()
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to Ollama")
            return self.llm_client.embed(text)  # This is async, but for simplicity

    async def health_check(self) -> dict:
        """Check health of RAG components."""
        return {
            "vector_db": await self.vector_db.health_check(),
            "llm": await self.llm_client.health_check(),
        }
