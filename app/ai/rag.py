"""
RAG (Retrieval-Augmented Generation) pipeline.
"""

import math
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
                page_documents = self.pdf_processor.extract_page_documents(filepath, file_id=str(file_id))
                chunks = self.text_processor.build_pdf_chunks(page_documents, filename)
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

            # Combine and deduplicate
            documents_by_id = {}
            for document in semantic_documents + bm25_documents + keyword_documents:
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
            for doc in documents:
                logger.debug(f"Doc: {doc.get('content_type')} page {doc.get('page_number')} score {doc.get('relevance_score', 0):.3f} text: {doc.get('chunk_text', '')[:100]}...")

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

        return (lexical_score * 2.0) + semantic_score + content_bonus

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
