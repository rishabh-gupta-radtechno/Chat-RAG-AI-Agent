"""
Vector database (Qdrant) client and operations.
Supports dense-only and hybrid dense+sparse (BGE-M3 / SPLADE) collections
with Reciprocal Rank Fusion (RRF) for multilingual retrieval.
"""

import re
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    Range,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Sparse vector dimension is token-vocabulary-sized for BGE-M3 (30522 tokens)
_SPARSE_VECTOR_NAME = "sparse"
_DENSE_VECTOR_NAME = "dense"


class VectorDBClient:
    """Qdrant vector database client with hybrid search support."""

    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection_name = "documents"
        # BGE-M3 uses 1024-dim dense vectors; legacy Ollama models use 768-dim.
        self.vector_size = 1024 if settings.use_bge_m3_embeddings else settings.embedding_dimension
        self._use_sparse = settings.use_bge_m3_embeddings or settings.enable_sparse_vectors

    async def initialize(self):
        """Initialize vector database collection."""
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await self._create_collection()
            else:
                logger.info("Collection already exists: %s", self.collection_name)
                await self._ensure_payload_indexes()

        except Exception as e:
            logger.error("Error initializing vector database: %s", e)
            raise

    async def _create_collection(self):
        """Create collection with HNSW tuning and optional sparse vector support."""
        logger.info("Creating collection: %s (sparse=%s)", self.collection_name, self._use_sparse)

        hnsw_config = HnswConfigDiff(
            m=32,               # Higher than default 16 — better recall for multilingual RAG
            ef_construct=200,   # Better index quality at ingestion time
            full_scan_threshold=10_000,
            max_indexing_threads=0,  # 0 = auto
            on_disk=False,
        )

        if self._use_sparse:
            # Hybrid collection: named dense vector + sparse vector
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=hnsw_config,
                    )
                },
                sparse_vectors_config={
                    _SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    default_segment_number=4,
                ),
                on_disk_payload=False,
            )
        else:
            # Dense-only collection (legacy / Ollama embeddings)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    hnsw_config=hnsw_config,
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    default_segment_number=4,
                ),
                on_disk_payload=False,
            )

        logger.info("Collection created: %s", self.collection_name)
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self):
        """Create payload field indexes for fast metadata filtering."""
        index_fields = [
            ("user_id", PayloadSchemaType.KEYWORD),
            ("file_id", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("script", PayloadSchemaType.KEYWORD),
            ("content_type", PayloadSchemaType.KEYWORD),
            ("has_table", PayloadSchemaType.BOOL),
            ("page_number", PayloadSchemaType.INTEGER),
            ("ocr_confidence", PayloadSchemaType.FLOAT),
        ]
        for field, schema_type in index_fields:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception:
                # Index may already exist — safe to ignore
                pass

    async def upsert_vectors(self, vectors: list[dict]) -> bool:
        """Upsert vectors (dense-only or dense+sparse) to the collection."""
        try:
            points_by_id: dict[str, PointStruct] = {}

            for v in vectors:
                point_id = v.get("id")
                if not point_id:
                    continue
                if point_id in points_by_id:
                    logger.warning("Duplicate vector id skipped: %s", point_id)
                    continue

                embedding = v["embedding"]
                sparse = v.get("sparse_embedding")  # dict with 'indices' and 'values'

                if self._use_sparse and sparse and isinstance(embedding, list):
                    # Hybrid point: named dense + sparse vectors
                    point = PointStruct(
                        id=point_id,
                        vector={
                            _DENSE_VECTOR_NAME: embedding,
                            _SPARSE_VECTOR_NAME: SparseVector(
                                indices=sparse["indices"],
                                values=sparse["values"],
                            ),
                        },
                        payload=v["metadata"],
                    )
                elif self._use_sparse and isinstance(embedding, list):
                    # Hybrid collection but no sparse provided — store dense only
                    point = PointStruct(
                        id=point_id,
                        vector={_DENSE_VECTOR_NAME: embedding},
                        payload=v["metadata"],
                    )
                else:
                    # Legacy dense-only collection
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=v["metadata"],
                    )

                points_by_id[point_id] = point

            points = list(points_by_id.values())
            if not points:
                logger.warning("No valid points to upsert")
                return True

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info("Upserted %d vectors", len(points))
            return True

        except Exception as e:
            logger.error("Error upserting vectors: %s", e)
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.5,
        user_id: Optional[str] = None,
        min_ocr_confidence: Optional[float] = None,
    ) -> list[dict]:
        """Dense vector similarity search."""
        try:
            query_filter = self._build_filter(
                user_id=user_id,
                min_ocr_confidence=min_ocr_confidence,
            )

            if self._use_sparse:
                # Named vector query for hybrid collections
                response = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    using=_DENSE_VECTOR_NAME,
                    limit=top_k,
                    score_threshold=threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )
                results = response.points
            elif hasattr(self.client, "query_points"):
                response = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )
                results = response.points
            else:
                results = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    query_filter=query_filter,
                )

            documents = [
                {"id": r.id, "relevance_score": r.score, **(r.payload or {})}
                for r in results
            ]
            logger.info("Dense search returned %d documents", len(documents))
            return documents

        except Exception as e:
            logger.error("Error searching vectors: %s", e)
            raise

    async def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict,
        top_k: int = 5,
        prefetch_k: int = 50,
        user_id: Optional[str] = None,
        min_ocr_confidence: Optional[float] = None,
    ) -> list[dict]:
        """
        Hybrid search using Qdrant's built-in Reciprocal Rank Fusion (RRF).
        Fuses dense semantic search + BGE-M3 sparse (lexical) search.
        Significantly improves recall for OCR-heavy and multilingual documents.
        """
        if not self._use_sparse:
            # Fall back to dense-only if collection doesn't have sparse vectors
            return await self.search(
                query_embedding=query_dense,
                top_k=top_k,
                user_id=user_id,
                min_ocr_confidence=min_ocr_confidence,
            )

        try:
            payload_filter = self._build_filter(
                user_id=user_id,
                min_ocr_confidence=min_ocr_confidence,
            )

            response = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    {
                        "query": query_dense,
                        "using": _DENSE_VECTOR_NAME,
                        "limit": prefetch_k,
                        "filter": payload_filter,
                    },
                    {
                        "query": SparseVector(
                            indices=query_sparse["indices"],
                            values=query_sparse["values"],
                        ),
                        "using": _SPARSE_VECTOR_NAME,
                        "limit": prefetch_k,
                        "filter": payload_filter,
                    },
                ],
                query={"fusion": "rrf"},
                limit=top_k,
                with_payload=True,
            )

            documents = [
                {"id": r.id, "relevance_score": r.score, **(r.payload or {})}
                for r in response.points
            ]
            logger.info("Hybrid RRF search returned %d documents", len(documents))
            return documents

        except Exception as e:
            logger.error("Error in hybrid search: %s", e)
            # Degrade gracefully to dense-only
            logger.warning("Falling back to dense-only search")
            return await self.search(
                query_embedding=query_dense,
                top_k=top_k,
                user_id=user_id,
                min_ocr_confidence=min_ocr_confidence,
            )

    async def keyword_search(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Search payload text for exact keyword matches (complements vector search)."""
        try:
            query_terms = self._keyword_terms(query)
            if not query_terms:
                return []

            results = []
            offset = None
            while True:
                points, offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    scroll_filter=self._user_filter(user_id),
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload or {}
                    chunk_text = payload.get("chunk_text") or ""
                    score = self._keyword_score(query_terms, chunk_text)
                    if score > 0:
                        results.append({"id": point.id, "relevance_score": score, **payload})

                if offset is None:
                    break

            results.sort(key=lambda item: item["relevance_score"], reverse=True)
            logger.info("Keyword search returned %d documents", len(results))
            return results[:limit]

        except Exception as e:
            logger.error("Error in keyword search: %s", e)
            raise

    async def _get_all_documents(self, user_id: Optional[str] = None) -> list[dict]:
        """Return all point payloads for BM25 / neighbor-context retrieval."""
        documents = []
        offset = None

        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                scroll_filter=self._user_filter(user_id),
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                documents.append({"id": point.id, **(point.payload or {})})

            if offset is None:
                break

        return documents

    async def get_diagrams_for_sources(
        self,
        sources: list[dict],
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Find diagram chunks from the same files/pages as retrieved text chunks."""
        source_pages = {
            (str(source.get("file_id")), source.get("page_number"))
            for source in sources
            if source.get("file_id") and source.get("page_number")
        }
        if not source_pages:
            return []

        diagrams = []
        seen: set = set()
        for document in await self._get_all_documents(user_id=user_id):
            if document.get("content_type") != "diagram":
                continue
            key = (str(document.get("file_id")), document.get("page_number"))
            if key not in source_pages:
                continue
            diagram_id = document.get("id")
            if diagram_id in seen:
                continue
            seen.add(diagram_id)
            diagrams.append(document)
            if len(diagrams) >= limit:
                break

        return diagrams

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _build_filter(
        user_id: Optional[str] = None,
        min_ocr_confidence: Optional[float] = None,
        language: Optional[str] = None,
    ) -> Optional[Filter]:
        must = []
        if user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=str(user_id))))
        if min_ocr_confidence is not None:
            must.append(FieldCondition(key="ocr_confidence", range=Range(gte=min_ocr_confidence)))
        if language:
            must.append(FieldCondition(key="language", match=MatchValue(value=language)))
        return Filter(must=must) if must else None

    @staticmethod
    def _user_filter(user_id: Optional[str]) -> Optional[Filter]:
        return VectorDBClient._build_filter(user_id=user_id)

    @staticmethod
    def _keyword_terms(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "for", "how", "in", "is", "of", "on",
            "or", "the", "to", "what", "when", "where", "which", "with",
        }
        return [
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) > 2 and term not in stop_words
        ]

    @staticmethod
    def _keyword_score(query_terms: list[str], text: str) -> float:
        raw_lines = [
            " ".join(re.findall(r"[a-z0-9]+", line.lower()))
            for line in text.splitlines()
            if line.strip()
        ]
        normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        if not normalized_text:
            return 0.0

        matched_terms = [term for term in query_terms if term in normalized_text]
        if not matched_terms:
            return 0.0

        score = len(matched_terms) / len(query_terms)
        query_phrase = " ".join(query_terms)
        if query_phrase in normalized_text:
            score += 0.75

        phrase_count = normalized_text.count(query_phrase) if query_phrase else 0
        if phrase_count:
            score += min(0.5, phrase_count * 0.1)

        words = normalized_text.split()
        first_words = " ".join(words[:24])
        if query_phrase and query_phrase in first_words:
            score += 0.5

        section_heading = re.search(
            rf"\b\d+(\.\d+)*\s+{re.escape(query_phrase)}\b",
            normalized_text,
        )
        if section_heading:
            score += 1.0

        for line in raw_lines:
            if re.fullmatch(rf"\d+(\.\d+)*\s+{re.escape(query_phrase)}", line):
                score += 2.0
                break
            if re.match(rf"\d+(\.\d+)*\s+{re.escape(query_phrase)}\s+\w+", line):
                score += 0.25
                break

        if "table of contents" in normalized_text or "table of figures" in normalized_text:
            score *= 0.45

        if len(words) < 20:
            score *= 0.5

        return score

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete all vectors for a file."""
        try:
            file_filter = Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=str(file_id)))]
            )
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=file_filter),
                wait=True,
            )
            logger.info("Deleted vectors for file: %s", file_id)
            return True
        except Exception as e:
            logger.error("Error deleting vectors: %s", e)
            raise

    async def health_check(self) -> bool:
        """Check if vector database is healthy."""
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error("Vector database health check failed: %s", e)
            return False
