"""
Vector database (Qdrant) client and operations.
"""

import re
from typing import Optional

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorDBClient:
    """Qdrant vector database client."""

    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.collection_name = "documents"
        self.vector_size = settings.embedding_dimension

    async def initialize(self):
        """Initialize vector database with collection."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await self._create_collection()
            else:
                existing_size = await self._existing_vector_size()
                if existing_size is not None and existing_size != self.vector_size:
                    # A changed embedding model invalidates every stored vector.
                    # Recreate the collection so dimensions match the new model.
                    logger.warning(
                        "Collection %s has vector size %s but config expects %s; recreating it.",
                        self.collection_name,
                        existing_size,
                        self.vector_size,
                    )
                    await self.client.delete_collection(collection_name=self.collection_name)
                    await self._create_collection()
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            # repr() because timeout/connection exceptions often have an empty str()
            logger.exception(f"Error initializing vector database at {settings.qdrant_url}: {e!r}")
            raise

    async def _create_collection(self) -> None:
        logger.info(f"Creating collection: {self.collection_name} (size={self.vector_size})")
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection created: {self.collection_name}")

    async def _existing_vector_size(self) -> Optional[int]:
        """Return the configured vector size of the existing collection, if readable."""
        try:
            info = await self.client.get_collection(collection_name=self.collection_name)
            vectors = info.config.params.vectors
            # Single unnamed vector -> VectorParams; named vectors -> dict.
            if isinstance(vectors, dict):
                params = next(iter(vectors.values()), None)
                return getattr(params, "size", None)
            return getattr(vectors, "size", None)
        except Exception as exc:
            logger.warning("Could not read existing collection config: %r", exc)
            return None

    async def upsert_vectors(
        self,
        vectors: list[dict],
    ) -> bool:
        """Upsert vectors to the collection."""
        try:
            points_by_id: dict[str, PointStruct] = {}
            for v in vectors:
                point_id = v.get("id")
                if not point_id:
                    continue
                if point_id in points_by_id:
                    logger.warning(f"Duplicate vector id detected and skipped: {point_id}")
                    continue
                points_by_id[point_id] = PointStruct(
                    id=point_id,
                    vector=v["embedding"],
                    payload=v["metadata"],
                )

            points = list(points_by_id.values())
            if not points:
                logger.warning("No valid points to upsert")
                return True

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Upserted {len(points)} vectors")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.5,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Search for similar vectors."""
        try:
            query_filter = self._user_filter(user_id)
            if hasattr(self.client, "query_points"):
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

            documents = []
            for result in results:
                documents.append(
                    {
                        "id": result.id,
                        "relevance_score": result.score,
                        **(result.payload or {}),
                    }
                )

            logger.info(f"Found {len(documents)} similar documents")
            return documents

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    async def keyword_search(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Search payload text for exact keyword matches.

        This complements vector search for manuals where section titles and
        part names must match literally.
        """
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
                        results.append({
                            "id": point.id,
                            "relevance_score": score,
                            **payload,
                        })

                if offset is None:
                    break

            results.sort(key=lambda item: item["relevance_score"], reverse=True)
            logger.info(f"Found {len(results)} keyword matched documents")
            return results[:limit]

        except Exception as e:
            logger.error(f"Error keyword searching vectors: {e}")
            raise

    async def search_by_section(
        self,
        section_number: str,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Return all chunks whose section_number matches exactly (e.g. '3.6').

        Lets a query that names a section retrieve that whole section by metadata,
        independent of embedding/keyword similarity.
        """
        try:
            must = [
                FieldCondition(key="section_number", match=MatchValue(value=str(section_number)))
            ]
            if user_id:
                must.append(FieldCondition(key="user_id", match=MatchValue(value=str(user_id))))
            section_filter = Filter(must=must)

            results: list[dict] = []
            offset = None
            while True:
                points, offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    scroll_filter=section_filter,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    results.append({"id": point.id, "relevance_score": 0.0, **payload})
                if offset is None or len(results) >= limit:
                    break

            logger.info(f"Found {len(results)} chunks in section {section_number}")
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching section {section_number}: {e}")
            return []

    async def _get_all_documents(self, user_id: Optional[str] = None) -> list[dict]:
        """Return all point payloads for lightweight lexical retrieval."""
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
                documents.append({
                    "id": point.id,
                    **(point.payload or {}),
                })

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
        seen = set()
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

    @staticmethod
    def _user_filter(user_id: Optional[str]) -> Optional[Filter]:
        if not user_id:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=str(user_id)),
                )
            ]
        )

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

        words = normalized_text.split()
        phrase_count = normalized_text.count(query_phrase) if query_phrase else 0
        if phrase_count:
            score += min(0.5, phrase_count * 0.1)

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
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchValue(value=str(file_id)),
                    )
                ]
            )
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=file_filter),
                wait=True,
            )
            logger.info(f"Deleted vectors for file: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if vector database is healthy."""
        try:
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return False
