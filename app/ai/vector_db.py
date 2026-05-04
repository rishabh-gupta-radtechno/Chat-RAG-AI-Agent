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
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection_name = "documents"
        self.vector_size = settings.embedding_dimension

    async def initialize(self):
        """Initialize vector database with collection."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection created: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise

    async def upsert_vectors(
        self,
        vectors: list[dict],
    ) -> bool:
        """Upsert vectors to the collection."""
        try:
            points = [
                PointStruct(
                    id=v["id"],
                    vector=v["embedding"],
                    payload=v["metadata"],
                )
                for v in vectors
            ]

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
    ) -> list[dict]:
        """Search for similar vectors."""
        try:
            if hasattr(self.client, "query_points"):
                response = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    with_payload=True,
                )
                results = response.points
            else:
                results = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
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
