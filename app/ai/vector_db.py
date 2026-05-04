"""
Vector database (Qdrant) client and operations.
"""

from typing import Optional

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete all vectors for a file."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "file_id",
                                "match": {"value": file_id},
                            }
                        ]
                    }
                },
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
