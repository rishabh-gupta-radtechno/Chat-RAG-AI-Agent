"""
RAG (Retrieval-Augmented Generation) pipeline.
"""

import uuid
from typing import Optional

from app.ai.llm import OllamaClient
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
        self._initialized = False

    async def initialize(self):
        """Initialize RAG pipeline."""
        if not self._initialized:
            await self.vector_db.initialize()
            self._initialized = True
            logger.info("RAG pipeline initialized")

    async def process_document(
        self,
        filepath: str,
        file_id: uuid.UUID,
        filename: str,
    ) -> int:
        """Process and embed a document."""
        try:
            # Extract text
            if filepath.endswith(".pdf"):
                text = self.text_processor.extract_text_from_pdf(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

            logger.info(f"Extracted {len(text)} characters from {filename}")

            # Clean text
            text = self.text_processor.clean_text(text)

            # Chunk text
            chunks = self.text_processor.chunk_by_sentences(text)
            logger.info(f"Created {len(chunks)} chunks")

            # Generate embeddings and upsert
            vectors = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self.llm_client.embed(chunk)

                    vectors.append({
                        "id": str(uuid.uuid5(file_id, str(i))),
                        "embedding": embedding,
                        "metadata": {
                            "file_id": str(file_id),
                            "filename": filename,
                            "chunk_index": i,
                            "chunk_text": chunk,
                            "chunk_size": len(chunk),
                        },
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
    ) -> list[dict]:
        """Retrieve relevant documents for a query."""
        try:
            if not self._initialized:
                await self.initialize()

            top_k = top_k or settings.vector_search_top_k

            # Generate query embedding
            query_embedding = await self.llm_client.embed(query)
            logger.info(f"Generated embedding for query: {query}")

            # Search vector database
            semantic_documents = await self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=settings.similarity_threshold,
            )
            keyword_documents = await self.vector_db.keyword_search(
                query=query,
                limit=top_k,
            )

            documents_by_id = {}
            for document in semantic_documents + keyword_documents:
                document_id = document.get("id")
                existing = documents_by_id.get(document_id)
                if (
                    existing is None
                    or document.get("relevance_score", 0.0) > existing.get("relevance_score", 0.0)
                ):
                    documents_by_id[document_id] = document

            documents = sorted(
                documents_by_id.values(),
                key=lambda document: document.get("relevance_score", 0.0),
                reverse=True,
            )[:top_k]

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    async def health_check(self) -> dict:
        """Check health of RAG components."""
        return {
            "vector_db": await self.vector_db.health_check(),
            "llm": await self.llm_client.health_check(),
        }
