"""
RAG (Retrieval-Augmented Generation) pipeline.
Supports multilingual retrieval with optional BGE-M3 dense+sparse embeddings
and cross-lingual reranking for Hindi/English/Devanagari documents.
"""

import math
import uuid
from typing import Optional

from app.ai.llm import OllamaClient
from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor, normalize_unicode
from app.ai.vector_db import VectorDBClient
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BGE_M3_Embedder:
    """
    Lazy-loaded BGE-M3 embedder producing dense + sparse vectors.
    Single model handles multilingual including Devanagari (Hindi/Marathi/Nepali).
    Activated via use_bge_m3_embeddings=True in settings.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                settings.bge_m3_model,
                use_fp16=True,
                device=settings.bge_m3_device,
            )
            logger.info("BGE-M3 model loaded: %s on %s", settings.bge_m3_model, settings.bge_m3_device)
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 embeddings. "
                "Install with: pip install FlagEmbedding"
            )

    def embed_batch(self, texts: list[str]) -> list[dict]:
        """Return list of {'dense': [...], 'sparse': {'indices': [...], 'values': [...]}}."""
        self._load()
        # Clean and normalize text before embedding
        clean_texts = [normalize_unicode(t) for t in texts]
        output = self._model.encode(
            clean_texts,
            batch_size=settings.bge_m3_batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results = []
        for i in range(len(clean_texts)):
            dense = output["dense_vecs"][i].tolist()
            lexical_weights = output["lexical_weights"][i]  # dict {token_id: weight}
            results.append({
                "dense": dense,
                "sparse": {
                    "indices": list(lexical_weights.keys()),
                    "values": [float(v) for v in lexical_weights.values()],
                },
            })
        return results

    def embed_single(self, text: str) -> dict:
        return self.embed_batch([text])[0]


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
        self._bge_m3: Optional[BGE_M3_Embedder] = None
        self._initialized = False

    async def initialize(self):
        """Initialize RAG pipeline."""
        if not self._initialized:
            await self.vector_db.initialize()
            if settings.enable_bm25_search:
                self._initialize_bm25()
            if settings.enable_reranking:
                self._initialize_reranker()
            if settings.use_bge_m3_embeddings:
                self._bge_m3 = BGE_M3_Embedder()
            self._initialized = True
            logger.info(
                "RAG pipeline initialized (bge_m3=%s, reranking=%s)",
                settings.use_bge_m3_embeddings,
                settings.enable_reranking,
            )

    def _initialize_bm25(self):
        try:
            import rank_bm25  # noqa: F401
            logger.info("BM25 search enabled")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            settings.enable_bm25_search = False

    def _initialize_reranker(self):
        """Initialize multilingual cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            # BAAI/bge-reranker-v2-m3 supports Hindi + English cross-lingual reranking
            self._reranker = CrossEncoder(settings.reranker_model)
            logger.info("Reranker loaded: %s", settings.reranker_model)
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            settings.enable_reranking = False
        except Exception as exc:
            logger.warning("Reranker load failed (%s): %s. Reranking disabled.", settings.reranker_model, exc)
            settings.enable_reranking = False

    # ------------------------------------------------------------------ embedding

    def _get_embedding(self, text: str) -> dict:
        """
        Produce embedding(s) for a single text.
        Returns dict with 'dense' (list[float]) and optionally 'sparse'.
        """
        text = normalize_unicode(text)

        if settings.use_bge_m3_embeddings and self._bge_m3:
            return self._bge_m3.embed_single(text)

        if settings.use_local_embeddings:
            dense = self._embed_locally(text)
        else:
            # Sync wrapper — called from async context; use await version below
            raise RuntimeError("Use _get_embedding_async for Ollama embeddings")

        return {"dense": dense, "sparse": None}

    async def _get_embedding_async(self, text: str) -> dict:
        """Async embedding dispatch."""
        text = normalize_unicode(text)

        if settings.use_bge_m3_embeddings and self._bge_m3:
            # BGE-M3 is CPU/GPU sync — run in thread pool in production;
            # for simplicity call directly here (fast on GPU, tolerable on CPU)
            return self._bge_m3.embed_single(text)

        if settings.use_local_embeddings:
            dense = self._embed_locally(text)
        else:
            dense = await self.llm_client.embed(text)

        return {"dense": dense, "sparse": None}

    async def _embed_batch_async(self, texts: list[str]) -> list[dict]:
        """Batch embedding for ingestion (much faster than one-by-one)."""
        if settings.use_bge_m3_embeddings and self._bge_m3:
            clean = [normalize_unicode(t) for t in texts]
            return self._bge_m3.embed_batch(clean)

        # Non-BGE path: embed one at a time (Ollama doesn't have a batch endpoint)
        results = []
        for text in texts:
            emb = await self._get_embedding_async(text)
            results.append(emb)
        return results

    # ------------------------------------------------------------------ ingestion

    async def process_document(
        self,
        filepath: str,
        file_id: uuid.UUID,
        filename: str,
        user_id: Optional[uuid.UUID] = None,
    ) -> int:
        """Process and embed a document."""
        try:
            if not self._initialized:
                await self.initialize()

            if filepath.endswith(".pdf"):
                page_documents = self.pdf_processor.extract_page_documents(
                    filepath, file_id=str(file_id)
                )
                chunks = self.text_processor.build_pdf_chunks(page_documents, filename)
                logger.info(
                    "Extracted %d pages → %d chunks from %s",
                    len(page_documents), len(chunks), filename,
                )
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                text = self.text_processor.clean_text(text)
                chunks = [
                    {
                        "text": chunk,
                        "metadata": {
                            "file_name": filename,
                            "page_number": 0,
                            "content_type": "text",
                            "ocr_confidence": 1.0,
                            "script": "latin",
                            "chunk_id": f"{filename}|text|{i:03d}",
                        },
                    }
                    for i, chunk in enumerate(self.text_processor.chunk_by_sentences(text))
                ]

            # Generate embeddings in batches
            batch_size = settings.bge_m3_batch_size
            vectors = []
            seen_chunk_ids: set[str] = set()

            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                texts = [c["text"] for c in batch]

                try:
                    embeddings = await self._embed_batch_async(texts)
                except Exception as e:
                    logger.warning("Batch embedding failed, falling back to individual: %s", e)
                    embeddings = []
                    for text in texts:
                        try:
                            embeddings.append(await self._get_embedding_async(text))
                        except Exception as inner_e:
                            logger.warning("Embedding failed for chunk: %s", inner_e)
                            embeddings.append({"dense": None, "sparse": None})

                for i, (chunk, emb) in enumerate(zip(batch, embeddings)):
                    if emb.get("dense") is None:
                        continue

                    metadata = chunk.get("metadata", {})
                    metadata["file_id"] = str(file_id)
                    if user_id:
                        metadata["user_id"] = str(user_id)
                    metadata["filename"] = filename
                    metadata["chunk_index"] = batch_start + i
                    metadata["chunk_size"] = len(chunk["text"])
                    metadata["chunk_text"] = chunk["text"]

                    # Propagate script/ocr_confidence defaults
                    metadata.setdefault("ocr_confidence", 1.0)
                    metadata.setdefault("script", "latin")
                    metadata.setdefault("has_table", False)

                    chunk_id = metadata.get("chunk_id") or (
                        f"{filename}|page{metadata.get('page_number', 0)}"
                        f"|{metadata.get('content_type', 'text')}|{batch_start + i:03d}"
                    )
                    metadata["chunk_id"] = chunk_id

                    if chunk_id in seen_chunk_ids:
                        logger.warning("Skipping duplicate chunk id: %s", chunk_id)
                        continue
                    seen_chunk_ids.add(chunk_id)

                    vectors.append({
                        "id": str(uuid.uuid5(file_id, chunk_id)),
                        "embedding": emb["dense"],
                        "sparse_embedding": emb.get("sparse"),
                        "metadata": metadata,
                    })

            if vectors:
                await self.vector_db.upsert_vectors(vectors)
                logger.info("Upserted %d vectors for %s", len(vectors), filename)

            return len(vectors)

        except Exception as e:
            logger.error("Error processing document: %s", e)
            raise

    # ------------------------------------------------------------------ retrieval

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        """Retrieve relevant documents with hybrid search and cross-lingual reranking."""
        try:
            if not self._initialized:
                await self.initialize()

            top_k = top_k or settings.vector_search_top_k
            min_ocr_conf = settings.min_ocr_confidence_for_retrieval

            # Generate query embedding
            query_emb = await self._get_embedding_async(query)
            query_dense = query_emb["dense"]
            query_sparse = query_emb.get("sparse")

            logger.info("Embedding generated for query (sparse=%s)", query_sparse is not None)

            # Primary search: hybrid RRF (dense + sparse) or dense-only
            if query_sparse and settings.use_bge_m3_embeddings:
                semantic_documents = await self.vector_db.hybrid_search(
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    top_k=top_k * 2,
                    prefetch_k=settings.hybrid_search_prefetch_k,
                    user_id=str(user_id) if user_id else None,
                    min_ocr_confidence=min_ocr_conf,
                )
            else:
                semantic_documents = await self.vector_db.search(
                    query_embedding=query_dense,
                    top_k=top_k * 2,
                    threshold=settings.similarity_threshold,
                    user_id=str(user_id) if user_id else None,
                    min_ocr_confidence=min_ocr_conf,
                )

            # BM25 search (optional, English-centric)
            bm25_documents = []
            if settings.enable_bm25_search:
                bm25_documents = await self._bm25_search(query, top_k * 2, user_id=user_id)

            # Keyword/lexical search
            keyword_documents = await self.vector_db.keyword_search(
                query,
                top_k * 2,
                user_id=str(user_id) if user_id else None,
            )

            # Merge and deduplicate
            documents_by_id: dict = {}
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

            # Cross-lingual reranking
            if settings.enable_reranking and self._reranker:
                candidates = self._rerank_documents(query, candidates, settings.rerank_top_k)
            else:
                candidates.sort(key=self._combined_retrieval_score, reverse=True)

            documents = await self._add_neighbor_context(
                candidates[:top_k],
                user_id=user_id,
                max_documents=settings.rag_context_docs + max(top_k, 6),
            )

            logger.info("Retrieved %d documents", len(documents))
            for doc in documents:
                logger.debug(
                    "Doc: %s page=%s score=%.3f script=%s ocr_conf=%.2f text=%s...",
                    doc.get("content_type"),
                    doc.get("page_number"),
                    doc.get("relevance_score", 0),
                    doc.get("script", "?"),
                    doc.get("ocr_confidence", 1.0),
                    doc.get("chunk_text", "")[:80],
                )

            return documents

        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            raise

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[dict]:
        try:
            from rank_bm25 import BM25Okapi
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
                {"id": all_docs[i]["id"], "relevance_score": scores[i], **all_docs[i]}
                for i in top_indices if scores[i] > 0
            ]
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
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
                candidate_file == fid and abs(candidate_page - pn) <= neighbor_radius
                for fid, pn in source_keys
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
        """Rerank with cross-encoder (multilingual bge-reranker-v2-m3)."""
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
            logger.warning("Reranking failed: %s", e)
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
            return model.encode(text).tolist()
        except ImportError:
            raise ImportError("sentence-transformers is required for local embeddings")

    async def health_check(self) -> dict:
        return {
            "vector_db": await self.vector_db.health_check(),
            "llm": await self.llm_client.health_check(),
        }
