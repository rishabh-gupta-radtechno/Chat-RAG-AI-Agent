"""
RAG (Retrieval-Augmented Generation) pipeline.
"""
import collections

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
        self._bm25_tokenizer = None # Store tokenizer for BM25
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
            # We don't initialize BM25Okapi here, as it needs the corpus.
            # We'll dynamically build it in _bm25_search.
            # However, we can initialize a tokenizer if needed.
            self._bm25_tokenizer = lambda text: text.split() # Simple whitespace tokenizer for now
            logger.info("BM25 search enabled")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 search disabled")
            settings.enable_bm25_search = False

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker using BAAI bge-reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            reranker_model = getattr(settings, 'reranker_model', 'BAAI/bge-reranker-v2-m3')
            self._reranker = CrossEncoder(reranker_model)
            logger.info(f"Reranking enabled with model: {reranker_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            settings.enable_reranking = False
        except Exception as e:
            logger.warning(f"Failed to load reranker model {getattr(settings, 'reranker_model', 'BAAI/bge-reranker-v2-m3')}: {e}")
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
                    metadata = chunk.get("metadata", {})
                    # Prepend section title to chunk text before embedding for better context
                    if metadata.get("section_title"):
                        chunk["text"] = f"{metadata['section_title']}. {chunk['text']}"

                    if settings.use_local_embeddings:
                        embedding = self._embed_locally(chunk["text"])
                    else:
                        embedding = await self.llm_client.embed(chunk["text"])

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

            # Merge results using Reciprocal Rank Fusion (RRF)
            # Filter out diagrams from LLM context before RRF
            semantic_documents = [d for d in semantic_documents if d.get("content_type") != "diagram"]
            bm25_documents = [d for d in bm25_documents if d.get("content_type") != "diagram"]
            keyword_documents = [d for d in keyword_documents if d.get("content_type") != "diagram"]

            # Add lexical score to semantic documents for potential later use or logging
            for doc in semantic_documents:
                doc["lexical_score"] = self.vector_db._keyword_score(self.vector_db._keyword_terms(query), doc.get("chunk_text", ""), doc.get("metadata"))

            candidates = self._reciprocal_rank_fusion([
                semantic_documents,
                bm25_documents,
                keyword_documents,
            ], k=60) # RRF needs a larger pool before reranking/final selection

            # Rerank if enabled
            if settings.enable_reranking and self._reranker:
                candidates = self._rerank_documents(query, candidates, settings.rerank_top_k)
            else:
                candidates.sort(key=self._combined_retrieval_score, reverse=True)

            # Deduplicate and take top 20 candidates (or rerank_top_k if reranking was applied)
            final_candidates = self._deduplicate_documents(candidates[:getattr(settings, 'rerank_top_k', 20)]) # Default to 20

            documents = await self._add_neighbor_context( # Add neighbor context to the top candidates
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
                ) # Log the documents after neighbor context

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

        try:
            from rank_bm25 import BM25Okapi

            all_docs = await self.vector_db._get_all_documents(
                user_id=str(user_id) if user_id else None
            )

            if not all_docs:
                logger.warning("BM25: No documents found")
                return []

            valid_docs = []

            for doc in all_docs:
                text = (doc.get("chunk_text") or "").strip()

                if text:
                    valid_docs.append(doc)

            if not valid_docs:
                logger.warning("BM25: No valid chunks found")
                return []
            
            tokenized_corpus = [
                doc["chunk_text"].split()
                for doc in valid_docs
            ]

            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = self._bm25_tokenizer(query) if self._bm25_tokenizer else query.split()

            if not tokenized_query:
                return []

            scores = bm25.get_scores(tokenized_query)

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:limit]

            return [
                {
                    "id": valid_docs[i]["id"],
                    "relevance_score": float(scores[i]),
                    **valid_docs[i],
                }
                for i in top_indices
                if scores[i] > 0
            ]

        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Performs Reciprocal Rank Fusion (RRF) on multiple ranked lists of documents.
        
        Args:
            ranked_lists: A list of lists, where each inner list is a ranked list of documents.
                          Each document is a dictionary with an 'id' key.
            k: A constant that determines the impact of lower ranks.
        
        Returns:
            A single list of documents, ranked by their RRF score.
        """
        fused_scores = collections.defaultdict(float)
        document_map = {} # To store the full document object for each ID

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get("id")
                if doc_id is None:
                    continue
                
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
                
                # Store the document with the highest relevance_score if multiple sources
                # or simply the first one encountered if scores are not comparable
                if doc_id not in document_map or \
                   doc.get("relevance_score", 0.0) > document_map[doc_id].get("relevance_score", 0.0):
                    document_map[doc_id] = doc

        # Sort documents by their fused scores in descending order
        reranked_documents = []
        for doc_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
            doc = document_map[doc_id]
            doc["rrf_score"] = score # Add RRF score to metadata
            reranked_documents.append(doc)
            
        return reranked_documents

    @staticmethod
    def _deduplicate_documents(documents: list[dict]) -> list[dict]:
        """Removes duplicate documents based on their 'id'."""
        seen_ids = set()
        deduplicated = []
        for doc in documents:
            if doc.get("id") not in seen_ids:
                deduplicated.append(doc)
                seen_ids.add(doc.get("id"))
        return deduplicated

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
        """Combine scores from semantic, BM25, and reranking with configurable weights."""
        lexical_score = float(document.get("lexical_score") or 0.0)
        semantic_score = float(document.get("relevance_score") or 0.0)
        rerank_score = document.get("rerank_score")

        # If reranker was applied, use the reranked score as the primary semantic score
        if rerank_score is not None:
            semantic_score = max(semantic_score, float(rerank_score))

        # Apply configurable weights for semantic vs lexical (BM25) search
        semantic_weight = getattr(settings, 'semantic_weight', 0.6) # Default to 60% semantic
        bm25_weight = getattr(settings, 'bm25_weight', 0.4) # Default to 40% BM25

        # If RRF was applied, use its score as the primary combined score
        # RRF score already combines different retrieval methods, so it should be prioritized.
        if "rrf_score" in document:
            return document["rrf_score"]
        
        combined_score = (semantic_score * semantic_weight * 2.0) + (lexical_score * bm25_weight)

        content_bonus = {
            "text": 0.2,
            "ocr": 0.1,
            "table": 0.05,
            "diagram": 0.0,
        }.get(document.get("content_type"), 0.0)

        page_type_penalty = {
            "toc": 0.3, # Penalize Table of Contents
        }.get(document.get("page_type", "content"), 0.0)

        return combined_score + content_bonus - page_type_penalty

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

    @staticmethod
    def extract_entities_from_query(query: str) -> list[str]:
        """Extract important entities from question for grounding check.
        
        For manuals, named entities like VTA, C3W2, damper piston are critical signals
        that retrieval should find relevant content.
        """
        # Use a more sophisticated approach to extract entities,
        # focusing on technical terms, proper nouns, and potential part numbers.
        # This can be enhanced with NLP libraries like spaCy for better entity recognition.
        
        # For now, a regex-based approach:
        # 1. Uppercase words (potential proper nouns, acronyms, part numbers)
        # 2. Words with numbers and hyphens (common in part numbers)
        # 3. Quoted phrases (often specific terms)
        
        extracted_entities = []
        
        # Pattern for technical terms/part numbers (e.g., "VTA", "C3W2", "DAMPER-PISTON")
        extracted_entities.extend(re.findall(r'\b[A-Z0-9]+(?:[\-][A-Z0-9]+)*\b', query))
        
        # Pattern for quoted phrases
        extracted_entities.extend(re.findall(r'"([^"]*)"', query))
        extracted_entities.extend(re.findall(r"'([^']*)'", query))
        
        # Filter out common stop words and single characters, convert to lowercase for consistency
        return list(set([e.lower() for e in extracted_entities if len(e) > 1 and e.lower() not in TextProcessor._important_terms("")]))

    @staticmethod
    def validate_retrieval_grounding(query: str, documents: list[dict]) -> tuple[bool, str]:
        """Check if retrieved documents contain relevant entities from the query.
        
        Returns: (is_grounded, grounding_message)
        
        This prevents hallucination by ensuring retrieved context actually addresses
        the question before generating an answer.
        """
        if not documents or not query:
            return False, "No documents retrieved"
        
        entities = RAGPipeline.extract_entities_from_query(query)
        if not entities:
            # If no entities to check, allow generation (generic questions)
            return True, "No specific entities to validate"
        
        # Check if at least one key entity appears in retrieved chunks
        combined_text = " ".join(
            doc.get("chunk_text", "").lower() for doc in documents
        )
        
        found_entities = [e for e in entities if e.lower() in combined_text]

        if found_entities:
            logger.info(f"Retrieval grounded: found entities {found_entities}")
            return True, f"Found entities: {', '.join(found_entities)}"
        
        # If primary entities not found, consider it ungrounded
        logger.warning(f"Retrieval not grounded: entities {entities} not in retrieved docs")
        return False, f"Query entities not found in retrieved context"

    async def _compress_context_for_llm(self, query: str, documents: list[dict], top_sentences: int = 3) -> list[dict]:
        """Context Filtering and Compression Layer:
        Compresses the context for the LLM by selecting the most relevant sentences
        from each document chunk based on the query.
        """
        compressed_documents = []
        for doc in documents:
            chunk_text = doc.get("chunk_text", "")
            if not chunk_text:
                continue

            # Use TextProcessor's sentence splitting for consistency
            sentences = TextProcessor.split_into_sentences(chunk_text)
            if not sentences:
                continue

            # Generate embeddings for query and each sentence
            # Only embed the query once
            query_embedding = await self.llm_client.embed(query) 
            
            sentence_embeddings = []
            for s in sentences:
                # Avoid embedding very short or noisy sentences
                if len(s.strip()) > 10: 
                    sentence_embeddings.append(await self.llm_client.embed(s))
                else:
                    sentence_embeddings.append(None) # Placeholder for short sentences

            # Calculate cosine similarity between query and each sentence
            similarities = []
            for i, s_emb in enumerate(sentence_embeddings):
                if s_emb is not None:
                    similarity = self._cosine_similarity(query_embedding, s_emb)
                    similarities.append((similarity, sentences[i]))
            
            # Sort sentences by similarity and select the top N
            similarities.sort(key=lambda x: x[0], reverse=True)
            selected_sentences = [s for score, s in similarities[:top_sentences]]

            if selected_sentences:
                compressed_text = " ".join(selected_sentences)
                compressed_doc = doc.copy()
                compressed_doc["chunk_text"] = compressed_text
                compressed_documents.append(compressed_doc)
        return compressed_documents

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude = math.sqrt(sum(v1**2 for v1 in vec1)) * math.sqrt(sum(v2**2 for v2 in vec2))
        return dot_product / magnitude if magnitude else 0.0

    async def health_check(self) -> dict:
        """Check health of RAG components."""
        return {
            "vector_db": await self.vector_db.health_check(),
            "llm": await self.llm_client.health_check(),
        }
