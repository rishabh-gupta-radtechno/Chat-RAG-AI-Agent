"""
Chat and RAG service with LLM integration.
"""
from datetime import datetime

import json
import re
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.rag import RAGPipeline
from app.ai.agent import ReActAgent
from app.ai.llm import OllamaClient
from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.chat import ChatHistoryRepository
from app.schemas import (
    ChatResponse,
    ConversationChatResponse,
    ConversationSummaryResponse,
    ConversationTurnResponse,
    DiagramReference,
    SourceReference,
)

logger = get_logger(__name__)
settings = get_settings()


class ChatService:
    """Chat service with RAG and agent integration."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.chat_repo = ChatHistoryRepository(session)
        self.rag_pipeline = RAGPipeline()
        self.agent = ReActAgent(self.rag_pipeline)
        self.llm_client = OllamaClient()

    async def ask_question(
        self,
        user_id: uuid.UUID,
        question: str,
    ) -> ChatResponse:
        """Process a question with RAG and agent."""
        response = await self.chat(user_id=user_id, message=question)
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            diagrams=response.diagrams,
            model=response.model,
            # thinking=response.thinking,
        )

    async def chat(
        self,
        user_id: uuid.UUID,
        message: str,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> ConversationChatResponse:
        """Process a multi-turn chat message with RAG and conversation memory."""
        logger.info(f"Processing chat message from user {user_id}: {message}")

        try:
            conversation_id = conversation_id or uuid.uuid4()

            history = await self.chat_repo.get_recent_by_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=6,
            )

            retrieval_query = self._build_retrieval_query(message, history)

            # Retrieve relevant documents
            documents = await self.rag_pipeline.retrieve(retrieval_query, user_id=user_id)
            logger.info(f"Retrieved {len(documents)} documents")
            diagram_user_id = None if any(
                doc.get("retrieval_scope") == "global_fallback" for doc in documents
            ) else str(user_id)
            diagrams = await self.rag_pipeline.vector_db.get_diagrams_for_sources(
                documents,
                user_id=diagram_user_id,
            )
            logger.info(f"Retrieved {len(diagrams)} related diagrams")

            # Process with conversation-aware RAG
            response = await self._generate_conversation_answer(
                message=message,
                documents=documents,
                diagrams=diagrams,
                history=history,
            )
            logger.info("Agent generated response")

            # Extract sources from documents
            sources = [
                SourceReference(
                    filename=doc.get("filename", "Unknown"),
                    file_id=uuid.UUID(doc.get("file_id", "00000000-0000-0000-0000-000000000000")),
                    chunk_index=doc.get("chunk_index", 0),
                    relevance_score=doc.get("relevance_score", 0.0),
                    page_number=doc.get("page_number", 0) or 0,
                    document_page_number=doc.get("document_page_number"),
                    content_type=doc.get("content_type"),
                    excerpt=self._excerpt(doc.get("chunk_text", "")),
                )
                for doc in documents
            ]
            sources = sorted(sources, key=lambda source: source.relevance_score, reverse=True)[:5]

            diagram_references = [
                DiagramReference(
                    filename=diagram.get("filename", "Unknown"),
                    file_id=uuid.UUID(diagram.get("file_id", "00000000-0000-0000-0000-000000000000")),
                    page_number=diagram.get("page_number", 0) or 0,
                    document_page_number=diagram.get("document_page_number"),
                    image_index=diagram.get("image_index"),
                    description=diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                    image_url=diagram.get("image_url"),
                    relevance_score=diagram.get("relevance_score", 0.0),
                )
                for diagram in diagrams
            ]

            unsupported_answer = "The uploaded documents do not provide enough information to answer this question."
            if response.get("answer", "").strip() == unsupported_answer:
                sources = []
                diagram_references = []

            # Store chat history
            await self.chat_repo.create_with_sources(
                user_id=user_id,
                conversation_id=conversation_id,
                question=message,
                answer=response.get("answer", ""),
                sources=[s.model_dump(mode="json") for s in sources],
                model=settings.ollama_chat_model,
            )
            await self.chat_repo.commit()

            return ConversationChatResponse(
                conversation_id=conversation_id,
                answer=response.get("answer", ""),
                sources=sources,
                diagrams=diagram_references,
                model=settings.ollama_chat_model,
                # thinking=response.get("thinking"),
            )

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def _build_retrieval_query(self, message: str, history: list) -> str:
        """Expand retrieval query with recent user turns for follow-up questions."""
        prior_questions = [turn.question.strip() for turn in history[-2:] if getattr(turn, "question", "").strip()]
        if not prior_questions:
            return message
        return "\n".join(prior_questions + [message])

    async def _generate_conversation_answer(
        self,
        message: str,
        documents: list[dict],
        diagrams: list[dict],
        history: list,
    ) -> dict:
        """Generate an answer with retrieved docs and recent conversation context."""
        extraction_documents = documents + [
            {
                **diagram,
                "chunk_text": diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                "content_type": diagram.get("content_type", "diagram"),
            }
            for diagram in diagrams
        ]
        extracted_answer = self.agent._extract_procedure_answer(message, extraction_documents)
        if extracted_answer:
            source_pages = self._source_page_summary(documents)
            diagram_pages = self._diagram_page_summary(diagrams)
            if source_pages:
                extracted_answer = f"{extracted_answer}\n\nSources: {source_pages}."
            if diagram_pages:
                extracted_answer = f"{extracted_answer}\nDiagrams: {diagram_pages}."
            return {
                "answer": extracted_answer,
                "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks and {len(diagrams)} related diagrams.",
            }

        direct_answer = self._extract_direct_answer(message, documents, diagrams)
        if direct_answer:
            return {
                "answer": direct_answer,
                "thinking": f"Used direct matching text from {len(documents)} retrieved document chunks and {len(diagrams)} related diagrams.",
            }

        if not self._has_sufficient_evidence(message, documents):
            return {
                "answer": "The uploaded documents do not provide enough information to answer this question.",
                "thinking": f"Found {len(documents)} retrieved chunks, but none provided enough direct evidence.",
            }

        context = self.agent._format_context(documents)
        diagram_context = self._format_diagram_context(diagrams)
        history_lines = []
        for turn in history:
            history_lines.append(f"User: {turn.question}")
            history_lines.append(f"Assistant: {turn.answer}")
        history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

        prompt = f"""Answer the user's latest message using the retrieved context and recent conversation.
Use the conversation history for follow-up references such as 'it', 'that', or 'the above part'.
Answer with page numbers for important facts, using short citations like "(page 3)".
Prefer exact wording from the context for definitions, names, numbers, limits, and procedures.
If related diagrams are available, include a short "Diagrams" line with their page numbers.
If the retrieved context does not support the answer, say the uploaded documents do not provide enough information.
Do not use outside knowledge.
Keep the answer direct and concise.

Recent conversation:
{history_text}

Latest user message:
{message}

Retrieved context:
{context}

Related diagrams:
{diagram_context}"""

        answer = await self.llm_client.generate(
            prompt,
            system=f"You are a careful RAG assistant. Use recent chat history only as conversational context, and use the retrieved documents as the factual source of truth.",
            temperature=0.2,
            top_p=0.9,
        )

        # Validation: Check if answer is grounded in retrieved chunks
        is_grounded = self._validate_answer_grounding(answer, documents)
        has_citation = "do not provide enough information" in answer.lower() or bool(
            re.search(r"\bpage\s+\d+\b|\(page\s+\d+\)", answer, re.IGNORECASE)
        )
        if not is_grounded or not has_citation:
            logger.warning("Answer not grounded in retrieved documents")
            answer = "The uploaded documents do not provide enough information to answer this question."

        return {
            "answer": answer,
            "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks, {len(diagrams)} related diagrams, and {len(history)} prior turns.",
        }

    @staticmethod
    def _excerpt(text: str, max_chars: int = 240) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _format_diagram_context(self, diagrams: list[dict]) -> str:
        if not diagrams:
            return "No related diagrams found."

        lines = []
        for index, diagram in enumerate(diagrams, start=1):
            filename = diagram.get("filename", "Unknown")
            page_number = diagram.get("page_number", 0)
            image_index = diagram.get("image_index")
            description = self._excerpt(
                diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                max_chars=400,
            )
            image_label = f", image {image_index}" if image_index is not None else ""
            page_label = self._format_page_reference(page_number, diagram.get("document_page_number"))
            lines.append(f"Diagram {index} ({filename}, {page_label}{image_label}): {description}")

        return "\n".join(lines)

    def _extract_direct_answer(self, message: str, documents: list[dict], diagrams: list[dict]) -> str:
        query_terms = self._important_terms(message)
        if not query_terms:
            return ""

        best_document = None
        best_score = 0.0
        for document in documents:
            chunk_text = document.get("chunk_text", "")
            normalized = self._normalize_text(chunk_text)
            matched_terms = sum(1 for term in query_terms if term in normalized)
            score = matched_terms + float(document.get("lexical_score") or 0.0)
            if score > best_score:
                best_score = score
                best_document = document

        if not best_document or best_score < max(1, len(query_terms) - 1):
            return ""

        chunk_text = re.sub(r"\s+", " ", best_document.get("chunk_text", "")).strip()
        if not chunk_text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        normalized_query = " ".join(query_terms)
        selected = []

        for sentence in sentences:
            normalized_sentence = self._normalize_text(sentence)
            if normalized_query in normalized_sentence and re.search(r"\b(is|are|used|set|designed|connected)\b", sentence, re.IGNORECASE):
                selected.append(sentence.strip())
                break

        if not selected:
            for sentence in sentences:
                normalized_sentence = self._normalize_text(sentence)
                matched_terms = sum(1 for term in query_terms if term in normalized_sentence)
                if matched_terms >= max(1, len(query_terms) - 1) and len(sentence.split()) > 8:
                    selected.append(sentence.strip())
                    break

        if not selected:
            return ""

        page_number = best_document.get("page_number", 0) or 0
        document_page_number = best_document.get("document_page_number")
        answer_text = self._strip_leading_heading(selected[0], query_terms)
        page_reference = self._format_page_reference(page_number, document_page_number)
        answer = f"{answer_text} ({page_reference})"

        diagram_pages = self._diagram_page_summary(diagrams)
        if page_number:
            answer = f"{answer}\n\nSources: {page_reference}."
        if diagram_pages:
            answer = f"{answer}\nDiagrams: {diagram_pages}."
        return answer

    @staticmethod
    def _strip_leading_heading(text: str, query_terms: list[str]) -> str:
        query_phrase = r"\s+".join(re.escape(term) for term in query_terms)
        stripped = re.sub(
            rf"^\s*\d+(\.\d+)*\s+{query_phrase}\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return stripped or text

    @staticmethod
    def _format_page_reference(page_number: int, document_page_number: Optional[int] = None) -> str:
        if document_page_number and document_page_number != page_number:
            return f"page {page_number} / document page {document_page_number}"
        return f"page {page_number}"

    @staticmethod
    def _important_terms(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "for", "how", "in", "is", "of", "on",
            "or", "the", "to", "what", "when", "where", "which", "with",
            "tell", "about", "explain", "describe", "give",
        }
        return [
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) > 2 and term not in stop_words
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _source_page_summary(documents: list[dict]) -> str:
        pages = sorted({
            int(doc.get("page_number"))
            for doc in documents
            if doc.get("page_number")
        })
        return ", ".join(f"page {page}" for page in pages)

    @staticmethod
    def _diagram_page_summary(diagrams: list[dict]) -> str:
        pages = sorted({
            int(diagram.get("page_number"))
            for diagram in diagrams
            if diagram.get("page_number")
        })
        return ", ".join(f"page {page}" for page in pages)

    async def get_chat_history(
        self,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> list:
        """Get chat history for user."""
        histories = await self.chat_repo.get_by_user(user_id, skip, limit)
        result = []
        for h in histories:
            try:
                sources = json.loads(h.sources) if h.sources else []
            except json.JSONDecodeError:
                sources = []

            result.append({
                "id": h.id,
                "conversation_id": h.conversation_id,
                "question": h.question,
                "answer": h.answer,
                "sources": [SourceReference(**source) for source in sources],
                "created_at": h.created_at,
            })
        return result

    async def get_conversation_history(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ConversationTurnResponse]:
        """Get ordered turns for a conversation."""
        histories = await self.chat_repo.get_by_conversation(user_id, conversation_id, limit)
        result = []
        for history in histories:
            try:
                sources = json.loads(history.sources) if history.sources else []
            except json.JSONDecodeError:
                sources = []

            result.append(
                ConversationTurnResponse(
                    id=history.id,
                    conversation_id=history.conversation_id,
                    question=history.question,
                    answer=history.answer,
                    sources=[SourceReference(**source) for source in sources],
                    model=history.model,
                    created_at=history.created_at,
                )
            )
        return result

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> list[ConversationSummaryResponse]:
        """List conversation summaries for a user."""
        histories = await self.chat_repo.list_conversations(user_id, limit)
        return [
            ConversationSummaryResponse(
                conversation_id=history.conversation_id,
                last_question=history.question,
                last_answer=history.answer,
                model=history.model,
                created_at=history.created_at,
            )
            for history in histories
        ]

    def _validate_answer_grounding(self, answer: str, documents: list[dict]) -> bool:
        """Validate that the answer is grounded in the retrieved documents."""
        if not documents:
            return False

        answer_lower = answer.lower()
        combined_text = " ".join([doc.get("chunk_text", "") for doc in documents]).lower()
        if not combined_text.strip():
            return False

        if "do not provide enough information" in answer_lower:
            return True

        # Keep this permissive: local models often paraphrase manual text, and
        # over-strict validation turns good retrieved context into a refusal.
        key_phrases = [
            word
            for word in re.findall(r"[a-z0-9]+", answer_lower)
            if len(word) > 4
        ]
        matches = sum(1 for phrase in key_phrases if phrase in combined_text)

        return matches / len(key_phrases) >= 0.2 if key_phrases else True

    def _has_sufficient_evidence(self, message: str, documents: list[dict]) -> bool:
        if not documents:
            return False

        query_terms = self._important_terms(message)
        if not query_terms:
            return True

        for document in documents:
            chunk_text = document.get("chunk_text", "")
            lexical_score = float(document.get("lexical_score") or 0.0)
            relevance_score = float(document.get("relevance_score") or 0.0)
            normalized = self._normalize_text(chunk_text)
            matched_terms = sum(1 for term in set(query_terms) if term in normalized)
            if matched_terms >= max(1, min(3, len(set(query_terms)))):
                return True
            if lexical_score >= 0.45 or relevance_score >= settings.similarity_threshold:
                return True

        return False
    
    async def list_all_conversations(
        self,
        limit: int = 100,
        user_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """List all conversation summaries across the system (all users)."""
        rows = await self.chat_repo.list_all_conversations(limit, user_name=user_name, start_date=start_date, end_date=end_date)
        return [
            {
                "user": {
                    "id": row.ChatHistory.user_id,
                    "name": row.name,
                },
                "conversation_id": row.ChatHistory.conversation_id,
                "conversation_title": row.ChatHistory.question[:50] + "..." if len(row.ChatHistory.question) > 50 else row.ChatHistory.question,
                "last_question": row.ChatHistory.question,
                "last_answer": row.ChatHistory.answer,
                "model": row.ChatHistory.model,
                "startdate": row.start_date,
                "last_activity": row.ChatHistory.created_at,
            }
            for row in rows
        ]
 
    async def get_conversation_history_by_id(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ConversationTurnResponse]:
        """Get ordered turns for a conversation."""
        histories = await self.chat_repo.get_by_conversation_history(conversation_id, limit)
        result = []
        for history in histories:
            try:
                sources = json.loads(history.sources) if history.sources else []
            except json.JSONDecodeError:
                sources = []
 
            result.append(
                ConversationTurnResponse(
                    id=history.id,
                    conversation_id=history.conversation_id,
                    question=history.question,
                    answer=history.answer,
                    sources=[SourceReference(**source) for source in sources],
                    model=history.model,
                    created_at=history.created_at,
                )
            )
        return result
