"""
Chat and RAG service with LLM integration.
"""

import json
import re
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.rag import RAGPipeline
from app.ai.agent import ReActAgent
from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.chat import ChatHistoryRepository
from app.schemas import ChatResponse, SourceReference

logger = get_logger(__name__)
settings = get_settings()


class ChatService:
    """Chat service with RAG and agent integration."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.chat_repo = ChatHistoryRepository(session)
        self.rag_pipeline = RAGPipeline()
        self.agent = ReActAgent(self.rag_pipeline)

    async def ask_question(
        self,
        user_id: uuid.UUID,
        question: str,
    ) -> ChatResponse:
        """Process a question with RAG and agent."""
        logger.info(f"Processing question from user {user_id}: {question}")

        try:
            # Retrieve relevant documents
            documents = await self.rag_pipeline.retrieve(question)
            logger.info(f"Retrieved {len(documents)} documents")

            language_mode = self._detect_language_mode(question)
            logger.info(f"Detected language mode: {language_mode}")

            # Process with ReAct agent
            response = await self.agent.process(question, documents, language_mode)
            logger.info(f"Agent generated response")

            # Extract sources from documents
            sources = [
                SourceReference(
                    filename=doc.get("filename", "Unknown"),
                    file_id=uuid.UUID(doc.get("file_id", "00000000-0000-0000-0000-000000000000")),
                    chunk_index=doc.get("chunk_index", 0),
                    relevance_score=doc.get("relevance_score", 0.0),
                )
                for doc in documents
            ]

            # Store chat history
            await self.chat_repo.create_with_sources(
                user_id=user_id,
                question=question,
                answer=response.get("answer", ""),
                sources=[s.model_dump(mode="json") for s in sources],
                model=settings.ollama_chat_model,
            )
            await self.chat_repo.commit()

            return ChatResponse(
                answer=response.get("answer", ""),
                sources=sources,
                model=settings.ollama_chat_model,
                thinking=response.get("thinking"),
            )

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def _detect_language_mode(self, question: str) -> str:
        """Detect whether the question is English, Hindi, or mixed (Hinglish)."""
        text = question.strip()
        if not text:
            return "english"

        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))
        lower_text = text.lower()

        roman_hindi_words = {
            "kya",
            "hai",
            "ka",
            "ke",
            "ki",
            "kahan",
            "kaha",
            "kaise",
            "kuch",
            "nahi",
            "nahin",
            "mujhe",
            "tum",
            "aap",
            "yeh",
            "woh",
            "agar",
            "toh",
            "bhi",
            "mera",
            "meri",
            "hum",
            "tha",
            "thi",
            "haan",
            "nahi",
            "kisi",
            "kaun",
            "kab",
        }

        contains_roman_hindi = any(word in lower_text.split() for word in roman_hindi_words)

        if has_devanagari and not has_latin:
            return "hindi"
        if has_latin and contains_roman_hindi and not has_devanagari:
            return "hinglish"
        if has_latin and not contains_roman_hindi:
            return "english"
        if has_devanagari:
            return "hindi"

        return "english"

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
                "question": h.question,
                "answer": h.answer,
                "sources": [SourceReference(**source) for source in sources],
                "created_at": h.created_at,
            })
        return result
