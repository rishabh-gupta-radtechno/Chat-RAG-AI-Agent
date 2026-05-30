"""
Chat and RAG service with LLM integration.
"""
from datetime import datetime

import json
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
            model=response.model,
            thinking=response.thinking,
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
            documents = await self.rag_pipeline.retrieve(retrieval_query)
            logger.info(f"Retrieved {len(documents)} documents")

            # Process with conversation-aware RAG
            response = await self._generate_conversation_answer(
                message=message,
                documents=documents,
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
                    page_number=doc.get("page_number"),
                )
                for doc in documents
            ]

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
                model=settings.ollama_chat_model,
                thinking=response.get("thinking"),
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
        history: list,
    ) -> dict:
        """Generate an answer with retrieved docs and recent conversation context."""
        extracted_answer = self.agent._extract_procedure_answer(message, documents)
        if extracted_answer:
            return {
                "answer": extracted_answer,
                "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks.",
            }

        context = self.agent._format_context(documents)
        history_lines = []
        for turn in history:
            history_lines.append(f"User: {turn.question}")
            history_lines.append(f"Assistant: {turn.answer}")
        history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

        prompt = f"""Answer the user's latest message using the retrieved context and recent conversation.
Use the conversation history for follow-up references such as 'it', 'that', or 'the above part'.
If the retrieved context does not support the answer, say the uploaded documents do not provide enough information.
Keep the answer direct and concise.

Recent conversation:
{history_text}

Latest user message:
{message}

Retrieved context:
{context}"""

        answer = await self.llm_client.generate(
            prompt,
            system="You are a careful RAG assistant. Use recent chat history only as conversational context, and use the retrieved documents as the factual source of truth.",
            temperature=0.2,
            top_p=0.9,
        )
        return {
            "answer": answer,
            "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks and {len(history)} prior turns.",
        }

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