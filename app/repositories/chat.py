"""
Chat history repository for database access.
"""

import json
import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatHistory
from app.repositories.base import BaseRepository


class ChatHistoryRepository(BaseRepository[ChatHistory]):
    """Chat history repository with additional queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ChatHistory)

    async def get_by_user(self, user_id: uuid.UUID, skip: int = 0, limit: int = 100) -> list[ChatHistory]:
        """Get chat history for a user."""
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ChatHistory]:
        """Get ordered chat turns for a conversation."""
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .where(ChatHistory.conversation_id == conversation_id)
            .order_by(ChatHistory.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_recent_by_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 6,
    ) -> list[ChatHistory]:
        """Get recent chat turns for context injection."""
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .where(ChatHistory.conversation_id == conversation_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> list[ChatHistory]:
        """Get latest turn for each conversation."""
        latest_per_conversation = (
            select(
                ChatHistory.conversation_id,
                func.max(ChatHistory.created_at).label("latest_created_at"),
            )
            .where(ChatHistory.user_id == user_id)
            .group_by(ChatHistory.conversation_id)
            .subquery()
        )

        stmt = (
            select(ChatHistory)
            .join(
                latest_per_conversation,
                (ChatHistory.conversation_id == latest_per_conversation.c.conversation_id)
                & (ChatHistory.created_at == latest_per_conversation.c.latest_created_at),
            )
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create_with_sources(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        question: str,
        answer: str,
        sources: list[dict],
        model: Optional[str] = None,
    ) -> ChatHistory:
        """Create chat history with sources."""
        sources_json = json.dumps(sources, default=str)
        return await self.create(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=sources_json,
            model=model or "",
        )
