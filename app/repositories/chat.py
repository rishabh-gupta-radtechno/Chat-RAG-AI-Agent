"""
Chat history repository for database access.
"""

import json
import uuid
from typing import Optional

from sqlalchemy import select
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

    async def create_with_sources(
        self,
        user_id: uuid.UUID,
        question: str,
        answer: str,
        sources: list[dict],
        model: Optional[str] = None,
    ) -> ChatHistory:
        """Create chat history with sources."""
        sources_json = json.dumps(sources, default=str)
        return await self.create(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources_json,
            model=model or "",
        )
