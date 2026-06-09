"""
Chat history repository for database access.
"""
from datetime import datetime, time, timezone

import json
import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatHistory, User
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
    
    async def get_by_conversation_history(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ChatHistory]:
        """Get ordered chat turns for a conversation."""
        stmt = (
            select(ChatHistory)
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
        diagrams: Optional[list[dict]] = None,
        model: Optional[str] = None,
    ) -> ChatHistory:
        """Create chat history with sources and diagrams."""
        sources_json = json.dumps(sources, default=str)
        diagrams_json = json.dumps(diagrams, default=str) if diagrams else None
        return await self.create(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=sources_json,
            diagrams=diagrams_json,
            model=model or "",
        )

    async def list_all_conversations(
        self,
        limit: int = 100,
        user_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[any]:
        """Get latest turn for each conversation across all users with metadata."""
        # Subquery for the start date of each conversation
        start_dates = (
            select(
                ChatHistory.conversation_id,
                func.min(ChatHistory.created_at).label("start_date"),
            )
            .group_by(ChatHistory.conversation_id)
            .subquery()
        )
 
        # Get the latest message timestamp for each conversation
        latest_per_conversation = (
            select(
                ChatHistory.conversation_id,
                func.max(ChatHistory.created_at).label("latest_created_at"),
            )
            .group_by(ChatHistory.conversation_id)
            .subquery()
        )
 
        # Get the first question and answer for each conversation
        first_turns = (
            select(
                ChatHistory.conversation_id,
                ChatHistory.question.label("first_question"),
                ChatHistory.answer.label("first_answer"),
            )
            .join(
                start_dates,
                (ChatHistory.conversation_id == start_dates.c.conversation_id)
                & (ChatHistory.created_at == start_dates.c.start_date),
            )
            .subquery()
        )
 
        stmt = (
            select(
                ChatHistory,
                User.name,
                start_dates.c.start_date,
                first_turns.c.first_question,
                first_turns.c.first_answer,
            )
            .join(User, ChatHistory.user_id == User.id)
            .join(
                latest_per_conversation,
                (ChatHistory.conversation_id == latest_per_conversation.c.conversation_id)
                & (ChatHistory.created_at == latest_per_conversation.c.latest_created_at),
            )
            .join(
                start_dates,
                ChatHistory.conversation_id == start_dates.c.conversation_id,
            )
            .join(
                first_turns,
                ChatHistory.conversation_id == first_turns.c.conversation_id,
            )
        )

        if user_name:
            stmt = stmt.where(User.name.ilike(f"%{user_name}%"))

        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            start_date = datetime.combine(start_date.date(), time.min, tzinfo=start_date.tzinfo)
            stmt = stmt.where(ChatHistory.created_at >= start_date)
        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            end_date = datetime.combine(end_date.date(), time.max, tzinfo=end_date.tzinfo)
            stmt = stmt.where(ChatHistory.created_at <= end_date)

        stmt = stmt.order_by(ChatHistory.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.all()
