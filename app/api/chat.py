"""
Chat API routes with RAG integration.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id
from app.db.database import get_db
from app.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.chat import ChatService
from app.core.logging import get_logger

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Ask a question with RAG and agent."""
    try:
        chat_service = ChatService(session)
        response = await chat_service.ask_question(user_id, request.question)
        return response

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing question",
        )


@router.get("/history", response_model=List[ChatHistoryResponse])
async def get_chat_history(
    skip: int = 0,
    limit: int = 50,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Get chat history for current user."""
    try:
        chat_service = ChatService(session)
        history = await chat_service.get_chat_history(user_id, skip, limit)
        return history

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving chat history",
        )
