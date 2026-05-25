"""
Chat API routes with RAG integration.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id
from app.db.database import get_db
from app.schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationChatRequest,
    ConversationChatResponse,
    ConversationSummaryResponse,
    ConversationTurnResponse,
)
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


@router.post("/message", response_model=ConversationChatResponse)
async def chat_message(
    request: ConversationChatRequest,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Send a message within a conversation, creating one if needed."""
    try:
        chat_service = ChatService(session)
        response = await chat_service.chat(
            user_id=user_id,
            message=request.message,
            conversation_id=request.conversation_id,
        )
        return response

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat message",
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


@router.get("/conversations", response_model=List[ConversationSummaryResponse])
async def list_conversations(
    limit: int = 50,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """List conversation summaries for the current user."""
    try:
        chat_service = ChatService(session)
        return await chat_service.list_conversations(user_id, limit)
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing conversations",
        )


@router.get("/conversations/{conversation_id}", response_model=List[ConversationTurnResponse])
async def get_conversation_history(
    conversation_id: str,
    limit: int = 100,
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Get ordered message history for one conversation."""
    try:
        from app.utils.helpers import validate_uuid

        conversation_uuid = validate_uuid(conversation_id)
        if not conversation_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation ID")

        chat_service = ChatService(session)
        return await chat_service.get_conversation_history(user_id, conversation_uuid, limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving conversation history",
        )

@router.get("/chatall")
async def list_all_user_chats(
    limit: int = 100,
    session: AsyncSession = Depends(get_db),
):
    """List conversation summaries for all users (Admin view)."""
    try:
        chat_service = ChatService(session)
        return await chat_service.list_all_conversations(limit)
    except Exception as e:
        logger.error(f"Error listing all conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing all conversations",
        )
 
@router.get("/conversations_history/{conversation_id}", response_model=List[ConversationTurnResponse])
async def get_conversation_history(
    conversation_id: str,
    limit: int = 100,    
    session: AsyncSession = Depends(get_db),
):
    """Get ordered message history for one conversation."""
    try:
        from app.utils.helpers import validate_uuid
 
        conversation_uuid = validate_uuid(conversation_id)
        if not conversation_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid conversation ID")
 
        chat_service = ChatService(session)
        return await chat_service.get_conversation_history_by_id(conversation_uuid, limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving conversation history",
        )