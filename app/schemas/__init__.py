"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ============ Auth Schemas ============


class UserRegisterRequest(BaseModel):
    """User registration request schema."""

    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    is_active: bool = Field(default=True)


class UserLoginRequest(BaseModel):
    """User login request schema."""

    email: EmailStr
    password: str

class UserUpdateRequest(BaseModel):
    """User update request schema for partial updates."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """User response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: Optional[str] = None
    email: str
    is_active: bool
    created_at: datetime


# ============ File Schemas ============


class FileUploadResponse(BaseModel):
    """File upload response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    filepath: str
    file_size: int
    file_type: str
    is_active: bool
    is_embedded: bool
    created_at: datetime


class FileListResponse(BaseModel):
    """File list response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_size: int
    file_type: str
    is_embedded: bool
    created_at: datetime


class SyncEmbeddingsRequest(BaseModel):
    """Sync embeddings request schema."""

    file_id: UUID


class SyncEmbeddingsResponse(BaseModel):
    """Sync embeddings response schema."""

    file_id: UUID
    chunks_created: int
    status: str


# ============ Chat Schemas ============


class SourceReference(BaseModel):
    """Source reference for chat response."""

    filename: str
    file_id: UUID
    chunk_index: int
    relevance_score: float


class ChatRequest(BaseModel):
    """Chat request schema."""

    question: str = Field(..., min_length=1, max_length=5000)


class ConversationChatRequest(BaseModel):
    """Conversation-aware chat request schema."""

    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    """Chat response schema."""

    model_config = ConfigDict(from_attributes=True)

    answer: str
    sources: list[SourceReference]
    model: str
    thinking: Optional[str] = None


class ConversationChatResponse(BaseModel):
    """Conversation-aware chat response schema."""

    model_config = ConfigDict(from_attributes=True)

    conversation_id: UUID
    answer: str
    sources: list[SourceReference]
    model: str
    thinking: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Chat history response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    question: str
    answer: str
    sources: list[SourceReference]
    created_at: datetime


class ConversationTurnResponse(BaseModel):
    """Single turn within a conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    question: str
    answer: str
    sources: list[SourceReference]
    model: str
    created_at: datetime


class ConversationSummaryResponse(BaseModel):
    """Conversation summary row."""

    conversation_id: UUID
    last_question: str
    last_answer: str
    model: str
    created_at: datetime


# ============ Health Check Schemas ============


class HealthCheckResponse(BaseModel):
    """Health check response schema."""

    status: str
    database: str
    vector_db: str
    llm: str


class DetailedHealthResponse(BaseModel):
    """Detailed health response schema."""

    status: str
    database: Optional[str] = None
    vector_db: Optional[str] = None
    llm: Optional[str] = None
    errors: list[str] = []


# ============ Admin Schemas ============


class AdminBase(BaseModel):
    name: str
    email: EmailStr
    is_enabled: bool = True


class AdminCreate(AdminBase):
    password: str


class AdminUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_enabled: Optional[bool] = None

class AdminLoginRequest(BaseModel):
    """Admin login request schema."""
    email: EmailStr
    password: str

class AdminLoginResponse(BaseModel):
    """Admin login response schema."""
    token: str
    name: str
    email: EmailStr

class AdminResponse(AdminBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminDeleteResponse(BaseModel):
    id: UUID
    message: str = "Admin deleted successfully"
