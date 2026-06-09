"""
Database models for the application.
"""

from typing import Optional
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    designation: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mobile: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    chat_histories: Mapped[list["ChatHistory"]] = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class File(Base):
    """File model for uploaded documents."""

    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    filepath: Mapped[str] = mapped_column(String(512))
    file_size: Mapped[int] = mapped_column(default=0)
    file_type: Mapped[str] = mapped_column(String(50))  # pdf, txt, docx, etc.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("admins.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    admin: Mapped["Admin"] = relationship("Admin", foreign_keys=[uploaded_by])

    def __repr__(self) -> str:
        return f"<File {self.filename}>"


class ChatHistory(Base):
    """Chat history model."""

    __tablename__ = "chat_histories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sources: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string of sources
    diagrams: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string of diagrams
    model: Mapped[str] = mapped_column(String(100), default="qwen3:8b")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_histories")

    def __repr__(self) -> str:
        return f"<ChatHistory {self.id}>"


class Admin(Base):
    """Admin model for system management."""

    __tablename__ = "admins"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password: Mapped[str] = mapped_column(String(255))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    files: Mapped[list["File"]] = relationship("File", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Admin {self.email}>"
