"""
Database connection and session management.
"""

from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=settings.db_pool_pre_ping,
    future=True,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_all_tables():
    """Create all tables."""
    from app.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
        await conn.execute(
            text(
                """
                ALTER TABLE chat_histories
                ADD COLUMN IF NOT EXISTS conversation_id UUID
                """
            )
        )
        await conn.execute(
            text(
                """
                UPDATE chat_histories
                SET conversation_id = gen_random_uuid()
                WHERE conversation_id IS NULL
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE chat_histories
                ALTER COLUMN conversation_id SET NOT NULL
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_chat_histories_conversation_id
                ON chat_histories (conversation_id)
                """
            )
        )


async def drop_all_tables():
    """Drop all tables."""
    from app.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
