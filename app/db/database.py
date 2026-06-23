"""
Database connection and session management.
"""

from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

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
    """Dependency to get database session.

    Closing is guarded: if the underlying connection was dropped (e.g. during a
    long request), close()/rollback can raise asyncpg "connection is closed".
    That must not surface as an ASGI 500 during teardown, so swallow it.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        try:
            await session.close()
        except Exception as exc:  # dead connection on teardown — log, don't 500
            logger.warning("Error closing DB session: %s", exc)


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
        # Add diagrams column to chat_histories if it doesn't exist
        await conn.execute(
            text("ALTER TABLE chat_histories ADD COLUMN IF NOT EXISTS diagrams TEXT")
        )
        # Add new columns to users table if they don't exist
        await conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255)")
        )
        await conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE")
        )
        await conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS department VARCHAR(255)")
        )
        await conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS designation VARCHAR(255)")
        )
        await conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS mobile BIGINT")
        )
        await conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ix_users_mobile
                ON users (mobile) WHERE mobile IS NOT NULL
                """
            )
        )
        # Fix files.uploaded_by FK: drop old reference to users, add reference to admins
        await conn.execute(
            text(
                """
                DO $$
                DECLARE
                    v_constraint TEXT;
                BEGIN
                    SELECT tc.constraint_name INTO v_constraint
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.referential_constraints rc
                        ON tc.constraint_name = rc.constraint_name
                        AND tc.table_schema = rc.constraint_schema
                    JOIN information_schema.table_constraints tc2
                        ON rc.unique_constraint_name = tc2.constraint_name
                        AND rc.unique_constraint_schema = tc2.table_schema
                    WHERE tc.table_name = 'files'
                        AND tc.constraint_type = 'FOREIGN KEY'
                        AND kcu.column_name = 'uploaded_by'
                        AND tc2.table_name = 'users';

                    IF v_constraint IS NOT NULL THEN
                        EXECUTE 'ALTER TABLE files DROP CONSTRAINT ' || quote_ident(v_constraint);
                        ALTER TABLE files ADD CONSTRAINT files_uploaded_by_fkey
                            FOREIGN KEY (uploaded_by) REFERENCES admins(id);
                    END IF;
                END $$
                """
            )
        )


async def drop_all_tables():
    """Drop all tables."""
    from app.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
