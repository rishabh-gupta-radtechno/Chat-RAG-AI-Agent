"""
Pytest tests for authentication endpoints.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth import AuthService
from app.repositories.user import UserRepository


@pytest.mark.asyncio
async def test_register_user(async_session: AsyncSession):
    """Test user registration."""
    auth_service = AuthService(async_session)

    user = await auth_service.register(
        email="test@example.com",
        password="testpassword123",
    )

    assert user.email == "test@example.com"
    assert user.is_active is True


@pytest.mark.asyncio
async def test_register_duplicate_email(async_session: AsyncSession):
    """Test registering with duplicate email."""
    auth_service = AuthService(async_session)

    await auth_service.register(
        email="test@example.com",
        password="testpassword123",
    )

    with pytest.raises(ValueError):
        await auth_service.register(
            email="test@example.com",
            password="anotherpassword123",
        )


@pytest.mark.asyncio
async def test_login_user(async_session: AsyncSession):
    """Test user login."""
    auth_service = AuthService(async_session)

    await auth_service.register(
        email="test@example.com",
        password="testpassword123",
    )

    tokens = await auth_service.login(
        email="test@example.com",
        password="testpassword123",
    )

    assert tokens.access_token is not None
    assert tokens.refresh_token is not None


@pytest.mark.asyncio
async def test_login_invalid_password(async_session: AsyncSession):
    """Test login with invalid password."""
    auth_service = AuthService(async_session)

    await auth_service.register(
        email="test@example.com",
        password="testpassword123",
    )

    with pytest.raises(ValueError):
        await auth_service.login(
            email="test@example.com",
            password="wrongpassword",
        )


@pytest.mark.asyncio
async def test_login_nonexistent_user(async_session: AsyncSession):
    """Test login with nonexistent user."""
    auth_service = AuthService(async_session)

    with pytest.raises(ValueError):
        await auth_service.login(
            email="nonexistent@example.com",
            password="anypassword",
        )
