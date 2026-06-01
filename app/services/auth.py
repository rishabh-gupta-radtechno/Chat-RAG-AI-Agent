"""
Authentication service with user management.
"""
from datetime import datetime
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.models import User
from app.repositories.user import UserRepository
from app.schemas import TokenResponse, UserResponse, UserUpdateRequest


class AuthService:
    """Authentication service."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.user_repo = UserRepository(session)

    async def register(self, email: str, password: str, name: str, department: Optional[str] = None, mobile: Optional[int] = None, is_active: bool = True) -> UserResponse:
        """Register a new user."""
        # Check if user already exists
        existing_user = await self.user_repo.get_by_email(email)
        if existing_user:
            raise ValueError("Email already registered")
        
        existing_mobile = await self.user_repo.get_by_mobile(mobile)
        # Check if mobile already exists
        if existing_mobile:
            raise ValueError("Mobile number already registered")

        # Hash password
        password_hash = hash_password(password)

        # Create user
        user = await self.user_repo.create(
            name=name,
            email=email,
            password_hash=password_hash,
            is_active=is_active,
            department=department,
            mobile=mobile,
            is_deleted=False,
        )
        await self.user_repo.commit()

        return UserResponse.model_validate(user)

    async def login(self, email: str, password: str) -> TokenResponse:
        """Login user and return tokens."""
        # Get user by email
        user = await self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("Invalid email or password")

        # Verify password
        if not verify_password(password, user.password_hash):
            raise ValueError("Invalid email or password")

        # Check if user is active
        if not user.is_active:
            raise ValueError("User account is inactive")

        # Create tokens
        access_token = create_access_token(str(user.id))
        refresh_token = create_refresh_token(str(user.id))

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def update_user(self, user_id: uuid.UUID, update_in: UserUpdateRequest) -> Optional[UserResponse]:
        """Update user details selectively."""
        update_data = update_in.model_dump(exclude_unset=True)

        if "mobile" in update_data and update_data["mobile"]:
            existing = await self.user_repo.get_by_mobile(update_data["mobile"])
            if existing and existing.id != user_id:
                raise ValueError("Mobile number already in use by another user")
        
        user = await self.user_repo.update(user_id, **update_data)
        if not user:
            return None
            
        await self.user_repo.commit()
        return UserResponse.model_validate(user)

    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[UserResponse]:
        """Get user by ID."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return None
        return UserResponse.model_validate(user)

    async def get_all_users(
        self, 
        skip: int = 0, 
        limit: int = 100,
        name: Optional[str] = None,
        email: Optional[str] = None,
        department: Optional[str] = None,
        mobile: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[UserResponse]:
        """Get all users with filtering."""
        users = await self.user_repo.get_all(
            skip=skip, 
            limit=limit, 
            name=name, 
            email=email, 
            department=department,
            mobile=mobile,
            start_date=start_date, 
            end_date=end_date
        )
        return [UserResponse.model_validate(user) for user in users]

    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email."""
        user = await self.user_repo.get_by_email(email)
        if not user:
            return None
        return UserResponse.model_validate(user)

    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """Delete user (soft delete)."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return False

        # Soft delete by setting is_deleted to True and is_active to False
        # This ensures the user is hidden from lists and lookups
        await self.user_repo.update(user_id, is_deleted=True, is_active=False)
        await self.user_repo.commit()
        return True
