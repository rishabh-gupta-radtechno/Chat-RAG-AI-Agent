"""
Authentication API routes.
"""
from datetime import datetime
import uuid
from typing import List, Optional
from app.api.dependencies import get_current_user_id
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas import PhoneLoginRequest, TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse, UserUpdateRequest, EmailStr
from app.services.auth import AuthService
from app.utils.exceptions import InvalidCredentialsException, UserAlreadyExistsException, UnauthorizedException
from app.core.logging import get_logger


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: UserRegisterRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Register a new user."""
    try:
        auth_service = AuthService(session)
        user = await auth_service.register(
            email=request.email,
            password=request.password,
            name=request.name,
            department=request.department,
            designation=request.designation,
            mobile=request.mobile,
            is_active=request.is_active
        )
        return user
    except ValueError as e:
        # Use the specific error message from the service (e.g., mobile vs email conflict)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Login user and get JWT tokens."""
    try:
        auth_service = AuthService(session)
        tokens = await auth_service.login(request.email, request.password)
        return tokens
    except ValueError as e:
        raise InvalidCredentialsException()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/loginphone", response_model=TokenResponse)
async def login_phone(
    request: PhoneLoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Login user with phone number and get JWT tokens."""
    try:
        auth_service = AuthService(session)
        tokens = await auth_service.login_with_phone(request.phone, request.password)
        return tokens
    except ValueError as e:
        raise InvalidCredentialsException(detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Get current user information."""
    auth_service = AuthService(session)
    user = await auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.get("/getalluser", response_model=List[UserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = Query(None, description="Filter by user name"),
    email: Optional[EmailStr] = Query(None, description="Filter by user email"),
    department: Optional[str] = Query(None, description="Filter by department"),
    designation: Optional[str] = Query(None, description="Filter by designation"),
    mobile: Optional[int] = Query(None, description="Filter by mobile number"),
    start_date: Optional[datetime] = Query(None, description="Filter by creation date (start)"),
    end_date: Optional[datetime] = Query(None, description="Filter by creation date (end)"),
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Get all registered users."""
    auth_service = AuthService(session)
    return await auth_service.get_all_users(skip=skip, limit=limit, name=name, email=email, department=department, designation=designation, mobile=mobile, start_date=start_date, end_date=end_date)


@router.put("/updateUser/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    request: UserUpdateRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Update user information (partial updates supported, password excluded)."""
    auth_service = AuthService(session)
    user = await auth_service.update_user(user_id, request)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.delete("/deleteUser/{user_id}")
async def delete_user(
    user_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
):
    """Delete a user account (soft delete)."""
    auth_service = AuthService(session)
    if not await auth_service.delete_user(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {"message": "User deleted successfully", "id": user_id}
