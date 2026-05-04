"""
Authentication API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from app.services.auth import AuthService
from app.utils.exceptions import InvalidCredentialsException, UserAlreadyExistsException

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: UserRegisterRequest,
    session: AsyncSession = Depends(get_db),
):
    """Register a new user."""
    try:
        auth_service = AuthService(session)
        user = await auth_service.register(request.email, request.password)
        return user
    except ValueError as e:
        raise UserAlreadyExistsException()
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


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    authorization: str = None,
    session: AsyncSession = Depends(get_db),
):
    """Get current user information."""
    from app.api.dependencies import get_current_user_id

    try:
        user_id = await get_current_user_id(authorization)
        auth_service = AuthService(session)
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return user
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
