import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import hash_password, verify_password, create_access_token
from app.models import Admin
from app.repositories.admin import AdminRepository
from app.schemas import AdminCreate, AdminUpdate, AdminResponse, AdminLoginRequest, AdminLoginResponse

class AdminService:
    """Service for managing admin users."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.admin_repo = AdminRepository(session)

    async def register_admin(self, admin_in: AdminCreate) -> AdminResponse:
        """Register a new admin."""
        existing_admin = await self.admin_repo.get_by_email(admin_in.email)
        if existing_admin:
            raise ValueError("Admin with this email already exists")

        # Logic referenced from auth/register
        hashed_pwd = hash_password(admin_in.password)
        
        admin = await self.admin_repo.create(
            name=admin_in.name,
            email=admin_in.email,
            password=hashed_pwd,
            is_enabled=admin_in.is_enabled
        )
        await self.admin_repo.commit()
        return AdminResponse.model_validate(admin)

    async def login(self, login_in: AdminLoginRequest) -> AdminLoginResponse:
        """Login admin and return access token."""
        admin = await self.admin_repo.get_by_email(login_in.email)
        if not admin:
            raise ValueError("Invalid email or password")

        if not verify_password(login_in.password, admin.password):
            raise ValueError("Invalid email or password")

        if not admin.is_enabled:
            raise ValueError("Admin account is disabled")

        # Create access token (referencing logic from auth/login)
        token = create_access_token(str(admin.id))
        
        return AdminLoginResponse(token=token, name=admin.name, email=admin.email)

    async def get_admin_by_id(self, admin_id: uuid.UUID) -> Optional[AdminResponse]:
        """Get admin details by ID."""
        admin = await self.admin_repo.get_by_id(admin_id)
        if not admin:
            return None
        return AdminResponse.model_validate(admin)

    async def update_admin(self, admin_id: uuid.UUID, admin_in: AdminUpdate) -> Optional[AdminResponse]:
        """Update admin details."""
        admin = await self.admin_repo.get_by_id(admin_id)
        if not admin:
            return None

        update_data = admin_in.model_dump(exclude_unset=True)
        
        if "password" in update_data and update_data["password"]:
            update_data["password"] = hash_password(update_data["password"])
            
        updated_admin = await self.admin_repo.update(admin_id, **update_data)
        await self.admin_repo.commit()
        
        return AdminResponse.model_validate(updated_admin)

    async def delete_admin(self, admin_id: uuid.UUID) -> bool:
        """Delete an admin."""
        admin = await self.admin_repo.get_by_id(admin_id)
        if not admin:
            return False
        
        # Using hard delete as requested for the delete endpoint
        # If soft-delete is preferred, use update(admin_id, is_active=False)
        query = await self.admin_repo.delete(admin_id)
        await self.admin_repo.commit()
        return True