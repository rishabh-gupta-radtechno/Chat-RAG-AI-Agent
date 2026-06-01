import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.admin import AdminService
from app.schemas import AdminCreate, AdminUpdate, AdminResponse, AdminDeleteResponse, AdminLoginRequest, AdminLoginResponse

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.post("/register", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def register_admin(
    admin_in: AdminCreate,
    session: AsyncSession = Depends(get_db)
):
    """Register a new system admin."""
    service = AdminService(session)
    try:
        return await service.register_admin(admin_in)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=AdminLoginResponse)
async def login_admin(
    login_in: AdminLoginRequest,
    session: AsyncSession = Depends(get_db)
):
    """Authenticate an admin user."""
    service = AdminService(session)
    try:
        return await service.login(login_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/getbyadminId/{admin_id}", response_model=AdminResponse)
async def get_admin_by_id(
    admin_id: uuid.UUID,
    session: AsyncSession = Depends(get_db)
):
    """Get admin details by ID."""
    service = AdminService(session)
    admin = await service.get_admin_by_id(admin_id)
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    return admin

@router.put("/update/{admin_id}", response_model=AdminResponse)
async def update_admin(
    admin_id: uuid.UUID,
    admin_in: AdminUpdate,
    session: AsyncSession = Depends(get_db)
):
    """Update admin profile information."""
    service = AdminService(session)
    admin = await service.update_admin(admin_id, admin_in)
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    return admin

@router.delete("/delete/{admin_id}", response_model=AdminDeleteResponse)
async def delete_admin(
    admin_id: uuid.UUID,
    session: AsyncSession = Depends(get_db)
):
    """Delete an admin account."""
    service = AdminService(session)
    if not await service.delete_admin(admin_id):
        raise HTTPException(status_code=404, detail="Admin not found")
    return AdminDeleteResponse(id=admin_id)