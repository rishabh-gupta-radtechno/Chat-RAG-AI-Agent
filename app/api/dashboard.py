from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.dashboard import DashboardStatsResponse
from app.services.dashboard import DashboardService
from app.api.dependencies import get_current_user_id
from app.db.database import get_db
router = APIRouter()

@router.get("/dashboardcount", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def get_dashboard_statistics(
    user_id = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
    # current_user: dict = Depends(get_current_active_user), # Uncomment if authentication is required for dashboard stats
):
    """
    Retrieve various statistics for the dashboard.
    """
    dashboard_service = DashboardService(session)
    stats = await dashboard_service.get_dashboard_stats()
    return stats