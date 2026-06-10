from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user import UserRepository
from app.repositories.file import FileRepository
from app.repositories.chat import ChatHistoryRepository
from app.schemas.dashboard import DashboardStatsResponse

class DashboardService:
    def __init__(self, session: AsyncSession):
        self.user_repo = UserRepository(session)
        self.file_repo = FileRepository(session)
        self.chat_repo = ChatHistoryRepository(session)

    async def get_dashboard_stats(self) -> DashboardStatsResponse:
        total_registered_users = await self.user_repo.count_total()
        total_active_users = await self.chat_repo.count_active_users_total()
        total_knowledge_files = await self.file_repo.count_total()
        total_synced_files = await self.file_repo.count_synced()
        total_conversations = await self.chat_repo.count_total_conversations()
        total_active_conversations_today = await self.chat_repo.count_active_conversations_today()

        return DashboardStatsResponse(
            total_registered_users=total_registered_users,
            total_active_users=total_active_users,
            total_knowledge_files=total_knowledge_files,
            total_synced_files=total_synced_files,
            total_conversations=total_conversations,
            total_active_conversations_today=total_active_conversations_today,
        )