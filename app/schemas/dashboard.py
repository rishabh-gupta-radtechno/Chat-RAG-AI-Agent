from pydantic import BaseModel

class DashboardStatsResponse(BaseModel):
    total_registered_users: int
    total_active_users: int
    total_knowledge_files: int
    total_synced_files: int
    total_conversations: int
    total_active_conversations_today: int