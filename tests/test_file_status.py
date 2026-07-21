import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Admin
from app.services.file import FileService


@pytest.mark.asyncio
async def test_toggle_file_status_updates_database_value(async_session: AsyncSession):
    user_id = uuid.uuid4()
    async_session.add(
        Admin(
            id=user_id,
            name="Admin",
            email="admin@example.com",
            password="hashed",
        )
    )
    await async_session.commit()

    service = FileService(async_session)
    created_file = await service.upload_file(
        user_id=user_id,
        filename="sample.pdf",
        file_content=b"%PDF-1.4",
        file_type="pdf",
    )

    assert created_file.file_status is True

    updated = await service.toggle_file_status(created_file.id, user_id, False)

    assert updated is True
    refreshed = await service.get_file(created_file.id)
    assert refreshed is not None
    assert refreshed.file_status is False
