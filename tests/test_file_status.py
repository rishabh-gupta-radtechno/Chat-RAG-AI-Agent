import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.vector_db import VectorDBClient
from app.models import Admin
from app.repositories.file import FileRepository
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


@pytest.mark.asyncio
async def test_get_disabled_file_ids_returns_only_disabled_files(async_session: AsyncSession):
    user_id = uuid.uuid4()
    async_session.add(
        Admin(id=user_id, name="Admin", email="admin2@example.com", password="hashed")
    )
    await async_session.commit()

    service = FileService(async_session)
    enabled = await service.upload_file(
        user_id=user_id, filename="enabled.pdf", file_content=b"%PDF-1.4", file_type="pdf",
    )
    disabled = await service.upload_file(
        user_id=user_id, filename="disabled.pdf", file_content=b"%PDF-1.4", file_type="pdf",
    )
    await service.toggle_file_status(disabled.id, user_id, False)

    disabled_ids = await FileRepository(async_session).get_disabled_file_ids()

    assert disabled_ids == {str(disabled.id)}
    assert str(enabled.id) not in disabled_ids


def test_build_filter_excludes_disabled_files():
    disabled_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    qfilter = VectorDBClient._build_filter(user_id, {disabled_id})

    # The user scope is a `must` and the disabled file is a `must_not`, so Qdrant
    # itself drops the disabled file's chunks — the search limit is spent on active
    # chunks only, never diluted by chunks that would be discarded afterwards.
    assert qfilter is not None
    user_conditions = [c for c in (qfilter.must or []) if c.key == "user_id"]
    assert user_conditions and user_conditions[0].match.value == user_id

    excluded_conditions = [c for c in (qfilter.must_not or []) if c.key == "file_id"]
    assert excluded_conditions
    assert disabled_id in excluded_conditions[0].match.any


def test_build_filter_none_when_no_scope_or_exclusions():
    assert VectorDBClient._build_filter(None, None) is None
    assert VectorDBClient._build_filter(None, set()) is None
