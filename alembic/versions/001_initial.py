"""Initial migration

Revision ID: 001
Revises:
Create Date: 2024-01-15 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial tables."""
    pass


def downgrade() -> None:
    """Drop initial tables."""
    pass
