"""Add diagrams column to chat_histories

Revision ID: 002
Revises: 001
Create Date: 2024-01-16 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add diagrams column to chat_histories table."""
    op.add_column('chat_histories', sa.Column('diagrams', sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop diagrams column from chat_histories table."""
    op.drop_column('chat_histories', 'diagrams')
