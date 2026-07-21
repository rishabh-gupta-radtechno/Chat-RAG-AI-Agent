"""Add file_status column to files

Revision ID: 003
Revises: 002
Create Date: 2026-07-21 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add file_status column to files table with a default of true."""
    op.add_column(
        'files',
        sa.Column('file_status', sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    """Drop file_status column from files table."""
    op.drop_column('files', 'file_status')
