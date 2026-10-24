"""drop the retired catamine integration tables

Revision ID: 20261023_01
Revises: 20260913_01
Create Date: 2026-10-23 00:00:00
"""

from alembic import op


revision = "20261023_01"
down_revision = "20260913_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF EXISTS: databases created from this repo never had them.
    op.execute("DROP TABLE IF EXISTS catamine_shares, catamine_runs, catamine_conversations")
    op.execute("DROP FUNCTION IF EXISTS catamine_terminal_immutable(), catamine_share_immutable()")


def downgrade() -> None:
    # The original schema was never committed; restore from the Railway backup if needed.
    pass
