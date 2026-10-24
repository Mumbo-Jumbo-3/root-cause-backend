"""catamine integration (never committed; superseded by 20261023_01)

Production was migrated to this revision from an uncommitted build that added the
catamine_conversations, catamine_runs and catamine_shares tables plus the
catamine_terminal_immutable and catamine_share_immutable trigger functions. The
integration was retired, so this stub only keeps that history resolvable;
20261023_01 removes whatever the original created.

Revision ID: 20260913_01
Revises: 20260524_01
Create Date: 2026-09-13 00:00:00
"""


revision = "20260913_01"
down_revision = "20260524_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
