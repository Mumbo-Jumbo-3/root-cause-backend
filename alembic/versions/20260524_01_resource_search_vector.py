"""add generated full-text search vector for resource chunks

Revision ID: 20260524_01
Revises: 20260421_02
Create Date: 2026-05-24 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260524_01"
down_revision = "20260421_02"
branch_labels = None
depends_on = None


SEARCH_VECTOR_SQL = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(header_path, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(content, '')), 'C')"
)


def upgrade() -> None:
    op.add_column(
        "agent_resource_chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(SEARCH_VECTOR_SQL, persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_agent_resource_chunks_search_vector",
        "agent_resource_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_resource_chunks_search_vector",
        table_name="agent_resource_chunks",
    )
    op.drop_column("agent_resource_chunks", "search_vector")
