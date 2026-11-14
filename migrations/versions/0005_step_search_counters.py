"""step search counters

execution_steps.grounded_calls: provider calls sent with web search attached, over every
attempt (each one took a slot of the task's grounded-call budget).
execution_steps.search_queries: the search queries the provider reported for those calls.
Written by hand: autogenerate does not detect CHECK constraints.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-14 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "ck_execution_steps_search_counters_non_negative"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "execution_steps",
        sa.Column("grounded_calls", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column(
        "execution_steps",
        sa.Column("search_queries", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.create_check_constraint(
        op.f(CONSTRAINT), "execution_steps", "grounded_calls >= 0 AND search_queries >= 0"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f(CONSTRAINT), "execution_steps", type_="check")
    op.drop_column("execution_steps", "search_queries")
    op.drop_column("execution_steps", "grounded_calls")
