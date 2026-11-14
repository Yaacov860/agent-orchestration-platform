"""add the 'planned' task status

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-13 16:00:00.000000

Statuses are VARCHAR + CHECK (see 0001), so adding one means replacing the CHECK constraint;
Alembic autogenerate does not detect CHECK changes, so this migration is written by hand.

The status lists are spelled out here on purpose instead of being read from TaskStatus:
a migration is a frozen snapshot of the schema at one point in time, and must keep producing
the same schema even after the Python enum changes again.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Name verified against the live database (pg_constraint) before writing this migration.
CONSTRAINT = "ck_tasks_task_status"

STATUSES_0001 = ("pending", "planning", "executing", "completed", "failed", "cancelled")
STATUSES_0002 = ("pending", "planning", "planned", "executing", "completed", "failed", "cancelled")


def _status_check(statuses: Sequence[str]) -> str:
    return "status IN (" + ", ".join(f"'{s}'" for s in statuses) + ")"


def _replace_check(statuses: Sequence[str]) -> None:
    # op.f(): the name is final as written; do not run it through the naming convention again.
    op.drop_constraint(op.f(CONSTRAINT), "tasks", type_="check")
    op.create_check_constraint(op.f(CONSTRAINT), "tasks", _status_check(statuses))


def upgrade() -> None:
    _replace_check(STATUSES_0002)


def downgrade() -> None:
    # Refuse rather than silently rewrite data: there is no correct older status for a task
    # that already has a stored plan, so the operator has to decide what happens to it.
    planned = op.get_bind().scalar(sa.text("SELECT count(*) FROM tasks WHERE status = 'planned'"))
    if planned:
        raise RuntimeError(
            f"cannot downgrade below 0002: {planned} task(s) have status 'planned', which 0001 "
            "does not allow. Change or delete them first."
        )
    _replace_check(STATUSES_0001)
