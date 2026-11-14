"""synthesis: 'synthesizing' status, tasks.failure_stage, task_results

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-13 20:00:00.000000

Written by hand, like 0002: autogenerate does not see CHECK changes, and status lists are a
frozen snapshot here rather than read from the (changing) Python enums.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TASK_STATUS_CHECK = "ck_tasks_task_status"  # verified in pg_constraint before writing this
STATUSES_0002 = ("pending", "planning", "planned", "executing", "completed", "failed", "cancelled")
STATUSES_0003 = (
    "pending",
    "planning",
    "planned",
    "executing",
    "synthesizing",
    "completed",
    "failed",
    "cancelled",
)
FAILURE_STAGES = ("planning", "execution", "synthesis")


def _in(column: str, values: Sequence[str]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _replace_status_check(statuses: Sequence[str]) -> None:
    op.drop_constraint(op.f(TASK_STATUS_CHECK), "tasks", type_="check")
    op.create_check_constraint(op.f(TASK_STATUS_CHECK), "tasks", _in("status", statuses))


def upgrade() -> None:
    _replace_status_check(STATUSES_0003)

    op.add_column("tasks", sa.Column("failure_stage", sa.String(length=20), nullable=True))
    op.create_check_constraint(
        op.f("ck_tasks_failure_stage"), "tasks", _in("failure_stage", FAILURE_STAGES)
    )

    op.create_table(
        "task_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.String(length=40), nullable=False),
        sa.Column("plan_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("key_findings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("conflicts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("limitations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_basis", sa.Text(), nullable=False),
        sa.Column("partial", sa.Boolean(), nullable=False),
        sa.Column("synthesized_from", postgresql.ARRAY(sa.String(length=50)), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_task_results_confidence_range"),
        ),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND attempts >= 1",
            name=op.f("ck_task_results_counters_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name=op.f("fk_task_results_task_id_tasks"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["execution_plans.id"],
            name=op.f("fk_task_results_plan_id_execution_plans"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_results")),
        sa.UniqueConstraint("task_id", name=op.f("uq_task_results_task_id")),
    )


def downgrade() -> None:
    # Refuse rather than rewrite data: 0002 has no 'synthesizing' status, and a task in it
    # has no correct older status. Results themselves are dropped with their table.
    blocked = op.get_bind().scalar(
        sa.text("SELECT count(*) FROM tasks WHERE status = 'synthesizing'")
    )
    if blocked:
        raise RuntimeError(
            f"cannot downgrade below 0003: {blocked} task(s) have status 'synthesizing', "
            "which 0002 does not allow. Let them finish, or change them first."
        )
    op.drop_table("task_results")
    op.drop_constraint(op.f("ck_tasks_failure_stage"), "tasks", type_="check")
    op.drop_column("tasks", "failure_stage")
    _replace_status_check(STATUSES_0002)
