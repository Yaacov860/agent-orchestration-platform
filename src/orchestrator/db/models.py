"""Database tables.

  tasks (1) ──< execution_plans (N) ──< execution_steps (N)

- tasks: what the user asked for, and its overall status.
- execution_plans: what the planner proposed. Written once, never edited; a re-plan adds a
  new row with a higher `version`, so earlier plans stay inspectable.
- execution_steps: what actually happened to each step of a plan. Updated as it runs.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from orchestrator.domain.status import StepStatus, TaskStatus

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def status_column(enum: type[StrEnum], name: str) -> Enum:
    """VARCHAR + CHECK (not a native Postgres ENUM, which needs ALTER TYPE to extend).
    Python still gets the enum member back on load."""
    return Enum(
        enum,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda members: [m.value for m in members],
    )


class Task(Base):
    __tablename__ = "tasks"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012 - SQLAlchemy's documented form

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    constraints: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    output_format: Mapped[str] = mapped_column(String(20))
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[TaskStatus] = mapped_column(status_column(TaskStatus, "task_status"), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    plans: Mapped[list["ExecutionPlan"]] = relationship(
        back_populates="task", lazy="raise", passive_deletes=True
    )


class ExecutionPlan(Base):
    __tablename__ = "execution_plans"
    __table_args__ = (
        UniqueConstraint("task_id", "version"),
        CheckConstraint("version >= 1", name="version_positive"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    plan: Mapped[dict[str, JsonValue]] = mapped_column(JSONB)
    """The planner's output exactly as validated: the record of what was planned."""
    parallel_groups: Mapped[list[list[str]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    task: Mapped[Task] = relationship(back_populates="plans", lazy="raise")
    steps: Mapped[list["ExecutionStep"]] = relationship(
        back_populates="plan", lazy="raise", passive_deletes=True
    )


class ExecutionStep(Base):
    __tablename__ = "execution_steps"
    __table_args__ = (
        UniqueConstraint("plan_id", "step_id"),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND attempts >= 0",
            name="counters_non_negative",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    plan_id: Mapped[UUID] = mapped_column(ForeignKey("execution_plans.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(String(50))
    position: Mapped[int] = mapped_column(Integer)
    """Order in the plan as written; sorting by step_id would put step_10 before step_2."""
    agent: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(50))
    input: Mapped[str] = mapped_column(Text)
    dependencies: Mapped[list[str]] = mapped_column(ARRAY(String(50)), server_default=text("'{}'"))
    status: Mapped[StepStatus] = mapped_column(status_column(StepStatus, "step_status"))
    output: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB)
    error: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB)
    input_tokens: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    execution_time_ms: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    plan: Mapped[ExecutionPlan] = relationship(back_populates="steps", lazy="raise")
