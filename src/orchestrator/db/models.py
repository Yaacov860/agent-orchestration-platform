"""Database tables.

  tasks (1) ──< execution_plans (N) ──< execution_steps (N)
    │
    └── task_results (0..1)
  (and task_events: every status change, in order)

- tasks: what the user asked for, and its overall status.
- execution_plans: what the planner proposed. Written once, never edited; a re-plan adds a
  new row with a higher `version`, so earlier plans stay inspectable.
- execution_steps: what actually happened to each step of a plan. Updated as it runs.
- task_events: every status change of a task, in order: its trace (append-only).
- task_results: the final, synthesized answer. At most one per task, written in the same
  transaction that marks the task completed, so `completed` always has a result.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import JsonValue
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
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

from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus

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
    failure_stage: Mapped[FailureStage | None] = mapped_column(
        status_column(FailureStage, "failure_stage")
    )
    """Set with status `failed`: planning, execution or synthesis."""
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
        CheckConstraint(
            "planner_input_tokens >= 0 AND planner_output_tokens >= 0 AND planner_attempts >= 1",
            name="planner_counters_valid",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    plan: Mapped[dict[str, JsonValue]] = mapped_column(JSONB)
    """The planner's output exactly as validated: the record of what was planned."""
    parallel_groups: Mapped[list[list[str]]] = mapped_column(JSONB)
    planner_input_tokens: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    planner_output_tokens: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    """Tokens of every planner call behind this plan: rejected answers and retries included."""
    planner_attempts: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    task: Mapped[Task] = relationship(back_populates="plans", lazy="raise")
    steps: Mapped[list["ExecutionStep"]] = relationship(
        back_populates="plan",
        lazy="raise",
        passive_deletes=True,
        order_by="ExecutionStep.position",  # plan order whenever steps are loaded
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


class TaskResult(Base):
    """The final answer of a completed task, with its provenance.

    One row per task (UNIQUE task_id): a re-run or a recovery can never add a second one.
    The JSON columns hold lists of plain objects; their shape is defined and validated by
    orchestrator.synthesis.types before anything is written.
    """

    __tablename__ = "task_results"
    __table_args__ = (
        UniqueConstraint("task_id"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"
        ),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND attempts >= 1",
            name="counters_valid",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    plan_id: Mapped[UUID] = mapped_column(ForeignKey("execution_plans.id", ondelete="CASCADE"))
    """The plan whose step outputs this result was synthesized from."""
    content: Mapped[str] = mapped_column(Text)
    key_findings: Mapped[list[dict[str, JsonValue]]] = mapped_column(JSONB)
    sources: Mapped[list[dict[str, JsonValue]]] = mapped_column(JSONB)
    conflicts: Mapped[list[dict[str, JsonValue]]] = mapped_column(JSONB)
    limitations: Mapped[list[str]] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_basis: Mapped[str] = mapped_column(Text)
    """How `confidence` was computed, in words: it is a heuristic, so it explains itself."""
    partial: Mapped[bool] = mapped_column(Boolean)
    """True when some steps did not complete: the answer rests on part of the plan."""
    synthesized_from: Mapped[list[str]] = mapped_column(ARRAY(String(50)))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer)
    """Synthesis model calls that produced this result (2 = one repair was needed)."""
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TaskEvent(Base):
    """One status change of a task. Written by the repository in the same transaction as the
    change itself, so the trace can neither miss a change nor record one that rolled back."""

    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    """Insertion order: the order of events, even when two share a timestamp."""
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("clock_timestamp()")
    )
    """clock_timestamp(), not now(): now() is the transaction's start, the same for every
    event of one transaction; the trace wants when each change actually happened."""
    status: Mapped[TaskStatus] = mapped_column(status_column(TaskStatus, "event_status"))
    error: Mapped[str | None] = mapped_column(Text)
    failure_stage: Mapped[FailureStage | None] = mapped_column(
        status_column(FailureStage, "event_failure_stage")
    )
    detail: Mapped[dict[str, JsonValue] | None] = mapped_column(JSONB)
    """Extra facts about the change, e.g. tokens spent by a stage that failed."""
