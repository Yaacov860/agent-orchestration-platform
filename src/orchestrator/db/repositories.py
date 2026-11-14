from collections.abc import Collection, Sequence
from typing import Any
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task, TaskEvent, TaskResult
from orchestrator.domain.status import (
    FailureStage,
    StepStatus,
    TaskStatus,
    check_step_transition,
    check_transition,
)


class TaskRepository:
    """The only place that knows how tasks, plans and steps are queried."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, task: Task) -> None:
        """A new task, and the first event of its trace."""
        self._session.add(task)
        self._session.add(TaskEvent(task_id=task.id, status=task.status))

    async def get(self, task_id: str) -> Task | None:
        return await self._session.get(Task, task_id)

    async def latest_plan_steps(self, task_id: str) -> Sequence[ExecutionStep]:
        """Steps of the task's newest plan, in plan order. Empty if it has no plan yet."""
        latest_version = (
            select(func.max(ExecutionPlan.version))
            .where(ExecutionPlan.task_id == task_id)
            .scalar_subquery()
        )
        query = (
            select(ExecutionStep)
            .join(ExecutionPlan)
            .where(ExecutionPlan.task_id == task_id, ExecutionPlan.version == latest_version)
            .order_by(ExecutionStep.position)
        )
        return (await self._session.scalars(query)).all()

    async def transition(
        self,
        task_id: str,
        *,
        expected: TaskStatus,
        to: TaskStatus,
        error: str | None = None,
        failure_stage: FailureStage | None = None,
        detail: dict[str, JsonValue] | None = None,
    ) -> Task | None:
        """Move a task to `to` only if it is still in `expected`; return it, or None if not.

        One conditional UPDATE, so the check and the write cannot be interleaved by another
        process: of two runners claiming the same pending task, exactly one gets it back, and
        a task that recovery already failed is not resurrected by a late planner.

        Raises InvalidTransitionError, before touching the database, if `expected -> to` is
        not in TASK_TRANSITIONS: that is a bug in the caller, not a race to lose quietly.
        A move to `failed` must say at which stage (and only such a move may).
        """
        check_transition(expected, to)
        _check_failure_stage(to, failure_stage)
        statement = (
            update(Task)
            .where(Task.id == task_id, Task.status == expected)
            .values(status=to, error=error, failure_stage=failure_stage, updated_at=func.now())
            .returning(Task)
        )
        task = (await self._session.scalars(statement)).one_or_none()
        if task is not None:  # recorded only if the change happened, in its transaction
            await self._record([task_id], to, error, failure_stage, detail)
        return task

    async def transition_from_any(
        self, task_id: str, *, expected: Collection[TaskStatus], to: TaskStatus
    ) -> Task | None:
        """Like `transition`, from whichever of `expected` the task is in (used to cancel)."""
        for status in expected:
            check_transition(status, to)
        statement = (
            update(Task)
            .where(Task.id == task_id, Task.status.in_(expected))
            .values(status=to, updated_at=func.now())
            .returning(Task)
        )
        task = (await self._session.scalars(statement)).one_or_none()
        if task is not None:
            await self._record([task_id], to)
        return task

    async def fail_all_in(
        self, status: TaskStatus, error: str, failure_stage: FailureStage
    ) -> Sequence[str]:
        """Fail every task currently in `status`; return their ids."""
        check_transition(status, TaskStatus.FAILED)
        statement = (
            update(Task)
            .where(Task.status == status)
            .values(
                status=TaskStatus.FAILED,
                error=error,
                failure_stage=failure_stage,
                updated_at=func.now(),
            )
            .returning(Task.id)
        )
        failed = (await self._session.scalars(statement)).all()
        await self._record(failed, TaskStatus.FAILED, error, failure_stage)
        return failed

    async def _record(
        self,
        task_ids: Sequence[str],
        status: TaskStatus,
        error: str | None = None,
        failure_stage: FailureStage | None = None,
        detail: dict[str, JsonValue] | None = None,
    ) -> None:
        if task_ids:
            await self._session.execute(
                insert(TaskEvent),
                [
                    {
                        "task_id": task_id,
                        "status": status,
                        "error": error,
                        "failure_stage": failure_stage,
                        "detail": detail,
                    }
                    for task_id in task_ids
                ],
            )

    async def events(self, task_id: str) -> Sequence[TaskEvent]:
        query = select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.id)
        return (await self._session.scalars(query)).all()

    async def steps_of_every_plan(self, task_id: str) -> Sequence[ExecutionStep]:
        """Every step row of every plan of the task (superseded plans included)."""
        query = (
            select(ExecutionStep)
            .join(ExecutionPlan)
            .where(ExecutionPlan.task_id == task_id)
            .order_by(ExecutionPlan.version, ExecutionStep.position)
        )
        return (await self._session.scalars(query)).all()

    async def plans(self, task_id: str) -> Sequence[ExecutionPlan]:
        query = (
            select(ExecutionPlan)
            .where(ExecutionPlan.task_id == task_id)
            .order_by(ExecutionPlan.version)
        )
        return (await self._session.scalars(query)).all()

    async def ids_in(self, status: TaskStatus) -> Sequence[str]:
        """Ids of every task in `status`, oldest first (first come, first served)."""
        query = select(Task.id).where(Task.status == status).order_by(Task.created_at, Task.id)
        return (await self._session.scalars(query)).all()

    async def next_plan_version(self, task_id: str) -> int:
        current = await self._session.scalar(
            select(func.max(ExecutionPlan.version)).where(ExecutionPlan.task_id == task_id)
        )
        return (current or 0) + 1

    def add_plan(self, plan: ExecutionPlan) -> None:
        self._session.add(plan)

    async def latest_plan(self, task_id: str) -> ExecutionPlan | None:
        """The task's newest plan with its steps loaded, in plan order."""
        query = (
            select(ExecutionPlan)
            .where(ExecutionPlan.task_id == task_id)
            .order_by(ExecutionPlan.version.desc())
            .limit(1)
            .options(selectinload(ExecutionPlan.steps))
        )
        return await self._session.scalar(query)


def _check_failure_stage(to: TaskStatus, stage: FailureStage | None) -> None:
    if (to is TaskStatus.FAILED) != (stage is not None):
        raise ValueError(
            "a failure stage goes with status 'failed', and only with it "
            f"(got status '{to}', stage {stage!r})"
        )


class ResultRepository:
    """Final results. At most one per task: the table's UNIQUE(task_id) enforces it."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, result: TaskResult) -> None:
        self._session.add(result)

    async def get(self, task_id: str) -> TaskResult | None:
        return await self._session.scalar(select(TaskResult).where(TaskResult.task_id == task_id))


class StepRepository:
    """Step state changes. Each is a conditional UPDATE checked against STEP_TRANSITIONS."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def move(
        self,
        plan_id: UUID,
        step_ids: Collection[str],
        *,
        expected: Collection[StepStatus],
        to: StepStatus,
        **values: Any,
    ) -> int:
        """Move the given steps of a plan to `to`, only those still in `expected`.
        Returns how many rows changed: fewer than asked means another writer got there first
        (a cancellation, say), and that writer's state wins."""
        for status in expected:
            check_step_transition(status, to)
        if not step_ids:
            return 0
        statement = (
            update(ExecutionStep)
            .where(
                ExecutionStep.plan_id == plan_id,
                ExecutionStep.step_id.in_(step_ids),
                ExecutionStep.status.in_(expected),
            )
            .values(status=to, **values)
            .returning(ExecutionStep.id)
        )
        return len((await self._session.scalars(statement)).all())

    async def move_all_of_tasks(
        self,
        task_ids: Collection[str],
        *,
        expected: StepStatus,
        to: StepStatus,
        **values: Any,
    ) -> int:
        """Move every step in `expected`, in any plan of the given tasks."""
        check_step_transition(expected, to)
        if not task_ids:
            return 0
        plans = select(ExecutionPlan.id).where(ExecutionPlan.task_id.in_(task_ids))
        statement = (
            update(ExecutionStep)
            .where(ExecutionStep.plan_id.in_(plans), ExecutionStep.status == expected)
            .values(status=to, **values)
            .returning(ExecutionStep.id)
        )
        return len((await self._session.scalars(statement)).all())
