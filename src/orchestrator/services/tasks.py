from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task, TaskEvent, TaskResult
from orchestrator.db.repositories import ResultRepository, StepRepository, TaskRepository
from orchestrator.domain.ids import is_task_id, new_task_id
from orchestrator.domain.progress import Progress, StepState, compute_progress
from orchestrator.domain.status import CANCELLABLE_TASK_STATUSES, StepStatus, TaskStatus
from orchestrator.services.usage import TaskUsage, task_usage


class TaskNotFoundError(Exception):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task '{task_id}' not found")
        self.task_id = task_id


@dataclass(frozen=True)
class TaskSnapshot:
    task: Task
    progress: Progress
    usage: TaskUsage


@dataclass(frozen=True)
class TaskTrace:
    task: Task
    events: Sequence[TaskEvent]
    plans: Sequence[ExecutionPlan]
    steps: Sequence[ExecutionStep]
    """The steps of the newest plan."""
    result: TaskResult | None
    usage: TaskUsage


@dataclass(frozen=True)
class TaskResultView:
    task: Task
    result: TaskResult | None
    """Present exactly when the task is completed (both are written in one transaction)."""


@dataclass(frozen=True)
class TaskSteps:
    task: Task
    plan: ExecutionPlan | None
    """The newest plan, its steps loaded in plan order; None before planning succeeds."""


class TaskNotCancellableError(Exception):
    def __init__(self, task_id: str, status: TaskStatus) -> None:
        allowed = ", ".join(
            s for s in TaskStatus if s in CANCELLABLE_TASK_STATUSES
        )  # lifecycle order
        super().__init__(
            f"task '{task_id}' is already {status}; only {allowed} tasks can be cancelled"
        )
        self.task_id = task_id
        self.status = status


class PlanningScheduler(Protocol):
    """Starts planning a stored task in the background (PlanningService implements it)."""

    def schedule(self, task_id: str) -> None: ...


class CancellationListener(Protocol):
    """Told after a task is cancelled, to stop work in flight (execution and synthesis)."""

    def cancel_requested(self, task_id: str) -> None: ...


class CancellationFanout:
    """Tells every listener; TaskService still knows none of them by name."""

    def __init__(self, *listeners: CancellationListener) -> None:
        self._listeners = listeners

    def cancel_requested(self, task_id: str) -> None:
        for listener in self._listeners:
            listener.cancel_requested(task_id)


class TaskService:
    """Task use cases. Knows nothing about HTTP; the API layer translates to and from it."""

    def __init__(
        self,
        session: AsyncSession,
        planning: PlanningScheduler,
        execution: CancellationListener,
    ) -> None:
        self._session = session
        self._tasks = TaskRepository(session)
        self._planning = planning
        self._execution = execution

    async def submit(
        self,
        *,
        goal: str,
        constraints: dict[str, JsonValue],
        output_format: str,
        deadline: datetime | None,
    ) -> Task:
        task = Task(
            id=new_task_id(),
            goal=goal,
            constraints=constraints,
            output_format=output_format,
            deadline=deadline,
            status=TaskStatus.PENDING,
        )
        self._tasks.add(task)
        await self._session.commit()
        # After the commit, never before: the background job reads the task in its own
        # session, and must find it. The client gets its 201 without waiting for the plan.
        self._planning.schedule(task.id)
        return task

    async def cancel(self, task_id: str) -> Task:
        """Cancel a task that has not finished. The database is changed first, in one
        transaction: the task, and every step still waiting (nobody would ever start them).

        Whatever is in flight then notices: a planner's late plan is discarded (its
        planning -> planned move fails), an executor here stops its running agents at once,
        and an executor in another process stops before its next group.
        """
        if not is_task_id(task_id):
            raise TaskNotFoundError(task_id)
        task = await self._tasks.transition_from_any(
            task_id, expected=CANCELLABLE_TASK_STATUSES, to=TaskStatus.CANCELLED
        )
        if task is None:
            await self._session.rollback()
            existing = await self._tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            raise TaskNotCancellableError(task_id, existing.status)
        await StepRepository(self._session).move_all_of_tasks(
            [task_id], expected=StepStatus.PENDING, to=StepStatus.CANCELLED
        )
        await self._session.commit()
        self._execution.cancel_requested(task_id)
        return task

    async def get_snapshot(self, task_id: str) -> TaskSnapshot:
        task = await self._existing(task_id)
        steps = await self._tasks.latest_plan_steps(task_id)
        progress = compute_progress(StepState(s.step_id, s.status) for s in steps)
        return TaskSnapshot(task=task, progress=progress, usage=await self._usage(task_id))

    async def get_trace(self, task_id: str) -> TaskTrace:
        """Everything that happened to the task, from what the database recorded."""
        task = await self._existing(task_id)
        plan = await self._tasks.latest_plan(task_id)
        return TaskTrace(
            task=task,
            events=await self._tasks.events(task_id),
            plans=await self._tasks.plans(task_id),
            steps=plan.steps if plan else [],
            result=await ResultRepository(self._session).get(task_id),
            usage=await self._usage(task_id),
        )

    async def _usage(self, task_id: str) -> TaskUsage:
        return task_usage(
            plans=await self._tasks.plans(task_id),
            steps=await self._tasks.steps_of_every_plan(task_id),
            events=await self._tasks.events(task_id),
            result=await ResultRepository(self._session).get(task_id),
        )

    async def get_steps(self, task_id: str) -> TaskSteps:
        """Every step of the newest plan with its state, result and error."""
        task = await self._existing(task_id)
        return TaskSteps(task=task, plan=await self._tasks.latest_plan(task_id))

    async def get_result(self, task_id: str) -> TaskResultView:
        task = await self._existing(task_id)
        return TaskResultView(task=task, result=await ResultRepository(self._session).get(task_id))

    async def _existing(self, task_id: str) -> Task:
        if not is_task_id(task_id):
            raise TaskNotFoundError(task_id)  # malformed: no point asking the database
        task = await self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task
