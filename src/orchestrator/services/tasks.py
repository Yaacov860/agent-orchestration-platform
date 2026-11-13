from dataclasses import dataclass
from datetime import datetime

from pydantic import JsonValue
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import Task
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.ids import is_task_id, new_task_id
from orchestrator.domain.progress import Progress, StepState, compute_progress
from orchestrator.domain.status import TaskStatus


class TaskNotFoundError(Exception):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"task '{task_id}' not found")
        self.task_id = task_id


@dataclass(frozen=True)
class TaskSnapshot:
    task: Task
    progress: Progress


class TaskService:
    """Task use cases. Knows nothing about HTTP; the API layer translates to and from it."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._tasks = TaskRepository(session)

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
        # Stage 3 starts planning here, in the background, after the commit.
        return task

    async def get_snapshot(self, task_id: str) -> TaskSnapshot:
        if not is_task_id(task_id):
            raise TaskNotFoundError(task_id)  # malformed: no point asking the database
        task = await self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        steps = await self._tasks.latest_plan_steps(task_id)
        progress = compute_progress(StepState(s.step_id, s.status) for s in steps)
        return TaskSnapshot(task=task, progress=progress)
