from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task


class TaskRepository:
    """The only place that knows how tasks, plans and steps are queried."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, task: Task) -> None:
        self._session.add(task)

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
