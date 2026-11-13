"""Builders for rows that later stages will create (plans, steps), so tests can set them up."""

from collections.abc import Sequence

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.domain.ids import new_task_id
from orchestrator.domain.status import StepStatus, TaskStatus


def make_task(**overrides: object) -> Task:
    fields: dict[str, object] = {
        "id": new_task_id(),
        "goal": "Explain asyncio",
        "constraints": {},
        "output_format": "markdown",
        "status": TaskStatus.PENDING,
    }
    return Task(**(fields | overrides))


def make_plan(task_id: str, version: int, steps: Sequence[tuple[str, StepStatus]]) -> ExecutionPlan:
    """A plan whose steps are given as (step_id, status), in plan order."""
    plan = ExecutionPlan(
        task_id=task_id,
        version=version,
        plan={"steps": [step_id for step_id, _ in steps]},
        parallel_groups=[[step_id for step_id, _ in steps]],
    )
    plan.steps = [
        ExecutionStep(
            step_id=step_id,
            position=position,
            agent="research",
            action="research",
            input=f"input of {step_id}",
            status=status,
        )
        for position, (step_id, status) in enumerate(steps)
    ]
    return plan
