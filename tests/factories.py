"""Builders for test data: database rows, and planner answers as the LLM would return them."""

import json
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


def plan_step(
    step_id: str,
    *dependencies: str,
    agent: str = "research",
    action: str = "research",
    input: str | None = None,
) -> dict[str, object]:
    """One step of a planner answer, as a dict ready for `plan_json`."""
    return {
        "step_id": step_id,
        "agent": agent,
        "action": action,
        "input": input if input is not None else f"instruction for {step_id}",
        "dependencies": list(dependencies),
    }


def plan_json(*steps: dict[str, object], reasoning: str = "test plan") -> str:
    """A planner answer as the model returns it: JSON text."""
    return json.dumps({"reasoning": reasoning, "steps": list(steps)})


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


def make_runnable_plan(
    task_id: str, steps: Sequence[tuple[str, Sequence[str]]], *, agent: str = "research"
) -> ExecutionPlan:
    """A plan as the planner stores it: steps (step_id, dependencies) in order, all pending,
    with real parallel groups. Each step's instruction is `do <step_id>`."""
    from orchestrator.planning.graph import parallel_groups

    groups = parallel_groups({step_id: list(deps) for step_id, deps in steps})
    plan = ExecutionPlan(
        task_id=task_id,
        version=1,
        plan={"steps": [step_id for step_id, _ in steps]},
        parallel_groups=groups,
    )
    plan.steps = [
        ExecutionStep(
            step_id=step_id,
            position=position,
            agent=agent,
            action="research",
            input=f"do {step_id}",
            dependencies=list(deps),
            status=StepStatus.PENDING,
        )
        for position, (step_id, deps) in enumerate(steps)
    ]
    return plan


def synthesis_json(
    content: str = "The final answer.",
    *,
    findings: Sequence[tuple[str, Sequence[str]]] = (),
    conflicts: Sequence[tuple[str, Sequence[tuple[str, Sequence[str]]]]] = (),
    limitations: Sequence[str] = (),
) -> str:
    """A synthesis answer as the model returns it. findings: (statement, step_ids);
    conflicts: (topic, [(claim, step_ids), ...])."""
    return json.dumps(
        {
            "content": content,
            "key_findings": [{"statement": s, "step_ids": list(ids)} for s, ids in findings],
            "conflicts": [
                {
                    "topic": topic,
                    "claims": [{"claim": c, "step_ids": list(ids)} for c, ids in claims],
                }
                for topic, claims in conflicts
            ],
            "limitations": list(limitations),
        }
    )
