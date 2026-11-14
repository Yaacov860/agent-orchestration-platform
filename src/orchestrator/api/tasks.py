from typing import Any

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from orchestrator.api.deps import TaskServiceDep
from orchestrator.api.schemas import (
    NoResultOut,
    TaskCancelledOut,
    TaskCreate,
    TaskResultOut,
    TaskStatusOut,
    TaskStepsOut,
    TaskSubmitted,
    TaskTraceOut,
)
from orchestrator.domain.status import StepStatus

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_task(
    body: TaskCreate, response: Response, service: TaskServiceDep
) -> TaskSubmitted:
    """Store the task and return its id at once. Planning, execution and synthesis follow."""
    task = await service.submit(
        goal=body.goal,
        constraints=body.constraints.model_dump(exclude_none=True),
        output_format=body.output_format,
        deadline=body.deadline,
    )
    response.headers["Location"] = f"/tasks/{task.id}"
    return TaskSubmitted(task_id=task.id, status=task.status)


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str, service: TaskServiceDep) -> TaskCancelledOut:
    """Stop a task that has not finished. 409 if it already completed, failed or was cancelled.
    Steps that already completed keep their results, and are listed in `completed_steps`."""
    task = await service.cancel(task_id)
    steps = await service.get_steps(task_id)
    completed = (
        [s.step_id for s in steps.plan.steps if s.status is StepStatus.COMPLETED]
        if steps.plan
        else []
    )
    return TaskCancelledOut(task_id=task.id, status=task.status, completed_steps=completed)


@router.get("/{task_id}")
async def get_task(task_id: str, service: TaskServiceDep) -> TaskStatusOut:
    """Current state, read from the database on every call, with its execution trace."""
    snapshot = await service.get_snapshot(task_id)
    trace = TaskTraceOut.from_trace(await service.get_trace(task_id))
    return TaskStatusOut.from_snapshot(snapshot, execution_trace=trace.timeline)


@router.get("/{task_id}/steps")
async def get_task_steps(task_id: str, service: TaskServiceDep) -> TaskStepsOut:
    """Every step of the newest plan: state, attempts, full output or error, tokens, timings.
    Empty `steps` (and `plan_version: null`) until a plan is stored."""
    return TaskStepsOut.from_steps(await service.get_steps(task_id))


_RESULT_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"description": "No such task."},
    409: {
        "model": NoResultOut,
        "description": "No result: the task is not completed yet, was cancelled, or failed "
        "(`failure_stage` says whether before synthesis or during it).",
    },
}


@router.get("/{task_id}/result", response_model=TaskResultOut, responses=_RESULT_RESPONSES)
async def get_task_result(task_id: str, service: TaskServiceDep) -> TaskResultOut | JSONResponse:
    """The final answer of a completed task: content, key findings with the steps they came
    from, the sources the agents actually returned, confidence and how it was computed,
    conflicts between steps, and limitations; and, as the assignment specifies, `result`
    (content, format, word count), the execution trace, total tokens and total time."""
    view = await service.get_result(task_id)
    if view.result is None:
        no_result = NoResultOut.from_view(view)
        return JSONResponse(status_code=409, content=no_result.model_dump(mode="json"))
    trace = TaskTraceOut.from_trace(await service.get_trace(task_id))
    return TaskResultOut.from_row(
        view.task.id, view.result, output_format=view.task.output_format, trace=trace
    )


@router.get("/{task_id}/trace")
async def get_task_trace(task_id: str, service: TaskServiceDep) -> TaskTraceOut:
    """What happened, in time order: every status change (with errors and failure stage),
    each step's start and end (attempts, tokens, time), the stored result; stage durations
    and token usage by stage."""
    return TaskTraceOut.from_trace(await service.get_trace(task_id))
