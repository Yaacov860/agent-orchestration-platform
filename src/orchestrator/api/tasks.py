from fastapi import APIRouter, Response, status

from orchestrator.api.deps import TaskServiceDep
from orchestrator.api.schemas import TaskCreate, TaskStatusOut, TaskSubmitted

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_task(
    body: TaskCreate, response: Response, service: TaskServiceDep
) -> TaskSubmitted:
    """Store the task and return its id at once. Planning and execution happen afterwards."""
    task = await service.submit(
        goal=body.goal,
        constraints=body.constraints.model_dump(exclude_none=True),
        output_format=body.output_format,
        deadline=body.deadline,
    )
    response.headers["Location"] = f"/tasks/{task.id}"
    return TaskSubmitted(task_id=task.id, status=task.status)


@router.get("/{task_id}")
async def get_task(task_id: str, service: TaskServiceDep) -> TaskStatusOut:
    """Current state, read from the database on every call."""
    return TaskStatusOut.from_snapshot(await service.get_snapshot(task_id))
