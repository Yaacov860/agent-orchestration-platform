"""HTTP request/response shapes. Validation happens here, before any of our code runs."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
)

from orchestrator.domain.status import TaskStatus
from orchestrator.services.tasks import TaskSnapshot

OutputFormat = Literal["markdown", "text", "json"]


class TaskConstraints(BaseModel):
    """Known constraints are typed and checked; extra ones are allowed but must be JSON.

    Strict types on purpose: lax mode would accept "yes" as True and "1500" as 1500,
    hiding client bugs instead of reporting them.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    max_words: StrictInt | None = Field(default=None, gt=0, le=20_000)
    tone: str | None = Field(default=None, min_length=1, max_length=50)
    include_code_examples: StrictBool | None = None


class TaskCreate(BaseModel):
    # forbid: a typo such as "constraint" is a 422, not a silently dropped field.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1, max_length=4_000)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)
    output_format: OutputFormat = "markdown"
    deadline: AwareDatetime | None = None
    """Must carry a timezone: a naive time is ambiguous."""

    @field_validator("deadline")
    @classmethod
    def deadline_in_future(cls, value: datetime | None) -> datetime | None:
        if value is not None and value <= datetime.now(UTC):
            raise ValueError("deadline must be in the future")
        return value


class TaskSubmitted(BaseModel):
    task_id: str
    status: TaskStatus


class ProgressOut(BaseModel):
    total_steps: int
    completed_steps: int
    failed_steps: int
    current_step: str | None
    """The assignment's field. With parallel steps several run at once: see running_steps."""
    running_steps: list[str]


class TaskStatusOut(BaseModel):
    task_id: str
    status: TaskStatus
    goal: str
    constraints: dict[str, JsonValue]
    output_format: str
    deadline: datetime | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    progress: ProgressOut

    @classmethod
    def from_snapshot(cls, snapshot: TaskSnapshot) -> "TaskStatusOut":
        task, progress = snapshot.task, snapshot.progress
        return cls(
            task_id=task.id,
            status=task.status,
            goal=task.goal,
            constraints=task.constraints,
            output_format=task.output_format,
            deadline=task.deadline,
            error=task.error,
            created_at=task.created_at,
            updated_at=task.updated_at,
            progress=ProgressOut(
                total_steps=progress.total_steps,
                completed_steps=progress.completed_steps,
                failed_steps=progress.failed_steps,
                current_step=progress.running_steps[0] if progress.running_steps else None,
                running_steps=list(progress.running_steps),
            ),
        )
