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

from orchestrator.agents.catalog import AgentSpec
from orchestrator.db.models import ExecutionStep, TaskResult
from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus
from orchestrator.llm.types import TokenUsage
from orchestrator.services.tasks import TaskResultView, TaskSnapshot, TaskSteps, TaskTrace
from orchestrator.services.usage import TaskUsage
from orchestrator.synthesis import Conflict, KeyFinding, ResultSource

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


class TokenCountOut(BaseModel):
    input: int
    output: int
    total: int

    @classmethod
    def of(cls, usage: TokenUsage) -> "TokenCountOut":
        return cls(input=usage.input_tokens, output=usage.output_tokens, total=usage.total)


class UsageOut(BaseModel):
    """Tokens by stage, failed and retried calls included (they were paid for)."""

    planning: TokenCountOut
    execution: TokenCountOut
    synthesis: TokenCountOut
    total: TokenCountOut

    @classmethod
    def of(cls, usage: TaskUsage) -> "UsageOut":
        return cls(
            planning=TokenCountOut.of(usage.planning),
            execution=TokenCountOut.of(usage.execution),
            synthesis=TokenCountOut.of(usage.synthesis),
            total=TokenCountOut.of(usage.total),
        )


class TaskStatusOut(BaseModel):
    task_id: str
    status: TaskStatus
    goal: str
    constraints: dict[str, JsonValue]
    output_format: str
    deadline: datetime | None
    error: str | None
    failure_stage: FailureStage | None
    """With status `failed`: planning, execution or synthesis."""
    created_at: datetime
    updated_at: datetime
    progress: ProgressOut
    usage: UsageOut

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
            failure_stage=task.failure_stage,
            created_at=task.created_at,
            updated_at=task.updated_at,
            progress=ProgressOut(
                total_steps=progress.total_steps,
                completed_steps=progress.completed_steps,
                failed_steps=progress.failed_steps,
                current_step=progress.running_steps[0] if progress.running_steps else None,
                running_steps=list(progress.running_steps),
            ),
            usage=UsageOut.of(snapshot.usage),
        )


class SourceOut(BaseModel):
    title: str
    url: str


class StepOutputOut(BaseModel):
    content: str
    """The agent's full output, as stored (inputs to later steps may have been cut)."""
    sources: list[SourceOut]
    confidence: float | None


class StepErrorOut(BaseModel):
    type: str
    message: str
    retryable: bool


class TokensOut(BaseModel):
    input: int
    output: int
    total: int


class StepOut(BaseModel):
    step_id: str
    agent: str
    action: str
    input: str
    """The step's instruction, from the plan."""
    dependencies: list[str]
    status: StepStatus
    attempts: int
    output: StepOutputOut | None
    error: StepErrorOut | None
    tokens: TokensOut
    """Over every attempt."""
    execution_time_ms: int | None
    """Time spent in the agent over every attempt, without waits for a slot or a retry."""
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_row(cls, row: ExecutionStep) -> "StepOut":
        return cls(
            step_id=row.step_id,
            agent=row.agent,
            action=row.action,
            input=row.input,
            dependencies=list(row.dependencies),
            status=row.status,
            attempts=row.attempts,
            output=StepOutputOut.model_validate(row.output) if row.output else None,
            error=StepErrorOut.model_validate(row.error) if row.error else None,
            tokens=TokensOut(
                input=row.input_tokens,
                output=row.output_tokens,
                total=row.input_tokens + row.output_tokens,
            ),
            execution_time_ms=row.execution_time_ms,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )


class TaskStepsOut(BaseModel):
    task_id: str
    status: TaskStatus
    plan_version: int | None
    """None until a plan is stored."""
    parallel_groups: list[list[str]]
    steps: list[StepOut]
    """In plan order."""

    @classmethod
    def from_steps(cls, result: TaskSteps) -> "TaskStepsOut":
        plan = result.plan
        return cls(
            task_id=result.task.id,
            status=result.task.status,
            plan_version=plan.version if plan else None,
            parallel_groups=plan.parallel_groups if plan else [],
            steps=[StepOut.from_row(row) for row in plan.steps] if plan else [],
        )


class SynthesisUsageOut(BaseModel):
    attempts: int
    """Model calls that produced the result (2 means one repair was needed)."""
    tokens: TokensOut


class TaskResultOut(BaseModel):
    """The final answer, with where each part came from and how far to trust it."""

    task_id: str
    status: TaskStatus
    content: str
    key_findings: list[KeyFinding]
    """Each with the steps it came from and `verification`: corroborated (two independent
    steps), single_step (not verified), or unattributed."""
    sources: list[ResultSource]
    """Exactly the pages the agents' providers returned, each with the steps that got it."""
    confidence: float | None
    confidence_basis: str
    """How `confidence` was computed: a heuristic that explains itself."""
    conflicts: list[Conflict]
    """Every disagreement between steps, with all sides and their steps. None is resolved
    by picking a side."""
    limitations: list[str]
    partial: bool
    """True when some steps did not complete: the answer rests on part of the plan."""
    synthesized_from: list[str]
    synthesis: SynthesisUsageOut
    created_at: datetime

    @classmethod
    def from_row(cls, task_id: str, row: TaskResult) -> "TaskResultOut":
        return cls(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            content=row.content,
            key_findings=[KeyFinding.model_validate(f) for f in row.key_findings],
            sources=[ResultSource.model_validate(s) for s in row.sources],
            confidence=row.confidence,
            confidence_basis=row.confidence_basis,
            conflicts=[Conflict.model_validate(c) for c in row.conflicts],
            limitations=list(row.limitations),
            partial=row.partial,
            synthesized_from=list(row.synthesized_from),
            synthesis=SynthesisUsageOut(
                attempts=row.attempts,
                tokens=TokensOut(
                    input=row.input_tokens,
                    output=row.output_tokens,
                    total=row.input_tokens + row.output_tokens,
                ),
            ),
            created_at=row.created_at,
        )


class NoResultOut(BaseModel):
    """Why a task has no result (yet). `failure_stage` tells a task that failed before
    synthesis (planning, execution) from one whose synthesis failed."""

    task_id: str
    status: TaskStatus
    failure_stage: FailureStage | None
    error: str | None
    detail: str

    @classmethod
    def from_view(cls, view: TaskResultView) -> "NoResultOut":
        task = view.task
        return cls(
            task_id=task.id,
            status=task.status,
            failure_stage=task.failure_stage,
            error=task.error,
            detail=_no_result_detail(task.id, task.status, task.failure_stage),
        )


def _no_result_detail(task_id: str, status: TaskStatus, stage: FailureStage | None) -> str:
    steps = f"/tasks/{task_id}/steps"
    if status is TaskStatus.CANCELLED:
        return f"the task was cancelled, so it has no result; finished step outputs are at {steps}"
    if status is TaskStatus.FAILED and stage is FailureStage.SYNTHESIS:
        return (
            "the steps ran, but combining their outputs into a final result failed (see "
            f"error); the step outputs are at {steps}"
        )
    if status is TaskStatus.FAILED:
        return (
            f"the task failed during {stage or 'processing'}, before a result could be "
            "synthesized (see error); no final result exists"
        )
    return f"the task is {status}; its result is available once it is completed"


# --- GET /agents ---------------------------------------------------------------------


class ActionOut(BaseModel):
    name: str
    description: str


class AgentOut(BaseModel):
    name: str
    description: str
    actions: list[ActionOut]
    searches_web: bool
    """Only an agent that searches can return grounded sources."""

    @classmethod
    def of(cls, spec: AgentSpec) -> "AgentOut":
        actions = spec.actions or tuple((a, "") for a in sorted(spec.capabilities))
        return cls(
            name=spec.name,
            description=spec.description,
            actions=[ActionOut(name=a, description=d) for a, d in actions],
            searches_web=spec.searches_web,
        )


# --- GET /tasks/{id}/trace -----------------------------------------------------------


class TraceEntryOut(BaseModel):
    at: datetime
    source: Literal["task", "step", "result"]
    event: str
    """task: the status entered; step: started / completed / failed / cancelled; result: stored."""
    status: str | None = None
    step_id: str | None = None
    agent: str | None = None
    action: str | None = None
    attempts: int | None = None
    error: str | None = None
    failure_stage: FailureStage | None = None
    tokens: TokenCountOut | None = None
    duration_ms: int | None = None


class NotRunStepOut(BaseModel):
    step_id: str
    agent: str
    action: str
    status: StepStatus
    reason: str | None


class StageDurationsOut(BaseModel):
    """Milliseconds each stage took, from the trace events; null while it has not ended."""

    planning: int | None
    execution: int | None
    synthesis: int | None
    total: int | None


class TaskTraceOut(BaseModel):
    task_id: str
    status: TaskStatus
    durations_ms: StageDurationsOut
    usage: UsageOut
    timeline: list[TraceEntryOut]
    """In time order: every status change, every step start and end, the stored result."""
    not_run: list[NotRunStepOut]
    """Steps that never started (skipped, cancelled or expired), with why."""

    @classmethod
    def from_trace(cls, trace: TaskTrace) -> "TaskTraceOut":
        timeline: list[TraceEntryOut] = [
            TraceEntryOut(
                at=e.at,
                source="task",
                event=e.status.value,
                status=e.status.value,
                error=e.error,
                failure_stage=e.failure_stage,
            )
            for e in trace.events
        ]
        not_run: list[NotRunStepOut] = []
        for step in trace.steps:
            if step.started_at is None:
                if step.status is not StepStatus.PENDING:
                    not_run.append(
                        NotRunStepOut(
                            step_id=step.step_id,
                            agent=step.agent,
                            action=step.action,
                            status=step.status,
                            reason=str(step.error.get("message")) if step.error else None,
                        )
                    )
                continue
            timeline.append(
                TraceEntryOut(
                    at=step.started_at,
                    source="step",
                    event="started",
                    step_id=step.step_id,
                    agent=step.agent,
                    action=step.action,
                )
            )
            if step.completed_at is not None:
                timeline.append(
                    TraceEntryOut(
                        at=step.completed_at,
                        source="step",
                        event=step.status.value,
                        status=step.status.value,
                        attempts=step.attempts,
                        error=str(step.error.get("message")) if step.error else None,
                        tokens=TokenCountOut(
                            input=step.input_tokens,
                            output=step.output_tokens,
                            total=step.input_tokens + step.output_tokens,
                        ),
                        duration_ms=step.execution_time_ms,
                        step_id=step.step_id,
                        agent=step.agent,
                        action=step.action,
                    )
                )
        if trace.result is not None:
            timeline.append(
                TraceEntryOut(
                    at=trace.result.created_at,
                    source="result",
                    event="stored",
                    attempts=trace.result.attempts,
                    tokens=TokenCountOut(
                        input=trace.result.input_tokens,
                        output=trace.result.output_tokens,
                        total=trace.result.input_tokens + trace.result.output_tokens,
                    ),
                )
            )
        timeline.sort(key=lambda entry: entry.at)
        return cls(
            task_id=trace.task.id,
            status=trace.task.status,
            durations_ms=_durations(trace),
            usage=UsageOut.of(trace.usage),
            timeline=timeline,
            not_run=not_run,
        )


def _durations(trace: TaskTrace) -> StageDurationsOut:
    events = list(trace.events)

    def stage(status: TaskStatus) -> int | None:
        for i, event in enumerate(events):
            if event.status is status and i + 1 < len(events):
                return round((events[i + 1].at - event.at).total_seconds() * 1000)
        return None

    terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    total = (
        round((events[-1].at - events[0].at).total_seconds() * 1000)
        if events and events[-1].status in terminal
        else None
    )
    return StageDurationsOut(
        planning=stage(TaskStatus.PLANNING),
        execution=stage(TaskStatus.EXECUTING),
        synthesis=stage(TaskStatus.SYNTHESIZING),
        total=total,
    )
