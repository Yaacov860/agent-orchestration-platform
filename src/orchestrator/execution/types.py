"""What the executor works on and what it reports. Plain data: no database, no planner types.

The executor depends on neither the planning package nor the ORM. It gets a plan as these
shapes, read back from the stored rows, so planning and execution only meet in the database.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from pydantic import JsonValue

from orchestrator.agents.types import AgentOutput, ErrorInfo
from orchestrator.llm.grounding import SearchUsage
from orchestrator.llm.types import TokenUsage

DEADLINE_EXCEEDED = "DeadlineExceeded"
"""Error type of a step stopped, or never started, because the task's deadline passed."""


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    agent: str
    action: str
    instruction: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionJob:
    task_id: str
    goal: str
    steps: tuple[StepSpec, ...]
    parallel_groups: tuple[tuple[str, ...], ...]
    constraints: Mapping[str, JsonValue] = field(default_factory=dict)
    deadline: datetime | None = None
    """When the task must be finished by (timezone-aware). Nothing starts after it, and a
    step still running when it passes is stopped."""


@dataclass(frozen=True)
class StepOutcome:
    """How one step ended, over all its attempts."""

    step_id: str
    agent: str
    status: Literal["completed", "failed", "cancelled"]
    attempts: int
    usage: TokenUsage
    """Tokens of every attempt, failed ones included: they were paid for."""
    execution_time_ms: int
    output: AgentOutput | None = None
    error: ErrorInfo | None = None
    search: SearchUsage = field(default_factory=SearchUsage)
    """Grounded calls and reported search queries of every attempt, failed ones included."""


@dataclass(frozen=True)
class ExecutionSummary:
    outcomes: tuple[StepOutcome, ...]
    skipped: Mapping[str, str]
    """Steps never run, each with the dependency that did not complete."""
    cancelled: tuple[str, ...]
    """Steps stopped or never started because the task was cancelled."""
    expired: tuple[str, ...] = ()
    """Steps never started because the deadline had passed."""

    @property
    def failed(self) -> tuple[StepOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == "failed")

    @property
    def was_cancelled(self) -> bool:
        return bool(self.cancelled)

    @property
    def deadline_exceeded(self) -> bool:
        return bool(self.expired) or any(
            o.error is not None and o.error.type == DEADLINE_EXCEEDED for o in self.outcomes
        )

    @property
    def succeeded(self) -> bool:
        return not (self.failed or self.skipped or self.cancelled or self.expired)


class StepRecorder(Protocol):
    """Where the executor reports progress (the database, in production)."""

    async def step_started(self, step_id: str, attempt: int) -> None: ...

    async def step_finished(self, outcome: StepOutcome) -> None: ...

    async def steps_skipped(self, blocked_by: Mapping[str, str]) -> None: ...

    async def steps_expired(self, step_ids: Sequence[str]) -> None:
        """Never started: the task's deadline passed first."""
        ...

    async def steps_cancelled(self, step_ids: Sequence[str]) -> None: ...

    async def cancel_requested(self) -> bool:
        """Was the task cancelled elsewhere (another process)? Checked between groups."""
        ...
