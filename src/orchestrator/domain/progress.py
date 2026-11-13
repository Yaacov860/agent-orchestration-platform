from collections.abc import Iterable
from dataclasses import dataclass

from orchestrator.domain.status import StepStatus


@dataclass(frozen=True)
class StepState:
    step_id: str
    status: StepStatus


@dataclass(frozen=True)
class Progress:
    total_steps: int
    completed_steps: int
    failed_steps: int
    running_steps: tuple[str, ...]


def compute_progress(steps: Iterable[StepState]) -> Progress:
    """Summarize a plan's steps. Derived from the step rows on every read, never stored,
    so it can never drift out of sync with them."""
    steps = list(steps)
    return Progress(
        total_steps=len(steps),
        completed_steps=sum(s.status == StepStatus.COMPLETED for s in steps),
        failed_steps=sum(s.status == StepStatus.FAILED for s in steps),
        running_steps=tuple(s.step_id for s in steps if s.status == StepStatus.RUNNING),
    )
