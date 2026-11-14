from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    PLANNED = "planned"
    """A validated plan is stored; execution has not started yet."""
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    """Every step has finished; the step outputs are being combined into the final result."""
    COMPLETED = "completed"
    """The final result is stored (GET /tasks/{id}/result). Reachable only from synthesizing."""
    FAILED = "failed"
    CANCELLED = "cancelled"


TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.PLANNING: frozenset({TaskStatus.PLANNED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.PLANNED: frozenset({TaskStatus.EXECUTING, TaskStatus.CANCELLED}),
    # Not straight to completed: a task is completed only once its result is stored.
    TaskStatus.EXECUTING: frozenset(
        {TaskStatus.SYNTHESIZING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.SYNTHESIZING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    # Terminal: a finished, failed or cancelled task never changes again.
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}
"""Every status change the system may make. Anything else is a bug and is refused before it
reaches the database (see TaskRepository.transition)."""


class FailureStage(StrEnum):
    """Where a failed task failed: tells "no result because execution failed" apart from
    "execution worked, but combining the outputs did not"."""

    PLANNING = "planning"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"


class InvalidTransitionError(Exception):
    def __init__(self, current: TaskStatus, new: TaskStatus) -> None:
        super().__init__(f"a task cannot move from '{current}' to '{new}'")
        self.current = current
        self.new = new


def check_transition(current: TaskStatus, new: TaskStatus) -> None:
    if new not in TASK_TRANSITIONS[current]:
        raise InvalidTransitionError(current, new)


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Never ran: a step it depends on did not complete, or the deadline passed first.
    `error.type` says which (DependencyNotCompleted or DeadlineExceeded)."""
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = frozenset(s for s, targets in TASK_TRANSITIONS.items() if not targets)
CANCELLABLE_TASK_STATUSES = frozenset(
    s for s, targets in TASK_TRANSITIONS.items() if TaskStatus.CANCELLED in targets
)

STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset(
        {
            StepStatus.RUNNING,
            StepStatus.FAILED,  # could not even start (its agent does not exist)
            StepStatus.SKIPPED,
            StepStatus.CANCELLED,
        }
    ),
    # running -> running is a retry: the same step, the next attempt.
    StepStatus.RUNNING: frozenset(
        {StepStatus.RUNNING, StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED}
    ),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
}


class InvalidStepTransitionError(Exception):
    def __init__(self, current: StepStatus, new: StepStatus) -> None:
        super().__init__(f"a step cannot move from '{current}' to '{new}'")


def check_step_transition(current: StepStatus, new: StepStatus) -> None:
    if new not in STEP_TRANSITIONS[current]:
        raise InvalidStepTransitionError(current, new)
