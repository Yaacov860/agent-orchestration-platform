import itertools

import pytest

from orchestrator.domain.status import (
    CANCELLABLE_TASK_STATUSES,
    STEP_TRANSITIONS,
    TASK_TRANSITIONS,
    TERMINAL_TASK_STATUSES,
    InvalidStepTransitionError,
    InvalidTransitionError,
    StepStatus,
    TaskStatus,
    check_step_transition,
    check_transition,
)

TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


def test_every_status_has_an_entry() -> None:
    assert set(TASK_TRANSITIONS) == set(TaskStatus)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (TaskStatus.PENDING, TaskStatus.PLANNING),  # claim
        (TaskStatus.PLANNING, TaskStatus.PLANNED),  # plan stored
        (TaskStatus.PLANNING, TaskStatus.FAILED),  # planning failed, or startup recovery
    ],
)
def test_the_transitions_planning_makes_are_allowed(current: TaskStatus, new: TaskStatus) -> None:
    check_transition(current, new)


@pytest.mark.parametrize("terminal", sorted(TERMINAL))
def test_a_terminal_task_never_changes_again(terminal: TaskStatus) -> None:
    for new in TaskStatus:
        with pytest.raises(InvalidTransitionError):
            check_transition(terminal, new)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (TaskStatus.PENDING, TaskStatus.PLANNED),  # skipping planning
        (TaskStatus.PENDING, TaskStatus.FAILED),  # recovery must reschedule pending, not fail it
        (TaskStatus.PLANNED, TaskStatus.PENDING),  # going back
        (TaskStatus.PLANNED, TaskStatus.PLANNING),
        (TaskStatus.PLANNING, TaskStatus.PLANNING),  # a no-op "transition" hides a double claim
        (TaskStatus.FAILED, TaskStatus.PENDING),  # resurrecting a failed task
    ],
)
def test_invalid_transitions_are_refused_with_a_clear_message(
    current: TaskStatus, new: TaskStatus
) -> None:
    with pytest.raises(InvalidTransitionError) as caught:
        check_transition(current, new)

    assert str(caught.value) == f"a task cannot move from '{current}' to '{new}'"


def test_no_status_can_move_to_itself() -> None:
    for status in TaskStatus:
        assert status not in TASK_TRANSITIONS[status]


def test_pending_is_reachable_from_nowhere() -> None:
    # A task is born pending (POST) and never returns there: re-running means a new plan.
    sources = [s for s, targets in TASK_TRANSITIONS.items() if TaskStatus.PENDING in targets]
    assert sources == []


def test_every_non_terminal_status_can_still_reach_a_terminal_one() -> None:
    for status in set(TaskStatus) - TERMINAL:
        reachable: set[TaskStatus] = set()
        frontier = {status}
        while frontier:
            frontier = set(itertools.chain.from_iterable(TASK_TRANSITIONS[s] for s in frontier))
            frontier -= reachable
            reachable |= frontier
        assert reachable & TERMINAL, status


# --- steps -------------------------------------------------------------------------------


def test_every_step_status_has_an_entry() -> None:
    assert set(STEP_TRANSITIONS) == set(StepStatus)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (StepStatus.PENDING, StepStatus.RUNNING),
        (StepStatus.RUNNING, StepStatus.RUNNING),  # a retry
        (StepStatus.RUNNING, StepStatus.COMPLETED),
        (StepStatus.RUNNING, StepStatus.FAILED),
        (StepStatus.PENDING, StepStatus.FAILED),  # its agent does not exist
        (StepStatus.PENDING, StepStatus.SKIPPED),
        (StepStatus.PENDING, StepStatus.CANCELLED),
        (StepStatus.RUNNING, StepStatus.CANCELLED),
    ],
)
def test_the_step_transitions_the_executor_makes_are_allowed(
    current: StepStatus, new: StepStatus
) -> None:
    check_step_transition(current, new)


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (StepStatus.PENDING, StepStatus.COMPLETED),  # completing without ever running
        (StepStatus.COMPLETED, StepStatus.RUNNING),  # re-running a finished step
        (StepStatus.SKIPPED, StepStatus.RUNNING),
        (StepStatus.CANCELLED, StepStatus.COMPLETED),  # a late result after a cancel
        (StepStatus.FAILED, StepStatus.COMPLETED),
    ],
)
def test_invalid_step_transitions_are_refused(current: StepStatus, new: StepStatus) -> None:
    with pytest.raises(InvalidStepTransitionError, match=f"from '{current}' to '{new}'"):
        check_step_transition(current, new)


def test_cancellable_tasks_are_exactly_the_unfinished_ones() -> None:
    assert CANCELLABLE_TASK_STATUSES == set(TaskStatus) - TERMINAL
    assert TERMINAL_TASK_STATUSES == TERMINAL
