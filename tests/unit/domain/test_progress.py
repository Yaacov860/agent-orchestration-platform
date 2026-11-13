from orchestrator.domain.progress import Progress, StepState, compute_progress
from orchestrator.domain.status import StepStatus


def test_no_steps_means_zero_progress() -> None:
    assert compute_progress([]) == Progress(0, 0, 0, ())


def test_counts_each_status_and_lists_running_steps_in_plan_order() -> None:
    steps = [
        StepState("step_1", StepStatus.COMPLETED),
        StepState("step_2", StepStatus.RUNNING),
        StepState("step_3", StepStatus.FAILED),
        StepState("step_4", StepStatus.RUNNING),
        StepState("step_5", StepStatus.SKIPPED),
        StepState("step_6", StepStatus.PENDING),
    ]

    assert compute_progress(steps) == Progress(
        total_steps=6, completed_steps=1, failed_steps=1, running_steps=("step_2", "step_4")
    )
