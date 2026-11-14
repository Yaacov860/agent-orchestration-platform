"""ExecutionService against the real database: lifecycle, stored results, failures, real
concurrency, cancellation (here and from another process), shutdown and recovery."""

import asyncio
from collections.abc import Sequence

import pytest
from factories import make_runnable_plan, make_task
from fakes import GatedLLM, ManualRunner, StepLLM, output_of, step_of
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.agents import AgentOutput
from orchestrator.agents.factory import AgentFactory
from orchestrator.db.models import ExecutionStep, Task
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus
from orchestrator.llm import (
    LLMClient,
    LLMPermanentError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)
from orchestrator.llm.fake import fake_response
from orchestrator.llm.unconfigured import NO_PROVIDER_MESSAGE, UnconfiguredLLM
from orchestrator.runner import BackgroundRunner, Runner
from orchestrator.services.execution import INTERRUPTED_ERROR, ExecutionService
from orchestrator.services.tasks import TaskNotCancellableError, TaskNotFoundError, TaskService

Sessions = async_sessionmaker[AsyncSession]

DIAMOND = [("step_1", ()), ("step_2", ()), ("step_3", ("step_1", "step_2"))]
CHAIN = [("step_1", ()), ("step_2", ("step_1",)), ("step_3", ("step_2",))]


def service(
    sessionmaker: Sessions, llm: LLMClient, runner: Runner | None = None, **options: object
) -> ExecutionService:
    async def no_sleep(seconds: float) -> None:
        return None

    return ExecutionService(
        sessionmaker,
        AgentFactory(llm),
        runner or ManualRunner(),
        sleep=no_sleep,
        **options,  # type: ignore[arg-type]
    )


async def planned_task(
    sessionmaker: Sessions,
    steps: Sequence[tuple[str, Sequence[str]]],
    *,
    status: TaskStatus = TaskStatus.PLANNED,
    agent: str = "research",
) -> str:
    task = make_task(status=status, goal="Compare Python and JavaScript")
    async with sessionmaker() as session:
        session.add(task)
        await session.flush()
        session.add(make_runnable_plan(task.id, steps, agent=agent))
        await session.commit()
    return task.id


async def task_state(sessionmaker: Sessions, task_id: str) -> tuple[TaskStatus, str | None]:
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
    assert task is not None
    return task.status, task.error


async def step_rows(sessionmaker: Sessions, task_id: str) -> dict[str, ExecutionStep]:
    async with sessionmaker() as session:
        plan = await TaskRepository(session).latest_plan(task_id)
    assert plan is not None
    return {s.step_id: s for s in plan.steps}


async def step_statuses(sessionmaker: Sessions, task_id: str) -> dict[str, StepStatus]:
    return {k: v.status for k, v in (await step_rows(sessionmaker, task_id)).items()}


class _NoPlanning:
    def schedule(self, task_id: str) -> None:
        raise AssertionError("not used here")


async def cancel_via_service(
    sessionmaker: Sessions, execution: ExecutionService, task_id: str
) -> None:
    async with sessionmaker() as session:
        await TaskService(session, _NoPlanning(), execution).cancel(task_id)


class HoldLLM(GatedLLM):
    """Answers at once, except for the `held` steps, which wait for `release`."""

    def __init__(self, *held: str) -> None:
        super().__init__(output_of)
        self._held = set(held)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if step_of(request) not in self._held:
            self.calls.append(request)
            return fake_response(output_of(request), input_tokens=10, output_tokens=5)
        return await super().generate(request)


# --- the happy path ----------------------------------------------------------------------


async def test_a_planned_task_runs_every_step_stores_it_and_hands_over_to_synthesis(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    handed_over: list[str] = []

    await service(sessionmaker, StepLLM(), on_executed=handed_over.append).execute_task(task_id)

    # Not completed: only synthesis completes a task, once its result is stored.
    assert await task_state(sessionmaker, task_id) == (TaskStatus.SYNTHESIZING, None)
    assert handed_over == [task_id]
    rows = await step_rows(sessionmaker, task_id)
    for step_id, row in rows.items():
        assert (row.status, row.attempts, row.error) == (StepStatus.COMPLETED, 1, None)
        assert AgentOutput.model_validate(row.output).content == f"output of {step_id}"
        assert (row.input_tokens, row.output_tokens) == (100, 50)
        assert row.started_at is not None and row.completed_at is not None
        assert row.started_at <= row.completed_at
        assert row.execution_time_ms is not None and row.execution_time_ms >= 0


async def test_while_it_runs_the_database_shows_executing_and_the_running_steps(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    llm = GatedLLM(output_of)
    running = asyncio.create_task(service(sessionmaker, llm).execute_task(task_id))

    await llm.until_in_flight(2)
    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.EXECUTING
    rows = await step_rows(sessionmaker, task_id)
    assert {k: (r.status, r.attempts) for k, r in rows.items()} == {
        "step_1": (StepStatus.RUNNING, 1),
        "step_2": (StepStatus.RUNNING, 1),
        "step_3": (StepStatus.PENDING, 0),
    }
    assert rows["step_1"].started_at is not None

    llm.release.set()
    await running
    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.SYNTHESIZING


async def test_no_transaction_is_held_open_while_agents_run(sessionmaker: Sessions) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    llm = GatedLLM(output_of)
    running = asyncio.create_task(service(sessionmaker, llm).execute_task(task_id))
    await llm.until_in_flight(2)

    async with sessionmaker() as session:
        idle_in_transaction = await session.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND state LIKE 'idle in transaction%' AND pid <> pg_backend_pid()"
            )
        )

    llm.release.set()
    await running
    assert idle_in_transaction == 0


# --- failures ----------------------------------------------------------------------------


async def test_a_failed_step_skips_its_dependents_and_the_rest_goes_on_to_synthesis(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    llm = StepLLM({"step_1": [LLMPermanentError("content blocked")]})
    handed_over: list[str] = []

    await service(sessionmaker, llm, on_executed=handed_over.append).execute_task(task_id)

    # step_2 completed, so there is something to synthesize: a partial result follows.
    assert await task_state(sessionmaker, task_id) == (TaskStatus.SYNTHESIZING, None)
    assert handed_over == [task_id]
    rows = await step_rows(sessionmaker, task_id)
    assert rows["step_1"].status is StepStatus.FAILED
    assert rows["step_1"].error == {
        "type": "LLMPermanentError",
        "message": "content blocked",
        "retryable": False,
    }
    assert rows["step_2"].status is StepStatus.COMPLETED  # independent: still ran
    assert rows["step_3"].status is StepStatus.SKIPPED
    assert rows["step_3"].error == {
        "type": "DependencyNotCompleted",
        "message": "not run: depends on step_1, which did not complete",
        "retryable": False,
    }
    assert rows["step_3"].attempts == 0


async def test_retries_are_stored_as_attempts_with_every_attempts_tokens(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, [("step_1", ())])
    llm = StepLLM(
        {
            "step_1": [
                LLMTransientError("429"),
                fake_response("found", input_tokens=7, output_tokens=3),
            ]
        }
    )

    await service(sessionmaker, llm).execute_task(task_id)

    row = (await step_rows(sessionmaker, task_id))["step_1"]
    assert (row.status, row.attempts) == (StepStatus.COMPLETED, 2)
    # The failed call raised before returning, so it reported no tokens: only the
    # successful call's usage is there to count (tokens a provider does report are summed).
    assert (row.input_tokens, row.output_tokens) == (7, 3)


async def test_without_a_provider_every_step_fails_clearly_and_is_not_retried(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)

    await service(sessionmaker, UnconfiguredLLM()).execute_task(task_id)

    status, error = await task_state(sessionmaker, task_id)
    assert status is TaskStatus.FAILED  # no step completed: nothing to synthesize from
    assert error == (
        f"no step produced a usable result: step_1 failed: {NO_PROVIDER_MESSAGE}; "
        f"step_2 failed: {NO_PROVIDER_MESSAGE}; not run: step_3"
    )
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
    assert task is not None and task.failure_stage is FailureStage.EXECUTION
    rows = await step_rows(sessionmaker, task_id)
    assert rows["step_1"].attempts == 1  # permanent: one attempt only
    assert rows["step_1"].error is not None and rows["step_1"].error["retryable"] is False


async def test_a_stored_step_for_a_missing_agent_fails_without_running(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, [("step_1", ())], agent="translation")
    llm = StepLLM()

    await service(sessionmaker, llm).execute_task(task_id)

    row = (await step_rows(sessionmaker, task_id))["step_1"]
    assert (row.status, row.attempts) == (StepStatus.FAILED, 0)
    assert row.error is not None and row.error["type"] == "UnknownAgentError"
    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.FAILED
    assert llm.calls == []


@pytest.mark.parametrize(
    "status",
    [TaskStatus.PENDING, TaskStatus.EXECUTING, TaskStatus.COMPLETED, TaskStatus.CANCELLED],
)
async def test_a_task_that_is_not_planned_is_left_alone(
    sessionmaker: Sessions, status: TaskStatus
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND, status=status)
    llm = StepLLM()

    await service(sessionmaker, llm).execute_task(task_id)

    assert llm.calls == []
    assert (await task_state(sessionmaker, task_id))[0] is status


# --- real concurrency against the database -----------------------------------------------


async def test_two_executors_racing_for_one_task_run_it_once(sessionmaker: Sessions) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    llm = StepLLM()
    # Two processes: own service, own semaphore.
    first, second = service(sessionmaker, llm), service(sessionmaker, llm)

    await asyncio.gather(first.execute_task(task_id), second.execute_task(task_id))

    assert sorted(llm.calls) == ["step_1", "step_2", "step_3"]
    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.SYNTHESIZING


async def test_the_step_cap_holds_across_tasks_with_the_real_runner(
    sessionmaker: Sessions,
) -> None:
    wide = [("step_1", ()), ("step_2", ()), ("step_3", ())]
    task_ids = [await planned_task(sessionmaker, wide) for _ in range(2)]
    llm = GatedLLM(output_of)
    runner = BackgroundRunner()
    execution = service(sessionmaker, llm, runner, max_concurrent_steps=2)

    for task_id in task_ids:
        execution.schedule(task_id)
    await llm.until_in_flight(2)
    await asyncio.sleep(0.05)  # room for a third call to wrongly start

    async with sessionmaker() as session:
        running_rows = await session.scalar(
            select(func.count())
            .select_from(ExecutionStep)
            .where(ExecutionStep.status == StepStatus.RUNNING)
        )
    assert (llm.in_flight, running_rows) == (2, 2)

    llm.release.set()
    await runner.wait_idle()
    assert llm.peak == 2
    for task_id in task_ids:
        assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.SYNTHESIZING


# --- cancellation ------------------------------------------------------------------------


async def test_cancelling_mid_run_stops_the_agent_keeps_finished_results_and_ends_cancelled(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, CHAIN)
    llm = HoldLLM("step_2")  # step_1 answers at once; step_2 is mid-thought when cancelled
    execution = service(sessionmaker, llm)
    running = asyncio.create_task(execution.execute_task(task_id))
    await llm.until_in_flight(1)

    await cancel_via_service(sessionmaker, execution, task_id)
    await running

    assert await task_state(sessionmaker, task_id) == (TaskStatus.CANCELLED, None)
    assert await step_statuses(sessionmaker, task_id) == {
        "step_1": StepStatus.COMPLETED,  # finished work is kept
        "step_2": StepStatus.CANCELLED,  # stopped mid-call
        "step_3": StepStatus.CANCELLED,  # never started
    }
    assert (llm.cancelled, llm.in_flight) == (1, 0)


async def test_a_cancel_from_another_process_stops_execution_before_the_next_group(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, CHAIN)
    llm = HoldLLM("step_1")
    running = asyncio.create_task(service(sessionmaker, llm).execute_task(task_id))
    await llm.until_in_flight(1)

    # Another process cancels: only the database changes, no in-process signal.
    async with sessionmaker() as session:
        await session.execute(
            text("UPDATE tasks SET status = 'cancelled' WHERE id = :id"), {"id": task_id}
        )
        await session.commit()
    llm.release.set()
    await running

    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.CANCELLED
    assert await step_statuses(sessionmaker, task_id) == {
        "step_1": StepStatus.COMPLETED,  # was already running; allowed to finish
        "step_2": StepStatus.CANCELLED,
        "step_3": StepStatus.CANCELLED,
    }
    assert [step_of(r) for r in llm.calls] == ["step_1"]


async def test_a_cancel_after_the_last_step_wins_over_completing(sessionmaker: Sessions) -> None:
    task_id = await planned_task(sessionmaker, [("step_1", ())])

    class CancelledWhileThinking:
        async def generate(self, request: LLMRequest) -> LLMResponse:
            async with sessionmaker() as session:
                await session.execute(
                    text("UPDATE tasks SET status = 'cancelled' WHERE id = :id"), {"id": task_id}
                )
                await session.commit()
            return fake_response("done anyway")

    await service(sessionmaker, CancelledWhileThinking()).execute_task(task_id)

    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.CANCELLED
    assert (await step_statuses(sessionmaker, task_id))["step_1"] is StepStatus.COMPLETED


async def test_cancelling_a_planned_task_cancels_its_steps_and_execution_never_starts(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND)
    llm = StepLLM()
    execution = service(sessionmaker, llm)

    await cancel_via_service(sessionmaker, execution, task_id)
    await execution.execute_task(task_id)

    assert llm.calls == []
    assert set((await step_statuses(sessionmaker, task_id)).values()) == {StepStatus.CANCELLED}


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED])
async def test_a_finished_task_cannot_be_cancelled(
    sessionmaker: Sessions, status: TaskStatus
) -> None:
    task_id = await planned_task(sessionmaker, DIAMOND, status=status)

    with pytest.raises(TaskNotCancellableError, match=f"is already {status}"):
        await cancel_via_service(sessionmaker, service(sessionmaker, StepLLM()), task_id)


async def test_cancelling_an_unknown_task_is_not_found(sessionmaker: Sessions) -> None:
    with pytest.raises(TaskNotFoundError):
        await cancel_via_service(sessionmaker, service(sessionmaker, StepLLM()), "task_" + "0" * 32)


# --- shutdown and recovery ---------------------------------------------------------------


async def test_after_a_shutdown_mid_execution_recovery_fails_it_and_keeps_finished_steps(
    sessionmaker: Sessions,
) -> None:
    task_id = await planned_task(sessionmaker, CHAIN)
    llm = HoldLLM("step_2")
    old_runner = BackgroundRunner()
    service(sessionmaker, llm, old_runner).schedule(task_id)
    await llm.until_in_flight(1)

    await old_runner.shutdown()  # the process stops with step_2 mid-call

    assert (await task_state(sessionmaker, task_id))[0] is TaskStatus.EXECUTING
    assert (await step_statuses(sessionmaker, task_id))["step_2"] is StepStatus.RUNNING

    report = await service(sessionmaker, StepLLM()).recover()

    assert list(report.failed) == [task_id]
    assert await task_state(sessionmaker, task_id) == (TaskStatus.FAILED, INTERRUPTED_ERROR)
    rows = await step_rows(sessionmaker, task_id)
    assert rows["step_1"].status is StepStatus.COMPLETED
    assert rows["step_2"].status is StepStatus.FAILED
    assert rows["step_2"].error is not None and rows["step_2"].error["type"] == "Interrupted"
    assert rows["step_3"].status is StepStatus.CANCELLED


async def test_recovery_restarts_planned_tasks_and_leaves_finished_ones_alone(
    sessionmaker: Sessions,
) -> None:
    planned = await planned_task(sessionmaker, DIAMOND)
    done = await planned_task(sessionmaker, DIAMOND, status=TaskStatus.COMPLETED)
    runner = ManualRunner()
    execution = service(sessionmaker, StepLLM(), runner)

    report = await execution.recover()
    await execution.recover()  # twice: still one job per task
    assert (list(report.failed), list(report.rescheduled)) == ([], [planned])
    assert runner.names == [f"execute:{planned}"]

    await runner.run_all()
    assert (await task_state(sessionmaker, planned))[0] is TaskStatus.SYNTHESIZING
    assert (await task_state(sessionmaker, done))[0] is TaskStatus.COMPLETED
