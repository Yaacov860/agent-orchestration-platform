"""PlanningService against the real database: status transitions, what gets stored, failures."""

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from factories import make_plan, make_task, plan_json, plan_step
from fakes import ManualRunner
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.agents.catalog import default_catalog
from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.status import FailureStage, InvalidTransitionError, StepStatus, TaskStatus
from orchestrator.llm import (
    LLMClient,
    LLMPermanentError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.llm.unconfigured import NO_PROVIDER_MESSAGE, UnconfiguredLLM
from orchestrator.planning import Plan, Planner
from orchestrator.services.planning import INTERNAL_ERROR, PlanningService

Sessions = async_sessionmaker[AsyncSession]

VALID = plan_json(
    plan_step("step_1", input="Research Python for beginners"),
    plan_step("step_2", input="Research JavaScript for beginners"),
    plan_step("step_3", "step_1", "step_2", action="summarize", input="Compare both"),
    reasoning="Two research branches, then a summary.",
)
INVALID = plan_json(plan_step("step_1", "step_9"))


class Sleeps:
    """Records retry delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def service(sessionmaker: Sessions, llm: LLMClient, sleep: Sleeps | None = None) -> PlanningService:
    return PlanningService(
        sessionmaker,
        Planner(llm, default_catalog()),
        ManualRunner(),
        retry_delays=(1.0, 3.0),
        sleep=sleep or Sleeps(),
    )


async def stored_task(sessionmaker: Sessions, **overrides: object) -> Task:
    task = make_task(**overrides)
    async with sessionmaker() as session:
        session.add(task)
        await session.commit()
    return task


async def reload(sessionmaker: Sessions, task_id: str) -> Task:
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
    assert task is not None
    return task


async def plans_of(sessionmaker: Sessions, task_id: str) -> list[ExecutionPlan]:
    async with sessionmaker() as session:
        query = (
            select(ExecutionPlan)
            .where(ExecutionPlan.task_id == task_id)
            .order_by(ExecutionPlan.version)
        )
        return list((await session.scalars(query)).all())


async def steps_of(sessionmaker: Sessions, plan: ExecutionPlan) -> list[ExecutionStep]:
    async with sessionmaker() as session:
        query = (
            select(ExecutionStep)
            .where(ExecutionStep.plan_id == plan.id)
            .order_by(ExecutionStep.position)
        )
        return list((await session.scalars(query)).all())


class ProbeLLM:
    """Runs `probe` while the 'model is thinking', then answers. Lets a test see (or change)
    the database exactly in the middle of planning."""

    def __init__(self, probe: Callable[[], Awaitable[None]], answer: str = VALID) -> None:
        self._probe = probe
        self._answer = answer

    async def generate(self, request: LLMRequest) -> LLMResponse:
        await self._probe()
        return fake_response(self._answer)


# --- success -----------------------------------------------------------------------------


async def test_a_pending_task_becomes_planned_with_its_plan_and_steps_stored(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(
        sessionmaker, goal="Python vs JavaScript", constraints={"tone": "friendly"}
    )
    llm = FakeLLM([fake_response(VALID)])

    await service(sessionmaker, llm).plan_task(task.id)

    planned = await reload(sessionmaker, task.id)
    assert (planned.status, planned.error) == (TaskStatus.PLANNED, None)
    assert planned.updated_at > task.updated_at

    [plan] = await plans_of(sessionmaker, task.id)
    assert plan.version == 1
    assert plan.parallel_groups == [["step_1", "step_2"], ["step_3"]]
    # The JSON record is the validated plan itself: it loads back into one.
    record = Plan.model_validate_json(json.dumps(plan.plan))
    assert record.reasoning == "Two research branches, then a summary."
    assert [s.step_id for s in record.steps] == ["step_1", "step_2", "step_3"]

    steps = await steps_of(sessionmaker, plan)
    assert [
        (s.step_id, s.position, s.agent, s.action, s.dependencies, s.status) for s in steps
    ] == [
        ("step_1", 0, "research", "research", [], StepStatus.PENDING),
        ("step_2", 1, "research", "research", [], StepStatus.PENDING),
        ("step_3", 2, "research", "summarize", ["step_1", "step_2"], StepStatus.PENDING),
    ]
    assert steps[2].input == "Compare both"
    assert all((s.attempts, s.input_tokens, s.output) == (0, 0, None) for s in steps)

    # The planner got the task as stored.
    assert "<task>\nPython vs JavaScript\n</task>" in llm.calls[0].user_prompt
    assert '{"tone": "friendly"}' in llm.calls[0].user_prompt


async def test_the_task_is_planning_and_committed_while_the_model_thinks(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    seen: list[TaskStatus] = []

    async def look() -> None:
        # A separate session: it sees only committed data.
        seen.append((await reload(sessionmaker, task.id)).status)

    await service(sessionmaker, ProbeLLM(look)).plan_task(task.id)

    assert seen == [TaskStatus.PLANNING]
    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PLANNED


async def test_no_database_connection_is_held_while_the_model_thinks(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    open_transactions: list[int] = []

    async def count() -> None:
        async with sessionmaker() as session:
            open_transactions.append(
                await session.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                        "AND state LIKE 'idle in transaction%' AND pid <> pg_backend_pid()"
                    )
                )
                or 0
            )

    await service(sessionmaker, ProbeLLM(count)).plan_task(task.id)

    assert open_transactions == [0]


async def test_a_replan_gets_the_next_version_and_keeps_the_old_plan(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    async with sessionmaker() as session:
        session.add(make_plan(task.id, 1, [("step_1", StepStatus.FAILED)]))
        await session.commit()

    await service(sessionmaker, FakeLLM([fake_response(VALID)])).plan_task(task.id)

    assert [p.version for p in await plans_of(sessionmaker, task.id)] == [1, 2]


# --- planning failures -------------------------------------------------------------------


async def test_without_a_configured_provider_the_task_fails_with_a_clear_message(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    sleeps = Sleeps()

    await service(sessionmaker, UnconfiguredLLM(), sleeps).plan_task(task.id)

    failed = await reload(sessionmaker, task.id)
    assert failed.status is TaskStatus.FAILED
    assert failed.error == f"LLM call failed during planning: {NO_PROVIDER_MESSAGE}"
    assert sleeps.delays == []  # permanent: not retried
    assert await plans_of(sessionmaker, task.id) == []


async def test_a_plan_still_invalid_after_the_repair_fails_the_task_and_stores_nothing(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    llm = FakeLLM([fake_response(INVALID), fake_response(INVALID)])

    await service(sessionmaker, llm).plan_task(task.id)

    failed = await reload(sessionmaker, task.id)
    assert failed.status is TaskStatus.FAILED
    assert failed.error == (
        "no valid plan after 2 attempts: "
        "step_1: depends on 'step_9', which is not a step in this plan"
    )
    assert len(llm.calls) == 2  # an invalid plan is not retried beyond the planner's repair
    assert await plans_of(sessionmaker, task.id) == []


async def test_a_transient_llm_failure_is_retried_after_a_delay(sessionmaker: Sessions) -> None:
    task = await stored_task(sessionmaker)
    sleeps = Sleeps()
    llm = FakeLLM([LLMTransientError("429 rate limited"), fake_response(VALID)])

    await service(sessionmaker, llm, sleeps).plan_task(task.id)

    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PLANNED
    assert sleeps.delays == [1.0]


async def test_transient_failures_give_up_after_the_last_retry(sessionmaker: Sessions) -> None:
    task = await stored_task(sessionmaker)
    sleeps = Sleeps()
    llm = FakeLLM([LLMTransientError("timeout")] * 3)

    await service(sessionmaker, llm, sleeps).plan_task(task.id)

    failed = await reload(sessionmaker, task.id)
    assert (failed.status, failed.error) == (
        TaskStatus.FAILED,
        "LLM call failed during planning: timeout",
    )
    assert sleeps.delays == [1.0, 3.0]
    assert len(llm.calls) == 3


async def test_a_permanent_llm_failure_is_not_retried(sessionmaker: Sessions) -> None:
    task = await stored_task(sessionmaker)
    sleeps = Sleeps()

    await service(sessionmaker, FakeLLM([LLMPermanentError("content blocked")]), sleeps).plan_task(
        task.id
    )

    assert (await reload(sessionmaker, task.id)).status is TaskStatus.FAILED
    assert sleeps.delays == []


async def test_a_bug_fails_the_task_without_leaking_details_to_the_client(
    sessionmaker: Sessions, caplog: pytest.LogCaptureFixture
) -> None:
    task = await stored_task(sessionmaker)

    async def explode() -> None:
        raise RuntimeError("secret internal detail")

    await service(sessionmaker, ProbeLLM(explode)).plan_task(task.id)

    failed = await reload(sessionmaker, task.id)
    assert (failed.status, failed.error) == (TaskStatus.FAILED, INTERNAL_ERROR)
    assert "secret internal detail" in caplog.text  # the details are in the server log


# --- concurrency and races ---------------------------------------------------------------


@pytest.mark.parametrize("status", [TaskStatus.PLANNING, TaskStatus.PLANNED, TaskStatus.FAILED])
async def test_a_task_that_is_not_pending_is_left_alone(
    sessionmaker: Sessions, status: TaskStatus
) -> None:
    task = await stored_task(sessionmaker, status=status)
    llm = FakeLLM()  # any call would fail loudly: the script is empty

    await service(sessionmaker, llm).plan_task(task.id)

    assert llm.calls == []
    assert (await reload(sessionmaker, task.id)).status is status


async def test_an_unknown_task_is_ignored(sessionmaker: Sessions) -> None:
    llm = FakeLLM()

    await service(sessionmaker, llm).plan_task("task_" + "0" * 32)

    assert llm.calls == []


async def test_only_one_of_two_concurrent_planners_gets_the_task(sessionmaker: Sessions) -> None:
    task = await stored_task(sessionmaker)
    llm = FakeLLM([fake_response(VALID)])  # one reply only: a second planning would fail loudly
    planning = service(sessionmaker, llm)

    await asyncio.gather(planning.plan_task(task.id), planning.plan_task(task.id))

    assert len(llm.calls) == 1
    assert len(await plans_of(sessionmaker, task.id)) == 1
    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PLANNED


async def test_a_plan_finished_after_the_task_was_failed_elsewhere_is_discarded(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)

    async def recovery_fails_it_meanwhile() -> None:
        async with sessionmaker() as session:
            await session.execute(
                text("UPDATE tasks SET status = 'failed', error = 'interrupted' WHERE id = :id"),
                {"id": task.id},
            )
            await session.commit()

    await service(sessionmaker, ProbeLLM(recovery_fails_it_meanwhile)).plan_task(task.id)

    failed = await reload(sessionmaker, task.id)
    assert (failed.status, failed.error) == (TaskStatus.FAILED, "interrupted")
    assert await plans_of(sessionmaker, task.id) == []


async def test_schedule_hands_the_job_to_the_runner_without_running_it(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    runner = ManualRunner()
    planning = PlanningService(
        sessionmaker, Planner(FakeLLM([fake_response(VALID)]), default_catalog()), runner
    )

    planning.schedule(task.id)

    assert runner.names == [f"plan:{task.id}"]
    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PENDING
    await runner.run_all()
    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PLANNED


async def test_scheduling_a_task_twice_queues_one_job_until_it_finishes(
    sessionmaker: Sessions,
) -> None:
    task = await stored_task(sessionmaker)
    runner = ManualRunner()
    planning = PlanningService(
        sessionmaker, Planner(FakeLLM([fake_response(VALID)]), default_catalog()), runner
    )

    planning.schedule(task.id)
    planning.schedule(task.id)
    assert runner.names == [f"plan:{task.id}"]

    await runner.run_all()
    # Once the job is done the task may be scheduled again; the claim then sees it is no
    # longer pending and does nothing (the empty script would fail loudly otherwise).
    planning.schedule(task.id)
    await runner.run_all()
    assert (await reload(sessionmaker, task.id)).status is TaskStatus.PLANNED


# --- invalid transitions -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "to"),
    [
        (TaskStatus.PENDING, TaskStatus.FAILED),
        (TaskStatus.PLANNED, TaskStatus.PENDING),
        (TaskStatus.FAILED, TaskStatus.PLANNING),
    ],
)
async def test_the_repository_refuses_an_invalid_transition_before_touching_the_row(
    sessionmaker: Sessions, expected: TaskStatus, to: TaskStatus
) -> None:
    task = await stored_task(sessionmaker, status=expected)

    async with sessionmaker() as session:
        with pytest.raises(InvalidTransitionError):
            await TaskRepository(session).transition(task.id, expected=expected, to=to)
        await session.commit()

    assert (await reload(sessionmaker, task.id)).status is expected


async def test_failing_every_task_in_a_terminal_status_is_refused(sessionmaker: Sessions) -> None:
    task = await stored_task(sessionmaker, status=TaskStatus.COMPLETED)

    async with sessionmaker() as session:
        with pytest.raises(InvalidTransitionError, match="from 'completed' to 'failed'"):
            await TaskRepository(session).fail_all_in(
                TaskStatus.COMPLETED, "x", FailureStage.EXECUTION
            )

    assert (await reload(sessionmaker, task.id)).status is TaskStatus.COMPLETED
