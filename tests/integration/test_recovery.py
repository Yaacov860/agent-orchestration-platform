"""Startup recovery: `planning` (interrupted mid-plan) is failed, `pending` (never started) is
scheduled again, everything else is left alone, and no task is ever planned twice."""

import asyncio
import logging

import pytest
from factories import make_task, plan_json, plan_step, synthesis_json
from fakes import GatedLLM, ManualRunner, RoutingLLM
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.agents.catalog import default_catalog
from orchestrator.config import Settings
from orchestrator.db.models import ExecutionPlan, Task
from orchestrator.domain.status import TaskStatus
from orchestrator.llm import LLMClient
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.llm.unconfigured import NO_PROVIDER_MESSAGE, UnconfiguredLLM
from orchestrator.main import create_app
from orchestrator.planning import Planner
from orchestrator.runner import BackgroundRunner, Runner
from orchestrator.services.planning import INTERRUPTED_ERROR, PlanningService

Sessions = async_sessionmaker[AsyncSession]

VALID = plan_json(plan_step("step_1"), plan_step("step_2", "step_1", action="summarize"))


def planning_service(
    sessionmaker: Sessions, llm: LLMClient, runner: Runner, max_concurrent: int = 4
) -> PlanningService:
    return PlanningService(
        sessionmaker,
        Planner(llm, default_catalog()),
        runner,
        retry_delays=(),
        max_concurrent=max_concurrent,
    )


def replies(count: int) -> FakeLLM:
    return FakeLLM([fake_response(VALID)] * count)


async def store(sessionmaker: Sessions, *statuses: TaskStatus) -> list[str]:
    """One transaction per task, so created_at (now() = transaction start) is ordered."""
    ids = []
    for status in statuses:
        task = make_task(status=status)
        async with sessionmaker() as session:
            session.add(task)
            await session.commit()
        ids.append(task.id)
    return ids


async def states(sessionmaker: Sessions) -> dict[str, tuple[TaskStatus, str | None]]:
    async with sessionmaker() as session:
        return {t.id: (t.status, t.error) for t in (await session.scalars(select(Task))).all()}


async def plan_count(sessionmaker: Sessions, task_id: str) -> int:
    async with sessionmaker() as session:
        query = (
            select(func.count()).select_from(ExecutionPlan).where(ExecutionPlan.task_id == task_id)
        )
        return await session.scalar(query) or 0


# --- what recovery does to each status ---------------------------------------------------


async def test_planning_is_failed_pending_is_rescheduled_and_the_rest_is_untouched(
    sessionmaker: Sessions, caplog: pytest.LogCaptureFixture
) -> None:
    stuck_1, pending_1, stuck_2, planned, failed, pending_2 = await store(
        sessionmaker,
        TaskStatus.PLANNING,
        TaskStatus.PENDING,
        TaskStatus.PLANNING,
        TaskStatus.PLANNED,
        TaskStatus.FAILED,
        TaskStatus.PENDING,
    )
    runner = ManualRunner()

    with caplog.at_level(logging.INFO):
        report = await planning_service(sessionmaker, FakeLLM(), runner).recover()

    assert sorted(report.failed) == sorted([stuck_1, stuck_2])
    assert report.rescheduled == [pending_1, pending_2]  # oldest first
    assert runner.names == [f"plan:{pending_1}", f"plan:{pending_2}"]
    assert await states(sessionmaker) == {
        stuck_1: (TaskStatus.FAILED, INTERRUPTED_ERROR),
        stuck_2: (TaskStatus.FAILED, INTERRUPTED_ERROR),
        pending_1: (TaskStatus.PENDING, None),  # scheduled, not touched yet
        pending_2: (TaskStatus.PENDING, None),
        planned: (TaskStatus.PLANNED, None),
        failed: (TaskStatus.FAILED, None),
    }
    assert "marked 2 interrupted task(s) as failed" in caplog.text
    assert "rescheduled 2 pending task(s) for planning" in caplog.text


async def test_a_rescheduled_task_gets_planned(sessionmaker: Sessions) -> None:
    [pending] = await store(sessionmaker, TaskStatus.PENDING)
    runner = ManualRunner()

    await planning_service(sessionmaker, replies(1), runner).recover()
    await runner.run_all()

    assert (await states(sessionmaker))[pending] == (TaskStatus.PLANNED, None)
    assert await plan_count(sessionmaker, pending) == 1


async def test_an_interrupted_task_is_failed_not_planned_again(sessionmaker: Sessions) -> None:
    [stuck] = await store(sessionmaker, TaskStatus.PLANNING)
    runner = ManualRunner()
    llm = FakeLLM()  # any call would fail loudly

    await planning_service(sessionmaker, llm, runner).recover()
    await runner.run_all()

    assert runner.jobs == [] and llm.calls == []
    assert (await states(sessionmaker))[stuck] == (TaskStatus.FAILED, INTERRUPTED_ERROR)


async def test_recovery_with_nothing_to_recover_changes_nothing(sessionmaker: Sessions) -> None:
    await store(sessionmaker, TaskStatus.PLANNED, TaskStatus.FAILED)
    before = await states(sessionmaker)
    runner = ManualRunner()

    report = await planning_service(sessionmaker, FakeLLM(), runner).recover()

    assert (list(report.failed), list(report.rescheduled)) == ([], [])
    assert runner.jobs == []
    assert await states(sessionmaker) == before


async def test_a_rescheduled_task_without_a_provider_fails_clearly(sessionmaker: Sessions) -> None:
    [pending] = await store(sessionmaker, TaskStatus.PENDING)
    runner = ManualRunner()

    await planning_service(sessionmaker, UnconfiguredLLM(), runner).recover()
    await runner.run_all()

    assert (await states(sessionmaker))[pending] == (
        TaskStatus.FAILED,
        f"LLM call failed during planning: {NO_PROVIDER_MESSAGE}",
    )


# --- never planned twice -----------------------------------------------------------------


async def test_running_recovery_twice_in_one_process_schedules_each_task_once(
    sessionmaker: Sessions,
) -> None:
    pending = await store(sessionmaker, TaskStatus.PENDING, TaskStatus.PENDING)
    runner = ManualRunner()
    planning = planning_service(sessionmaker, replies(2), runner)

    await planning.recover()
    second = await planning.recover()

    assert second.rescheduled == pending  # still pending, so reported again...
    assert runner.names == [f"plan:{task_id}" for task_id in pending]  # ...but queued once


async def test_two_processes_recovering_at_once_plan_each_task_exactly_once(
    sessionmaker: Sessions,
) -> None:
    # Two app processes (own service, own runner) start together and both reschedule the
    # same pending tasks. The conditional claim lets exactly one of them plan each task.
    pending = await store(sessionmaker, *[TaskStatus.PENDING] * 3)
    llm = replies(3)  # one reply per task: a second planning of any task would fail loudly
    runner_a, runner_b = ManualRunner(), ManualRunner()
    process_a = planning_service(sessionmaker, llm, runner_a)
    process_b = planning_service(sessionmaker, llm, runner_b)

    await asyncio.gather(process_a.recover(), process_b.recover())
    await asyncio.gather(runner_a.run_all(), runner_b.run_all())

    assert len(llm.calls) == 3
    for task_id in pending:
        assert (await states(sessionmaker))[task_id] == (TaskStatus.PLANNED, None)
        assert await plan_count(sessionmaker, task_id) == 1


async def test_a_task_submitted_and_rescheduled_is_still_planned_once(
    sessionmaker: Sessions,
) -> None:
    # POST schedules a task; a recovery in another process reschedules the same task.
    [task_id] = await store(sessionmaker, TaskStatus.PENDING)
    llm = replies(1)
    here, elsewhere = ManualRunner(), ManualRunner()
    planning_service(sessionmaker, llm, here).schedule(task_id)
    await planning_service(sessionmaker, llm, elsewhere).recover()

    await asyncio.gather(here.run_all(), elsewhere.run_all())

    assert len(llm.calls) == 1
    assert await plan_count(sessionmaker, task_id) == 1


# --- shutdown and restart ----------------------------------------------------------------


async def test_a_backlog_is_planned_a_few_at_a_time_and_the_queue_stays_pending(
    sessionmaker: Sessions,
) -> None:
    pending = await store(sessionmaker, *[TaskStatus.PENDING] * 5)
    llm = GatedLLM(lambda request: VALID)
    runner = BackgroundRunner()

    await planning_service(sessionmaker, llm, runner, max_concurrent=2).recover()
    await llm.until_in_flight(2)
    await asyncio.sleep(0.05)  # room for a job that would wrongly start to show up

    statuses = [status for status, _ in (await states(sessionmaker)).values()]
    assert statuses.count(TaskStatus.PLANNING) == 2
    assert statuses.count(TaskStatus.PENDING) == 3  # queued tasks are not claimed yet

    llm.release.set()
    await runner.wait_idle()
    assert llm.peak == 2
    final = await states(sessionmaker)
    assert {task_id: final[task_id] for task_id in pending} == {
        task_id: (TaskStatus.PLANNED, None) for task_id in pending
    }


async def test_after_a_shutdown_mid_plan_the_next_start_fails_it_and_plans_the_queue(
    sessionmaker: Sessions,
) -> None:
    """The cap's other half: the task being planned is lost (failed), the queued one is not."""
    first, queued = await store(sessionmaker, TaskStatus.PENDING, TaskStatus.PENDING)
    gated = GatedLLM(lambda request: VALID)
    old_runner = BackgroundRunner()
    old = planning_service(sessionmaker, gated, old_runner, max_concurrent=1)
    old.schedule(first)
    old.schedule(queued)
    await gated.until_in_flight(1)

    await old_runner.shutdown()  # the process stops: `first` mid-plan, `queued` waiting

    assert await states(sessionmaker) == {
        first: (TaskStatus.PLANNING, None),
        queued: (TaskStatus.PENDING, None),
    }

    new_runner = ManualRunner()
    report = await planning_service(sessionmaker, replies(1), new_runner).recover()
    await new_runner.run_all()

    assert (list(report.failed), list(report.rescheduled)) == ([first], [queued])
    assert await states(sessionmaker) == {
        first: (TaskStatus.FAILED, INTERRUPTED_ERROR),
        queued: (TaskStatus.PLANNED, None),
    }


# --- wiring into the app -----------------------------------------------------------------


async def test_the_app_recovers_and_runs_the_backlog_when_it_starts(
    database_url: str, sessionmaker: Sessions
) -> None:
    stuck, pending = await store(sessionmaker, TaskStatus.PLANNING, TaskStatus.PENDING)
    # The pending task is planned (2 steps), executed and synthesized.
    app = create_app(Settings(database_url=database_url), llm=RoutingLLM(VALID, synthesis_json()))
    runner: BackgroundRunner = app.state.runner

    async with app.router.lifespan_context(app):
        await runner.wait_idle()
        assert await states(sessionmaker) == {
            stuck: (TaskStatus.FAILED, INTERRUPTED_ERROR),
            pending: (TaskStatus.COMPLETED, None),
        }


async def test_the_app_still_starts_when_the_database_is_down_at_startup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(
        Settings(database_url="postgresql+asyncpg://u:p@127.0.0.1:1/nothing"), llm=FakeLLM()
    )

    with caplog.at_level(logging.ERROR):
        async with app.router.lifespan_context(app):
            pass

    assert "startup recovery skipped, database unavailable" in caplog.text
