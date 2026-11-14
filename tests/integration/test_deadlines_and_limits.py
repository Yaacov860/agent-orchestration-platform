"""Deadlines at every stage, and the limits from Settings, against the real database."""

import asyncio
from datetime import UTC, datetime, timedelta

from factories import make_runnable_plan, make_task, plan_json, plan_step, synthesis_json
from fakes import GatedLLM, ManualRunner, RoutingLLM, StepLLM, is_planner, is_synthesis
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from orchestrator.agents.catalog import default_catalog
from orchestrator.agents.factory import AgentFactory
from orchestrator.config import Settings
from orchestrator.db.models import Task
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.status import StepStatus, TaskStatus
from orchestrator.llm import LLMRequest, LLMResponse
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.main import create_app
from orchestrator.planning import Planner
from orchestrator.runner import BackgroundRunner
from orchestrator.services.execution import (
    DEADLINE_BEFORE_EXECUTION,
    DEADLINE_DURING_EXECUTION,
    ExecutionService,
)
from orchestrator.services.planning import (
    DEADLINE_BEFORE_PLANNING,
    DEADLINE_DURING_PLANNING,
    PlanningService,
)

Sessions = async_sessionmaker[AsyncSession]

CHAIN = [("step_1", ()), ("step_2", ("step_1",)), ("step_3", ("step_2",))]
WIDE = [("step_1", ()), ("step_2", ()), ("step_3", ())]
PLAN = plan_json(plan_step("step_1"), plan_step("step_2", "step_1", action="summarize"))


def soon(seconds: float) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=seconds)


async def store_task(sessionmaker: Sessions, *, steps: object = None, **fields: object) -> str:
    task = make_task(**fields)
    async with sessionmaker() as session:
        session.add(task)
        await session.flush()
        if steps is not None:
            session.add(make_runnable_plan(task.id, steps))  # type: ignore[arg-type]
        await session.commit()
    return task.id


async def state(sessionmaker: Sessions, task_id: str) -> tuple[TaskStatus, str | None]:
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
    assert task is not None
    return task.status, task.error


async def steps(sessionmaker: Sessions, task_id: str) -> dict[str, tuple[StepStatus, str | None]]:
    async with sessionmaker() as session:
        plan = await TaskRepository(session).latest_plan(task_id)
    assert plan is not None
    return {s.step_id: (s.status, (s.error or {}).get("type")) for s in plan.steps}  # type: ignore[misc]


def planning(sessionmaker: Sessions, llm: object) -> PlanningService:
    return PlanningService(sessionmaker, Planner(llm, default_catalog()), ManualRunner())  # type: ignore[arg-type]


def execution(sessionmaker: Sessions, llm: object, **options: object) -> ExecutionService:
    return ExecutionService(sessionmaker, AgentFactory(llm), ManualRunner(), **options)  # type: ignore[arg-type]


# --- deadlines: planning -----------------------------------------------------------------


async def test_a_task_whose_deadline_passed_before_planning_fails_without_a_model_call(
    sessionmaker: Sessions,
) -> None:
    task_id = await store_task(sessionmaker, deadline=soon(-1))
    llm = FakeLLM()  # any call would fail loudly

    await planning(sessionmaker, llm).plan_task(task_id)

    assert await state(sessionmaker, task_id) == (TaskStatus.FAILED, DEADLINE_BEFORE_PLANNING)
    assert llm.calls == []


async def test_planning_that_outlasts_the_deadline_is_stopped(sessionmaker: Sessions) -> None:
    task_id = await store_task(sessionmaker, deadline=soon(0.2))
    llm = GatedLLM()  # the planner's model never answers

    await planning(sessionmaker, llm).plan_task(task_id)

    assert await state(sessionmaker, task_id) == (TaskStatus.FAILED, DEADLINE_DURING_PLANNING)
    assert (llm.cancelled, llm.in_flight) == (1, 0)


async def test_planning_within_the_deadline_is_unaffected(sessionmaker: Sessions) -> None:
    task_id = await store_task(sessionmaker, deadline=soon(60))

    await planning(sessionmaker, FakeLLM([fake_response(PLAN)])).plan_task(task_id)

    assert (await state(sessionmaker, task_id))[0] is TaskStatus.PLANNED


# --- deadlines: execution ----------------------------------------------------------------


async def test_a_deadline_passed_before_execution_skips_every_step(sessionmaker: Sessions) -> None:
    task_id = await store_task(
        sessionmaker, steps=CHAIN, status=TaskStatus.PLANNED, deadline=soon(-1)
    )
    llm = StepLLM()

    await execution(sessionmaker, llm).execute_task(task_id)

    assert await state(sessionmaker, task_id) == (TaskStatus.FAILED, DEADLINE_BEFORE_EXECUTION)
    assert await steps(sessionmaker, task_id) == {
        step_id: (StepStatus.SKIPPED, "DeadlineExceeded")
        for step_id in ("step_1", "step_2", "step_3")
    }
    assert llm.calls == []


async def test_a_deadline_passing_mid_execution_stops_the_running_step_and_keeps_finished_ones(
    sessionmaker: Sessions,
) -> None:
    task_id = await store_task(
        sessionmaker, steps=CHAIN, status=TaskStatus.PLANNED, deadline=soon(0.3)
    )

    class FirstStepOnly(GatedLLM):
        """step_1 answers at once; step_2 hangs past the deadline."""

        async def generate(self, request: LLMRequest) -> LLMResponse:
            if "do step_1" in request.user_prompt:
                return fake_response("step_1 done")
            return await super().generate(request)

    llm = FirstStepOnly()
    await execution(sessionmaker, llm, step_timeout=30.0).execute_task(task_id)

    status, error = await state(sessionmaker, task_id)
    assert status is TaskStatus.FAILED
    assert error == (
        f"{DEADLINE_DURING_EXECUTION}; "
        "step_2 failed: the task's deadline passed while it was running; not run: step_3"
    )
    assert await steps(sessionmaker, task_id) == {
        "step_1": (StepStatus.COMPLETED, None),
        "step_2": (StepStatus.FAILED, "DeadlineExceeded"),
        "step_3": (StepStatus.SKIPPED, "DeadlineExceeded"),
    }
    assert llm.cancelled == 1


async def test_a_deadline_set_at_submission_is_enforced_through_the_api(
    database_url: str, engine: AsyncEngine
) -> None:
    llm = GatedLLM()  # the planner never answers
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url), llm=llm, runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        submitted = await client.post(
            "/tasks", json={"goal": "Explain asyncio", "deadline": soon(0.3).isoformat()}
        )
        await runner.run_all()
        body = (await client.get(f"/tasks/{submitted.json()['task_id']}")).json()
    await app.state.engine.dispose()

    assert submitted.status_code == 201
    assert (body["status"], body["error"]) == ("failed", DEADLINE_DURING_PLANNING)


# --- limits from Settings ----------------------------------------------------------------


async def run_app(
    database_url: str, settings: dict[str, object], llm: object, gate: GatedLLM | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Submit one task to an app built from these settings; return its status and steps.
    With a `gate`, its calls are held until it is seen that only one is in flight."""
    app = create_app(Settings(database_url=database_url, **settings), llm=llm)  # type: ignore[arg-type]
    runner: BackgroundRunner = app.state.runner
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json={"goal": "Compare things"})).json()["task_id"]
        if gate is not None:
            await gate.until_in_flight(1)
            await asyncio.sleep(0.1)  # three steps are ready; room for a second to wrongly start
            assert gate.in_flight == 1
            gate.release.set()
        await runner.wait_idle()
        status = (await client.get(f"/tasks/{task_id}")).json()
        step_list = (await client.get(f"/tasks/{task_id}/steps")).json()
    await app.state.engine.dispose()
    return status, step_list


def wide_plan_then(answer: str = "findings") -> GatedLLM:
    """Plans three independent steps, then answers every step once released."""
    plan = plan_json(plan_step("step_1"), plan_step("step_2"), plan_step("step_3"))

    class PlanThenGate(GatedLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            if is_planner(request):
                return fake_response(plan)
            if is_synthesis(request):
                return fake_response(synthesis_json())
            return await super().generate(request)

    return PlanThenGate(lambda request: answer)


async def test_max_concurrent_steps_from_settings_caps_parallel_steps(
    database_url: str, engine: AsyncEngine
) -> None:
    llm = wide_plan_then()

    status, _ = await run_app(database_url, {"max_concurrent_steps": 1}, llm, gate=llm)

    assert status["status"] == "completed"
    assert llm.peak == 1  # three independent steps, one at a time


async def test_step_timeout_and_attempts_from_settings_apply(
    database_url: str, engine: AsyncEngine
) -> None:
    plan = plan_json(plan_step("step_1"))

    class PlanThenHang(GatedLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            if "You are the planner" in request.system_prompt:
                return fake_response(plan)
            return await super().generate(request)  # never released

    llm = PlanThenHang()
    app = create_app(
        Settings(
            database_url=database_url,
            step_timeout_seconds=0.05,
            step_max_attempts=2,
            step_retry_base_delay_seconds=0,
        ),
        llm=llm,
    )
    runner: BackgroundRunner = app.state.runner
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json={"goal": "g"})).json()["task_id"]
        await runner.wait_idle()
        [step] = (await client.get(f"/tasks/{task_id}/steps")).json()["steps"]
    await app.state.engine.dispose()

    assert (step["status"], step["attempts"]) == ("failed", 2)  # timed out twice, no third try
    assert step["error"] == {
        "type": "StepTimeoutError",
        "message": "step did not finish within 0.05s",
        "retryable": True,
    }
    assert llm.cancelled == 2


async def test_the_input_size_limit_from_settings_cuts_what_the_next_step_receives(
    database_url: str, engine: AsyncEngine
) -> None:
    long_findings = "F" * 500
    llm = RoutingLLM(PLAN, synthesis_json(), step=lambda request: long_findings)

    _, step_list = await run_app(database_url, {"max_context_chars_per_input": 200}, llm)

    [summary_prompt] = [
        r.user_prompt for r in llm.calls if "## Your step (summarize)" in r.user_prompt
    ]
    assert "F" * 200 + "\n\n[... truncated: 300 of 500 characters omitted]" in summary_prompt
    assert "F" * 201 not in summary_prompt
    # The stored, and served, output of step_1 is complete.
    first = step_list["steps"][0]  # type: ignore[index]
    assert first["output"]["content"] == long_findings
