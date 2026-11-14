"""Trace, token usage, GET /agents, /metrics and /health/ready, against the real database."""

from factories import make_task, plan_json, plan_step, synthesis_json
from fakes import ManualRunner
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from orchestrator.config import Settings
from orchestrator.db.models import ExecutionPlan, TaskEvent
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.status import TaskStatus
from orchestrator.llm import LLMPermanentError, LLMTransientError
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.main import create_app

Sessions = async_sessionmaker[AsyncSession]
GOAL = {"goal": "Compare Python and JavaScript for beginners"}
PLAN = plan_json(
    plan_step("step_1"),
    plan_step("step_2"),
    plan_step("step_3", "step_1", "step_2", action="summarize"),
)


async def run_task(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM, *replies: object
) -> str:
    llm.add(*replies)  # type: ignore[arg-type]
    task_id: str = (await client.post("/tasks", json=GOAL)).json()["task_id"]
    await runner.run_all()
    return task_id


def full_run() -> list[object]:
    return [
        fake_response(PLAN, input_tokens=300, output_tokens=60),
        fake_response("- a", input_tokens=100, output_tokens=20),
        fake_response("- b", input_tokens=100, output_tokens=20),
        fake_response("summary", input_tokens=150, output_tokens=30),
        fake_response(
            synthesis_json(findings=[("x", ["step_1"])]), input_tokens=400, output_tokens=80
        ),
    ]


async def statuses(sessionmaker: Sessions, task_id: str) -> list[TaskStatus]:
    async with sessionmaker() as session:
        return [e.status for e in await TaskRepository(session).events(task_id)]


# --- the trace ---------------------------------------------------------------------------


async def test_every_status_change_is_recorded_in_order(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM, sessionmaker: Sessions
) -> None:
    task_id = await run_task(client, runner, llm, *full_run())

    assert await statuses(sessionmaker, task_id) == [
        TaskStatus.PENDING,
        TaskStatus.PLANNING,
        TaskStatus.PLANNED,
        TaskStatus.EXECUTING,
        TaskStatus.SYNTHESIZING,
        TaskStatus.COMPLETED,
    ]
    async with sessionmaker() as session:
        times = [e.at for e in await TaskRepository(session).events(task_id)]
    assert times == sorted(times)
    assert len(set(times)) == len(times)  # clock_timestamp(): real times, not one per transaction


async def test_a_change_that_did_not_happen_is_not_recorded(sessionmaker: Sessions) -> None:
    task = make_task(status=TaskStatus.PLANNED)
    async with sessionmaker() as session:
        TaskRepository(session).add(task)
        await session.commit()
        moved = await TaskRepository(session).transition(
            task.id, expected=TaskStatus.PENDING, to=TaskStatus.PLANNING
        )
        await session.commit()

    assert moved is None
    assert await statuses(sessionmaker, task.id) == [TaskStatus.PLANNED]


async def test_a_rolled_back_change_leaves_no_event(sessionmaker: Sessions) -> None:
    task = make_task()
    task_id = task.id  # the instance expires on rollback
    async with sessionmaker() as session:
        TaskRepository(session).add(task)
        await session.commit()
        await TaskRepository(session).transition(
            task_id, expected=TaskStatus.PENDING, to=TaskStatus.PLANNING
        )
        await session.rollback()

    assert await statuses(sessionmaker, task_id) == [TaskStatus.PENDING]


async def test_the_trace_endpoint_gives_the_timeline_durations_and_usage(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await run_task(client, runner, llm, *full_run())

    trace = (await client.get(f"/tasks/{task_id}/trace")).json()

    assert trace["status"] == "completed"
    timeline = trace["timeline"]
    assert [e["at"] for e in timeline] == sorted(e["at"] for e in timeline)
    task_events = [e["event"] for e in timeline if e["source"] == "task"]
    assert task_events == [
        "pending",
        "planning",
        "planned",
        "executing",
        "synthesizing",
        "completed",
    ]
    step_events = [(e["step_id"], e["event"]) for e in timeline if e["source"] == "step"]
    assert sorted(step_events) == sorted(
        [(f"step_{n}", event) for n in (1, 2, 3) for event in ("started", "completed")]
    )
    finished = next(
        e
        for e in timeline
        if e["source"] == "step" and e["step_id"] == "step_3" and e["event"] == "completed"
    )
    assert (finished["agent"], finished["action"], finished["attempts"]) == (
        "research",
        "summarize",
        1,
    )
    assert finished["tokens"] == {"input": 150, "output": 30, "total": 180}
    assert [e["event"] for e in timeline if e["source"] == "result"] == ["stored"]
    durations = trace["durations_ms"]
    assert all(
        durations[k] is not None and durations[k] >= 0
        for k in ("planning", "execution", "synthesis", "total")
    )
    assert trace["not_run"] == []


async def test_the_trace_names_steps_that_never_ran_and_why(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await run_task(
        client,
        runner,
        llm,
        fake_response(PLAN),
        LLMPermanentError("blocked"),
        fake_response("- b"),
        fake_response(synthesis_json()),
    )

    trace = (await client.get(f"/tasks/{task_id}/trace")).json()

    assert trace["not_run"] == [
        {
            "step_id": "step_3",
            "agent": "research",
            "action": "summarize",
            "status": "skipped",
            "reason": "not run: depends on step_1, which did not complete",
        }
    ]


async def test_a_failed_task_shows_where_and_why_in_its_trace(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await run_task(client, runner, llm, LLMPermanentError("bad key"))

    trace = (await client.get(f"/tasks/{task_id}/trace")).json()

    last = trace["timeline"][-1]
    assert (last["event"], last["failure_stage"]) == ("failed", "planning")
    assert last["error"] == "LLM call failed during planning: bad key"
    assert trace["durations_ms"]["planning"] is not None
    assert trace["durations_ms"]["execution"] is None


async def test_a_cancel_is_in_the_trace(client: AsyncClient) -> None:
    task_id = (await client.post("/tasks", json=GOAL)).json()["task_id"]
    await client.post(f"/tasks/{task_id}/cancel")

    events = [e["event"] for e in (await client.get(f"/tasks/{task_id}/trace")).json()["timeline"]]

    assert events == ["pending", "cancelled"]


async def test_the_trace_of_an_unknown_task_is_404(client: AsyncClient) -> None:
    assert (await client.get("/tasks/task_" + "0" * 32 + "/trace")).status_code == 404


# --- token usage -------------------------------------------------------------------------


async def test_usage_adds_up_every_stage(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await run_task(client, runner, llm, *full_run())

    usage = (await client.get(f"/tasks/{task_id}")).json()["usage"]

    assert usage == {
        "planning": {"input": 300, "output": 60, "total": 360},
        "execution": {"input": 350, "output": 70, "total": 420},
        "synthesis": {"input": 400, "output": 80, "total": 480},
        "total": {"input": 1050, "output": 210, "total": 1260},
    }


async def test_planning_tokens_include_rejected_answers_and_retries(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM, sessionmaker: Sessions
) -> None:
    # A rejected plan (repair), then a transient error (service retry), then a valid plan.
    invalid = plan_json(plan_step("step_1", "step_9"))
    task_id = await run_task(
        client,
        runner,
        llm,
        fake_response(invalid, input_tokens=10, output_tokens=1),
        LLMTransientError("429"),
        fake_response(PLAN, input_tokens=20, output_tokens=2),
        fake_response("- a"),
        fake_response("- b"),
        fake_response("s"),
        fake_response(synthesis_json()),
    )

    async with sessionmaker() as session:
        plan = await session.scalar(select(ExecutionPlan).where(ExecutionPlan.task_id == task_id))
    assert plan is not None
    assert (plan.planner_input_tokens, plan.planner_output_tokens, plan.planner_attempts) == (
        30,
        3,
        3,
    )


async def test_the_tokens_of_a_failed_planning_are_kept(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    invalid = plan_json(plan_step("step_1", "step_9"))
    task_id = await run_task(
        client,
        runner,
        llm,
        fake_response(invalid, input_tokens=10, output_tokens=1),
        fake_response(invalid, input_tokens=12, output_tokens=2),
    )

    usage = (await client.get(f"/tasks/{task_id}")).json()["usage"]

    assert usage["planning"] == {"input": 22, "output": 3, "total": 25}
    assert usage["total"]["total"] == 25


# --- GET /agents -------------------------------------------------------------------------


async def test_agents_lists_every_agent_its_actions_and_who_searches(client: AsyncClient) -> None:
    agents = (await client.get("/agents")).json()

    assert [a["name"] for a in agents] == ["analysis", "code", "research", "writing"]
    by_name = {a["name"]: a for a in agents}
    assert [x["name"] for x in by_name["code"]["actions"]] == [
        "explain_code",
        "fix_bug",
        "review_code",
        "write_code",
    ]
    assert all(x["description"] for a in agents for x in a["actions"])
    assert {a["name"] for a in agents if a["searches_web"]} == {"research"}


# --- monitoring --------------------------------------------------------------------------


async def test_metrics_count_tasks_steps_and_tokens(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    await run_task(client, runner, llm, *full_run())
    await client.post("/tasks", json=GOAL)  # one more, still pending

    metrics = (await client.get("/metrics")).json()

    assert metrics["tasks_by_status"] == {"completed": 1, "pending": 1}
    assert metrics["steps_by_status"] == {"completed": 3}
    assert metrics["tokens"]["total"] == 1260
    assert metrics["llm"] == {"provider": "FakeLLM", "configured": True}
    assert metrics["uptime_seconds"] >= 0


async def test_ready_when_the_database_answers_and_a_provider_is_configured(
    client: AsyncClient,
) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["database"] == {"ok": True}


async def test_not_ready_without_a_provider(database_url: str, engine: AsyncEngine) -> None:
    app = create_app(
        Settings(database_url=database_url, llm_provider="none"), runner=ManualRunner()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/ready")
        live = await client.get("/health")
    await app.state.engine.dispose()

    assert response.status_code == 503
    llm = response.json()["checks"]["llm"]
    assert (llm["ok"], llm["provider"]) == (False, "none")
    assert live.status_code == 200  # alive, just not ready


async def test_not_ready_when_the_database_is_down() -> None:
    app = create_app(
        Settings(database_url="postgresql+asyncpg://u:p@127.0.0.1:1/nothing"), llm=FakeLLM()
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/ready")
    await app.state.engine.dispose()

    assert response.status_code == 503
    assert response.json()["checks"]["database"]["ok"] is False


async def test_deleting_a_task_deletes_its_trace(sessionmaker: Sessions) -> None:
    task = make_task()
    async with sessionmaker() as session:
        TaskRepository(session).add(task)
        await session.commit()
        await session.execute(text("DELETE FROM tasks WHERE id = :id"), {"id": task.id})
        await session.commit()
        assert await session.scalar(select(func.count()).select_from(TaskEvent)) == 0
