from datetime import UTC, datetime, timedelta

import pytest
from factories import make_plan, make_task, plan_json, plan_step, synthesis_json
from fakes import GatedLLM, ManualRunner
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from orchestrator.agents import AgentOutput
from orchestrator.config import Settings
from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.domain.ids import is_task_id, new_task_id
from orchestrator.domain.status import StepStatus, TaskStatus
from orchestrator.llm import LLMPermanentError, LLMRequest, LLMResponse, Source
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.llm.unconfigured import NO_PROVIDER_MESSAGE
from orchestrator.main import create_app
from orchestrator.runner import BackgroundRunner

Sessions = async_sessionmaker[AsyncSession]

BLOG_POST = {
    "goal": "Write a comparison blog post about Python vs JavaScript for beginners",
    "constraints": {"max_words": 1500, "tone": "friendly", "include_code_examples": True},
    "output_format": "markdown",
}


async def count_tasks(sessionmaker: Sessions) -> int:
    async with sessionmaker() as session:
        return await session.scalar(select(func.count()).select_from(Task)) or 0


# --- POST /tasks: success ---------------------------------------------------------------


async def test_submit_stores_the_task_and_returns_its_id(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    response = await client.post("/tasks", json=BLOG_POST)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert is_task_id(body["task_id"])
    assert response.headers["location"] == f"/tasks/{body['task_id']}"

    async with sessionmaker() as session:
        row = await session.get(Task, body["task_id"])
    assert row is not None
    assert row.goal == BLOG_POST["goal"]
    assert row.constraints == BLOG_POST["constraints"]
    assert row.status is TaskStatus.PENDING


async def test_submit_applies_defaults_and_trims_the_goal(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    response = await client.post("/tasks", json={"goal": "  Explain asyncio  "})

    assert response.status_code == 201
    async with sessionmaker() as session:
        row = await session.get(Task, response.json()["task_id"])
    assert row is not None
    assert (row.goal, row.constraints, row.output_format, row.deadline) == (
        "Explain asyncio",
        {},
        "markdown",
        None,
    )


async def test_unknown_constraints_are_kept_alongside_known_ones(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    constraints = {"max_words": 300, "audience": "teenagers", "avoid": ["jargon", "slang"]}

    response = await client.post("/tasks", json={"goal": "g", "constraints": constraints})

    async with sessionmaker() as session:
        row = await session.get(Task, response.json()["task_id"])
    assert row is not None
    assert row.constraints == constraints


async def test_a_future_timezone_aware_deadline_is_accepted(client: AsyncClient) -> None:
    deadline = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    response = await client.post("/tasks", json={"goal": "g", "deadline": deadline})

    assert response.status_code == 201


# --- POST /tasks: validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        pytest.param({}, "goal", id="missing goal"),
        pytest.param({"goal": "   "}, "goal", id="blank goal"),
        pytest.param({"goal": "x" * 4001}, "goal", id="goal too long"),
        pytest.param({"goal": "g", "constraint": {}}, "constraint", id="unknown field (typo)"),
        pytest.param({"goal": "g", "constraints": {"max_words": 0}}, "max_words", id="max_words 0"),
        pytest.param(
            {"goal": "g", "constraints": {"max_words": "1500"}}, "max_words", id="max_words string"
        ),
        pytest.param(
            {"goal": "g", "constraints": {"include_code_examples": "yes"}},
            "include_code_examples",
            id="bool as 'yes'",
        ),
        pytest.param({"goal": "g", "output_format": "pdf"}, "output_format", id="bad format"),
        pytest.param(
            {"goal": "g", "deadline": "2020-01-01T00:00:00Z"}, "deadline", id="past deadline"
        ),
        pytest.param(
            {"goal": "g", "deadline": "2099-01-01T00:00:00"}, "deadline", id="naive deadline"
        ),
    ],
)
async def test_invalid_submissions_are_rejected_and_nothing_is_stored(
    client: AsyncClient,
    sessionmaker: Sessions,
    runner: ManualRunner,
    payload: dict[str, object],
    bad_field: str,
) -> None:
    response = await client.post("/tasks", json=payload)

    assert response.status_code == 422
    error_locations = [error["loc"] for error in response.json()["detail"]]
    assert any(bad_field in loc for loc in error_locations), error_locations
    assert await count_tasks(sessionmaker) == 0
    assert runner.jobs == []  # nothing stored, so nothing to plan


async def test_a_body_that_is_not_json_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/tasks", content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422


# --- GET /tasks/{id} --------------------------------------------------------------------


async def test_get_returns_the_stored_task_with_empty_progress_before_planning(
    client: AsyncClient,
) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    response = await client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "pending"
    assert body["goal"] == BLOG_POST["goal"]
    assert body["constraints"] == BLOG_POST["constraints"]
    assert body["error"] is None
    assert body["created_at"] and body["updated_at"]
    assert body["progress"] == {
        "total_steps": 0,
        "completed_steps": 0,
        "failed_steps": 0,
        "current_step": None,
        "running_steps": [],
    }


async def test_progress_counts_the_steps_of_the_latest_plan_only(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    async with sessionmaker() as session:
        session.add(make_plan(task_id, 1, [("step_1", StepStatus.FAILED)]))  # superseded
        session.add(
            make_plan(
                task_id,
                2,
                [
                    ("step_1", StepStatus.COMPLETED),
                    ("step_2", StepStatus.RUNNING),
                    ("step_3", StepStatus.RUNNING),
                    ("step_4", StepStatus.PENDING),
                ],
            )
        )
        await session.commit()

    progress = (await client.get(f"/tasks/{task_id}")).json()["progress"]

    assert progress == {
        "total_steps": 4,
        "completed_steps": 1,
        "failed_steps": 0,
        "current_step": "step_2",
        "running_steps": ["step_2", "step_3"],
    }


async def test_get_an_unknown_task_is_404(client: AsyncClient) -> None:
    response = await client.get(f"/tasks/{new_task_id()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize("bad_id", ["nope", "task_123", "task_" + "G" * 32, "1; DROP TABLE tasks"])
async def test_get_a_malformed_id_is_404(client: AsyncClient, bad_id: str) -> None:
    response = await client.get(f"/tasks/{bad_id}")

    assert response.status_code == 404


# --- background planning ----------------------------------------------------------------

BLOG_PLAN = plan_json(
    plan_step("step_1", input="Research Python's strengths for beginners"),
    plan_step("step_2", input="Research JavaScript's strengths for beginners"),
    plan_step("step_3", "step_1", "step_2", action="summarize", input="Compare the findings"),
)


async def test_submit_schedules_planning_for_the_stored_task_and_returns_at_once(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    response = await client.post("/tasks", json=BLOG_POST)

    task_id = response.json()["task_id"]
    assert runner.names == [f"plan:{task_id}"]
    assert llm.calls == []  # the 201 did not wait for the model


async def test_a_submitted_task_is_planned_and_get_shows_its_steps(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    llm.add(fake_response(BLOG_PLAN))
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    await runner.run_all("plan:")  # plan only; execution stays queued
    body = (await client.get(f"/tasks/{task_id}")).json()

    assert (body["status"], body["error"]) == ("planned", None)
    assert runner.names == [f"execute:{task_id}"]  # a stored plan hands over to execution
    assert body["progress"] == {
        "total_steps": 3,
        "completed_steps": 0,
        "failed_steps": 0,
        "current_step": None,
        "running_steps": [],
    }


async def test_a_task_that_cannot_be_planned_shows_failed_with_the_reason(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    invalid = plan_json(plan_step("step_1", agent="translation", action="translate"))
    llm.add(fake_response(invalid), fake_response(invalid))
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    await runner.run_all()
    body = (await client.get(f"/tasks/{task_id}")).json()

    assert body["status"] == "failed"
    assert body["error"] == (
        "no valid plan after 2 attempts: "
        "step_1: unknown agent 'translation'; available agents: analysis, code, research, writing"
    )
    assert body["progress"]["total_steps"] == 0


async def test_with_no_llm_provider_configured_tasks_are_accepted_then_fail_clearly(
    database_url: str,
    engine: AsyncEngine,  # requested for its TRUNCATE: this test builds its own app
) -> None:
    # The production wiring: no LLM injected, so the app builds one from LLM_PROVIDER=none.
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url, llm_provider="none"), runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        submit = await client.post("/tasks", json=BLOG_POST)
        await runner.run_all()
        body = (await client.get(f"/tasks/{submit.json()['task_id']}")).json()
    await app.state.engine.dispose()

    assert submit.status_code == 201
    assert body["status"] == "failed"
    assert body["error"] == f"LLM call failed during planning: {NO_PROVIDER_MESSAGE}"


async def test_post_tasks_plans_executes_and_stores_every_step_result(
    database_url: str, sessionmaker: Sessions
) -> None:
    """End to end with the real BackgroundRunner: no test double between POST and the rows."""
    llm = FakeLLM(
        [
            fake_response(BLOG_PLAN),
            fake_response("- Python reads like English", input_tokens=10, output_tokens=5),
            fake_response("- JavaScript runs in every browser", input_tokens=10, output_tokens=5),
            fake_response("Python is easier to start; JavaScript is everywhere."),
            fake_response(synthesis_json("Python is easier to start with.")),
        ]
    )
    app = create_app(Settings(database_url=database_url), llm=llm)
    runner: BackgroundRunner = app.state.runner
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
        await runner.wait_idle()
        body = (await client.get(f"/tasks/{task_id}")).json()
    await app.state.engine.dispose()

    assert (body["status"], body["error"]) == ("completed", None)
    assert body["progress"]["completed_steps"] == 3
    async with sessionmaker() as session:
        plan = await session.scalar(select(ExecutionPlan).where(ExecutionPlan.task_id == task_id))
        assert plan is not None
        steps = (
            await session.scalars(
                select(ExecutionStep)
                .where(ExecutionStep.plan_id == plan.id)
                .order_by(ExecutionStep.position)
            )
        ).all()
    assert (plan.version, plan.parallel_groups) == (1, [["step_1", "step_2"], ["step_3"]])
    assert [(s.step_id, s.dependencies, s.status, s.attempts) for s in steps] == [
        ("step_1", [], StepStatus.COMPLETED, 1),
        ("step_2", [], StepStatus.COMPLETED, 1),
        ("step_3", ["step_1", "step_2"], StepStatus.COMPLETED, 1),
    ]
    assert f"<task>\n{BLOG_POST['goal']}\n</task>" in llm.calls[0].user_prompt
    # The two research steps ran in parallel, so which got which reply is not fixed;
    # the summary step got both their outputs.
    # The stored output is an AgentOutput: it loads back into one.
    research_outputs = {AgentOutput.model_validate(s.output).content for s in steps[:2]}
    assert research_outputs == {
        "- Python reads like English",
        "- JavaScript runs in every browser",
    }
    assert all((s.input_tokens, s.output_tokens) == (10, 5) for s in steps[:2])
    summary_prompt = llm.calls[3].user_prompt
    assert "## Input from step_1 (research agent)" in summary_prompt
    assert "## Input from step_2 (research agent)" in summary_prompt
    assert all(content in summary_prompt for content in research_outputs)


# --- GET /tasks/{id}/steps --------------------------------------------------------------


async def test_steps_are_empty_until_a_plan_is_stored(client: AsyncClient) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    response = await client.get(f"/tasks/{task_id}/steps")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "status": "pending",
        "plan_version": None,
        "parallel_groups": [],
        "steps": [],
    }


async def test_steps_show_state_result_tokens_and_timings_after_a_run(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    llm.add(
        fake_response(BLOG_PLAN),
        fake_response("- findings A", input_tokens=30, output_tokens=10),
        fake_response("- findings B", input_tokens=30, output_tokens=10),
        fake_response("the comparison", input_tokens=50, output_tokens=20),
        fake_response(synthesis_json()),
    )
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await runner.run_all()

    body = (await client.get(f"/tasks/{task_id}/steps")).json()

    assert (body["status"], body["plan_version"]) == ("completed", 1)
    assert body["parallel_groups"] == [["step_1", "step_2"], ["step_3"]]
    first, _, last = body["steps"]
    assert {k: first[k] for k in ("step_id", "agent", "action", "dependencies", "status")} == {
        "step_id": "step_1",
        "agent": "research",
        "action": "research",
        "dependencies": [],
        "status": "completed",
    }
    assert first["input"] == "Research Python's strengths for beginners"
    assert first["attempts"] == 1
    assert first["output"]["content"] in {"- findings A", "- findings B"}  # ran in parallel
    assert first["output"]["sources"] == []
    assert first["output"]["confidence"] == 0.3  # no grounded sources: low evidence
    assert first["tokens"] == {"input": 30, "output": 10, "total": 40}
    assert first["error"] is None
    assert first["execution_time_ms"] >= 0
    assert first["started_at"] <= first["completed_at"]
    assert last["dependencies"] == ["step_1", "step_2"]
    assert last["output"] == {"content": "the comparison", "sources": [], "confidence": None}


async def test_steps_show_the_error_of_a_failed_step_and_why_others_did_not_run(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    plan = plan_json(plan_step("step_1"), plan_step("step_2", "step_1", action="summarize"))
    llm.add(fake_response(plan), LLMPermanentError("content blocked"))
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await runner.run_all()

    failed, skipped = (await client.get(f"/tasks/{task_id}/steps")).json()["steps"]

    assert (failed["status"], failed["output"]) == ("failed", None)
    assert failed["error"] == {
        "type": "LLMPermanentError",
        "message": "content blocked",
        "retryable": False,
    }
    assert (skipped["status"], skipped["attempts"], skipped["started_at"]) == ("skipped", 0, None)
    assert skipped["error"] == {
        "type": "DependencyNotCompleted",
        "message": "not run: depends on step_1, which did not complete",
        "retryable": False,
    }


async def test_steps_come_in_plan_order_not_id_order(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    llm.add(fake_response(plan_json(plan_step("step_2"), plan_step("step_1", "step_2"))))
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await runner.run_all("plan:")

    body = (await client.get(f"/tasks/{task_id}/steps")).json()

    assert [s["step_id"] for s in body["steps"]] == ["step_2", "step_1"]
    assert {s["status"] for s in body["steps"]} == {"pending"}


@pytest.mark.parametrize("bad_id", ["nope", "task_" + "0" * 32])
async def test_steps_of_an_unknown_task_are_404(client: AsyncClient, bad_id: str) -> None:
    assert (await client.get(f"/tasks/{bad_id}/steps")).status_code == 404


# --- GET /tasks/{id}/result -------------------------------------------------------------

PY_DOCS = Source(title="Python docs", url="https://docs.python.org")
RESULT = synthesis_json(
    "Python is the easier first language; JavaScript is unavoidable on the web.",
    findings=[
        ("Python reads like English", ["step_1"]),
        ("Both are taught widely", ["step_1", "step_2"]),
    ],
    conflicts=[
        (
            "time to first app",
            [("a week with Python", ["step_1"]), ("a day with JavaScript", ["step_2"])],
        )
    ],
)


async def submit_and_run(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM, *replies: object
) -> str:
    llm.add(*replies)  # type: ignore[arg-type]
    task_id: str = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await runner.run_all()
    return task_id


async def test_the_full_flow_ends_completed_with_a_sourced_result(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await submit_and_run(
        client,
        runner,
        llm,
        fake_response(BLOG_PLAN),
        fake_response("- Python findings", sources=[PY_DOCS]),
        fake_response("- JavaScript findings"),
        fake_response("the comparison"),
        fake_response(RESULT, input_tokens=900, output_tokens=300),
    )
    assert runner.jobs == []  # plan -> execute -> synthesize all ran

    response = await client.get(f"/tasks/{task_id}/result")

    assert response.status_code == 200
    body = response.json()
    assert (body["task_id"], body["status"]) == (task_id, "completed")
    assert body["content"].startswith("Python is the easier first language")
    assert body["key_findings"] == [
        {
            "statement": "Python reads like English",
            "step_ids": ["step_1"],
            "verification": "single_step",
        },
        {
            "statement": "Both are taught widely",
            "step_ids": ["step_1", "step_2"],
            "verification": "corroborated",
        },
    ]
    assert body["conflicts"] == [
        {
            "topic": "time to first app",
            "claims": [
                {"claim": "a week with Python", "step_ids": ["step_1"]},
                {"claim": "a day with JavaScript", "step_ids": ["step_2"]},
            ],
        }
    ]
    # Exactly the page a provider returned, tied to the step that got it. step_1 and step_2
    # ran in parallel, so which of them got the reply with the source is not fixed.
    [source] = body["sources"]
    assert (source["url"], len(source["step_ids"])) == (PY_DOCS.url, 1)
    assert body["partial"] is False
    assert body["synthesized_from"] == ["step_1", "step_2", "step_3"]
    assert "for 1 conflict(s)" in body["confidence_basis"]
    assert body["synthesis"] == {
        "attempts": 1,
        "tokens": {"input": 900, "output": 300, "total": 1200},
    }
    status = (await client.get(f"/tasks/{task_id}")).json()
    assert (status["status"], status["failure_stage"]) == ("completed", None)


async def test_a_task_with_failed_steps_completes_with_a_partial_result(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    plan = plan_json(plan_step("step_1"), plan_step("step_2"))
    task_id = await submit_and_run(
        client,
        runner,
        llm,
        fake_response(plan),
        fake_response("- findings"),
        LLMPermanentError("content blocked"),
        fake_response(synthesis_json("Partial answer.")),
    )

    body = (await client.get(f"/tasks/{task_id}/result")).json()

    assert body["partial"] is True
    assert len(body["synthesized_from"]) == 1
    assert any("failed: content blocked" in note for note in body["limitations"])


async def test_an_unfinished_task_has_no_result_yet(client: AsyncClient) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    response = await client.get(f"/tasks/{task_id}/result")

    assert response.status_code == 409
    assert response.json() == {
        "task_id": task_id,
        "status": "pending",
        "failure_stage": None,
        "error": None,
        "detail": "the task is pending; its result is available once it is completed",
    }


async def test_a_task_that_failed_before_synthesis_says_so(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    plan = plan_json(plan_step("step_1"))
    task_id = await submit_and_run(
        client, runner, llm, fake_response(plan), LLMPermanentError("blocked")
    )

    body = (await client.get(f"/tasks/{task_id}/result")).json()

    assert (body["status"], body["failure_stage"]) == ("failed", "execution")
    assert body["error"] == "no step produced a usable result: step_1 failed: blocked"
    assert body["detail"].startswith("the task failed during execution, before a result")


async def test_a_task_whose_synthesis_failed_says_so_and_points_to_the_steps(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    plan = plan_json(plan_step("step_1"))
    task_id = await submit_and_run(
        client,
        runner,
        llm,
        fake_response(plan),
        fake_response("- findings"),
        LLMPermanentError("bad key"),
    )

    response = await client.get(f"/tasks/{task_id}/result")

    assert response.status_code == 409
    body = response.json()
    assert (body["status"], body["failure_stage"]) == ("failed", "synthesis")
    assert body["error"] == "synthesis failed: LLM call failed during synthesis: bad key"
    assert f"/tasks/{task_id}/steps" in body["detail"]
    steps = (await client.get(f"/tasks/{task_id}/steps")).json()["steps"]
    assert steps[0]["status"] == "completed"  # the work done is still there


async def test_a_cancelled_task_has_no_result(client: AsyncClient) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await client.post(f"/tasks/{task_id}/cancel")

    body = (await client.get(f"/tasks/{task_id}/result")).json()

    assert (body["status"], body["failure_stage"]) == ("cancelled", None)
    assert body["detail"].startswith("the task was cancelled")


@pytest.mark.parametrize("bad_id", ["nope", "task_" + "0" * 32])
async def test_the_result_of_an_unknown_task_is_404(client: AsyncClient, bad_id: str) -> None:
    assert (await client.get(f"/tasks/{bad_id}/result")).status_code == 404


async def test_openapi_documents_both_the_result_and_the_no_result_response(
    client: AsyncClient,
) -> None:
    spec = (await client.get("/openapi.json")).json()
    responses = spec["paths"]["/tasks/{task_id}/result"]["get"]["responses"]

    assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/TaskResultOut"
    )
    assert responses["409"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/NoResultOut"
    )
    assert "404" in responses


# --- POST /tasks/{id}/cancel -------------------------------------------------------------


async def test_a_pending_task_can_be_cancelled_and_is_then_never_planned(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    response = await client.post(f"/tasks/{task_id}/cancel")
    await runner.run_all()  # the queued planning job finds the task no longer pending

    assert response.status_code == 200
    assert response.json() == {"task_id": task_id, "status": "cancelled"}
    assert llm.calls == []
    assert (await client.get(f"/tasks/{task_id}")).json()["status"] == "cancelled"


async def test_a_planned_task_can_be_cancelled_and_its_steps_with_it(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM, sessionmaker: Sessions
) -> None:
    llm.add(fake_response(BLOG_PLAN))
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    await runner.run_all("plan:")

    assert (await client.post(f"/tasks/{task_id}/cancel")).status_code == 200
    await runner.run_all()  # the queued execution job finds the task no longer planned

    async with sessionmaker() as session:
        statuses = (
            await session.scalars(
                select(ExecutionStep.status)
                .join(ExecutionPlan)
                .where(ExecutionPlan.task_id == task_id)
            )
        ).all()
    assert statuses == [StepStatus.CANCELLED] * 3
    assert len(llm.calls) == 1  # only the planner ran


async def test_cancelling_a_finished_task_is_a_conflict(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    task = make_task(status=TaskStatus.COMPLETED)
    async with sessionmaker() as session:
        session.add(task)
        await session.commit()

    response = await client.post(f"/tasks/{task.id}/cancel")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        f"task '{task.id}' is already completed; "
        "only pending, planning, planned, executing, synthesizing tasks can be cancelled"
    )


@pytest.mark.parametrize("bad_id", ["nope", "task_" + "0" * 32])
async def test_cancelling_an_unknown_task_is_404(client: AsyncClient, bad_id: str) -> None:
    assert (await client.post(f"/tasks/{bad_id}/cancel")).status_code == 404


async def test_cancel_through_the_api_stops_a_running_execution_end_to_end(
    database_url: str, sessionmaker: Sessions
) -> None:
    """Real runner: POST, plan, start executing, cancel while two agents are mid-call."""

    class PlanThenHold(GatedLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            if "You are the planner" in request.system_prompt:
                return fake_response(BLOG_PLAN)
            return await super().generate(request)

    llm = PlanThenHold()
    app = create_app(Settings(database_url=database_url), llm=llm)
    runner: BackgroundRunner = app.state.runner
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
        await llm.until_in_flight(2)  # step_1 and step_2 are running
        during = (await client.get(f"/tasks/{task_id}")).json()

        cancel = await client.post(f"/tasks/{task_id}/cancel")
        await runner.wait_idle()
        after = (await client.get(f"/tasks/{task_id}")).json()
    await app.state.engine.dispose()

    assert (during["status"], during["progress"]["running_steps"]) == (
        "executing",
        ["step_1", "step_2"],
    )
    assert cancel.status_code == 200
    assert after["status"] == "cancelled"
    assert after["progress"]["running_steps"] == []
    assert (llm.cancelled, llm.in_flight) == (2, 0)  # the agents were stopped, not awaited


# --- infrastructure failure -------------------------------------------------------------


async def test_database_unavailable_is_503_with_retry_after() -> None:
    app = create_app(Settings(database_url="postgresql+asyncpg://u:p@127.0.0.1:1/nothing"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        submit = await client.post("/tasks", json={"goal": "g"})
        status = await client.get(f"/tasks/{new_task_id()}")

    for response in (submit, status):
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.json() == {"detail": "database unavailable, try again shortly"}
