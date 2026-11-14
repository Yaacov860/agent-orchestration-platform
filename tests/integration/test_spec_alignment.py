"""The response fields the assignment specifies, through the API and the real database:
cancel's `completed_steps`, `GET /agents` as `{"agents": [...]}`, the execution trace on
`GET /tasks/{id}`, and the Final Result Format fields on `GET /tasks/{id}/result`."""

import asyncio

from factories import plan_json, plan_step, synthesis_json
from fakes import ManualRunner, is_planner
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from orchestrator.config import Settings
from orchestrator.llm import LLMRequest, LLMResponse
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.main import create_app

GOAL = {"goal": "Compare Python and JavaScript for beginners", "output_format": "markdown"}
PLAN = plan_json(
    plan_step("step_1"),
    plan_step("step_2"),
    plan_step("step_3", "step_1", "step_2", action="summarize"),
)
ANSWER = "# Python first\n\nIt reads like English."


async def completed_task(client: AsyncClient, runner: ManualRunner, llm: FakeLLM) -> str:
    llm.add(
        fake_response(PLAN, input_tokens=300, output_tokens=60),
        fake_response("- a", input_tokens=100, output_tokens=20),
        fake_response("- b", input_tokens=100, output_tokens=20),
        fake_response("summary", input_tokens=150, output_tokens=30),
        fake_response(synthesis_json(ANSWER), input_tokens=400, output_tokens=80),
    )
    task_id: str = (await client.post("/tasks", json=GOAL)).json()["task_id"]
    await runner.run_all()
    return task_id


# --- GET /tasks/{id}/result ---------------------------------------------------------------


async def test_the_result_carries_the_final_result_format_fields(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await completed_task(client, runner, llm)

    result = (await client.get(f"/tasks/{task_id}/result")).json()
    trace = (await client.get(f"/tasks/{task_id}/trace")).json()

    assert result["content"] == ANSWER  # existing fields stay
    assert (result["format"], result["word_count"]) == ("markdown", 7)  # "#" counts as a word
    assert result["result"] == {"content": ANSWER, "format": "markdown", "word_count": 7}
    assert result["total_tokens"] == 300 + 60 + 2 * (100 + 20) + 150 + 30 + 400 + 80
    assert result["total_tokens"] == trace["usage"]["total"]["total"]
    assert result["total_time_ms"] == trace["durations_ms"]["total"]
    assert result["total_time_ms"] is not None and result["total_time_ms"] >= 0
    assert result["execution_trace"] == trace["timeline"]


# --- GET /tasks/{id} ----------------------------------------------------------------------


async def test_the_status_carries_the_execution_trace(
    client: AsyncClient, runner: ManualRunner, llm: FakeLLM
) -> None:
    task_id = await completed_task(client, runner, llm)

    status = (await client.get(f"/tasks/{task_id}")).json()
    trace = (await client.get(f"/tasks/{task_id}/trace")).json()

    assert status["execution_trace"] == trace["timeline"]
    assert [e["event"] for e in status["execution_trace"] if e["source"] == "task"] == [
        "pending",
        "planning",
        "planned",
        "executing",
        "synthesizing",
        "completed",
    ]


async def test_a_new_task_already_has_its_first_trace_event(client: AsyncClient) -> None:
    task_id = (await client.post("/tasks", json=GOAL)).json()["task_id"]

    trace = (await client.get(f"/tasks/{task_id}")).json()["execution_trace"]

    assert [(e["source"], e["event"]) for e in trace] == [("task", "pending")]


# --- GET /agents --------------------------------------------------------------------------


async def test_agents_are_listed_as_the_assignment_specifies(client: AsyncClient) -> None:
    body = (await client.get("/agents")).json()

    assert list(body) == ["agents"]
    for agent in body["agents"]:
        assert agent["status"] == "available"
        assert agent["capabilities"] == sorted(a["name"] for a in agent["actions"])
        assert {"name", "description", "actions", "searches_web"} <= set(agent)  # kept
    by_name = {a["name"]: a for a in body["agents"]}
    assert by_name["research"]["capabilities"] == ["research", "summarize"]


# --- POST /tasks/{id}/cancel --------------------------------------------------------------


class HoldsTheSecondStep:
    """Answers the planner and step_1 at once; holds step_2 until it is cancelled."""

    def __init__(self) -> None:
        self.second_step_started = asyncio.Event()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if is_planner(request):
            return fake_response(plan_json(plan_step("step_1"), plan_step("step_2", "step_1")))
        if "instruction for step_2" in request.user_prompt:
            self.second_step_started.set()
            await asyncio.Event().wait()  # never answers: only a cancel ends it
        return fake_response("- finding", input_tokens=10, output_tokens=5)


async def test_cancelling_mid_execution_lists_the_steps_that_completed(
    database_url: str, engine: AsyncEngine
) -> None:
    llm = HoldsTheSecondStep()
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url), llm=llm, runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json=GOAL)).json()["task_id"]
        running = asyncio.create_task(runner.run_all())
        await asyncio.wait_for(llm.second_step_started.wait(), timeout=5)

        response = await client.post(f"/tasks/{task_id}/cancel")
        await asyncio.wait_for(running, timeout=5)  # the executor stops step_2 at once
        steps = (await client.get(f"/tasks/{task_id}/steps")).json()["steps"]
    await app.state.engine.dispose()

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "status": "cancelled",
        "completed_steps": ["step_1"],
    }
    assert [(s["step_id"], s["status"]) for s in steps] == [
        ("step_1", "completed"),
        ("step_2", "cancelled"),
    ]
