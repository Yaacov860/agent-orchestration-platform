"""POST /tasks to GET /result with all four agents, against the real database: the plan routes
steps to research, analysis, writing and code; outputs flow along the dependencies; synthesis
combines them with provenance."""

import json

from factories import plan_json, plan_step, synthesis_json
from fakes import ByCallerLLM, ManualRunner
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from orchestrator.config import Settings
from orchestrator.llm import LLMPermanentError, LLMRequest, LLMResponse, Source
from orchestrator.llm.fake import fake_response
from orchestrator.main import create_app

GOAL = {
    "goal": "Pick a first language for a beginner, write them a short guide and a starter script"
}
PLAN = plan_json(
    plan_step("step_1", agent="research", action="research", input="Python for beginners"),
    plan_step("step_2", agent="research", action="research", input="JavaScript for beginners"),
    plan_step(
        "step_3", "step_1", "step_2", agent="analysis", action="compare", input="Compare both"
    ),
    plan_step("step_4", "step_3", agent="writing", action="write", input="Write the guide"),
    plan_step("step_5", "step_3", agent="code", action="write_code", input="Starter script"),
)
PY = Source(title="Python docs", url="https://docs.python.org")
MDN = Source(title="MDN", url="https://developer.mozilla.org")
CODE = 'Here it is:\n```python\nimport urllib.request\nurllib.request.urlopen("https://api.example.com/v1")\n```'


def research(request: LLMRequest) -> LLMResponse:
    """Each research step gets its own findings and the page its search "used"."""
    if "Python for beginners" in request.user_prompt:
        return fake_response("- Python reads like English", sources=[PY])
    return fake_response("- JavaScript runs in every browser", sources=[MDN])


async def run_task(database_url: str, llm: ByCallerLLM) -> tuple[dict, dict, dict]:  # type: ignore[type-arg]
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url), llm=llm, runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json=GOAL)).json()["task_id"]
        await runner.run_all()  # plan, execute, synthesize
        status = (await client.get(f"/tasks/{task_id}")).json()
        steps = (await client.get(f"/tasks/{task_id}/steps")).json()
        result = await client.get(f"/tasks/{task_id}/result")
    await app.state.engine.dispose()
    return status, steps, result.json()


def replies(**overrides: object) -> ByCallerLLM:
    base: dict[str, object] = {
        "planner": PLAN,
        "research": [research, research],
        "analysis": "| | Python | JavaScript |\n## Verdict\nPython, per step_1",
        "writing": "# Your first language: Python\nStart here.",
        "code": CODE,
        "synthesis": synthesis_json(
            "Start with Python. The guide and the script below are your first steps.\n\n"
            'Script: urlopen("https://api.example.com/v1")',
            findings=[
                ("Python reads like English; JavaScript runs everywhere", ["step_1", "step_2"]),
                ("Python is the better first language here", ["step_3"]),
                ("The guide recommends starting with Python", ["step_3", "step_4"]),
            ],
        ),
    }
    return ByCallerLLM(base | overrides)  # type: ignore[arg-type]


async def test_all_four_agents_run_and_their_outputs_end_in_one_sourced_result(
    database_url: str, engine: AsyncEngine
) -> None:
    research_calls = [research, research]
    llm = replies(research=research_calls)

    status, steps, result = await run_task(database_url, llm)

    assert (status["status"], status["failure_stage"]) == ("completed", None)
    assert [(s["step_id"], s["agent"], s["action"], s["status"]) for s in steps["steps"]] == [
        ("step_1", "research", "research", "completed"),
        ("step_2", "research", "research", "completed"),
        ("step_3", "analysis", "compare", "completed"),
        ("step_4", "writing", "write", "completed"),
        ("step_5", "code", "write_code", "completed"),
    ]
    assert steps["parallel_groups"] == [["step_1", "step_2"], ["step_3"], ["step_4", "step_5"]]
    assert [who for who, _ in llm.calls].count("synthesis") == 1

    # Data flowed along the declared edges only.
    [analysis] = llm.requests("analysis")
    assert "Input from step_1 (research agent)" in analysis.user_prompt
    assert "Input from step_2 (research agent)" in analysis.user_prompt
    [writing] = llm.requests("writing")
    assert "Input from step_3 (analysis agent)" in writing.user_prompt
    assert "Input from step_1" not in writing.user_prompt

    # Synthesis saw every agent's output, labelled, and was told to keep the deliverables.
    [synthesis] = llm.requests("synthesis")
    for label in (
        "(analysis agent, action compare)",
        "(writing agent, action write)",
        "(code agent, action write_code)",
    ):
        assert label in synthesis.user_prompt
    assert "build the answer on it and keep it intact" in synthesis.system_prompt

    assert result["synthesized_from"] == ["step_1", "step_2", "step_3", "step_4", "step_5"]
    assert result["partial"] is False
    # Sources: only what the research providers returned, tied to those steps.
    assert [(s["url"], s["step_ids"]) for s in result["sources"]] == [
        (PY.url, ["step_1"]),
        (MDN.url, ["step_2"]),
    ]
    assert [f["verification"] for f in result["key_findings"]] == [
        "corroborated",  # two independent research steps
        "single_step",  # one analysis step
        "single_step",  # step_4 derives from step_3: not a second opinion
    ]
    # The endpoint from the generated code is quoted, not invented: accepted, but no source.
    assert "https://api.example.com/v1" in result["content"]
    assert "https://api.example.com/v1" not in json.dumps(result["sources"])


async def test_a_failed_writing_step_still_yields_a_partial_result_that_says_so(
    database_url: str, engine: AsyncEngine
) -> None:
    llm = replies(
        research=[research, research],
        writing=LLMPermanentError("content blocked"),
        synthesis=synthesis_json(
            "Start with Python; the guide could not be written.", findings=[("x", ["step_3"])]
        ),
    )

    status, steps, result = await run_task(database_url, llm)

    assert status["status"] == "completed"
    assert {s["step_id"]: s["status"] for s in steps["steps"]}["step_4"] == "failed"
    assert result["partial"] is True
    assert "step_4" not in result["synthesized_from"]
    assert any(
        note.startswith("step_4 (writing: write) failed: content blocked")
        for note in result["limitations"]
    )


async def test_without_a_provider_no_agent_runs_and_the_task_fails_at_planning(
    database_url: str, engine: AsyncEngine
) -> None:
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url, llm_provider="none"), runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json=GOAL)).json()["task_id"]
        await runner.run_all()
        result = (await client.get(f"/tasks/{task_id}/result")).json()
    await app.state.engine.dispose()

    assert (result["status"], result["failure_stage"]) == ("failed", "planning")
