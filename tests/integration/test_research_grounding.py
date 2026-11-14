"""Bounded grounding through the API and the real database: the per-task budget, the step's
grounding status, search counts per step, task, trace and metrics, and the result's
limitations. A scripted FakeLLM stands in for a provider with search."""

from typing import Any

import pytest
from factories import make_plan, make_task, plan_json, plan_step, synthesis_json
from fakes import ManualRunner
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from orchestrator.config import Settings
from orchestrator.domain.status import StepStatus
from orchestrator.llm import LLMResponse, Source
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.main import create_app

PAGE = Source(title="Python docs", url="https://docs.python.org")
PLAN = plan_json(
    plan_step("step_1"),
    plan_step("step_2"),
    plan_step("step_3", "step_1", "step_2", action="summarize"),
)


def researched(text: str) -> LLMResponse:
    return fake_response(text, sources=[PAGE], search_queries=["python for beginners", "pep 8"])


async def run_task(database_url: str, **settings: object) -> dict[str, Any]:
    llm = FakeLLM(
        [
            fake_response(PLAN),
            researched("- Python reads like English"),
            researched("- Python is widely taught"),
            fake_response("- summary"),
            fake_response(synthesis_json(findings=[("Python suits beginners", ["step_1"])])),
        ]
    )
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url, **settings), llm=llm, runner=runner)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json={"goal": "Python for beginners"})).json()[
            "task_id"
        ]
        await runner.run_all()
        out = {
            "status": (await client.get(f"/tasks/{task_id}")).json(),
            "steps": (await client.get(f"/tasks/{task_id}/steps")).json()["steps"],
            "trace": (await client.get(f"/tasks/{task_id}/trace")).json(),
            "result": (await client.get(f"/tasks/{task_id}/result")).json(),
            "metrics": (await client.get("/metrics")).json(),
        }
    await app.state.engine.dispose()
    return out


async def test_a_budget_of_one_grounds_one_parallel_research_step_and_labels_the_other(
    database_url: str, engine: AsyncEngine
) -> None:
    out = await run_task(database_url, gemini_max_grounded_calls_per_task=1)

    assert out["status"]["status"] == "completed"
    research = {s["step_id"]: s for s in out["steps"] if s["action"] == "research"}
    by_status = {s["output"]["grounding"]["status"]: s for s in research.values()}
    assert set(by_status) == {"grounded", "unverified"}

    grounded, unverified = by_status["grounded"], by_status["unverified"]
    assert grounded["search"] == {"grounded_calls": 1, "queries": 2}
    assert grounded["output"]["grounding"]["search_queries"] == ["python for beginners", "pep 8"]
    assert grounded["output"]["sources"] == [{"title": "Python docs", "url": PAGE.url}]

    reason = unverified["output"]["grounding"]["reason"]
    assert "grounded-call budget is used up (1 of 1" in reason
    assert unverified["output"]["content"].startswith(f"Unverified: {reason}.")
    assert unverified["output"]["sources"] == []
    assert unverified["search"] == {"grounded_calls": 0, "queries": 0}

    summary = next(s for s in out["steps"] if s["action"] == "summarize")
    assert "grounding" not in summary["output"]
    assert summary["search"] == {"grounded_calls": 0, "queries": 0}


async def test_search_is_counted_for_the_task_the_trace_and_metrics(
    database_url: str, engine: AsyncEngine
) -> None:
    out = await run_task(database_url, gemini_max_grounded_calls_per_task=1)

    expected = {"grounded_calls": 1, "queries": 2}
    assert out["status"]["search"] == expected
    assert out["trace"]["search"] == expected
    ends = [
        e for e in out["trace"]["timeline"] if e["source"] == "step" and e["event"] != "started"
    ]
    assert sum(e["search"]["grounded_calls"] for e in ends) == 1
    assert out["metrics"]["search"] == expected
    # Tokens are reported as before: search is a separate figure, not folded into usage.
    assert set(out["status"]["usage"]) == {"planning", "execution", "synthesis", "total"}


async def test_the_result_names_the_unverified_step_among_its_limitations(
    database_url: str, engine: AsyncEngine
) -> None:
    out = await run_task(database_url, gemini_max_grounded_calls_per_task=1)

    unverified = next(
        s["step_id"]
        for s in out["steps"]
        if s["action"] == "research" and s["output"]["grounding"]["status"] == "unverified"
    )
    notes = [n for n in out["result"]["limitations"] if n.startswith(f"{unverified} (research")]
    assert len(notes) == 1
    assert "is unverified: the task's grounded-call budget is used up (1 of 1" in notes[0]
    assert "Its findings come from the model, not from cited pages." in notes[0]


async def test_with_the_default_budget_both_research_steps_are_grounded(
    database_url: str, engine: AsyncEngine
) -> None:
    out = await run_task(database_url)  # GEMINI_MAX_GROUNDED_CALLS_PER_TASK defaults to 2

    research = [s for s in out["steps"] if s["action"] == "research"]
    assert [s["output"]["grounding"]["status"] for s in research] == ["grounded", "grounded"]
    assert out["status"]["search"] == {"grounded_calls": 2, "queries": 4}
    assert not any("is unverified" in n for n in out["result"]["limitations"])


@pytest.mark.parametrize("column", ["grounded_calls", "search_queries"])
async def test_the_database_refuses_a_negative_search_counter(
    sessionmaker: async_sessionmaker[AsyncSession], column: str
) -> None:
    task = make_task()
    async with sessionmaker() as session:
        session.add(task)
        session.add(make_plan(task.id, 1, [("step_1", StepStatus.PENDING)]))
        await session.commit()
        with pytest.raises(IntegrityError, match="ck_execution_steps_search_counters_non_negative"):
            await session.execute(text(f"UPDATE execution_steps SET {column} = -1"))
