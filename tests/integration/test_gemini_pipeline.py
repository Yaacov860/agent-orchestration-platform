"""The whole pipeline on ONE GeminiLLM: planner, all four agents and synthesis share it, through
the API and the real database. Only the network call is replaced: answers are real SDK
response objects, so the provider adapter runs end to end, retries and timeouts included."""

import asyncio
from typing import Any

import httpx
import pytest
from factories import plan_json, plan_step, synthesis_json
from fakes import ManualRunner, caller_of
from google.genai import errors, types
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from orchestrator.config import Settings
from orchestrator.llm import LLMRequest
from orchestrator.llm.gemini import GeminiLLM
from orchestrator.main import create_app

PLAN = plan_json(
    plan_step("step_1", agent="research", action="research", input="Python for beginners"),
    plan_step("step_2", "step_1", agent="analysis", action="evaluate", input="Evaluate Python"),
    plan_step("step_3", "step_2", agent="writing", action="write", input="Write a short guide"),
    plan_step("step_4", "step_2", agent="code", action="write_code", input="A hello-world script"),
)
SYNTHESIS = synthesis_json(
    "Start with Python: the guide and script follow.",
    findings=[("Python reads like English", ["step_1"]), ("Python suits beginners", ["step_2"])],
)
ANSWERS = {
    "planner": PLAN,
    "research": "- Python reads like English",
    "analysis": "## Strengths\n- readable (per step_1)",
    "writing": "# Your first language",
    "code": "```python\nprint('hello')\n```",
    "synthesis": SYNTHESIS,
}


def answer(text: str, *, search: bool) -> types.GenerateContentResponse:
    grounding = (
        types.GroundingMetadata(
            web_search_queries=["python for beginners"],
            grounding_chunks=[
                types.GroundingChunk(
                    web=types.GroundingChunkWeb(uri="https://docs.python.org", title="Python docs")
                )
            ],
        )
        if search
        else None
    )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
                grounding_metadata=grounding,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50, candidates_token_count=20
        ),
        model_version="gemini-test-001",
    )


class GeminiNetwork:
    """The network behind `client.aio.models`: answers by caller, after scripted failures."""

    def __init__(self, failures: dict[str, list[BaseException]] | None = None) -> None:
        self._failures = failures or {}
        self.calls: list[tuple[str, types.GenerateContentConfig]] = []

    async def generate_content(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        assert isinstance(config.system_instruction, str)
        caller = caller_of(
            LLMRequest(system_prompt=config.system_instruction, user_prompt=str(contents))
        )
        self.calls.append((caller, config))
        await asyncio.sleep(0)
        if self._failures.get(caller):
            raise self._failures[caller].pop(0)
        return answer(ANSWERS[caller], search=config.tools is not None)


async def run(
    database_url: str, network: GeminiNetwork, *, search_grounding: bool = True, **settings: object
) -> dict[str, Any]:
    # Grounding is off by default; these tests drive the grounded path on purpose.
    llm = GeminiLLM(
        api_key="AIza-test-only",
        model="gemini-test",
        models=network,
        search_grounding=search_grounding,
    )
    runner = ManualRunner()
    app = create_app(Settings(database_url=database_url, **settings), llm=llm, runner=runner)  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task_id = (await client.post("/tasks", json={"goal": "Teach a beginner to code"})).json()[
            "task_id"
        ]
        await runner.run_all()
        out = {
            "status": (await client.get(f"/tasks/{task_id}")).json(),
            "steps": (await client.get(f"/tasks/{task_id}/steps")).json()["steps"],
            "result": (await client.get(f"/tasks/{task_id}/result")).json(),
        }
    await app.state.engine.dispose()
    return out


async def test_the_planner_all_four_agents_and_synthesis_run_on_the_same_gemini_client(
    database_url: str, engine: AsyncEngine
) -> None:
    network = GeminiNetwork()

    out = await run(database_url, network)

    assert out["status"]["status"] == "completed"
    callers = [caller for caller, _ in network.calls]
    assert callers[0] == "planner" and callers[-1] == "synthesis"
    assert sorted(set(callers[1:-1])) == ["analysis", "code", "research", "writing"]
    # Only the research step asked for Google Search grounding.
    searched = {caller for caller, config in network.calls if config.tools is not None}
    assert searched == {"research"}
    # Every step's tokens came from the provider's usage metadata.
    assert all((s["tokens"]["input"], s["tokens"]["output"]) == (50, 20) for s in out["steps"])
    # The grounded page reached the result as a source of the research step.
    assert out["result"]["sources"] == [
        {"title": "Python docs", "url": "https://docs.python.org", "step_ids": ["step_1"]}
    ]
    assert out["result"]["synthesis"]["tokens"] == {"input": 50, "output": 20, "total": 70}
    # The grounded call's queries, as Gemini reported them, counted for the step and the task.
    research = next(s for s in out["steps"] if s["agent"] == "research")
    assert research["output"]["grounding"] == {
        "status": "grounded",
        "reason": None,
        "search_queries": ["python for beginners"],
    }
    assert research["search"] == {"grounded_calls": 1, "queries": 1}
    assert out["status"]["search"] == {"grounded_calls": 1, "queries": 1}


async def test_transient_gemini_errors_are_retried_at_every_stage(
    database_url: str, engine: AsyncEngine
) -> None:
    def rate_limited() -> errors.APIError:
        return errors.ClientError(
            429, {"error": {"code": 429, "message": "Quota", "status": "RESOURCE_EXHAUSTED"}}, None
        )

    network = GeminiNetwork(
        {
            "planner": [rate_limited()],
            "research": [httpx.ReadTimeout("slow")],
            "code": [
                errors.ServerError(
                    503,
                    {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}},
                    None,
                )
            ],
            "synthesis": [rate_limited()],
        }
    )

    # Fast retries: no real waiting in the test.
    out = await run(database_url, network, step_retry_base_delay_seconds=0)

    assert out["status"]["status"] == "completed"
    attempts = {s["agent"]: s["attempts"] for s in out["steps"]}
    assert (attempts["research"], attempts["code"], attempts["writing"]) == (2, 2, 1)
    callers = [caller for caller, _ in network.calls]
    assert callers.count("planner") == 2 and callers.count("synthesis") == 2


async def test_a_rejected_key_fails_the_task_at_planning_without_retrying(
    database_url: str, engine: AsyncEngine
) -> None:
    network = GeminiNetwork(
        {
            "planner": [
                errors.ClientError(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": "API key not valid.",
                            "status": "INVALID_ARGUMENT",
                        }
                    },
                    None,
                )
            ]
        }
    )

    out = await run(database_url, network)

    assert (out["status"]["status"], out["status"]["failure_stage"]) == ("failed", "planning")
    assert "Gemini rejected the API key (400); check GEMINI_API_KEY" in out["status"]["error"]
    assert [caller for caller, _ in network.calls] == ["planner"]  # permanent: no retry


async def test_a_step_whose_gemini_call_keeps_timing_out_fails_and_the_rest_completes(
    database_url: str, engine: AsyncEngine
) -> None:
    def timeouts() -> list[BaseException]:
        return [httpx.ReadTimeout("slow") for _ in range(3)]

    network = GeminiNetwork({"code": timeouts()})

    out = await run(database_url, network, step_retry_base_delay_seconds=0)

    code = next(s for s in out["steps"] if s["agent"] == "code")
    assert (code["status"], code["attempts"]) == ("failed", 3)
    assert code["error"]["message"] == "Gemini did not answer within 60s (GEMINI_TIMEOUT_SECONDS)"
    assert code["error"]["retryable"] is True
    assert out["result"]["partial"] is True  # synthesis went ahead without the code step


@pytest.mark.parametrize(
    ("search_grounding", "budget", "reason"),
    [
        (False, 2, "Google Search grounding is disabled (GEMINI_SEARCH_GROUNDING=false)"),
        (True, 0, "the task's grounded-call budget is used up (0 of 0 calls"),
    ],
)
async def test_without_grounding_research_runs_unsearched_and_the_result_says_so(
    database_url: str, engine: AsyncEngine, search_grounding: bool, budget: int, reason: str
) -> None:
    network = GeminiNetwork()

    out = await run(
        database_url,
        network,
        search_grounding=search_grounding,
        gemini_max_grounded_calls_per_task=budget,
    )

    assert out["status"]["status"] == "completed"
    assert all(config.tools is None for _, config in network.calls)  # no call searched
    research = next(s for s in out["steps"] if s["agent"] == "research")
    assert research["output"]["grounding"]["status"] == "unverified"
    assert research["output"]["grounding"]["reason"].startswith(reason)
    assert research["output"]["content"].startswith("Unverified: ")
    assert research["output"]["sources"] == []
    assert out["status"]["search"] == {"grounded_calls": 0, "queries": 0}
    assert any(
        "step_1 (research: research) is unverified" in n for n in out["result"]["limitations"]
    )
