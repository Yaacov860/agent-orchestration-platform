"""The per-task grounded-call budget inside the executor: shared by parallel research steps,
taken by every attempt, never touched by the other agents."""

import asyncio
from collections.abc import Mapping, Sequence

from orchestrator.agents.factory import AgentFactory
from orchestrator.execution import DependencyExecutor, ExecutionJob, StepOutcome, StepSpec
from orchestrator.llm import LLMTransientError, SearchUsage, Source
from orchestrator.llm.fake import FakeLLM, ScriptItem, fake_response

PAGE = Source(title="Python", url="https://python.org")


class Recorder:
    def __init__(self) -> None:
        self.outcomes: dict[str, StepOutcome] = {}

    async def step_started(self, step_id: str, attempt: int) -> None:
        pass

    async def step_finished(self, outcome: StepOutcome) -> None:
        self.outcomes[outcome.step_id] = outcome

    async def steps_skipped(self, blocked_by: Mapping[str, str]) -> None:
        pass

    async def steps_expired(self, step_ids: Sequence[str]) -> None:
        pass

    async def steps_cancelled(self, step_ids: Sequence[str]) -> None:
        pass

    async def cancel_requested(self) -> bool:
        return False


async def no_sleep(seconds: float) -> None:
    pass


def spec(step_id: str, agent: str = "research", action: str = "research") -> StepSpec:
    return StepSpec(step_id=step_id, agent=agent, action=action, instruction=f"do {step_id}")


def one_group(*steps: StepSpec) -> ExecutionJob:
    """Every step in one parallel group: they all run at once."""
    return ExecutionJob(
        task_id="task_1",
        goal="Compare Python and JavaScript",
        steps=steps,
        parallel_groups=(tuple(s.step_id for s in steps),),
    )


async def run(
    script: Sequence[ScriptItem], job: ExecutionJob, max_grounded_calls: int | None
) -> tuple[dict[str, StepOutcome], FakeLLM]:
    llm = FakeLLM(script)
    executor = DependencyExecutor(
        AgentFactory(llm),
        slots=asyncio.Semaphore(4),
        retry_delays=(0.0, 0.0),
        step_timeout=5.0,
        sleep=no_sleep,
        max_grounded_calls=max_grounded_calls,
    )
    recorder = Recorder()
    await executor.run(job, recorder, asyncio.Event())
    return recorder.outcomes, llm


def status(outcome: StepOutcome) -> str:
    assert outcome.output is not None and outcome.output.grounding is not None
    return outcome.output.grounding.status


async def test_parallel_research_steps_share_one_budget() -> None:
    answers = [fake_response(f"- finding {n}", sources=[PAGE], search_queries=["q"]) for n in "123"]

    outcomes, llm = await run(answers, one_group(spec("s1"), spec("s2"), spec("s3")), 2)

    assert len(llm.calls) == 3 and all(call.use_search for call in llm.calls)
    statuses = sorted(status(o) for o in outcomes.values())
    assert statuses == ["grounded", "grounded", "unverified"]
    assert sum(o.search.grounded_calls for o in outcomes.values()) == 2
    unverified = next(o for o in outcomes.values() if status(o) == "unverified")
    assert unverified.output is not None and unverified.output.grounding is not None
    assert "grounded-call budget is used up (2 of 2" in (unverified.output.grounding.reason or "")
    assert unverified.output.sources == ()
    assert unverified.search == SearchUsage()  # it never got a slot


async def test_every_retry_takes_a_slot_and_the_last_one_runs_unverified() -> None:
    script: list[ScriptItem] = [
        LLMTransientError("429"),
        LLMTransientError("429"),
        fake_response("- finally", sources=[PAGE]),
    ]

    outcomes, llm = await run(script, one_group(spec("s1")), 2)

    # Every attempt asked for search; the budget alone decided which ones searched.
    assert [call.use_search for call in llm.calls] == [True, True, True]
    outcome = outcomes["s1"]
    assert (outcome.status, outcome.attempts) == ("completed", 3)
    assert outcome.search.grounded_calls == 2  # the two failed attempts used both slots
    assert status(outcome) == "unverified"
    assert outcome.output is not None and outcome.output.content.startswith("Unverified:")


async def test_queries_are_added_up_over_a_steps_attempts() -> None:
    script: list[ScriptItem] = [
        fake_response("", search_queries=["first query"]),  # empty answer: retried
        fake_response("- found", sources=[PAGE], search_queries=["second", "third"]),
    ]

    outcomes, _ = await run(script, one_group(spec("s1")), 5)

    assert outcomes["s1"].search == SearchUsage(grounded_calls=2, queries=3)
    assert status(outcomes["s1"]) == "grounded"


async def test_the_other_agents_never_search_or_touch_the_budget() -> None:
    script = [
        fake_response("A short draft."),
        fake_response("## Analysis\n- point (per the inputs)"),
        fake_response("```python\nprint('hi')\n```"),
        fake_response("- summary"),
    ]
    job = one_group(
        spec("w", "writing", "write"),
        spec("a", "analysis", "analyze"),
        spec("c", "code", "write_code"),
        spec("r", "research", "summarize"),
    )

    outcomes, llm = await run(script, job, 1)

    assert [call.use_search for call in llm.calls] == [False, False, False, False]
    assert all(o.status == "completed" for o in outcomes.values())
    assert all(o.search == SearchUsage() for o in outcomes.values())
    for outcome in outcomes.values():
        assert outcome.output is not None
        assert "grounding" not in outcome.output.model_dump(mode="json")


async def test_without_a_configured_limit_the_executor_does_not_cap_grounding() -> None:
    answers = [fake_response(f"- {n}", sources=[PAGE]) for n in "1234"]

    outcomes, _ = await run(answers, one_group(*(spec(f"s{n}") for n in "1234")), None)

    assert [status(o) for o in outcomes.values()] == ["grounded"] * 4
    assert sum(o.search.grounded_calls for o in outcomes.values()) == 4


async def test_a_budget_of_zero_means_no_grounded_call_at_all() -> None:
    outcomes, _ = await run([fake_response("- x", sources=[PAGE])], one_group(spec("s1")), 0)

    assert status(outcomes["s1"]) == "unverified"
    assert outcomes["s1"].search.grounded_calls == 0
