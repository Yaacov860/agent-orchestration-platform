"""The executor running steps on all four agents: routing, data flow, retries, timeouts."""

import asyncio
from collections.abc import Sequence

from fakes import ByCallerLLM, GatedLLM, caller_of

from orchestrator.agents.factory import AgentFactory
from orchestrator.execution import (
    DependencyExecutor,
    ExecutionJob,
    ExecutionSummary,
    StepOutcome,
    StepSpec,
)
from orchestrator.llm import LLMClient, LLMRequest, LLMResponse, LLMTransientError, Source
from orchestrator.llm.fake import fake_response
from orchestrator.planning.graph import parallel_groups

CODE = "```python\ndef pick(a, b):\n    return a\n```\nNotes: untested."


def step(step_id: str, agent: str, action: str, *dependencies: str) -> StepSpec:
    return StepSpec(
        step_id=step_id,
        agent=agent,
        action=action,
        instruction=f"do {step_id}",
        dependencies=dependencies,
    )


def job(*steps: StepSpec) -> ExecutionJob:
    groups = parallel_groups({s.step_id: s.dependencies for s in steps})
    return ExecutionJob(
        task_id="task_1",
        goal="Pick a first language and give a starter snippet",
        steps=steps,
        parallel_groups=tuple(tuple(g) for g in groups),
    )


class Recorder:
    def __init__(self) -> None:
        self.outcomes: dict[str, StepOutcome] = {}

    async def step_started(self, step_id: str, attempt: int) -> None: ...

    async def step_finished(self, outcome: StepOutcome) -> None:
        self.outcomes[outcome.step_id] = outcome

    async def steps_skipped(self, blocked_by: object) -> None: ...

    async def steps_expired(self, step_ids: Sequence[str]) -> None: ...

    async def steps_cancelled(self, step_ids: Sequence[str]) -> None: ...

    async def cancel_requested(self) -> bool:
        return False


async def no_sleep(seconds: float) -> None:
    return None


async def run(
    llm: LLMClient, the_job: ExecutionJob, *, step_timeout: float = 5.0
) -> ExecutionSummary:
    executor = DependencyExecutor(
        AgentFactory(llm), slots=asyncio.Semaphore(4), step_timeout=step_timeout, sleep=no_sleep
    )
    return await executor.run(the_job, Recorder(), asyncio.Event())


CHAIN = job(
    step("step_1", "research", "research"),
    step("step_2", "analysis", "compare", "step_1"),
    step("step_3", "writing", "write", "step_2"),
    step("step_4", "code", "write_code", "step_2"),
)


async def test_each_step_runs_on_its_own_agent_and_gets_only_its_dependencies_outputs() -> None:
    llm = ByCallerLLM(
        {
            "research": "- Python reads like English",
            "analysis": "| | Python | JS |\nVerdict: Python",
            "writing": "# Start with Python",
            "code": CODE,
        }
    )

    summary = await run(llm, CHAIN)

    assert summary.succeeded
    assert {o.step_id: o.agent for o in summary.outcomes} == {
        "step_1": "research",
        "step_2": "analysis",
        "step_3": "writing",
        "step_4": "code",
    }
    [analysis] = llm.requests("analysis")
    assert (
        "## Input from step_1 (research agent)\n- Python reads like English" in analysis.user_prompt
    )
    [writing] = llm.requests("writing")
    assert "## Input from step_2 (analysis agent)\n| | Python | JS |" in writing.user_prompt
    assert "Input from step_1" not in writing.user_prompt  # it declared only step_2
    [code] = llm.requests("code")
    assert "## Input from step_2 (analysis agent)" in code.user_prompt
    assert [request.use_search for _, request in llm.calls] == [True, False, False, False]


async def test_only_research_outputs_carry_sources() -> None:
    page = Source(title="Python docs", url="https://docs.python.org")
    llm = ByCallerLLM(
        {
            "research": fake_response("- facts", sources=[page]),
            "analysis": fake_response("verdict", sources=[page]),  # would be a stray source
            "writing": "text",
            "code": CODE,
        }
    )

    summary = await run(llm, CHAIN)

    sources = {o.step_id: (o.output.sources if o.output else None) for o in summary.outcomes}
    assert sources == {"step_1": (page,), "step_2": (), "step_3": (), "step_4": ()}


async def test_a_transient_failure_of_a_writing_step_is_retried() -> None:
    llm = ByCallerLLM(
        {
            "research": "facts",
            "analysis": "verdict",
            "writing": [LLMTransientError("429"), "# Text"],
            "code": CODE,
        }
    )

    summary = await run(llm, CHAIN)

    writing = next(o for o in summary.outcomes if o.agent == "writing")
    assert (writing.status, writing.attempts) == ("completed", 2)


async def test_code_without_a_code_block_is_retried_as_an_invalid_output() -> None:
    llm = ByCallerLLM(
        {
            "research": "facts",
            "analysis": "verdict",
            "writing": "text",
            "code": ["Use a loop.", CODE],
        }
    )

    summary = await run(llm, CHAIN)

    code = next(o for o in summary.outcomes if o.agent == "code")
    assert (code.status, code.attempts) == ("completed", 2)


async def test_code_that_never_returns_code_fails_and_its_dependents_are_skipped() -> None:
    the_job = job(
        step("step_1", "analysis", "analyze"),
        step("step_2", "code", "fix_bug", "step_1"),
        step("step_3", "code", "review_code", "step_2"),
    )
    llm = ByCallerLLM({"analysis": "requirements", "code": "I would change line 3."})

    summary = await run(llm, the_job)

    [failed] = summary.failed
    assert (failed.step_id, failed.attempts) == ("step_2", 3)
    assert failed.error is not None and failed.error.type == "InvalidAgentOutputError"
    assert summary.skipped == {"step_3": "step_2"}


async def test_a_hung_analysis_step_times_out_while_the_others_finish() -> None:
    gate = GatedLLM()

    class HangOnAnalysis:
        async def generate(self, request: LLMRequest) -> LLMResponse:
            if caller_of(request) == "analysis":
                return await gate.generate(request)  # never released
            return fake_response(CODE if caller_of(request) == "code" else "text")

    the_job = job(
        step("step_1", "research", "research"),
        step("step_2", "analysis", "evaluate"),
        step("step_3", "writing", "write", "step_1"),
    )
    summary = await run(HangOnAnalysis(), the_job, step_timeout=0.05)

    by_id = {o.step_id: o for o in summary.outcomes}
    assert by_id["step_2"].error is not None
    assert (by_id["step_2"].error.type, by_id["step_2"].attempts) == ("StepTimeoutError", 3)
    assert (by_id["step_1"].status, by_id["step_3"].status) == ("completed", "completed")
    assert gate.in_flight == 0
