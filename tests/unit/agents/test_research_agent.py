import asyncio
from typing import ClassVar

import pytest

from orchestrator.agents import AgentOutput, AgentTask, BaseAgent, ContextItem, ResearchAgent
from orchestrator.llm import LLMClient, LLMPermanentError, LLMRequest, LLMTransientError, Source
from orchestrator.llm.fake import FakeLLM, fake_response

PYTHON_ORG = Source(title="Python", url="https://python.org")
REAL_PYTHON = Source(title="Real Python", url="https://realpython.com")


def make_task(action: str = "research", **overrides: object) -> AgentTask:
    fields: dict[str, object] = {
        "step_id": "step_1",
        "action": action,
        "instruction": "Python advantages for beginners",
        "goal": "Write a comparison blog post about Python vs JavaScript for beginners",
        "constraints": {"tone": "friendly"},
    }
    return AgentTask.model_validate(fields | overrides)


# --- happy path -------------------------------------------------------------------------


async def test_research_returns_grounded_findings_with_sources_tokens_and_timing() -> None:
    llm = FakeLLM(
        [
            fake_response(
                "- Readable syntax",
                sources=[PYTHON_ORG, REAL_PYTHON],
                input_tokens=300,
                output_tokens=150,
            )
        ]
    )

    result = await ResearchAgent(llm).run(make_task())

    assert result.status == "completed"
    assert result.step_id == "step_1"
    assert result.agent == "research"
    assert result.output is not None
    assert result.output.content == "- Readable syntax"
    assert result.output.sources == (PYTHON_ORG, REAL_PYTHON)
    assert result.tokens_used == 450
    assert result.execution_time_ms >= 0
    assert result.error is None


async def test_research_asks_for_web_search_and_sends_goal_constraints_and_step() -> None:
    llm = FakeLLM([fake_response("findings")])

    await ResearchAgent(llm).run(make_task())

    [request] = llm.calls
    assert request.use_search is True
    assert "Python vs JavaScript" in request.user_prompt  # the overall goal
    assert '"tone": "friendly"' in request.user_prompt  # the constraints
    assert "Python advantages for beginners" in request.user_prompt  # the step itself


async def test_upstream_step_outputs_are_passed_into_the_prompt() -> None:
    llm = FakeLLM([fake_response("summary")])
    task = make_task(
        "summarize",
        context=[ContextItem(step_id="step_0", agent="research", content="JS runs in browsers")],
    )

    await ResearchAgent(llm).run(task)

    assert "JS runs in browsers" in llm.calls[0].user_prompt
    assert "step_0" in llm.calls[0].user_prompt


async def test_summarize_does_not_search_and_reports_no_confidence() -> None:
    llm = FakeLLM([fake_response("summary")])

    result = await ResearchAgent(llm).run(make_task("summarize"))

    assert llm.calls[0].use_search is False
    assert result.output is not None
    assert result.output.confidence is None


# --- sources and confidence -------------------------------------------------------------


async def test_duplicate_sources_are_collapsed_keeping_first_seen_order() -> None:
    llm = FakeLLM([fake_response("x", sources=[REAL_PYTHON, PYTHON_ORG, REAL_PYTHON])])

    result = await ResearchAgent(llm).run(make_task())

    assert result.output is not None
    assert result.output.sources == (REAL_PYTHON, PYTHON_ORG)


@pytest.mark.parametrize(
    ("sources", "expected"),
    [([], 0.3), ([PYTHON_ORG], 0.6), ([PYTHON_ORG, REAL_PYTHON], 0.85)],
)
async def test_confidence_reflects_how_much_evidence_backs_the_answer(
    sources: list[Source], expected: float
) -> None:
    llm = FakeLLM([fake_response("x", sources=sources)])

    result = await ResearchAgent(llm).run(make_task())

    assert result.output is not None
    assert result.output.confidence == expected


# --- failures ---------------------------------------------------------------------------


async def test_unsupported_action_fails_without_calling_the_llm() -> None:
    llm = FakeLLM()

    result = await ResearchAgent(llm).run(make_task("generate_code"))

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.type == "UnsupportedActionError"
    assert result.error.retryable is False
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "retryable"),
    [(LLMTransientError("429 rate limited"), True), (LLMPermanentError("blocked"), False)],
)
async def test_llm_errors_become_a_failed_result_that_says_whether_to_retry(
    error: Exception, retryable: bool
) -> None:
    result = await ResearchAgent(FakeLLM([error])).run(make_task())

    assert result.status == "failed"
    assert result.output is None
    assert result.error is not None
    assert result.error.retryable is retryable


async def test_an_empty_answer_is_a_retryable_failure() -> None:
    result = await ResearchAgent(FakeLLM([fake_response("   ")])).run(make_task())

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.retryable is True


async def test_an_unexpected_bug_is_recorded_as_a_non_retryable_failure() -> None:
    result = await ResearchAgent(FakeLLM([ValueError("boom")])).run(make_task())

    assert result.status == "failed"
    assert result.error is not None
    assert (result.error.type, result.error.retryable) == ("ValueError", False)


async def test_cancellation_is_not_swallowed_as_a_step_failure() -> None:
    result_task = asyncio.create_task(
        ResearchAgent(FakeLLM([asyncio.CancelledError()])).run(make_task())
    )

    with pytest.raises(asyncio.CancelledError):
        await result_task


# --- token metering (BaseAgent) ---------------------------------------------------------


class DraftThenEditAgent(BaseAgent):
    """Test double: an agent that makes two LLM calls in one run."""

    name: ClassVar[str] = "two_calls"
    description: ClassVar[str] = "test"
    capabilities: ClassVar[frozenset[str]] = frozenset({"go"})

    async def execute(self, task: AgentTask, llm: LLMClient) -> AgentOutput:
        await llm.generate(LLMRequest(system_prompt="s", user_prompt="draft"))
        final = await llm.generate(LLMRequest(system_prompt="s", user_prompt="edit"))
        return AgentOutput(content=final.text)


async def test_tokens_from_every_llm_call_in_a_run_are_summed() -> None:
    llm = FakeLLM(
        [
            fake_response("d", input_tokens=10, output_tokens=5),
            fake_response("e", input_tokens=20, output_tokens=7),
        ]
    )

    result = await DraftThenEditAgent(llm).run(make_task("go"))

    assert (result.usage.input_tokens, result.usage.output_tokens) == (30, 12)


async def test_tokens_already_spent_are_kept_when_a_later_call_fails() -> None:
    llm = FakeLLM(
        [fake_response("d", input_tokens=10, output_tokens=5), LLMTransientError("timeout")]
    )

    result = await DraftThenEditAgent(llm).run(make_task("go"))

    assert result.status == "failed"
    assert result.tokens_used == 15


async def test_concurrent_runs_on_one_agent_instance_keep_separate_token_counts() -> None:
    llm = FakeLLM(
        [
            fake_response("a", input_tokens=1, output_tokens=0),
            fake_response("b", input_tokens=1000, output_tokens=0),
        ]
    )
    agent = ResearchAgent(llm)

    first, second = await asyncio.gather(
        agent.run(make_task(step_id="step_1")), agent.run(make_task(step_id="step_2"))
    )

    assert sorted([first.tokens_used, second.tokens_used]) == [1, 1000]
