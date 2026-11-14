"""The writing, analysis and code agents, each on its own, with a scripted LLM."""

from collections.abc import Mapping
from typing import ClassVar

import pytest

from orchestrator.agents import (
    AgentResult,
    AgentTask,
    AnalysisAgent,
    BaseAgent,
    CodeAgent,
    ContextItem,
    ResearchAgent,
    WritingAgent,
)
from orchestrator.agents.prompted import SHARED_RULES, PromptedAgent
from orchestrator.llm import LLMPermanentError, LLMTransientError, Source
from orchestrator.llm.fake import FakeLLM, fake_response

SPECIALIZED: list[type[PromptedAgent]] = [WritingAgent, AnalysisAgent, CodeAgent]
ACTIONS = [(agent, action) for agent in SPECIALIZED for action in sorted(agent.capabilities)]
CODE_BLOCK = "```python\nprint('hi')\n```"


def task(action: str, **overrides: object) -> AgentTask:
    fields: dict[str, object] = {
        "step_id": "step_2",
        "action": action,
        "instruction": "Do the step",
        "goal": "Compare Python and JavaScript for beginners",
        "constraints": {"tone": "friendly", "max_words": 300},
        "context": (
            ContextItem(step_id="step_1", agent="research", content="- Python is readable"),
        ),
    }
    return AgentTask.model_validate(fields | overrides)


def answer_for(agent: type[PromptedAgent], action: str) -> str:
    return f"{CODE_BLOCK}\nNotes." if action in agent.output_checks else "A useful answer."


# --- every action of every new agent -----------------------------------------------------


@pytest.mark.parametrize(("agent", "action"), ACTIONS, ids=lambda x: getattr(x, "name", x))
async def test_each_action_sends_its_own_prompt_and_the_steps_inputs(
    agent: type[PromptedAgent], action: str
) -> None:
    llm = FakeLLM([fake_response(answer_for(agent, action))])

    result = await agent(llm).run(task(action))

    assert result.status == "completed", result.error
    [request] = llm.calls
    assert request.system_prompt == agent.system_prompts[action].rstrip() + "\n" + SHARED_RULES
    assert f"You are the {agent.name} agent" in request.system_prompt
    assert request.use_search is False  # only research searches
    assert (request.temperature, request.max_output_tokens) == (
        agent.temperature,
        agent.max_output_tokens,
    )
    prompt = request.user_prompt
    assert "Compare Python and JavaScript for beginners" in prompt  # the goal
    assert '"tone": "friendly"' in prompt  # the constraints
    assert "## Input from step_1 (research agent)\n- Python is readable" in prompt  # its dependency
    assert f"## Your step ({action})\nDo the step" in prompt


@pytest.mark.parametrize(("agent", "action"), ACTIONS, ids=lambda x: getattr(x, "name", x))
async def test_no_new_agent_reports_sources_or_confidence_it_does_not_have(
    agent: type[PromptedAgent], action: str
) -> None:
    # Even if a provider attached sources to a non-search call, they would not be this
    # step's evidence: only research steps carry sources.
    stray = Source(title="x", url="https://stray.example")
    llm = FakeLLM([fake_response(answer_for(agent, action), sources=[stray])])

    result = await agent(llm).run(task(action))

    assert result.output is not None
    assert (result.output.sources, result.output.confidence) == ((), None)


def test_the_temperatures_follow_the_work() -> None:
    assert CodeAgent.temperature < AnalysisAgent.temperature < WritingAgent.temperature


# --- one AgentResult shape for all four agents -------------------------------------------


async def test_every_agent_returns_the_same_result_shape() -> None:
    results: list[AgentResult] = []
    for agent, action in [
        (ResearchAgent, "research"),
        (WritingAgent, "write"),
        (AnalysisAgent, "compare"),
        (CodeAgent, "write_code"),
    ]:
        llm = FakeLLM([fake_response(f"{CODE_BLOCK}\nanswer", input_tokens=30, output_tokens=10)])
        results.append(await agent(llm).run(task(action)))

    shapes = {tuple(r.model_dump()) for r in results}
    assert shapes == {
        (
            "step_id",
            "agent",
            "status",
            "output",
            "error",
            "usage",
            "execution_time_ms",
            "tokens_used",
        )
    }
    assert [r.agent for r in results] == ["research", "writing", "analysis", "code"]
    assert all(r.status == "completed" and r.tokens_used == 40 for r in results)
    assert all(r.execution_time_ms >= 0 for r in results)


# --- failures ----------------------------------------------------------------------------


@pytest.mark.parametrize("agent", SPECIALIZED, ids=lambda a: a.name)
async def test_an_empty_answer_is_a_retryable_failure(agent: type[PromptedAgent]) -> None:
    action = sorted(agent.capabilities)[0]

    result = await agent(FakeLLM([fake_response("   ")])).run(task(action))

    assert result.status == "failed" and result.error is not None
    assert (result.error.type, result.error.retryable) == ("LLMTransientError", True)


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [(LLMTransientError("429"), True), (LLMPermanentError("blocked"), False)],
)
async def test_llm_errors_keep_their_retryability(failure: Exception, retryable: bool) -> None:
    result = await WritingAgent(FakeLLM([failure])).run(task("write"))

    assert result.error is not None and result.error.retryable is retryable


async def test_an_action_of_another_agent_is_refused_without_a_model_call() -> None:
    llm = FakeLLM()

    result = await WritingAgent(llm).run(task("compare"))  # an analysis action

    assert result.error is not None
    assert result.error.type == "UnsupportedActionError"
    assert "supported: ['draft_email', 'rewrite', 'write']" in result.error.message
    assert llm.calls == []


# --- the code agent ----------------------------------------------------------------------


@pytest.mark.parametrize("action", ["write_code", "fix_bug"])
async def test_code_that_must_return_code_fails_retryably_without_a_code_block(action: str) -> None:
    result = await CodeAgent(FakeLLM([fake_response("You could use a loop here.")])).run(
        task(action)
    )

    assert result.status == "failed" and result.error is not None
    assert (result.error.type, result.error.retryable) == ("InvalidAgentOutputError", True)
    assert result.error.message == (
        f"code.{action}: the answer has no fenced code block; code must be inside ``` fences"
    )


async def test_explaining_or_reviewing_code_needs_no_code_block() -> None:
    for action in ("explain_code", "review_code"):
        result = await CodeAgent(FakeLLM([fake_response("It loops over the list.")])).run(
            task(action)
        )
        assert result.status == "completed"


def test_the_code_agent_is_told_it_runs_nothing() -> None:
    for action in CodeAgent.capabilities:
        prompt = CodeAgent(FakeLLM()).system_prompt(action)
        assert "You cannot run code, read or write files" in prompt
        assert "Never claim" in prompt


def test_the_analysis_agent_is_told_to_invent_nothing() -> None:
    for action in AnalysisAgent.capabilities:
        prompt = AnalysisAgent(FakeLLM()).system_prompt(action)
        assert "Never invent numbers, data points, studies or sources" in prompt
        assert "insufficient information" in prompt


# --- contracts are checked when a class is defined ---------------------------------------


def test_action_descriptions_must_match_capabilities() -> None:
    with pytest.raises(TypeError, match="do not match capabilities"):

        class Drifted(BaseAgent):
            name: ClassVar[str] = "drifted"
            description: ClassVar[str] = "d"
            capabilities: ClassVar[frozenset[str]] = frozenset({"a", "b"})
            action_descriptions: ClassVar[Mapping[str, str]] = {"a": "only a"}

            async def execute(self, task: AgentTask, llm: object) -> object:  # type: ignore[override]
                raise NotImplementedError


def test_a_prompted_agent_needs_a_prompt_for_every_action() -> None:
    with pytest.raises(TypeError, match="system_prompts must cover exactly its capabilities"):

        class Unprompted(PromptedAgent):
            name: ClassVar[str] = "unprompted"
            description: ClassVar[str] = "u"
            capabilities: ClassVar[frozenset[str]] = frozenset({"a", "b"})
            system_prompts: ClassVar[Mapping[str, str]] = {"a": "prompt a"}


@pytest.mark.parametrize("agent", [ResearchAgent, *SPECIALIZED], ids=lambda a: a.name)
def test_every_implemented_agent_describes_every_action(agent: type[BaseAgent]) -> None:
    assert set(agent.action_descriptions) == set(agent.capabilities)
    assert all(len(text) > 20 for text in agent.action_descriptions.values())
