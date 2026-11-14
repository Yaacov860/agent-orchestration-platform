import asyncio

import pytest
from factories import plan_json, plan_step

from orchestrator.agents.catalog import AgentCatalog, AgentSpec, default_catalog
from orchestrator.llm import LLMPermanentError, LLMTransientError
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.planning import MAX_STEPS, Planner, PlanningError, PlanResult
from orchestrator.planning.planner import PREVIOUS_ANSWER_LIMIT

GOAL = "Write a comparison blog post about Python vs JavaScript for beginners"
VALID = plan_json(
    plan_step("step_1"),
    plan_step("step_2"),
    plan_step("step_3", "step_1", "step_2", action="summarize"),
)
CYCLIC = plan_json(plan_step("step_1", "step_2"), plan_step("step_2", "step_1"))


async def plan_with(llm: FakeLLM, catalog: AgentCatalog | None = None) -> PlanResult:
    return await Planner(llm, catalog or default_catalog()).plan(
        goal=GOAL, constraints={"tone": "friendly", "max_words": 1500}, output_format="markdown"
    )


async def planning_error(llm: FakeLLM) -> PlanningError:
    with pytest.raises(PlanningError) as caught:
        await plan_with(llm)
    return caught.value


# --- first attempt succeeds --------------------------------------------------------------


async def test_a_valid_answer_becomes_a_validated_plan_in_one_attempt() -> None:
    llm = FakeLLM([fake_response(VALID, input_tokens=400, output_tokens=120)])

    result = await plan_with(llm)

    assert result.attempts == 1
    assert result.validated.parallel_groups == (("step_1", "step_2"), ("step_3",))
    assert (result.usage.input_tokens, result.usage.output_tokens) == (400, 120)
    assert len(llm.calls) == 1


async def test_the_prompt_lists_the_catalogs_agents_and_the_rules() -> None:
    llm = FakeLLM([fake_response(VALID)])

    await plan_with(llm)

    [request] = llm.calls
    assert default_catalog().describe() in request.system_prompt
    assert f"At most {MAX_STEPS} steps" in request.system_prompt
    assert request.use_search is False  # planning needs no web search


async def test_the_prompt_carries_the_goal_constraints_and_output_format() -> None:
    llm = FakeLLM([fake_response(VALID)])

    await plan_with(llm)

    [request] = llm.calls
    assert f"<task>\n{GOAL}\n</task>" in request.user_prompt
    assert '{"tone": "friendly", "max_words": 1500}' in request.user_prompt
    assert "## Final output format\nmarkdown" in request.user_prompt


async def test_the_prompt_is_built_from_the_injected_catalog() -> None:
    catalog = AgentCatalog([AgentSpec("writing", "Writes text.", frozenset({"write"}))])
    llm = FakeLLM([fake_response(plan_json(plan_step("step_1", agent="writing", action="write")))])

    result = await plan_with(llm, catalog)

    assert (
        "- writing: Writes text. Actions: write." in llm.calls[0].system_prompt
    )  # no actions given
    assert "research" not in llm.calls[0].system_prompt.split("## Choosing agents")[0]
    assert result.validated.plan.steps[0].agent == "writing"


@pytest.mark.parametrize(
    "wrapped",
    [f"```json\n{VALID}\n```", f"```\n{VALID}\n```", f"\n\n  {VALID}  \n"],
)
async def test_json_wrapped_in_a_code_fence_or_whitespace_is_accepted(wrapped: str) -> None:
    result = await plan_with(FakeLLM([fake_response(wrapped)]))

    assert result.attempts == 1


# --- one repair attempt ------------------------------------------------------------------


async def test_an_invalid_plan_is_repaired_by_showing_the_model_its_problems() -> None:
    llm = FakeLLM(
        [
            fake_response(CYCLIC, input_tokens=400, output_tokens=100),
            fake_response(VALID, input_tokens=600, output_tokens=120),
        ]
    )

    result = await plan_with(llm)

    assert result.attempts == 2
    assert result.validated.parallel_groups == (("step_1", "step_2"), ("step_3",))
    # Tokens of the rejected attempt count too: they were paid for.
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1000, 220)

    repair = llm.calls[1]
    assert repair.system_prompt == llm.calls[0].system_prompt
    assert f"<task>\n{GOAL}\n</task>" in repair.user_prompt  # the task is not forgotten
    assert f"## Your previous answer (rejected)\n{CYCLIC}" in repair.user_prompt
    assert "- dependency cycle (each step depends on the next): step_1 -> step_2 -> step_1" in (
        repair.user_prompt
    )


async def test_a_plan_using_an_unimplemented_agent_is_sent_back_for_repair() -> None:
    writing = plan_json(
        plan_step("step_1"), plan_step("step_2", "step_1", agent="translation", action="translate")
    )
    llm = FakeLLM([fake_response(writing), fake_response(VALID)])

    result = await plan_with(llm)

    assert result.attempts == 2
    assert (
        "- step_2: unknown agent 'translation'; available agents: analysis, code, research, writing"
        in llm.calls[1].user_prompt
    )


async def test_an_answer_that_is_not_json_is_sent_back_for_repair() -> None:
    llm = FakeLLM([fake_response("Sure! Step 1: research Python."), fake_response(VALID)])

    result = await plan_with(llm)

    assert result.attempts == 2
    assert "- Invalid JSON" in llm.calls[1].user_prompt


async def test_a_huge_rejected_answer_is_truncated_in_the_repair_prompt() -> None:
    huge = "x" * (PREVIOUS_ANSWER_LIMIT * 3)
    llm = FakeLLM([fake_response(huge), fake_response(VALID)])

    await plan_with(llm)

    assert "x" * PREVIOUS_ANSWER_LIMIT in llm.calls[1].user_prompt
    assert "x" * (PREVIOUS_ANSWER_LIMIT + 1) not in llm.calls[1].user_prompt


async def test_planning_gives_up_after_one_failed_repair() -> None:
    llm = FakeLLM(
        [
            fake_response(CYCLIC, input_tokens=10, output_tokens=1),
            fake_response(
                plan_json(plan_step("step_1", "step_5")), input_tokens=20, output_tokens=2
            ),
        ]
    )

    error = await planning_error(llm)

    assert len(llm.calls) == 2  # exactly one repair; FakeLLM would fail loudly on a third call
    assert error.attempts == 2
    assert error.retryable is False  # the same request would most likely fail the same way
    assert error.problems == ("step_1: depends on 'step_5', which is not a step in this plan",)
    assert str(error) == (
        "no valid plan after 2 attempts: "
        "step_1: depends on 'step_5', which is not a step in this plan"
    )
    assert error.usage.total == 33


# --- LLM failures ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [(LLMTransientError("rate limited"), True), (LLMPermanentError("invalid API key"), False)],
)
async def test_an_llm_error_fails_planning_and_keeps_its_retryability(
    failure: Exception, retryable: bool
) -> None:
    error = await planning_error(FakeLLM([failure]))

    assert error.retryable is retryable
    assert error.attempts == 1
    assert isinstance(error.__cause__, type(failure))
    assert str(error) == f"LLM call failed during planning: {failure}"


async def test_an_llm_error_during_the_repair_keeps_the_tokens_already_spent() -> None:
    llm = FakeLLM(
        [fake_response(CYCLIC, input_tokens=40, output_tokens=10), LLMTransientError("timeout")]
    )

    error = await planning_error(llm)

    assert (error.attempts, error.retryable, error.usage.total) == (2, True, 50)


async def test_cancellation_propagates_instead_of_becoming_a_planning_failure() -> None:
    task = asyncio.create_task(plan_with(FakeLLM([asyncio.CancelledError()])))

    with pytest.raises(asyncio.CancelledError):
        await task


# --- concurrency -------------------------------------------------------------------------


async def test_concurrent_plans_on_one_planner_keep_separate_token_counts() -> None:
    llm = FakeLLM(
        [
            fake_response(VALID, input_tokens=1, output_tokens=0),
            fake_response(VALID, input_tokens=1000, output_tokens=0),
        ]
    )
    planner = Planner(llm, default_catalog())

    results = await asyncio.gather(
        *(planner.plan(goal=GOAL, constraints={}, output_format="text") for _ in range(2))
    )

    assert sorted(r.usage.total for r in results) == [1, 1000]
