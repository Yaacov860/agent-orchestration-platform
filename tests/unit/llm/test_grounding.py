"""The grounded-call budget and ledger on their own (llm/grounding.py)."""

import asyncio

import pytest

from orchestrator.llm import GroundingBudget, LLMRequest, SearchUsage, may_ground

SEARCH = LLMRequest(system_prompt="s", user_prompt="u", use_search=True)


def test_the_budget_grants_exactly_max_calls_and_then_refuses() -> None:
    budget = GroundingBudget(2)
    step = budget.for_step("step_1")

    assert [step.try_reserve() for _ in range(4)] == [True, True, False, False]
    assert budget.used == 2
    assert step.usage == SearchUsage(grounded_calls=2)


def test_steps_share_one_budget_and_keep_their_own_ledger() -> None:
    budget = GroundingBudget(3)
    a, b = budget.for_step("step_1"), budget.for_step("step_2")

    assert (a.try_reserve(), b.try_reserve(), a.try_reserve(), b.try_reserve()) == (
        True,
        True,
        True,
        False,
    )
    assert (budget.usage_of("step_1").grounded_calls, budget.usage_of("step_2").grounded_calls) == (
        2,
        1,
    )


@pytest.mark.parametrize(("max_calls", "granted"), [(None, 10), (0, 0), (1, 1)])
def test_none_means_no_limit_and_zero_means_no_grounded_call(
    max_calls: int | None, granted: int
) -> None:
    step = GroundingBudget(max_calls).for_step("s")

    assert sum(step.try_reserve() for _ in range(10)) == granted


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_calls"):
        GroundingBudget(-1)


async def test_concurrent_reservations_never_exceed_the_budget() -> None:
    budget = GroundingBudget(3)

    async def reserve(step_id: str) -> bool:
        await asyncio.sleep(0)  # interleave the coroutines before they reserve
        return budget.for_step(step_id).try_reserve()

    granted = await asyncio.gather(*(reserve(f"step_{n}") for n in range(10)))

    assert sum(granted) == 3
    assert budget.used == 3


def test_only_non_empty_queries_are_counted() -> None:
    step = GroundingBudget(1).for_step("s")

    step.record_queries(["python gil", "", "   ", "pep 703"])

    assert step.usage == SearchUsage(queries=2)


def test_may_ground_grants_a_search_request_and_takes_its_slot() -> None:
    step = GroundingBudget(1).for_step("s")

    report = may_ground(
        SEARCH.model_copy(update={"grounding": step}), enabled=True, disabled_reason="off"
    )

    assert (report.used, report.reason) == (True, None)
    assert step.usage.grounded_calls == 1


def test_may_ground_refuses_when_the_budget_is_used_up_and_says_so() -> None:
    step = GroundingBudget(0).for_step("s")

    report = may_ground(
        SEARCH.model_copy(update={"grounding": step}), enabled=True, disabled_reason="off"
    )

    assert report.used is False
    assert report.reason is not None and "grounded-call budget is used up (0 of 0" in report.reason
    assert "GEMINI_MAX_GROUNDED_CALLS_PER_TASK" in report.reason


def test_may_ground_takes_no_slot_when_grounding_is_off_or_search_not_requested() -> None:
    step = GroundingBudget(5).for_step("s")
    with_budget = SEARCH.model_copy(update={"grounding": step})

    off = may_ground(with_budget, enabled=False, disabled_reason="grounding is off")
    plain = may_ground(
        with_budget.model_copy(update={"use_search": False}), enabled=True, disabled_reason="x"
    )

    assert (off.used, off.reason) == (False, "grounding is off")
    assert plain.used is False
    assert step.usage.grounded_calls == 0


def test_the_budget_handle_never_appears_in_a_dumped_request() -> None:
    request = SEARCH.model_copy(update={"grounding": GroundingBudget(1).for_step("s")})

    assert "grounding" not in request.model_dump()
    assert "grounding" not in repr(request)
