import json

import pytest
from factories import plan_json, plan_step
from pydantic import ValidationError

from orchestrator.agents.catalog import AgentCatalog, AgentSpec, default_catalog
from orchestrator.planning import (
    MAX_STEPS,
    Plan,
    PlanValidationError,
    ValidatedPlan,
    parse_plan,
    validate_plan,
)

CATALOG = default_catalog()  # research: research, summarize


def check(*steps: dict[str, object], catalog: AgentCatalog = CATALOG) -> ValidatedPlan:
    return validate_plan(parse_plan(plan_json(*steps)), catalog)


def problems_of(*steps: dict[str, object], catalog: AgentCatalog = CATALOG) -> tuple[str, ...]:
    with pytest.raises(PlanValidationError) as caught:
        check(*steps, catalog=catalog)
    return caught.value.problems


def parse_problems(raw: str) -> tuple[str, ...]:
    with pytest.raises(PlanValidationError) as caught:
        parse_plan(raw)
    return caught.value.problems


# --- a valid plan ------------------------------------------------------------------------


def test_a_valid_plan_comes_back_with_its_parallel_groups() -> None:
    validated = check(
        plan_step("step_1"),
        plan_step("step_2"),
        plan_step("step_3", "step_1", "step_2", action="summarize"),
    )

    assert validated.parallel_groups == (("step_1", "step_2"), ("step_3",))
    assert [s.step_id for s in validated.plan.steps] == ["step_1", "step_2", "step_3"]
    assert validated.plan.reasoning == "test plan"


def test_a_dependency_listed_twice_is_normalized_not_rejected() -> None:
    validated = check(plan_step("step_1"), plan_step("step_2", "step_1", "step_1"))

    assert validated.plan.steps[1].dependencies == ("step_1",)


def test_text_fields_are_trimmed() -> None:
    validated = check(plan_step("step_1", input="  find facts  "))

    assert validated.plan.steps[0].input == "find facts"


def test_reasoning_is_optional() -> None:
    raw = json.dumps({"steps": [plan_step("step_1")]})

    assert validate_plan(parse_plan(raw), CATALOG).plan.reasoning == ""


# --- limits ------------------------------------------------------------------------------


def test_a_plan_without_steps_is_rejected() -> None:
    assert problems_of() == ("the plan has no steps; it needs at least 1",)


def test_a_plan_over_the_step_limit_is_rejected() -> None:
    steps = [plan_step(f"step_{n}") for n in range(1, MAX_STEPS + 2)]

    assert problems_of(*steps) == (f"the plan has {MAX_STEPS + 1} steps; the maximum is 8",)


def test_a_plan_at_the_step_limit_is_accepted() -> None:
    assert len(check(*[plan_step(f"step_{n}") for n in range(1, MAX_STEPS + 1)]).plan.steps) == 8


# --- agents and actions ------------------------------------------------------------------


def test_an_agent_that_is_not_implemented_is_rejected_and_the_real_ones_are_named() -> None:
    assert problems_of(plan_step("step_1", agent="translation", action="translate")) == (
        "step_1: unknown agent 'translation'; available agents: analysis, code, research, writing",
    )


def test_an_action_the_agent_does_not_offer_is_rejected_and_the_real_ones_are_named() -> None:
    assert problems_of(plan_step("step_1", action="write")) == (
        "step_1: agent 'research' does not support action 'write'; "
        "supported actions: research, summarize",
    )


def test_validation_follows_the_catalog_it_is_given_not_a_fixed_agent_list() -> None:
    catalog = AgentCatalog(
        [
            AgentSpec("research", "r.", frozenset({"research"})),
            AgentSpec("writing", "w.", frozenset({"write"})),
        ]
    )

    validated = check(
        plan_step("step_1"),
        plan_step("step_2", "step_1", agent="writing", action="write"),
        catalog=catalog,
    )

    assert validated.parallel_groups == (("step_1",), ("step_2",))


# --- the dependency graph ----------------------------------------------------------------


def test_a_repeated_step_id_is_rejected() -> None:
    assert problems_of(plan_step("step_1"), plan_step("step_1")) == (
        "step_id 'step_1' is used by more than one step",
    )


def test_a_dependency_on_a_missing_step_is_rejected() -> None:
    assert problems_of(plan_step("step_1"), plan_step("step_2", "step_7")) == (
        "step_2: depends on 'step_7', which is not a step in this plan",
    )


def test_a_step_that_depends_on_itself_is_rejected_once_not_also_as_a_cycle() -> None:
    assert problems_of(plan_step("step_1", "step_1")) == ("step_1: depends on itself",)


def test_a_cycle_is_rejected_with_the_steps_that_form_it() -> None:
    assert problems_of(plan_step("step_1", "step_2"), plan_step("step_2", "step_1")) == (
        "dependency cycle (each step depends on the next): step_1 -> step_2 -> step_1",
    )


def test_a_cycle_is_still_found_when_other_edges_are_broken() -> None:
    # Only one repair attempt follows, so the model must see the cycle in the same round
    # as the missing dependency, not discover it after fixing that.
    problems = problems_of(
        plan_step("step_1", "step_9"),
        plan_step("step_2", "step_3"),
        plan_step("step_3", "step_2"),
    )

    assert problems == (
        "step_1: depends on 'step_9', which is not a step in this plan",
        "dependency cycle (each step depends on the next): step_2 -> step_3 -> step_2",
    )


def test_no_cycle_is_guessed_at_when_step_ids_repeat() -> None:
    # With two step_1s, "depends on step_1" has no single meaning: report only the duplicate.
    problems = problems_of(
        plan_step("step_1", "step_2"), plan_step("step_2", "step_1"), plan_step("step_1")
    )

    assert problems == ("step_id 'step_1' is used by more than one step",)


def test_every_problem_is_reported_together() -> None:
    problems = problems_of(
        plan_step("step_1", agent="translation", action="translate"),
        plan_step("step_2", "step_2", action="code"),
        plan_step("step_3", "step_8"),
    )

    assert problems == (
        "step_1: unknown agent 'translation'; available agents: analysis, code, research, writing",
        "step_2: agent 'research' does not support action 'code'; "
        "supported actions: research, summarize",
        "step_2: depends on itself",
        "step_3: depends on 'step_8', which is not a step in this plan",
    )


# --- the shape of the JSON ---------------------------------------------------------------


def test_text_that_is_not_json_is_a_problem_not_a_crash() -> None:
    [problem] = parse_problems("Here is your plan: step 1, research...")

    assert problem.startswith("Invalid JSON")


def test_json_that_is_not_an_object_is_rejected() -> None:
    assert parse_problems("[]") == ("Input should be an object",)


def test_shape_problems_name_their_exact_location() -> None:
    raw = json.dumps(
        {
            "steps": [
                {
                    "step_id": "first",
                    "agent": 3,
                    "action": "research",
                    "input": "   ",
                    "dependecies": [],
                },
                {"step_id": "step_2", "agent": "research", "action": "research"},
            ]
        }
    )

    assert parse_problems(raw) == (
        "steps[0].step_id: String should match pattern '^step_[1-9][0-9]*$'",
        "steps[0].agent: Input should be a valid string",
        "steps[0].input: String should have at least 1 character",
        "steps[0]: unknown field 'dependecies'",
        "steps[1].input: Field required",
    )


def test_an_unknown_top_level_field_is_rejected() -> None:
    raw = json.dumps({"steps": [plan_step("step_1")], "notes": "extra"})

    assert parse_problems(raw) == ("unknown field 'notes'",)


def test_a_parsed_plan_is_immutable() -> None:
    plan = parse_plan(plan_json(plan_step("step_1")))

    assert isinstance(plan, Plan)
    with pytest.raises(ValidationError, match="frozen"):
        plan.steps[0].input = "changed"
