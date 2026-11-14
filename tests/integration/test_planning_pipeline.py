"""The whole planning pipeline with every real part: catalog from the agent classes, prompt,
parsing, validation, dependency graph, repair loop. Only the LLM is scripted, at the network
boundary. No database: storing plans belongs to the planning service, not the planner.
"""

import json

import pytest
from factories import plan_step
from pydantic import JsonValue

from orchestrator.agents import ResearchAgent
from orchestrator.agents.catalog import AgentCatalog, AgentSpec, default_catalog
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.planning import Planner, parse_plan, validate_plan

GOAL = "Write a comparison blog post about Python vs JavaScript for beginners"
CONSTRAINTS: dict[str, JsonValue] = {
    "max_words": 1500,
    "tone": "friendly",
    "include_code_examples": True,
}


def answer(*steps: dict[str, object], reasoning: str) -> str:
    return json.dumps({"reasoning": reasoning, "steps": list(steps)}, indent=2)


async def test_a_flawed_first_plan_is_repaired_into_a_runnable_graph() -> None:
    # Attempt 1 has the two mistakes models make most: a cycle and an action that does not
    # exist, and it comes wrapped in a code fence. Attempt 2 is the fixed plan.
    flawed = answer(
        plan_step("step_1", input="Research Python's strengths for beginners"),
        plan_step("step_2", "step_3", input="Research JavaScript's strengths for beginners"),
        plan_step("step_3", "step_1", "step_2", action="compare", input="Compare the two"),
        reasoning="Two research branches, then a comparison.",
    )
    fixed = answer(
        plan_step("step_1", input="Research Python's strengths for beginners"),
        plan_step("step_2", input="Research JavaScript's strengths for beginners"),
        plan_step("step_3", input="Research which one beginners find easier to start with"),
        plan_step(
            "step_4",
            "step_1",
            "step_2",
            "step_3",
            action="summarize",
            input="Summarize the findings into a side-by-side comparison",
        ),
        reasoning="Three independent research branches in parallel, then one summary.",
    )
    llm = FakeLLM([fake_response(f"```json\n{flawed}\n```"), fake_response(fixed)])

    result = await Planner(llm, default_catalog()).plan(
        goal=GOAL, constraints=CONSTRAINTS, output_format="markdown"
    )

    assert result.attempts == 2
    plan = result.validated.plan
    assert plan.reasoning.startswith("Three independent research branches")
    assert [(s.step_id, s.action, s.dependencies) for s in plan.steps] == [
        ("step_1", "research", ()),
        ("step_2", "research", ()),
        ("step_3", "research", ()),
        ("step_4", "summarize", ("step_1", "step_2", "step_3")),
    ]
    assert result.validated.parallel_groups == (("step_1", "step_2", "step_3"), ("step_4",))

    # The repair prompt named both mistakes of attempt 1 at once.
    repair_prompt = llm.calls[1].user_prompt
    assert "step_3: agent 'research' does not support action 'compare'" in repair_prompt
    assert "dependency cycle (each step depends on the next): step_2 -> step_3 -> step_2" in (
        repair_prompt
    )


async def test_more_agents_in_the_catalog_reach_the_prompt_and_validation_alike() -> None:
    # A catalog with a (test-only) writing agent next to the real research agent: the
    # blog-post shape the assignment describes. Nothing in the planner changes.
    catalog = AgentCatalog(
        [
            *AgentCatalog.from_agents([ResearchAgent]).specs,
            AgentSpec("writing", "Writes and edits text.", frozenset({"write", "edit"})),
        ]
    )
    blog = answer(
        plan_step("step_1"),
        plan_step("step_2"),
        plan_step("step_3", "step_1", "step_2", agent="writing", action="write"),
        plan_step("step_4", "step_3", agent="writing", action="edit"),
        reasoning="Research in parallel, then write, then edit.",
    )
    llm = FakeLLM([fake_response(blog)])

    result = await Planner(llm, catalog).plan(goal=GOAL, constraints={}, output_format="markdown")

    assert "- writing: Writes and edits text. Actions: edit, write." in llm.calls[0].system_prompt
    assert result.validated.parallel_groups == (("step_1", "step_2"), ("step_3",), ("step_4",))


async def test_a_validated_plan_survives_a_round_trip_through_json() -> None:
    # The plan is stored as JSONB; what is stored must load back into the same valid plan.
    llm = FakeLLM(
        [fake_response(answer(plan_step("step_1"), plan_step("step_2", "step_1"), reasoning="r"))]
    )
    result = await Planner(llm, default_catalog()).plan(
        goal=GOAL, constraints={}, output_format="text"
    )

    stored = result.validated.plan.model_dump_json()
    reloaded = validate_plan(parse_plan(stored), default_catalog())

    assert reloaded == result.validated


# --- multi-agent plans: routing to the right agent and action ----------------------------


def step(step_id: str, agent: str, action: str, *dependencies: str) -> dict[str, object]:
    return plan_step(step_id, *dependencies, agent=agent, action=action, input=f"{agent} {action}")


SCENARIOS = {
    "research then a document": (
        [
            step("step_1", "research", "research"),
            step("step_2", "research", "research"),
            step("step_3", "writing", "write", "step_1", "step_2"),
        ],
        (("step_1", "step_2"), ("step_3",)),
    ),
    "research then an analysis of alternatives": (
        [
            step("step_1", "research", "research"),
            step("step_2", "research", "research"),
            step("step_3", "analysis", "compare", "step_1", "step_2"),
        ],
        (("step_1", "step_2"), ("step_3",)),
    ),
    "requirements analysis then code": (
        [
            step("step_1", "analysis", "analyze"),
            step("step_2", "code", "write_code", "step_1"),
            step("step_3", "code", "review_code", "step_2"),
        ],
        (("step_1",), ("step_2",), ("step_3",)),
    ),
    "research, analysis, writing and code together": (
        [
            step("step_1", "research", "research"),
            step("step_2", "analysis", "compare", "step_1"),
            step("step_3", "writing", "write", "step_2"),
            step("step_4", "code", "write_code", "step_2"),
            step("step_5", "writing", "draft_email", "step_3"),
        ],
        (("step_1",), ("step_2",), ("step_3", "step_4"), ("step_5",)),
    ),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_a_multi_agent_plan_is_accepted_and_scheduled(scenario: str) -> None:
    steps, groups = SCENARIOS[scenario]
    llm = FakeLLM([fake_response(answer(*steps, reasoning=scenario))])

    result = await Planner(llm, default_catalog()).plan(
        goal=GOAL, constraints={}, output_format="markdown"
    )

    assert result.attempts == 1
    assert [(s.agent, s.action) for s in result.validated.plan.steps] == [
        (s["agent"], s["action"]) for s in steps
    ]
    assert result.validated.parallel_groups == groups


async def test_an_action_routed_to_the_wrong_agent_is_sent_back_naming_the_right_ones() -> None:
    # writing cannot research and research cannot write code: the repair prompt says what
    # each agent can do, so the model can re-route the step.
    wrong = answer(
        step("step_1", "writing", "research"),
        step("step_2", "research", "write_code", "step_1"),
        reasoning="x",
    )
    right = answer(
        step("step_1", "research", "research"),
        step("step_2", "code", "write_code", "step_1"),
        reasoning="x",
    )
    llm = FakeLLM([fake_response(wrong), fake_response(right)])

    result = await Planner(llm, default_catalog()).plan(
        goal=GOAL, constraints={}, output_format="markdown"
    )

    assert result.attempts == 2
    repair = llm.calls[1].user_prompt
    assert (
        "step_1: agent 'writing' does not support action 'research'; "
        "supported actions: draft_email, rewrite, write" in repair
    )
    assert (
        "step_2: agent 'research' does not support action 'write_code'; "
        "supported actions: research, summarize" in repair
    )


async def test_the_planner_prompt_describes_every_agent_and_action() -> None:
    llm = FakeLLM([fake_response(answer(step("step_1", "research", "research"), reasoning="r"))])

    await Planner(llm, default_catalog()).plan(goal=GOAL, constraints={}, output_format="text")

    prompt = llm.calls[0].system_prompt
    assert default_catalog().describe() in prompt
    assert "Pick each step's agent and action by what that step must produce" in prompt
    assert "gathers no information itself" in prompt


async def test_the_planner_prompt_names_no_agent_the_catalog_lacks() -> None:
    catalog = AgentCatalog([AgentSpec("writing", "Writes text.", frozenset({"write"}))])
    llm = FakeLLM([fake_response(answer(step("step_1", "writing", "write"), reasoning="r"))])

    await Planner(llm, catalog).plan(goal=GOAL, constraints={}, output_format="text")

    prompt = llm.calls[0].system_prompt
    for absent in ("research", "analysis", "code agent", "summarize"):
        assert absent not in prompt, absent
