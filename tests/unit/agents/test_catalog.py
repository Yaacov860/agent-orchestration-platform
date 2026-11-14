import pytest

from orchestrator.agents import AnalysisAgent, CodeAgent, ResearchAgent, WritingAgent
from orchestrator.agents.catalog import AgentCatalog, AgentSpec, default_catalog


def spec(name: str, *capabilities: str) -> AgentSpec:
    return AgentSpec(name=name, description=f"{name} agent.", capabilities=frozenset(capabilities))


def test_the_catalog_reads_each_agents_declared_contract() -> None:
    research = AgentCatalog.from_agents([ResearchAgent]).get("research")

    assert research == AgentSpec(
        name="research",
        description=ResearchAgent.description,
        capabilities=frozenset({"research", "summarize"}),
        actions=tuple(sorted(ResearchAgent.action_descriptions.items())),
        searches_web=True,
    )


def test_the_default_catalog_holds_only_implemented_agents() -> None:
    # Decision: the planner may only plan agents that exist. Add an agent here when it lands.
    assert default_catalog().names == ("analysis", "code", "research", "writing")


def test_supports_checks_both_the_agent_and_the_action() -> None:
    catalog = AgentCatalog([spec("research", "research", "summarize")])

    assert catalog.supports("research", "summarize")
    assert not catalog.supports("research", "write")
    assert not catalog.supports("writing", "write")
    assert catalog.get("writing") is None


def test_describe_lists_agents_and_actions_in_a_stable_order() -> None:
    catalog = AgentCatalog([spec("writing", "write", "edit"), spec("analysis", "compare")])

    assert catalog.names == ("analysis", "writing")
    assert catalog.describe() == (
        "- analysis: analysis agent. Actions: compare.\n"
        "- writing: writing agent. Actions: edit, write."
    )


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([spec("research", "a"), spec("research", "b")], "registered twice"),
        ([spec("research")], "has no capabilities"),
        ([], "at least one agent"),
    ],
)
def test_an_inconsistent_catalog_fails_at_construction(
    specs: list[AgentSpec], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentCatalog(specs)


def test_the_planner_sees_what_every_action_of_every_agent_does() -> None:
    text = default_catalog().describe()

    for agent in (ResearchAgent, WritingAgent, AnalysisAgent, CodeAgent):
        assert f"- {agent.name}: {agent.description}" in text
        for action, what in agent.action_descriptions.items():
            assert f"  - {action}: {what}" in text


def test_action_descriptions_that_miss_a_capability_are_refused() -> None:
    with pytest.raises(ValueError, match="action descriptions do not match"):
        AgentCatalog(
            [AgentSpec("writing", "w.", frozenset({"write", "rewrite"}), actions=(("write", "w"),))]
        )
