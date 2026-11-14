import pytest

from orchestrator.agents import AnalysisAgent, CodeAgent, ResearchAgent, WritingAgent
from orchestrator.agents.catalog import default_catalog
from orchestrator.agents.factory import AgentFactory, UnknownAgentError
from orchestrator.llm.fake import FakeLLM


def test_every_agent_the_planner_may_name_can_be_run() -> None:
    # The catalog (planner side) and the factory (executor side) read the same list.
    assert AgentFactory(FakeLLM()).names == default_catalog().names


def test_one_shared_instance_per_agent() -> None:
    factory = AgentFactory(FakeLLM())

    agent = factory.get("research")

    assert isinstance(agent, ResearchAgent)
    assert factory.get("research") is agent


def test_an_unknown_agent_names_the_ones_available() -> None:
    with pytest.raises(
        UnknownAgentError,
        match=r"no agent named 'translation' \(available: analysis, code, research, writing\)",
    ):
        AgentFactory(FakeLLM()).get("translation")


def test_registering_an_agent_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="registered twice"):
        AgentFactory(FakeLLM(), [ResearchAgent, ResearchAgent])


@pytest.mark.parametrize(
    ("name", "agent_type"),
    [
        ("research", ResearchAgent),
        ("writing", WritingAgent),
        ("analysis", AnalysisAgent),
        ("code", CodeAgent),
    ],
)
def test_the_factory_builds_each_implemented_agent(name: str, agent_type: type) -> None:
    agent = AgentFactory(FakeLLM()).get(name)

    assert isinstance(agent, agent_type)
    assert agent.name == name
