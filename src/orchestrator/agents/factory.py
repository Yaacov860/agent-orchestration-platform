"""Agent instances for the executor, built from the same list the planner's catalog uses.

The catalog (planning side) and the factory (execution side) both read IMPLEMENTED_AGENTS,
so every agent a plan may name is an agent the executor can run.
"""

from collections.abc import Iterable

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.catalog import IMPLEMENTED_AGENTS
from orchestrator.llm import LLMClient


class UnknownAgentError(Exception):
    def __init__(self, name: str, available: Iterable[str]) -> None:
        super().__init__(f"no agent named '{name}' (available: {', '.join(sorted(available))})")
        self.name = name


class AgentFactory:
    """One instance per agent class, shared by every step and task.

    Sharing is safe: agents keep no per-run state (BaseAgent meters each run separately),
    so concurrent steps on one instance do not mix their tokens.
    """

    def __init__(
        self, llm: LLMClient, agents: Iterable[type[BaseAgent]] = IMPLEMENTED_AGENTS
    ) -> None:
        self._agents: dict[str, BaseAgent] = {}
        for agent_class in agents:
            if agent_class.name in self._agents:
                raise ValueError(f"agent '{agent_class.name}' is registered twice")
            self._agents[agent_class.name] = agent_class(llm)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))

    def get(self, name: str) -> BaseAgent:
        try:
            return self._agents[name]
        except KeyError:
            raise UnknownAgentError(name, self._agents) from None
