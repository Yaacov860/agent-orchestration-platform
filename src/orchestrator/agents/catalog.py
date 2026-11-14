"""What agents exist and what each can do: the single source the planner works from.

The planner uses the same catalog twice: to tell the model which agents and actions it may
use (the prompt), and to check the plan it gets back (validation). One source, so the prompt
can never offer something validation rejects, or the other way round.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from orchestrator.agents.analysis import AnalysisAgent
from orchestrator.agents.base import BaseAgent
from orchestrator.agents.code import CodeAgent
from orchestrator.agents.research import ResearchAgent
from orchestrator.agents.writing import WritingAgent

IMPLEMENTED_AGENTS: tuple[type[BaseAgent], ...] = (
    ResearchAgent,
    WritingAgent,
    AnalysisAgent,
    CodeAgent,
)
"""Only agents that are actually implemented, so the planner never plans a step nobody can run.
A new agent becomes plannable by being added here; nothing else changes."""


@dataclass(frozen=True)
class AgentSpec:
    """What the planner needs to know about an agent. No LLM, no instance: just its contract."""

    name: str
    description: str
    capabilities: frozenset[str]
    actions: tuple[tuple[str, str], ...] = ()
    """(action, what it does), sorted. Optional; when given, it covers every capability."""
    searches_web: bool = False


class AgentCatalog:
    def __init__(self, specs: Iterable[AgentSpec]) -> None:
        by_name: dict[str, AgentSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"agent '{spec.name}' is registered twice")
            if not spec.capabilities:
                raise ValueError(f"agent '{spec.name}' has no capabilities")
            if spec.actions and {a for a, _ in spec.actions} != set(spec.capabilities):
                raise ValueError(
                    f"agent '{spec.name}': action descriptions do not match its capabilities"
                )
            by_name[spec.name] = spec
        if not by_name:
            raise ValueError("a catalog needs at least one agent")
        self._specs: Mapping[str, AgentSpec] = dict(sorted(by_name.items()))

    @classmethod
    def from_agents(cls, agents: Iterable[type[BaseAgent]]) -> "AgentCatalog":
        """Read each agent class's declared contract (its ClassVars)."""
        return cls(
            AgentSpec(
                name=a.name,
                description=a.description,
                capabilities=a.capabilities,
                actions=tuple(sorted(a.action_descriptions.items())),
                searches_web=a.searches_web,
            )
            for a in agents
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def specs(self) -> tuple[AgentSpec, ...]:
        return tuple(self._specs.values())

    def get(self, name: str) -> AgentSpec | None:
        return self._specs.get(name)

    def supports(self, agent: str, action: str) -> bool:
        spec = self._specs.get(agent)
        return spec is not None and action in spec.capabilities

    def describe(self) -> str:
        """The agents as the planner prompt lists them, with what each action does, so the
        planner chooses by what a step needs rather than by names. Sorted: a stable prompt."""
        blocks = []
        for spec in self._specs.values():
            if spec.actions:
                actions = "\n".join(f"  - {action}: {what}" for action, what in spec.actions)
                blocks.append(f"- {spec.name}: {spec.description}\n{actions}")
            else:
                actions_list = ", ".join(sorted(spec.capabilities))
                blocks.append(f"- {spec.name}: {spec.description} Actions: {actions_list}.")
        return "\n".join(blocks)


def default_catalog() -> AgentCatalog:
    return AgentCatalog.from_agents(IMPLEMENTED_AGENTS)
