"""Turn the planner model's raw output into a ValidatedPlan, or say everything wrong with it.

Every problem is collected and reported at once, never just the first: the planner sends the
whole list back to the model for one repair attempt, and a model asked to fix one problem at a
time would need many round trips. For the same reason each message names the step and says
what IS allowed, so the model can act on it without seeing this code.
"""

from collections import Counter
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from orchestrator.agents.catalog import AgentCatalog
from orchestrator.planning.graph import DependencyCycleError, parallel_groups
from orchestrator.planning.types import Plan, PlanStep, ValidatedPlan

MAX_STEPS = 8
"""Enough for research, analysis, writing and review on a few branches. More steps mostly
means a model over-splitting the task: more LLM calls, cost and latency, not better output."""


class PlanValidationError(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("invalid plan: " + "; ".join(self.problems))


def parse_plan(raw_json: str) -> Plan:
    """Parse the model's JSON text into a Plan, or raise with every shape problem found.

    Takes the text, not a dict: JSON arrays then load as the strict tuple fields, and
    malformed JSON is reported the same way as any other problem.
    """
    try:
        return Plan.model_validate_json(raw_json)
    except ValidationError as exc:
        raise PlanValidationError([_describe(error) for error in exc.errors()]) from None


def validate_plan(plan: Plan, catalog: AgentCatalog) -> ValidatedPlan:
    """Check what the schema cannot: limits, agents, actions and the dependency graph."""
    if not plan.steps:
        raise PlanValidationError(["the plan has no steps; it needs at least 1"])

    plan = _without_repeated_dependencies(plan)
    problems: list[str] = []
    if len(plan.steps) > MAX_STEPS:
        problems.append(f"the plan has {len(plan.steps)} steps; the maximum is {MAX_STEPS}")

    id_counts = Counter(step.step_id for step in plan.steps)
    duplicates = [step_id for step_id, count in id_counts.items() if count > 1]
    problems += [f"step_id '{step_id}' is used by more than one step" for step_id in duplicates]
    for step in plan.steps:
        problems += _agent_problems(step, catalog)
        problems += _dependency_problems(step, id_counts.keys())

    # With duplicate ids the graph is undefined (which step_2 does a dependency mean?), so
    # there is nothing to check. Otherwise look for cycles among the valid edges even when
    # other edges are broken: those were already reported above, and the model gets only
    # one repair attempt, so a cycle must not wait until the other problems are fixed.
    groups: list[list[str]] = []
    if not duplicates:
        valid_edges = {
            step.step_id: [d for d in step.dependencies if d in id_counts and d != step.step_id]
            for step in plan.steps
        }
        try:
            groups = parallel_groups(valid_edges)
        except DependencyCycleError as exc:
            problems.append(str(exc))

    if problems:
        raise PlanValidationError(problems)
    return ValidatedPlan(plan=plan, parallel_groups=tuple(tuple(g) for g in groups))


def _agent_problems(step: PlanStep, catalog: AgentCatalog) -> list[str]:
    spec = catalog.get(step.agent)
    if spec is None:
        return [
            f"{step.step_id}: unknown agent '{step.agent}'; "
            f"available agents: {', '.join(catalog.names)}"
        ]
    if step.action not in spec.capabilities:
        return [
            f"{step.step_id}: agent '{step.agent}' does not support action '{step.action}'; "
            f"supported actions: {', '.join(sorted(spec.capabilities))}"
        ]
    return []


def _dependency_problems(step: PlanStep, step_ids: AbstractSet[str]) -> list[str]:
    problems: list[str] = []
    for dependency in step.dependencies:
        if dependency == step.step_id:
            problems.append(f"{step.step_id}: depends on itself")
        elif dependency not in step_ids:
            problems.append(
                f"{step.step_id}: depends on '{dependency}', which is not a step in this plan"
            )
    return problems


def _without_repeated_dependencies(plan: Plan) -> Plan:
    """["step_1", "step_1"] is harmless sloppiness, not worth a repair round trip: normalize it."""
    steps = tuple(
        step.model_copy(update={"dependencies": tuple(dict.fromkeys(step.dependencies))})
        for step in plan.steps
    )
    return plan.model_copy(update={"steps": steps})


def _describe(error: ErrorDetails) -> str:
    """One Pydantic error as a readable line: `steps[1].agent: Field required`."""
    loc = error["loc"]
    if error["type"] == "extra_forbidden":
        return _prefixed(_location(loc[:-1]), f"unknown field '{loc[-1]}'")
    return _prefixed(_location(loc), error["msg"])


def _prefixed(location: str, message: str) -> str:
    return f"{location}: {message}" if location else message


def _location(loc: tuple[int | str, ...]) -> str:
    """('steps', 1, 'agent') -> 'steps[1].agent'."""
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text
