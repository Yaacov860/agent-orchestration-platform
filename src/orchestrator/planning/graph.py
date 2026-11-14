"""Dependency graph of plan steps: which steps can run together, and in what order.

A step's `dependencies` are the ids of the steps whose output it needs. Dependencies are
declared explicitly by the plan, never inferred from the step text.
"""

from collections.abc import Mapping, Sequence


class DependencyError(Exception):
    """The dependencies do not form a runnable graph."""


class UnknownDependencyError(DependencyError):
    def __init__(self, missing: Mapping[str, Sequence[str]]) -> None:
        self.missing = {step: tuple(deps) for step, deps in missing.items()}
        details = "; ".join(f"{step} -> {', '.join(deps)}" for step, deps in self.missing.items())
        super().__init__(f"dependencies on steps that do not exist: {details}")


class DependencyCycleError(DependencyError):
    def __init__(self, cycle: Sequence[str], unresolved: Sequence[str]) -> None:
        self.cycle = tuple(cycle)
        """One cycle, closed: the first id is repeated at the end. Each step depends on the
        next one: (step_1, step_2, step_1) means step_1 needs step_2 and step_2 needs step_1."""
        self.unresolved = tuple(unresolved)
        """Every step that can never start: the cycle plus anything waiting on it."""
        super().__init__(
            f"dependency cycle (each step depends on the next): {' -> '.join(self.cycle)}"
        )


def parallel_groups(dependencies: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Split steps into groups that can run in parallel, in execution order.

    `dependencies` maps each step id to the ids it depends on; its iteration order is the
    plan order. Group N holds every step whose dependencies all sit in groups before N,
    so all steps of a group can run at once. Kahn's algorithm, level by level: O(V + E).
    Within a group, steps keep their plan order, so the result is deterministic.

    Raises UnknownDependencyError or DependencyCycleError instead of returning a partial order.
    """
    position = {step: index for index, step in enumerate(dependencies)}
    # A dependency listed twice is still one edge; counting it twice would never reach zero.
    needs = {step: set(deps) for step, deps in dependencies.items()}

    missing = {
        step: sorted(deps - position.keys())
        for step, deps in needs.items()
        if deps - position.keys()
    }
    if missing:
        raise UnknownDependencyError(missing)

    waiting = {step: len(deps) for step, deps in needs.items()}
    """How many of a step's dependencies have not finished yet (its in-degree)."""
    dependents: dict[str, list[str]] = {step: [] for step in dependencies}
    for step, deps in needs.items():
        for dependency in deps:
            dependents[dependency].append(step)

    groups: list[list[str]] = []
    ready = [step for step in dependencies if waiting[step] == 0]
    while ready:
        groups.append(ready)
        unblocked: list[str] = []
        for step in ready:
            for dependent in dependents[step]:
                waiting[dependent] -= 1
                if waiting[dependent] == 0:
                    unblocked.append(dependent)
        ready = sorted(unblocked, key=position.__getitem__)

    unresolved = [step for step in dependencies if waiting[step] > 0]
    if unresolved:
        raise DependencyCycleError(_find_cycle(unresolved, needs), unresolved)
    return groups


def _find_cycle(unresolved: Sequence[str], needs: Mapping[str, set[str]]) -> list[str]:
    """Walk backwards along dependencies until a step repeats.

    Every unresolved step depends on at least one other unresolved step (otherwise it would
    have been scheduled), so the walk never gets stuck and must eventually repeat a step.
    """
    remaining = set(unresolved)
    order = {step: index for index, step in enumerate(unresolved)}
    path: list[str] = []
    seen: dict[str, int] = {}
    step = unresolved[0]
    while step not in seen:
        seen[step] = len(path)
        path.append(step)
        # Earliest in plan order, so the reported cycle is deterministic.
        step = min(needs[step] & remaining, key=order.__getitem__)
    cycle = path[seen[step] :]
    return [*cycle, cycle[0]]
