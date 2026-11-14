"""What code, not the model, decides about a result: where it came from and how far to trust it.

Rules, all deterministic and testable:
- sources are exactly the pages the steps' providers returned, each with the steps that got it;
- a finding is `corroborated` only if it cites two steps that do not derive from each other
  (a summary of step_1 agreeing with step_1 is not a second opinion);
- confidence is computed from evidence, coverage and conflicts, and states its own basis;
- limitations name every step that failed or did not run, and every weakness in support.
"""

from collections.abc import Iterable, Mapping, Sequence

from orchestrator.synthesis.types import (
    Conflict,
    KeyFinding,
    ResultSource,
    StepEvidence,
    SynthesisInput,
    Verification,
)

CONFLICT_FACTOR = 0.8
"""Confidence is multiplied by this when the outputs contradict each other."""


def ancestors(steps: Sequence[StepEvidence]) -> dict[str, frozenset[str]]:
    """For each step, every step it derives from, directly or not."""
    direct = {s.step_id: set(s.dependencies) for s in steps}
    result: dict[str, frozenset[str]] = {}

    def visit(step_id: str) -> frozenset[str]:
        if step_id not in result:
            found: set[str] = set()
            for dependency in direct.get(step_id, ()):
                found |= {dependency, *visit(dependency)}
            result[step_id] = frozenset(found)
        return result[step_id]

    for step in steps:
        visit(step.step_id)
    return result


def verification(step_ids: Sequence[str], lineage: Mapping[str, frozenset[str]]) -> Verification:
    if not step_ids:
        return "unattributed"
    for i, first in enumerate(step_ids):
        for second in step_ids[i + 1 :]:
            independent = first not in lineage.get(second, ()) and second not in lineage.get(
                first, ()
            )
            if first != second and independent:
                return "corroborated"
    return "single_step"


def collect_sources(steps: Iterable[StepEvidence]) -> tuple[ResultSource, ...]:
    """Every page a step's provider returned, once per URL, with all steps that returned it."""
    titles: dict[str, str] = {}
    returned_by: dict[str, list[str]] = {}
    for step in steps:
        for source in step.sources:
            titles.setdefault(source.url, source.title)
            if step.step_id not in returned_by.setdefault(source.url, []):
                returned_by[source.url].append(step.step_id)
    return tuple(
        ResultSource(title=titles[url], url=url, step_ids=tuple(ids))
        for url, ids in returned_by.items()
    )


def confidence(
    synthesis_input: SynthesisInput, conflicts: Sequence[Conflict]
) -> tuple[float | None, str]:
    """(value, basis). An explainable heuristic, not a probability:

        mean evidence confidence of the completed steps that report one
        x coverage (completed steps / all steps of the plan)
        x 0.8 if the outputs conflict

    Step confidence is itself evidence-based (grounded sources, DECISIONS.md 0.1). With no
    step reporting one (only summaries ran), there is nothing to base a number on: None.
    """
    usable = synthesis_input.usable_steps
    scores = [s.confidence for s in usable if s.confidence is not None]
    if not scores:
        return None, "no completed step reported evidence-based confidence, so none is given"
    evidence = sum(scores) / len(scores)
    coverage = len(usable) / len(synthesis_input.steps)
    factor = CONFLICT_FACTOR if conflicts else 1.0
    value = round(evidence * coverage * factor, 2)
    basis = (
        f"mean evidence confidence {evidence:.2f} of {len(scores)} step(s)"
        f" x coverage {len(usable)}/{len(synthesis_input.steps)}"
    )
    if conflicts:
        basis += f" x {CONFLICT_FACTOR} for {len(conflicts)} conflict(s)"
    return value, basis


def step_limitations(
    synthesis_input: SynthesisInput, truncated: Iterable[str], max_chars: int
) -> list[str]:
    """What is missing, stated by code: never left to the model to remember."""
    notes: list[str] = []
    for step in synthesis_input.steps:
        if step.usable:
            continue
        why = step.error or f"its status is {step.status}"
        if step.status == "failed":
            notes.append(
                f"{step.step_id} ({step.agent}: {step.action}) failed: {why}. "
                "Its findings are missing from this result."
            )
        else:
            notes.append(f"{step.step_id} ({step.agent}: {step.action}) produced no output: {why}.")
    for step in synthesis_input.usable_steps:
        if step.grounding_status == "unverified":
            notes.append(
                f"{step.step_id} ({step.agent}: {step.action}) is unverified: "
                f"{step.grounding_reason or 'no web search ran'}. Its findings come from the "
                "model, not from cited pages."
            )
    for step_id in truncated:
        notes.append(
            f"{step_id}'s output was longer than {max_chars} characters and was cut "
            "for synthesis; the full output is in the step results."
        )
    if not any(s.sources for s in synthesis_input.usable_steps):
        notes.append("No step returned web sources: the findings are not backed by cited pages.")
    return notes


def support_limitations(findings: Sequence[KeyFinding]) -> list[str]:
    notes: list[str] = []
    single = sum(f.verification == "single_step" for f in findings)
    unattributed = sum(f.verification == "unattributed" for f in findings)
    if single:
        notes.append(
            f"{single} of {len(findings)} key finding(s) rest on a single step and are "
            "not corroborated."
        )
    if unattributed:
        notes.append(f"{unattributed} key finding(s) could not be attributed to any step.")
    return notes


def merge_unique(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for group in groups:
        for item in group:
            seen.setdefault(item.strip(), None)
    return tuple(item for item in seen if item)
