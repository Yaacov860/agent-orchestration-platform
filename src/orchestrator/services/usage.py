"""Token usage of a task, by stage, from what the database recorded.

    planning   = planner tokens of every plan row        + failed planning (trace events)
    execution  = tokens of every step of every plan      (every attempt, failed ones included)
    synthesis  = tokens of the stored result             + failed synthesis (trace events)
    search     = grounded calls and reported search queries of every step (execution only)

Failed work counts: it was paid for. Superseded plans count too, for the same reason.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import JsonValue

from orchestrator.db.models import ExecutionPlan, ExecutionStep, TaskEvent, TaskResult
from orchestrator.domain.status import FailureStage
from orchestrator.llm.grounding import SearchUsage
from orchestrator.llm.types import TokenUsage


def usage_detail(usage: TokenUsage | None) -> dict[str, JsonValue] | None:
    """Tokens a failed stage spent, for its trace event (nothing else would record them)."""
    if usage is None or usage.total == 0:
        return None
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


@dataclass(frozen=True)
class TaskUsage:
    planning: TokenUsage
    execution: TokenUsage
    synthesis: TokenUsage
    search: SearchUsage = field(default_factory=SearchUsage)
    """Web search, which is billed apart from tokens: only research steps search."""

    @property
    def total(self) -> TokenUsage:
        return self.planning + self.execution + self.synthesis


def task_usage(
    plans: Iterable[ExecutionPlan],
    steps: Iterable[ExecutionStep],
    events: Iterable[TaskEvent],
    result: TaskResult | None,
) -> TaskUsage:
    planning = _sum(
        TokenUsage(input_tokens=p.planner_input_tokens, output_tokens=p.planner_output_tokens)
        for p in plans
    )
    steps_seen = list(steps)
    execution = _sum(
        TokenUsage(input_tokens=s.input_tokens, output_tokens=s.output_tokens) for s in steps_seen
    )
    synthesis = (
        TokenUsage(input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        if result is not None
        else TokenUsage()
    )
    search = SearchUsage()
    for step in steps_seen:
        search = search + SearchUsage(
            grounded_calls=step.grounded_calls, queries=step.search_queries
        )
    for event in events:
        spent = _from_detail(event.detail)
        if event.failure_stage is FailureStage.PLANNING:
            planning = planning + spent
        elif event.failure_stage is FailureStage.SYNTHESIS:
            synthesis = synthesis + spent
    return TaskUsage(planning=planning, execution=execution, synthesis=synthesis, search=search)


def _sum(usages: Iterable[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total = total + usage
    return total


def _from_detail(detail: dict[str, JsonValue] | None) -> TokenUsage:
    if not detail:
        return TokenUsage()
    input_tokens, output_tokens = detail.get("input_tokens", 0), detail.get("output_tokens", 0)
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens=output_tokens if isinstance(output_tokens, int) else 0,
    )
