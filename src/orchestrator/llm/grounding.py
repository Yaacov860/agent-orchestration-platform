"""Bounded web-search grounding: a per-task budget of grounded provider calls, and a ledger.

    executor ──► GroundingBudget(max_calls)      one per task execution, shared by its steps
                   └─ for_step(step_id) ──► StepGrounding   rides on each search request
    provider ──► may_ground(request)  before EVERY HTTP call that would attach search:
                   reserves a slot, or says why the call goes without search
             ──► record_queries(n)   after the call, from what the provider reports

What is hard and what is not:

- The number of grounded *requests* per task is a hard limit. A slot is reserved before each
  provider call that would attach search: first attempts, step retries and model fallbacks
  alike. A call that fails still used its slot, because it may still have searched and been
  billed. The check and the increment run with no `await` in between, so concurrent steps of
  one task, all on one event loop, cannot both take the last slot.
- The number of Google *queries* is not limited. The model decides how many queries one
  grounded call runs, and the API offers no parameter that caps them. Queries are counted
  afterwards, from the provider's report of a successful call. A call that failed or timed
  out reports nothing, so the count can be lower than what was billed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from orchestrator.llm.types import LLMRequest


class SearchUsage(BaseModel):
    """Grounded provider calls made and the search queries they reported."""

    model_config = ConfigDict(frozen=True)

    grounded_calls: int = Field(default=0, ge=0)
    """Provider calls sent with web search attached (each one reserved a budget slot)."""
    queries: int = Field(default=0, ge=0)
    """Search queries the provider reported running, for the calls that returned."""

    def __add__(self, other: SearchUsage) -> SearchUsage:
        return SearchUsage(
            grounded_calls=self.grounded_calls + other.grounded_calls,
            queries=self.queries + other.queries,
        )


class GroundingReport(BaseModel):
    """What happened to search on one provider call that asked for it."""

    model_config = ConfigDict(frozen=True)

    used: bool
    """Web search was attached to the call that produced this response."""
    reason: str | None = None
    """Why search was not attached, when it was not."""
    queries: tuple[str, ...] = ()
    """The search queries the provider reports it ran, in its words."""


class GroundingBudget:
    """The grounded-call budget of one task execution, and the ledger of what it spent.

    `max_calls=None` means no limit (the ledger still counts). Not thread-safe by design:
    it is shared by coroutines on one event loop, and reservation never awaits.
    """

    def __init__(self, max_calls: int | None) -> None:
        if max_calls is not None and max_calls < 0:
            raise ValueError("max_calls must be >= 0 or None")
        self.max_calls = max_calls
        self._used = 0
        self._by_step: dict[str, SearchUsage] = {}

    @property
    def used(self) -> int:
        return self._used

    def for_step(self, step_id: str) -> StepGrounding:
        return StepGrounding(self, step_id)

    def usage_of(self, step_id: str) -> SearchUsage:
        return self._by_step.get(step_id, SearchUsage())

    @property
    def exhausted_reason(self) -> str:
        return (
            f"the task's grounded-call budget is used up ({self._used} of {self.max_calls} "
            "calls, GEMINI_MAX_GROUNDED_CALLS_PER_TASK)"
        )

    def _reserve(self, step_id: str) -> bool:
        # No await between the check and the increment: atomic on one event loop.
        if self.max_calls is not None and self._used >= self.max_calls:
            return False
        self._used += 1
        self._add(step_id, SearchUsage(grounded_calls=1))
        return True

    def _add(self, step_id: str, usage: SearchUsage) -> None:
        self._by_step[step_id] = self.usage_of(step_id) + usage


class StepGrounding:
    """One step's handle on its task's budget. Attached to that step's search requests."""

    def __init__(self, budget: GroundingBudget, step_id: str) -> None:
        self._budget = budget
        self.step_id = step_id

    def try_reserve(self) -> bool:
        """Take one grounded-call slot before a provider call; False if none is left."""
        return self._budget._reserve(self.step_id)

    def record_queries(self, queries: Iterable[str]) -> None:
        """Count the queries a returned grounded call reports (empty ones are ignored)."""
        count = sum(1 for q in queries if q and q.strip())
        if count:
            self._budget._add(self.step_id, SearchUsage(queries=count))

    @property
    def exhausted_reason(self) -> str:
        return self._budget.exhausted_reason

    @property
    def usage(self) -> SearchUsage:
        return self._budget.usage_of(self.step_id)

    def __repr__(self) -> str:
        return f"StepGrounding(step_id={self.step_id!r})"


def may_ground(request: LLMRequest, *, enabled: bool, disabled_reason: str) -> GroundingReport:
    """Decide, just before one provider call, whether it may attach web search.

    Call it once per HTTP call (fallbacks included): a granted call has reserved its slot.
    The result is a provisional report (`queries` empty): the provider fills them in.
    """
    if not request.use_search:
        return GroundingReport(used=False, reason="search was not requested")
    if not enabled:
        return GroundingReport(used=False, reason=disabled_reason)
    if request.grounding is not None and not request.grounding.try_reserve():
        return GroundingReport(used=False, reason=request.grounding.exhausted_reason)
    return GroundingReport(used=True)
