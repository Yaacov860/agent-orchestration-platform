"""A scripted, in-memory LLMClient for tests and key-less demos.

The test writes the "conversation" up front: each call to `generate` pops the next item.
An item is either a response to return or an exception to raise, so failure paths
(rate limits, timeouts, blocked content) are as easy to test as the happy path.
"""

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import TypeAlias

from orchestrator.llm.grounding import GroundingReport, may_ground
from orchestrator.llm.types import LLMRequest, LLMResponse, Source, TokenUsage

# BaseException, not Exception, so a test can script asyncio.CancelledError too.
ScriptItem: TypeAlias = LLMResponse | BaseException

SEARCH_OFF = "the fake provider was built without search"


class FakeLLM:
    model = "fake-llm"

    def __init__(self, script: Iterable[ScriptItem] = (), *, search: bool = True) -> None:
        self._script: deque[ScriptItem] = deque(script)
        self.calls: list[LLMRequest] = []
        """Every request received, in order, so tests can assert on what an agent sent."""
        self.search = search
        """A provider with web search, like GeminiLLM with grounding on: a search request
        reserves a slot from its step's grounding budget before the call, and gets the
        scripted sources and queries only if the slot was granted."""

    def add(self, *items: ScriptItem) -> None:
        """Append to the script, for a FakeLLM already wired into an app by a fixture."""
        self._script.extend(items)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        # Before the call, as a real provider must: a failed call still used its slot.
        report = (
            may_ground(request, enabled=self.search, disabled_reason=SEARCH_OFF)
            if request.use_search
            else None
        )
        # A real suspension point, like a network call: lets other coroutines run meanwhile.
        await asyncio.sleep(0)
        if not self._script:
            raise AssertionError(
                f"FakeLLM script exhausted: call #{len(self.calls)} has no scripted reply"
            )
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        if report is None:
            return item
        queries = item.grounding.queries if report.used and item.grounding else ()
        if report.used and request.grounding is not None:
            request.grounding.record_queries(queries)
        return item.model_copy(
            update={
                # A call that went without search has nothing to cite.
                "sources": item.sources if report.used else (),
                "grounding": GroundingReport(
                    used=report.used, reason=report.reason, queries=queries
                ),
            }
        )


def fake_response(
    text: str,
    *,
    sources: Iterable[Source] = (),
    search_queries: Iterable[str] = (),
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    """Shorthand for building a scripted reply. `search_queries`: what the provider reports
    the search ran, for a reply to a search request."""
    queries = tuple(search_queries)
    return LLMResponse(
        text=text,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=FakeLLM.model,
        sources=tuple(sources),
        grounding=GroundingReport(used=True, queries=queries) if queries else None,
    )
