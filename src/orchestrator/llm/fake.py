"""A scripted, in-memory LLMClient for tests and key-less demos.

The test writes the "conversation" up front: each call to `generate` pops the next item.
An item is either a response to return or an exception to raise, so failure paths
(rate limits, timeouts, blocked content) are as easy to test as the happy path.
"""

import asyncio
from collections import deque
from collections.abc import Iterable

from orchestrator.llm.types import LLMRequest, LLMResponse, Source, TokenUsage

# BaseException, not Exception, so a test can script asyncio.CancelledError too.
type ScriptItem = LLMResponse | BaseException


class FakeLLM:
    model = "fake-llm"

    def __init__(self, script: Iterable[ScriptItem] = ()) -> None:
        self._script: deque[ScriptItem] = deque(script)
        self.calls: list[LLMRequest] = []
        """Every request received, in order, so tests can assert on what an agent sent."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        # A real suspension point, like a network call: lets other coroutines run meanwhile.
        await asyncio.sleep(0)
        if not self._script:
            raise AssertionError(
                f"FakeLLM script exhausted: call #{len(self.calls)} has no scripted reply"
            )
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


def fake_response(
    text: str,
    *,
    sources: Iterable[Source] = (),
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    """Shorthand for building a scripted reply."""
    return LLMResponse(
        text=text,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=FakeLLM.model,
        sources=tuple(sources),
    )
