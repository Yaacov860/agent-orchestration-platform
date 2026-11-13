from typing import Protocol

from orchestrator.llm.types import LLMRequest, LLMResponse


class LLMClient(Protocol):
    """The contract every LLM provider fulfils.

    A `Protocol` is structural typing: any class with a matching `generate` method *is* an
    LLMClient, no inheritance needed. Agents depend on this, never on a concrete provider.

    Implementations must raise only `LLMTransientError` / `LLMPermanentError` (see errors.py).
    """

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
