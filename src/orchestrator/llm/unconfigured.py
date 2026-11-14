from orchestrator.llm.errors import LLMPermanentError
from orchestrator.llm.types import LLMRequest, LLMResponse

NO_PROVIDER_MESSAGE = (
    "no LLM provider is configured (LLM_PROVIDER=none), so tasks cannot be planned or run; "
    "configure a provider and submit the task again"
)
MISSING_GEMINI_KEY_MESSAGE = (
    "no LLM provider is configured: LLM_PROVIDER=gemini but GEMINI_API_KEY is not set, so "
    "tasks cannot be planned or run; set the key and submit the task again"
)


class UnconfiguredLLM:
    """The LLMClient used when no provider is set up.

    The API still starts, accepts tasks and serves them; anything that needs a model fails
    with one clear, permanent (non-retryable) error instead of a crash or a hang. Retrying
    cannot help: only configuration can.
    """

    model = "none"

    def __init__(self, message: str = NO_PROVIDER_MESSAGE) -> None:
        self.message = message

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMPermanentError(self.message)
