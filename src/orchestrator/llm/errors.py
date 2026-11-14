"""The only LLM errors code outside the `llm` package ever sees.

Each provider maps its SDK's exceptions onto these two. The split exists for the retry
logic: a transient error (rate limit, timeout, 5xx, empty reply) is worth another try;
a permanent one (bad request, auth, content blocked) will fail the same way every time.
"""


class LLMError(Exception):
    retryable: bool = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        """Seconds the provider asked to wait before retrying, when it said (a rate limit's
        RetryInfo). Retrying sooner would only hit the same limit again."""


class LLMTransientError(LLMError):
    retryable = True


class LLMPermanentError(LLMError):
    retryable = False


MAX_RETRY_WAIT = 60.0
"""The longest a stage waits before a retry, whatever the provider asks: past that, failing
and saying why beats holding a slot (and the task) for minutes."""


def retry_wait(planned: float, provider_hint: float | None) -> float:
    """The stage's own delay, stretched to what the provider asked for, within the cap."""
    return min(max(planned, provider_hint or 0.0), max(planned, MAX_RETRY_WAIT))
