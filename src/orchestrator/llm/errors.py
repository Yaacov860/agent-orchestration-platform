"""The only LLM errors code outside the `llm` package ever sees.

Each provider maps its SDK's exceptions onto these two. The split exists for the retry
logic: a transient error (rate limit, timeout, 5xx, empty reply) is worth another try;
a permanent one (bad request, auth, content blocked) will fail the same way every time.
"""


class LLMError(Exception):
    retryable: bool = False


class LLMTransientError(LLMError):
    retryable = True


class LLMPermanentError(LLMError):
    retryable = False
