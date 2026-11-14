"""Choose the LLMClient from configuration. The one place that knows which providers exist."""

import logging

from orchestrator.config import Settings
from orchestrator.llm.client import LLMClient
from orchestrator.llm.unconfigured import MISSING_GEMINI_KEY_MESSAGE, UnconfiguredLLM

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "none")
"""FakeLLM is deliberately not an option: unscripted, it can only fail, so tests inject a
scripted one directly instead."""


def build_llm(settings: Settings) -> LLMClient:
    """Raises ValueError for an unknown provider, so a typo in LLM_PROVIDER stops the app at
    startup instead of surfacing later as a failed task. A missing key does not stop it: the
    API still serves, and tasks fail with a message that says what to set."""
    match settings.llm_provider.strip().lower():
        case "gemini":
            if settings.gemini_api_key is None:
                logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; tasks will fail")
                return UnconfiguredLLM(MISSING_GEMINI_KEY_MESSAGE)
            # Imported here: the SDK loads only when Gemini is actually used.
            from orchestrator.llm.gemini import GeminiLLM

            logger.info(
                "LLM provider: gemini, model %s, fallbacks %s, search grounding %s",
                settings.gemini_model,
                list(settings.gemini_fallbacks),
                settings.gemini_search_grounding,
            )
            return GeminiLLM(
                api_key=settings.gemini_api_key.get_secret_value(),
                model=settings.gemini_model,
                timeout_seconds=settings.gemini_timeout_seconds,
                fallback_models=settings.gemini_fallbacks,
                search_grounding=settings.gemini_search_grounding,
            )
        case "none":
            return UnconfiguredLLM()
        case _:
            raise ValueError(
                f"unknown LLM_PROVIDER {settings.llm_provider!r}; "
                f"supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )


def describe_llm(llm: LLMClient) -> dict[str, object]:
    """What /metrics and /health/ready report about the provider. Never the key."""
    from orchestrator.llm.gemini import GeminiLLM

    if isinstance(llm, GeminiLLM):
        return {
            "provider": "gemini",
            "configured": True,
            "model": llm.model,
            "fallback_models": list(llm.fallback_models),
            "search_grounding": llm.search_grounding,
        }
    if isinstance(llm, UnconfiguredLLM):
        return {"provider": "none", "configured": False, "reason": llm.message}
    return {"provider": type(llm).__name__, "configured": True}
