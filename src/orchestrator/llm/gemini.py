"""GeminiLLM: the LLMClient contract on Google's Gen AI SDK (Gemini Developer API).

    LLMRequest ──► GenerateContentConfig ──► client.aio.models.generate_content ──► LLMResponse
                    system_instruction          (one HTTP call: SDK retries off,
                    temperature                  timeout per call)
                    max_output_tokens
                    tools: google_search  (only when use_search)

Like every provider, it raises only LLMTransientError / LLMPermanentError (errors.py), so the
callers' retry policies (planning, execution, synthesis) never see an SDK type. The SDK's own
retries are switched off: retrying is decided in one place per caller, not multiplied by
a second layer underneath.

The API key travels in the `x-goog-api-key` header (checked in the SDK source), is held as
the SDK client's only copy, and is scrubbed from any error text before it leaves this module.
"""

import logging
import re
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol

import httpx
import httpx2
from google.genai import Client, errors, types
from pydantic import ValidationError

from orchestrator.llm.errors import LLMPermanentError, LLMTransientError
from orchestrator.llm.types import LLMRequest, LLMResponse, Source, TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
"""A fast text model with Google Search grounding: the model the official SDK's README uses
for text generation. Flash rather than Pro because one task makes many calls (planner, every
step, synthesis). Not verified against a live account here: see AI_USAGE.md."""

BLOCKED_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
        types.FinishReason.LANGUAGE,
    }
)
"""The model refused or was stopped by a filter: the same request will be stopped again."""

_MAX_ERROR_TEXT = 300

# google-genai 3.3.0 sends through `httpx2`, a separate package whose exceptions do not
# inherit from httpx's (found from a real timeout that escaped unmapped): catch both.
_TIMEOUTS = (httpx.TimeoutException, httpx2.TimeoutException)
_TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)

NEXT_MODEL_CODES = frozenset({429, 503, 504})
"""This model cannot serve the call now (overloaded, timed out, or its quota is used up) but
another may: Gemini's quotas are per model, and so is its load."""

_RETRY_DELAY = re.compile(r"^(\d+(?:\.\d+)?)s$")


class _Overloaded(Exception):
    def __init__(self, code: int, error: LLMTransientError | LLMPermanentError) -> None:
        self.code = code
        self.error = error


class GenerateContent(Protocol):
    """The one SDK call used: `client.aio.models`. A seam for tests (no network)."""

    def generate_content(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> Awaitable[types.GenerateContentResponse]: ...


class GeminiLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
        fallback_models: Sequence[str] = (),
        search_grounding: bool = True,
        models: GenerateContent | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GeminiLLM needs an API key")
        self.model = model
        self.fallback_models = tuple(m for m in dict.fromkeys(fallback_models) if m != model)
        """Tried in order, for one call, when the model before them is overloaded (503/504)."""
        self.search_grounding = search_grounding
        """False for accounts without Google Search grounding quota: research then runs
        without search and reports no sources (and so low confidence), instead of failing."""
        self._timeout_seconds = timeout_seconds
        self._secret = api_key
        if models is None:
            client = Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=round(timeout_seconds * 1000),  # the SDK takes milliseconds
                    retry_options=types.HttpRetryOptions(attempts=1),  # we retry, not the SDK
                ),
            )
            # Keep the Client itself, not only `client.aio.models`: when the Client is
            # garbage-collected its __del__ closes the HTTP connection, and every later call
            # would fail with "client has been closed".
            self._client: Client | None = client
            models = client.aio.models
        else:
            self._client = None
        self._models = models

    def __repr__(self) -> str:  # never the key
        return f"GeminiLLM(model={self.model!r}, timeout_seconds={self._timeout_seconds})"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        search = request.use_search and self.search_grounding
        config = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            tools=[types.Tool(google_search=types.GoogleSearch())] if search else None,
        )
        chain = (self.model, *self.fallback_models)
        for index, model in enumerate(chain):
            try:
                return await self._generate_once(model, request, config)
            except _Overloaded as overloaded:
                if index == len(chain) - 1:
                    raise overloaded.error from None
                logger.warning(
                    "Gemini model %s unavailable (%s); trying %s",
                    model,
                    overloaded.code,
                    chain[index + 1],
                )
        raise AssertionError("unreachable: the last model either answers or raises")

    async def _generate_once(
        self, model: str, request: LLMRequest, config: types.GenerateContentConfig
    ) -> LLMResponse:
        try:
            response = await self._models.generate_content(
                model=model, contents=request.user_prompt, config=config
            )
        except errors.APIError as exc:
            error = self._api_error(exc, model, searched=config.tools is not None)
            if (exc.code or 0) in NEXT_MODEL_CODES:
                raise _Overloaded(exc.code or 0, error) from None
            raise error from None
        except _TIMEOUTS:
            raise LLMTransientError(
                f"Gemini did not answer within {self._timeout_seconds:g}s (GEMINI_TIMEOUT_SECONDS)"
            ) from None
        except _TRANSPORT_ERRORS as exc:
            raise LLMTransientError(f"could not reach Gemini: {type(exc).__name__}") from None
        except (errors.UnknownApiResponseError, ValidationError) as exc:
            raise LLMTransientError(
                f"Gemini returned a response that could not be parsed: {type(exc).__name__}"
            ) from None
        return self._to_response(response, model)

    def _to_response(self, response: types.GenerateContentResponse, model: str) -> LLMResponse:
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            raise LLMPermanentError(f"Gemini blocked the prompt ({feedback.block_reason.name})")
        if not response.candidates:
            raise LLMTransientError("Gemini returned no candidates")
        candidate = response.candidates[0]
        reason = candidate.finish_reason
        if reason in BLOCKED_FINISH_REASONS:
            assert reason is not None
            raise LLMPermanentError(f"Gemini stopped the answer ({reason.name})")

        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        # Thought summaries are not part of the answer (they are not requested anyway).
        text = "".join(p.text for p in parts if p.text and not p.thought)
        if not text.strip():
            if reason == types.FinishReason.MAX_TOKENS:
                # On thinking models, thinking counts against max_output_tokens: the same
                # budget would run out the same way, so retrying cannot help.
                raise LLMPermanentError(
                    "Gemini used the whole max_output_tokens budget before writing any text"
                )
            raise LLMTransientError("Gemini returned an empty answer")
        if reason == types.FinishReason.MAX_TOKENS:
            logger.warning("Gemini answer cut at max_output_tokens (%d characters)", len(text))

        return LLMResponse(
            text=text,
            usage=_usage(response.usage_metadata),
            model=response.model_version or model,
            sources=_sources(candidate.grounding_metadata),
        )

    def _api_error(
        self, exc: errors.APIError, model: str, *, searched: bool = False
    ) -> LLMTransientError | LLMPermanentError:
        code = exc.code or 0
        detail = self._safe(f"{exc.status or ''} {exc.message or ''}".strip())
        lowered = detail.lower()
        if code in (401, 403) or (code == 400 and "api key" in lowered):
            return LLMPermanentError(f"Gemini rejected the API key ({code}); check GEMINI_API_KEY")
        if code == 404:
            return LLMPermanentError(
                f"Gemini model {model!r} was not found ({code}); check GEMINI_MODEL: {detail}"
            )
        if code == 429 and (daily := _daily_quota(exc.details)):
            # The model's per-day quota is used up: it resets in hours, so retrying in seconds
            # cannot help (seen for real: GenerateRequestsPerDayPerProjectPerModel-FreeTier,
            # retryDelay 52691s). Another model has its own quota: see NEXT_MODEL_CODES.
            wait = _retry_after(exc.details)
            resets = f"; it resets in about {wait / 3600:.0f}h" if wait else ""
            return LLMPermanentError(
                f"Gemini daily quota for {model!r} is used up (429, {daily}){resets}; "
                "use another model (GEMINI_MODEL, GEMINI_FALLBACK_MODELS) or a paid plan"
            )
        if code == 429:
            # Retryable: a per-minute limit and a missing quota share this exact text, and the
            # per-minute one clears within a minute (seen on real responses, 2026-11-14). The
            # provider's RetryInfo, when present, says how long to wait.
            hint = (
                "; if this persists for search requests, the account may have no Google Search "
                "grounding quota: set GEMINI_SEARCH_GROUNDING=false"
                if searched
                else ""
            )
            return LLMTransientError(
                f"Gemini rate limit or quota reached (429){hint}: {detail}",
                retry_after=_retry_after(exc.details),
            )
        if code in (408, 499) or code >= 500:
            return LLMTransientError(f"Gemini server error ({code}): {detail}")
        if 400 <= code < 500:
            return LLMPermanentError(f"Gemini rejected the request ({code}): {detail}")
        return LLMTransientError(f"Gemini error ({code or 'no status'}): {detail}")

    def _safe(self, text: str) -> str:
        """Error text as it may be logged and stored: never the key, never unbounded."""
        return text.replace(self._secret, "[redacted]")[:_MAX_ERROR_TEXT]


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> TokenUsage:
    """Tokens as the provider reports them; zero if it reports none.
    Thinking tokens are billed as output, so they count as output here."""
    if metadata is None:
        return TokenUsage()
    output = (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0)
    return TokenUsage(input_tokens=metadata.prompt_token_count or 0, output_tokens=output)


def _sources(metadata: types.GroundingMetadata | None) -> tuple[Source, ...]:
    """The pages Google Search grounding used, as the provider reports them (DECISIONS.md 0.1:
    sources come from the provider, never from the model's text)."""
    if metadata is None or not metadata.grounding_chunks:
        return ()
    sources = []
    for chunk in metadata.grounding_chunks:
        web = chunk.web
        if web is not None and web.uri:
            sources.append(Source(title=web.title or web.domain or web.uri, url=web.uri))
    return tuple(sources)


def _retry_after(details: Any) -> float | None:
    """`RetryInfo.retryDelay` ("31s") from an error payload, in seconds, if one was sent."""
    for entry in _details(details):
        if str(entry.get("@type", "")).endswith("RetryInfo"):
            match = _RETRY_DELAY.match(str(entry.get("retryDelay", "")))
            if match:
                return float(match[1])
    return None


def _details(details: Any) -> list[dict[str, Any]]:
    error = details.get("error") if isinstance(details, dict) else None
    entries = error.get("details") if isinstance(error, dict) else None
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _daily_quota(details: Any) -> str | None:
    """The id of a per-day quota the error says was exceeded, if any (QuotaFailure)."""
    for entry in _details(details):
        if str(entry.get("@type", "")).endswith("QuotaFailure"):
            for violation in entry.get("violations") or []:
                quota = str(violation.get("quotaId", "")) if isinstance(violation, dict) else ""
                if "PerDay" in quota:
                    return quota
    return None
