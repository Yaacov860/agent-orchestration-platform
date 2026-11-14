"""GeminiLLM without any network: the SDK's real types, only `generate_content` replaced."""

import asyncio
import gc
from collections.abc import Sequence
from typing import Any

import httpx
import httpx2
import pytest
from google.genai import errors, types

from orchestrator.llm import LLMPermanentError, LLMRequest, LLMTransientError, Source
from orchestrator.llm.gemini import GeminiLLM

KEY = "AIza-test-secret-0123456789abcdef"
REQUEST = LLMRequest(
    system_prompt="You are a test.", user_prompt="Say hi.", temperature=0.4, max_output_tokens=321
)


def response(
    *texts: str,
    finish: types.FinishReason = types.FinishReason.STOP,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    grounding: types.GroundingMetadata | None = None,
    thought: str | None = None,
    blocked: types.BlockedReason | None = None,
    candidates: bool = True,
) -> types.GenerateContentResponse:
    parts = [types.Part(text=t) for t in texts]
    if thought is not None:
        parts.insert(0, types.Part(text=thought, thought=True))
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=finish,
                grounding_metadata=grounding,
            )
        ]
        if candidates
        else [],
        usage_metadata=usage,
        model_version="gemini-test-001",
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=blocked)
        if blocked
        else None,
    )


class FakeModels:
    """Stands in for `client.aio.models`: records calls, returns or raises scripted items."""

    def __init__(self, *items: types.GenerateContentResponse | BaseException) -> None:
        self._items = list(items)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        await asyncio.sleep(0)
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def gemini(*items: types.GenerateContentResponse | BaseException) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(*items)
    return GeminiLLM(api_key=KEY, model="gemini-test", timeout_seconds=30, models=models), models


def api_error(code: int, status: str, message: str) -> errors.APIError:
    payload = {"error": {"code": code, "message": message, "status": status}}
    kind = errors.ClientError if code < 500 else errors.ServerError
    return kind(code, payload, None)


# --- the request -------------------------------------------------------------------------


async def test_the_request_maps_onto_the_sdk_config() -> None:
    llm, models = gemini(response("hi"))

    await llm.generate(REQUEST)

    [call] = models.calls
    config: types.GenerateContentConfig = call["config"]
    assert (call["model"], call["contents"]) == ("gemini-test", "Say hi.")
    assert config.system_instruction == "You are a test."
    assert (config.temperature, config.max_output_tokens) == (0.4, 321)
    assert config.tools is None  # no search unless asked


async def test_use_search_turns_on_google_search_grounding() -> None:
    llm, models = gemini(response("hi"))

    await llm.generate(REQUEST.model_copy(update={"use_search": True}))

    tools = models.calls[0]["config"].tools
    assert tools is not None and len(tools) == 1
    assert tools[0].google_search is not None


def test_the_real_client_is_built_offline_with_our_timeout_and_no_sdk_retries() -> None:
    llm = GeminiLLM(api_key=KEY, model="gemini-test", timeout_seconds=12.5)

    # SDK internals, read only to check the two options this module depends on.
    sdk_models: Any = llm._models
    options: types.HttpOptions = sdk_models._api_client._http_options
    assert options.timeout == 12_500  # milliseconds
    assert options.retry_options is not None and options.retry_options.attempts == 1
    assert options.headers is not None
    assert options.headers["x-goog-api-key"] == KEY  # in a header, not in the URL


def test_a_blank_key_is_refused() -> None:
    with pytest.raises(ValueError, match="needs an API key"):
        GeminiLLM(api_key="  ", models=FakeModels())


def test_the_key_is_not_in_the_clients_repr() -> None:
    llm, _ = gemini()

    assert KEY not in repr(llm)
    assert KEY not in str(vars(llm).get("model"))


# --- the response ------------------------------------------------------------------------


async def test_text_usage_model_and_sources_are_read_from_the_response() -> None:
    grounding = types.GroundingMetadata(
        grounding_chunks=[
            types.GroundingChunk(
                web=types.GroundingChunkWeb(uri="https://docs.python.org", title="Python docs")
            ),
            types.GroundingChunk(
                web=types.GroundingChunkWeb(uri="https://mdn.dev", domain="mdn.dev")
            ),
            types.GroundingChunk(web=types.GroundingChunkWeb(title="no uri")),  # dropped
            types.GroundingChunk(),  # not a web chunk: dropped
        ]
    )
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=120, candidates_token_count=30, thoughts_token_count=50
    )
    llm, _ = gemini(
        response("Hello, ", "world.", usage=usage, grounding=grounding, thought="(thinking)")
    )

    result = await llm.generate(REQUEST)

    assert result.text == "Hello, world."  # thought parts are not the answer
    assert (result.usage.input_tokens, result.usage.output_tokens) == (
        120,
        80,
    )  # thinking billed as output
    assert result.model == "gemini-test-001"
    assert result.sources == (
        Source(title="Python docs", url="https://docs.python.org"),
        Source(title="mdn.dev", url="https://mdn.dev"),
    )


async def test_missing_usage_counts_as_zero_rather_than_failing() -> None:
    llm, _ = gemini(response("hi", usage=None))

    result = await llm.generate(REQUEST)

    assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)


async def test_an_answer_cut_at_max_tokens_is_returned_as_it_is() -> None:
    llm, _ = gemini(response('{"steps": [', finish=types.FinishReason.MAX_TOKENS))

    assert (await llm.generate(REQUEST)).text == '{"steps": ['  # callers' validation catches it


# --- invalid responses -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "kind", "message"),
    [
        (
            response(candidates=False, blocked=types.BlockedReason.SAFETY),
            LLMPermanentError,
            "Gemini blocked the prompt (SAFETY)",
        ),
        (
            response("x", finish=types.FinishReason.SAFETY),
            LLMPermanentError,
            "Gemini stopped the answer (SAFETY)",
        ),
        (
            response("x", finish=types.FinishReason.RECITATION),
            LLMPermanentError,
            "Gemini stopped the answer (RECITATION)",
        ),
        (response(candidates=False), LLMTransientError, "Gemini returned no candidates"),
        (response("  "), LLMTransientError, "Gemini returned an empty answer"),
        (
            response(finish=types.FinishReason.MAX_TOKENS),
            LLMPermanentError,
            "Gemini used the whole max_output_tokens budget before writing any text",
        ),
    ],
    ids=[
        "blocked prompt",
        "safety stop",
        "recitation",
        "no candidates",
        "empty",
        "budget spent thinking",
    ],
)
async def test_an_unusable_response_is_mapped_to_the_right_error(
    bad: types.GenerateContentResponse, kind: type[Exception], message: str
) -> None:
    llm, _ = gemini(bad)

    with pytest.raises(kind) as caught:
        await llm.generate(REQUEST)

    assert str(caught.value) == message


async def test_a_response_the_sdk_cannot_parse_is_retryable() -> None:
    llm, _ = gemini(errors.UnknownApiResponseError("unexpected shape"))

    with pytest.raises(LLMTransientError, match="could not be parsed: UnknownApiResponseError"):
        await llm.generate(REQUEST)


# --- API errors --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "status", "message", "retryable", "expected"),
    [
        (
            400,
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
            False,
            "rejected the API key (400)",
        ),
        (
            401,
            "UNAUTHENTICATED",
            "Request had invalid authentication credentials.",
            False,
            "rejected the API key (401)",
        ),
        (
            403,
            "PERMISSION_DENIED",
            "The caller does not have permission.",
            False,
            "rejected the API key (403)",
        ),
        (
            404,
            "NOT_FOUND",
            "models/gemini-test is not found.",
            False,
            "model 'gemini-test' was not found (404); check GEMINI_MODEL",
        ),
        (
            400,
            "INVALID_ARGUMENT",
            "Request contains an invalid argument.",
            False,
            "rejected the request (400)",
        ),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Resource has been exhausted.",
            True,
            "rate limit or quota reached (429)",
        ),
        (408, "DEADLINE_EXCEEDED", "Timed out.", True, "server error (408)"),
        (500, "INTERNAL", "Internal error.", True, "server error (500)"),
        (503, "UNAVAILABLE", "The model is overloaded.", True, "server error (503)"),
        (504, "DEADLINE_EXCEEDED", "Deadline exceeded.", True, "server error (504)"),
    ],
)
async def test_api_errors_are_mapped_and_say_whether_to_retry(
    code: int, status: str, message: str, retryable: bool, expected: str
) -> None:
    llm, _ = gemini(api_error(code, status, message))

    with pytest.raises((LLMTransientError, LLMPermanentError)) as caught:
        await llm.generate(REQUEST)

    assert caught.value.retryable is retryable
    assert expected in str(caught.value)


async def test_the_key_never_appears_in_an_error_even_if_the_server_echoes_it() -> None:
    llm, _ = gemini(api_error(400, "INVALID_ARGUMENT", f"bad request for key {KEY}"))

    with pytest.raises(LLMPermanentError) as caught:
        await llm.generate(REQUEST)

    assert KEY not in str(caught.value)
    assert "[redacted]" in str(caught.value)
    assert caught.value.__cause__ is None  # the SDK exception (and its payload) is not chained


async def test_a_long_server_message_is_cut() -> None:
    llm, _ = gemini(api_error(500, "INTERNAL", "x" * 5_000))

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST)

    assert len(str(caught.value)) < 400


# --- transport errors and timeouts -------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            httpx.ReadTimeout("read timed out"),
            "Gemini did not answer within 30s (GEMINI_TIMEOUT_SECONDS)",
        ),
        (
            httpx.ConnectTimeout("connect timed out"),
            "Gemini did not answer within 30s (GEMINI_TIMEOUT_SECONDS)",
        ),
        (httpx.ConnectError("name resolution failed"), "could not reach Gemini: ConnectError"),
        (httpx.RemoteProtocolError("peer closed"), "could not reach Gemini: RemoteProtocolError"),
        # What the SDK actually raises: httpx2, whose exceptions are not httpx's.
        (
            httpx2.ReadTimeout("read timed out"),
            "Gemini did not answer within 30s (GEMINI_TIMEOUT_SECONDS)",
        ),
        (httpx2.ConnectError("refused"), "could not reach Gemini: ConnectError"),
    ],
)
async def test_timeouts_and_network_failures_are_retryable(
    failure: Exception, message: str
) -> None:
    llm, _ = gemini(failure)

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST)

    assert str(caught.value) == message


async def test_cancellation_propagates() -> None:
    llm, _ = gemini(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await llm.generate(REQUEST)


async def test_concurrent_calls_share_one_client() -> None:
    llm, models = gemini(*[response(f"answer {n}") for n in range(5)])

    answers: Sequence[Any] = await asyncio.gather(*(llm.generate(REQUEST) for _ in range(5)))

    assert sorted(a.text for a in answers) == [f"answer {n}" for n in range(5)]
    assert len(models.calls) == 5


async def test_the_sdk_client_stays_open_for_the_life_of_the_provider() -> None:
    # Regression: holding only `client.aio.models` let the Client be garbage-collected, and its
    # __del__ schedules the HTTP connection's close on the running loop: every real call then
    # failed with "client has been closed". Fakes cannot catch this; only a real Client, a
    # forced collection and a turn of a running loop can (a sync test passes either way).
    llm = GeminiLLM(api_key=KEY, model="gemini-test")
    gc.collect()
    await asyncio.sleep(0.05)  # let any scheduled close run

    sdk_models: Any = llm._models
    http = sdk_models._api_client._async_httpx_client
    assert http is not None and not http.is_closed


# --- quota, model fallback, grounding switch (from the real responses of 2026-11-14) ----

QUOTA_TEXT = (
    "You exceeded your current quota, please check your plan and billing details. "
    "For more information on this error, head to: https://aistudio.google.com/docs/rate-limits."
)


async def test_a_quota_429_stays_retryable_because_per_minute_limits_share_its_text() -> None:
    # Real responses, 2026-11-14: plain requests got this exact 429 and succeeded a minute
    # later; it is not safe to treat the text as "quota gone for good".
    llm, _ = gemini(api_error(429, "RESOURCE_EXHAUSTED", QUOTA_TEXT))

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST)

    assert caught.value.retryable is True
    assert "GEMINI_SEARCH_GROUNDING" not in str(caught.value)


async def test_a_search_429_points_to_the_grounding_switch() -> None:
    llm, _ = gemini(api_error(429, "RESOURCE_EXHAUSTED", QUOTA_TEXT))

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST.model_copy(update={"use_search": True}))

    assert "no Google Search grounding quota: set GEMINI_SEARCH_GROUNDING=false" in str(
        caught.value
    )


@pytest.mark.parametrize(("delay", "expected"), [("31s", 31.0), ("0.5s", 0.5), ("soon", None)])
async def test_the_providers_retry_delay_is_read_when_it_sends_one(
    delay: str, expected: float | None
) -> None:
    payload = {
        "error": {
            "code": 429,
            "message": QUOTA_TEXT,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            ],
        }
    }
    llm, _ = gemini(errors.ClientError(429, payload, None))

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST)

    assert caught.value.retry_after == expected


async def test_no_retry_delay_when_the_provider_sends_none() -> None:
    llm, _ = gemini(api_error(429, "RESOURCE_EXHAUSTED", QUOTA_TEXT))

    with pytest.raises(LLMTransientError) as caught:
        await llm.generate(REQUEST)

    assert caught.value.retry_after is None


def overloaded(code: int = 503) -> errors.APIError:
    status = "UNAVAILABLE" if code == 503 else "DEADLINE_EXCEEDED"
    return api_error(code, status, "This model is currently experiencing high demand.")


def with_fallbacks(
    *items: types.GenerateContentResponse | BaseException,
) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(*items)
    llm = GeminiLLM(
        api_key=KEY, model="primary", fallback_models=("second", "third"), models=models
    )
    return llm, models


async def test_an_overloaded_model_falls_back_to_the_next_one_for_that_call() -> None:
    reply = response("ok")
    llm, models = with_fallbacks(overloaded(503), overloaded(504), reply)

    result = await llm.generate(REQUEST)

    assert [c["model"] for c in models.calls] == ["primary", "second", "third"]
    assert result.text == "ok"
    assert result.model == "gemini-test-001"  # the provider's own report of who answered


async def test_when_every_model_is_overloaded_the_last_error_is_raised_and_is_retryable() -> None:
    llm, models = with_fallbacks(overloaded(), overloaded(), overloaded(504))

    with pytest.raises(LLMTransientError, match="server error \\(504\\)"):
        await llm.generate(REQUEST)

    assert len(models.calls) == 3


@pytest.mark.parametrize(
    "failure",
    [api_error(401, "UNAUTHENTICATED", "bad credentials"), api_error(500, "INTERNAL", "internal")],
    ids=["auth", "500"],
)
async def test_errors_another_model_cannot_fix_do_not_fall_back(failure: errors.APIError) -> None:
    llm, models = with_fallbacks(failure)

    with pytest.raises((LLMTransientError, LLMPermanentError)):
        await llm.generate(REQUEST)

    assert [c["model"] for c in models.calls] == ["primary"]


def daily_quota() -> errors.APIError:
    """The shape of the real response of 2026-11-14 for an exhausted free-tier daily quota."""
    payload = {
        "error": {
            "code": 429,
            "message": QUOTA_TEXT,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "52691s"},
            ],
        }
    }
    return errors.ClientError(429, payload, None)


async def test_a_used_up_daily_quota_is_permanent_and_says_when_it_resets() -> None:
    llm, _ = gemini(daily_quota())

    with pytest.raises(LLMPermanentError) as caught:
        await llm.generate(REQUEST)

    message = str(caught.value)
    assert (
        "daily quota for 'gemini-test' is used up "
        "(429, GenerateRequestsPerDayPerProjectPerModel-FreeTier)" in message
    )
    assert "resets in about 15h" in message


async def test_a_model_out_of_daily_quota_falls_back_to_one_that_has_quota() -> None:
    llm, models = with_fallbacks(daily_quota(), response("answered by the second model"))

    result = await llm.generate(REQUEST)

    assert [c["model"] for c in models.calls] == ["primary", "second"]
    assert result.text == "answered by the second model"


async def test_when_every_model_is_out_of_daily_quota_the_failure_is_permanent() -> None:
    llm, _ = with_fallbacks(daily_quota(), daily_quota(), daily_quota())

    with pytest.raises(LLMPermanentError, match="daily quota for 'third' is used up"):
        await llm.generate(REQUEST)


async def test_a_per_minute_limit_falls_back_and_stays_retryable_if_all_are_limited() -> None:
    limited = api_error(429, "RESOURCE_EXHAUSTED", QUOTA_TEXT)
    llm, models = with_fallbacks(limited, api_error(429, "RESOURCE_EXHAUSTED", QUOTA_TEXT), limited)

    with pytest.raises(LLMTransientError):
        await llm.generate(REQUEST)

    assert len(models.calls) == 3


async def test_a_missing_fallback_model_is_named_in_the_error() -> None:
    llm, _ = with_fallbacks(overloaded(), api_error(404, "NOT_FOUND", "no such model"))

    with pytest.raises(LLMPermanentError, match="model 'second' was not found"):
        await llm.generate(REQUEST)


def test_fallbacks_skip_duplicates_and_the_primary_model() -> None:
    llm = GeminiLLM(
        api_key=KEY, model="a", fallback_models=("a", "b", "b", "c"), models=FakeModels()
    )

    assert llm.fallback_models == ("b", "c")


async def test_with_grounding_off_search_requests_run_without_the_search_tool() -> None:
    models = FakeModels(response("from training data only"))
    llm = GeminiLLM(api_key=KEY, model="m", search_grounding=False, models=models)

    result = await llm.generate(REQUEST.model_copy(update={"use_search": True}))

    assert models.calls[0]["config"].tools is None
    assert result.sources == ()  # nothing grounded, so nothing claimed
