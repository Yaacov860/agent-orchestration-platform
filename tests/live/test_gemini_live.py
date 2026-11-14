"""Real calls to the Gemini API, with the app's own configuration (key, model, fallbacks,
grounding switch, from the environment or .env). Never part of a plain `pytest` run:

    uv run pytest -m live tests/live -v

They cost a few cents and need the network. Skipped, with the reason, without a key.
"""

import pytest
from conftest import LIVE_GEMINI_API_KEY

from orchestrator.config import Settings
from orchestrator.llm import LLMPermanentError, LLMRequest
from orchestrator.llm.factory import build_llm
from orchestrator.llm.gemini import GeminiLLM

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_GEMINI_API_KEY, reason="GEMINI_API_KEY is not set: no real Gemini call made"
    ),
]


def live() -> GeminiLLM:
    llm = build_llm(Settings())
    assert isinstance(llm, GeminiLLM)
    return llm


async def test_a_plain_request_returns_text_and_usage() -> None:
    response = await live().generate(
        LLMRequest(
            system_prompt="Answer in one word.",
            user_prompt="What colour is a clear daytime sky?",
            temperature=0.0,
            max_output_tokens=512,
        )
    )

    assert response.text.strip()
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0
    assert response.model


async def test_search_grounding_returns_provider_sources() -> None:
    llm = live()
    if not llm.search_grounding:
        pytest.skip("GEMINI_SEARCH_GROUNDING=false: grounding deliberately off")
    request = LLMRequest(
        system_prompt="Answer briefly, using web search.",
        user_prompt="What is the latest stable Python release?",
        max_output_tokens=1024,
        use_search=True,
    )
    try:
        response = await llm.generate(request)
    except LLMPermanentError as exc:
        if "quota exhausted" in str(exc):
            pytest.skip(f"this account has no Google Search grounding quota: {exc}")
        raise

    assert response.text.strip()
    # Grounding is the model's choice per query, so sources may be empty; when present they
    # must be the provider's URLs.
    assert all(s.url.startswith("http") for s in response.sources)


async def test_a_bad_key_is_an_authentication_error_and_does_not_echo_the_key() -> None:
    bad_key = "AIza-deliberately-invalid-key-for-tests"
    llm = GeminiLLM(api_key=bad_key, model=live().model, timeout_seconds=30)

    with pytest.raises(LLMPermanentError) as caught:
        await llm.generate(LLMRequest(system_prompt="s", user_prompt="hi"))

    assert "check GEMINI_API_KEY" in str(caught.value)
    assert bad_key not in str(caught.value)
