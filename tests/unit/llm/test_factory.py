import pytest
from pydantic import SecretStr, ValidationError

from orchestrator.config import Settings
from orchestrator.llm import LLMPermanentError, LLMRequest
from orchestrator.llm.factory import build_llm
from orchestrator.llm.gemini import GeminiLLM
from orchestrator.llm.unconfigured import (
    MISSING_GEMINI_KEY_MESSAGE,
    NO_PROVIDER_MESSAGE,
    UnconfiguredLLM,
)

REQUEST = LLMRequest(system_prompt="s", user_prompt="u")
KEY = "AIza-not-a-real-key-0123456789"


def settings(**values: object) -> Settings:
    """Explicit values only; the autouse fixture has already blanked any real key."""
    return Settings.model_validate(values)


def test_gemini_is_the_default_provider() -> None:
    assert Settings.model_fields["llm_provider"].default == "gemini"
    assert Settings.model_fields["gemini_model"].default == "gemini-3.8-flash"


def test_gemini_with_a_key_builds_the_gemini_client_without_any_network_call() -> None:
    llm = build_llm(
        settings(gemini_api_key=KEY, gemini_model="gemini-x", gemini_timeout_seconds=12)
    )

    assert isinstance(llm, GeminiLLM)
    assert (llm.model, repr(llm)) == (
        "gemini-x",
        "GeminiLLM(model='gemini-x', timeout_seconds=12.0)",
    )


@pytest.mark.parametrize("key", [None, "", "   "])
async def test_gemini_without_a_key_still_starts_and_fails_tasks_clearly(key: str | None) -> None:
    llm = build_llm(settings(gemini_api_key=key))

    assert isinstance(llm, UnconfiguredLLM)
    with pytest.raises(LLMPermanentError) as caught:
        await llm.generate(REQUEST)
    assert str(caught.value) == MISSING_GEMINI_KEY_MESSAGE
    assert caught.value.retryable is False


@pytest.mark.parametrize("value", ["none", "NONE", "  none "])
async def test_none_builds_the_unconfigured_client_even_with_a_key(value: str) -> None:
    llm = build_llm(settings(llm_provider=value, gemini_api_key=KEY))

    assert isinstance(llm, UnconfiguredLLM)
    with pytest.raises(LLMPermanentError, match="LLM_PROVIDER=none"):
        await llm.generate(REQUEST)
    assert NO_PROVIDER_MESSAGE.startswith("no LLM provider is configured")


@pytest.mark.parametrize("value", ["fake", "openai", "", "gemini-pro"])
def test_an_unknown_provider_is_rejected_with_the_supported_ones(value: str) -> None:
    with pytest.raises(ValueError, match=r"unknown LLM_PROVIDER .*; supported: gemini, none"):
        build_llm(settings(llm_provider=value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gemini_model", ""),
        ("gemini_model", "   "),
        ("gemini_timeout_seconds", 0),
        ("gemini_timeout_seconds", 601),
    ],
)
def test_invalid_gemini_settings_are_refused_at_startup(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        settings(**{field: value})


def test_the_key_never_shows_in_settings_output() -> None:
    configured = settings(gemini_api_key=KEY)

    assert isinstance(configured.gemini_api_key, SecretStr)
    for rendered in (repr(configured), str(configured), configured.model_dump_json()):
        assert KEY not in rendered


def test_the_key_and_model_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_MODEL", " gemini-y ")

    from_env = Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings' own kwarg

    assert from_env.gemini_api_key is not None
    assert from_env.gemini_api_key.get_secret_value() == KEY
    assert from_env.gemini_model == "gemini-y"


def test_fallback_models_and_grounding_reach_the_client() -> None:
    llm = build_llm(
        settings(
            gemini_api_key=KEY,
            gemini_fallback_models=" gemini-b , ,gemini-c ",
            gemini_search_grounding=False,
        )
    )

    assert isinstance(llm, GeminiLLM)
    assert llm.fallback_models == ("gemini-b", "gemini-c")
    assert llm.search_grounding is False


def test_the_default_fallbacks_are_the_models_that_answered_when_checked() -> None:
    assert settings().gemini_fallbacks == ("gemini-3.6-flash", "gemini-3.5-flash-lite")
    assert settings(gemini_fallback_models="").gemini_fallbacks == ()
