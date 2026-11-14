import pytest
from pydantic import ValidationError

from orchestrator.config import Settings
from orchestrator.execution.executor import (
    MAX_CONTEXT_CHARS,
    STEP_RETRY_DELAYS,
    STEP_TIMEOUT_SECONDS,
)
from orchestrator.services.execution import MAX_CONCURRENT_STEPS
from orchestrator.services.planning import MAX_CONCURRENT_PLANS


def settings(**values: object) -> Settings:
    """Validate explicit values. Explicit values win over the environment, so only fields
    a test does not pass could come from the developer's machine; tests pass all they check."""
    return Settings.model_validate(values)


def defaults() -> Settings:
    """The declared defaults only: model_construct reads no environment and no .env
    (unlike Settings() and even model_validate, which both do)."""
    return Settings.model_construct()


def test_settings_defaults_match_the_services_own_defaults() -> None:
    # Two places hold each default (Settings for the app, the constants for code that builds
    # a service directly, such as tests); this keeps them from drifting apart.
    default = defaults()
    assert default.max_concurrent_plans == MAX_CONCURRENT_PLANS
    assert default.max_concurrent_steps == MAX_CONCURRENT_STEPS
    assert default.step_timeout_seconds == STEP_TIMEOUT_SECONDS
    assert default.step_retry_delays == STEP_RETRY_DELAYS
    assert default.max_context_chars_per_input == MAX_CONTEXT_CHARS


@pytest.mark.parametrize(
    ("attempts", "base", "delays"),
    [(3, 1.0, (1.0, 3.0)), (1, 1.0, ()), (4, 0.5, (0.5, 1.5, 4.5)), (2, 0.0, (0.0,))],
)
def test_retry_delays_follow_from_attempts_and_base(
    attempts: int, base: float, delays: tuple[float, ...]
) -> None:
    configured = settings(step_max_attempts=attempts, step_retry_base_delay_seconds=base)
    assert configured.step_retry_delays == delays


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_concurrent_steps", 0),
        ("max_concurrent_plans", 0),
        ("step_timeout_seconds", 0),
        ("step_timeout_seconds", -5),
        ("step_max_attempts", 0),
        ("step_retry_base_delay_seconds", -1),
        ("max_context_chars_per_input", 10),
    ],
)
def test_an_out_of_range_limit_is_refused_naming_the_field(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        settings(**{field: value})


def test_limits_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_STEPS", "2")
    monkeypatch.setenv("STEP_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("STEP_MAX_ATTEMPTS", "1")

    from_env = Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings' own kwarg

    assert (from_env.max_concurrent_steps, from_env.step_timeout_seconds) == (2, 30.0)
    assert from_env.step_retry_delays == ()


def test_the_defaults_used_here_ignore_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_STEPS", "7")

    assert defaults().max_concurrent_steps == 4
