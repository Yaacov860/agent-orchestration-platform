from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables (or a local .env file).

    Every limit is bounded: a value outside its range (MAX_CONCURRENT_STEPS=0, a negative
    timeout) stops the app at startup with the field named, instead of misbehaving later.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "agent-orchestration-platform"
    database_url: str = "postgresql+asyncpg://orchestrator:orchestrator@localhost:5433/orchestrator"
    llm_provider: str = "gemini"
    """See llm/factory.py: "gemini", or "none". Without GEMINI_API_KEY (or with "none") the API
    still runs, and planning fails with a clear error."""
    gemini_api_key: SecretStr | None = None
    """Never logged or shown: SecretStr masks it in repr and in model dumps."""
    gemini_model: str = Field(default="gemini-3.8-flash", min_length=1, max_length=100)
    gemini_fallback_models: str = "gemini-3.6-flash,gemini-3.5-flash-lite"
    """Comma-separated. Tried in order for a call when the model before them is overloaded
    (503/504). Both answered for this project's key on 2026-11-14 while gemini-3.8-flash
    returned 503; empty disables fallback."""
    gemini_search_grounding: bool = False
    """Off by default: Google Search grounding is billed per search query and needs quota on
    the key (a free-tier key has none: 429 "check your plan and billing"). Off, research steps
    run without search, are labelled unverified and report no sources. On, only research
    steps search, within GEMINI_MAX_GROUNDED_CALLS_PER_TASK."""
    gemini_max_grounded_calls_per_task: int = Field(default=2, ge=0, le=50)
    """A hard limit on provider calls sent with web search attached, per task, over all its
    research steps, retries and model fallbacks (0: none at all). Not a limit on Google
    queries: the model decides how many one call runs, and they are only counted afterwards."""
    gemini_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    """One HTTP call to Gemini. Below STEP_TIMEOUT_SECONDS, so a hung call fails as a provider
    timeout (and is retried) before the step timeout has to stop it."""

    # --- limits (see DECISIONS.md 3 and 4) ---
    max_concurrent_plans: int = Field(default=4, ge=1, le=64)
    """Plans computed at once by this process."""
    max_concurrent_steps: int = Field(default=4, ge=1, le=64)
    """Agent runs at once by this process, across all tasks (the LLM's rate limit is shared)."""
    step_timeout_seconds: float = Field(default=180.0, gt=0, le=3_600)
    """One attempt of one step; a step that runs longer counts as a retryable timeout."""
    step_max_attempts: int = Field(default=3, ge=1, le=10)
    """Attempts per step, the first included. 1 means never retry."""
    step_retry_base_delay_seconds: float = Field(default=1.0, ge=0, le=60)
    """Wait before the first retry; each further retry waits 3 times longer."""
    max_context_chars_per_input: int = Field(default=8_000, ge=200, le=200_000)
    """Characters of one step's output given to a dependent step, and to synthesis (about 4
    per token). Longer outputs are cut, with a marker; the stored output stays complete."""
    max_concurrent_syntheses: int = Field(default=4, ge=1, le=64)
    """Syntheses computed at once by this process."""
    synthesis_timeout_seconds: float = Field(default=120.0, gt=0, le=3_600)
    """The whole synthesis of one task, its retries and repair included."""

    @field_validator("gemini_api_key")
    @classmethod
    def blank_key_is_no_key(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value is not None and value.get_secret_value().strip() else None

    @field_validator("gemini_model")
    @classmethod
    def model_name_is_trimmed(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("GEMINI_MODEL must not be blank")
        return value.strip()

    @property
    def gemini_fallbacks(self) -> tuple[str, ...]:
        return tuple(m.strip() for m in self.gemini_fallback_models.split(",") if m.strip())

    @property
    def step_retry_delays(self) -> tuple[float, ...]:
        """(1.0, 3.0) with the defaults: one delay per retry, growing 3x each time."""
        base = self.step_retry_base_delay_seconds
        return tuple(base * 3**retry for retry in range(self.step_max_attempts - 1))


@lru_cache
def get_settings() -> Settings:
    return Settings()
