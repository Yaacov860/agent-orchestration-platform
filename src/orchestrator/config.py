from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "agent-orchestration-platform"
    database_url: str = "postgresql+asyncpg://orchestrator:orchestrator@localhost:5433/orchestrator"
    llm_provider: str = "fake"


@lru_cache
def get_settings() -> Settings:
    return Settings()
