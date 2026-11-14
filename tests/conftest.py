"""Shared fixtures.

Database tests run against a real Postgres (JSONB, ARRAY, CHECK and CASCADE behave
differently elsewhere), in a separate `orchestrator_test` database that is dropped,
recreated and migrated with Alembic at the start of every run. So every run also tests
the migration from scratch. Each test starts from empty tables.

Unit tests never request these fixtures, so they run even with Postgres down.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fakes import ManualRunner
from httpx import ASGITransport, AsyncClient
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from orchestrator.config import Settings
from orchestrator.db.session import create_sessionmaker
from orchestrator.llm.fake import FakeLLM
from orchestrator.main import create_app

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://orchestrator:orchestrator@localhost:5433/orchestrator_test",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


async def recreate_database(database_url: str) -> None:
    target = make_url(database_url)
    admin = create_async_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as connection:
            await connection.execute(
                text(f'DROP DATABASE IF EXISTS "{target.database}" WITH (FORCE)')
            )
            await connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        await admin.dispose()


def _live_key() -> str:
    key = Settings().gemini_api_key  # the environment, then .env: as the app would see it
    return key.get_secret_value() if key is not None else ""


LIVE_GEMINI_API_KEY = _live_key()
"""Read once, at import, before the fixture below blanks it: only tests marked `live` may use
it (tests/live). Never printed: test ids and skip reasons do not contain it."""


@pytest.fixture(autouse=True)
def no_real_llm_unless_live(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No test reaches a real model by accident. An empty environment variable wins over a
    key in .env (pydantic-settings reads the environment first), so any Settings() built in a
    test sees no key and falls back to UnconfiguredLLM. Live tests opt in explicitly."""
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")


@pytest.fixture(scope="session")
def database_url() -> str:
    try:
        asyncio.run(recreate_database(TEST_DATABASE_URL))
    except OSError as exc:
        pytest.fail(
            f"Postgres for tests is unreachable ({exc}). Start it: docker compose up -d postgres"
        )
    command.upgrade(alembic_config(TEST_DATABASE_URL), "head")
    return TEST_DATABASE_URL


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE tasks CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_sessionmaker(engine)


@pytest.fixture
def llm() -> FakeLLM:
    """The app's LLM in API tests. Empty: a test that lets planning run scripts it first."""
    return FakeLLM()


@pytest.fixture
def runner() -> ManualRunner:
    """The app's background runner in API tests: nothing runs until `await runner.run_all()`."""
    return ManualRunner()


@pytest.fixture
async def client(
    database_url: str, engine: AsyncEngine, llm: FakeLLM, runner: ManualRunner
) -> AsyncIterator[AsyncClient]:
    """HTTP client wired to an app that uses the test database. `engine` truncates first.
    Lifespan events do not run under ASGITransport, so startup recovery is tested directly."""
    app = create_app(Settings(database_url=database_url), llm=llm, runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    await app.state.engine.dispose()
