import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.agents.catalog import AgentCatalog, default_catalog
from orchestrator.agents.factory import AgentFactory
from orchestrator.api.errors import DATABASE_UNAVAILABLE, register_error_handlers
from orchestrator.api.tasks import router as tasks_router
from orchestrator.config import Settings, get_settings
from orchestrator.db.session import create_engine, create_sessionmaker
from orchestrator.llm import LLMClient
from orchestrator.llm.factory import build_llm, describe_llm
from orchestrator.planning import Planner
from orchestrator.runner import BackgroundRunner, Runner
from orchestrator.services.execution import ExecutionService
from orchestrator.services.planning import PlanningService
from orchestrator.services.synthesis import SynthesisService
from orchestrator.services.tasks import CancellationFanout
from orchestrator.synthesis import Synthesizer

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    catalog: AgentCatalog | None = None,
    runner: Runner | None = None,
) -> FastAPI:
    """App factory. Every collaborator can be injected: tests pass their own database, a
    scripted FakeLLM and a runner they drive by hand; production builds them from settings."""
    settings = settings or get_settings()
    # Creating an engine does not connect; the first query does.
    engine = create_engine(settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    runner = runner or BackgroundRunner()
    llm = llm or build_llm(settings)
    catalog = catalog or default_catalog()
    # Each stage hands the next one a task id through a callback, so planning, execution
    # and synthesis never import each other: they meet only here and in the database.
    synthesis = SynthesisService(
        sessionmaker,
        Synthesizer(llm, max_chars_per_step=settings.max_context_chars_per_input),
        runner,
        timeout=settings.synthesis_timeout_seconds,
        max_concurrent=settings.max_concurrent_syntheses,
    )
    execution = ExecutionService(
        sessionmaker,
        AgentFactory(llm),
        runner,
        max_concurrent_steps=settings.max_concurrent_steps,
        retry_delays=settings.step_retry_delays,
        step_timeout=settings.step_timeout_seconds,
        max_context_chars=settings.max_context_chars_per_input,
        on_executed=synthesis.schedule,
    )
    planning = PlanningService(
        sessionmaker,
        Planner(llm, catalog),
        runner,
        max_concurrent=settings.max_concurrent_plans,
        on_planned=execution.schedule,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await execution.recover()
            await synthesis.recover()
            await planning.recover()
        except DATABASE_UNAVAILABLE as exc:
            # Same stance as requests (503, not a dead process): start anyway. The stuck
            # tasks stay as they are until a startup that can reach the database.
            logger.error("startup recovery skipped, database unavailable: %s", exc)
        yield
        # Cancel background work first: it may still need the connections disposed below.
        await runner.shutdown()
        await engine.dispose()  # close pooled connections

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.sessionmaker = sessionmaker
    app.state.runner = runner
    app.state.planning = planning
    app.state.execution = execution
    app.state.synthesis = synthesis
    app.state.cancellation = CancellationFanout(execution, synthesis)
    app.state.llm_info = describe_llm(llm)

    register_error_handlers(app)
    app.include_router(tasks_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
