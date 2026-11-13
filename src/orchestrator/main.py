from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.api.errors import register_error_handlers
from orchestrator.api.tasks import router as tasks_router
from orchestrator.config import Settings, get_settings
from orchestrator.db.session import create_engine, create_sessionmaker


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory: tests build a fresh app (with their own database) per test."""
    settings = settings or get_settings()
    # Creating an engine does not connect; the first query does.
    engine = create_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await engine.dispose()  # close pooled connections on shutdown

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.sessionmaker = create_sessionmaker(engine)

    register_error_handlers(app)
    app.include_router(tasks_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
