from fastapi import FastAPI

from orchestrator.config import get_settings


def create_app() -> FastAPI:
    """App factory: tests build a fresh app per test instead of sharing a module-level global."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
