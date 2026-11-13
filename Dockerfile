# --- build stage: resolve dependencies with uv into a venv ---
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first, so code edits don't invalidate this cached layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN uv sync --frozen --no-dev

# --- runtime stage: only the venv and the code, no uv, no build tools ---
FROM python:3.12-slim
RUN useradd --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
# Migrate, then hand the process over to uvicorn (exec) so it receives SIGTERM directly.
# Fine for one instance; with several replicas, migrations belong in a separate deploy step.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000"]
