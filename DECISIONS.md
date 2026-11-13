# Design Decisions

_Filled in as each part is built._

## 0. Foundations

**Runtime:** Python 3.12, FastAPI, Pydantic v2, managed with `uv` (lockfile committed).

**Execution model:** asyncio inside the API process, not Celery/Redis. LLM calls are network-bound,
which is exactly what asyncio handles well, and a separate broker would add a moving part without
adding capability at this scale. Trade-off: a process crash stops in-flight tasks. State lives in
PostgreSQL, so on startup interrupted tasks are marked `failed` with a reason rather than left hanging.

**Persistence:** PostgreSQL (JSONB for plans and outputs) with SQLAlchemy 2 async + Alembic.

**LLM access:** one `LLMClient` protocol with a real provider and a scripted `FakeLLM`. Tests never
call a real model, so they are fast, free and deterministic.

## 1. Task Decomposition Strategy

## 2. Dependency Management

## 3. Parallel Execution

## 4. Failure Recovery

## 5. One Thing I Would Do Differently With More Time
