# Agent Orchestration Platform

An API that takes a complex task, has a planner break it into a dependency graph of steps,
runs each step on a specialized agent (research, writing, analysis, code), runs independent
steps in parallel, and synthesizes the outputs into one result with provenance.

> Work in progress. Built stage by stage; see `DECISIONS.md` for the reasoning behind each choice.

## Run

```bash
docker compose up --build
curl http://localhost:8000/health   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

## Test

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```
