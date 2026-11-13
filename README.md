# Agent Orchestration Platform

An API that takes a complex task, has a planner break it into a dependency graph of steps,
runs each step on a specialized agent (research, writing, analysis, code), runs independent
steps in parallel, and synthesizes the outputs into one result with provenance.

> Work in progress. Built stage by stage; see `DECISIONS.md` for the reasoning behind each choice.

## Run

```bash
docker compose up --build        # API on :8000, Postgres on 127.0.0.1:5433; migrations run on start
curl http://localhost:8000/health   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

## Use

Submit a task:

```bash
curl -i -X POST localhost:8000/tasks -H 'content-type: application/json' -d '{
  "goal": "Write a comparison blog post about Python vs JavaScript for beginners",
  "constraints": {"max_words": 1500, "tone": "friendly", "include_code_examples": true},
  "output_format": "markdown"
}'
# 201 Created
# location: /tasks/task_952e7faa8f2d42168baf28e712bc95b9
# {"task_id":"task_952e7faa8f2d42168baf28e712bc95b9","status":"pending"}
```

Track it:

```bash
curl localhost:8000/tasks/task_952e7faa8f2d42168baf28e712bc95b9
# {"task_id": "...", "status": "pending", "goal": "...", "constraints": {...},
#  "progress": {"total_steps": 0, "completed_steps": 0, "failed_steps": 0,
#               "current_step": null, "running_steps": []}, ...}
```

| Field | Rules |
|---|---|
| `goal` | required, 1-4000 characters after trimming |
| `constraints` | optional; `max_words` (int 1-20000), `tone` (string), `include_code_examples` (bool), plus any other JSON values |
| `output_format` | `markdown` (default), `text` or `json` |
| `deadline` | optional ISO-8601 time with a timezone, in the future |

Errors: `422` invalid input (every problem listed), `404` unknown task, `503` database unavailable.

## Test

Unit tests need nothing. Integration tests need the Postgres container
(`docker compose up -d postgres`); they create and migrate their own `orchestrator_test` database.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
```
