# Agent Orchestration Platform

An API that takes a complex task, has a planner break it into a dependency graph of steps,
runs each step on a specialized agent (research, writing, analysis, code), runs independent
steps in parallel, and synthesizes the outputs into one result with provenance.

See `DECISIONS.md` for the reasoning behind each choice, and `AI_USAGE.md` for how the system
uses an LLM and how it was built with an AI assistant.

## Run

```bash
docker compose up --build        # API on :8000, Postgres on 127.0.0.1:5433; migrations run on start
curl http://localhost:8000/health   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

## Flow

```
POST /tasks ─► planning ─► planned ─► executing ─► synthesizing ─► completed ─► GET /tasks/{id}/result
                  │                       │              │
                  └──────────── failed (failure_stage: planning | execution | synthesis) ──┘
             any unfinished stage ─► cancelled (POST /tasks/{id}/cancel)
```

1. **Planning**: the planner turns the goal into a dependency graph of steps and gives each
   step the agent and action that fit it, chosen from the agents' own descriptions and
   checked against the agents that exist.
2. **Execution**: the steps run on their agents, group by group, independent ones in parallel.
3. **Synthesis**: the step outputs are combined into one answer. The model writes the answer,
   its key findings (each citing the steps it came from) and the conflicts between steps.
   Code adds the sources the agents actually returned, whether each finding is corroborated,
   a confidence that states its basis, and the limitations (what failed, what was cut).
4. **Completed**: set in the same transaction that stores the result, never before.

## API

| Endpoint | What it returns |
|---|---|
| `POST /tasks` | `201` with `task_id`; planning, execution and synthesis follow in the background |
| `GET /tasks/{id}` | status, `failure_stage`, `error`, `progress`, `usage` (tokens by stage) and `search` (grounded calls, queries) |
| `GET /tasks/{id}/steps` | every step of the newest plan: state, attempts, full output or error, tokens, `search`, timings; research outputs also carry `grounding` |
| `GET /tasks/{id}/trace` | the timeline (status changes, step starts and ends, result stored), stage durations, usage, `search` |
| `GET /tasks/{id}/result` | the final answer with sources, findings, conflicts, limitations; `409` while there is none |
| `POST /tasks/{id}/cancel` | stops an unfinished task; `409` if it already finished |
| `GET /agents` | every agent, what each action does, and whether it searches the web |
| `GET /metrics` | tasks and steps by status, tokens by stage, `search` totals, background jobs, provider, uptime |
| `GET /health` | liveness: the process is up |
| `GET /health/ready` | readiness: the database answers and a provider is configured (`503` otherwise) |

OpenAPI: `/docs` and `/openapi.json`.

## Agents

| Agent | Actions | Works from | Sources |
|---|---|---|---|
| `research` | `research` (web search, when grounding is on), `summarize` | the web; earlier steps | yes: the pages its search used |
| `writing` | `write`, `rewrite`, `draft_email` | earlier steps (facts), constraints (tone, length) | none |
| `analysis` | `analyze`, `compare`, `evaluate` | earlier steps or data in the task; invents none | none |
| `code` | `write_code`, `explain_code`, `review_code`, `fix_bug` | the step and earlier steps; text only, runs nothing | none |

Every agent returns the same result shape (output, error, tokens, time). Only research
searches and cites; the others are told to invent no facts or sources. `write_code` and
`fix_bug` must return a fenced code block, or the attempt counts as a retryable failure. The
code agent never runs code or touches files.

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

The `201` comes back at once; the task is planned and then executed in the background.
Track it:

```bash
curl localhost:8000/tasks/task_952e7faa8f2d42168baf28e712bc95b9
# {"task_id": "...", "status": "executing", "goal": "...", "constraints": {...}, "error": null,
#  "progress": {"total_steps": 3, "completed_steps": 0, "failed_steps": 0,
#               "current_step": "step_1", "running_steps": ["step_1", "step_2"]}, ...}
```

See every step: its state, attempts, full output or error, tokens and timings, in plan order
(`steps` is empty and `plan_version` is `null` until a plan is stored):

```bash
curl localhost:8000/tasks/task_952e7faa8f2d42168baf28e712bc95b9/steps
# {"task_id": "...", "status": "failed", "plan_version": 1,
#  "parallel_groups": [["step_1", "step_2"], ["step_3"]],
#  "steps": [
#    {"step_id": "step_1", "agent": "research", "action": "research", "input": "...",
#     "dependencies": [], "status": "completed", "attempts": 1,
#     "output": {"content": "...", "sources": [{"title": "...", "url": "..."}], "confidence": 0.85},
#     "error": null, "tokens": {"input": 812, "output": 240, "total": 1052},
#     "execution_time_ms": 4210, "started_at": "...", "completed_at": "..."},
#    {"step_id": "step_2", ..., "status": "failed", "output": null,
#     "error": {"type": "LLMTransientError", "message": "...", "retryable": true}},
#    {"step_id": "step_3", ..., "status": "skipped", "attempts": 0,
#     "error": {"type": "DependencyNotCompleted",
#               "message": "not run: depends on step_2, which did not complete", ...}}]}
```

Cancel it (`200`; `409` if it already finished; steps that completed keep their results):

```bash
curl -X POST localhost:8000/tasks/task_952e7faa8f2d42168baf28e712bc95b9/cancel
# {"task_id": "...", "status": "cancelled"}
```

| Status | Meaning |
|---|---|
| `pending` | stored, planning not started yet |
| `planning` | the planner is breaking the task into steps |
| `planned` | a validated plan is stored; execution has not started |
| `executing` | steps are running, in dependency order, independent ones in parallel |
| `synthesizing` | the step outputs are being combined into the final result |
| `completed` | the final result is stored: `GET /tasks/{id}/result` |
| `failed` | see `error` and `failure_stage` (`planning`, `execution` or `synthesis`) |
| `cancelled` | stopped by `POST /tasks/{id}/cancel`; no result |

Get the result (`200` once completed; otherwise `409` with why there is none):

```bash
curl localhost:8000/tasks/task_952e7faa8f2d42168baf28e712bc95b9/result
# {"task_id": "...", "status": "completed",
#  "content": "Python is the easier first language; JavaScript is unavoidable on the web.",
#  "key_findings": [
#    {"statement": "Python reads like English", "step_ids": ["step_1"], "verification": "single_step"},
#    {"statement": "Both are taught widely", "step_ids": ["step_1", "step_2"], "verification": "corroborated"}],
#  "sources": [{"title": "Python docs", "url": "https://docs.python.org", "step_ids": ["step_1"]}],
#  "confidence": 0.68,
#  "confidence_basis": "mean evidence confidence 0.85 of 2 step(s) x coverage 3/3 x 0.8 for 1 conflict(s)",
#  "conflicts": [{"topic": "time to first app", "claims": [
#    {"claim": "a week with Python", "step_ids": ["step_1"]},
#    {"claim": "a day with JavaScript", "step_ids": ["step_2"]}]}],
#  "limitations": ["1 of 2 key finding(s) rest on a single step and are not corroborated."],
#  "partial": false, "synthesized_from": ["step_1", "step_2", "step_3"],
#  "synthesis": {"attempts": 1, "tokens": {"input": 900, "output": 300, "total": 1200}}, ...}

# 409 when there is no result, e.g. synthesis failed after the steps ran:
# {"task_id": "...", "status": "failed", "failure_stage": "synthesis",
#  "error": "synthesis failed: ...", "detail": "the steps ran, but combining their outputs ..."}
```

- `verification`: `corroborated` = cited by two steps that do not derive from each other;
  `single_step` = one step only (or a step and its own summary): **not verified**;
  `unattributed` = the model could not tie it to a step.
- `sources` are only pages the agents' providers returned; a draft that mentions any other URL
  is rejected.
- `conflicts` keep every side; none is resolved by picking one.
- `partial: true`: some steps did not complete; `limitations` names them.

**Execution.** Steps run group by group from the plan's `parallel_groups`; a step receives the
outputs of the steps it depends on, each cut to 8,000 characters (marked as cut; the stored
output stays complete). At most 4 agent runs at once across all tasks. A step whose failure is
retryable (rate limit, timeout) is tried up to 3 times; a step that does not finish in 180 s
counts as a timeout. When a step fails, steps that depend on it are `skipped` and independent
steps still run. If at least one step completed, the task goes on to synthesis and its result
is `partial`; if none did, the task fails (`failure_stage: execution`) with no synthesis.

**Synthesis.** One model call over every completed step's output (each cut to the same 8,000
characters), with at most one repair call if the draft is invalid. A draft that cites a step
without output, mentions a URL no step returned, or (for `output_format: json`) is not JSON,
is never stored. Transient LLM errors are retried; the whole synthesis is bounded by
`SYNTHESIS_TIMEOUT_SECONDS` and by the deadline. Any failure leaves `failed` with
`failure_stage: synthesis`; the step results stay available at `/steps`.

**Deadline.** A task's `deadline` is enforced at every stage. Passed before planning or before
execution starts: nothing runs, the task fails (`deadline passed before ...`). Reached while the
planner or a step is working: that call is stopped, nothing new starts, and the task fails
(`deadline passed during ...`). Completed steps keep their results, a retry that would end
past the deadline is not started, and steps that never ran are `skipped` with
`DeadlineExceeded`.

| Setting (env) | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_PLANS` | 4 | plans computed at once |
| `MAX_CONCURRENT_STEPS` | 4 | agent runs at once, across all tasks |
| `STEP_TIMEOUT_SECONDS` | 180 | one attempt of one step |
| `STEP_MAX_ATTEMPTS` | 3 | attempts per step, the first included (1 = no retry) |
| `STEP_RETRY_BASE_DELAY_SECONDS` | 1 | wait before the first retry; each next one waits 3x longer |
| `MAX_CONTEXT_CHARS_PER_INPUT` | 8000 | characters of one step's output passed to a step or to synthesis |
| `MAX_CONCURRENT_SYNTHESES` | 4 | syntheses computed at once |
| `SYNTHESIS_TIMEOUT_SECONDS` | 120 | the whole synthesis of one task, retries and repair included |

An out-of-range value (`MAX_CONCURRENT_STEPS=0`, say) stops the app at startup, naming the field.
`docker compose` forwards every setting in this table and in the Gemini table below from your
shell or `.env`, with the same defaults.

**After a restart**, a task left `planning` or `executing` becomes `failed`
(`interrupted: the server restarted ...`; completed steps keep their results), a task still
`pending` or `planned` is picked up again, and a task left `synthesizing` is synthesized
again (its inputs are the stored step outputs; it can never get two results). At most 4 tasks are planned at once; the rest
wait as `pending`.

**LLM provider: Gemini.** Every model call (planner, all four agents, synthesis) goes to
Google Gemini through the official `google-genai` SDK. Put your key in your shell or in a
local `.env` (git-ignored; copy `.env.example`):

```bash
export GEMINI_API_KEY=...          # personal Google AI Studio key; never logged or returned
docker compose up --build          # compose forwards the key and the settings below
```

| Setting (env) | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, or `none` to run with no model at all |
| `GEMINI_API_KEY` | (unset) | the key; unset or blank: the API runs, tasks fail clearly |
| `GEMINI_MODEL` | `gemini-3.8-flash` | any Gemini text model your key can use |
| `GEMINI_TIMEOUT_SECONDS` | 60 | one HTTP call to Gemini |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.6-flash,gemini-3.5-flash-lite` | tried in order for a call when the model before them is overloaded (503/504) or out of quota (429) |
| `GEMINI_SEARCH_GROUNDING` | `false` | Google Search grounding for `research` steps (no other agent, and not `summarize`, ever searches). Off by default: it needs Google Search quota on the key and is billed per search query. Off: research runs without search, is labelled unverified, and reports no sources |
| `GEMINI_MAX_GROUNDED_CALLS_PER_TASK` | 2 | a **hard** limit on Gemini requests sent with search attached, per task: every research attempt, retry and model fallback takes one (0: none). It does **not** limit Google queries: see below |

Without a key the app still starts: each task fails at planning with
`no LLM provider is configured: LLM_PROVIDER=gemini but GEMINI_API_KEY is not set ...`. An
unknown `LLM_PROVIDER`, a blank `GEMINI_MODEL` or an out-of-range timeout stops the app at
startup. With grounding on, research steps use Google Search grounding, and their `sources`
are the pages Gemini reports it used. Rate limits, timeouts and 5xx are retried by the stages, waiting as
long as Gemini asks (up to 60 s); a rejected key, an unknown model, a blocked prompt, or a
model whose daily quota is used up (once every fallback model is too) are not.

**On a free-tier key** (what was available while building this): quotas are per model and per
day, Google Search grounding may have no quota at all, and popular models are often
overloaded. What worked end to end on 2026-11-14 was `GEMINI_SEARCH_GROUNDING=false` with the
default fallbacks, and tasks submitted one at a time. Without grounding, results have no
sources and low confidence, and say so in their limitations.

**Bounded grounding: what is limited, and what is only counted.**

- **Limited, hard: grounded requests.** Each task gets `GEMINI_MAX_GROUNDED_CALLS_PER_TASK`
  slots, shared by all its research steps, parallel ones included. Before every Gemini
  request that would attach Google Search, a slot is reserved. That covers a step's first
  attempt, each retry and each model fallback. A request that then fails still used its
  slot, because it may have searched and been billed. When no slot is left, the step still
  runs, without search.
- **Not limited: Google queries.** The model decides how many queries one grounded request
  runs, and the Gemini API has no parameter that caps them. The research prompt asks for "a
  few focused queries" on one step, but that is an instruction, not a limit. Queries are
  counted afterwards from Gemini's `web_search_queries`, per step and per task. A request
  that failed or timed out reports none, so the count can be lower than what was billed.
  Gemini 3 bills grounding per query: one task can cost more queries than its grounded
  requests.
- **Verified or not, per research step.** Each research output has
  `grounding: {status, reason, search_queries}`:
  - `grounded`: search ran and Gemini returned the pages used, which are its `sources`.
  - `unverified`, with a reason: grounding off, the budget used up, or search ran but
    returned no pages.

  When no search ran at all, the step's text also starts with `Unverified: <reason>.`.
  Synthesis adds every unverified step to the result's `limitations`. Only grounded pages
  ever become `sources`.

Per step, `search` is `{grounded_calls, queries}` over every attempt. The task, the trace
and `/metrics` add these up. Search is counted apart from `usage`, which stays tokens only.

| Field | Rules |
|---|---|
| `goal` | required, 1-4000 characters after trimming |
| `constraints` | optional; `max_words` (int 1-20000), `tone` (string), `include_code_examples` (bool), plus any other JSON values |
| `output_format` | `markdown` (default), `text` or `json` |
| `deadline` | optional ISO-8601 time with a timezone, in the future |

Errors: `422` invalid input (every problem listed), `404` unknown task (also for `/steps`,
`/result` and `/cancel`), `409` cancelling a finished task or asking for a result that does
not exist (yet), `503` database unavailable.

## Test

Unit tests need nothing. Integration tests need the Postgres container
(`docker compose up -d postgres`); they create and migrate their own `orchestrator_test` database.
No test calls a real model or needs an API key: a fixture blanks any key from your shell or
`.env` for every test, so a test cannot reach Gemini by accident.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests scripts
```

A Docker smoke test checks the image itself: the tests above run in the development
environment, which has packages the image does not install (a runtime dependency missing from
the image was once found only this way). It builds the image, starts it next to a throwaway
Postgres, and requires `GET /health` to answer 200, once with `LLM_PROVIDER=none` and once with
`LLM_PROVIDER=gemini`, both without a key, so it never calls a model. It does not use
`docker-compose.yml` or its ports, and removes everything it creates:

```bash
scripts/docker_smoke.sh                   # needs Docker; about 20 s with a warm build cache
SMOKE_PORT=18010 scripts/docker_smoke.sh  # another loopback port (default 18000)
SMOKE_NO_BUILD=1 scripts/docker_smoke.sh  # reuse an image built before (SMOKE_IMAGE)
```

With a key (shell or `.env`), two more checks reach the real Gemini API. They are never part
of a plain `pytest` run:

```bash
uv run pytest -m live tests/live -v       # skipped, with the reason, without a key
uv run python scripts/gemini_e2e.py       # against a running API (docker compose up): 3 tasks
                                          # via POST /tasks, one at a time; steps, trace, usage
                                          # and result of each saved in e2e-results/ (git-ignored)
```
