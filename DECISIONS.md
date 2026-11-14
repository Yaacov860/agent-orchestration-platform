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

## 0.1 LLM layer and the agent contract

**One provider-neutral contract.** Agents call `LLMClient.generate(LLMRequest) -> LLMResponse` and
never import a provider SDK. Each provider translates its SDK objects into these shapes and maps its
exceptions onto exactly two: `LLMTransientError` (rate limit, timeout, 5xx, empty reply: worth a
retry) and `LLMPermanentError` (bad request, auth, blocked content: will fail the same way again).
The executor's retry decision reads only `retryable`, so it never learns any provider's error types.

**Sources come from the provider, never from the model's text.** A model asked to "cite sources"
invents plausible URLs. With search grounding the provider returns the pages it actually used, as
data separate from the text, and the research prompt tells the model not to write URLs at all.

**Agents: Template Method.** `BaseAgent.run` is written once: check the action is supported, time
the run, meter tokens, and turn any failure into a structured `AgentResult` with `retryable`.
Subclasses implement only `execute`. A new agent cannot forget timing, metering or error handling.

**Token metering is per run, not per agent.** One agent instance serves many steps concurrently
(two research steps in the same parallel group), so a counter stored on the agent would mix their
numbers. `run` wraps the client in a fresh `MeteredLLM` for each run. Tokens spent before a failure
are kept: they were paid for. A test runs two steps concurrently on one instance; injecting a shared
counter makes it fail (`[1, 1001] != [1, 1000]`).

**Cancellation is not a failure.** `run` catches `Exception`, not `BaseException`, so
`asyncio.CancelledError` propagates and a cancelled task actually stops.

**Research confidence is evidence-based, not self-reported.** Models' self-rated confidence is
poorly calibrated. Confidence is derived from the number of distinct grounded sources (0 → 0.3,
1 → 0.6, 2+ → 0.85). It is a crude heuristic, but it is explainable and testable. `summarize`
does no search, so it reports `None` rather than a made-up number.

## 0.2 Data model and the task API

**Three tables, split by "planned" vs "happened".**
`tasks` (what the user asked) ─< `execution_plans` (what the planner proposed) ─< `execution_steps`
(what actually happened to each step). A plan is written once and never edited; a re-plan adds a
row with a higher `version`, so both plans stay inspectable. Steps are the mutable runtime state.
The assignment lists both `ExecutionPlan.steps (JSON)` and `ExecutionStep`: the JSON is the record
of the plan as validated, the rows are where status, output, tokens and timings get updated.

**Postgres, not memory.** A task runs for tens of seconds while the client polls in separate
requests, so state must outlive the request that created it, survive a restart, and be shared by
every API process. Restarting the API container and fetching the same task proves it.

**The database enforces the rules too.** `UNIQUE(plan_id, step_id)` (dependencies refer to steps
by that id), `UNIQUE(task_id, version)`, `CHECK` on statuses and counters, `ON DELETE CASCADE`.
Statuses are `VARCHAR + CHECK` rather than a native Postgres `ENUM`, which needs `ALTER TYPE` to
extend; Python still gets enum members back. FK columns are covered by the leading column of those
unique indexes, so no extra indexes are needed.

**Async SQLAlchemy guard rails.** Relationships are `lazy="raise"`: an implicit lazy load cannot
work under asyncio, so it fails loudly at the line that tried it. `expire_on_commit=False` for the
same reason.

**Progress is derived, never stored.** `GET /tasks/{id}` reads the task and the steps of its
newest plan and computes the counts on every call, so they cannot drift from the step rows.
The assignment's `current_step` is kept, but with parallel steps several run at once, so
`running_steps` lists all of them.

**`POST /tasks` returns `pending`, not `planning`** (the assignment's example), because nothing
plans yet. The status moves to `planning` when planning actually starts.

**Validation at the edge.** Unknown top-level fields are rejected (a typo such as `constraint`
would otherwise be silently dropped). Known constraints use strict types: lax mode would accept
`"yes"` as `true` and `"1500"` as `1500`, hiding client bugs. Extra constraints are allowed but
must be JSON. A deadline must carry a timezone and be in the future. A malformed task id is a 404
without a database round trip.

**Errors.** 422 validation (all problems at once), 404 unknown task, 503 + `Retry-After` when the
database is unreachable. Only connection-type errors become 503: a constraint violation is a bug
in our code, so it stays a 500.

**Migrations.** Alembic, autogenerated from the models and then reviewed by hand. The test suite
drops and recreates its database and runs `alembic upgrade head` on every run, so the migration is
tested from scratch every time; another test runs downgrade-to-base and back up. The container runs
`alembic upgrade head` before starting uvicorn; with several replicas that belongs in a separate
deploy step instead, to avoid concurrent migrations.

**Adding a status means replacing a CHECK by hand.** Autogenerate does not detect CHECK changes,
so 0002 is hand-written, with the status lists spelled out instead of read from the enum (a
migration is a frozen snapshot). Its downgrade refuses while `planned` rows exist instead of
silently rewriting them. A test inserts every `TaskStatus` value into the migrated database, so
an enum member added without a migration fails in CI, not in production.

## 0.5 Specialized agents: writing, analysis, code

**Four agents, one contract.** Research (existing), writing (`write`, `rewrite`,
`draft_email`), analysis (`analyze`, `compare`, `evaluate`) and code (`write_code`,
`explain_code`, `review_code`, `fix_bug`). All four take an `AgentTask` and return the same
`AgentResult`; tests check the shape is identical. All four run through `BaseAgent.run`, which
does the action check, timing, token metering and error mapping, and build their prompt with
`BaseAgent.compose_user_prompt`: the goal, the constraints, the outputs of this step's direct
dependencies, and the instruction. Nothing in BaseAgent was duplicated or rewritten.

**One small base for the three new ones.** Writing, analysis and code differ only in prompts,
sampling and output checks, so `PromptedAgent` (one model call, a system prompt per action,
shared rules appended) holds the common `execute`, and each agent only declares data:
`system_prompts`, `temperature` (code 0.1 < analysis 0.2 < writing 0.7), `max_output_tokens`,
`output_checks`. Research keeps its own `execute`: it searches, collects sources and derives
an evidence-based confidence, which the others do not.

**No search, no sources, no confidence outside research.** The three new agents never request
web search, never return sources (even if a provider attached some to a non-search call), and
report `confidence: null`. Sources and evidence therefore enter the system through exactly one
door, and synthesis's provenance (0.4) needs no special cases.

**Do not invent.** Every new agent's prompt says to work only from the goal, the constraints,
the inputs and the step; to invent no facts, figures or sources; and to say when the inputs are
not enough. Analysis must name the input each claim rests on and write "insufficient
information" rather than guess. This is instruction, not enforcement: a model can still
invent, and only research outputs are grounded (see AI_USAGE.md).

**Code is text only.** The code agent has no tool to run code, touch files or act on a machine,
and its prompt forbids claiming code was run or tested. `write_code` and `fix_bug` must return a
fenced code block. An answer without one is an `InvalidAgentOutputError`, retryable like an
empty answer, because a second sample usually complies. After the last attempt the step fails
and its dependents are skipped.

## 1. Task Decomposition Strategy

## 2. Dependency Management

## 3. Parallel Execution

## 4. Failure Recovery

## 5. One Thing I Would Do Differently With More Time
