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

## 0.4 Synthesis, provenance and the final result

**The model writes, code decides what a model would invent.** The model's draft is only
`content`, `key_findings` (each with the `step_ids` it came from), `conflicts` (every side
with its steps) and `limitations` it noticed. Strict Pydantic, unknown fields rejected.
Code then computes:
- `sources`: exactly the pages the steps' providers returned, each with the steps that got it.
  The model never sees a URL in its prompt (only a count), and a draft that mentions any URL no
  step returned is rejected.
- `verification` per finding: `corroborated` only when it cites two steps that do not derive
  from each other. A summary step agreeing with the step it summarized is not a second opinion,
  and neither is a summary alone, even of several steps. `single_step` is shown as not
  verified, and `unattributed` (an empty `step_ids`) is allowed and labelled, rather than
  forcing the model to guess a step.
- `confidence`: mean evidence-based step confidence x coverage (completed / all steps) x 0.8
  if there are conflicts, with `confidence_basis` saying exactly that. With no step reporting
  evidence, it is `null`, not a made-up number.
- `limitations`: every failed or skipped step with its error, every output cut for the prompt,
  "no web sources" when none came back, and the count of findings that are single-step or
  unattributed. The model's own limitations are appended, without duplicates.

**Conflicts are kept, never resolved by fiat.** The prompt asks for every side with its steps
and forbids picking one unless the outputs settle it. Validation requires at least two
different claims, each citing steps that have output. Conflicts lower the confidence and are
stored and served whole. Detecting them is the model's job: no deterministic comparison of
free text was attempted, which is a real limitation of the approach.

**Invalid output is never stored.** Same pattern as planning: all problems collected, one
repair call that shows the model its answer and the problems, then failure. A draft citing a
failed step, an unknown step or an invented URL, or invalid JSON for `output_format: json`,
counts as invalid.

**What reaches synthesis.** Every completed step, cut to `MAX_CONTEXT_CHARS_PER_INPUT` with a
marker. Each is labelled with its agent, action, instruction and the steps it built on, so
both the model and the provenance code know the lineage. Failed and skipped steps are listed
as "without output (cannot be cited)".

**Separation.** `synthesis/` is pure. It imports none of planning, execution, the database or
the services, and gets the task and step rows as plain data (`SynthesisInput`). Execution hands
over through an injected `on_executed(task_id)` callback, as planning does with `on_planned`.
The architecture test enforces all of it.

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

**The planner chooses by description, not by name.** Each agent declares
`action_descriptions`: what each action produces and what it needs. The catalog passes them
into the planner prompt, together with rules that are catalog-agnostic and name no agent
(agents that gather no information need earlier steps as dependencies; gather facts first,
produce the deliverable last). A test builds a planner from a writing-only catalog and checks
that its prompt mentions no other agent. A class whose descriptions do not match its
capabilities fails when it is defined. The model's routing is then checked like everything
else: an action given to the wrong agent is a validation problem whose message lists what that
agent can do, so the single repair attempt can re-route it.

**Synthesis keeps deliverables.** When a step already produced what the task asks for (a
document from writing, code from code), the synthesis prompt asks to build on it rather than
rewrite it. A URL that appears in a step's own output (an endpoint in generated code, say) may
be quoted, since it is copied rather than invented, but it never becomes a source: `sources`
stays provider-only.

## 0.6 The Gemini provider

**One adapter behind the existing contract.** `GeminiLLM` implements `LLMClient.generate`
exactly, and nothing outside `llm/` changed. The planner, the four agents and synthesis all
use the one instance `create_app` builds, each still metered per call by `MeteredLLM`.
`build_llm(settings)` is the only place that knows providers. The SDK is imported only when
Gemini is actually configured.

**Request mapping.** `system_prompt` becomes `system_instruction`, `user_prompt` the contents,
and `temperature` and `max_output_tokens` pass through. `use_search` adds the Google Search
grounding tool, and only research asks for it.

**Response mapping.**
- Text is the first candidate's parts, without thought parts.
- Usage is `prompt_token_count` in, and `candidates_token_count + thoughts_token_count` out,
  since thinking is billed as output. When the provider reports no usage it counts as zero,
  not as an error.
- Sources are the grounding chunks' web pages (URI and title), exactly as the provider
  reports them.

**One retry layer, not two.** The SDK would retry 408/429/5xx itself. It is configured with
`attempts=1`, so retries happen only in the stage that owns the policy (planning, the step
executor, synthesis). Otherwise one step could make up to 5 x 3 calls. The SDK timeout
(milliseconds, per HTTP call) is `GEMINI_TIMEOUT_SECONDS`, 60 s by default, under the 180 s
step timeout: a hung call ends as a provider timeout, which the stage retries, before the
step timeout would have to stop it.

**Error mapping and retryability.**

| What Gemini did | Mapped to | Retried |
|---|---|---|
| 401, 403, or 400 "API key not valid" | `LLMPermanentError` ("check GEMINI_API_KEY") | no |
| 404 (model not found) | `LLMPermanentError` ("check GEMINI_MODEL") | no |
| other 4xx (bad request) | `LLMPermanentError` | no |
| 429 with a per-day `quotaId` (the model's daily quota is used up) | next model; if none is left, `LLMPermanentError` with the reset time | no |
| other 429 (per-minute limit, or grounding quota) | next model; if none is left, `LLMTransientError` with the provider's `retryDelay` | yes, after that delay (60 s cap) |
| 503, 504 (overloaded) | next model; if none is left, `LLMTransientError` | yes |
| 408, 499, other 5xx | `LLMTransientError` | yes |
| network error, HTTP timeout (`httpx` or `httpx2`) | `LLMTransientError` | yes |
| blocked prompt, or stopped for safety / recitation / blocklist / prohibited content / SPII / language | `LLMPermanentError` | no |
| no candidates, empty answer, unparseable response | `LLMTransientError` | yes |
| `MAX_TOKENS` with no text at all | `LLMPermanentError` (the same budget runs out the same way) | no |
| `MAX_TOKENS` with some text | returned as-is; JSON callers' validation catches the cut | (repair) |

**What the real API taught (2026-11-14, a free-tier key).** Every rule above about 429, 503,
504 and timeouts was revised after real responses, not designed up front.
- **The 429 text is identical** for a per-minute limit, a used-up daily quota and a missing
  grounding quota ("You exceeded your current quota, please check your plan and billing
  details"). A first version made it permanent on that text. A minute later the same plain
  requests succeeded, so that was wrong, and it failed three end-to-end tasks without a
  retry. The structured details tell the cases apart: `QuotaFailure.quotaId`
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`) and `RetryInfo.retryDelay`
  (`52691s` for the daily one).
- **Quotas and load are per model.** `gemini-3.8-flash` was out of its daily quota and
  overloaded (503) while `gemini-3.6-flash` and `gemini-3.5-flash-lite` answered, so a call
  moves to the next of `GEMINI_FALLBACK_MODELS` on 429, 503 or 504. In the successful
  end-to-end run, 25 calls fell back.
- **Waiting.** A stage's own retry delays (1 to 5 s) are far shorter than a per-minute
  window. Every stage now waits `max(its delay, the provider's retryDelay)`, capped at 60 s,
  within the deadline.
- **Two transports.** The SDK sends through `httpx2`, a separate package whose exceptions do
  not inherit from httpx's. A real read timeout escaped unmapped and was recorded as an
  internal error. Both families are mapped now.
- **Grounding.** Google Search grounding returned 429 instantly on every model while plain
  requests worked: this key has no grounding quota. `GEMINI_SEARCH_GROUNDING=false` lets such
  an account run, with results that have no sources, low confidence, and say so.
- **Research without search.** Told to "use web search" with no search tool, the model wrote a
  tool call as its answer (`fn:authoritative_search(query="...")`), and the writing step built
  on nothing. The research prompt now covers running without search. A tool-call-only answer
  is an `InvalidAgentOutputError`, which is retried. The model did not add the "Unverified"
  line the prompt asks for, so transparency rests on the code-side limitation ("No step
  returned web sources") and the 0.3 confidence, not on the model.

**The key stays secret.** It is a `SecretStr` in `Settings`, so repr and dumps mask it. It goes
to the SDK only, which sends it in the `x-goog-api-key` header, not in the URL (checked in the
SDK source). Error text is built from status and message only, scrubbed of the key, cut to
300 characters, and raised `from None` so the SDK exception's payload is not chained into
logs. Tests check every one of these.

**Model choice: `gemini-3.8-flash` (configurable with `GEMINI_MODEL`).** A fast text model with
Google Search grounding. Flash rather than Pro because one task makes many calls (planner,
every step, synthesis), so latency and cost dominate. It is the model the official SDK's
README (google-genai 3.3.0) uses for text and search grounding. The live API backs it: the key's
model list includes it, and a request to `gemini-2.5-flash` was refused as "no longer
available to new users", with `gemini-3.8-flash` named as the replacement. No successful
generation has been observed yet (503 and 504 under high demand, 429 on grounding quota): see
AI_USAGE.md.

**Hold the SDK client, not just its models.** The SDK `Client` closes its HTTP connection when
it is garbage-collected. `GeminiLLM` keeps the client itself for its whole life. A regression
test forces a collection on a running loop and checks the connection is still open.

**Tests can never reach Gemini by accident.** An autouse fixture sets `GEMINI_API_KEY` (and
`GOOGLE_API_KEY`) to empty for every test not marked `live`. An empty variable wins over a key
in `.env`; this was checked, not assumed. Live tests are opt-in, skip with a reason without a
key, and the end-to-end script runs against a deployed API rather than inside the test suite.

## 1. Task Decomposition Strategy

**The model proposes, the code decides.** The planner asks the LLM for a JSON plan; nothing it
returns is trusted until `parse_plan` and `validate_plan` accept it. The goal is fenced in the
prompt as user data, but the real protection is that validation, not the model, decides what runs.

**One catalog, used twice.** `AgentCatalog` is built from the agent classes' own `name`,
`description`, `capabilities` and `action_descriptions`, and contains only implemented agents
(research, writing, analysis, code; see 0.5). The prompt lists exactly those agents and
actions, and validation checks the answer against the same catalog, so the prompt can never
offer what validation rejects. Adding an agent to `IMPLEMENTED_AGENTS` is all it takes to make
it plannable, and runnable, since the executor's `AgentFactory` reads the same list.

**Strict plan schema.** Unknown fields are errors (`dependecies` would otherwise drop the
dependencies silently), step ids follow `step_<n>`, lengths match the database columns, and a
plan has 1 to 8 steps. More steps mostly means a model over-splitting the task: more calls, cost
and latency, not a better result.

**All problems at once, then one repair.** Validation collects every problem, each naming the
step and what is allowed instead (`unknown agent 'writing'; available agents: research`). If the
first answer is invalid, the model gets its own answer back with that list, once. A model that
cannot fix a plan when shown the exact problems rarely does on a third try, and each try is a
full LLM call. Harmless sloppiness (a dependency listed twice) is normalized rather than sent back.

**Holding a `ValidatedPlan` is proof.** Only `validate_plan` builds one, and only it carries the
parallel groups, so later code never re-checks whether a plan is sound.

## 2. Dependency Management

**Dependencies are declared, never inferred.** Each step lists the ids whose output it needs.

**Parallel groups by Kahn's algorithm, level by level** (O(steps + dependencies)): group 0 is every
step with no dependencies; each next group is every step whose dependencies are all in earlier
groups. Within a group steps keep plan order, so the same plan always yields the same groups.

**Cycles are reported as the cycle itself.** Steps that never reach zero waiting dependencies are
stuck; walking back along stuck dependencies until a step repeats yields the actual loop
(`step_2 -> step_3 -> step_2`), told apart from steps merely blocked behind it. Validation looks
for cycles even when other edges are broken (those are reported already): with a single repair
attempt, the model must see the cycle in the same round as everything else. Only duplicate step
ids skip the check, since a dependency on a duplicated id has no single meaning.

## 3. Parallel Execution

**Planning and execution meet only in the database.** Once a plan is committed, planning calls
an injected `on_planned(task_id)`; the app wires it to `ExecutionService.schedule`. The executor
reads the stored rows (steps, `parallel_groups`) and never imports the planning package, and
planning never imports execution. An architecture test parses the source and fails on any such
import. A plan stored by an older planner runs the same way as a fresh one.

**Group by group.** The executor walks the plan's `parallel_groups`: every step of a group starts
together, and the next group starts when the whole group has finished. Steps in a group never
depend on each other, so running them at once is always safe. The cost: a step that is ready
early still waits for its slowest group-mate. Scheduling each step the moment its own
dependencies finish ("eager") would shorten the critical path. It was not chosen for now, since
the groups are already validated, stored and visible, which makes progress easy to reason about.

**One cap for the whole process.** At most 4 agent runs at once, across all tasks: the LLM's rate
limit is shared by everything, so a per-task cap would still let ten tasks fire forty calls. A
slot is held per attempt and released during the wait before a retry, so a step that is backing
off does not block others. Tests hold calls "mid-thought" (a gated fake LLM) and check that two
tasks with three parallel steps each never have more than 2 calls in flight, and that the
database shows exactly 2 `running` rows at that moment.

**Inputs flow along declared edges only.** A step gets the outputs of its *direct* dependencies,
labelled by step and agent. What it needs from further back, it must declare.

**Each input is cut to a set size, visibly.** One dependency's output is passed in up to
`MAX_CONTEXT_CHARS_PER_INPUT` characters (8,000, about 2,000 tokens). The limit is counted in
characters because no tokenizer is available for every provider. The beginning is kept, since
agents put their findings first. The cut is marked (`[... truncated: N of M characters
omitted]`) so the model knows text is missing rather than taking a fragment for the whole. Only
the prompt is cut: the stored output, and what `GET /tasks/{id}/steps` serves, stays complete.
The limit is per input, so a step with many dependencies gets proportionally more. A per-step
budget would be the next refinement.

**One agent instance per agent, shared.** `AgentFactory` builds every agent once from
`IMPLEMENTED_AGENTS`, the same list the planner's catalog reads, so the planner can only name
agents the executor can run (a test compares the two). Sharing is safe because `BaseAgent`
meters each run separately.

**No transaction is open while an agent runs.** Every state change (started, finished, skipped,
cancelled) is its own short, conditional UPDATE, checked against `STEP_TRANSITIONS`. That table
caught a real bug during development: finishing a step was allowed from `pending` too, which
would have let a step "complete" without ever running.

**A bug in one step does not orphan its siblings.** A group runs in an `asyncio.TaskGroup`, not
`gather`: if a step raises (say, the database is gone), the other steps of the group are
cancelled instead of running on after the task has already been failed.

## 4. Failure Recovery

## 5. One Thing I Would Do Differently With More Time
