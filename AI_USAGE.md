# AI Tool Usage

Two different things are documented here: how the **system** uses an LLM (and how tests replace
it), and how an **AI coding assistant** was used to build it. The second part currently covers
stage 3 (planner and its integration), the execution engine and the synthesis engine.

> **To review (author):** sections marked _[author to confirm]_ state judgements or reflect a
> way of working as seen from the assistant's side of the sessions. They need your
> confirmation, or your own words, before this is submitted. Earlier stages (0 to 0.2) are not
> covered and need to be added by you.

## How the system uses an LLM

**Where a model is called today**, per task:

| Caller | Calls | Purpose |
|---|---|---|
| Planner | 1, plus at most 1 repair | break the goal into a validated step graph |
| Research agent | 1 per step attempt, up to 3 attempts | do one step (`research` is web-grounded only when `GEMINI_SEARCH_GROUNDING` is on, and only within the task's grounded-call budget; otherwise it is labelled unverified) |
| Writing, analysis, code agents | 1 per step attempt, up to 3 attempts | write, analyze or work with code from earlier steps' outputs; no search |
| Synthesizer | 1, plus at most 1 repair; up to 2 retries of a transient error | combine the step outputs into the final answer |

All three go through the same `LLMClient` contract and meter their tokens with `MeteredLLM`.

**Which model.** Google Gemini through the official `google-genai` SDK, `gemini-3.8-flash` by
default (`GEMINI_MODEL`), for every caller. Without `GEMINI_API_KEY` the app builds
`UnconfiguredLLM`: the API runs and tasks fail with a message saying the key is missing.

> **Real API runs (2026-11-14, a free-tier key from `.env`).**
>
> - **End to end, through the API, succeeded:** `POST /tasks` -> planning -> agents ->
>   synthesis -> `GET /tasks/{id}/result` (HTTP 200), with `GEMINI_SEARCH_GROUNDING=false`,
>   the default fallback models, and tasks one at a time:
>   - task_9a34934a3637489986130824d363a1ef: analysis and writing (research, analysis, writing), completed, 9402 tokens;
>   - task_6d8cdd84eff948b0b32a38effe1db5ad: code (code, writing), completed, 7019 tokens;
>   - task_87b97762d2564e1fa0b87a469e53034d: research and summary (research, writing), completed, 5457 tokens,
>     after the fix described in "What I Had to Fix". An earlier run of the same task,
>     task_15524165d4e84af38ff0e44ef6940819, also "completed", but with an empty research step.
> - **Live tests:** a plain request passed (real text and token counts), and a bad key was
>   reported as an auth error without echoing it. The search-grounding test is skipped: this
>   key has no grounding quota (instant 429 on every model).
> - **Not verified against the real API:** grounded sources (no quota), and therefore
>   `corroborated` findings and evidence-based confidence above 0.3. Result quality was read
>   by eye on three tasks, not evaluated.
> - **Earlier failed runs, kept for the record:** task_30ba0792820548dab42ade68cd333db7,
>   task_f9c997e11362458d932c04002220addf, task_03b1c78da69345c99b3d1a453a7300bd (503/504),
>   then task_28f1a2c6490944198c35a931c9cf7a80, task_f2d68d0473c84ef0acff4db56cec023d,
>   task_610a2fdaa872414181ed97c9aa7c1e3e (429 wrongly treated as permanent), then three
>   failed at planning on a used-up daily quota before model fallback covered 429.
> - **After billing was enabled on the key (2026-11-14): one more end-to-end task**, through
>   the Docker image of the final code (`docker compose`, PostgreSQL in its container), sent
>   with `POST /tasks` and polled to the end: `task_243afa2a37504ae7b62f36dedde32b19`,
>   `completed`, not partial. The plan had three steps in two parallel groups (research and
>   code, then writing); the planner, every step and the synthesis succeeded on the first
>   attempt, with no retry and no fallback model. Tokens, as reported by Gemini: 8,018 in total,
>   3,519 input and 4,499 output (planning 2,187, execution 3,678, synthesis 2,153).
>   **Google Search grounding was disabled for this run** (`GEMINI_SEARCH_GROUNDING=false`):
>   the research step ran without search, began with "Unverified: no web search was
>   available", and returned no sources and confidence 0.3, and the result lists "no web
>   sources" among its limitations. Grounded search therefore remains unverified live.
>
> _[author: with a key that has grounding quota, rerun `uv run pytest -m live tests/live -v`
> and `uv run python scripts/gemini_e2e.py` with grounding on, and record the outcome.]_

**What the model is trusted with.** Nothing it returns is used unchecked. A plan is parsed with
a strict schema and validated against the agent catalog and the dependency graph before it is
stored. A synthesis draft is parsed strictly and checked against the steps that produced
output: no citing a failed or unknown step, and no URL that no step returned. Sources,
"corroborated", confidence and the failure-based limitations are computed by code, never
taken from the model. The user's goal and the step outputs are fenced in prompts as data, but
validation is the real guard.

**What has not been verified.** No prompt (planner, research, writing, analysis, code,
synthesis) has been run against a real model. In particular, whether a real planner routes
steps to the right agents is untested: the tests check that the descriptions reach it and that
wrong routing is caught and sent back, not the model's choices. The tests prove the logic around the model (parsing, validation, repair, retries,
provenance, storage), not the quality of what a real model would produce. In particular,
conflict detection relies on the model noticing disagreements: the tests check that reported
conflicts are kept whole and validated, not that a real model finds them.

## How tests replace the LLM: FakeLLM

No test calls a model, needs an API key or touches the network. `FakeLLM` is a scripted
`LLMClient`: each call pops the next item of a script written by the test, either a response or
an exception, and every request is recorded.

- **Failure paths are as easy to test as success:** a script can hold `LLMTransientError("429")`,
  `LLMPermanentError`, or `asyncio.CancelledError`.
- **Over-calling fails loudly.** A call beyond the script raises, so "exactly one repair" and
  "never planned twice" are asserted by giving the fake exactly as many replies as allowed.
- **Recorded requests** let tests check what was sent: the catalog in the system prompt, the
  goal and constraints, and the rejected answer plus its problems in the repair prompt.
- **Small purpose-built fakes** cover what a script cannot. `ProbeLLM` runs code while the "model
  is thinking", to check the task is already `planning` and committed, and that no database
  connection is held. `GatedLLM` blocks until released, to test the concurrency cap and a
  shutdown mid-plan. `UnconfiguredLLM`, the production class, is used directly to test the
  no-provider path.
- **FakeLLM is not a provider.** Unscripted it can only fail, so it is injected through
  `create_app(llm=...)` rather than selected by `LLM_PROVIDER`.

## Tools I Used

- **Claude Code** (VS Code extension), model Claude Opus 5.5, for stage 3, the execution and
  synthesis engines, the specialized agents and the Gemini provider: design proposals,
  implementation, tests and documentation.
- Way of working: environment checks first (read-only), then a written plan per unit, approved
  before any code. Commits and pushes only on explicit approval. Each unit ended with the full
  suite, `ruff` and `mypy --strict`.

## What Helped Most _[author to confirm]_

- **Plan before code, with decisions surfaced as questions.** Choices with real trade-offs went
  back to the author instead of being made silently: a separate `planned` status (migration) versus
  reusing `planning`, a catalog limited to implemented agents, failing clearly when no provider
  is configured, and what startup recovery does to each status.
- **Mutation checks on green tests.** When new tests passed on the first run, a deliberate bug was
  planted to confirm a test catches it, then removed. Examples: allowing a third planner attempt,
  making status transitions unconditional, storing a plan for a task that already failed,
  removing the concurrency cap. Every planted bug was caught.
- **Deterministic tests for background work.** A `ManualRunner` that only records jobs removed the
  race between `POST /tasks` and the planner in API tests, with one end-to-end test on the real
  runner.

## What I Had to Fix _[author to confirm the framing: these were caught by the checks and by review within the sessions]_

Problems in AI-written code or tests, caught during review or by the checks:

- **Cycle detection skipped a broken graph.** The first validation ran the cycle check only when
  there were no other dependency problems. With a single repair attempt, the model would fix the
  visible problems and never see the cycle. Found by printing the actual messages for a sample
  plan before writing tests. Cycles are now checked over the valid edges.
- **Strict parsing rejected valid JSON.** Pydantic strict mode refuses a `list` for a tuple field,
  which is what `json.loads` produces. `parse_plan` now parses the JSON text directly.
- **A test that tested nothing.** `test_types_are_not_coerced` would pass with or without strict
  mode. Removed.
- **A false confirmation.** One mutation script failed to apply, because the formatter had changed
  the indentation, and reported "17 passed" on unmutated code. Caught by checking that the
  mutation was really in place; rerun properly, it failed as it should.
- **A test that depended on the developer's machine:** it read `Settings()`, which reads the
  environment and `.env`. It now checks the declared default.
- **A polling helper** in tests (`while ...: sleep`), flagged by `ruff`, replaced with an
  `asyncio.Condition` that wakes exactly when the state changes.
- **A step could "complete" without running.** The execution recorder allowed finishing a step
  from `pending` as well as `running`. The step-transition table refused `pending -> completed`
  on the first end-to-end run, and the allowed source states now depend on the outcome.
- **Orphaned steps on a bug.** The first executor ran a group with `asyncio.gather`, which leaves
  sibling steps running when one raises. It now uses a `TaskGroup`, and a test checks that no
  agent task is left behind.
- **A wrong assumption about a library.** Test helpers were switched to
  `Settings.model_validate` on the belief that it skips environment variables. Checked before
  relying on it: it does not. The defaults now come from `model_construct`, and a test sets an
  environment variable to prove the defaults ignore it.
- **A concurrency test that could not fail.** The first "cap from Settings" test released the
  fake model as soon as one call arrived, so the other calls returned at once whether or not a
  cap existed. It now holds the calls and checks that only one is in flight.
- **A wrong example in the README:** a progress sample showed a step running alongside the
  step it depends on. Corrected.
- **Assumptions the real API overturned.** All of these were found only by running against
  Gemini, and each was fixed and covered by a test built from the real response:
  - a 429 rule written from the error text alone made per-minute limits permanent, which is
    wrong (the same text covers three different cases);
  - an exception mapping that missed `httpx2`, the transport the SDK actually uses, so a real
    timeout escaped as an internal error;
  - a research prompt that broke when search was off: the model wrote a tool call as its
    answer, and it passed as findings;
  - live tests that ran on every plain `pytest` once a key existed in `.env`.
- **Diagnosing by evidence, not by guess.** The 503/504/429 failures were narrowed down
  step by step: one call per model, then the structured `QuotaFailure` and `RetryInfo`
  details, with no key ever printed. The first explanation (a per-minute limit) was itself
  partly wrong; the `quotaId` showed a used-up daily quota on the default model.
- **A Gemini client that closed itself.** The first `GeminiLLM` kept only `client.aio.models`.
  The SDK `Client` was then garbage-collected, and its `__del__` closed the HTTP connection,
  so every real call failed with "client has been closed". Offline tests use a fake and could
  not see it. It surfaced in a startup check with a real `Client`. The first regression test
  written for it passed even on the broken code (it was synchronous, so the SDK's scheduled
  close never ran); it was rewritten as an async test and confirmed to fail without the fix.
- **A risk of tests calling the real API.** Once a key exists in `.env`, any test building
  `Settings()` without injecting an LLM would have called Gemini, costing money and depending
  on the network. Spotted while adding the provider: an autouse fixture now blanks the key
  for every non-live test. The claim it relies on, that an empty environment variable
  overrides `.env`, was checked with a temporary `.env` before being used.
- **Reading the SDK instead of assuming it.** The installed google-genai (3.3.0) is newer than
  the assistant's training data. The error classes, the timeout unit (milliseconds), the
  default retry behaviour and how the key is sent were all read from the installed source
  before writing the adapter.
- **A planner prompt that named agents it might not have.** The first version of the new
  "choosing agents" guidance listed research, writing, analysis and code by name. A planner
  built from a smaller catalog would have been told about agents it could not use. An
  existing test caught it; the guidance now names no agent, and a new test checks that a
  writing-only planner's prompt mentions no other.
- **A test double that broke the system under test.** A scripted fake LLM wrapped a full
  response object as if it were text, so every research step "failed". The fix was in the
  fake, not in the code; worth noting because the failure looked like a real bug at first.
- **A wrong expected value in a test.** A confidence-basis test expected `0.73` for a mean of
  0.85 and 0.6. That mean is 0.725, which is 0.7249... in binary floating point, so the code
  correctly prints `0.72`. The test was wrong, not the code.
- **A type check silenced instead of fixed.** The first `/result` route returned a
  `JSONResponse` with a `type: ignore`. It now declares `response_model` and a union return
  type, so both responses are typed and both appear in OpenAPI.
- **Type errors under `mypy --strict`** in tests: a `dict` inferred too loosely, an unused
  `type: ignore`, and indexing raw JSON. The last one was replaced by loading the stored JSON back
  into a `Plan`, which also proves what is stored is a valid plan.
- **A runtime dependency declared only for development.** `llm/gemini.py` imports `httpx` to
  map its timeouts and transport errors, but `httpx` was listed only in the `dev` dependency
  group, where the tests use it as an HTTP client. The Docker image installs without dev
  dependencies, so the API container exited at startup with
  `ModuleNotFoundError: No module named 'httpx'`, whatever `LLM_PROVIDER` was, while all 541
  tests passed in the development environment. Found by the final Docker validation. `httpx`
  is now a runtime dependency (the same 0.28.1 in `uv.lock`), and `scripts/docker_smoke.sh`
  checks that the container starts and answers `GET /health` with 200. It was run on the code
  before the fix and failed with the same error.

## What AI Struggled With _[author to confirm]_

- **Interactions between parts:** each check in validation was right on its own; the bug was how
  the cycle check combined with the one-repair rule in the planner.
- **Trusting its own green tests:** tests passing on the first run said little until a planted
  bug showed they could fail.
- **Tooling side effects:** the formatter rewrote lines between writing code and scripting
  against its exact text.
