# AI Tool Usage

Two different things are documented here: how the **system** uses an LLM (and how tests
replace it), and how an **AI coding assistant** was used to build it. The second part covers
the whole project, stage by stage:

| Stage | What was built | Now in |
|---|---|---|
| 0 | Project skeleton: health endpoint, Docker, tooling | `main` |
| 0.1 | LLM client contract, FakeLLM, BaseAgent, ResearchAgent | `main` |
| 0.2 | Task data model, first migration, `POST /tasks`, `GET /tasks/{id}` | `main` |
| 3 | Planner: `planned` status, dependency graph, plan validation, LLM planner, planning service, background runner | PR1, PR3, PR4, PR7 |
| Hardening | Startup recovery (reschedule `pending`, fail interrupted `planning`), no planning twice, edge-case tests, this document's account of LLM use | PR7, PR8 |
| Execution engine | Group-by-group executor, concurrency cap, retries, cancellation; then `/steps`, input-size limit, `Settings` limits, deadlines | PR4, PR6, PR7, PR8 |
| Synthesis engine | Provenance, draft validation, synthesizer, synthesis service, `/result` | PR5, PR7, PR8 |
| Specialized agents | Writing, analysis and code agents; agent catalog and factory | PR2 |
| Gemini provider | `GeminiLLM`, provider factory, settings | PR6 |
| Readiness | Gemini failures diagnosed and fixed against the real API; first real end-to-end runs; `GET /agents`, trace, token usage, `/metrics`, `/health/ready` | PR1, PR6, PR8 |
| PR split | The finished work organized into 8 reviewable PRs, after the fact; the stack's tree is identical to the working version | - |
| Final validation | Docker build and Compose: the `httpx` runtime dependency fixed, a Docker smoke test added; a paid live run with grounding off | PR6, PR8 |
| Publication fixes | Compose forwards every documented setting; documentation updated | PR6, PR8 |
| PR9 | Bounded Google Search grounding; a live run with grounding on | PR9 |

**Who did what.** Claude wrote the code, the tests and the documentation in every stage,
stages 0–0.2 included. The author wrote no code. The author:
- gave the instructions for each unit and set its scope and constraints;
- decided the questions that were raised along the way;
- approved each plan before work started;
- approved every commit and every paid live run.

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

> **Real API runs (2026-11-14, the key from `.env`: free tier at first, then with billing enabled).**
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
> - **Not verified on that key:** grounded sources (no quota), and therefore
>   evidence-based confidence above 0.3. Both were verified later, with billing enabled (the
>   last item below). `corroborated` findings are still unverified live. Result quality was
>   read by eye on three tasks, not evaluated.
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
>   sources" among its limitations. Grounded search was verified in the next run.
> - **With Google Search grounding on (2026-11-14, billing enabled, bounded grounding):**
>   - **How it ran:** one end-to-end task through the Docker image of the final code, with
>     `GEMINI_SEARCH_GROUNDING=true` and `GEMINI_MAX_GROUNDED_CALLS_PER_TASK=1`.
>   - **Result:** `task_ac4108c471374a8e8217b47e8bfea101`, `completed`, not partial, in
>     34.5 s. Planning took 4.9 s, execution 24.0 s and synthesis 5.6 s.
>   - **Steps:** the plan had two steps in sequence, research then writing. Every call
>     succeeded on its first attempt, with no retry and no fallback model.
>   - **Tokens:** 5,769 in total, as reported by Gemini (2,716 input, 3,053 output).
>   - **Grounding:** the research step made **1 grounded request**, the whole budget, and
>     was marked `grounded`. It returned **5 sources** (Google grounding-redirect links,
>     titled by domain), and its evidence-based confidence was 0.85. The writing step did not
>     search.
>   - **Queries:** Gemini reported **3 search queries** for that request. That is the
>     provider's count, as recorded by the app. The number actually billed was not checked
>     against Google's billing.
>   - **What the result says:** both of its findings rest on that single research step and
>     are labelled not corroborated.
>   - **Not run with grounding on:** `tests/live` and `scripts/gemini_e2e.py`.

**What the model is trusted with.** Nothing it returns is used unchecked. A plan is parsed with
a strict schema and validated against the agent catalog and the dependency graph before it is
stored. A synthesis draft is parsed strictly and checked against the steps that produced
output: no citing a failed or unknown step, and no URL that no step returned. Sources,
"corroborated", confidence and the failure-based limitations are computed by code, never
taken from the model. The user's goal and the step outputs are fenced in prompts as data, but
validation is the real guard.

**What has not been verified.** Every prompt has run against the real model in the
end-to-end tasks listed above: the planner, research, writing, analysis, code and synthesis.
None of them has been evaluated: their results were read by eye on a few tasks.
- **Agent routing.** The tests check that the agent descriptions reach the planner, and that
  wrong routing is caught and sent back. They do not check the model's choices. In the live
  runs, the planner's plans used the research, analysis, writing and code agents. That is an
  observation, not an evaluation.
- **What the tests prove.** The tests prove the logic around the model (parsing, validation, repair, retries,
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

- **Stages 0–0.2:** Claude, model Claude Opus 5.5, wrote the code from the author's
  instructions. The commits carry a `Co-Authored-By: Claude Opus 5.5` trailer.
- **Stage 3 through PR9:** Claude Code (VS Code extension), model Claude Opus 5.5. It was used
  for design proposals, implementation, tests, documentation, the after-the-fact PR split,
  and the validation runs: Docker, Compose, coverage and live API.
- **How the work was done:**
  - Environment checks first, read-only.
  - A written plan per unit, approved before any code.
  - Commits only on approval, and no push without approval.
  - Live API runs only on explicit approval, one task at a time, with the key never printed.
  - Each unit ended with the full suite, `ruff` and `mypy --strict`.

## What Helped Most

- **A plan before code, with decisions surfaced as questions.** Choices with real trade-offs
  went back to the author instead of being made silently. Examples: a separate `planned`
  status versus reusing `planning`; a catalog limited to implemented agents; failing clearly
  when no provider is configured; what startup recovery does to each status; and, in PR9,
  whether search should be capped on requests or on queries.
- **Mutation checks on green tests.** When new tests passed on the first run, a deliberate
  bug was planted to confirm that a test catches it, then removed. Examples: allowing a
  third planner attempt, making status transitions unconditional, storing a plan for a task
  that already failed, removing the concurrency cap. Every planted bug was caught.
- **Deterministic tests for background work.** A `ManualRunner` that only records jobs
  removed the race between `POST /tasks` and the planner in API tests. One end-to-end test
  runs on the real runner.
- **The real API, early and cheaply.** Running against Gemini overturned four assumptions
  that offline tests could not (listed under "Assumptions the real API overturned" below).
  Live runs stayed small:
  - metadata-only model checks before a run;
  - tasks submitted one at a time;
  - one task for each paid run;
  - a grounded-call budget of 1 for the grounding run.
- **Checking against a reference instead of trusting a summary.**
  - The PR split was accepted only when the stack's tree hash equalled the working version's.
  - The Compose defaults were compared, value by value, with `Settings`.
  - Coverage of the PR1/PR2 code was measured, not inferred.
- **Testing the shipped artefact.** Building and running the Docker image found a missing
  runtime dependency that 541 green tests could not see. The smoke test written for it fails
  on the code before the fix.

## What I Had to Fix

Problems in Claude-written code, tests, configuration or documentation. Each was caught by
the checks or by review within the sessions, then fixed by Claude and covered by a test (or,
for configuration, by a script check):

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
- **Documented settings that Docker ignored.** The README documented
  `GEMINI_SEARCH_GROUNDING`, `GEMINI_FALLBACK_MODELS`, the Gemini timeout and the execution
  limits as environment settings. `docker compose up` forwarded only `LLM_PROVIDER`,
  `GEMINI_API_KEY` and `GEMINI_MODEL`. Found when the first live run needed an override file
  just to turn grounding off. Compose now forwards every documented setting, and its
  defaults were compared value by value with `Settings`.

## What AI Struggled With

- **Interactions between parts.** Each check in plan validation was right on its own; the
  bug was how the cycle check combined with the planner's one-repair rule.
- **Trusting its own green tests.** Tests that passed on the first run said little until a
  planted bug showed they could fail.
- **The gap between the development environment and the shipped image.** The suite ran in
  an environment that had `httpx`. The image, installed without dev dependencies, did not.
  This was found only when the image built from the final code was started, in the Docker
  validation.
- **Documentation drifting from configuration.** Settings were documented as configurable
  while the Docker path ignored them. Each was right where it was written; the gap showed
  only when the two were used together.
- **Precision in its own reports.**
  - A summary gave the test count on `main` as 59. It was 51; corrected in the next report.
  - A first coverage check counted calls by name, which matched unrelated methods such as
    `session.add`. It was replaced by an AST analysis and a measured coverage run.
- **Tooling side effects.**
  - The formatter rewrote lines between writing code and scripting against its exact text.
  - zsh expanded `=` inside an `echo`.
  - macOS's system Python (3.9) rejected f-string syntax that the project's Python (3.12)
    accepts. It broke a status-polling loop during the paid grounding run: the task had
    been sent once and the loop only read status, so there was no extra cost.
  - A failed `git stash` in a throwaway worktree let the next command overwrite files there.
    No branch was affected.
- **A library newer than its training data.** google-genai 3.3.0 had to be read from its
  installed source (error classes, timeout units, retries, how the key is sent) rather than
  recalled.
