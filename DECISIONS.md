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

## 1. Task Decomposition Strategy

## 2. Dependency Management

## 3. Parallel Execution

## 4. Failure Recovery

## 5. One Thing I Would Do Differently With More Time
