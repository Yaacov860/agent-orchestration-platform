"""Run a plan's steps in dependency order, in parallel where the plan allows, within a cap.

    for group in parallel_groups:                    (groups come from the stored plan)
        cancelled?                 -> mark the rest cancelled, stop
        deadline passed?           -> mark the rest expired, stop
        a dependency not completed -> skipped
        otherwise                  -> run, all of the group at once (each attempt holds a slot)

A group starts when the previous one has finished. Steps inside a group never depend on each
other, so they run concurrently; the shared semaphore caps how many run at once across every
task in the process.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.factory import AgentFactory, UnknownAgentError
from orchestrator.agents.types import AgentOutput, AgentResult, AgentTask, ContextItem, ErrorInfo
from orchestrator.domain.clock import Clock, seconds_left, utc_now
from orchestrator.execution.types import (
    DEADLINE_EXCEEDED,
    ExecutionJob,
    ExecutionSummary,
    StepOutcome,
    StepRecorder,
    StepSpec,
)
from orchestrator.llm.errors import retry_wait
from orchestrator.llm.types import TokenUsage

logger = logging.getLogger(__name__)

STEP_RETRY_DELAYS = (1.0, 3.0)
"""Seconds before each retry of a step whose failure is retryable: two retries, then it fails."""

STEP_TIMEOUT_SECONDS = 180.0
"""One attempt of one step. A hung provider call must not hold a slot forever."""

MAX_CONTEXT_CHARS = 8_000
"""Characters of one dependency's output passed into a dependent step's prompt (about 2,000
tokens). Characters, not tokens: no tokenizer is available for every provider."""

TRUNCATION_NOTE = "\n\n[... truncated: {omitted} of {total} characters omitted]"


class DependencyExecutor:
    def __init__(
        self,
        agents: AgentFactory,
        *,
        slots: asyncio.Semaphore,
        retry_delays: Sequence[float] = STEP_RETRY_DELAYS,
        step_timeout: float = STEP_TIMEOUT_SECONDS,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Clock = utc_now,
    ) -> None:
        self._agents = agents
        self._slots = slots
        self._retry_delays = tuple(retry_delays)
        self._step_timeout = step_timeout
        self._max_context_chars = max_context_chars
        self._sleep = sleep
        self._clock = clock

    async def run(
        self, job: ExecutionJob, recorder: StepRecorder, cancel: asyncio.Event
    ) -> ExecutionSummary:
        steps = {step.step_id: step for step in job.steps}
        done: dict[str, StepOutcome] = {}
        skipped: dict[str, str] = {}
        cancelled: list[str] = []
        expired: list[str] = []

        for index, group in enumerate(job.parallel_groups):
            rest = [step_id for later in job.parallel_groups[index:] for step_id in later]
            if not cancel.is_set() and await recorder.cancel_requested():
                cancel.set()
            if cancel.is_set():
                await recorder.steps_cancelled(rest)
                cancelled += rest
                break
            if self._remaining(job) == 0:
                await recorder.steps_expired(rest)
                expired += rest
                break

            runnable: list[StepSpec] = []
            blocked: dict[str, str] = {}
            for step_id in group:
                step = steps[step_id]
                blocker = next(
                    (
                        d
                        for d in step.dependencies
                        if d not in done or done[d].status != "completed"
                    ),
                    None,
                )
                if blocker is None:
                    runnable.append(step)
                else:
                    blocked[step_id] = blocker
            if blocked:
                await recorder.steps_skipped(blocked)
                skipped.update(blocked)

            # A TaskGroup, not gather: if one step hits a bug (say, the database is gone), its
            # siblings are cancelled instead of running on, orphaned, after the task failed.
            async with asyncio.TaskGroup() as group_tasks:
                running = [
                    group_tasks.create_task(self._run_step(job, step, done, recorder, cancel))
                    for step in runnable
                ]
            for outcome in (r.result() for r in running):
                done[outcome.step_id] = outcome
                if outcome.status == "cancelled":
                    cancelled.append(outcome.step_id)

        return ExecutionSummary(
            outcomes=tuple(done.values()),
            skipped=skipped,
            cancelled=tuple(cancelled),
            expired=tuple(expired),
        )

    def _remaining(self, job: ExecutionJob) -> float | None:
        """Seconds left before the deadline (0 once it has passed), or None without one."""
        return seconds_left(job.deadline, self._clock())

    async def _run_step(
        self,
        job: ExecutionJob,
        step: StepSpec,
        done: Mapping[str, StepOutcome],
        recorder: StepRecorder,
        cancel: asyncio.Event,
    ) -> StepOutcome:
        try:
            agent = self._agents.get(step.agent)
        except UnknownAgentError as exc:
            # A plan stored before an agent was removed, or a planner/executor mismatch.
            error = ErrorInfo(type=type(exc).__name__, message=str(exc), retryable=False)
            return await self._finish(recorder, step, step.agent, "failed", 0, error=error)

        task = AgentTask(
            step_id=step.step_id,
            action=step.action,
            instruction=step.instruction,
            goal=job.goal,
            constraints=dict(job.constraints),
            # Direct dependencies only: each step declares exactly whose output it needs.
            context=tuple(self._context(done[d], job, step) for d in step.dependencies),
        )

        attempts = 0
        usage = TokenUsage()
        elapsed_ms = 0
        result: AgentResult | None = None
        for delay in (*self._retry_delays, None):
            async with self._slots:
                if cancel.is_set():
                    result = None
                    break
                remaining = self._remaining(job)
                if remaining == 0:
                    # The deadline passed while this step waited for a slot or a retry.
                    result = _deadline_result(agent, step, "before this attempt could start")
                    break
                attempts += 1
                await recorder.step_started(step.step_id, attempts)
                result = await self._attempt(agent, task, cancel, remaining)
            if result is None:  # cancelled while running
                break
            usage = usage + result.usage
            elapsed_ms += result.execution_time_ms
            retry = (
                result.status == "failed" and result.error is not None and result.error.retryable
            )
            if not retry or delay is None:
                break
            assert result.error is not None
            delay = retry_wait(delay, result.error.retry_after_seconds)
            remaining = self._remaining(job)
            if remaining is not None and remaining <= delay:
                # Waiting would only reach the deadline: keep the failure as it is.
                logger.warning(
                    "step %s of task %s: no retry, the deadline is %.1fs away",
                    step.step_id,
                    job.task_id,
                    remaining,
                )
                break
            logger.warning(
                "step %s of task %s failed (attempt %d), retrying in %.1fs: %s",
                step.step_id,
                job.task_id,
                attempts,
                delay,
                result.error.message if result.error else "",
            )
            if await _sleep_unless_cancelled(self._sleep, delay, cancel):
                result = None
                break

        if result is None:
            return await self._finish(
                recorder, step, agent.name, "cancelled", attempts, usage, elapsed_ms
            )
        return await self._finish(
            recorder,
            step,
            agent.name,
            result.status,
            attempts,
            usage,
            elapsed_ms,
            output=result.output,
            error=result.error,
        )

    async def _finish(
        self,
        recorder: StepRecorder,
        step: StepSpec,
        agent: str,
        status: Literal["completed", "failed", "cancelled"],
        attempts: int,
        usage: TokenUsage | None = None,
        elapsed_ms: int = 0,
        *,
        output: AgentOutput | None = None,
        error: ErrorInfo | None = None,
    ) -> StepOutcome:
        outcome = StepOutcome(
            step_id=step.step_id,
            agent=agent,
            status=status,
            attempts=attempts,
            usage=usage or TokenUsage(),
            execution_time_ms=elapsed_ms,
            output=output,
            error=error,
        )
        await recorder.step_finished(outcome)
        return outcome

    async def _attempt(
        self,
        agent: BaseAgent,
        task: AgentTask,
        cancel: asyncio.Event,
        remaining: float | None,
    ) -> AgentResult | None:
        """One run of the agent; None if the task was cancelled meanwhile.

        Bounded by the step timeout or the time left before the deadline, whichever is
        shorter. Only the agent call is cancelled, never a database write: the step's state
        is always recorded afterwards, by the caller, from a coroutine nobody interrupts.
        """
        by_deadline = remaining is not None and remaining < self._step_timeout
        limit = remaining if remaining is not None and by_deadline else self._step_timeout
        started = time.perf_counter()
        run = asyncio.create_task(agent.run(task), name=f"agent:{task.step_id}")
        stop = asyncio.create_task(cancel.wait())
        try:
            async with asyncio.timeout(limit):
                await asyncio.wait({run, stop}, return_when=asyncio.FIRST_COMPLETED)
        except TimeoutError:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            if by_deadline:
                return _deadline_result(agent, task, "while it was running", elapsed_ms)
            return AgentResult(
                step_id=task.step_id,
                agent=agent.name,
                status="failed",
                error=ErrorInfo(
                    type="StepTimeoutError",
                    message=f"step did not finish within {self._step_timeout:g}s",
                    retryable=True,
                ),
                # The run's token meter is lost with the cancelled run: tokens unknown.
                execution_time_ms=elapsed_ms,
            )
        finally:
            stop.cancel()
            if not run.done():
                run.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run
        return run.result() if not run.cancelled() else None

    def _context(self, outcome: StepOutcome, job: ExecutionJob, step: StepSpec) -> ContextItem:
        """A dependency's output as input for `step`, cut to the configured size.

        The beginning is kept (agents put their findings first) and the cut is marked, so the
        model knows text is missing. The stored output itself stays complete.
        """
        assert outcome.output is not None, "only completed dependencies reach this point"
        content = outcome.output.content
        if len(content) > self._max_context_chars:
            omitted = len(content) - self._max_context_chars
            logger.info(
                "task %s: input from %s to %s truncated (%d of %d characters omitted)",
                job.task_id,
                outcome.step_id,
                step.step_id,
                omitted,
                len(content),
            )
            content = content[: self._max_context_chars] + TRUNCATION_NOTE.format(
                omitted=omitted, total=len(content)
            )
        return ContextItem(step_id=outcome.step_id, agent=outcome.agent, content=content)


def _deadline_result(
    agent: BaseAgent, step: StepSpec | AgentTask, when: str, elapsed_ms: int = 0
) -> AgentResult:
    return AgentResult(
        step_id=step.step_id,
        agent=agent.name,
        status="failed",
        # Not retryable: there is no time left to retry in.
        error=ErrorInfo(
            type=DEADLINE_EXCEEDED, message=f"the task's deadline passed {when}", retryable=False
        ),
        execution_time_ms=elapsed_ms,
    )


async def _sleep_unless_cancelled(
    sleep: Callable[[float], Awaitable[None]], seconds: float, cancel: asyncio.Event
) -> bool:
    """Sleep before a retry; True if the task was cancelled during the wait."""
    nap = asyncio.ensure_future(sleep(seconds))
    stop = asyncio.create_task(cancel.wait())
    try:
        await asyncio.wait({nap, stop}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in (nap, stop):
            waiter.cancel()
    return cancel.is_set()
