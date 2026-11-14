"""Synthesis use cases: turn an executed task's step outputs into its final, stored result.

    synthesizing ──result stored──► completed     (one transaction: result row + status)
          │
          ├──invalid output twice / LLM error / timeout / deadline──► failed (synthesis)
          └──cancelled (POST /tasks/{id}/cancel)──► cancelled         (no result written)

A task is completed only by this service, and only in the transaction that stores its result:
the conditional `synthesizing -> completed` UPDATE and the INSERT commit together or not at
all, and UNIQUE(task_id) makes a second result impossible even across processes.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.agents.types import AgentOutput
from orchestrator.db.models import ExecutionPlan, Task, TaskResult
from orchestrator.db.repositories import ResultRepository, TaskRepository
from orchestrator.domain.clock import Clock, seconds_left, utc_now
from orchestrator.domain.status import FailureStage, TaskStatus
from orchestrator.llm import retry_wait
from orchestrator.llm.types import TokenUsage
from orchestrator.runner import Runner
from orchestrator.services.usage import usage_detail
from orchestrator.synthesis import (
    StepEvidence,
    SynthesisError,
    SynthesisInput,
    SynthesisOutcome,
    Synthesizer,
)

logger = logging.getLogger(__name__)


SYNTHESIS_TIMEOUT_SECONDS = 120.0
"""The whole synthesis of one task: its model calls, repair and retries included."""

MAX_CONCURRENT_SYNTHESES = 4
TRANSIENT_RETRY_DELAYS = (2.0, 5.0)
"""Same policy as planning: two retries of a transient LLM failure, inside the timeout."""

INTERNAL_ERROR = "internal error during synthesis"
DEADLINE_BEFORE_SYNTHESIS = "deadline passed before synthesis started"
DEADLINE_DURING_SYNTHESIS = "deadline passed during synthesis"


@dataclass(frozen=True)
class SynthesisRecoveryReport:
    rescheduled: Sequence[str]
    """Tasks left `synthesizing` by a previous process. Their inputs are the stored step
    outputs and synthesis writes nothing until it succeeds, so it is simply run again."""


class SynthesisService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        synthesizer: Synthesizer,
        runner: Runner,
        *,
        timeout: float = SYNTHESIS_TIMEOUT_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_SYNTHESES,
        retry_delays: Sequence[float] = TRANSIENT_RETRY_DELAYS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Clock = utc_now,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._synthesizer = synthesizer
        self._runner = runner
        self._timeout = timeout
        self._slots = asyncio.Semaphore(max_concurrent)
        self._retry_delays = tuple(retry_delays)
        self._sleep = sleep
        self._clock = clock
        self._scheduled: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}

    # --- scheduling and cancellation -----------------------------------------------------

    def schedule(self, task_id: str) -> None:
        """Synthesize a task in the background. Safe to repeat (see `_store`)."""
        if task_id in self._scheduled:
            return
        self._scheduled.add(task_id)
        self._runner.spawn(f"synthesize:{task_id}", lambda: self._run(task_id))

    async def _run(self, task_id: str) -> None:
        try:
            await self.synthesize_task(task_id)
        finally:
            self._scheduled.discard(task_id)

    def cancel_requested(self, task_id: str) -> None:
        """The task was cancelled in the database: stop its synthesis here, if running."""
        if (event := self._cancel_events.get(task_id)) is not None:
            event.set()

    # --- one synthesis -------------------------------------------------------------------

    async def synthesize_task(self, task_id: str) -> None:
        """Synthesize one task in `synthesizing`. Never raises, except cancellation."""
        async with self._slots:
            cancel = self._cancel_events.setdefault(task_id, asyncio.Event())
            try:
                await self._synthesize(task_id, cancel)
            except Exception:
                logger.exception("unexpected error while synthesizing task %s", task_id)
                try:
                    await self._fail(task_id, INTERNAL_ERROR)
                except Exception:
                    logger.exception("could not mark task %s as failed", task_id)
            finally:
                self._cancel_events.pop(task_id, None)

    async def _synthesize(self, task_id: str, cancel: asyncio.Event) -> None:
        loaded = await self._load(task_id)
        if loaded is None:
            return
        task, plan_id, synthesis_input = loaded

        remaining = seconds_left(task.deadline, self._clock())
        if remaining == 0:
            await self._fail(task_id, DEADLINE_BEFORE_SYNTHESIS)
            return
        by_deadline = remaining is not None and remaining < self._timeout
        limit = remaining if remaining is not None and by_deadline else self._timeout

        try:
            async with asyncio.timeout(limit):
                outcome = await _unless_cancelled(self._with_retries(synthesis_input), cancel)
        except TimeoutError:
            error = (
                DEADLINE_DURING_SYNTHESIS
                if by_deadline
                else f"synthesis timed out after {limit:g}s"
            )
            logger.warning("task %s: %s", task_id, error)
            await self._fail(task_id, error)
            return
        except SynthesisError as exc:
            logger.warning("synthesis of task %s failed: %s", task_id, exc)
            await self._fail(task_id, f"synthesis failed: {exc}", exc.usage)
            return
        if outcome is None:
            logger.info("task %s cancelled during synthesis; no result written", task_id)
            return
        await self._store(task_id, plan_id, outcome)

    async def _with_retries(self, synthesis_input: SynthesisInput) -> SynthesisOutcome:
        """Retried on transient failures; tokens and model calls add up over every try."""
        spent, calls = TokenUsage(), 0
        for attempt, delay in enumerate((*self._retry_delays, None), start=1):
            try:
                outcome = await self._synthesizer.synthesize(synthesis_input)
            except SynthesisError as exc:
                spent, calls = spent + exc.usage, calls + exc.attempts
                if not exc.retryable or delay is None:
                    raise SynthesisError(
                        str(exc),
                        retryable=exc.retryable,
                        usage=spent,
                        attempts=calls,
                        problems=exc.problems,
                        retry_after=exc.retry_after,
                    ) from exc
                logger.warning(
                    "synthesis of task %s: transient failure (try %d), retrying in %.1fs: %s",
                    synthesis_input.task_id,
                    attempt,
                    retry_wait(delay, exc.retry_after),
                    exc,
                )
                await self._sleep(retry_wait(delay, exc.retry_after))
                continue
            return replace(outcome, usage=spent + outcome.usage, attempts=calls + outcome.attempts)
        raise AssertionError("unreachable: the last try either returns or raises")

    async def _load(self, task_id: str) -> tuple[Task, UUID, SynthesisInput] | None:
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            task = await tasks.get(task_id)
            if task is None or task.status is not TaskStatus.SYNTHESIZING:
                logger.info("task %s is not synthesizing (or gone); nothing to do", task_id)
                return None
            plan = await tasks.latest_plan(task_id)
        if plan is None:
            await self._fail(task_id, "no plan to synthesize from")
            return None
        return task, plan.id, _input(task, plan)

    async def _store(self, task_id: str, plan_id: UUID, outcome: SynthesisOutcome) -> None:
        """The result and `completed`, in one transaction, or neither."""
        async with self._sessionmaker() as session:
            completed = await TaskRepository(session).transition(
                task_id, expected=TaskStatus.SYNTHESIZING, to=TaskStatus.COMPLETED
            )
            if completed is None:
                # Cancelled meanwhile, or another process stored the result first: either
                # way this result must not be written.
                logger.info("task %s left 'synthesizing' first; result discarded", task_id)
                await session.rollback()
                return
            ResultRepository(session).add(_result_row(task_id, plan_id, outcome))
            await session.commit()
        result = outcome.result
        logger.info(
            "task %s completed: %d finding(s), %d source(s), %d conflict(s), partial=%s",
            task_id,
            len(result.key_findings),
            len(result.sources),
            len(result.conflicts),
            result.partial,
        )

    async def _fail(self, task_id: str, error: str, usage: TokenUsage | None = None) -> None:
        async with self._sessionmaker() as session:
            await TaskRepository(session).transition(
                task_id,
                expected=TaskStatus.SYNTHESIZING,
                to=TaskStatus.FAILED,
                error=error,
                failure_stage=FailureStage.SYNTHESIS,
                detail=usage_detail(usage),
            )
            await session.commit()

    # --- startup recovery ----------------------------------------------------------------

    async def recover(self) -> SynthesisRecoveryReport:
        """At startup, run again every synthesis a previous process left unfinished.

        Unlike planning and execution (failed when interrupted), synthesis is re-run: its
        inputs are durable step rows and it writes nothing before it succeeds, so running it
        again is exactly what would have happened. Two processes doing so cannot store two
        results: only one `synthesizing -> completed` move succeeds.
        """
        async with self._sessionmaker() as session:
            pending = await TaskRepository(session).ids_in(TaskStatus.SYNTHESIZING)
        for task_id in pending:
            self.schedule(task_id)
        if pending:
            logger.info("rescheduled %d interrupted synthesis(es)", len(pending))
        return SynthesisRecoveryReport(rescheduled=pending)


_T = TypeVar("_T")


async def _unless_cancelled(work: Coroutine[Any, Any, _T], cancel: asyncio.Event) -> _T | None:
    """Run `work`; None if the task is cancelled first (the work is then cancelled too)."""
    run = asyncio.ensure_future(work)
    stop = asyncio.create_task(cancel.wait())
    try:
        await asyncio.wait({run, stop}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.cancel()
        if not run.done():
            run.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run
    return run.result() if not run.cancelled() else None


def _input(task: Task, plan: ExecutionPlan) -> SynthesisInput:
    """The stored rows as synthesis input. Plain data: no ORM objects leave this module."""
    steps = []
    for row in plan.steps:
        output = AgentOutput.model_validate(row.output) if row.output else None
        error = row.error.get("message") if row.error else None
        steps.append(
            StepEvidence(
                step_id=row.step_id,
                agent=row.agent,
                action=row.action,
                instruction=row.input,
                dependencies=tuple(row.dependencies),
                status=row.status.value,
                content=output.content if output else None,
                sources=output.sources if output else (),
                confidence=output.confidence if output else None,
                error=str(error) if error is not None else None,
                grounding_status=output.grounding.status if output and output.grounding else None,
                grounding_reason=output.grounding.reason if output and output.grounding else None,
            )
        )
    return SynthesisInput(
        task_id=task.id,
        goal=task.goal,
        output_format=task.output_format,
        constraints=dict(task.constraints),
        steps=tuple(steps),
    )


def _result_row(task_id: str, plan_id: UUID, outcome: SynthesisOutcome) -> TaskResult:
    result = outcome.result.model_dump(mode="json")
    return TaskResult(
        task_id=task_id,
        plan_id=plan_id,
        content=result["content"],
        key_findings=result["key_findings"],
        sources=result["sources"],
        conflicts=result["conflicts"],
        limitations=result["limitations"],
        confidence=result["confidence"],
        confidence_basis=result["confidence_basis"],
        partial=result["partial"],
        synthesized_from=result["synthesized_from"],
        input_tokens=outcome.usage.input_tokens,
        output_tokens=outcome.usage.output_tokens,
        attempts=outcome.attempts,
    )
