"""Execution use cases: run a planned task, cancel it, and recover after a restart.

    planned ──claim──► executing ──at least one step completed──► synthesizing (on_executed)
                           │
                           ├──no step completed, or the deadline passed──► failed (execution)
                           └──cancelled (POST /tasks/{id}/cancel)──► cancelled

Execution never marks a task completed: only synthesis does, once the result is stored.

Like planning, nothing holds a database transaction while an agent runs: every state change
is its own short transaction, and every one is a conditional UPDATE.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.agents.factory import AgentFactory
from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.db.repositories import StepRepository, TaskRepository
from orchestrator.domain.clock import Clock, utc_now
from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus
from orchestrator.execution import (
    DependencyExecutor,
    ExecutionJob,
    ExecutionSummary,
    StepOutcome,
    StepSpec,
)
from orchestrator.execution.executor import (
    MAX_CONTEXT_CHARS,
    STEP_RETRY_DELAYS,
    STEP_TIMEOUT_SECONDS,
)
from orchestrator.execution.types import DEADLINE_EXCEEDED
from orchestrator.runner import Runner

logger = logging.getLogger(__name__)

MAX_CONCURRENT_STEPS = 4
"""Agent runs at once in this process, across all tasks: the LLM's rate limit is shared."""

MAX_GROUNDED_CALLS_PER_TASK = 2
"""Provider calls with web search attached, per task: a hard limit on grounded requests (every
attempt and model fallback takes a slot), not on Google queries, which the model chooses."""

INTERNAL_ERROR = "internal error during execution"
INTERRUPTED_ERROR = "interrupted: the server restarted while this task was being executed"
NO_PLAN_ERROR = "no plan to execute"
NO_USABLE_RESULT = "no step produced a usable result"
DEADLINE_BEFORE_EXECUTION = "deadline passed before execution started"
DEADLINE_DURING_EXECUTION = "deadline passed during execution"

_INTERRUPTED_STEP = {"type": "Interrupted", "message": INTERRUPTED_ERROR, "retryable": False}
_EXPIRED_STEP = {
    "type": DEADLINE_EXCEEDED,
    "message": "not run: the task's deadline passed first",
    "retryable": False,
}

_FINISHED_FROM: dict[StepStatus, tuple[StepStatus, ...]] = {
    StepStatus.COMPLETED: (StepStatus.RUNNING,),  # only a step that ran can complete
    # pending: a step whose agent does not exist fails without ever running
    StepStatus.FAILED: (StepStatus.PENDING, StepStatus.RUNNING),
    # pending: cancelled before its first attempt took a slot
    StepStatus.CANCELLED: (StepStatus.PENDING, StepStatus.RUNNING),
}
"""Which states a step may finish from, by how it finished."""


@dataclass(frozen=True)
class ExecutionRecoveryReport:
    failed: Sequence[str]
    """Tasks left `executing` by a previous process: their outcome is unknown, now `failed`."""
    rescheduled: Sequence[str]
    """Tasks still `planned`: execution never started, so it is started now."""


class ExecutionService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        agents: AgentFactory,
        runner: Runner,
        *,
        max_concurrent_steps: int = MAX_CONCURRENT_STEPS,
        retry_delays: Sequence[float] = STEP_RETRY_DELAYS,
        step_timeout: float = STEP_TIMEOUT_SECONDS,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_grounded_calls_per_task: int = MAX_GROUNDED_CALLS_PER_TASK,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Clock = utc_now,
        on_executed: Callable[[str], None] = lambda task_id: None,
    ) -> None:
        self._on_executed = on_executed
        """Called with the task id once it moved to `synthesizing` (the app: start synthesis).
        A callback, so execution knows nothing about the synthesis engine."""
        self._sessionmaker = sessionmaker
        self._runner = runner
        self._executor = DependencyExecutor(
            agents,
            slots=asyncio.Semaphore(max_concurrent_steps),
            retry_delays=retry_delays,
            step_timeout=step_timeout,
            max_context_chars=max_context_chars,
            sleep=sleep,
            clock=clock,
            max_grounded_calls=max_grounded_calls_per_task,
        )
        self._scheduled: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}
        """Executions running in this process, so a cancel here stops them at once."""

    # --- scheduling ----------------------------------------------------------------------

    def schedule(self, task_id: str) -> None:
        """Execute a planned task in the background. Returns at once. Safe to repeat: only one
        claim per task can succeed, in this process or any other."""
        if task_id in self._scheduled:
            return
        self._scheduled.add(task_id)
        self._runner.spawn(f"execute:{task_id}", lambda: self._run(task_id))

    async def _run(self, task_id: str) -> None:
        try:
            await self.execute_task(task_id)
        finally:
            self._scheduled.discard(task_id)

    def cancel_requested(self, task_id: str) -> None:
        """The task was just cancelled in the database: stop its execution here, if any.
        An execution in another process notices at its next group (see the executor)."""
        if (event := self._cancel_events.get(task_id)) is not None:
            event.set()

    # --- one execution -------------------------------------------------------------------

    async def execute_task(self, task_id: str) -> None:
        """Execute one planned task. Never raises, except cancellation (shutdown)."""
        # Registered before the claim, so a cancel arriving right after the claim is not missed.
        cancel = self._cancel_events.setdefault(task_id, asyncio.Event())
        try:
            job = await self._claim(task_id)
            if job is None:
                return
            recorder = DbStepRecorder(self._sessionmaker, task_id, job.plan_id)
            summary = await self._executor.run(job.job, recorder, cancel)
            await self._finish(task_id, summary)
        except Exception:
            logger.exception("unexpected error while executing task %s", task_id)
            try:
                await self._fail(task_id, INTERNAL_ERROR)
            except Exception:
                logger.exception("could not mark task %s as failed", task_id)
        finally:
            self._cancel_events.pop(task_id, None)

    async def _claim(self, task_id: str) -> "_ClaimedJob | None":
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            task = await tasks.transition(
                task_id, expected=TaskStatus.PLANNED, to=TaskStatus.EXECUTING
            )
            if task is None:
                logger.info("task %s is not planned (or gone); not executing it", task_id)
                return None
            plan = await tasks.latest_plan(task_id)
            if plan is None or not plan.steps:
                # Cannot happen through the API (planned means a plan was stored); a stray
                # row must not leave the task `executing` forever.
                await tasks.transition(
                    task_id,
                    expected=TaskStatus.EXECUTING,
                    to=TaskStatus.FAILED,
                    error=NO_PLAN_ERROR,
                    failure_stage=FailureStage.EXECUTION,
                )
                await session.commit()
                return None
            await session.commit()
            return _ClaimedJob(plan_id=plan.id, job=_job(task, plan))

    async def _finish(self, task_id: str, summary: ExecutionSummary) -> None:
        """Hand over to synthesis, or fail. What counts:

        - cancelled: nothing to do, the canceller already set `cancelled`;
        - the deadline passed: no time left to synthesize, so `failed`;
        - not a single step completed: nothing to synthesize from, so `failed`;
        - otherwise `synthesizing`, even if some steps failed or were skipped: the result
          will be partial, and say what is missing (DECISIONS.md 5).
        """
        if summary.was_cancelled:
            logger.info("task %s cancelled; %d step(s) stopped", task_id, len(summary.cancelled))
            return
        completed = [o for o in summary.outcomes if o.status == "completed"]
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            if completed and not summary.deadline_exceeded:
                moved = await tasks.transition(
                    task_id, expected=TaskStatus.EXECUTING, to=TaskStatus.SYNTHESIZING
                )
            else:
                error = _failure_message(summary)
                if not completed and not summary.deadline_exceeded:
                    error = f"{NO_USABLE_RESULT}: {error}"
                moved = await tasks.transition(
                    task_id,
                    expected=TaskStatus.EXECUTING,
                    to=TaskStatus.FAILED,
                    error=error,
                    failure_stage=FailureStage.EXECUTION,
                )
            await session.commit()
        if moved is None:
            # Cancelled after the last step finished: the cancel stands.
            logger.info("task %s left 'executing' before it finished; leaving it", task_id)
            return
        logger.info(
            "task %s %s: %d of %d step(s) completed, %d tokens",
            task_id,
            moved.status,
            len(completed),
            len(summary.outcomes) + len(summary.skipped) + len(summary.expired),
            sum(o.usage.total for o in summary.outcomes),
        )
        if moved.status is TaskStatus.SYNTHESIZING:
            self._on_executed(task_id)

    async def _fail(self, task_id: str, error: str) -> None:
        async with self._sessionmaker() as session:
            await TaskRepository(session).transition(
                task_id,
                expected=TaskStatus.EXECUTING,
                to=TaskStatus.FAILED,
                error=error,
                failure_stage=FailureStage.EXECUTION,
            )
            await session.commit()

    # --- startup recovery ----------------------------------------------------------------

    async def recover(self) -> ExecutionRecoveryReport:
        """At startup: fail what was interrupted mid-execution, restart what never started.

        `executing` -> `failed`; its running steps -> `failed` (interrupted), its waiting
        steps -> `cancelled`. Steps that had completed keep their results.
        `planned` -> scheduled again.

        Same single-process assumption as planning recovery (DECISIONS.md 4).
        """
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            steps = StepRepository(session)
            failed = await tasks.fail_all_in(
                TaskStatus.EXECUTING, INTERRUPTED_ERROR, FailureStage.EXECUTION
            )
            await steps.move_all_of_tasks(
                failed,
                expected=StepStatus.RUNNING,
                to=StepStatus.FAILED,
                error=_INTERRUPTED_STEP,
                completed_at=func.now(),
            )
            await steps.move_all_of_tasks(
                failed, expected=StepStatus.PENDING, to=StepStatus.CANCELLED
            )
            await session.commit()
            planned = await tasks.ids_in(TaskStatus.PLANNED)
        if failed:
            logger.warning(
                "marked %d interrupted execution(s) as failed: %s", len(failed), ", ".join(failed)
            )
        for task_id in planned:
            self.schedule(task_id)
        if planned:
            logger.info("rescheduled %d planned task(s) for execution", len(planned))
        return ExecutionRecoveryReport(failed=failed, rescheduled=planned)


class DbStepRecorder:
    """StepRecorder backed by execution_steps: one short transaction per state change."""

    def __init__(
        self, sessionmaker: async_sessionmaker[AsyncSession], task_id: str, plan_id: UUID
    ) -> None:
        self._sessionmaker = sessionmaker
        self._task_id = task_id
        self._plan_id = plan_id

    async def step_started(self, step_id: str, attempt: int) -> None:
        async with self._sessionmaker() as session:
            await StepRepository(session).move(
                self._plan_id,
                [step_id],
                expected=[StepStatus.PENDING, StepStatus.RUNNING],
                to=StepStatus.RUNNING,
                attempts=attempt,
                # The first attempt's start: a retry must not move it.
                started_at=func.coalesce(ExecutionStep.started_at, func.now()),
            )
            await session.commit()

    async def step_finished(self, outcome: StepOutcome) -> None:
        values: dict[str, object] = {
            "attempts": outcome.attempts,
            "input_tokens": outcome.usage.input_tokens,
            "output_tokens": outcome.usage.output_tokens,
            "grounded_calls": outcome.search.grounded_calls,
            "search_queries": outcome.search.queries,
            "execution_time_ms": outcome.execution_time_ms,
            "completed_at": func.now(),
            "output": outcome.output.model_dump(mode="json") if outcome.output else None,
            # exclude_none: an error without a retry hint keeps its plain three-field shape.
            "error": outcome.error.model_dump(mode="json", exclude_none=True)
            if outcome.error
            else None,
        }
        to = StepStatus(outcome.status)
        async with self._sessionmaker() as session:
            await StepRepository(session).move(
                self._plan_id,
                [outcome.step_id],
                expected=_FINISHED_FROM[to],
                to=to,
                **values,
            )
            await session.commit()

    async def steps_skipped(self, blocked_by: Mapping[str, str]) -> None:
        async with self._sessionmaker() as session:
            steps = StepRepository(session)
            for step_id, blocker in blocked_by.items():
                await steps.move(
                    self._plan_id,
                    [step_id],
                    expected=[StepStatus.PENDING],
                    to=StepStatus.SKIPPED,
                    error={
                        "type": "DependencyNotCompleted",
                        "message": f"not run: depends on {blocker}, which did not complete",
                        "retryable": False,
                    },
                )
            await session.commit()

    async def steps_expired(self, step_ids: Sequence[str]) -> None:
        async with self._sessionmaker() as session:
            await StepRepository(session).move(
                self._plan_id,
                step_ids,
                expected=[StepStatus.PENDING],
                to=StepStatus.SKIPPED,
                error=_EXPIRED_STEP,
            )
            await session.commit()

    async def steps_cancelled(self, step_ids: Sequence[str]) -> None:
        async with self._sessionmaker() as session:
            await StepRepository(session).move(
                self._plan_id, step_ids, expected=[StepStatus.PENDING], to=StepStatus.CANCELLED
            )
            await session.commit()

    async def cancel_requested(self) -> bool:
        async with self._sessionmaker() as session:
            status = await session.scalar(select(Task.status).where(Task.id == self._task_id))
        return status is TaskStatus.CANCELLED


@dataclass(frozen=True)
class _ClaimedJob:
    plan_id: UUID
    job: ExecutionJob


def _job(task: Task, plan: ExecutionPlan) -> ExecutionJob:
    """The stored plan rows as the executor's input. Plain data, no planner types."""
    return ExecutionJob(
        task_id=task.id,
        goal=task.goal,
        constraints=dict(task.constraints),
        steps=tuple(
            StepSpec(
                step_id=row.step_id,
                agent=row.agent,
                action=row.action,
                instruction=row.input,
                dependencies=tuple(row.dependencies),
            )
            for row in plan.steps
        ),
        parallel_groups=tuple(tuple(group) for group in plan.parallel_groups),
        deadline=task.deadline,
    )


def _failure_message(summary: ExecutionSummary) -> str:
    """One line for `task.error`: `step_1 failed: content blocked; not run: step_3`."""
    if summary.deadline_exceeded and not any(o.attempts for o in summary.outcomes):
        return DEADLINE_BEFORE_EXECUTION
    parts = [DEADLINE_DURING_EXECUTION] if summary.deadline_exceeded else []
    parts += [
        f"{o.step_id} failed: {o.error.message if o.error else 'unknown error'}"
        for o in summary.failed
    ]
    not_run = [*summary.skipped, *summary.expired]
    if not_run:
        parts.append(f"not run: {', '.join(not_run)}")
    return "; ".join(parts)
