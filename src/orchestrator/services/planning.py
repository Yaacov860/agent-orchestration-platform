"""Planning use cases: plan a submitted task in the background, and recover after a restart.

    pending ──claim──► planning ──plan stored──► planned
                          │
                          └──no valid plan / error──► failed

Three short transactions, none of them open while the model thinks (tens of seconds):
  1. claim:  pending -> planning, committed at once, so pollers see `planning` immediately;
  2. plan:   the LLM call, with no database connection held;
  3. store:  the plan, its steps and planning -> planned, all or nothing.

Planning twice is impossible by construction, not by care: the claim is a conditional
UPDATE, so however often a task gets scheduled (by POST, by recovery, by two processes),
only one claim can succeed and everyone else finds it no longer pending.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.db.repositories import TaskRepository
from orchestrator.domain.clock import Clock, seconds_left, utc_now
from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus
from orchestrator.llm import retry_wait
from orchestrator.llm.types import TokenUsage
from orchestrator.planning import Planner, PlanningError, PlanResult
from orchestrator.runner import Runner
from orchestrator.services.usage import usage_detail

logger = logging.getLogger(__name__)

TRANSIENT_RETRY_DELAYS = (2.0, 5.0)
"""Seconds to wait before each retry of a transient LLM failure (rate limit, timeout): two
retries, then the task fails. A permanent failure or an invalid plan is never retried."""

INTERNAL_ERROR = "internal error during planning"
"""What the client sees for a bug on our side; the details go to the server log only."""

INTERRUPTED_ERROR = "interrupted: the server restarted while this task was being planned"
DEADLINE_BEFORE_PLANNING = "deadline passed before planning started"
DEADLINE_DURING_PLANNING = "deadline passed during planning"

MAX_CONCURRENT_PLANS = 4
"""Plans computed at once by this process. Without a cap, a restart that finds a backlog of
pending tasks (or a burst of submissions) would fire every LLM call at the same moment and
mostly collect rate-limit errors."""


@dataclass(frozen=True)
class RecoveryReport:
    failed: Sequence[str]
    """Tasks a previous process left in `planning`: interrupted mid-plan, now `failed`."""
    rescheduled: Sequence[str]
    """Tasks still `pending`: planning never started, so it is simply started now."""


class PlanningService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        planner: Planner,
        runner: Runner,
        *,
        retry_delays: Sequence[float] = TRANSIENT_RETRY_DELAYS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_concurrent: int = MAX_CONCURRENT_PLANS,
        on_planned: Callable[[str], None] = lambda task_id: None,
        clock: Clock = utc_now,
    ) -> None:
        self._clock = clock
        self._on_planned = on_planned
        """Called with the task id once its plan is committed (the app: start execution).
        A callback, so planning knows nothing about the execution engine."""
        self._sessionmaker = sessionmaker
        self._planner = planner
        self._runner = runner
        self._retry_delays = tuple(retry_delays)
        self._sleep = sleep
        self._slots = asyncio.Semaphore(max_concurrent)
        self._scheduled: set[str] = set()
        """Tasks with a planning job queued or running in this process."""

    def schedule(self, task_id: str) -> None:
        """Plan the task in the background. Returns at once.

        Scheduling a task that already has a job here is a no-op. That is only a saving:
        correctness comes from the claim, which also covers other processes.
        """
        if task_id in self._scheduled:
            logger.info("task %s is already scheduled for planning", task_id)
            return
        self._scheduled.add(task_id)
        self._runner.spawn(f"plan:{task_id}", lambda: self._run(task_id))

    async def _run(self, task_id: str) -> None:
        try:
            await self.plan_task(task_id)
        finally:
            self._scheduled.discard(task_id)

    async def recover(self) -> RecoveryReport:
        """At startup, settle what a previous process left behind.

        `planning` -> `failed`: a plan was in progress and its outcome is unknown (decision:
        fail it visibly rather than silently start over). `pending` -> scheduled again:
        nothing had started, so planning it now is exactly what would have happened.
        Failing first matters: the rescheduled jobs must not race this UPDATE.

        Assumes a single API process (DECISIONS.md 4): another live replica's in-flight
        plans would be failed. Rescheduling, by contrast, is safe with any number of
        processes, because only one claim per task can succeed.
        """
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            failed = await tasks.fail_all_in(
                TaskStatus.PLANNING, INTERRUPTED_ERROR, FailureStage.PLANNING
            )
            await session.commit()
            pending = await tasks.ids_in(TaskStatus.PENDING)
        if failed:
            logger.warning(
                "marked %d interrupted task(s) as failed: %s", len(failed), ", ".join(failed)
            )
        for task_id in pending:
            self.schedule(task_id)
        if pending:
            logger.info("rescheduled %d pending task(s) for planning", len(pending))
        return RecoveryReport(failed=failed, rescheduled=pending)

    async def plan_task(self, task_id: str) -> None:
        """Plan one pending task. Never raises, except cancellation (shutdown)."""
        # Wait for a slot BEFORE claiming: a queued task stays `pending`, so if the process
        # dies while it waits, recovery reschedules it instead of failing it.
        async with self._slots:
            await self._plan_claimed(task_id)

    async def _plan_claimed(self, task_id: str) -> None:
        try:
            task = await self._claim(task_id)
            if task is None:
                logger.info("task %s is not pending (or gone); not planning it", task_id)
                return
            remaining = seconds_left(task.deadline, self._clock())
            if remaining == 0:
                logger.warning("task %s: deadline passed before planning started", task_id)
                await self._fail(task_id, DEADLINE_BEFORE_PLANNING)
                return
            try:
                # The planner call (and its retries) may use only the time that is left.
                async with asyncio.timeout(remaining):
                    result = await self._plan_with_retries(task)
            except TimeoutError:
                logger.warning("task %s: deadline passed during planning", task_id)
                await self._fail(task_id, DEADLINE_DURING_PLANNING)
                return
            except PlanningError as exc:
                logger.warning("planning task %s failed: %s", task_id, exc)
                await self._fail(task_id, str(exc), exc.usage)
                return
            await self._store(task_id, result)
        except Exception:
            # A bug or an unreachable database. If the database is down, failing the task
            # fails too; it then stays `planning` and startup recovery takes care of it.
            logger.exception("unexpected error while planning task %s", task_id)
            try:
                await self._fail(task_id, INTERNAL_ERROR)
            except Exception:
                logger.exception("could not mark task %s as failed", task_id)

    async def _claim(self, task_id: str) -> Task | None:
        async with self._sessionmaker() as session:
            task = await TaskRepository(session).transition(
                task_id, expected=TaskStatus.PENDING, to=TaskStatus.PLANNING
            )
            await session.commit()
            return task

    async def _plan_with_retries(self, task: Task) -> PlanResult:
        """The planner, retried on transient failures. Tokens and attempts add up over every
        try, the failed ones included: they were paid for, and the plan row records them."""
        spent, calls = TokenUsage(), 0
        for attempt, delay in enumerate((*self._retry_delays, None), start=1):
            try:
                result = await self._planner.plan(
                    goal=task.goal, constraints=task.constraints, output_format=task.output_format
                )
            except PlanningError as exc:
                spent, calls = spent + exc.usage, calls + exc.attempts
                if not exc.retryable or delay is None:
                    raise PlanningError(
                        str(exc),
                        retryable=exc.retryable,
                        usage=spent,
                        attempts=calls,
                        problems=exc.problems,
                        retry_after=exc.retry_after,
                    ) from exc
                wait = retry_wait(delay, exc.retry_after)
                logger.warning(
                    "planning task %s: transient failure (try %d), retrying in %.1fs: %s",
                    task.id,
                    attempt,
                    wait,
                    exc,
                )
                await self._sleep(wait)
                continue
            logger.info(
                "planned task %s: %d steps, %d planner attempt(s), %d tokens",
                task.id,
                len(result.validated.plan.steps),
                result.attempts,
                result.usage.total,
            )
            return replace(result, usage=spent + result.usage, attempts=calls + result.attempts)
        raise AssertionError("unreachable: the last try either returns or raises")

    async def _store(self, task_id: str, result: PlanResult) -> None:
        async with self._sessionmaker() as session:
            tasks = TaskRepository(session)
            # Transition first: if the task left `planning` meanwhile (recovery failed it,
            # or later a user cancelled it), the plan is discarded, not attached to it.
            planned = await tasks.transition(
                task_id, expected=TaskStatus.PLANNING, to=TaskStatus.PLANNED
            )
            if planned is None:
                logger.warning(
                    "task %s left 'planning' while it was planned; plan discarded", task_id
                )
                await session.rollback()
                return
            version = await tasks.next_plan_version(task_id)
            tasks.add_plan(_plan_rows(task_id, version, result))
            await session.commit()
        # After the commit: the executor reads the plan in its own session and must find it.
        self._on_planned(task_id)

    async def _fail(self, task_id: str, error: str, usage: TokenUsage | None = None) -> None:
        async with self._sessionmaker() as session:
            await TaskRepository(session).transition(
                task_id,
                expected=TaskStatus.PLANNING,
                to=TaskStatus.FAILED,
                error=error,
                failure_stage=FailureStage.PLANNING,
                detail=usage_detail(usage),
            )
            await session.commit()


def _plan_rows(task_id: str, version: int, result: PlanResult) -> ExecutionPlan:
    """The validated plan as rows: the JSON record of what was planned, plus one mutable
    runtime row per step (DECISIONS.md 0.2), plus what planning it cost."""
    validated = result.validated
    plan = ExecutionPlan(
        task_id=task_id,
        version=version,
        plan=validated.plan.model_dump(mode="json"),
        parallel_groups=[list(group) for group in validated.parallel_groups],
        planner_input_tokens=result.usage.input_tokens,
        planner_output_tokens=result.usage.output_tokens,
        planner_attempts=result.attempts,
    )
    plan.steps = [
        ExecutionStep(
            step_id=step.step_id,
            position=position,
            agent=step.agent,
            action=step.action,
            input=step.input,
            dependencies=list(step.dependencies),
            status=StepStatus.PENDING,
        )
        for position, step in enumerate(validated.plan.steps)
    ]
    return plan
