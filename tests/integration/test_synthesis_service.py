"""SynthesisService against the real database: the stored result, atomic completion, failures,
timeouts, deadlines, cancellation, duplicates and recovery."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from factories import make_runnable_plan, make_task, synthesis_json
from fakes import GatedLLM, ManualRunner, is_synthesis
from pydantic import JsonValue
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.db.models import ExecutionStep, Task, TaskResult
from orchestrator.db.repositories import ResultRepository, TaskRepository
from orchestrator.domain.status import FailureStage, StepStatus, TaskStatus
from orchestrator.llm import (
    LLMClient,
    LLMPermanentError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.runner import Runner
from orchestrator.services.synthesis import (
    DEADLINE_BEFORE_SYNTHESIS,
    DEADLINE_DURING_SYNTHESIS,
    INTERNAL_ERROR,
    SynthesisService,
)
from orchestrator.services.tasks import TaskService
from orchestrator.synthesis import Synthesizer

Sessions = async_sessionmaker[AsyncSession]

DIAMOND = [("step_1", ()), ("step_2", ()), ("step_3", ("step_1", "step_2"))]
PY: JsonValue = {"title": "Python docs", "url": "https://docs.python.org"}
MDN: JsonValue = {"title": "MDN", "url": "https://developer.mozilla.org"}
GOOD = synthesis_json(
    "Python is the easier first language.",
    findings=[
        ("Python reads like English", ["step_1"]),
        ("Both are taught widely", ["step_1", "step_2"]),
    ],
)


def service(
    sessionmaker: Sessions, llm: LLMClient, runner: Runner | None = None, **options: object
) -> SynthesisService:
    async def no_sleep(seconds: float) -> None:
        return None

    return SynthesisService(
        sessionmaker,
        Synthesizer(llm),
        runner or ManualRunner(),
        sleep=no_sleep,
        **options,  # type: ignore[arg-type]
    )


async def synthesizing_task(
    sessionmaker: Sessions,
    *,
    status: TaskStatus = TaskStatus.SYNTHESIZING,
    deadline: datetime | None = None,
    failed_steps: Sequence[str] = (),
) -> str:
    """A task whose plan ran: every step completed with an output, except `failed_steps`."""
    task = make_task(status=status, goal="Compare Python and JavaScript", deadline=deadline)
    async with sessionmaker() as session:
        session.add(task)
        await session.flush()
        plan = make_runnable_plan(task.id, DIAMOND)
        sources: dict[str, list[JsonValue]] = {"step_1": [PY], "step_2": [MDN], "step_3": []}
        for step in plan.steps:
            if step.step_id in failed_steps:
                step.status = StepStatus.FAILED
                step.error = {
                    "type": "LLMPermanentError",
                    "message": "content blocked",
                    "retryable": False,
                }
            else:
                step.status = StepStatus.COMPLETED
                step.attempts = 1
                step.output = {
                    "content": f"output of {step.step_id}",
                    "sources": sources[step.step_id],
                    "confidence": 0.85 if step.step_id != "step_3" else None,
                }
        session.add(plan)
        await session.commit()
    return task.id


async def state(
    sessionmaker: Sessions, task_id: str
) -> tuple[TaskStatus, str | None, FailureStage | None]:
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
    assert task is not None
    return task.status, task.error, task.failure_stage


async def result_of(sessionmaker: Sessions, task_id: str) -> TaskResult | None:
    async with sessionmaker() as session:
        return await ResultRepository(session).get(task_id)


async def result_count(sessionmaker: Sessions) -> int:
    async with sessionmaker() as session:
        return await session.scalar(select(func.count()).select_from(TaskResult)) or 0


# --- success -----------------------------------------------------------------------------


async def test_a_synthesized_result_is_stored_with_its_provenance_and_the_task_completed(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = FakeLLM([fake_response(GOOD, input_tokens=900, output_tokens=300)])

    await service(sessionmaker, llm).synthesize_task(task_id)

    assert await state(sessionmaker, task_id) == (TaskStatus.COMPLETED, None, None)
    row = await result_of(sessionmaker, task_id)
    assert row is not None
    assert row.content == "Python is the easier first language."
    assert row.key_findings == [
        {
            "statement": "Python reads like English",
            "step_ids": ["step_1"],
            "verification": "single_step",
        },
        {
            "statement": "Both are taught widely",
            "step_ids": ["step_1", "step_2"],
            "verification": "corroborated",
        },
    ]
    assert row.sources == [
        {"title": "Python docs", "url": "https://docs.python.org", "step_ids": ["step_1"]},
        {"title": "MDN", "url": "https://developer.mozilla.org", "step_ids": ["step_2"]},
    ]
    assert (row.confidence, row.partial, row.conflicts) == (0.85, False, [])
    assert row.synthesized_from == ["step_1", "step_2", "step_3"]
    assert (row.input_tokens, row.output_tokens, row.attempts) == (900, 300, 1)
    async with sessionmaker() as session:
        plan = await TaskRepository(session).latest_plan(task_id)
    assert plan is not None and row.plan_id == plan.id


async def test_a_partial_result_is_stored_with_what_is_missing(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker, failed_steps=["step_2"])
    llm = FakeLLM(
        [
            fake_response(
                synthesis_json("Only Python was covered.", findings=[("readable", ["step_1"])])
            )
        ]
    )

    await service(sessionmaker, llm).synthesize_task(task_id)

    row = await result_of(sessionmaker, task_id)
    assert row is not None and row.partial is True
    assert row.limitations[0] == (
        "step_2 (research: research) failed: content blocked. "
        "Its findings are missing from this result."
    )
    assert row.synthesized_from == ["step_1", "step_3"]


# --- the task is completed only together with its result ---------------------------------


async def test_if_storing_the_result_fails_the_task_is_not_completed(
    sessionmaker: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = await synthesizing_task(sessionmaker)

    def broken_add(self: ResultRepository, result: TaskResult) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(ResultRepository, "add", broken_add)
    await service(sessionmaker, FakeLLM([fake_response(GOOD)])).synthesize_task(task_id)

    # The completed move was in the same transaction as the insert: both rolled back.
    assert await state(sessionmaker, task_id) == (
        TaskStatus.FAILED,
        INTERNAL_ERROR,
        FailureStage.SYNTHESIS,
    )
    assert await result_of(sessionmaker, task_id) is None


async def test_the_database_refuses_a_second_result_for_a_task(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker)
    await service(sessionmaker, FakeLLM([fake_response(GOOD)])).synthesize_task(task_id)
    first = await result_of(sessionmaker, task_id)
    assert first is not None

    async with sessionmaker() as session:
        session.add(
            TaskResult(
                task_id=task_id,
                plan_id=first.plan_id,
                content="second",
                key_findings=[],
                sources=[],
                conflicts=[],
                limitations=[],
                confidence=None,
                confidence_basis="",
                partial=False,
                synthesized_from=[],
                input_tokens=0,
                output_tokens=0,
                attempts=1,
            )
        )
        with pytest.raises(IntegrityError, match="uq_task_results_task_id"):
            await session.commit()


# --- failures ----------------------------------------------------------------------------


async def test_a_draft_invalid_after_the_repair_fails_synthesis_and_stores_nothing(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = FakeLLM(
        [fake_response("not json"), fake_response(synthesis_json(findings=[("x", ["step_9"])]))]
    )

    await service(sessionmaker, llm).synthesize_task(task_id)

    status, error, stage = await state(sessionmaker, task_id)
    assert (status, stage) == (TaskStatus.FAILED, FailureStage.SYNTHESIS)
    assert error is not None and error.startswith(
        "synthesis failed: no valid result after 2 attempts"
    )
    assert await result_of(sessionmaker, task_id) is None
    async with sessionmaker() as session:  # the step outputs are untouched
        statuses = (await session.scalars(select(ExecutionStep.status))).all()
    assert set(statuses) == {StepStatus.COMPLETED}


async def test_a_permanent_llm_error_fails_synthesis(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker)

    await service(sessionmaker, FakeLLM([LLMPermanentError("invalid API key")])).synthesize_task(
        task_id
    )

    assert await state(sessionmaker, task_id) == (
        TaskStatus.FAILED,
        "synthesis failed: LLM call failed during synthesis: invalid API key",
        FailureStage.SYNTHESIS,
    )


async def test_a_transient_llm_error_is_retried(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = FakeLLM([LLMTransientError("429"), fake_response(GOOD)])

    await service(sessionmaker, llm).synthesize_task(task_id)

    assert (await state(sessionmaker, task_id))[0] is TaskStatus.COMPLETED
    assert len(llm.calls) == 2


async def test_a_synthesis_that_takes_too_long_times_out(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = GatedLLM()  # never answers

    await service(sessionmaker, llm, timeout=0.05).synthesize_task(task_id)

    assert await state(sessionmaker, task_id) == (
        TaskStatus.FAILED,
        "synthesis timed out after 0.05s",
        FailureStage.SYNTHESIS,
    )
    assert (llm.cancelled, llm.in_flight) == (1, 0)
    assert await result_of(sessionmaker, task_id) is None


async def test_a_deadline_passed_before_synthesis_fails_it_without_a_model_call(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(
        sessionmaker, deadline=datetime.now(UTC) - timedelta(seconds=1)
    )
    llm = FakeLLM()

    await service(sessionmaker, llm).synthesize_task(task_id)

    assert await state(sessionmaker, task_id) == (
        TaskStatus.FAILED,
        DEADLINE_BEFORE_SYNTHESIS,
        FailureStage.SYNTHESIS,
    )
    assert llm.calls == []


async def test_a_deadline_reached_during_synthesis_stops_it(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(
        sessionmaker, deadline=datetime.now(UTC) + timedelta(seconds=0.2)
    )
    llm = GatedLLM()

    await service(sessionmaker, llm, timeout=30).synthesize_task(task_id)

    assert await state(sessionmaker, task_id) == (
        TaskStatus.FAILED,
        DEADLINE_DURING_SYNTHESIS,
        FailureStage.SYNTHESIS,
    )


@pytest.mark.parametrize(
    "status", [TaskStatus.EXECUTING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
async def test_a_task_that_is_not_synthesizing_is_left_alone(
    sessionmaker: Sessions, status: TaskStatus
) -> None:
    task_id = await synthesizing_task(sessionmaker, status=status)
    llm = FakeLLM()

    await service(sessionmaker, llm).synthesize_task(task_id)

    assert llm.calls == []
    assert (await state(sessionmaker, task_id))[0] is status


# --- cancellation ------------------------------------------------------------------------


class _NoPlanning:
    def schedule(self, task_id: str) -> None:
        raise AssertionError("not used here")


async def test_cancelling_during_synthesis_stops_the_model_and_writes_no_result(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = GatedLLM()
    synthesis = service(sessionmaker, llm)
    running = asyncio.create_task(synthesis.synthesize_task(task_id))
    await llm.until_in_flight(1)

    async with sessionmaker() as session:
        await TaskService(session, _NoPlanning(), synthesis).cancel(task_id)
    await running

    assert (await state(sessionmaker, task_id))[0] is TaskStatus.CANCELLED
    assert (llm.cancelled, llm.in_flight) == (1, 0)
    assert await result_of(sessionmaker, task_id) is None


async def test_a_cancel_from_another_process_mid_synthesis_still_prevents_the_result(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)

    class CancelledWhileWriting:
        async def generate(self, request: LLMRequest) -> LLMResponse:
            async with sessionmaker() as session:  # no in-process signal: only the DB changes
                await session.execute(
                    text("UPDATE tasks SET status = 'cancelled' WHERE id = :id"), {"id": task_id}
                )
                await session.commit()
            return fake_response(GOOD)

    await service(sessionmaker, CancelledWhileWriting()).synthesize_task(task_id)

    assert (await state(sessionmaker, task_id))[0] is TaskStatus.CANCELLED
    assert await result_of(sessionmaker, task_id) is None


async def test_a_task_cancelled_before_synthesis_starts_is_never_synthesized(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    runner = ManualRunner()
    llm = FakeLLM()
    synthesis = service(sessionmaker, llm, runner)
    synthesis.schedule(task_id)  # queued, not started

    async with sessionmaker() as session:
        await TaskService(session, _NoPlanning(), synthesis).cancel(task_id)
    await runner.run_all()

    assert llm.calls == []
    assert (await state(sessionmaker, task_id))[0] is TaskStatus.CANCELLED


# --- duplicates, concurrency and recovery ------------------------------------------------


async def test_two_processes_synthesizing_one_task_store_exactly_one_result(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = FakeLLM([fake_response(GOOD), fake_response(GOOD)])  # both get to call the model

    await asyncio.gather(
        service(sessionmaker, llm).synthesize_task(task_id),
        service(sessionmaker, llm).synthesize_task(task_id),
    )

    assert await result_count(sessionmaker) == 1
    assert (await state(sessionmaker, task_id))[0] is TaskStatus.COMPLETED


async def test_recovery_reruns_an_interrupted_synthesis_once_per_task(
    sessionmaker: Sessions,
) -> None:
    task_id = await synthesizing_task(sessionmaker)
    done_id = await synthesizing_task(sessionmaker, status=TaskStatus.COMPLETED)
    runner = ManualRunner()
    synthesis = service(sessionmaker, FakeLLM([fake_response(GOOD)]), runner)

    report = await synthesis.recover()
    await synthesis.recover()  # twice in one process: still one job

    assert list(report.rescheduled) == [task_id]
    assert runner.names == [f"synthesize:{task_id}"]
    await runner.run_all()
    assert (await state(sessionmaker, task_id))[0] is TaskStatus.COMPLETED
    assert (await state(sessionmaker, done_id))[0] is TaskStatus.COMPLETED
    assert await result_count(sessionmaker) == 1


async def test_recovery_in_two_processes_still_stores_one_result(sessionmaker: Sessions) -> None:
    task_id = await synthesizing_task(sessionmaker)
    llm = FakeLLM([fake_response(GOOD), fake_response(GOOD)])
    a, b = ManualRunner(), ManualRunner()

    await asyncio.gather(
        service(sessionmaker, llm, a).recover(), service(sessionmaker, llm, b).recover()
    )
    await asyncio.gather(a.run_all(), b.run_all())

    assert await result_count(sessionmaker) == 1
    assert (await state(sessionmaker, task_id))[0] is TaskStatus.COMPLETED


async def test_the_synthesis_cap_holds_across_tasks(sessionmaker: Sessions) -> None:
    task_ids = [await synthesizing_task(sessionmaker) for _ in range(3)]

    class Gate(GatedLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            assert is_synthesis(request)
            await super().generate(request)
            return fake_response(GOOD)

    llm = Gate()
    synthesis = service(sessionmaker, llm, max_concurrent=1)
    running = [asyncio.create_task(synthesis.synthesize_task(t)) for t in task_ids]
    await llm.until_in_flight(1)
    await asyncio.sleep(0.05)
    assert llm.in_flight == 1

    llm.release.set()
    await asyncio.gather(*running)
    assert llm.peak == 1
    assert await result_count(sessionmaker) == 3
