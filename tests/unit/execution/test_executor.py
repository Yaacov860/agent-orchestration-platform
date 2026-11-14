"""DependencyExecutor on its own: an in-memory recorder, the real ResearchAgent, a scripted LLM."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fakes import GatedLLM, StepLLM, output_of, step_of

from orchestrator.agents.factory import AgentFactory
from orchestrator.execution import (
    DependencyExecutor,
    ExecutionJob,
    ExecutionSummary,
    StepOutcome,
    StepSpec,
)
from orchestrator.llm import (
    LLMClient,
    LLMPermanentError,
    LLMTransientError,
)
from orchestrator.llm.fake import fake_response
from orchestrator.planning.graph import parallel_groups


def step(step_id: str, *dependencies: str, agent: str = "research") -> StepSpec:
    return StepSpec(
        step_id=step_id,
        agent=agent,
        action="research",
        instruction=f"do {step_id}",
        dependencies=dependencies,
    )


def job(*steps: StepSpec, task_id: str = "task_1") -> ExecutionJob:
    groups = parallel_groups({s.step_id: s.dependencies for s in steps})
    return ExecutionJob(
        task_id=task_id,
        goal="Compare Python and JavaScript",
        constraints={"tone": "friendly"},
        steps=steps,
        parallel_groups=tuple(tuple(g) for g in groups),
    )


class Recorder:
    """In-memory StepRecorder: an ordered log of what the executor reported."""

    def __init__(self, *, cancel_after: str | None = None) -> None:
        self.events: list[tuple[object, ...]] = []
        self.outcomes: dict[str, StepOutcome] = {}
        self._cancel_after = cancel_after
        self._cancelled_elsewhere = False

    async def step_started(self, step_id: str, attempt: int) -> None:
        self.events.append(("started", step_id, attempt))

    async def step_finished(self, outcome: StepOutcome) -> None:
        self.events.append(("finished", outcome.step_id, outcome.status))
        self.outcomes[outcome.step_id] = outcome
        if outcome.step_id == self._cancel_after:
            self._cancelled_elsewhere = True  # as if another process cancelled the task now

    async def steps_skipped(self, blocked_by: Mapping[str, str]) -> None:
        self.events.append(("skipped", dict(blocked_by)))

    async def steps_expired(self, step_ids: Sequence[str]) -> None:
        self.events.append(("expired", list(step_ids)))

    async def steps_cancelled(self, step_ids: Sequence[str]) -> None:
        self.events.append(("cancelled", list(step_ids)))

    async def cancel_requested(self) -> bool:
        return self._cancelled_elsewhere


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def executor(
    llm: LLMClient,
    *,
    slots: int = 4,
    retry_delays: Sequence[float] = (1.0, 3.0),
    step_timeout: float = 5.0,
    max_context_chars: int = 8_000,
    sleep: Sleeps | None = None,
) -> DependencyExecutor:
    return DependencyExecutor(
        AgentFactory(llm),
        slots=asyncio.Semaphore(slots),
        retry_delays=retry_delays,
        step_timeout=step_timeout,
        max_context_chars=max_context_chars,
        sleep=sleep or Sleeps(),
    )


async def run(
    llm: LLMClient,
    the_job: ExecutionJob,
    recorder: Recorder | None = None,
    cancel: asyncio.Event | None = None,
    **options: object,
) -> ExecutionSummary:
    return await executor(llm, **options).run(  # type: ignore[arg-type]
        the_job, recorder or Recorder(), cancel or asyncio.Event()
    )


def statuses(summary: ExecutionSummary) -> dict[str, str]:
    return {o.step_id: o.status for o in summary.outcomes}


# --- order and data flow -----------------------------------------------------------------


async def test_dependents_start_after_their_dependencies_and_receive_their_outputs() -> None:
    llm = GatedLLM(output_of)
    llm.release.set()
    recorder = Recorder()

    summary = await run(
        llm, job(step("step_1"), step("step_2"), step("step_3", "step_1", "step_2")), recorder
    )

    assert summary.succeeded
    assert statuses(summary) == {
        "step_1": "completed",
        "step_2": "completed",
        "step_3": "completed",
    }
    finished = [e[1] for e in recorder.events if e[0] == "finished"]
    started_3 = recorder.events.index(("started", "step_3", 1))
    assert all(
        recorder.events.index(("finished", s, "completed")) < started_3
        for s in ("step_1", "step_2")
    )
    assert finished[-1] == "step_3"

    [summary_request] = [r for r in llm.calls if step_of(r) == "step_3"]
    prompt = summary_request.user_prompt
    assert "## Input from step_1 (research agent)\noutput of step_1" in prompt
    assert "## Input from step_2 (research agent)\noutput of step_2" in prompt
    assert summary.outcomes[-1].output is not None
    assert summary.outcomes[-1].output.content == "output of step_3"


async def test_only_direct_dependencies_are_passed_as_input() -> None:
    llm = GatedLLM(output_of)
    llm.release.set()

    await run(llm, job(step("step_1"), step("step_2", "step_1"), step("step_3", "step_2")))

    [last] = [r for r in llm.calls if step_of(r) == "step_3"]
    assert "Input from step_2" in last.user_prompt
    assert "Input from step_1" not in last.user_prompt  # it declared only step_2


async def test_the_goal_and_constraints_reach_every_step() -> None:
    llm = GatedLLM(output_of)
    llm.release.set()

    await run(llm, job(step("step_1")))

    assert "Compare Python and JavaScript" in llm.calls[0].user_prompt
    assert '{"tone": "friendly"}' in llm.calls[0].user_prompt


# --- parallelism and the cap -------------------------------------------------------------


async def test_the_steps_of_a_group_really_run_at_the_same_time() -> None:
    llm = GatedLLM(output_of)
    running = asyncio.create_task(run(llm, job(step("step_1"), step("step_2"), step("step_3"))))

    await llm.until_in_flight(3)  # all three are inside the model at once
    llm.release.set()

    assert (await running).succeeded


async def test_the_cap_holds_across_steps_and_jobs_sharing_one_executor() -> None:
    llm = GatedLLM(output_of)
    shared = executor(llm, slots=2)
    jobs = [
        job(step("step_1"), step("step_2"), step("step_3"), task_id=f"task_{n}") for n in (1, 2)
    ]
    running = [asyncio.create_task(shared.run(j, Recorder(), asyncio.Event())) for j in jobs]

    await llm.until_in_flight(2)
    await asyncio.sleep(0.05)  # room for a third call to wrongly start
    assert llm.in_flight == 2

    llm.release.set()
    results = await asyncio.gather(*running)
    assert all(r.succeeded for r in results)
    assert llm.peak == 2
    assert len(llm.calls) == 6


# --- failures ----------------------------------------------------------------------------


async def test_a_failed_step_skips_its_dependents_transitively_but_not_independent_steps() -> None:
    llm = StepLLM({"step_1": [LLMPermanentError("content blocked")]})
    recorder = Recorder()
    the_job = job(
        step("step_1"),
        step("step_2"),
        step("step_3", "step_1"),
        step("step_4", "step_3"),
        step("step_5", "step_2"),
    )

    summary = await run(llm, the_job, recorder)

    assert statuses(summary) == {"step_1": "failed", "step_2": "completed", "step_5": "completed"}
    assert summary.skipped == {"step_3": "step_1", "step_4": "step_3"}
    assert sorted(llm.calls) == ["step_1", "step_2", "step_5"]  # skipped steps cost nothing
    assert not summary.succeeded
    failed = recorder.outcomes["step_1"]
    assert failed.error is not None
    assert (failed.error.type, failed.error.retryable, failed.attempts) == (
        "LLMPermanentError",
        False,
        1,
    )


async def test_a_transient_failure_is_retried_and_every_attempts_tokens_count() -> None:
    sleeps = Sleeps()
    llm = StepLLM(
        {
            "step_1": [
                LLMTransientError("429 rate limited"),
                fake_response("found it", input_tokens=40, output_tokens=10),
            ]
        }
    )
    recorder = Recorder()

    summary = await run(llm, job(step("step_1")), recorder, sleep=sleeps)

    [outcome] = summary.outcomes
    assert (outcome.status, outcome.attempts) == ("completed", 2)
    assert outcome.usage.total == 50
    assert sleeps.delays == [1.0]
    assert [e for e in recorder.events if e[0] == "started"] == [
        ("started", "step_1", 1),
        ("started", "step_1", 2),
    ]


async def test_retries_stop_after_the_last_delay() -> None:
    sleeps = Sleeps()
    llm = StepLLM({"step_1": [LLMTransientError("timeout")] * 3})

    summary = await run(llm, job(step("step_1"), step("step_2", "step_1")), sleep=sleeps)

    [outcome] = summary.failed
    assert (outcome.step_id, outcome.attempts) == ("step_1", 3)
    assert sleeps.delays == [1.0, 3.0]
    assert summary.skipped == {"step_2": "step_1"}


async def test_a_permanent_failure_is_not_retried() -> None:
    sleeps = Sleeps()

    summary = await run(
        StepLLM({"step_1": [LLMPermanentError("bad key")]}), job(step("step_1")), sleep=sleeps
    )

    assert summary.failed[0].attempts == 1
    assert sleeps.delays == []


async def test_a_hung_step_times_out_as_a_retryable_failure_and_its_call_is_cancelled() -> None:
    llm = GatedLLM()  # never released: the provider hangs

    summary = await run(llm, job(step("step_1")), retry_delays=(), step_timeout=0.05)

    [outcome] = summary.failed
    assert outcome.error is not None
    assert (outcome.error.type, outcome.error.retryable) == ("StepTimeoutError", True)
    assert outcome.error.message == "step did not finish within 0.05s"
    assert (llm.cancelled, llm.in_flight) == (1, 0)  # the hung call does not linger


async def test_a_step_for_an_agent_that_does_not_exist_fails_without_running() -> None:
    llm = StepLLM()

    summary = await run(llm, job(step("step_1", agent="translation"), step("step_2", "step_1")))

    [outcome] = summary.failed
    assert outcome.error is not None
    assert (outcome.attempts, outcome.error.type) == (0, "UnknownAgentError")
    assert "no agent named 'translation'" in outcome.error.message
    assert summary.skipped == {"step_2": "step_1"}
    assert llm.calls == []


# --- cancellation ------------------------------------------------------------------------


async def test_a_task_cancelled_before_it_starts_runs_nothing() -> None:
    llm = StepLLM()
    recorder = Recorder()
    cancel = asyncio.Event()
    cancel.set()

    summary = await run(llm, job(step("step_1"), step("step_2", "step_1")), recorder, cancel)

    assert summary.cancelled == ("step_1", "step_2")
    assert recorder.events == [("cancelled", ["step_1", "step_2"])]
    assert llm.calls == []


async def test_a_cancel_stops_the_running_agents_and_everything_after_them() -> None:
    llm = GatedLLM(output_of)  # never released: the steps are mid-thought when cancelled
    recorder = Recorder()
    cancel = asyncio.Event()
    the_job = job(step("step_1"), step("step_2"), step("step_3", "step_1", "step_2"))
    running = asyncio.create_task(run(llm, the_job, recorder, cancel))

    await llm.until_in_flight(2)
    cancel.set()
    summary = await running

    assert statuses(summary) == {"step_1": "cancelled", "step_2": "cancelled"}
    assert set(summary.cancelled) == {"step_1", "step_2", "step_3"}
    assert (llm.cancelled, llm.in_flight) == (2, 0)
    assert recorder.events[-1] == ("cancelled", ["step_3"])


async def test_a_cancel_from_another_process_is_noticed_before_the_next_group() -> None:
    llm = StepLLM()
    recorder = Recorder(cancel_after="step_1")

    summary = await run(llm, job(step("step_1"), step("step_2", "step_1")), recorder)

    assert statuses(summary) == {"step_1": "completed"}
    assert summary.cancelled == ("step_2",)
    assert llm.calls == ["step_1"]


async def test_a_cancel_during_the_wait_before_a_retry_stops_the_step() -> None:
    cancel = asyncio.Event()

    async def cancelled_while_waiting(seconds: float) -> None:
        cancel.set()
        await asyncio.Event().wait()  # would wait forever without the cancel

    llm = StepLLM({"step_1": [LLMTransientError("429")]})
    summary = await executor(llm, sleep=cancelled_while_waiting).run(  # type: ignore[arg-type]
        job(step("step_1")), Recorder(), cancel
    )

    [outcome] = summary.outcomes
    assert (outcome.status, outcome.attempts) == ("cancelled", 1)


async def test_shutting_down_mid_step_propagates_and_leaves_no_agent_running() -> None:
    llm = GatedLLM()
    running = asyncio.create_task(run(llm, job(step("step_1"), step("step_2"))))
    await llm.until_in_flight(2)

    running.cancel()  # what BackgroundRunner.shutdown does

    with pytest.raises(asyncio.CancelledError):
        await running
    assert (llm.cancelled, llm.in_flight) == (2, 0)


async def test_a_bug_in_one_step_cancels_its_siblings_instead_of_orphaning_them() -> None:
    llm = GatedLLM()

    class BrokenRecorder(Recorder):
        async def step_started(self, step_id: str, attempt: int) -> None:
            if step_id == "step_2":
                raise RuntimeError("database gone")
            await super().step_started(step_id, attempt)

    with pytest.raises(ExceptionGroup) as caught:
        await run(llm, job(step("step_1"), step("step_2")), BrokenRecorder())

    assert caught.group_contains(RuntimeError, match="database gone")
    assert llm.in_flight == 0
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("agent:")]


# --- the size of inputs passed between steps ---------------------------------------------


async def test_a_long_dependency_output_is_cut_for_the_next_step_but_kept_whole() -> None:
    long_output = "A" * 40 + "B" * 60  # 100 characters
    llm = StepLLM({"step_1": [fake_response(long_output)]})

    summary = await executor(llm, max_context_chars=40).run(
        job(step("step_1"), step("step_2", "step_1")), Recorder(), asyncio.Event()
    )

    first = summary.outcomes[0]
    assert first.output is not None and first.output.content == long_output  # stored whole
    assert llm.prompts["step_2"].count("A") == 40
    assert "B" not in llm.prompts["step_2"]
    assert "A" * 40 + "\n\n[... truncated: 60 of 100 characters omitted]" in llm.prompts["step_2"]


async def test_an_output_at_the_limit_is_passed_unchanged() -> None:
    llm = StepLLM({"step_1": [fake_response("x" * 40)]})

    await executor(llm, max_context_chars=40).run(
        job(step("step_1"), step("step_2", "step_1")), Recorder(), asyncio.Event()
    )

    assert "x" * 40 in llm.prompts["step_2"]
    assert "truncated" not in llm.prompts["step_2"]


# --- deadlines ---------------------------------------------------------------------------


def soon(seconds: float) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=seconds)


def with_deadline(the_job: ExecutionJob, deadline: datetime) -> ExecutionJob:
    return replace(the_job, deadline=deadline)


async def test_a_deadline_already_passed_runs_nothing() -> None:
    llm = StepLLM()
    recorder = Recorder()
    the_job = with_deadline(job(step("step_1"), step("step_2", "step_1")), soon(-1))

    summary = await run(llm, the_job, recorder)

    assert summary.expired == ("step_1", "step_2")
    assert summary.deadline_exceeded and not summary.succeeded
    assert recorder.events == [("expired", ["step_1", "step_2"])]
    assert llm.calls == []


async def test_a_deadline_passing_mid_step_stops_it_and_nothing_after_it_starts() -> None:
    llm = GatedLLM(output_of)  # never released
    the_job = with_deadline(job(step("step_1"), step("step_2", "step_1")), soon(0.2))

    summary = await run(llm, the_job, step_timeout=30.0)

    [stopped] = summary.outcomes
    assert stopped.error is not None
    assert (stopped.status, stopped.attempts) == ("failed", 1)
    assert (stopped.error.type, stopped.error.retryable) == ("DeadlineExceeded", False)
    assert stopped.error.message == "the task's deadline passed while it was running"
    assert summary.expired == ("step_2",)
    assert (llm.cancelled, llm.in_flight) == (1, 0)


async def test_the_step_timeout_still_applies_when_the_deadline_is_further_away() -> None:
    llm = GatedLLM()
    the_job = with_deadline(job(step("step_1")), soon(30))

    summary = await run(llm, the_job, retry_delays=(), step_timeout=0.05)

    [outcome] = summary.failed
    assert outcome.error is not None and outcome.error.type == "StepTimeoutError"
    assert not summary.deadline_exceeded


async def test_no_retry_is_started_when_its_wait_would_reach_the_deadline() -> None:
    sleeps = Sleeps()
    llm = StepLLM({"step_1": [LLMTransientError("429 rate limited")]})
    the_job = with_deadline(job(step("step_1")), soon(5))

    summary = await run(llm, the_job, retry_delays=(60.0,), sleep=sleeps)

    [outcome] = summary.failed
    assert outcome.attempts == 1
    assert outcome.error is not None and outcome.error.message == "429 rate limited"
    assert sleeps.delays == []


async def test_a_step_still_waiting_for_a_slot_at_the_deadline_never_starts() -> None:
    llm = GatedLLM(output_of)
    the_job = with_deadline(job(step("step_1"), step("step_2")), soon(0.2))

    summary = await run(llm, the_job, slots=1, step_timeout=30.0)

    by_id = {o.step_id: o for o in summary.outcomes}
    assert by_id["step_1"].attempts == 1  # was running at the deadline
    assert by_id["step_2"].attempts == 0  # got the slot only after it
    assert by_id["step_2"].error is not None
    assert (
        by_id["step_2"].error.message
        == "the task's deadline passed before this attempt could start"
    )
    assert len(llm.calls) == 1


async def test_a_retry_waits_as_long_as_the_provider_asked_within_the_cap() -> None:
    sleeps = Sleeps()
    llm = StepLLM(
        {
            "step_1": [
                LLMTransientError("429", retry_after=31.0),
                LLMTransientError("429", retry_after=600.0),  # beyond the cap
                fake_response("done"),
            ]
        }
    )

    summary = await run(llm, job(step("step_1")), sleep=sleeps)

    assert summary.outcomes[0].status == "completed"
    assert sleeps.delays == [31.0, 60.0]  # the planned 1 s and 3 s, stretched; 600 capped


def test_retry_wait_never_shortens_a_planned_delay() -> None:
    from orchestrator.llm import retry_wait

    assert retry_wait(5.0, None) == 5.0
    assert retry_wait(5.0, 2.0) == 5.0
    assert retry_wait(5.0, 30.0) == 30.0
    assert retry_wait(90.0, 600.0) == 90.0  # a planned delay above the cap is kept
