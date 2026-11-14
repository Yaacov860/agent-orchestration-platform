import asyncio
import gc
import logging

import pytest

from orchestrator.runner import BackgroundRunner


async def test_a_spawned_job_runs_in_the_background() -> None:
    runner = BackgroundRunner()
    done = asyncio.Event()

    async def job() -> None:
        done.set()

    runner.spawn("job", job)
    assert not done.is_set()  # spawn returns at once; the job runs on the next loop turn

    await runner.wait_idle()
    assert done.is_set()
    assert runner.running == 0


async def test_a_running_job_is_kept_alive_by_the_runner() -> None:
    # The loop holds tasks weakly; without the runner's reference this one could be
    # garbage-collected while it waits.
    runner = BackgroundRunner()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def job() -> None:
        await release.wait()
        finished.set()

    runner.spawn("job", job)
    await asyncio.sleep(0)
    gc.collect()
    assert runner.running == 1

    release.set()
    await runner.wait_idle()
    assert finished.is_set()


async def test_wait_idle_also_waits_for_jobs_started_meanwhile() -> None:
    runner = BackgroundRunner()
    order: list[str] = []

    async def second() -> None:
        order.append("second")

    async def first() -> None:
        order.append("first")
        runner.spawn("second", second)

    runner.spawn("first", first)
    await runner.wait_idle()

    assert order == ["first", "second"]


async def test_a_crashing_job_is_logged_and_does_not_affect_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = BackgroundRunner()
    survivor = asyncio.Event()

    async def crash() -> None:
        raise RuntimeError("boom")

    async def fine() -> None:
        survivor.set()

    runner.spawn("plan:task_1", crash)
    runner.spawn("plan:task_2", fine)
    with caplog.at_level(logging.ERROR, logger="orchestrator.runner"):
        await runner.wait_idle()

    assert survivor.is_set()
    [record] = caplog.records
    assert record.getMessage() == "background job plan:task_1 crashed"
    assert record.exc_info is not None and isinstance(record.exc_info[1], RuntimeError)


async def test_shutdown_cancels_running_jobs() -> None:
    runner = BackgroundRunner()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def forever() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    runner.spawn("forever", forever)
    await started.wait()
    await runner.shutdown()

    assert cancelled.is_set()
    assert runner.running == 0


async def test_nothing_starts_after_shutdown(caplog: pytest.LogCaptureFixture) -> None:
    runner = BackgroundRunner()
    await runner.shutdown()
    calls: list[str] = []

    async def job() -> None:
        calls.append("ran")

    with caplog.at_level(logging.WARNING, logger="orchestrator.runner"):
        runner.spawn("late", job)
    await asyncio.sleep(0)

    assert calls == []
    assert "not starting late" in caplog.text
