"""Running work in the background of the API process (asyncio, no broker: DECISIONS.md 0).

Why not FastAPI's BackgroundTasks: those are tied to one request and nothing tracks them
afterwards. Long work needs an owner that (1) keeps a reference to every running task, since
the event loop holds tasks only weakly and an unreferenced one can be garbage-collected
mid-run, (2) logs a job that crashed instead of losing the error, and (3) cancels everything
on shutdown. The executor (stage 4) will run on the same runner.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

logger = logging.getLogger(__name__)

type Job = Callable[[], Coroutine[Any, Any, None]]
"""A job is a coroutine *factory*: nothing starts until the runner calls it, so a runner
that never runs a job (a test double, or one shutting down) leaves no coroutine un-awaited."""


class Runner(Protocol):
    def spawn(self, name: str, job: Job) -> None: ...

    async def shutdown(self) -> None: ...


class BackgroundRunner:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def running(self) -> int:
        return len(self._tasks)

    def spawn(self, name: str, job: Job) -> None:
        if self._closed:
            # Only during shutdown. The work is not lost: its task is left in the database
            # and startup recovery deals with it.
            logger.warning("runner is shutting down; not starting %s", name)
            return
        task = asyncio.get_running_loop().create_task(job(), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    async def wait_idle(self) -> None:
        """Wait until every job, including jobs started meanwhile, has finished."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            # Jobs handle their own failures; reaching here means a bug in the job itself.
            logger.error("background job %s crashed", task.get_name(), exc_info=error)
