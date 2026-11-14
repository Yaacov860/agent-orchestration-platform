"""Test doubles for the app's collaborators (FakeLLM lives in the package: demos can use it)."""

import asyncio
import re
from collections.abc import Callable, Mapping
from typing import TypeAlias

from orchestrator.llm import LLMRequest, LLMResponse
from orchestrator.llm.fake import fake_response
from orchestrator.runner import Job


class ManualRunner:
    """A Runner that only records jobs; the test decides when they run.

    With the real BackgroundRunner, a test checking the state right after POST /tasks would
    race the planner. Here nothing runs until `run_all`, so every assertion is deterministic.
    """

    def __init__(self) -> None:
        self.jobs: list[tuple[str, Job]] = []

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.jobs]

    def spawn(self, name: str, job: Job) -> None:
        self.jobs.append((name, job))

    async def run_all(self, prefix: str = "") -> None:
        """Run every recorded job whose name starts with `prefix`, in order, including jobs
        recorded meanwhile. Others stay queued: `run_all("plan:")` plans without executing."""
        while (index := self._next(prefix)) is not None:
            _, job = self.jobs.pop(index)
            await job()

    def _next(self, prefix: str) -> int | None:
        return next((i for i, (name, _) in enumerate(self.jobs) if name.startswith(prefix)), None)

    async def shutdown(self) -> None:
        self.jobs.clear()


class GatedLLM:
    """An LLM that answers only once released, and tracks how many calls are in flight.

    Lets a test hold calls "mid-thought" to look at the system meanwhile: how many run at
    once, what the database shows, what a cancel does to a call that is still running.
    """

    def __init__(self, answer: Callable[[LLMRequest], str] = lambda request: "findings") -> None:
        self.release = asyncio.Event()
        self.in_flight = 0
        self.peak = 0
        self.calls: list[LLMRequest] = []
        self.cancelled = 0
        """Calls that were cancelled while waiting: proof a cancel reached the agent."""
        self._answer = answer
        self._changed = asyncio.Condition()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        await self._set_in_flight(+1)
        try:
            await self.release.wait()
            return fake_response(self._answer(request), input_tokens=10, output_tokens=5)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            await self._set_in_flight(-1)

    async def until_in_flight(self, count: int) -> None:
        """Wake exactly when `count` calls are in flight (no polling); fail after 2 s."""
        async with asyncio.timeout(2), self._changed:
            await self._changed.wait_for(lambda: self.in_flight == count)

    async def _set_in_flight(self, delta: int) -> None:
        async with self._changed:
            self.in_flight += delta
            self.peak = max(self.peak, self.in_flight)
            self._changed.notify_all()


# --- telling the callers apart ---------------------------------------------------------


def is_planner(request: LLMRequest) -> bool:
    return "You are the planner" in request.system_prompt


def is_synthesis(request: LLMRequest) -> bool:
    return "You are the synthesis stage" in request.system_prompt


# --- scripting replies per plan step -------------------------------------------------

_STEP = re.compile(r"## Your step \(\w+\)\ndo (step_\d+)")


def step_of(request: LLMRequest) -> str:
    match = _STEP.search(request.user_prompt)
    assert match, request.user_prompt
    return match[1]


def output_of(request: LLMRequest) -> str:
    return f"output of {step_of(request)}"


class StepLLM:
    """Replies per step: each step pops its own script; unscripted steps answer normally."""

    def __init__(self, scripts: Mapping[str, list[LLMResponse | Exception]] | None = None) -> None:
        self._scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.calls: list[str] = []
        self.prompts: dict[str, str] = {}
        """The last user prompt each step received, to check what was passed in."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        step_id = step_of(request)
        self.calls.append(step_id)
        self.prompts[step_id] = request.user_prompt
        await asyncio.sleep(0)
        script = self._scripts.get(step_id)
        item = script.pop(0) if script else fake_response(output_of(request))
        if isinstance(item, Exception):
            raise item
        return item


class RoutingLLM:
    """Answers by caller: the planner gets `plan`, synthesis gets `synthesis`, and every
    step gets `step(request)`. Records every request, in order."""

    def __init__(
        self,
        plan: str,
        synthesis: str | list[str | Exception],
        step: Callable[[LLMRequest], str] = lambda request: "findings",
    ) -> None:
        self._plan = plan
        self._synthesis = list(synthesis) if isinstance(synthesis, list) else synthesis
        self._step = step
        self.calls: list[LLMRequest] = []

    @property
    def synthesis_calls(self) -> list[LLMRequest]:
        return [r for r in self.calls if is_synthesis(r)]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        await asyncio.sleep(0)
        if is_planner(request):
            return fake_response(self._plan)
        if is_synthesis(request):
            item = self._synthesis.pop(0) if isinstance(self._synthesis, list) else self._synthesis
            if isinstance(item, Exception):
                raise item
            return fake_response(item, input_tokens=500, output_tokens=200)
        return fake_response(self._step(request))


_CALLERS = re.compile(r"You are the (planner|synthesis stage|\w+ agent)\b")


def caller_of(request: LLMRequest) -> str:
    """'planner', 'synthesis', or the agent name ('research', 'writing', 'analysis', 'code'),
    read from the system prompt each of them starts with."""
    match = _CALLERS.search(request.system_prompt)
    assert match, request.system_prompt[:80]
    return match[1].removesuffix(" agent").removesuffix(" stage")


Reply: TypeAlias = str | LLMResponse | Exception | Callable[[LLMRequest], str | LLMResponse]


class ByCallerLLM:
    """Scripts replies per caller. Each caller's value is one reply used every time, or a
    list consumed in order. A reply is text, a full LLMResponse (to attach sources), an
    exception to raise, or a function of the request. Records (caller, request) for every call."""

    def __init__(self, replies: Mapping[str, Reply | list[Reply]]) -> None:
        self._replies = {k: list(v) if isinstance(v, list) else v for k, v in replies.items()}
        self.calls: list[tuple[str, LLMRequest]] = []

    def requests(self, caller: str) -> list[LLMRequest]:
        return [request for who, request in self.calls if who == caller]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        caller = caller_of(request)
        self.calls.append((caller, request))
        await asyncio.sleep(0)
        script = self._replies[caller]
        reply = script.pop(0) if isinstance(script, list) else script
        if isinstance(reply, Exception):
            raise reply
        produced = reply(request) if callable(reply) else reply
        if isinstance(produced, LLMResponse):
            return produced
        return fake_response(produced, input_tokens=100, output_tokens=40)
