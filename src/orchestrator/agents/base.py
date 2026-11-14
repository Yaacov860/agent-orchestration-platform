import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Literal

from orchestrator.agents.types import AgentOutput, AgentResult, AgentTask, ErrorInfo
from orchestrator.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    StepGrounding,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class UnsupportedActionError(Exception):
    """The step asked an agent for an action it does not offer (a planning bug)."""


class MeteredLLM:
    """Wraps the LLM for ONE agent run and adds up the tokens of every call in that run.

    A fresh one is created per run because a single agent instance serves many steps
    concurrently; a token counter stored on the agent itself would mix their numbers.
    It satisfies the LLMClient protocol, so `execute` cannot tell it from the real client.
    It also attaches the step's grounding budget to every search request, so an agent never
    handles the budget itself and cannot forget it.
    """

    def __init__(self, llm: LLMClient, grounding: StepGrounding | None = None) -> None:
        self._llm = llm
        self._grounding = grounding
        self.usage = TokenUsage()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.use_search and request.grounding is None and self._grounding is not None:
            request = request.model_copy(update={"grounding": self._grounding})
        response = await self._llm.generate(request)
        self.usage = self.usage + response.usage
        return response


class BaseAgent(ABC):
    """Shared run loop for every agent (Template Method pattern).

    `run` is the same for all agents: check the action, time it, meter tokens, turn any
    failure into a structured result. Subclasses only implement `execute`, the actual work.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]
    """The actions this agent accepts; the planner may only assign these."""
    searches_web: ClassVar[bool] = False
    """Whether it asks the provider for web search (and so can return grounded sources)."""
    action_descriptions: ClassVar[Mapping[str, str]] = {}
    """What each action does, in a sentence the planner reads: it picks an agent by what a
    step must do, not by the agent's name. When given, it must describe exactly the
    capabilities (checked when the class is defined, so the two cannot drift apart)."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        described = set(cls.action_descriptions)
        if described and hasattr(cls, "capabilities") and described != set(cls.capabilities):
            raise TypeError(
                f"{cls.__name__}: action_descriptions {sorted(described)} do not match "
                f"capabilities {sorted(cls.capabilities)}"
            )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, task: AgentTask, *, grounding: StepGrounding | None = None) -> AgentResult:
        """`grounding`: this step's share of its task's grounded-call budget (the executor's).
        Without it, search requests are not budgeted."""
        started = time.perf_counter()
        meter = MeteredLLM(self._llm, grounding)
        error: ErrorInfo
        try:
            if task.action not in self.capabilities:
                raise UnsupportedActionError(
                    f"agent '{self.name}' does not support action '{task.action}' "
                    f"(supported: {sorted(self.capabilities)})"
                )
            output = await self.execute(task, meter)
        except LLMError as exc:
            error = ErrorInfo(
                type=type(exc).__name__,
                message=str(exc),
                retryable=exc.retryable,
                retry_after_seconds=exc.retry_after,
            )
        except Exception as exc:
            # Deliberately not BaseException: asyncio.CancelledError must propagate so that
            # cancelling a task actually stops it instead of being recorded as a step failure.
            logger.exception("agent %s failed on step %s", self.name, task.step_id)
            error = ErrorInfo(type=type(exc).__name__, message=str(exc), retryable=False)
        else:
            return self._result(task, "completed", started, meter.usage, output=output)
        return self._result(task, "failed", started, meter.usage, error=error)

    @abstractmethod
    async def execute(self, task: AgentTask, llm: LLMClient) -> AgentOutput:
        """Do the agent's work for one step. Raise on failure; `run` records it."""

    def compose_user_prompt(self, task: AgentTask) -> str:
        """The common frame every agent's prompt starts from: goal, constraints, inputs, step."""
        parts = [f"## Overall goal (the user's request)\n{task.goal}"]
        if task.constraints:
            parts.append(f"## Constraints\n{json.dumps(task.constraints, ensure_ascii=False)}")
        for item in task.context:
            parts.append(f"## Input from {item.step_id} ({item.agent} agent)\n{item.content}")
        parts.append(f"## Your step ({task.action})\n{task.instruction}")
        return "\n\n".join(parts)

    def _result(
        self,
        task: AgentTask,
        status: Literal["completed", "failed"],
        started: float,
        usage: TokenUsage,
        *,
        output: AgentOutput | None = None,
        error: ErrorInfo | None = None,
    ) -> AgentResult:
        return AgentResult(
            step_id=task.step_id,
            agent=self.name,
            status=status,
            output=output,
            error=error,
            usage=usage,
            execution_time_ms=round((time.perf_counter() - started) * 1000),
        )
