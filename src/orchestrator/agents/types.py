"""The agent contract: what goes into an agent (AgentTask) and what comes out (AgentResult)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, computed_field

from orchestrator.llm.types import Source, TokenUsage


class ContextItem(BaseModel):
    """The output of an upstream step that this step depends on."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    agent: str
    content: str


class AgentTask(BaseModel):
    """Everything one agent needs to run one plan step, and nothing more."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    action: str
    instruction: str
    """What this step must do (the plan step's `input`)."""
    goal: str
    """The user's overall goal, so the step works at the right level and angle."""
    constraints: dict[str, JsonValue] = Field(default_factory=dict)
    context: tuple[ContextItem, ...] = ()
    """Outputs of the steps this one depends on. Filled by the executor."""


class Grounding(BaseModel):
    """Whether a research answer rests on web search, and if not, why. Set by code from what
    the provider reported, never by the model."""

    model_config = ConfigDict(frozen=True)

    status: Literal["grounded", "unverified"]
    """grounded: search was attached and the provider returned the pages it used.
    unverified: no search ran, or it returned no pages: the answer is the model's own."""
    reason: str | None = None
    """Why the answer is unverified (search disabled, budget used up, no pages returned)."""
    search_queries: tuple[str, ...] = ()
    """The queries the provider reports for the call that produced this answer."""


class AgentOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    sources: tuple[Source, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    grounding: Grounding | None = Field(default=None, exclude_if=lambda value: value is None)
    """Research only (the one action that may search). Left out of the stored output of
    every other step, so their shape is unchanged."""


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    retryable: bool
    retry_after_seconds: float | None = None
    """How long the provider asked to wait before a retry, when it said."""


class AgentResult(BaseModel):
    """One agent run, successful or not. Mirrors the assignment's 'Agent Output Format'."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    agent: str
    status: Literal["completed", "failed"]
    output: AgentOutput | None = None
    error: ErrorInfo | None = None
    usage: TokenUsage = TokenUsage()
    execution_time_ms: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_used(self) -> int:
        return self.usage.total
