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


class AgentOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    sources: tuple[Source, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    message: str
    retryable: bool


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
