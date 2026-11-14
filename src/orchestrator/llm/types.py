"""Provider-neutral data shapes for talking to an LLM.

Every provider (Gemini, FakeLLM, anything later) translates its own SDK objects into these,
so nothing above the `llm` package ever imports a provider SDK.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.llm.grounding import GroundingReport, StepGrounding


class Source(BaseModel):
    """A citation the provider returned alongside the text (e.g. a search-grounding result)."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Self) -> Self:
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    system_prompt: str
    user_prompt: str
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, gt=0)
    use_search: bool = False
    """Ask the provider to ground the answer in live web search and return the sources it used."""
    grounding: StepGrounding | None = Field(default=None, exclude=True, repr=False)
    """The step's share of its task's grounded-call budget (grounding.py). Set by the agent
    run, never by the agent: every provider call that would attach search reserves from it."""


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    usage: TokenUsage
    model: str
    sources: tuple[Source, ...] = ()
    grounding: GroundingReport | None = None
    """For a request with `use_search`: whether search was attached, why not, and the queries
    the provider reports. None from a provider that does not report it."""
