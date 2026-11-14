"""Shapes of synthesis: what goes in (step evidence), what the model may return (a draft),
and what is stored and served (the final result).

The split matters. The model writes only the draft: the answer, findings with the steps they
came from, conflicts, limitations it noticed. Everything a model could invent is computed by
code instead: sources (from the providers, via the step outputs), whether a finding is
corroborated, the confidence, and the limitations that follow from what failed.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from orchestrator.llm.types import Source

# --- input -------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepEvidence:
    """One plan step as synthesis sees it, read from the stored step row."""

    step_id: str
    agent: str
    action: str
    instruction: str
    dependencies: tuple[str, ...]
    status: str
    content: str | None = None
    sources: tuple[Source, ...] = ()
    confidence: float | None = None
    error: str | None = None
    """Why a step has no output (failed or not run), in the executor's words."""
    grounding_status: str | None = None
    """Research steps: "grounded" or "unverified" (agents.types.Grounding). None otherwise."""
    grounding_reason: str | None = None
    """Why a research step is unverified, in code's words."""

    @property
    def usable(self) -> bool:
        return self.status == "completed" and bool(self.content and self.content.strip())


@dataclass(frozen=True)
class SynthesisInput:
    task_id: str
    goal: str
    output_format: str
    steps: tuple[StepEvidence, ...]
    """Every step of the plan, in plan order, whatever its state."""
    constraints: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def usable_steps(self) -> tuple[StepEvidence, ...]:
        return tuple(s for s in self.steps if s.usable)


# --- what the model returns (untrusted: strict, bounded, validated) ----------------------

_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)


class DraftFinding(BaseModel):
    model_config = _STRICT

    statement: str = Field(min_length=1, max_length=1_000)
    step_ids: tuple[str, ...] = Field(default=(), max_length=20)
    """The steps whose output this finding comes from. Empty: the model could not attribute
    it, and the result will say so."""


class DraftClaim(BaseModel):
    model_config = _STRICT

    claim: str = Field(min_length=1, max_length=1_000)
    step_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class DraftConflict(BaseModel):
    model_config = _STRICT

    topic: str = Field(min_length=1, max_length=300)
    claims: tuple[DraftClaim, ...] = Field(min_length=2, max_length=6)
    """Every side, as the steps stated it. None is dropped and none is declared the winner."""


class SynthesisDraft(BaseModel):
    model_config = _STRICT

    content: str = Field(min_length=1, max_length=20_000)
    key_findings: tuple[DraftFinding, ...] = Field(default=(), max_length=20)
    conflicts: tuple[DraftConflict, ...] = Field(default=(), max_length=10)
    limitations: tuple[str, ...] = Field(default=(), max_length=10)


# --- what is stored and served -----------------------------------------------------------

Verification = Literal["corroborated", "single_step", "unattributed"]
"""corroborated: backed by at least two steps that do not derive from each other.
single_step: backed by one step (or by steps where one derives from the other): not verified.
unattributed: the model could not tie it to any step."""

_FROZEN = ConfigDict(frozen=True)


class KeyFinding(BaseModel):
    model_config = _FROZEN

    statement: str
    step_ids: tuple[str, ...]
    verification: Verification


class ResultSource(BaseModel):
    model_config = _FROZEN

    title: str
    url: str
    step_ids: tuple[str, ...]
    """The steps whose provider returned this page."""


class ConflictClaim(BaseModel):
    model_config = _FROZEN

    claim: str
    step_ids: tuple[str, ...]


class Conflict(BaseModel):
    model_config = _FROZEN

    topic: str
    claims: tuple[ConflictClaim, ...]


class FinalResult(BaseModel):
    model_config = _FROZEN

    content: str
    key_findings: tuple[KeyFinding, ...]
    sources: tuple[ResultSource, ...]
    confidence: float | None = Field(ge=0.0, le=1.0)
    confidence_basis: str
    conflicts: tuple[Conflict, ...]
    limitations: tuple[str, ...]
    partial: bool
    synthesized_from: tuple[str, ...]
