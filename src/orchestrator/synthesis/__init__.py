"""Synthesis: combine the step outputs of an executed plan into one final, sourced answer.

Pure, like planning and execution: no database, and no import of either of them. It gets the
task and its stored step rows as plain data (SynthesisInput) and returns a FinalResult.
"""

from orchestrator.synthesis.synthesizer import SynthesisError, SynthesisOutcome, Synthesizer
from orchestrator.synthesis.types import (
    Conflict,
    ConflictClaim,
    FinalResult,
    KeyFinding,
    ResultSource,
    StepEvidence,
    SynthesisDraft,
    SynthesisInput,
)
from orchestrator.synthesis.validation import DraftValidationError, parse_draft, validate_draft

__all__ = [
    "Conflict",
    "ConflictClaim",
    "DraftValidationError",
    "FinalResult",
    "KeyFinding",
    "ResultSource",
    "StepEvidence",
    "SynthesisDraft",
    "SynthesisError",
    "SynthesisInput",
    "SynthesisOutcome",
    "Synthesizer",
    "parse_draft",
    "validate_draft",
]
