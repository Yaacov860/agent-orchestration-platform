"""Builders for synthesis inputs: steps as synthesis sees them, read from stored rows."""

from orchestrator.llm.types import Source
from orchestrator.synthesis import StepEvidence, SynthesisInput

PY = Source(title="Python docs", url="https://docs.python.org")
MDN = Source(title="MDN", url="https://developer.mozilla.org")
SURVEY = Source(title="Survey", url="https://survey.example.org/2026")


def done(
    step_id: str,
    content: str,
    *dependencies: str,
    sources: tuple[Source, ...] = (),
    confidence: float | None = 0.85,
    action: str = "research",
) -> StepEvidence:
    return StepEvidence(
        step_id=step_id,
        agent="research",
        action=action,
        instruction=f"do {step_id}",
        dependencies=dependencies,
        status="completed",
        content=content,
        sources=sources,
        confidence=confidence,
    )


def failed(step_id: str, error: str = "content blocked", *dependencies: str) -> StepEvidence:
    return StepEvidence(
        step_id=step_id,
        agent="research",
        action="research",
        instruction=f"do {step_id}",
        dependencies=dependencies,
        status="failed",
        error=error,
    )


def skipped(step_id: str, *dependencies: str) -> StepEvidence:
    blocker = dependencies[0] if dependencies else "?"
    return StepEvidence(
        step_id=step_id,
        agent="research",
        action="summarize",
        instruction=f"do {step_id}",
        dependencies=dependencies,
        status="skipped",
        error=f"not run: depends on {blocker}, which did not complete",
    )


def evidence(*steps: StepEvidence, output_format: str = "markdown") -> SynthesisInput:
    return SynthesisInput(
        task_id="task_1",
        goal="Compare Python and JavaScript for beginners",
        output_format=output_format,
        constraints={"tone": "friendly"},
        steps=steps,
    )
