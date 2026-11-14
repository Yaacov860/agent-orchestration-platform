from collections.abc import Mapping
from typing import ClassVar

from orchestrator.agents.prompted import PromptedAgent

_ANALYST = """\
You are the analysis agent in a multi-agent pipeline. You reason about information other
agents gathered; you gather none yourself.

- Every claim you make must come from your inputs, the goal or your step. Name the input
  (e.g. "per step_1") a claim rests on.
- Never invent numbers, data points, studies or sources. If the inputs do not support a
  conclusion, write "insufficient information" for it instead of guessing.
- Separate what the inputs state from what you infer, and mark inferences as such.
- Where inputs contradict each other, report the contradiction with both sides.
- Answer in concise markdown."""

ANALYZE_PROMPT = (
    _ANALYST
    + """
- Your step: analyze the inputs as instructed: patterns, trends, causes, risks. End with
  `## Conclusions`, each conclusion stating how strongly the inputs support it."""
)

COMPARE_PROMPT = (
    _ANALYST
    + """
- Your step: compare the alternatives named in your step on the criteria that matter for
  the goal. Give a markdown table (alternatives x criteria) built only from the inputs, then
  `## Verdict`: which fits the goal best and under what conditions, or why the inputs
  cannot decide."""
)

EVALUATE_PROMPT = (
    _ANALYST
    + """
- Your step: evaluate the option or claim in your step. Give `## Strengths`,
  `## Weaknesses` and `## Assessment`, each point tied to an input."""
)


class AnalysisAgent(PromptedAgent):
    name: ClassVar[str] = "analysis"
    description: ClassVar[str] = (
        "Reasons over information earlier steps gathered: compares alternatives, finds "
        "patterns and trends, weighs pros and cons, draws conclusions. Adds no new facts or "
        "sources, so it needs research steps (or data in the task) before it."
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({"analyze", "compare", "evaluate"})
    action_descriptions: ClassVar[Mapping[str, str]] = {
        "analyze": "find patterns, trends, causes or risks in the outputs of earlier steps "
        "and draw supported conclusions.",
        "compare": "compare two or more alternatives on criteria that matter for the goal, "
        "as a table plus a verdict, using only the inputs.",
        "evaluate": "weigh the strengths and weaknesses of one option, plan or claim.",
    }
    system_prompts: ClassVar[Mapping[str, str]] = {
        "analyze": ANALYZE_PROMPT,
        "compare": COMPARE_PROMPT,
        "evaluate": EVALUATE_PROMPT,
    }
    temperature: ClassVar[float] = 0.2
    """Analysis should be repeatable: the same inputs, the same reading."""
