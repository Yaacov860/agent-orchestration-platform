import re
from collections.abc import Mapping
from typing import ClassVar

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.prompted import InvalidAgentOutputError
from orchestrator.agents.types import AgentOutput, AgentTask, Grounding
from orchestrator.llm import GroundingReport, LLMClient, LLMRequest, LLMTransientError, Source

# The search instructions below are guidance for the model, not a limit: the model decides how
# many queries a grounded call runs. The hard limit is on grounded calls (llm/grounding.py).
RESEARCH_SYSTEM_PROMPT = """\
You are the research agent in a multi-agent pipeline. Another agent will turn your findings
into the final text, so deliver facts, not polished prose.

- Use web search if it is available to you, and base every factual claim on what you find.
- Keep it light: this is quick research on one step, not a survey. Run only the few focused
  queries this step needs, and stop when you have enough to answer it.
- If no search is available, answer from what you know and start with the line
  "Unverified: no web search was available"; never write tool or function calls as text.
- Do not write URLs or a reference list: the sources you used are collected automatically.
- If the evidence is thin, outdated or conflicting, say so explicitly instead of guessing.
- Stay on your step. Use the overall goal only to pick the right depth and angle.
- Answer in concise markdown bullet points, at most about 300 words."""

_TOOL_CALL_ONLY = re.compile(r"`*\s*[\w.:]+\([^\n]*\)\s*`*")
"""A whole answer that is one function-call-looking line: `fn:search(query="...")`."""

SUMMARIZE_SYSTEM_PROMPT = """\
You are the research agent in a multi-agent pipeline. Summarize ONLY the inputs you are given.

- Do not add facts that are not in the inputs.
- Keep claims attributed to the step they came from when they disagree.
- Answer in concise markdown bullet points, at most about 300 words."""


class ResearchAgent(BaseAgent):
    name: ClassVar[str] = "research"
    description: ClassVar[str] = "Searches for information, summarizes findings and cites sources."
    capabilities: ClassVar[frozenset[str]] = frozenset({"research", "summarize"})
    searches_web: ClassVar[bool] = True
    action_descriptions: ClassVar[Mapping[str, str]] = {
        "research": "find facts on the web about one question; returns findings with the "
        "grounded sources the search used. The only action that gathers new information.",
        "summarize": "condense the outputs of earlier steps into factual bullet points for "
        "other agents; adds nothing new and does no search.",
    }

    async def execute(self, task: AgentTask, llm: LLMClient) -> AgentOutput:
        searching = task.action == "research"
        response = await llm.generate(
            LLMRequest(
                system_prompt=RESEARCH_SYSTEM_PROMPT if searching else SUMMARIZE_SYSTEM_PROMPT,
                user_prompt=self.compose_user_prompt(task),
                temperature=0.2,
                use_search=searching,
            )
        )
        content = response.text.strip()
        if not content:
            raise LLMTransientError("model returned an empty answer")
        if _TOOL_CALL_ONLY.fullmatch(content):
            # Seen for real with search unavailable: the model "called" a search tool in its
            # text, e.g. fn:authoritative_search(query="..."). That is not a finding.
            raise InvalidAgentOutputError(
                "research: the answer is a tool call written as text, not findings"
            )

        sources = _unique_by_url(response.sources)
        if not searching:
            return AgentOutput(content=content, sources=sources, confidence=None)
        grounding = _grounding(response.grounding, sources)
        if response.grounding is not None and not response.grounding.used:
            # Known for certain that no search ran: say so in the text itself, in code's words
            # (the model was never told why it had no search, so its own line may be wrong).
            content = _label_unverified(content, grounding.reason)
        return AgentOutput(
            content=content,
            sources=sources,
            confidence=_evidence_confidence(len(sources)),
            grounding=grounding,
        )


NO_PAGES = "web search was available but returned no pages to cite"
NOT_REPORTED = "the provider did not report web search, and returned no pages to cite"


def _grounding(report: GroundingReport | None, sources: tuple[Source, ...]) -> Grounding:
    """Verified means: the provider ran search for this answer and returned the pages."""
    queries = report.queries if report is not None else ()
    if report is not None and not report.used:
        return Grounding(status="unverified", reason=report.reason)
    if sources:
        return Grounding(status="grounded", search_queries=queries)
    return Grounding(
        status="unverified",
        reason=NO_PAGES if report is not None else NOT_REPORTED,
        search_queries=queries,
    )


_UNVERIFIED_LINE = re.compile(r"\A\s*unverified\b[^\n]*\n*", re.IGNORECASE)


def _label_unverified(content: str, reason: str | None) -> str:
    body = _UNVERIFIED_LINE.sub("", content, count=1).strip() or content
    why = reason or "no web search ran"
    return f"Unverified: {why}. These findings come from the model, not from web search.\n\n{body}"


def _unique_by_url(sources: tuple[Source, ...]) -> tuple[Source, ...]:
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        if source.url not in seen:
            seen.add(source.url)
            unique.append(source)
    return tuple(unique)


def _evidence_confidence(source_count: int) -> float:
    """How much independent evidence backs the answer. Not the model's self-rating,
    which is poorly calibrated: models sound sure whether or not they are right."""
    if source_count == 0:
        return 0.3
    if source_count == 1:
        return 0.6
    return 0.85
