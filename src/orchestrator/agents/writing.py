from collections.abc import Mapping
from typing import ClassVar

from orchestrator.agents.prompted import PromptedAgent

_WRITER = """\
You are the writing agent in a multi-agent pipeline. You write finished text for people to
read, from the material you are given.

- Respect every constraint given (length such as max_words, tone, audience, format such as
  markdown) exactly; they override your own preferences.
- Use the facts in your inputs and only those. Where the inputs disagree, say so rather than
  picking one silently. Where they leave a gap, leave it visible instead of filling it."""

WRITE_PROMPT = (
    _WRITER
    + """
- Your step: write a complete piece (document, article, report, post, summary) from the
  inputs. Give it a clear structure: a title, short sections, and a closing that answers
  the task. Return only the piece itself."""
)

REWRITE_PROMPT = (
    _WRITER
    + """
- Your step: rewrite the text you are given (in your inputs or your step) as instructed:
  clearer, shorter, a different tone or audience. Keep its meaning and every fact; change
  nothing the instruction does not ask for. Return only the rewritten text."""
)

EMAIL_PROMPT = (
    _WRITER
    + """
- Your step: draft one email. Start with a line `Subject: ...`, then the body with a
  greeting, the message and a sign-off. Keep it as short as the purpose allows. Leave
  placeholders such as [name] for anything the inputs do not give you."""
)


class WritingAgent(PromptedAgent):
    name: ClassVar[str] = "writing"
    description: ClassVar[str] = (
        "Writes finished text for people from material it is given: documents, articles, "
        "summaries, emails; adapts tone, length and style to the constraints. Gathers no "
        "information itself, so it needs research or analysis steps before it for facts."
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({"write", "rewrite", "draft_email"})
    action_descriptions: ClassVar[Mapping[str, str]] = {
        "write": "write a complete, structured piece (document, article, report, summary, "
        "post) from the outputs of earlier steps, following length and tone constraints.",
        "rewrite": "rewrite or improve given text: clearer, shorter, another tone or "
        "audience, keeping its meaning and facts.",
        "draft_email": "draft one email (subject line and body) for the purpose described.",
    }
    system_prompts: ClassVar[Mapping[str, str]] = {
        "write": WRITE_PROMPT,
        "rewrite": REWRITE_PROMPT,
        "draft_email": EMAIL_PROMPT,
    }
    temperature: ClassVar[float] = 0.7
    """Prose benefits from some variety; the facts are pinned by the inputs, not the sampling."""
    max_output_tokens: ClassVar[int] = 4096
