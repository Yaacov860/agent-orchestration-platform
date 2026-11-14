"""The code agent: writes, explains, reviews and fixes code, as text only.

It never runs code, touches files or acts on a machine: it has no tool for any of that, and
its prompts forbid claiming otherwise. What it returns is a proposal for a person to check.
"""

import re
from collections.abc import Mapping
from typing import ClassVar

from orchestrator.agents.prompted import OutputCheck, PromptedAgent

_FENCED_BLOCK = re.compile(r"```[^\n`]*\n.*?\n```", re.DOTALL)

_CODER = """\
You are the code agent in a multi-agent pipeline. You work with code as text only.

- You cannot run code, read or write files, install packages or use a terminal. Never claim
  that code was run, tested or compiled; say what should be tested instead.
- Put all code in fenced markdown blocks with the language named (```python).
- Work from the code and requirements in your inputs and your step. If something needed is
  missing (a function you were not shown, a version), say so and state your assumption."""

WRITE_CODE_PROMPT = (
    _CODER
    + """
- Your step: write the code asked for. Give the code first, complete and runnable as far as
  you can tell, then a short `## Notes` section: assumptions, how to use it, what to test."""
)

EXPLAIN_CODE_PROMPT = (
    _CODER
    + """
- Your step: explain the code you are given: what it does, how, and anything surprising.
  Quote short fragments when it helps; do not rewrite the code."""
)

REVIEW_CODE_PROMPT = (
    _CODER
    + """
- Your step: review the code you are given. List findings by severity (bugs, security,
  correctness, then readability), each with where it is, why it matters and a fix. Say so
  plainly if you find nothing serious."""
)

FIX_BUG_PROMPT = (
    _CODER
    + """
- Your step: find the bug described (or the most likely one) in the code you are given.
  State the cause in one or two sentences, then give the corrected code in a fenced block,
  then what to test to confirm the fix."""
)


def _needs_code_block(content: str) -> str | None:
    if _FENCED_BLOCK.search(content):
        return None
    return "the answer has no fenced code block; code must be inside ``` fences"


class CodeAgent(PromptedAgent):
    name: ClassVar[str] = "code"
    description: ClassVar[str] = (
        "Works with code as text: writes code to a specification, explains, reviews, and "
        "fixes bugs in code it is given. Never runs code or touches files; returns code and "
        "advice for a person to check."
    )
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"write_code", "explain_code", "review_code", "fix_bug"}
    )
    action_descriptions: ClassVar[Mapping[str, str]] = {
        "write_code": "write code that meets requirements given in the step or in the "
        "outputs of earlier steps (e.g. an analysis of requirements).",
        "explain_code": "explain what given code does and how.",
        "review_code": "review given code for bugs, security and readability issues, with "
        "suggested fixes.",
        "fix_bug": "find the cause of a described bug in given code and return corrected code.",
    }
    system_prompts: ClassVar[Mapping[str, str]] = {
        "write_code": WRITE_CODE_PROMPT,
        "explain_code": EXPLAIN_CODE_PROMPT,
        "review_code": REVIEW_CODE_PROMPT,
        "fix_bug": FIX_BUG_PROMPT,
    }
    output_checks: ClassVar[Mapping[str, OutputCheck]] = {
        # The two actions whose whole point is code must actually return some.
        "write_code": _needs_code_block,
        "fix_bug": _needs_code_block,
    }
    temperature: ClassVar[float] = 0.1
    """Code wants the most likely tokens, not creative ones."""
    max_output_tokens: ClassVar[int] = 4096
