"""Turn the synthesis model's raw output into a SynthesisDraft, or say everything wrong with it.

Like plan validation (planning/validation.py): every problem at once, each naming what is
allowed, since the model gets exactly one repair attempt. Checks the schema cannot express:
- every cited step exists and actually produced output (no citing a failed step);
- no URL appears that is neither a source a provider returned nor text in a step's output
  (the synthesis model must not invent any; one copied from, say, code is not invented);
- for output_format "json", the content is valid JSON.
"""

import json
import re
from collections.abc import Iterable

from pydantic import ValidationError

from orchestrator.synthesis.types import SynthesisDraft, SynthesisInput

_URL = re.compile(r"https?://[^\s<>\"'()\[\]{}]+")


class DraftValidationError(Exception):
    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("invalid synthesis: " + "; ".join(self.problems))


def parse_draft(raw_json: str) -> SynthesisDraft:
    try:
        return SynthesisDraft.model_validate_json(raw_json)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            message = (
                f"unknown field '{error['loc'][-1]}'"
                if error["type"] == "extra_forbidden"
                else error["msg"]
            )
            problems.append(f"{location}: {message}" if location else message)
        raise DraftValidationError(problems) from None


def validate_draft(draft: SynthesisDraft, synthesis_input: SynthesisInput) -> SynthesisDraft:
    """Check the draft against the steps; return it with repeated step ids removed."""
    usable = {s.step_id for s in synthesis_input.usable_steps}
    without_output = {s.step_id for s in synthesis_input.steps} - usable
    allowed = ", ".join(sorted(usable))
    problems: list[str] = []

    def check_ids(where: str, step_ids: Iterable[str]) -> None:
        for step_id in step_ids:
            if step_id in without_output:
                problems.append(
                    f"{where}: {step_id} did not complete and has no output to cite; "
                    f"steps with output: {allowed}"
                )
            elif step_id not in usable:
                problems.append(
                    f"{where}: {step_id} is not a step of this plan; steps with output: {allowed}"
                )

    for i, finding in enumerate(draft.key_findings):
        check_ids(f"key_findings[{i}]", finding.step_ids)
    for i, conflict in enumerate(draft.conflicts):
        for j, claim in enumerate(conflict.claims):
            check_ids(f"conflicts[{i}].claims[{j}]", claim.step_ids)
        if len({c.claim.casefold() for c in conflict.claims}) < 2:
            problems.append(f"conflicts[{i}]: its claims are identical, so it is not a conflict")

    known_urls = {source.url for step in synthesis_input.usable_steps for source in step.sources}
    # A URL that appears in a step's own output (an endpoint in generated code, a link the
    # user gave) is copied, not invented. It is still not a source: `sources` stays
    # provider-only, computed in provenance.collect_sources.
    known_urls |= {
        url.rstrip(".,;:!?")
        for step in synthesis_input.usable_steps
        for url in _URL.findall(step.content or "")
    }
    texts = [("content", draft.content)]
    texts += [(f"key_findings[{i}]", f.statement) for i, f in enumerate(draft.key_findings)]
    texts += [
        (f"conflicts[{i}].claims[{j}]", c.claim)
        for i, conflict in enumerate(draft.conflicts)
        for j, c in enumerate(conflict.claims)
    ]
    for where, text in texts:
        for url in _URL.findall(text):
            if url.rstrip(".,;:!?") not in known_urls:
                problems.append(
                    f"{where}: mentions {url.rstrip('.,;:!?')}, which no step returned; "
                    "do not write URLs, sources are attached automatically"
                )

    if synthesis_input.output_format == "json":
        try:
            json.loads(draft.content)
        except ValueError:
            problems.append('content: output_format is "json", but content is not valid JSON')

    if problems:
        raise DraftValidationError(problems)
    return _without_repeated_ids(draft)


def _without_repeated_ids(draft: SynthesisDraft) -> SynthesisDraft:
    def unique(ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ids))

    return draft.model_copy(
        update={
            "key_findings": tuple(
                f.model_copy(update={"step_ids": unique(f.step_ids)}) for f in draft.key_findings
            ),
            "conflicts": tuple(
                c.model_copy(
                    update={
                        "claims": tuple(
                            claim.model_copy(update={"step_ids": unique(claim.step_ids)})
                            for claim in c.claims
                        )
                    }
                )
                for c in draft.conflicts
            ),
        }
    )
