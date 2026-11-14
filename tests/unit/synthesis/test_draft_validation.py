import json

import pytest
from evidence import MDN, PY, done, evidence, failed
from factories import synthesis_json

from orchestrator.synthesis import DraftValidationError, SynthesisInput, parse_draft, validate_draft

INPUT = evidence(
    done("step_1", "a", sources=(PY,)), done("step_2", "b", sources=(MDN,)), failed("step_3")
)


def problems(raw: str, synthesis_input: SynthesisInput = INPUT) -> tuple[str, ...]:
    with pytest.raises(DraftValidationError) as caught:
        validate_draft(parse_draft(raw), synthesis_input)
    return caught.value.problems


def test_a_valid_draft_passes_with_repeated_step_ids_removed() -> None:
    draft = validate_draft(
        parse_draft(
            synthesis_json(findings=[("Python is readable", ["step_1", "step_1", "step_2"])])
        ),
        INPUT,
    )

    assert draft.key_findings[0].step_ids == ("step_1", "step_2")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Here is my answer: Python wins.", "Invalid JSON"),
        (json.dumps({"key_findings": []}), "content: Field required"),
        (json.dumps({"content": "   "}), "content: String should have at least 1 character"),
        (json.dumps({"content": "x", "verdict": "python"}), "verdict: unknown field 'verdict'"),
        (
            json.dumps(
                {
                    "content": "x",
                    "conflicts": [
                        {"topic": "t", "claims": [{"claim": "a", "step_ids": ["step_1"]}]}
                    ],
                }
            ),
            "conflicts.0.claims: Tuple should have at least 2 items",
        ),
        (
            json.dumps(
                {
                    "content": "x",
                    "conflicts": [
                        {
                            "topic": "t",
                            "claims": [
                                {"claim": "a", "step_ids": []},
                                {"claim": "b", "step_ids": ["step_1"]},
                            ],
                        }
                    ],
                }
            ),
            "conflicts.0.claims.0.step_ids: Tuple should have at least 1 item",
        ),
    ],
)
def test_malformed_drafts_are_rejected_with_the_location(raw: str, expected: str) -> None:
    with pytest.raises(DraftValidationError) as caught:
        parse_draft(raw)

    assert any(expected in problem for problem in caught.value.problems), caught.value.problems


def test_citing_a_step_that_failed_or_does_not_exist_is_rejected() -> None:
    raw = synthesis_json(findings=[("a", ["step_3"]), ("b", ["step_9"])])

    assert problems(raw) == (
        "key_findings[0]: step_3 did not complete and has no output to cite; "
        "steps with output: step_1, step_2",
        "key_findings[1]: step_9 is not a step of this plan; steps with output: step_1, step_2",
    )


def test_a_url_no_step_returned_is_rejected_wherever_it_appears() -> None:
    raw = synthesis_json(
        "See https://made-up.example.com/guide for more.",
        findings=[("per https://another-invented.org", ["step_1"])],
    )

    assert problems(raw) == (
        "content: mentions https://made-up.example.com/guide, which no step returned; "
        "do not write URLs, sources are attached automatically",
        "key_findings[0]: mentions https://another-invented.org, which no step returned; "
        "do not write URLs, sources are attached automatically",
    )


def test_a_url_a_step_did_return_is_accepted_even_followed_by_punctuation() -> None:
    validate_draft(parse_draft(synthesis_json(f"As {PY.url}.")), INPUT)


def test_a_conflict_whose_sides_say_the_same_is_not_a_conflict() -> None:
    raw = synthesis_json(
        conflicts=[("speed", [("Python is fast", ["step_1"]), ("python is FAST", ["step_2"])])]
    )

    assert problems(raw) == ("conflicts[0]: its claims are identical, so it is not a conflict",)


def test_json_output_must_be_valid_json() -> None:
    json_input = evidence(done("step_1", "a"), output_format="json")

    assert problems(synthesis_json("not json"), json_input) == (
        'content: output_format is "json", but content is not valid JSON',
    )
    validate_draft(parse_draft(synthesis_json('{"winner": null}')), json_input)


def test_every_problem_is_reported_at_once() -> None:
    raw = synthesis_json(
        "https://invented.example",
        findings=[("a", ["step_9"])],
        conflicts=[("t", [("x", ["step_3"]), ("y", ["step_1"])])],
    )

    assert len(problems(raw)) == 3


def test_a_url_quoted_from_a_step_output_is_accepted_but_an_invented_one_is_not() -> None:
    code_step = done("step_1", 'requests.get("https://api.example.com/v1")', confidence=None)
    with_code = evidence(code_step)

    validate_draft(parse_draft(synthesis_json('Call "https://api.example.com/v1".')), with_code)
    assert problems(synthesis_json("Docs: https://api.example.com/docs"), with_code) == (
        "content: mentions https://api.example.com/docs, which no step returned; "
        "do not write URLs, sources are attached automatically",
    )
