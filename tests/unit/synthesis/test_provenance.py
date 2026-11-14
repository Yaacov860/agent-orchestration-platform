from evidence import MDN, PY, SURVEY, done, evidence, failed, skipped

from orchestrator.synthesis import Conflict, ConflictClaim, KeyFinding
from orchestrator.synthesis import provenance as p

STEPS = (
    done("step_1", "a"),
    done("step_2", "b"),
    done("step_3", "c", "step_1", "step_2", action="summarize"),
    done("step_4", "d", "step_3", action="summarize"),
)
LINEAGE = p.ancestors(STEPS)
CONFLICT = Conflict(
    topic="t",
    claims=(
        ConflictClaim(claim="x", step_ids=("step_1",)),
        ConflictClaim(claim="y", step_ids=("step_2",)),
    ),
)


def test_ancestors_are_transitive() -> None:
    assert LINEAGE["step_4"] == {"step_1", "step_2", "step_3"}
    assert LINEAGE["step_1"] == frozenset()


def test_a_finding_without_steps_is_unattributed() -> None:
    assert p.verification((), LINEAGE) == "unattributed"


def test_one_step_is_not_corroboration() -> None:
    assert p.verification(("step_1",), LINEAGE) == "single_step"


def test_a_step_and_what_derives_from_it_are_not_two_opinions() -> None:
    # step_3 summarizes step_1: agreeing with its own input proves nothing.
    assert p.verification(("step_1", "step_3"), LINEAGE) == "single_step"
    assert p.verification(("step_3", "step_4"), LINEAGE) == "single_step"


def test_a_summary_alone_is_not_corroborated_even_if_it_read_several_steps() -> None:
    assert p.verification(("step_3",), LINEAGE) == "single_step"


def test_two_independent_steps_corroborate() -> None:
    assert p.verification(("step_1", "step_2"), LINEAGE) == "corroborated"
    assert p.verification(("step_3", "step_1", "step_2"), LINEAGE) == "corroborated"


def test_sources_are_collected_once_per_url_with_every_step_that_returned_them() -> None:
    sources = p.collect_sources(
        [done("step_1", "a", sources=(PY, SURVEY)), done("step_2", "b", sources=(SURVEY, MDN))]
    )

    assert [(s.url, s.step_ids) for s in sources] == [
        (PY.url, ("step_1",)),
        (SURVEY.url, ("step_1", "step_2")),
        (MDN.url, ("step_2",)),
    ]


def test_confidence_is_evidence_times_coverage_and_explains_itself() -> None:
    value, basis = p.confidence(
        evidence(
            done("step_1", "a", confidence=0.85),
            done("step_2", "b", confidence=0.6),
            failed("step_3"),
        ),
        conflicts=(),
    )

    assert value == round((0.85 + 0.6) / 2 * (2 / 3), 2)
    assert (
        basis == "mean evidence confidence 0.72 of 2 step(s) x coverage 2/3"
    )  # 0.725 is 0.7249... in binary


def test_conflicts_lower_confidence_and_say_so() -> None:
    value, basis = p.confidence(
        evidence(done("step_1", "a", confidence=0.85)), conflicts=(CONFLICT,)
    )

    assert value == round(0.85 * 0.8, 2)
    assert basis.endswith("x 0.8 for 1 conflict(s)")


def test_no_confidence_is_invented_when_no_step_reported_evidence() -> None:
    value, basis = p.confidence(evidence(done("step_1", "a", confidence=None)), conflicts=())

    assert value is None
    assert "none is given" in basis


def test_every_step_without_output_becomes_a_limitation() -> None:
    notes = p.step_limitations(
        evidence(
            done("step_1", "a", sources=(PY,)),
            failed("step_2", "rate limited"),
            skipped("step_3", "step_2"),
        ),
        truncated=[],
        max_chars=8000,
    )

    assert notes == [
        "step_2 (research: research) failed: rate limited. "
        "Its findings are missing from this result.",
        "step_3 (research: summarize) produced no output: "
        "not run: depends on step_2, which did not complete.",
    ]


def test_truncation_and_missing_sources_are_limitations() -> None:
    notes = p.step_limitations(evidence(done("step_1", "a")), truncated=["step_1"], max_chars=100)

    assert "step_1's output was longer than 100 characters and was cut for synthesis" in notes[0]
    assert notes[1] == "No step returned web sources: the findings are not backed by cited pages."


def test_weak_support_is_stated() -> None:
    findings = [
        KeyFinding(statement="a", step_ids=("step_1",), verification="single_step"),
        KeyFinding(statement="b", step_ids=(), verification="unattributed"),
        KeyFinding(statement="c", step_ids=("step_1", "step_2"), verification="corroborated"),
    ]

    assert p.support_limitations(findings) == [
        "1 of 3 key finding(s) rest on a single step and are not corroborated.",
        "1 key finding(s) could not be attributed to any step.",
    ]


def test_limitations_are_merged_without_repeats() -> None:
    assert p.merge_unique(["a", "b"], ["b ", "c"], [""]) == ("a", "b", "c")


def test_a_url_in_a_step_output_is_never_turned_into_a_source() -> None:
    assert p.collect_sources([done("step_1", "see https://api.example.com/v1", sources=())]) == ()


def test_an_unverified_research_step_is_named_in_the_limitations_with_its_reason() -> None:
    from dataclasses import replace

    unverified = replace(
        done("step_1", "- from memory"),
        grounding_status="unverified",
        grounding_reason="the task's grounded-call budget is used up (2 of 2 calls)",
    )
    grounded = replace(
        done("step_2", "- cited", sources=(PY,)),
        grounding_status="grounded",
    )

    notes = p.step_limitations(evidence(unverified, grounded), truncated=[], max_chars=100)

    assert notes == [
        "step_1 (research: research) is unverified: the task's grounded-call budget is used up "
        "(2 of 2 calls). Its findings come from the model, not from cited pages."
    ]
