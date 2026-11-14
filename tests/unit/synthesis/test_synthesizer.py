import asyncio

import pytest
from evidence import MDN, PY, SURVEY, done, evidence, failed, skipped
from factories import synthesis_json

from orchestrator.llm import LLMPermanentError, LLMTransientError
from orchestrator.llm.fake import FakeLLM, fake_response
from orchestrator.synthesis import SynthesisError, SynthesisInput, SynthesisOutcome, Synthesizer
from orchestrator.synthesis.synthesizer import MAX_ATTEMPTS

FULL = evidence(
    done("step_1", "Python reads like English.", sources=(PY, SURVEY), confidence=0.85),
    done("step_2", "JavaScript runs in every browser.", sources=(MDN, SURVEY), confidence=0.85),
    done(
        "step_3",
        "Python is easier to start; JS is everywhere.",
        "step_1",
        "step_2",
        confidence=None,
        action="summarize",
    ),
)
GOOD = synthesis_json(
    "Python is the easier first language; JavaScript is the one you cannot avoid on the web.",
    findings=[
        ("Python's syntax is easy to read", ["step_1"]),
        ("Both are widely taught to beginners", ["step_1", "step_2"]),
        ("Overall Python is easier to start", ["step_3"]),
        ("Learning either takes weeks", []),
    ],
    limitations=["The outputs do not compare job markets."],
)


async def synthesize(
    llm: FakeLLM, synthesis_input: SynthesisInput = FULL, **options: int
) -> SynthesisOutcome:
    return await Synthesizer(llm, **options).synthesize(synthesis_input)


async def synthesis_error(llm: FakeLLM, synthesis_input: SynthesisInput = FULL) -> SynthesisError:
    with pytest.raises(SynthesisError) as caught:
        await synthesize(llm, synthesis_input)
    return caught.value


# --- a valid synthesis -------------------------------------------------------------------


async def test_the_model_writes_the_answer_and_code_adds_provenance_and_confidence() -> None:
    llm = FakeLLM([fake_response(GOOD, input_tokens=900, output_tokens=300)])

    outcome = await synthesize(llm)

    result = outcome.result
    assert outcome.attempts == 1 and outcome.usage.total == 1200
    assert result.content.startswith("Python is the easier first language")
    assert [(f.step_ids, f.verification) for f in result.key_findings] == [
        (("step_1",), "single_step"),
        (("step_1", "step_2"), "corroborated"),
        (("step_3",), "single_step"),  # a summary alone is not a second opinion
        ((), "unattributed"),
    ]
    # Sources are what the providers returned, never anything the model wrote.
    assert [(s.url, s.step_ids) for s in result.sources] == [
        (PY.url, ("step_1",)),
        (SURVEY.url, ("step_1", "step_2")),
        (MDN.url, ("step_2",)),
    ]
    assert (result.confidence, result.partial) == (0.85, False)
    assert result.synthesized_from == ("step_1", "step_2", "step_3")
    assert result.limitations == (
        "2 of 4 key finding(s) rest on a single step and are not corroborated.",
        "1 key finding(s) could not be attributed to any step.",
        "The outputs do not compare job markets.",
    )


async def test_the_prompt_carries_every_output_its_lineage_and_the_task_as_data() -> None:
    llm = FakeLLM([fake_response(GOOD)])

    await synthesize(llm)

    [request] = llm.calls
    prompt = request.user_prompt
    assert "<task>\nCompare Python and JavaScript for beginners\n</task>" in prompt
    assert '<output step="step_1">\nPython reads like English.\n</output>' in prompt
    assert "## Output of step_3 (research agent, action summarize)\n" in prompt
    assert "Built on: step_1, step_2" in prompt
    assert "Grounded web sources returned: 2" in prompt
    assert PY.url not in prompt  # sources are counted for the model, never shown to copy
    assert "Do not write URLs" in request.system_prompt
    assert request.use_search is False


async def test_a_conflict_keeps_every_side_and_lowers_confidence() -> None:
    llm = FakeLLM(
        [
            fake_response(
                synthesis_json(
                    "The sources disagree on which is faster to learn.",
                    conflicts=[
                        (
                            "time to first program",
                            [
                                ("Python: one week", ["step_1"]),
                                ("JavaScript: one week", ["step_2"]),
                            ],
                        )
                    ],
                )
            )
        ]
    )

    result = (await synthesize(llm)).result

    [conflict] = result.conflicts
    assert [(c.claim, c.step_ids) for c in conflict.claims] == [
        ("Python: one week", ("step_1",)),
        ("JavaScript: one week", ("step_2",)),
    ]
    assert result.confidence == round(0.85 * 0.8, 2)
    assert "for 1 conflict(s)" in result.confidence_basis


async def test_a_partial_input_gives_a_partial_result_that_names_what_is_missing() -> None:
    partial = evidence(
        done("step_1", "Python reads like English.", sources=(PY,)),
        failed("step_2", "LLM call failed: rate limited"),
        skipped("step_3", "step_2"),
    )
    llm = FakeLLM(
        [
            fake_response(
                synthesis_json("Only Python was covered.", findings=[("readable", ["step_1"])])
            )
        ]
    )

    result = (await synthesize(llm, partial)).result

    assert result.partial is True
    assert result.synthesized_from == ("step_1",)
    assert result.limitations[:2] == (
        "step_2 (research: research) failed: LLM call failed: rate limited. "
        "Its findings are missing from this result.",
        "step_3 (research: summarize) produced no output: "
        "not run: depends on step_2, which did not complete.",
    )
    assert result.confidence == round(0.85 * (1 / 3), 2)
    assert "## Steps without output (cannot be cited)\n- step_2" in llm.calls[0].user_prompt


async def test_long_outputs_are_cut_for_the_prompt_and_the_cut_is_a_limitation() -> None:
    long_input = evidence(done("step_1", "x" * 1_000, sources=(PY,)))

    llm = FakeLLM([fake_response(synthesis_json())])
    outcome = await synthesize(llm, long_input, max_chars_per_step=300)

    prompt = llm.calls[0].user_prompt
    assert "x" * 300 + "\n[... truncated for synthesis: 700 of 1000 characters omitted]" in prompt
    assert "x" * 301 not in prompt
    assert any("was cut for synthesis" in note for note in outcome.result.limitations)


async def test_a_fenced_answer_is_accepted() -> None:
    outcome = await synthesize(FakeLLM([fake_response(f"```json\n{GOOD}\n```")]))

    assert outcome.attempts == 1


# --- repair and failure ------------------------------------------------------------------


async def test_an_invalid_draft_is_repaired_once_with_its_problems_shown() -> None:
    invented = synthesis_json("See https://invented.example.com", findings=[("x", ["step_9"])])
    llm = FakeLLM(
        [
            fake_response(invented, input_tokens=10, output_tokens=1),
            fake_response(GOOD, input_tokens=20, output_tokens=2),
        ]
    )

    outcome = await synthesize(llm)

    assert (outcome.attempts, outcome.usage.total) == (2, 33)
    repair = llm.calls[1].user_prompt
    assert "## Your previous answer (rejected)" in repair
    assert "- content: mentions https://invented.example.com, which no step returned" in repair
    assert "- key_findings[0]: step_9 is not a step of this plan" in repair


async def test_a_draft_still_invalid_after_the_repair_is_never_returned() -> None:
    llm = FakeLLM(
        [fake_response("not json"), fake_response(synthesis_json(findings=[("x", ["step_9"])]))]
    )

    error = await synthesis_error(llm)

    assert len(llm.calls) == MAX_ATTEMPTS
    assert (error.retryable, error.attempts) == (False, 2)
    assert error.problems == (
        "key_findings[0]: step_9 is not a step of this plan; "
        "steps with output: step_1, step_2, step_3",
    )


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [(LLMTransientError("429"), True), (LLMPermanentError("no provider"), False)],
)
async def test_an_llm_error_is_reported_with_its_retryability(
    failure: Exception, retryable: bool
) -> None:
    error = await synthesis_error(FakeLLM([failure]))

    assert error.retryable is retryable
    assert str(error) == f"LLM call failed during synthesis: {failure}"


async def test_with_no_usable_output_the_model_is_not_called() -> None:
    llm = FakeLLM()

    error = await synthesis_error(llm, evidence(failed("step_1"), skipped("step_2", "step_1")))

    assert (error.attempts, error.retryable) == (0, False)
    assert llm.calls == []


async def test_cancellation_propagates() -> None:
    task = asyncio.create_task(synthesize(FakeLLM([asyncio.CancelledError()])))

    with pytest.raises(asyncio.CancelledError):
        await task
