import pytest

from orchestrator.llm import LLMRequest, LLMTransientError
from orchestrator.llm.fake import FakeLLM, fake_response

REQUEST = LLMRequest(system_prompt="sys", user_prompt="hello")


async def test_returns_scripted_replies_in_order_and_records_requests() -> None:
    llm = FakeLLM([fake_response("first"), fake_response("second")])

    first = await llm.generate(REQUEST)
    second = await llm.generate(REQUEST.model_copy(update={"user_prompt": "again"}))

    assert (first.text, second.text) == ("first", "second")
    assert [call.user_prompt for call in llm.calls] == ["hello", "again"]


async def test_raises_a_scripted_exception() -> None:
    llm = FakeLLM([LLMTransientError("429 rate limited")])

    with pytest.raises(LLMTransientError, match="rate limited"):
        await llm.generate(REQUEST)


async def test_fails_loudly_when_called_more_times_than_scripted() -> None:
    llm = FakeLLM([fake_response("only one")])
    await llm.generate(REQUEST)

    with pytest.raises(AssertionError, match="script exhausted"):
        await llm.generate(REQUEST)
