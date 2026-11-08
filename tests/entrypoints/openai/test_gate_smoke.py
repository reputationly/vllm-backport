import pytest
from vllm.entrypoints.openai.chat_completion import sampling_policy as sp
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.fixture
def deepseek():
    sp.set_active_profile(sp.DEEPSEEK_V4)
    yield sp.DEEPSEEK_V4
    sp.set_active_profile(None)


def test_top_logprobs_without_logprobs(deepseek):
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "top_logprobs": 3}
    )
    assert not req.logprobs


def test_reasoning_effort_ultra(deepseek):
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "ultra"}
    )
    assert req.reasoning_effort == "max"


def test_temperature_out_of_range(deepseek):
    with pytest.raises(VLLMValidationError) as ei:
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 2.5}
        )
    assert ei.value.parameter == "temperature"


def test_no_profile_unchanged():
    sp.set_active_profile(None)
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 2.5}
    )
    assert req.temperature == 2.5
