# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendor sampling semantics for thinking-mode chat models.

The hosted APIs do not forward the caller's sampling knobs unchanged while the
model is thinking (DeepSeek ignores temperature and floors top_p at 0.95;
MiniMax ignores temperature and pins top_p to 0.95; Qwen ships 1.0/0.95/20).
Forwarding e.g. temperature=1.8 into a long thinking stream is what makes the
same request degenerate here while the hosted channels stay clean.
"""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.sampling_policy import (
    DEEPSEEK_V4,
    GLM5,
    MINIMAX_M3,
    QWEN3_8,
    policy_for,
    thinking_enabled,
)

DEFAULTS: dict = {}


def _params(policy, **kwargs):
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )
    return req.to_sampling_params(8, DEFAULTS, sampling_policy=policy)


@pytest.mark.parametrize(
    "kwargs,expected_temperature,expected_top_p",
    [
        # thinking (default): temperature ignored, top_p defaults to 1.0
        ({"temperature": 1.8}, 1.0, 1.0),
        ({"temperature": 2.0, "top_p": 1.0}, 1.0, 1.0),
        # below the floor is raised, inside the range is kept
        ({"temperature": 1.8, "top_p": 0.5}, 1.0, 0.95),
        ({"temperature": 1.8, "top_p": 0.9}, 1.0, 0.95),
        ({"temperature": 1.8, "top_p": 0.99}, 1.0, 0.99),
        # non-thinking: temperature honoured, top_p pinned to 1.0
        (
            {"temperature": 1.8, "chat_template_kwargs": {"reasoning_effort": "none"}},
            1.8,
            1.0,
        ),
        ({"temperature": 1.8, "reasoning_effort": "none"}, 1.8, 1.0),
        ({"temperature": 1.8, "chat_template_kwargs": {"thinking": False}}, 1.8, 1.0),
        ({"temperature": 1.8, "chat_template_kwargs": {"thinking": True}}, 1.0, 1.0),
        # reasoning_effort="none" wins over thinking=True, as in the tokenizer
        (
            {
                "temperature": 1.8,
                "chat_template_kwargs": {"thinking": True, "reasoning_effort": "none"},
            },
            1.8,
            1.0,
        ),
    ],
)
def test_deepseek_thinking_semantics(kwargs, expected_temperature, expected_top_p):
    params = _params(DEEPSEEK_V4, **kwargs)
    assert params.temperature == expected_temperature
    assert params.top_p == expected_top_p


def test_deepseek_clears_penalties():
    params = _params(
        DEEPSEEK_V4, temperature=1.8, presence_penalty=1.5, frequency_penalty=1.5
    )
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0


def test_deepseek_top_p_default_with_tools():
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=1.8,
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    params = req.to_sampling_params(8, DEFAULTS, sampling_policy=DEEPSEEK_V4)
    assert params.temperature == 1.0
    assert params.top_p == 0.95


@pytest.mark.parametrize(
    "kwargs,expected_temperature,expected_top_p,expected_top_k",
    [
        # MiniMax always thinks: temperature ignored, top_p and top_k pinned
        ({"temperature": 1.8, "top_p": 0.5, "top_k": 5}, 1.0, 0.95, -1),
        ({"temperature": 0.2, "top_p": 0.99}, 1.0, 0.95, -1),
        (
            {"temperature": 1.8, "chat_template_kwargs": {"enable_thinking": False}},
            1.0,
            0.95,
            -1,
        ),
    ],
)
def test_minimax_semantics(
    kwargs, expected_temperature, expected_top_p, expected_top_k
):
    params = _params(MINIMAX_M3, **kwargs)
    assert params.temperature == expected_temperature
    assert params.top_p == expected_top_p
    assert params.top_k == expected_top_k


@pytest.mark.parametrize(
    "kwargs,expected_temperature,expected_top_p,expected_top_k",
    [
        # Qwen honours temperature but its documented thinking range is <= 1.0
        ({"temperature": 1.8}, 1.0, 0.95, 20),
        ({"temperature": 0.6, "top_p": 0.8}, 0.6, 0.95, 20),
        # non-thinking keeps the caller's values
        (
            {"temperature": 1.8, "chat_template_kwargs": {"enable_thinking": False}},
            1.8,
            1.0,
            5,
        ),
    ],
)
def test_qwen_semantics(kwargs, expected_temperature, expected_top_p, expected_top_k):
    kwargs.setdefault("top_k", 5)
    params = _params(QWEN3_8, **kwargs)
    assert params.temperature == expected_temperature
    assert params.top_p == expected_top_p
    assert params.top_k == expected_top_k


@pytest.mark.parametrize(
    "temperature,expected",
    [(1.8, 1.0), (2.0, 1.0), (1.0, 1.0), (0.7, 0.7)],
)
def test_glm_semantics(temperature, expected):
    """GLM-5.3 always thinks: temperature clamped to the documented 1.0."""
    params = _params(GLM5, temperature=temperature, top_p=0.5)
    assert params.temperature == expected
    assert params.top_p == 0.95


def test_unknown_model_untouched():
    params = _params(None, temperature=1.8, top_p=0.7, top_k=7)
    assert params.temperature == 1.8
    assert params.top_p == 0.7
    assert params.top_k == 7


@pytest.mark.parametrize(
    "model_type,architectures,expected",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"], DEEPSEEK_V4),
        ("deepseek_v4_text", None, DEEPSEEK_V4),
        ("minimax_m3_vl", ["MiniMaxM3SparseForConditionalGeneration"], MINIMAX_M3),
        ("qwen4_exp", ["Qwen4ExpForConditionalGeneration"], QWEN3_8),
        ("qwen3", ["Qwen3ForCausalLM"], None),
        ("llama", None, None),
    ],
)
def test_policy_lookup(model_type, architectures, expected):
    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures)
    )
    assert policy_for(cfg) is expected


def test_policy_lookup_without_hf_config():
    assert policy_for(SimpleNamespace()) is None


@pytest.mark.parametrize(
    "kwargs,style,expected",
    [
        ({}, "deepseek", True),
        ({"reasoning_effort": "none"}, "deepseek", False),
        ({"thinking": False}, "deepseek", False),
        ({"enable_thinking": False}, "deepseek", False),
        ({"thinking": True, "reasoning_effort": "none"}, "deepseek", False),
        ({"enable_thinking": False}, "always", True),
        ({"enable_thinking": False}, "qwen", False),
        ({}, "qwen", True),
    ],
)
def test_thinking_enabled(kwargs, style, expected):
    assert thinking_enabled(kwargs, style) is expected
