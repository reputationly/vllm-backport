# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-by-row checks of the vendor profile tables in the parity design doc.

Each hosted API rejects, ignores or rewrites a different subset of the OpenAI
request surface. The design doc's section 4 tables are the spec; this module
walks them one row at a time so a regression names the row it broke.

The gate runs as a wrap validator on ``ChatCompletionRequest``, so a rejected
row surfaces as ``VLLMValidationError`` with ``parameter`` set to the field the
hosted API blames.
"""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion import sampling_policy as sp
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError

DEFAULTS: dict = {}
BASE = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
STOP17 = [f"zz{i}" for i in range(17)]


@pytest.fixture
def profile(request):
    sp.set_active_profile(request.param)
    yield request.param
    sp.set_active_profile(None)


def _parse(profile, **kwargs):
    return ChatCompletionRequest.model_validate({**BASE, **kwargs})


def _rejects(profile, parameter, **kwargs):
    with pytest.raises(VLLMValidationError) as ei:
        _parse(profile, **kwargs)
    assert ei.value.parameter == parameter
    return ei.value


def _sampling(profile, **kwargs):
    return _parse(profile, **kwargs).to_sampling_params(
        8, DEFAULTS, sampling_policy=profile
    )


# ---------------------------------------------------------------------------
# 4.1 DeepSeek-V4.1-Flash
# ---------------------------------------------------------------------------

DEEPSEEK = pytest.mark.parametrize("profile", [sp.DEEPSEEK_V4], indirect=True)


@DEEPSEEK
def test_deepseek_temperature_range(profile):
    _rejects(profile, "temperature", temperature=2.5)
    _rejects(profile, "temperature", temperature=-0.1)
    assert _sampling(profile, temperature=2.0).temperature == 1.0


@DEEPSEEK
def test_deepseek_top_p_range(profile):
    _rejects(profile, "top_p", top_p=0)
    _rejects(profile, "top_p", top_p=1.5)


@DEEPSEEK
@pytest.mark.parametrize("key", ["presence_penalty", "frequency_penalty"])
def test_deepseek_penalty_range(profile, key):
    _rejects(profile, key, **{key: 3})
    _rejects(profile, key, **{key: -3})


@DEEPSEEK
def test_deepseek_penalties_ignored_in_both_modes(profile):
    """The hosted API deprecated the penalties; they must never take effect."""
    for extra in ({}, {"thinking": {"type": "disabled"}}):
        params = _sampling(
            profile, presence_penalty=1.5, frequency_penalty=-1.5, **extra
        )
        assert params.presence_penalty == 0.0
        assert params.frequency_penalty == 0.0


@DEEPSEEK
@pytest.mark.parametrize("key", ["top_k", "min_p", "repetition_penalty", "seed"])
def test_deepseek_unvalidated_knobs_ignored(profile, key):
    params = _sampling(profile, **{key: 123})
    assert getattr(params, key) in (-1, 0.0, 1.0, None)


@DEEPSEEK
def test_deepseek_n_coerced_to_one(profile):
    req = _parse(profile, n=3)
    assert req.n == 1


@DEEPSEEK
def test_deepseek_top_logprobs_range(profile):
    _rejects(profile, "top_logprobs", logprobs=True, top_logprobs=21)
    assert _parse(profile, logprobs=True, top_logprobs=20).top_logprobs == 20


@DEEPSEEK
def test_deepseek_top_logprobs_without_logprobs_ignored(profile):
    assert _parse(profile, top_logprobs=3).logprobs is False


@DEEPSEEK
def test_deepseek_logit_bias_rejected(profile):
    _rejects(profile, "logit_bias", logit_bias={"100": -100})


@DEEPSEEK
def test_deepseek_stop_limit(profile):
    assert _parse(profile, stop=STOP17[:16]).stop
    _rejects(profile, "stop", stop=STOP17)


@DEEPSEEK
@pytest.mark.parametrize("value", [0, -1])
def test_deepseek_max_tokens_minimum(profile, value):
    _rejects(profile, "max_tokens", max_tokens=value)


@DEEPSEEK
def test_deepseek_json_object_requires_hint(profile):
    _rejects(profile, "response_format", response_format={"type": "json_object"})
    assert _parse(
        profile,
        messages=[{"role": "user", "content": "give me JSON"}],
        response_format={"type": "json_object"},
    ).response_format


@DEEPSEEK
def test_deepseek_json_hint_ignores_tool_schemas(profile):
    """A "json" spelling inside a tool schema must not satisfy the check."""
    tools = [
        {
            "type": "function",
            "function": {"name": "f", "description": "returns json", "parameters": {}},
        }
    ]
    _rejects(
        profile,
        "response_format",
        tools=tools,
        response_format={"type": "json_object"},
    )


@DEEPSEEK
def test_deepseek_tool_choice_none_drops_tools(profile):
    """The hosted API answers from the prompt, not the tool schema."""
    req = _parse(profile, tools=TOOLS, tool_choice="none")
    assert not req.tools


@DEEPSEEK
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_deepseek_tool_choice_kept(profile, choice):
    req = _parse(profile, tools=TOOLS, tool_choice=choice)
    assert req.tools
    assert req.tool_choice == choice


@DEEPSEEK
def test_deepseek_legacy_function_fields_ignored(profile):
    req = _parse(
        profile,
        functions=[TOOLS[0]["function"]],
        function_call="auto",
    )
    assert not getattr(req, "functions", None)


@DEEPSEEK
def test_deepseek_thinking_type_validated(profile):
    _rejects(profile, "thinking.type", thinking={"type": "foo"})


@DEEPSEEK
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": "high"}, True),
        ({"thinking": {"type": "disabled"}, "reasoning_effort": "high"}, False),
        ({"thinking": {"type": "enabled"}, "reasoning_effort": "none"}, True),
    ],
)
def test_deepseek_thinking_priority(profile, kwargs, expected):
    sp.set_active_profile(sp.DEEPSEEK_V4)
    try:
        assert _parse(sp.DEEPSEEK_V4, **kwargs)._vendor_state.thinking is expected
    finally:
        sp.set_active_profile(None)


@DEEPSEEK
def test_deepseek_effort_ultra_maps_to_max(profile):
    assert _parse(profile, reasoning_effort="ultra").reasoning_effort == "max"


@DEEPSEEK
def test_deepseek_effort_validated(profile):
    _rejects(profile, "reasoning_effort", reasoning_effort="foo")


# ---------------------------------------------------------------------------
# 4.2 Qwen3.8-Flash
# ---------------------------------------------------------------------------

QWEN = pytest.mark.parametrize("profile", [sp.QWEN3_8], indirect=True)


@QWEN
def test_qwen_temperature_range_high_open(profile):
    _rejects(profile, "temperature", temperature=2.0)
    _rejects(profile, "temperature", temperature=-0.1)
    assert _sampling(profile, temperature=1.99).temperature == 1.0


@QWEN
def test_qwen_temperature_clamped_while_thinking(profile):
    assert _sampling(profile, temperature=0.2).temperature == 0.6
    assert _sampling(profile, temperature=1.8).temperature == 1.0
    assert _sampling(profile, temperature=0.8).temperature == 0.8


@QWEN
def test_qwen_temperature_honoured_without_thinking(profile):
    params = _sampling(profile, temperature=1.8, enable_thinking=False)
    assert params.temperature == 1.8


@QWEN
def test_qwen_top_p_zero_accepted(profile):
    params = _sampling(profile, top_p=0)
    assert params.top_p == pytest.approx(1e-6)
    _rejects(profile, "top_p", top_p=1.5)


@QWEN
def test_qwen_top_k_validated(profile):
    _rejects(profile, "top_k", top_k=-1)
    assert _sampling(profile, top_k=0).top_k == -1
    assert _sampling(profile, top_k=101).top_k == -1
    assert _sampling(profile, top_k=40).top_k == 40


@QWEN
@pytest.mark.parametrize("key", ["presence_penalty", "frequency_penalty"])
def test_qwen_penalty_range(profile, key):
    _rejects(profile, key, **{key: 3})
    assert getattr(_sampling(profile, **{key: 1.5}), key) == 1.5


@QWEN
def test_qwen_repetition_penalty_kept(profile):
    assert _sampling(profile, repetition_penalty=1.2).repetition_penalty == 1.2


@QWEN
def test_qwen_min_p_and_logit_bias_dropped(profile):
    params = _sampling(profile, min_p=0.5, logit_bias={"100": -100})
    assert params.min_p == 0.0
    assert not params.logit_bias


@QWEN
def test_qwen_top_logprobs_range(profile):
    _rejects(profile, "top_logprobs", logprobs=True, top_logprobs=6)
    assert _parse(profile, logprobs=True, top_logprobs=5).top_logprobs == 5
    assert _parse(profile, top_logprobs=3).logprobs is False


@QWEN
def test_qwen_n_thinking_must_be_one(profile):
    _rejects(profile, "n", n=2)
    assert _parse(profile, n=1).n == 1


@QWEN
def test_qwen_n_non_thinking_range(profile):
    assert _parse(profile, n=2, enable_thinking=False).n == 2
    _rejects(profile, "n", n=5, enable_thinking=False)
    assert _parse(profile, n=0, enable_thinking=False).n == 1


@QWEN
@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
def test_qwen_token_budget_range(profile, key):
    _rejects(profile, key, **{key: 0})
    _rejects(profile, key, **{key: 131073})


@QWEN
def test_qwen_max_tokens_sets_body_limit(profile):
    """D2: max_tokens caps the answer only; max_completion_tokens does not."""
    assert _parse(profile, max_tokens=32)._vendor_state.body_limit == 32
    assert (
        _parse(profile, max_completion_tokens=32)._vendor_state.body_limit is None
    )


@QWEN
def test_qwen_json_object_requires_hint(profile):
    _rejects(profile, "response_format", response_format={"type": "json_object"})


@QWEN
def test_qwen_tool_choice_none_drops_tools(profile):
    req = _parse(profile, tools=TOOLS, tool_choice="none")
    assert not req.tools


@QWEN
def test_qwen_thinking_type_validated(profile):
    _rejects(profile, "thinking.type", thinking={"type": "foo"})
    assert _parse(profile, thinking={"type": "auto"})


@QWEN
def test_qwen_effort_validated_and_mapped(profile):
    _rejects(profile, "reasoning_effort", reasoning_effort="ultra")
    assert _parse(profile, reasoning_effort="max")._vendor_state.effort == "xhigh"
    assert _parse(profile, reasoning_effort="minimal")._vendor_state.effort == "low"


@QWEN
def test_qwen_effort_and_budget_conflict(profile):
    _rejects(profile, "thinking_budget", reasoning_effort="low", thinking_budget=64)


@QWEN
def test_qwen_preserve_thinking_false_drops_history_reasoning(profile):
    messages = [
        {"role": "user", "content": "remember 7391"},
        {"role": "assistant", "content": "ok", "reasoning_content": "thinking..."},
        {"role": "user", "content": "what is it?"},
    ]
    req = _parse(profile, messages=messages, preserve_thinking=False)
    assert all("reasoning_content" not in m for m in req.messages)


@QWEN
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),
        ({"thinking": {"type": "enabled"}, "enable_thinking": False}, True),
        ({"thinking": {"type": "disabled"}, "reasoning_effort": "high"}, False),
        ({"enable_thinking": False}, False),
        ({"chat_template_kwargs": {"enable_thinking": False}}, False),
        ({"reasoning_effort": "none"}, False),
    ],
)
def test_qwen_thinking_priority(profile, kwargs, expected):
    sp.set_active_profile(sp.QWEN3_8)
    try:
        assert _parse(sp.QWEN3_8, **kwargs)._vendor_state.thinking is expected
    finally:
        sp.set_active_profile(None)


# ---------------------------------------------------------------------------
# 4.3 MiniMax-M3.1-Flash
# ---------------------------------------------------------------------------

MINIMAX = pytest.mark.parametrize("profile", [sp.MINIMAX_M3], indirect=True)


@MINIMAX
def test_minimax_temperature_ignored(profile):
    assert _sampling(profile, temperature=1.8).temperature == 1.0


@MINIMAX
def test_minimax_top_p_validated_and_pinned(profile):
    _rejects(profile, "top_p", top_p=0)
    _rejects(profile, "top_p", top_p=1.5)
    assert _sampling(profile, top_p=0.5).top_p == 0.95


@MINIMAX
@pytest.mark.parametrize(
    "key", ["top_k", "min_p", "repetition_penalty", "presence_penalty",
            "frequency_penalty", "logit_bias", "seed"]
)
def test_minimax_unvalidated_knobs_ignored(profile, key):
    params = _sampling(profile, **{key: 7})
    assert getattr(params, key) in (-1, 0.0, 1.0, None, {})


@MINIMAX
def test_minimax_n_above_one_rejected(profile):
    _rejects(profile, "n", n=2)
    assert _parse(profile, n=1).n == 1


@MINIMAX
def test_minimax_logprobs_ignored(profile):
    params = _sampling(profile, logprobs=True, top_logprobs=3)
    assert params.logprobs is None


@MINIMAX
def test_minimax_stop_ignored(profile):
    assert not _sampling(profile, stop=["7"]).stop


@MINIMAX
def test_minimax_max_tokens_zero_means_unset(profile):
    req = _parse(profile, max_tokens=0)
    assert req.max_tokens is None


@MINIMAX
def test_minimax_response_format_ignored(profile):
    req = _parse(profile, response_format={"type": "json_object"})
    assert req.response_format is None


@MINIMAX
def test_minimax_tool_choice_rewritten_to_auto(profile):
    for choice in ("required", "none"):
        assert _parse(profile, tools=TOOLS, tool_choice=choice).tool_choice == "auto"


@MINIMAX
def test_minimax_thinking_disabled_rejected(profile):
    _rejects(profile, "thinking.type", thinking={"type": "disabled"})
    _rejects(profile, "thinking.type", thinking={"type": "foo"})


@MINIMAX
@pytest.mark.parametrize("value", ["enabled", "adaptive"])
def test_minimax_thinking_accepted(profile, value):
    assert _parse(profile, thinking={"type": value})._vendor_state.thinking is True


@MINIMAX
def test_minimax_effort_rules(profile):
    _rejects(profile, "reasoning_effort", reasoning_effort="none")
    _rejects(profile, "reasoning_effort", reasoning_effort="foo")
    assert _parse(profile, reasoning_effort="minimal")._vendor_state.effort == "low"
    assert _parse(profile)._vendor_state.effort == "max"


@MINIMAX
@pytest.mark.parametrize(
    "key",
    ["enable_thinking", "chat_template_kwargs", "reasoning_split", "service_tier"],
)
def test_minimax_ignored_fields(profile, key):
    assert _parse(profile, **{key: {"enable_thinking": False}
                             if key == "chat_template_kwargs" else False})


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_no_profile_leaves_request_untouched():
    sp.set_active_profile(None)
    req = ChatCompletionRequest.model_validate(
        {**BASE, "temperature": 2.5, "top_p": 0.3, "top_k": 7, "n": 3}
    )
    assert req.temperature == 2.5
    assert req.top_p == 0.3
    assert req.n == 3


def test_profile_switch_disabled_by_env(monkeypatch):
    sp.set_active_profile(sp.DEEPSEEK_V4)
    monkeypatch.setattr(sp, "profile_gate_enabled", lambda: False)
    try:
        req = ChatCompletionRequest.model_validate({**BASE, "temperature": 2.5})
        assert req.temperature == 2.5
    finally:
        sp.set_active_profile(None)


def test_escape_hatch_keeps_allowlisted_field(monkeypatch):
    monkeypatch.setenv("VLLM_VENDOR_API_PROFILE_ALLOW", "chat_template_kwargs")
    sp.set_active_profile(sp.DEEPSEEK_V4)
    try:
        req = _parse(
            sp.DEEPSEEK_V4,
            chat_template_kwargs={"thinking": False},
        )
        assert req.chat_template_kwargs
    finally:
        sp.set_active_profile(None)


def test_policy_lookup_by_architecture():
    def cfg(model_type, architectures=None):
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, architectures=architectures
            )
        )

    assert sp.policy_for(cfg("deepseek_v41")) is sp.DEEPSEEK_V4
    assert sp.policy_for(cfg("qwen4_exp")) is sp.QWEN3_8
    assert sp.policy_for(cfg("minimax_m3_vl")) is sp.MINIMAX_M3
    assert sp.policy_for(cfg("llama")) is None
    assert sp.policy_for(SimpleNamespace()) is None
