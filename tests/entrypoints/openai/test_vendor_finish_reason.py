# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""finish_reason decisions at the serving boundary.

The vendor profiles deliberately change one thing about tool calls: every
emitted call reports ``finish_reason="tool_calls"``, including a named
``tool_choice`` where vLLM itself says ``"stop"``. A model without a profile
must keep vLLM's own rule, because OpenAI clients gate tool execution on this
field -- regressing it silently breaks their tool loop.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _stream_finish_reason,
    _tool_calls_finish_reason,
)


@pytest.mark.parametrize(
    "profiled,tool_choice,auto_tools_called,has_tool_calls,engine_reason,expected",
    [
        # no profile: vLLM's rule
        (False, "auto", True, True, "stop", True),
        (False, "auto", False, False, "stop", False),
        # a named tool_choice reports "stop" without a profile
        (False, {"type": "function"}, False, True, "stop", False),
        # "required" only counts when the engine itself stopped
        (False, "required", False, False, "stop", True),
        (False, "required", False, False, "length", False),
        # with a profile every emitted call is a tool call
        (True, "auto", True, True, "stop", True),
        (True, {"type": "function"}, False, True, "stop", True),
        (True, None, False, False, "stop", False),
    ],
)
def test_tool_calls_finish_reason(
    profiled, tool_choice, auto_tools_called, has_tool_calls, engine_reason, expected
):
    assert (
        _tool_calls_finish_reason(
            profiled=profiled,
            tool_choice=tool_choice,
            auto_tools_called=auto_tools_called,
            has_tool_calls=has_tool_calls,
            engine_finish_reason=engine_reason,
        )
        is expected
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # no profile: vLLM's rule, named tool choice ends as "stop"
        (dict(profiled=False, tools_streamed=True), "tool_calls"),
        (
            dict(profiled=False, tools_streamed=True, tool_choice_function_name="f"),
            "stop",
        ),
        (dict(profiled=False), "stop"),
        (dict(profiled=False, engine_finish_reason="length"), "length"),
        # with a profile: tool calls win, then the legacy shape, then the cap
        (dict(profiled=True, tools_streamed=True), "tool_calls"),
        (
            dict(profiled=True, tools_streamed=True, tool_choice_function_name="f"),
            "tool_calls",
        ),
        (
            dict(profiled=True, tools_streamed=True, legacy_functions=True),
            "function_call",
        ),
        (dict(profiled=True, budget_spent=True), "length"),
        (dict(profiled=True), "stop"),
        (dict(profiled=True, engine_finish_reason="length"), "length"),
    ],
)
def test_stream_finish_reason(kwargs, expected):
    kwargs.setdefault("legacy_functions", False)
    kwargs.setdefault("tools_streamed", False)
    kwargs.setdefault("budget_spent", False)
    kwargs.setdefault("tool_choice_function_name", None)
    kwargs.setdefault("engine_finish_reason", None)
    assert _stream_finish_reason(**kwargs) == expected


def test_only_measured_families_require_stop_strings():
    """GLM-5.3 has no measured request rules, so it must not inherit the cap."""
    from vllm.entrypoints.openai.chat_completion import sampling_policy as sp

    assert sp.DEEPSEEK_V4.requires_stop_strings is True
    assert sp.QWEN3_8.requires_stop_strings is True
    assert sp.MINIMAX_M3.requires_stop_strings is True
    assert sp.GLM5.requires_stop_strings is False
