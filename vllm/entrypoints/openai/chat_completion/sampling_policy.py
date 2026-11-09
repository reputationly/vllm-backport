# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendor API profiles: make a self-hosted model answer like its hosted API.

The hosted APIs of the models we serve are not "forward every knob" services.
They validate some parameters, silently ignore others, and only accept a
specific set of thinking switches. Measured behaviour (see
``docs/deployment/vendor-api-parity-design.md``) is encoded here as one
profile per model family, applied in four stages:

1. request gate  -- validate/rewrite the raw JSON before pydantic parsing;
2. template args -- the gate decides thinking, the user's kwargs are dropped;
3. sampling      -- temperature/top_p/top_k/penalties/n follow the hosted API;
4. output        -- trailing-newline stripping and answer-length limits.

Models without a profile (``policy_for`` returns ``None``) are untouched.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)

ThinkingStyle = Literal["deepseek", "qwen", "always"]

_DEEPSEEK_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)
_QWEN_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_MINIMAX_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})


# ---------------------------------------------------------------------------
# Values exchanged between the request and the sampling stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VendorState:
    """Thinking decision shared by stages 2-4 (derived once per request)."""

    thinking: bool = True
    effort: str | None = None
    #: Answer-only token budget (Qwen). ``None`` means "no extra limit".
    body_limit: int | None = None
    #: The request used the legacy ``functions``/``function_call`` shape, so
    #: the response must answer in that shape too (Qwen).
    legacy_functions: bool = False


@dataclass(frozen=True)
class SamplingInputs:
    """Resolved sampling knobs entering stage 3."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    frequency_penalty: float
    seed: int | None
    n: int
    thinking: bool
    effort: str | None
    #: Names of the knobs the caller set explicitly; the rest may take the
    #: family's documented default instead of the engine's.
    specified: frozenset[str] = frozenset()


@dataclass(frozen=True)
class VendorProfile:
    """Everything one model family needs to look like its hosted API."""

    name: str
    thinking_style: ThinkingStyle
    gate: Callable[[dict[str, Any]], dict[str, Any]]
    sampling: Callable[[SamplingInputs], SamplingInputs]
    state: Callable[[Mapping[str, Any]], VendorState]
    #: Qwen: ``max_tokens`` caps the answer only, thinking has its own budget.
    answer_only_max_tokens: bool = False
    #: Qwen: the template emits a blank line after ``</think>``.
    strip_leading_newlines: bool = False
    #: The family needs ``VLLM_MAX_STOP_STRINGS`` raised so its stop-string
    #: rule fits through pydantic. GLM-5.3 has no measured request rules yet,
    #: so it must not inherit that deployment requirement.
    requires_stop_strings: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reject(message: str, parameter: str, value: Any = None) -> None:
    raise VLLMValidationError(message, parameter=parameter, value=value)


def _num(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    _reject(f"`{key}` must be a number", key, value)
    return None


def _require_range(
    data: Mapping[str, Any],
    key: str,
    low: float,
    high: float,
    *,
    low_open: bool = False,
    high_open: bool = False,
    message: str | None = None,
) -> None:
    value = _num(data, key)
    if value is None:
        return
    too_low = value <= low if low_open else value < low
    too_high = value >= high if high_open else value > high
    if too_low or too_high:
        left = "(" if low_open else "["
        right = ")" if high_open else "]"
        _reject(
            message
            or f"`{key}` must be in {left}{low:g}, {high:g}{right}, got {value:g}",
            key,
            value,
        )


def _messages_text(messages: Any) -> str:
    """Concatenated text of every message part (tool schemas excluded)."""
    chunks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                # ``tools`` schemas live outside ``messages``; skip any nested
                # "json" spelling inside structured payloads only for tool defs.
                if key in {"image_url", "input_audio"}:
                    continue
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(messages)
    return "\n".join(chunks)


def _require_json_hint(data: Mapping[str, Any], message: str) -> None:
    response_format = data.get("response_format")
    if not isinstance(response_format, Mapping):
        return
    if response_format.get("type") != "json_object":
        return
    if "json" not in _messages_text(data.get("messages") or []).lower():
        _reject(message, "response_format", "json_object")


def _thinking_type(data: Mapping[str, Any]) -> Any:
    thinking = data.get("thinking")
    if isinstance(thinking, Mapping):
        return thinking.get("type")
    return None


def _stop_list(data: dict[str, Any]) -> list[str]:
    stop = data.get("stop")
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [item for item in stop if isinstance(item, str)]
    return []


def _drop(data: dict[str, Any], *keys: str) -> None:
    for key in keys:
        data.pop(key, None)


def _drop_stream_options_without_stream(data: dict[str, Any]) -> None:
    """``stream_options`` on a non-streaming request is accepted and ignored."""
    if data.get("stream_options") and not data.get("stream"):
        _drop(data, "stream_options")


def _allowed_escape_hatches() -> frozenset[str]:
    raw = os.environ.get("VLLM_VENDOR_API_PROFILE_ALLOW", "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


# ---------------------------------------------------------------------------
# DeepSeek V4.1
# ---------------------------------------------------------------------------


def _deepseek_gate(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    _drop_stream_options_without_stream(data)
    _require_range(data, "temperature", 0.0, 2.0)
    _require_range(
        data,
        "top_p",
        0.0,
        1.0,
        low_open=True,
        message="top_p must be in (0, 1]",
    )
    for key in ("presence_penalty", "frequency_penalty"):
        _require_range(data, key, -2.0, 2.0)
    for key in ("max_tokens", "max_completion_tokens"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value < 1:
            _reject(f"`{key}` must be at least 1", key, value)

    top_logprobs = data.get("top_logprobs")
    if (
        isinstance(top_logprobs, int)
        and not isinstance(top_logprobs, bool)
        and (top_logprobs < 0 or top_logprobs > 20)
    ):
        _reject("top_logprobs must be in [0, 20]", "top_logprobs", top_logprobs)
    if not data.get("logprobs"):
        # The hosted API accepts ``top_logprobs`` on its own and ignores it.
        _drop(data, "top_logprobs")

    if data.get("logit_bias"):
        _reject(
            "Reasoning model does not support n > 1, logit_bias",
            "logit_bias",
        )

    stop = _stop_list(data)
    if len(stop) > 16:
        _reject("stop must contain at most 16 strings", "stop", len(stop))

    thinking_type = _thinking_type(data)
    if thinking_type is not None and thinking_type not in {"enabled", "disabled"}:
        _reject(
            "The parameter `type` specified in the request are not valid: "
            "invalid value",
            "thinking.type",
            thinking_type,
        )

    effort = data.get("reasoning_effort")
    if isinstance(effort, str) and effort not in _DEEPSEEK_EFFORTS:
        _reject(
            f"`reasoning_effort` must be one of {sorted(_DEEPSEEK_EFFORTS)}",
            "reasoning_effort",
            effort,
        )
    if effort == "ultra":
        data["reasoning_effort"] = "max"

    _require_json_hint(
        data,
        "Prompt must contain the word 'json' in some form to use "
        "'response_format' of type 'json_object'",
    )

    # The hosted API always answers with a single choice.
    n = data.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n != 1:
        logger.info(
            "vendor-profile[deepseek]: coercing n=%s to 1 (hosted API returns "
            "a single choice)",
            n,
        )
        data["n"] = 1

    if data.get("tool_choice") == "none" and data.get("tools"):
        _drop(data, "tools", "tool_choice")

    allowed = _allowed_escape_hatches()
    for key in (
        "chat_template_kwargs",
        "enable_thinking",
        "top_k",
        "min_p",
        "repetition_penalty",
        "seed",
        "functions",
        "function_call",
    ):
        if key not in allowed:
            _drop(data, key)
    return data


def _deepseek_state(data: Mapping[str, Any]) -> VendorState:
    thinking_type = _thinking_type(data)
    effort = data.get("reasoning_effort")
    if thinking_type is not None:
        thinking = thinking_type != "disabled"
        # ``thinking.type`` outranks ``reasoning_effort``; forwarding "none"
        # would make the renderer drop thinking mode while the sampler still
        # applies thinking-mode semantics.
        if thinking and effort == "none":
            effort = None
    elif effort == "none":
        thinking = False
    else:
        thinking = True
    return VendorState(thinking=thinking, effort=effort if thinking else None)


def _deepseek_sampling(inputs: SamplingInputs) -> SamplingInputs:
    updates: dict[str, Any] = {
        # The hosted API accepts but never applies these.
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": None,
        "n": 1,
    }
    if inputs.thinking:
        updates["temperature"] = 1.0
        # Default 1.0 regardless of the model's generation_config (design 4.1),
        # floored to 0.95 when the caller asks for less.
        if "top_p" not in inputs.specified:
            updates["top_p"] = 1.0
        else:
            updates["top_p"] = max(0.95, min(1.0, inputs.top_p))
    else:
        updates["top_p"] = 1.0
    return replace(inputs, **updates)


DEEPSEEK_V4 = VendorProfile(
    name="deepseek-v4",
    thinking_style="deepseek",
    gate=_deepseek_gate,
    sampling=_deepseek_sampling,
    state=_deepseek_state,
    requires_stop_strings=True,
)


# ---------------------------------------------------------------------------
# Qwen3.8
# ---------------------------------------------------------------------------


def _qwen_gate(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    _drop_stream_options_without_stream(data)
    _require_range(
        data,
        "temperature",
        0.0,
        2.0,
        high_open=True,
        message="Temperature should be in [0.0, 2.0)",
    )
    _require_range(
        data,
        "top_p",
        0.0,
        1.0,
        message="'top_p' must be in (0.0, 1.0]",
    )
    top_k = data.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k < 0:
        _reject("top_k must be a non-negative integer", "top_k", top_k)
    for key in ("presence_penalty", "frequency_penalty"):
        _require_range(data, key, -2.0, 2.0)

    top_logprobs = data.get("top_logprobs")
    if (
        isinstance(top_logprobs, int)
        and not isinstance(top_logprobs, bool)
        and (top_logprobs < 0 or top_logprobs > 5)
    ):
        _reject("Range of top_logprobs should be [0, 5]", "top_logprobs", top_logprobs)
    if not data.get("logprobs"):
        _drop(data, "top_logprobs")

    thinking_type = _thinking_type(data)
    if thinking_type is not None and thinking_type not in {
        "enabled",
        "disabled",
        "auto",
    }:
        _reject(
            '\'type\' must be in ["enabled", "disabled", "auto"]',
            "thinking.type",
            thinking_type,
        )
    effort = data.get("reasoning_effort")
    if isinstance(effort, str) and effort not in _QWEN_EFFORTS:
        _reject(
            "'reasoning_effort' must be one of: 'none', 'minimal', 'low', "
            "'medium', 'high', 'xhigh', 'max'",
            "reasoning_effort",
            effort,
        )
    if effort is not None and data.get("thinking_budget") is not None:
        _reject(
            "reasoning_effort and thinking_budget cannot be set together",
            "thinking_budget",
        )

    state = _qwen_state(data)
    n = data.get("n")
    if isinstance(n, int) and not isinstance(n, bool):
        if state.thinking and n != 1:
            _reject(
                "The n parameter must be 1 when enable_thinking is true",
                "n",
                n,
            )
        if not state.thinking and (n < 0 or n > 4):
            _reject("Range of n should be [1, 4]", "n", n)
        if n == 0:
            data["n"] = 1

    for key in ("max_tokens", "max_completion_tokens"):
        value = data.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (value < 1 or value > 131072)
        ):
            _reject(f"Range of {key} should be [1, 131072]", key, value)

    budget = data.get("thinking_budget")
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
    ):
        _reject(
            "`thinking_budget` must be a non-negative integer",
            "thinking_budget",
            budget,
        )

    _require_json_hint(
        data,
        "'messages' must contain the word 'json' in some form, to use "
        "'response_format' of type 'json_object'",
    )

    if data.get("tool_choice") == "none" and data.get("tools"):
        _drop(data, "tools", "tool_choice")

    # ``preserve_thinking=False`` drops assistant reasoning from the history.
    if data.get("preserve_thinking") is False:
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    message.pop("reasoning_content", None)
                    message.pop("reasoning", None)

    # Legacy ``functions``/``function_call`` are still served by the hosted
    # API; translate them to the tools shape and answer in the legacy shape.
    functions = data.get("functions")
    if functions and not data.get("tools"):
        data["tools"] = [
            {"type": "function", "function": fn}
            for fn in functions
            if isinstance(fn, Mapping)
        ]
        function_call = data.get("function_call")
        if function_call == "none":
            data["tool_choice"] = "none"
        elif function_call == "auto":
            data["tool_choice"] = "auto"
        elif isinstance(function_call, Mapping):
            name = function_call.get("name")
            if isinstance(name, str):
                data["tool_choice"] = {
                    "type": "function",
                    "function": {"name": name},
                }
        data["_vendor_legacy_functions"] = True

    allowed = _allowed_escape_hatches()
    for key in ("min_p", "logit_bias", "functions", "function_call"):
        if key not in allowed:
            _drop(data, key)
    return data


def _qwen_effort_map(effort: str | None) -> str | None:
    if effort in {"max", "high"}:
        return "xhigh"
    if effort == "minimal":
        return "low"
    return effort


def _qwen_state(data: Mapping[str, Any]) -> VendorState:
    thinking_type = _thinking_type(data)
    if thinking_type is not None:
        thinking = thinking_type != "disabled"
    elif data.get("enable_thinking") is not None:
        thinking = bool(data["enable_thinking"])
    else:
        template_kwargs = data.get("chat_template_kwargs")
        if isinstance(template_kwargs, Mapping) and (
            "enable_thinking" in template_kwargs
        ):
            thinking = bool(template_kwargs["enable_thinking"])
        else:
            thinking = data.get("reasoning_effort") != "none"

    effort = _qwen_effort_map(data.get("reasoning_effort"))
    if thinking and effort == "none":
        # A higher-priority switch asked for thinking; the template would read
        # "none" as "do not think", so do not forward it.
        effort = None
    body_limit = None
    max_tokens = data.get("max_tokens")
    if (
        data.get("max_completion_tokens") is None
        and isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
    ):
        body_limit = max_tokens
    return VendorState(
        thinking=thinking,
        effort=effort,
        body_limit=body_limit,
        legacy_functions=bool(data.get("_vendor_legacy_functions")),
    )


def _qwen_sampling(inputs: SamplingInputs) -> SamplingInputs:
    # The hosted API accepts top_p=0; SamplingParams rejects it, so pin the
    # degenerate value to the smallest legal probability.
    updates: dict[str, Any] = {"min_p": 0.0}
    if inputs.thinking:
        updates["temperature"] = min(max(inputs.temperature, 0.6), 1.0)
        updates["top_k"] = inputs.top_k
        if inputs.top_k <= 0 or inputs.top_k > 100:
            updates["top_k"] = -1
        if "top_k" not in inputs.specified:
            updates["top_k"] = 20
        if "top_p" not in inputs.specified:
            updates["top_p"] = 0.95
        if "presence_penalty" not in inputs.specified:
            updates["presence_penalty"] = 0.0
    else:
        if "temperature" not in inputs.specified:
            updates["temperature"] = 0.7
        else:
            updates["temperature"] = inputs.temperature
        if "top_p" not in inputs.specified:
            updates["top_p"] = 0.8
        if "top_p" in updates and updates["top_p"] <= 0:
            updates["top_p"] = 1e-6
        if "presence_penalty" not in inputs.specified:
            updates["presence_penalty"] = 1.5
    if "top_p" not in updates:
        updates["top_p"] = inputs.top_p if inputs.top_p > 0 else 1e-6
    return replace(inputs, **updates)


QWEN3_8 = VendorProfile(
    name="qwen3.8",
    thinking_style="qwen",
    gate=_qwen_gate,
    sampling=_qwen_sampling,
    state=_qwen_state,
    answer_only_max_tokens=True,
    strip_leading_newlines=True,
    requires_stop_strings=True,
)


# ---------------------------------------------------------------------------
# MiniMax M3.1
# ---------------------------------------------------------------------------


def _minimax_gate(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    _drop_stream_options_without_stream(data)
    # The hosted API accepts top_p=0 (measured 3/3) even though its error text
    # advertises (0, 1]; only values outside [0, 1] are rejected.
    _require_range(
        data,
        "top_p",
        0.0,
        1.0,
        message="invalid params, param 'top_p' should be in (0,1] (2013)",
    )
    n = data.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        _reject(
            "invalid params, model[MiniMax-M3.1-Flash] does not support n > 1 (2013)",
            "n",
            n,
        )
    thinking_type = _thinking_type(data)
    if thinking_type == "disabled":
        _reject(
            'model "MiniMax-M3.1-Flash" requires adaptive thinking; '
            'thinking.type="disabled" (including reasoning.effort=none) is not '
            "allowed (2013)",
            "thinking.type",
            thinking_type,
        )
    if thinking_type is not None and thinking_type not in {
        "enabled",
        "adaptive",
        "disabled",
    }:
        _reject(
            "invalid params, `thinking.type` must be one of adaptive, enabled (2013)",
            "thinking.type",
            thinking_type,
        )
    effort = data.get("reasoning_effort")
    if effort == "none":
        _reject(
            'invalid params, model "MiniMax-M3.1-Flash" requires adaptive '
            "thinking; reasoning effort none is not allowed (2013)",
            "reasoning_effort",
            effort,
        )
    if isinstance(effort, str) and effort not in _MINIMAX_EFFORTS:
        _reject(
            f'invalid params, invalid reasoning_effort: "{effort}" '
            "(allowed: low, medium, high, xhigh, max) (2013)",
            "reasoning_effort",
            effort,
        )
    if data.get("max_tokens") == 0:
        _drop(data, "max_tokens")

    # The model always splits reasoning out of the answer, so the vendor
    # rejects an explicit opt-out (measured 3/3).
    if data.get("reasoning_split") is False:
        _reject(
            'invalid params, model "MiniMax-M3.1-Flash" requires '
            "reasoning_split=true (2013)",
            "reasoning_split",
            False,
        )

    # ``none`` is honoured (measured 3/3 against the vendor endpoint: no tool
    # call, a plain answer). Every other value behaves like ``auto``.
    if data.get("tool_choice") == "none" and data.get("tools"):
        _drop(data, "tools", "tool_choice")
    elif data.get("tool_choice") is not None and data.get("tool_choice") != "auto":
        logger.info(
            "vendor-profile[minimax]: rewriting tool_choice=%r to 'auto' "
            "(hosted API only implements auto)",
            data.get("tool_choice"),
        )
        data["tool_choice"] = "auto"

    allowed = _allowed_escape_hatches()
    for key in (
        "stop",
        "logprobs",
        "top_logprobs",
        "response_format",
        "logit_bias",
        "seed",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "parallel_tool_calls",
        "enable_thinking",
        "chat_template_kwargs",
        "functions",
        "function_call",
    ):
        if key not in allowed:
            _drop(data, key)
    return data


def _minimax_state(data: Mapping[str, Any]) -> VendorState:
    effort = data.get("reasoning_effort") or "max"
    if effort == "minimal":
        effort = "low"
    return VendorState(thinking=True, effort=effort)


def _minimax_sampling(inputs: SamplingInputs) -> SamplingInputs:
    # top_p is pinned to 0.95, but keep SamplingParams happy for top_p=0 callers.
    return replace(
        inputs,
        temperature=1.0,
        top_p=0.95 if inputs.top_p > 0 else 1e-6,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        seed=None,
        n=1,
    )


MINIMAX_M3 = VendorProfile(
    name="minimax-m3",
    thinking_style="always",
    gate=_minimax_gate,
    sampling=_minimax_sampling,
    state=_minimax_state,
    requires_stop_strings=True,
)


# ---------------------------------------------------------------------------
# GLM-5.3
# ---------------------------------------------------------------------------


def _glm5_gate(data: dict[str, Any]) -> dict[str, Any]:
    # No hosted-API request rules have been measured for GLM-5.3, so the
    # request is passed through verbatim: this profile only restores the
    # sampling clamp the family needs while thinking.
    return dict(data)


def _glm5_state(data: Mapping[str, Any]) -> VendorState:
    effort = data.get("reasoning_effort")
    return VendorState(
        thinking=True,
        effort=effort if isinstance(effort, str) else None,
    )


def _glm5_sampling(inputs: SamplingInputs) -> SamplingInputs:
    # GLM-5.3 cannot stop thinking; the hosted API caps temperature at 1.0 and
    # pins top_p to 0.95. Everything else is forwarded.
    return replace(
        inputs,
        temperature=min(inputs.temperature, 1.0),
        top_p=0.95,
    )


GLM5 = VendorProfile(
    name="glm-5.3",
    thinking_style="always",
    gate=_glm5_gate,
    sampling=_glm5_sampling,
    state=_glm5_state,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (model_type) -> profile. model_type is checked first because it is present
# on every config; architectures add the text-only aliases.
_POLICIES: dict[str, VendorProfile] = {
    "deepseek_v4": DEEPSEEK_V4,
    "deepseek_v41": DEEPSEEK_V4,
    "deepseek_v4_text": DEEPSEEK_V4,
    "deepseek_v41_text": DEEPSEEK_V4,
    "minimax_m3": MINIMAX_M3,
    "minimax_m3_vl": MINIMAX_M3,
    "qwen4_exp": QWEN3_8,
    "qwen4_exp_text": QWEN3_8,
    "glm5_next": GLM5,
}

_ARCHITECTURE_POLICIES: dict[str, VendorProfile] = {
    "DeepseekV4ForCausalLM": DEEPSEEK_V4,
    "DeepseekV41ForCausalLM": DEEPSEEK_V4,
    "MiniMaxM3SparseForCausalLM": MINIMAX_M3,
    "MiniMaxM3SparseForConditionalGeneration": MINIMAX_M3,
    "Qwen4ExpForCausalLM": QWEN3_8,
    "Qwen4ExpForConditionalGeneration": QWEN3_8,
    "Glm5NextForCausalLM": GLM5,
    "Glm5NextForConditionalGeneration": GLM5,
}

#: Profile of the model this process serves (one model per process).
ACTIVE_PROFILE: VendorProfile | None = None


def set_active_profile(profile: VendorProfile | None) -> None:
    global ACTIVE_PROFILE
    ACTIVE_PROFILE = profile


def profile_gate_enabled() -> bool:
    return bool(envs.VLLM_VENDOR_SAMPLING_POLICY) and bool(envs.VLLM_VENDOR_API_PROFILE)


def policy_for(model_config: Any) -> VendorProfile | None:
    """Profile for a model, or ``None`` when it has no vendor rules."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return None
    model_type = getattr(hf_config, "model_type", None)
    if model_type in _POLICIES:
        return _POLICIES[model_type]
    for architecture in getattr(hf_config, "architectures", None) or ():
        profile = _ARCHITECTURE_POLICIES.get(architecture)
        if profile is not None:
            return profile
    return None


def thinking_enabled(
    template_kwargs: dict[str, Any], style: ThinkingStyle = "deepseek"
) -> bool:
    """Fallback thinking detection for models without a profile."""
    if style == "always":
        return True
    if style == "qwen":
        if "enable_thinking" in template_kwargs:
            return bool(template_kwargs["enable_thinking"])
        return True
    if "thinking" in template_kwargs or "enable_thinking" in template_kwargs:
        enabled = bool(template_kwargs.get("thinking")) or bool(
            template_kwargs.get("enable_thinking")
        )
    else:
        enabled = True
    if template_kwargs.get("reasoning_effort") == "none":
        enabled = False
    return enabled
