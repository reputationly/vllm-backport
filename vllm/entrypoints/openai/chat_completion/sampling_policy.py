# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendor sampling semantics for thinking-mode chat models.

The hosted APIs of the models we serve do not simply forward the caller's
sampling knobs while the model is thinking. Measured and documented behaviour:

* DeepSeek V4/V4.1 (``deepseek-flash``): thinking mode does not support
  ``temperature``/``presence_penalty``/``frequency_penalty`` (accepted and
  ignored) and clamps ``top_p`` to ``[0.95, 1.0]``; non-thinking mode pins
  ``top_p`` to 1.0.  https://api-docs.deepseek.com/guides/thinking_mode
* MiniMax M3/M3.1-Flash: ``temperature`` is ignored (always 1) and ``top_p``
  is always 0.95; ``top_k`` is ignored.  M3.1-Flash always thinks.
  https://platform.minimax.io/docs/api-reference/text-anthropic-api
* Qwen3.8 (``generation_config.json``): temperature 1.0, top_p 0.95, top_k 20
  for thinking; Qwen honours ``temperature`` but documents this range.
* GLM-5.3/5.3-Flash: thinking cannot be disabled and the API defaults to
  temperature 1.0 / top_p 0.95.
  https://docs.bigmodel.cn/cn/guide/start/migrate-to-glm-new

Forwarding e.g. ``temperature=1.8`` into a long thinking stream is what makes
the same request degenerate here while the hosted channels stay clean, so the
chat entry applies the vendor rules below for the families that have them.
Any model without a policy is untouched.
"""

from dataclasses import dataclass
from typing import Any, Literal

from vllm.logger import init_logger

logger = init_logger(__name__)

ThinkingStyle = Literal["deepseek", "qwen", "always"]


@dataclass(frozen=True)
class SamplingPolicy:
    """Sampling semantics applied while a model is thinking."""

    name: str
    thinking_style: ThinkingStyle
    # ``None`` means "keep the caller's value".
    temperature: float | None = None
    max_temperature: float | None = None
    top_p: float | None = None
    top_p_floor: float | None = None
    top_p_default: float | None = None
    top_p_default_with_tools: float | None = None
    top_k: int | None = None
    clear_penalties: bool = False
    # Applied in non-thinking (chat) mode; ``None`` leaves the request alone.
    chat_top_p: float | None = None


DEEPSEEK_V4 = SamplingPolicy(
    name="deepseek-v4",
    thinking_style="deepseek",
    temperature=1.0,
    top_p_floor=0.95,
    top_p_default=1.0,
    top_p_default_with_tools=0.95,
    clear_penalties=True,
    chat_top_p=1.0,
)

MINIMAX_M3 = SamplingPolicy(
    name="minimax-m3",
    thinking_style="always",
    temperature=1.0,
    top_p=0.95,
    top_k=-1,  # ignored by the hosted API
    clear_penalties=True,
)

QWEN3_8 = SamplingPolicy(
    name="qwen3.8",
    thinking_style="qwen",
    max_temperature=1.0,
    top_p=0.95,
    top_k=20,
)

GLM5 = SamplingPolicy(
    name="glm-5.3",
    thinking_style="always",
    max_temperature=1.0,
    top_p=0.95,
)

# (model_type, architecture) -> policy. model_type is checked first because it
# is present on every config; architectures add the text-only aliases.
_POLICIES: dict[str, SamplingPolicy] = {
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

_ARCHITECTURE_POLICIES: dict[str, SamplingPolicy] = {
    "DeepseekV4ForCausalLM": DEEPSEEK_V4,
    "DeepseekV41ForCausalLM": DEEPSEEK_V4,
    "MiniMaxM3SparseForCausalLM": MINIMAX_M3,
    "MiniMaxM3SparseForConditionalGeneration": MINIMAX_M3,
    "Qwen4ExpForCausalLM": QWEN3_8,
    "Qwen4ExpForConditionalGeneration": QWEN3_8,
    "Glm5NextForCausalLM": GLM5,
    "Glm5NextForConditionalGeneration": GLM5,
}


def policy_for(model_config: Any) -> SamplingPolicy | None:
    """Sampling policy for a model, or ``None`` when it has no vendor rules."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return None
    model_type = getattr(hf_config, "model_type", None)
    if model_type in _POLICIES:
        return _POLICIES[model_type]
    for architecture in getattr(hf_config, "architectures", None) or ():
        policy = _ARCHITECTURE_POLICIES.get(architecture)
        if policy is not None:
            return policy
    return None


def thinking_enabled(
    template_kwargs: dict[str, Any], style: ThinkingStyle = "deepseek"
) -> bool:
    """Whether the chat template will emit thinking-mode tokens."""
    if style == "always":
        return True
    if style == "qwen":
        # Qwen templates take a plain boolean and think by default.
        if "enable_thinking" in template_kwargs:
            return bool(template_kwargs["enable_thinking"])
        return True
    # DeepSeek V4: mirrors _DeepseekV4Tokenizer.apply_chat_template.
    if "thinking" in template_kwargs or "enable_thinking" in template_kwargs:
        enabled = bool(template_kwargs.get("thinking")) or bool(
            template_kwargs.get("enable_thinking")
        )
    else:
        enabled = True
    if template_kwargs.get("reasoning_effort") == "none":
        enabled = False
    return enabled
