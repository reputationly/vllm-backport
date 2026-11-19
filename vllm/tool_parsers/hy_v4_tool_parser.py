# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool parser for HYV4 / HY4 Flash checkpoints.

The checkpoint's `chat_template.jinja` teaches the model to emit

    <tool_calls:opensource>
    <tool_call:opensource>{name}<arg_key:opensource>{key}</arg_key:opensource>\
<arg_value:opensource>{value}</arg_value:opensource>...</tool_call:opensource>
    </tool_calls:opensource>

so a call is a plain-text envelope around key/value pairs rather than JSON. The
parser turns that into the OpenAI `tool_calls` structure, keeping any text
before the envelope as `content`.
"""

from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

_TOKEN_SUFFIX = ":opensource"
CALLS_BEGIN = f"<tool_calls{_TOKEN_SUFFIX}>"
CALLS_END = f"</tool_calls{_TOKEN_SUFFIX}>"
CALL_BEGIN = f"<tool_call{_TOKEN_SUFFIX}>"
CALL_END = f"</tool_call{_TOKEN_SUFFIX}>"
KEY_BEGIN = f"<arg_key{_TOKEN_SUFFIX}>"
KEY_END = f"</arg_key{_TOKEN_SUFFIX}>"
VALUE_BEGIN = f"<arg_value{_TOKEN_SUFFIX}>"
VALUE_END = f"</arg_value{_TOKEN_SUFFIX}>"

_CALL_RE = re.compile(
    re.escape(CALL_BEGIN) + r"(.*?)" + re.escape(CALL_END),
    re.DOTALL,
)
_ARG_RE = re.compile(
    re.escape(KEY_BEGIN) + r"(.*?)" + re.escape(KEY_END)
    + r"\s*"
    + re.escape(VALUE_BEGIN) + r"(.*?)" + re.escape(VALUE_END),
    re.DOTALL,
)


def _strip_envelope(text: str) -> str:
    """Drop the `<tool_calls>` wrapper and whitespace around it."""
    start = text.find(CALLS_BEGIN)
    end = text.find(CALLS_END)
    if start != -1 and end != -1 and end > start:
        return (text[:start] + text[end + len(CALLS_END) :]).strip()
    if start != -1:
        return text[:start].strip()
    if end != -1:
        return text[end + len(CALLS_END) :].strip()
    return text


class HYV4ToolParser(ToolParser):
    """Parses the `<tool_call>` envelope used by HYV4 checkpoints."""

    def __init__(self, tokenizer, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Streaming keeps the raw text; calls are emitted once their closing
        # token shows up, which avoids guessing at partial argument names.
        self._emitted: int = 0

    @staticmethod
    def _parse_call(body: str) -> tuple[str, dict[str, str]] | None:
        name, _, rest = body.partition(KEY_BEGIN)
        name = name.strip()
        if not name:
            return None
        arguments: dict[str, str] = {}
        for key, value in _ARG_RE.findall(KEY_BEGIN + rest):
            arguments[key.strip()] = value
        return name, arguments

    def extract_tool_calls(
        self, model_output: str, request=None
    ) -> ExtractedToolCallInformation:
        from vllm.entrypoints.generate.base.protocol import (
            FunctionCall,
            ToolCall,
        )

        if CALL_BEGIN not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[])

        calls: list[ToolCall] = []
        for body in _CALL_RE.findall(model_output):
            parsed = self._parse_call(body)
            if parsed is None:
                continue
            name, arguments = parsed
            calls.append(
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=name, arguments=self._dump(arguments)
                    ),
                )
            )
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[])

        content = _strip_envelope(model_output)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=calls,
            content=content or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request=None,
    ) -> DeltaMessage | None:
        """Emit each call as soon as its closing tag is complete."""
        from vllm.entrypoints.generate.base.protocol import (
            DeltaFunctionCall,
            DeltaToolCall,
        )

        bodies = _CALL_RE.findall(current_text)
        if len(bodies) <= self._emitted:
            # Nothing complete yet: suppress content that belongs to the
            # envelope, otherwise stream it through untouched.
            if CALL_BEGIN in current_text or CALLS_BEGIN in current_text:
                return None
            return DeltaMessage(content=delta_text) if delta_text else None

        deltas: list[DeltaToolCall] = []
        for index in range(self._emitted, len(bodies)):
            parsed = self._parse_call(bodies[index])
            if parsed is None:
                continue
            name, arguments = parsed
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=name, arguments=self._dump(arguments)
                    ),
                )
            )
        self._emitted = len(bodies)
        return DeltaMessage(tool_calls=deltas) if deltas else None

    @staticmethod
    def _dump(arguments: dict[str, str]) -> str:
        import json

        return json.dumps(arguments, ensure_ascii=False)
