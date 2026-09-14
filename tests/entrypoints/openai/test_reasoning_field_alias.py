# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VLLM_REASONING_OUTPUT_AS_REASONING_CONTENT.

Requests accept either spelling, but responses only ever emitted
``reasoning``. A client written against an API that emits
``reasoning_content`` therefore read nothing and silently dropped the chain
of thought. The env var renames the wire key so such a client works against
a vLLM server unmodified.
"""

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage

ENV = "VLLM_REASONING_OUTPUT_AS_REASONING_CONTENT"
THOUGHT = "Let me work through this step by step."


def _messages():
    return (
        ChatMessage(role="assistant", content="4", reasoning=THOUGHT),
        DeltaMessage(role="assistant", content="4", reasoning=THOUGHT),
    )


@pytest.mark.parametrize("msg", _messages(), ids=["non_streaming", "streaming"])
def test_default_emits_reasoning(msg, monkeypatch):
    """Off by default: the wire key stays ``reasoning``."""
    monkeypatch.delenv(ENV, raising=False)

    data = msg.model_dump()

    assert data["reasoning"] == THOUGHT
    assert "reasoning_content" not in data


@pytest.mark.parametrize("msg", _messages(), ids=["non_streaming", "streaming"])
def test_enabled_emits_reasoning_content(msg, monkeypatch):
    """Enabled: the same text is emitted under ``reasoning_content`` only."""
    monkeypatch.setenv(ENV, "1")

    data = msg.model_dump()

    assert data["reasoning_content"] == THOUGHT
    assert "reasoning" not in data
    # The rename must not disturb the rest of the message.
    assert data["content"] == "4"
    assert data["role"] == "assistant"


def test_enabled_leaves_messages_without_reasoning_alone(monkeypatch):
    """A message carrying no chain of thought gains no new key."""
    monkeypatch.setenv(ENV, "1")

    data = ChatMessage(role="assistant", content="4").model_dump()

    assert data["content"] == "4"
    assert data.get("reasoning") is None
    assert data.get("reasoning_content") is None


def test_enabled_survives_json_round_trip(monkeypatch):
    """model_dump_json is the path the server actually serializes through."""
    monkeypatch.setenv(ENV, "1")

    raw = DeltaMessage(reasoning=THOUGHT).model_dump_json()

    assert '"reasoning_content"' in raw
    assert '"reasoning"' not in raw
