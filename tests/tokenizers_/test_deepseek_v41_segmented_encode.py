# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the segmented long-prompt encoder (WPF).

The parity checks against a real checkpoint tokenizer are opt-in because CI has
no DeepSeek-V4.1 tokenizer files; set ``VLLM_DSV41_TOKENIZER_PATH`` to a model
directory (or tokenizer.json) to run them.
"""

import json
import os

import pytest

from vllm.tokenizers.deepseek_v41 import (
    _SEGMENT_MAX_CHARS,
    _SegmentTokenCache,
    _split_segments,
    get_deepseek_v41_tokenizer,
)

TOKENIZER_PATH = os.environ.get("VLLM_DSV41_TOKENIZER_PATH")


class _LenTokenizer:
    """Deterministic tokenizer: one token per character of a chunk."""

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        return [len(text)]


def test_split_segments_roundtrips_and_respects_cap():
    text = "".join(f"line {i}\ncontent {i}\n" for i in range(30_000))
    segments = _split_segments(text)
    assert "".join(segments) == text
    assert len(segments) > 1
    assert all(len(segment) <= _SEGMENT_MAX_CHARS for segment in segments)
    # Every non-final chunk ends right after a newline, so the next chunk starts
    # at a pre-tokenizer boundary.
    assert all(segment.endswith("\n") for segment in segments[:-1])
    assert all(not segment[:1].isspace() for segment in segments[1:])


def test_split_segments_keeps_short_text_whole():
    assert _split_segments("hello\nworld") == ["hello\nworld"]


def test_split_segments_never_cuts_without_boundary():
    """Single-line JSON has no newline: the whole prompt stays one segment."""
    text = json.dumps({f"k{i}": "v" * 20 for i in range(12_000)})
    assert "\n" not in text
    assert len(text) > _SEGMENT_MAX_CHARS
    assert _split_segments(text) == [text]


def test_split_segments_extends_past_window_to_a_boundary():
    """A boundary past the window is used instead of cutting at the window."""
    text = "a" * (2 * _SEGMENT_MAX_CHARS) + "\nnext line\n" + "b" * 100
    segments = _split_segments(text)
    assert "".join(segments) == text
    assert len(segments) == 2
    assert segments[0].endswith("\n")
    assert segments[1].startswith("next line")
    assert len(segments[0]) > _SEGMENT_MAX_CHARS


def test_segment_cache_evicts_least_recently_used():
    cache = _SegmentTokenCache(capacity=4)
    cache.put(b"a", [1, 2])
    cache.put(b"b", [3, 4])
    assert cache.get(b"a") == [1, 2]
    cache.put(b"c", [5, 6])  # a is now most recently used, b is evicted
    assert cache.get(b"b") is None
    assert cache.get(b"a") == [1, 2]
    assert cache.get(b"c") == [5, 6]


def test_segmented_encode_keeps_ids_when_cache_evicts():
    """The assembly must use this call's ids, not re-read an evicting LRU."""
    prompt = "".join(f"line {i}\ncontent {i}\n" for i in range(30_000))
    segments = _split_segments(prompt)
    assert len(segments) > 1

    tokenizer = get_deepseek_v41_tokenizer(_LenTokenizer())
    tokenizer._vllm_dsv41_segment_cache = _SegmentTokenCache(capacity=0)
    assert tokenizer._dsv41_encode(prompt, {}) == [len(s) for s in segments]


@pytest.mark.skipif(
    not TOKENIZER_PATH, reason="set VLLM_DSV41_TOKENIZER_PATH to run parity"
)
@pytest.mark.parametrize(
    "case",
    ["multiline", "single_line_json", "digits", "base64"],
)
def test_segmented_encode_matches_whole_string(case):
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

    if case == "multiline":
        text = "".join(
            f"## turn {i}\n\nquestion {i} with 12345 and punctuation.\n"
            for i in range(30_000)
        )
    elif case == "single_line_json":
        text = json.dumps({f"k{i}": "v" * 20 for i in range(12_000)})
    elif case == "digits":
        text = "1234567890" * 20_000
    else:
        text = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 5_000
    assert len(text) > _SEGMENT_MAX_CHARS
    assert "\n" not in text if case != "multiline" else True

    tokenizer = DeepseekV41Tokenizer.from_pretrained(TOKENIZER_PATH)
    reference = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer._dsv41_encode(text, {}) == reference
    # Second call is served from the chunk cache and must still be identical.
    assert tokenizer._dsv41_encode(text, {}) == reference
