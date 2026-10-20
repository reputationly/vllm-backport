# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the segmented long-prompt encoder (WPF).

The parity check against a real checkpoint tokenizer is opt-in because CI has no
DeepSeek-V4.1 tokenizer files; set ``VLLM_DSV41_TOKENIZER_PATH`` to a model
directory (or tokenizer.json) to run it.
"""

import os

import pytest

from vllm.tokenizers.deepseek_v41 import (
    _SEGMENT_MAX_CHARS,
    _SegmentTokenCache,
    _split_segments,
)

TOKENIZER_PATH = os.environ.get("VLLM_DSV41_TOKENIZER_PATH")


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


def test_segment_cache_evicts_least_recently_used():
    cache = _SegmentTokenCache(capacity=4)
    cache.put(b"a", [1, 2])
    cache.put(b"b", [3, 4])
    assert cache.get(b"a") == [1, 2]
    cache.put(b"c", [5, 6])  # a is now most recently used, b is evicted
    assert cache.get(b"b") is None
    assert cache.get(b"a") == [1, 2]
    assert cache.get(b"c") == [5, 6]


@pytest.mark.skipif(
    not TOKENIZER_PATH, reason="set VLLM_DSV41_TOKENIZER_PATH to run parity"
)
def test_segmented_encode_matches_whole_string():
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

    tokenizer = DeepseekV41Tokenizer.from_pretrained(TOKENIZER_PATH)
    prompt = "".join(
        f"## turn {i}\n\nquestion {i} with 12345 and punctuation.\n"
        for i in range(30_000)
    )
    assert len(prompt) > _SEGMENT_MAX_CHARS

    reference = tokenizer.encode(prompt, add_special_tokens=False)
    assert tokenizer._dsv41_encode(prompt, {}) == reference
    # Second call is served from the chunk cache and must still be identical.
    assert tokenizer._dsv41_encode(prompt, {}) == reference
