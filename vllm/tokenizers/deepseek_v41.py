# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from transformers import TokenizersBackend

import vllm.envs as envs
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

logger = init_logger(__name__)

_SEGMENT_MAX_CHARS = 64 * 1024
_MIN_SEGMENTED_CHARS = 8 * 1024

# Cut right after a newline whose next character is not whitespace: the word
# regex never spans it.
_BOUNDARY_AFTER_NEWLINE = re.compile(r"\n(?=\S)")
# Cut right before a whitespace run (previous character is non-whitespace).
# The word regex attaches one leading space to the following word, so the run
# must stay with the segment that starts at it, never with the one that ends
# before it. Added tokens contain no whitespace, so this cannot split one.
_BOUNDARY_BEFORE_WHITESPACE = re.compile(r"(?<=\S)(?=\s)")
# Only the tail of a window is scanned for a boundary in the common case.
_BOUNDARY_TAIL_CHARS = 4 * 1024


def _last_boundary(
    pattern: re.Pattern[str], text: str, start: int, end: int, *, after: bool
) -> int:
    cut = -1
    for match in pattern.finditer(text, start, end):
        cut = match.end() if after else match.start()
    return cut


def _first_boundary(
    pattern: re.Pattern[str], text: str, start: int, end: int, *, after: bool
) -> int:
    match = pattern.search(text, start, end)
    if match is None:
        return -1
    return match.end() if after else match.start()


def _find_cut(text: str, start: int, end: int) -> int:
    """Return a pre-tokenizer-safe cut position in ``(start, len(text))``.

    Boundaries are dense in practice, so scan only the tail of the window first;
    the full-window scan runs only for long unbroken runs.
    """
    n = len(text)
    tail = max(start, end - _BOUNDARY_TAIL_CHARS)
    cut = _first_boundary(_BOUNDARY_AFTER_NEWLINE, text, tail, end, after=True)
    if cut > start:
        return cut
    cut = _first_boundary(_BOUNDARY_BEFORE_WHITESPACE, text, tail, end, after=False)
    if cut > start:
        return cut
    cut = _last_boundary(_BOUNDARY_AFTER_NEWLINE, text, start, end, after=True)
    if cut > start:
        return cut
    cut = _last_boundary(_BOUNDARY_BEFORE_WHITESPACE, text, start, end, after=False)
    if cut > start:
        return cut
    cut = _first_boundary(_BOUNDARY_AFTER_NEWLINE, text, end, n, after=True)
    if cut > start:
        return cut
    return _first_boundary(_BOUNDARY_BEFORE_WHITESPACE, text, end, n, after=False)


def _split_segments(text: str) -> list[str]:
    """Split ``text`` at pre-tokenizer-safe boundaries, at most 64k chars each.

    Boundaries are a newline followed by a non-whitespace character, or the
    start of a whitespace run whose previous character is not whitespace.
    Neither can change how the DeepSeek V4.1 pre-tokenizer groups the text, so
    the chunks encode to the same ids as the whole string.
    """
    n = len(text)
    if n <= _SEGMENT_MAX_CHARS:
        return [text]
    segments: list[str] = []
    start = 0
    while start < n:
        end = min(start + _SEGMENT_MAX_CHARS, n)
        if end < n:
            cut = _find_cut(text, start, end)
            if cut <= start:
                # No safe boundary anywhere in the rest: keep it whole.
                segments.append(text[start:])
                break
            end = cut
        segments.append(text[start:end])
        start = end
    return segments


def _segment_key(segment: str) -> bytes:
    return hashlib.blake2b(segment.encode("utf-8"), digest_size=16).digest()


class _SegmentTokenCache:
    """LRU cache of chunk token ids, bounded by total cached tokens."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: OrderedDict[bytes, list[int]] = OrderedDict()
        self._tokens = 0
        self._lock = threading.Lock()

    def get(self, key: bytes) -> list[int] | None:
        with self._lock:
            ids = self._items.get(key)
            if ids is not None:
                self._items.move_to_end(key)
            return ids

    def put(self, key: bytes, ids: list[int]) -> None:
        with self._lock:
            if key in self._items:
                return
            self._items[key] = ids
            self._tokens += len(ids)
            while self._tokens > self._capacity and self._items:
                _, evicted = self._items.popitem(last=False)
                self._tokens -= len(evicted)


def _reject_image_placeholder(text: Any) -> None:
    if isinstance(text, str) and IMAGE_PLACEHOLDER in text:
        raise ValueError(
            f"Message text contains image special token {IMAGE_PLACEHOLDER!r}; "
            "provide images as image content parts."
        )


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    # vLLM additions to the reference: Responses API part types, image_pil,
    # and ValueError (400) instead of the reference's asserts.
    result = [dict(message) for message in copy.deepcopy(messages)]
    image_idx = 0
    for message in result:
        image_indices = message["_image_indices"] = []
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _reject_image_placeholder(message.get("reasoning_content"))
        content = message.get("content")
        if isinstance(content, str):
            _reject_image_placeholder(content)
        elif isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    _reject_image_placeholder(block.get("text"))
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                    image_indices.append(image_idx)
                    image_idx += 1
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
    return result


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            image_order: list[int] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages)
            if tools:
                system = next((m for m in conversation if m["role"] == "system"), None)
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = tools

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                thinking = True
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be one of "
                    f"{sorted(REASONING_EFFORT_MAPPINGS)} or an integer within "
                    "[1, 100] in chat_template_kwargs"
                )

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                image_order=image_order,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                return self._dsv41_encode(prompt, tokenizer_kwargs)
            return prompt

        def _segment_cache(self) -> "_SegmentTokenCache":
            cache = getattr(self, "_vllm_dsv41_segment_cache", None)
            if cache is None:
                cache = _SegmentTokenCache(int(envs.VLLM_DSV41_ENCODE_CACHE_TOKENS))
                self._vllm_dsv41_segment_cache = cache
            return cache

        def _dsv41_encode(
            self, prompt: str, tokenizer_kwargs: dict[str, Any]
        ) -> list[int]:
            """Encode a long prompt in parallel chunks, reusing cached chunks.

            Long agent prompts repeat their prefix every turn; encoding each chunk
            once and caching it by content makes the first turn pay for the whole
            prompt while later turns only encode the new tail.
            """
            timing = envs.VLLM_DSV41_FRONTEND_TIMING
            started = time.monotonic() if timing else 0.0

            def _log_whole_string() -> None:
                logger.info(
                    "dsv41 whole-string encode: chars=%d segments=1 "
                    "cache_miss=0 took_ms=%.1f",
                    len(prompt),
                    (time.monotonic() - started) * 1000.0,
                )

            if (
                not envs.VLLM_DSV41_SEGMENTED_ENCODE
                or tokenizer_kwargs
                or len(prompt) < _MIN_SEGMENTED_CHARS
            ):
                ids = self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
                if timing:
                    _log_whole_string()
                return ids
            segments = _split_segments(prompt)
            if len(segments) <= 1:
                ids = self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
                if timing:
                    _log_whole_string()
                return ids

            cache = self._segment_cache()
            keys = [_segment_key(segment) for segment in segments]
            # Keep this call's ids locally: a concurrent encode may evict a
            # segment from the LRU between the lookup and the final assembly.
            encoded: list[list[int] | None] = [None] * len(segments)
            missing: list[tuple[int, str]] = []
            for idx, (segment, key) in enumerate(zip(segments, keys)):
                cached = cache.get(key)
                if cached is not None:
                    encoded[idx] = cached
                else:
                    missing.append((idx, segment))
            if missing:
                backend = getattr(self, "backend_tokenizer", None)
                if backend is not None and hasattr(backend, "encode_batch"):
                    encodings = backend.encode_batch(
                        [segment for _, segment in missing],
                        add_special_tokens=False,
                    )
                    encoded_missing = [list(encoding.ids) for encoding in encodings]
                else:
                    encoded_missing = [
                        list(self.encode(segment, add_special_tokens=False))
                        for _, segment in missing
                    ]
                for (idx, _), chunk_ids in zip(missing, encoded_missing):
                    encoded[idx] = chunk_ids
                    cache.put(keys[idx], chunk_ids)

            ids: list[int] = []
            for chunk in encoded:
                assert chunk is not None
                ids.extend(chunk)
            if timing:
                logger.info(
                    "dsv41 segmented encode: chars=%d segments=%d longest=%d "
                    "cache_miss=%d took_ms=%.1f",
                    len(prompt),
                    len(segments),
                    max(len(segment) for segment in segments),
                    len(missing),
                    (time.monotonic() - started) * 1000.0,
                )
            return ids

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
