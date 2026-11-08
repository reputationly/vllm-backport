# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import hashlib
import json
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


# A literal sentinel spelling in message text is wrapped in these marks. The
# reference encoding treats such spellings as plain text; only spellings the
# renderer inserts itself (e.g. the placeholder of an image content part)
# may become special tokens. The mark travels with the text through
# rendering; at tokenization time the wrapped spelling is spliced as the
# ids it produces without the added vocabulary.
_MARK_OPEN = "\ue000"
_MARK_CLOSE = "\ue001"

# Spellings protected even without a tokenizer lookup: the image placeholder
# itself and the reserved compressor-alignment pad it splices with.
_LITERAL_SPELLINGS = (
    IMAGE_PLACEHOLDER,
    "<|place_holder_mm_span_0436|>",
)


def _literal_pattern(spellings: tuple[str, ...]) -> "re.Pattern[str]":
    return re.compile(
        "|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True))
    )


def _escape_literal_spellings(text: Any, spellings: tuple[str, ...]) -> Any:
    if not isinstance(text, str) or not spellings:
        return text
    return _literal_pattern(spellings).sub(
        lambda match: _MARK_OPEN + match.group(0) + _MARK_CLOSE, text
    )


def _escape_json_value(value: Any, spellings: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _escape_literal_spellings(value, spellings)
    if isinstance(value, list):
        return [_escape_json_value(item, spellings) for item in value]
    if isinstance(value, dict):
        return {key: _escape_json_value(item, spellings) for key, item in value.items()}
    return value


def _escape_tool_arguments(arguments: Any, spellings: tuple[str, ...]) -> Any:
    """Escape literal sentinel spellings inside tool-call arguments."""
    if not isinstance(arguments, str):
        return _escape_json_value(arguments, spellings)
    decoded: Any = arguments
    rounds = 0
    # encode_arguments_to_dsml tolerates up to two layers of JSON encoding.
    while rounds < 2 and isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError:
            break
        rounds += 1
    if rounds == 0:
        return _escape_literal_spellings(arguments, spellings)
    escaped = _escape_json_value(decoded, spellings)
    if escaped == decoded:
        return arguments
    for _ in range(rounds):
        escaped = json.dumps(escaped, ensure_ascii=False)
    return escaped


def _marked_parts(text: str):
    """Yield ``(plain_text, None)`` or ``(None, spelling)`` segments.

    A marked spelling sits between ``_MARK_OPEN`` and the next
    ``_MARK_CLOSE``; a stray ``_MARK_OPEN`` is yielded back as plain text.
    """
    chunks = text.split(_MARK_OPEN)
    yield chunks[0], None
    for chunk in chunks[1:]:
        spelling, sep, rest = chunk.partition(_MARK_CLOSE)
        if sep:
            yield None, spelling
            yield rest, None
        else:
            yield _MARK_OPEN + chunk, None


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
    literal_spellings: tuple[str, ...] = _LITERAL_SPELLINGS,
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
        reasoning_content = _escape_literal_spellings(
            message.get("reasoning_content"), literal_spellings
        )
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if function is not None and function.get("arguments") is not None:
                function["arguments"] = _escape_tool_arguments(
                    function["arguments"], literal_spellings
                )
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _escape_literal_spellings(content, literal_spellings)
        elif isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(
                        _escape_literal_spellings(
                            block.get("text", ""), literal_spellings
                        )
                    )
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

    image_id = added_vocab.get(IMAGE_PLACEHOLDER)
    literal_spellings: tuple[str, ...] = _LITERAL_SPELLINGS
    if image_id is not None:
        # The vision path treats [image_id, image_id + 5) as image tokens,
        # so any added-token spelling in that range corrupts multimodal
        # routing when typed into message text.
        literal_spellings += tuple(
            spelling
            for spelling, token_id in added_vocab.items()
            if image_id < token_id < image_id + 5
        )

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
            conversation = _normalize_messages(messages, literal_spellings)
            if tools:
                system = next((m for m in conversation if m["role"] == "system"), None)
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = _escape_json_value(tools, literal_spellings)

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

        def _plain_tokenizer(self):
            """Backend tokenizer without the added vocabulary, or None."""
            plain = getattr(self, "_vllm_dsv41_plain_tokenizer", None)
            if plain is None:
                backend = getattr(self, "backend_tokenizer", None)
                if backend is None:
                    plain = False
                else:
                    from tokenizers import Tokenizer

                    spec = json.loads(backend.to_str())
                    spec["added_tokens"] = []
                    plain = Tokenizer.from_str(json.dumps(spec))
                self._vllm_dsv41_plain_tokenizer = plain
            return plain or None

        def _literal_token_ids(self, spelling: str) -> list[int]:
            """Token ids for a marked spelling encoded as plain text.

            The reference encoding renders spellings typed into message
            text through a tokenizer without the added vocabulary, so they
            cannot collide with real special tokens.
            """
            cache = getattr(self, "_vllm_dsv41_literal_ids", None)
            if cache is None:
                cache = self._vllm_dsv41_literal_ids = {}
            ids = cache.get(spelling)
            if ids is None:
                plain = self._plain_tokenizer()
                if plain is not None:
                    ids = list(plain.encode(spelling, add_special_tokens=False).ids)
                else:
                    # Per-character encode is a best-effort fallback: the
                    # added token only matches the full spelling.
                    ids = [
                        token_id
                        for char in spelling
                        for token_id in self.encode(char, add_special_tokens=False)
                    ]
                cache[spelling] = ids
            return ids

        def encode(self, text, *args, **kwargs):
            if not (isinstance(text, str) and _MARK_OPEN in text):
                return super().encode(text, *args, **kwargs)
            ids: list[int] = []
            for part, spelling in _marked_parts(text):
                if spelling is not None:
                    ids.extend(self._literal_token_ids(spelling))
                elif part:
                    ids.extend(super().encode(part, *args, **kwargs))
            return ids

        def __call__(self, text, *args, **kwargs):
            if not (isinstance(text, str) and _MARK_OPEN in text):
                return super().__call__(text, *args, **kwargs)

            # Per-chunk truncation would slice every chunk; it is stripped
            # here and re-applied to the merged sequence below.
            chunk_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("truncation", "max_length")
            }
            merged: dict[str, list[Any]] = {}
            char_base = 0
            for part, spelling in _marked_parts(text):
                if spelling is not None:
                    plain_ids = self._literal_token_ids(spelling)
                    mark_len = len(_MARK_OPEN) + len(spelling) + len(_MARK_CLOSE)
                    for key, target in merged.items():
                        if key == "input_ids":
                            filler = plain_ids
                        elif key == "offset_mapping":
                            filler = [(char_base, char_base + mark_len)] * len(
                                plain_ids
                            )
                        elif key == "attention_mask":
                            filler = [1] * len(plain_ids)
                        else:
                            filler = [0] * len(plain_ids)
                        target.extend(filler)
                    char_base += mark_len
                    continue
                encoding = super().__call__(part, *args, **chunk_kwargs)
                for key, value in encoding.items():
                    if not isinstance(value, list):
                        continue
                    if key == "offset_mapping":
                        merged.setdefault(key, []).extend(
                            (start + char_base, end + char_base) for start, end in value
                        )
                    else:
                        merged.setdefault(key, []).extend(value)
                char_base += len(part)

            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length:
                left = getattr(self, "truncation_side", "right") == "left"
                for key, value in merged.items():
                    if len(value) > max_length:
                        merged[key] = (
                            value[-max_length:] if left else value[:max_length]
                        )

            result = encoding.__class__(merged)
            if return_tensors := kwargs.get("return_tensors"):
                result = result.convert_to_tensors(return_tensors)
            return result

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
            if _MARK_OPEN in prompt:
                ids = []
                for part, spelling in _marked_parts(prompt):
                    if spelling is not None:
                        ids.extend(self._literal_token_ids(spelling))
                    elif part:
                        ids.extend(self._dsv41_encode(part, tokenizer_kwargs))
                return ids

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
