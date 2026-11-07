# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers.deepseek_v41 import (
    _MARK_CLOSE,
    _MARK_OPEN,
    _normalize_messages,
    get_deepseek_v41_tokenizer,
)
from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


def tool_call(name):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_reference_encoder_fixtures(case_id):
    # Copied from DeepSeek-V4.1-Flash-1017 encoding/tests.
    data = json.loads(
        (FIXTURES / "deepseek_v41" / f"test_input_{case_id}.json").read_text()
    )
    case = data if isinstance(data, dict) else {"messages": data}
    messages = case["messages"]
    thinking_mode = case.get("thinking_mode") or "chat"
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()

    if any(m["role"] == "latest_reminder" for m in messages):
        # latest_reminder is not an API role.
        assert case.get("tools") is None
        prompt = encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=case.get("reasoning_effort"),
        )
    else:
        prompt = render(
            messages,
            tools=case.get("tools"),
            thinking=thinking_mode == "thinking",
            reasoning_effort=case.get("reasoning_effort"),
        )
    assert prompt == expected


def test_drop_thinking_keeps_current_turn_across_tool_results():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = render(messages)
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")

    next_turn = render(messages + [{"role": "user", "content": "q3"}])
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")


def marked(spelling):
    return _MARK_OPEN + spelling + _MARK_CLOSE


def assert_only_marked(prompt, spelling, count=1):
    assert prompt.count(marked(spelling)) == count
    assert spelling not in prompt.replace(marked(spelling), "")


@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("as_part", [False, True])
def test_escapes_image_placeholder_in_text(role, as_part):
    text = f"hi {IMAGE_PLACEHOLDER}"
    content = [{"type": "text", "text": text}] if as_part else text
    prompt = render([{"role": role, "tool_call_id": "c1", "content": content}])
    assert_only_marked(prompt, IMAGE_PLACEHOLDER)


def test_escapes_image_pad_spelling_in_text():
    pad = "<|place_holder_mm_span_0436|>"
    prompt = render([{"role": "user", "content": f"hi {pad}"}])
    assert_only_marked(prompt, pad)


def test_escapes_image_placeholder_in_reasoning():
    prompt = render(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": IMAGE_PLACEHOLDER},
        ],
        drop_thinking=False,
    )
    assert_only_marked(prompt, IMAGE_PLACEHOLDER)


def test_escapes_image_placeholder_in_tool_arguments():
    call = tool_call("c1")
    # Default ensure_ascii encodes the spelling as \uXXXX escapes; escaping
    # must decode the JSON rather than replace on the raw string.
    call["function"]["arguments"] = json.dumps({"note": IMAGE_PLACEHOLDER})
    prompt = render(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "v"},
        ]
    )
    assert_only_marked(prompt, IMAGE_PLACEHOLDER)


def test_literal_image_placeholder_encodes_as_plain_text():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    messages = [{"role": "user", "content": f"hi {IMAGE_PLACEHOLDER}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    before, rest = prompt.split(_MARK_OPEN)
    spelling, after = rest.split(_MARK_CLOSE)
    assert spelling == IMAGE_PLACEHOLDER

    # FakeHfTokenizer encodes each string to [len(text)], so the marked
    # spelling splices as one id per plain character.
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    assert ids == [len(before)] + [1] * len(IMAGE_PLACEHOLDER) + [len(after)]


def test_escapes_image_placeholder_in_double_encoded_tool_arguments():
    call = tool_call("c1")
    # Double-encoded arguments are tolerated by encode_arguments_to_dsml;
    # escaping must decode as many layers as dsml does.
    call["function"]["arguments"] = json.dumps(json.dumps({"note": IMAGE_PLACEHOLDER}))
    prompt = render(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "v"},
        ]
    )
    assert_only_marked(prompt, IMAGE_PLACEHOLDER)


def test_tool_arguments_without_sentinels_pass_through_untouched():
    call = tool_call("c1")
    call["function"]["arguments"] = "[1,2,3]"
    normalized = _normalize_messages(
        [{"role": "assistant", "content": "a", "tool_calls": [call]}]
    )
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == "[1,2,3]"


PAD_SPELLING = "<|place_holder_mm_span_0436|>"


def _tiny_backend():
    from tokenizers import AddedToken, Tokenizer, models, trainers

    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=[])
    corpus = [
        "abcdefghijklmnopqrstuvwxyz 0123456789",
        "deepseek_image place_holder_mm_span_0436",
        "user assistant tool result hi there ",
        # Every char of the protected spellings needs a vocab entry so the
        # plain-text encode reproduces the spelling exactly.
        "<>|｜",
    ]
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.add_tokens([AddedToken(IMAGE_PLACEHOLDER, special=True)])
    tokenizer.add_tokens([AddedToken(PAD_SPELLING, special=True)])
    return tokenizer


class _BackendHfTokenizer:
    """HF-shaped stub backed by a real ``tokenizers.Tokenizer``."""

    def __init__(self, backend):
        self.backend_tokenizer = backend

    def get_added_vocab(self):
        full = self.backend_tokenizer.get_vocab(with_added_tokens=True)
        base = self.backend_tokenizer.get_vocab(with_added_tokens=False)
        return {k: v for k, v in full.items() if k not in base}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return list(
            self.backend_tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            ).ids
        )

    def __call__(self, text, *args, return_offsets_mapping=False, **kwargs):
        ids = self.encode(text, **kwargs)
        encoding = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if return_offsets_mapping:
            encoding["offset_mapping"] = [(0, len(text))] * len(ids)
        return encoding

    def decode(self, ids, **kwargs):
        return self.backend_tokenizer.decode(list(ids))


def _backend_tokenizer():
    return get_deepseek_v41_tokenizer(_BackendHfTokenizer(_tiny_backend()))


def test_call_splices_marked_spellings_through_backend():
    tokenizer = _backend_tokenizer()
    image_id = tokenizer.get_added_vocab()[IMAGE_PLACEHOLDER]
    pad_id = tokenizer.get_added_vocab()[PAD_SPELLING]

    messages = [
        {"role": "user", "content": f"hi {IMAGE_PLACEHOLDER} and {PAD_SPELLING}"},
        {"role": "tool", "tool_call_id": "c1", "content": f"out {IMAGE_PLACEHOLDER}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(prompt, add_special_tokens=False)
    ids = encoding["input_ids"]

    assert image_id not in ids and pad_id not in ids
    assert encoding["attention_mask"] == [1] * len(ids)
    # The tiny BPE has no decoder; join loosely instead of asserting
    # whitespace fidelity (verified verbatim on the real tokenizer).
    decoded = tokenizer.decode(ids).replace(" ", "")
    assert IMAGE_PLACEHOLDER in decoded and PAD_SPELLING in decoded

    direct = tokenizer.apply_chat_template(messages, tokenize=True)
    assert ids == direct


def test_call_rebases_offsets_and_honours_truncation():
    tokenizer = _backend_tokenizer()
    messages = [{"role": "user", "content": f"hi {IMAGE_PLACEHOLDER} tail"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    encoding = tokenizer(prompt, return_offsets_mapping=True)
    before = prompt.split(_MARK_OPEN)[0]
    mark_span = (len(before), len(before) + len(marked(IMAGE_PLACEHOLDER)))
    assert mark_span in encoding["offset_mapping"]

    truncated = tokenizer(prompt, truncation=True, max_length=5)
    assert truncated["input_ids"] == encoding["input_ids"][:5]
    assert truncated["attention_mask"] == [1] * 5


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_numeric_reasoning_effort(effort, budget):
    assert render(
        [{"role": "user", "content": "question"}], reasoning_effort=effort
    ) == (
        "<｜begin▁of▁sentence｜><｜System｜>"
        f"Reasoning Effort: {budget} (range 1-100, the higher the value, "
        "the more thorough the reasoning)\n\n"
        "<｜User｜>question<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", ["unknown", -1, 0, 101, True, 1.5, []])
def test_invalid_effort_is_a_request_error(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "question"}], reasoning_effort=effort)


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": False},
        {"enable_thinking": False},
        {"thinking": True, "reasoning_effort": "none"},
        {"enable_thinking": True, "reasoning_effort": "none"},
    ],
)
def test_chat_mode_has_closed_thinking_prefix(controls):
    assert render([{"role": "user", "content": "question"}], **controls) == (
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    )


def test_raw_text_parts_preserve_reference_separator():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    assert (
        render(
            messages,
            conversation=[{"role": "user", "content": "first\nsecond"}],
            thinking=False,
        )
        == "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond<｜Assistant｜></think>"
    )
    assert messages == original


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    # The Responses API carries text in input_text/output_text parts, which are
    # equivalent to the chat completions text part.
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    assert render([responses_message], thinking=False) == render(
        [chat_message], thinking=False
    )


def test_mid_system_gets_its_own_marker_and_generation_header():
    assert render(
        [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "answer briefly"},
        ],
        thinking=False,
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>question"
        "<｜System｜>answer briefly<｜Assistant｜></think>"
    )


def test_top_level_effort_overrides_template_effort():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "question"}],
        reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": 100},
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert "Reasoning Effort: 50 " in render(request.messages, **kwargs)


def test_encode_uses_one_bos_and_forwards_truncation():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        thinking=False,
        truncation=True,
        max_length=16,
    )
    text, add_special_tokens, kwargs = tokenizer.last_encode
    assert text.startswith("<｜begin▁of▁sentence｜>")
    assert add_special_tokens is False
    assert kwargs == {"truncation": True, "max_length": 16}


@pytest.mark.parametrize("image_type", ["image_url", "input_image", "image_pil"])
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
        url = f"https://example.com/{color}.png"
        return {
            "type": image_type,
            "image_url": {"url": url} if image_type == "image_url" else url,
        }

    parts = [
        {"type": "text", "text": "first"},
        image_part("red"),
        {"type": "text", "text": "second"},
        image_part("blue"),
        {"type": "text", "text": "compare"},
    ]
    assert render([{"role": "user", "content": parts}], thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\n"
        "second\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>"
    )
    assert parts[1]["type"] == parts[3]["type"] == image_type


def test_unsupported_media_is_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])
