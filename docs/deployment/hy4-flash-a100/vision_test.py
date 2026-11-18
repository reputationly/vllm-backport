#!/usr/bin/env python3
"""Hy4-flash vision accept test: four synthetic images with known answers.

Synthetic images make "really saw it" distinguishable from "guessed right" - the
expected answer is guaranteed by construction. Mirrors the DeepSeek-V4-Vision-Exp
bring-up gate (4/4) described in docs/deployment/vision-exp-bringup.md.

Usage: python3 vision_test.py [--base http://127.0.0.1:8000] [--model hy4-flash]
"""

import argparse
import base64
import io
import json
import sys
import urllib.request

from PIL import Image, ImageDraw


def _img_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def big_digit() -> tuple[Image.Image, str, str]:
    im = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(im).text((100, 110), "7", fill="black")
    return im, "图里只有一个巨大的阿拉伯数字,它是几?只回答数字。", "7"


def five_circles() -> tuple[Image.Image, str, str]:
    im = Image.new("RGB", (320, 320), "white")
    d = ImageDraw.Draw(im)
    for i in range(5):
        x = 30 + i * 55
        d.ellipse((x, 120, x + 40, 160), fill="red")
    return im, "图里有几个红色圆形?只回答数字。", "5"


def text_read() -> tuple[Image.Image, str, str]:
    im = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(im).text((40, 90), "VISION OK 42", fill="black")
    return im, "读出图中的文字,原样输出。", "VISION OK 42"


def color_order() -> tuple[Image.Image, str, str]:
    im = Image.new("RGB", (300, 120), "white")
    d = ImageDraw.Draw(im)
    for i, color in enumerate(("blue", "green", "red")):
        d.rectangle((20 + i * 90, 30, 70 + i * 90, 90), fill=color)
    return im, "从左到右,三个方块依次是什么颜色?按顺序回答。", "蓝绿红"


CASES = [big_digit, five_circles, text_read, color_order]


def ask(base: str, model: str, im: Image.Image, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + _img_to_b64(im)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.load(r)
    return body["choices"][0]["message"]["content"] or ""


def matches(answer: str, expect: str) -> bool:
    if expect.isdigit():
        return expect in answer
    if expect == "蓝绿红":
        return all(c in answer for c in expect)
    return expect.replace(" ", "") in answer.replace(" ", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="hy4-flash")
    args = ap.parse_args()

    passed = 0
    for make in CASES:
        im, prompt, expect = make()
        try:
            answer = ask(args.base, args.model, im, prompt)
            ok = matches(answer, expect)
        except Exception as exc:  # noqa: BLE001
            answer, ok = f"<request failed: {exc}>", False
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] expect={expect!r} answer={answer[:120]!r}")
    print(f"\n{passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
