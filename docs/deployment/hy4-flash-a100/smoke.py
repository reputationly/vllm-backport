#!/usr/bin/env python3
"""Hy4-flash accept test: text quality, instruction following, long-context recall.

Usage: python3 smoke.py [--base http://127.0.0.1:8000] [--model hy4-flash]
Exit code is non-zero when any case fails, so it can gate a rollout.
"""

import argparse
import json
import sys
import urllib.request

CASES = [
    {
        "name": "chinese",
        "prompt": "用一句话说明为什么 MoE 模型推理时只激活部分参数会更省算力。",
        "check": lambda t: any(k in t for k in ("专家", "路由", "激活")),
    },
    {
        "name": "math",
        "prompt": "一个数列前两项是 3 和 7,之后每项是前两项之和。第 8 项是多少?只回答数字。",
        "check": lambda t: "71" in t,
    },
    {
        "name": "code",
        "prompt": "写一个 Python 函数 add(a, b) 返回两数之和,只输出代码。",
        "check": lambda t: "def add" in t and "return" in t,
    },
    {
        "name": "long_context",
        "prompt": (
            "下面是一段日志,只有一行包含密码。请只回答密码本身。\n"
            + "\n".join(f"line {i}: status=ok" for i in range(2000))
            + "\nline 2001: password=ZEBRA-9134\n"
            + "\n".join(f"line {i}: status=ok" for i in range(2002, 4000))
        ),
        "check": lambda t: "ZEBRA-9134" in t,
    },
]


def chat(base: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="hy4-flash")
    args = ap.parse_args()

    failed = 0
    for case in CASES:
        try:
            out = chat(args.base, args.model, case["prompt"])
            ok = case["check"](out)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            out, ok = f"<request failed: {exc}>", False
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}: {out[:160]!r}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
