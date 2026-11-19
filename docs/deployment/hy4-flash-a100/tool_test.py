"""Verify the HYV4 tool parser: non-streaming and streaming tool calls."""
import json, urllib.request
BASE = "http://127.0.0.1:8000"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city"]}}}]
MSG = [{"role": "user", "content": "北京现在天气怎么样？用工具查一下。"}]

def post(payload):
    r = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1200) as resp:
        return json.load(resp)

out = post({"model": "hy4-flash", "messages": MSG, "tools": TOOLS,
            "tool_choice": "auto", "max_tokens": 256, "temperature": 0.6,
            "chat_template_kwargs": {"reasoning_effort": "no_think"}})
ch = out["choices"][0]
msg = ch["message"]
print("finish_reason:", ch.get("finish_reason"))
print("tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False)[:400])
print("content:", (msg.get("content") or "")[:120])

# streaming
payload = {"model": "hy4-flash", "messages": MSG, "tools": TOOLS, "tool_choice": "auto",
           "max_tokens": 256, "temperature": 0.6, "stream": True,
           "chat_template_kwargs": {"reasoning_effort": "no_think"}}
r = urllib.request.Request(BASE + "/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
names, args, chunks = [], [], 0
with urllib.request.urlopen(r, timeout=1200) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        obj = json.loads(line[5:].strip())
        for c in obj.get("choices", []):
            d = c.get("delta") or {}
            for tc in (d.get("tool_calls") or []):
                chunks += 1
                fn = tc.get("function") or {}
                if fn.get("name"):
                    names.append(fn["name"])
                if fn.get("arguments"):
                    args.append(fn["arguments"])
print("stream tool deltas:", chunks, "| names:", names, "| args:", "".join(args)[:200])
