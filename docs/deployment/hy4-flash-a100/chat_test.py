"""Verify chat API: reasoning split, plain answer, and vision through chat."""
import base64, io, json, urllib.request
BASE = "http://127.0.0.1:8000"
def post(payload, timeout=1200):
    r = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)

def show(tag, payload):
    out = post(payload)
    m = out["choices"][0]["message"]
    rc = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"[{tag}] content={ (m.get('content') or '')[:110]!r}")
    print(f"        reasoning_content={rc[:110]!r} (len={len(rc)})")
    return out

show("think=high", {"model": "hy4-flash",
     "messages": [{"role": "user", "content": "9.11 和 9.9 哪个大?只回答一个数字。"}],
     "max_tokens": 256, "temperature": 0.9,
     "chat_template_kwargs": {"reasoning_effort": "high"}})
show("no_think", {"model": "hy4-flash",
     "messages": [{"role": "user", "content": "巴黎是哪个国家的首都?一个词。"}],
     "max_tokens": 32, "temperature": 0.6,
     "chat_template_kwargs": {"reasoning_effort": "no_think"}})

# vision through the chat endpoint
from PIL import Image, ImageDraw
im = Image.new("RGB", (336, 336), "white")
d = ImageDraw.Draw(im)
d.text((40, 120), "HELLO 7", fill="black")
buf = io.BytesIO(); im.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
show("vision", {"model": "hy4-flash", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "图里的文字是什么?只回文字。"}]}],
     "max_tokens": 64, "temperature": 0,
     "chat_template_kwargs": {"reasoning_effort": "no_think"}})
