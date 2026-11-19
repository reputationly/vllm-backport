"""Per-request floor for an arbitrary endpoint (1-token prompt, cold + hit)."""
import json, sys, time, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
def once(prompt):
    p = {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0,
         "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE.rstrip("/") + "/completions",
                                 data=json.dumps(p).encode(), headers=h)
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            b = line[5:].strip()
            if b == "[DONE]":
                break
            o = json.loads(b)
            if o.get("usage"):
                usage = o["usage"]
            ch = (o.get("choices") or [{}])[0]
            if first is None and (ch.get("text") is not None or ch.get("delta")):
                first = time.time()
    if first is None:
        return float("nan"), (usage or {}).get("prompt_tokens", -1)
    return (first - t0) * 1000, (usage or {}).get("prompt_tokens", -1)
for label, prompt in (("1 token", "hi"), ("16 tokens", "请用一句话解释什么是混合专家模型。" * 2),
                      ("520 tokens", "人工智能推理系统的性能取决于预填充与解码两个阶段的效率。" * 40)):
    a, pt = once(prompt); b, _ = once(prompt); c, _ = once(prompt)
    print(f"{MODEL}: {label:11s} tokens={pt:>4} cold {a:7.1f} ms | hit {b:7.1f} | hit2 {c:7.1f}", flush=True)
