"""Single-stream sweep using usage.completion_tokens for an exact decode rate."""
import json, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
CTX = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["128", "512", "2048", "8192"])]
OUT = int(sys.argv[5]) if len(sys.argv) > 5 else 64

def run(prompt):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": OUT,
               "temperature": 0, "stream": True, "ignore_eos": True,
               "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE.rstrip("/") + "/completions",
                                 data=json.dumps(payload).encode(), headers=h)
    t0 = time.time(); first = None; last = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            if ch.get("text"):
                if first is None:
                    first = time.time()
                last = time.time()
    end = time.time()
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    ttft = (first - t0) * 1000
    dec = ((ct - 1) / (last - first)) if (ct and last and last > first) else float("nan")
    return pt, ttft, dec, ct

words = "the quick brown fox jumps over the lazy dog and then runs away".split()
mk = lambda n: (" ".join(words * (n // len(words) + 1)))[: n * 6]

print(f"== {MODEL}: single stream, out={OUT}, greedy ==")
print(f"{'ctx':>6} {'TTFT ms':>8} {'prefill tok/s':>14} {'decode tok/s':>13} {'TPOT ms':>8}")
for ctx in CTX:
    try:
        pt, ttft, dec, ct = run(mk(ctx))
        print(f"{pt:>6} {ttft:>8.0f} {pt/(ttft/1000):>14.0f} {dec:>13.2f} {1000/dec:>8.1f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{ctx:>6} FAILED {type(e).__name__}", flush=True)
