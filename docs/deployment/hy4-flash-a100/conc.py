"""Aggregate throughput at several concurrency levels."""
import json, sys, time, urllib.request, concurrent.futures
BASE, MODEL = "http://127.0.0.1:8000/v1", "hy4-flash"
def run(prompt, mt=96):
    p = {"model": MODEL, "prompt": prompt, "max_tokens": mt, "temperature": 0,
         "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(BASE + "/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            b = line[5:].strip()
            if b == "[DONE]":
                break
            o = json.loads(b)
            if o.get("usage"):
                usage = o["usage"]
            if first is None and (o.get("choices") or [{}])[0].get("text"):
                first = time.time()
    return (usage or {}).get("completion_tokens", 0), (first - t0) * 1000, time.time() - t0
prompt = "Summarize why MoE inference activates only a subset of parameters. " * 40
for c in (1, 8, 16):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda _: run(prompt), range(c * 2)))
    wall = time.time() - t0
    toks = sum(r[0] for r in res)
    ttfts = [r[1] for r in res]
    print(f"c={c:>2}: {toks} tokens in {wall:.2f}s -> {toks/wall:6.1f} tok/s aggregate | "
          f"TTFT p50 {sorted(ttfts)[len(ttfts)//2]:7.0f} ms max {max(ttfts):7.0f} ms", flush=True)
