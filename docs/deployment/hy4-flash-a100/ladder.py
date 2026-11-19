"""TTFT vs prompt length: fit TTFT = intercept + slope * tokens."""
import json, sys, time, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
SIZES = [int(x) for x in sys.argv[4].split(",")]
PARA = ("人工智能推理系统的性能取决于预填充与解码两个阶段的效率。预填充阶段一次处理整段提示，"
        "其耗时随上下文长度线性增长；解码阶段逐 token 生成，受显存带宽与并行策略影响。")
rows = []
for target in SIZES:
    reps = max(1, int(target * 1.55 / len(PARA)))
    prompt = PARA * reps
    p = {"model": MODEL, "prompt": prompt, "max_tokens": 2, "temperature": 0,
         "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE.rstrip("/") + "/completions",
                                 data=json.dumps(p).encode(), headers=h)
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
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
            if first is None and (o.get("choices") or [{}])[0].get("text"):
                first = time.time()
    pt = (usage or {}).get("prompt_tokens", target)
    ttft = (first - t0) * 1000
    rows.append((pt, ttft))
    print(f"  prompt={pt:>6} tok  TTFT {ttft:7.1f} ms", flush=True)

if len(rows) >= 2:
    n = len(rows)
    sx = sum(r[0] for r in rows); sy = sum(r[1] for r in rows)
    sxx = sum(r[0] ** 2 for r in rows); sxy = sum(r[0] * r[1] for r in rows)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    print(f"  => TTFT ≈ {intercept:.0f} ms + {slope*1000:.1f} us/token "
          f"(steady prefill {1e6/slope:.0f} tok/s)")
