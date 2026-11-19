"""Long-context cold TTFT for an arbitrary endpoint (with optional key)."""
import json, sys, time, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
TARGETS = [int(x) for x in sys.argv[4].split(",")]
OUT = int(sys.argv[5]) if len(sys.argv) > 5 else 64
PARA = ("人工智能推理系统的性能取决于预填充与解码两个阶段的效率。预填充阶段一次处理整段提示，"
        "其耗时随上下文长度线性增长；解码阶段逐 token 生成，受显存带宽与并行策略影响。"
        "在实际部署中，我们通常需要同时权衡首字延迟、每 token 时延与整体吞吐。")
for target in TARGETS:
    reps = max(1, int(target * 1.55 / len(PARA)))
    prompt = PARA * reps
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": OUT, "temperature": 0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE.rstrip("/") + "/completions",
                                 data=json.dumps(payload).encode(), headers=h)
    t0 = time.time(); first = None; last = None; usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
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
                if (o.get("choices") or [{}])[0].get("text"):
                    if first is None:
                        first = time.time()
                    last = time.time()
        pt = (usage or {}).get("prompt_tokens", -1)
        ct = (usage or {}).get("completion_tokens", 0)
        ttft = (first - t0) * 1000
        tpot = ((last - first) * 1000 / (ct - 1)) if ct > 1 and last and first else float("nan")
        print(f"{MODEL}: prompt={pt:>7} | TTFT {ttft/1000:7.2f} s | prefill {pt/(ttft/1000):7.0f} tok/s | TPOT {tpot:6.1f} ms", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{MODEL}: target {target} FAILED {type(exc).__name__}: {exc}", flush=True)
