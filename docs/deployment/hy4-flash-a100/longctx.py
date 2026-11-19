"""Cold long-context TTFT/TPOT on natural-language prompts."""
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8000/v1"
PARA = ("人工智能推理系统的性能取决于预填充与解码两个阶段的效率。预填充阶段一次处理整段提示，"
        "其耗时随上下文长度线性增长；解码阶段逐 token 生成，受显存带宽与并行策略影响。"
        "在实际部署中，我们通常需要同时权衡首字延迟、每 token 时延与整体吞吐。")
CHARS_PER_TOK = 1.55  # CJK-heavy prose tokenises at roughly 1.5 chars/token

def measure(target_tokens, out=64):
    reps = max(1, int(target_tokens * CHARS_PER_TOK / len(PARA)))
    prompt = PARA * reps
    payload = {"model": "hy4-flash", "prompt": prompt, "max_tokens": out,
               "temperature": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            o = json.loads(body)
            if o.get("usage"):
                usage = o["usage"]
            if (o.get("choices") or [{}])[0].get("text"):
                if first is None:
                    first = time.time()
                last = time.time()
    end = time.time()
    pt = (usage or {}).get("prompt_tokens", -1)
    ct = (usage or {}).get("completion_tokens", 0)
    ttft = (first - t0) * 1000
    tpot = ((last - first) * 1000 / (ct - 1)) if ct > 1 and last and first else float("nan")
    print(f"prompt={pt:>7} tok | TTFT {ttft/1000:8.2f} s | prefill {pt/(ttft/1000):7.0f} tok/s | "
          f"TPOT {tpot:7.1f} ms | total {end-t0:7.2f} s", flush=True)

for target in (int(x) for x in sys.argv[1].split(",")):
    try:
        measure(target)
    except Exception as exc:  # noqa: BLE001
        print(f"target {target}: FAILED {type(exc).__name__}: {exc}", flush=True)
