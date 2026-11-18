#!/usr/bin/env bash
# Hy4-flash baseline: TTFT / TPOT / throughput at c=1 and c=8, plus the KV and
# memory numbers the engine reports at startup (they go to the serve log).
#
#   BASE=http://127.0.0.1:8000 MODEL=hy4-flash bash bench.sh
#
# Runs `vllm bench serve` from inside the same container image, so no extra deps.
set -uo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-hy4-flash}"
IMG=/root/hy4flash/pkg
SITE=/root/hy4flash/site/tf

run() {
  local in_len=$1 out_len=$2 conc=$3
  echo "=== in=$in_len out=$out_len concurrency=$conc ==="
  docker run --rm --network=host --entrypoint bash \
    -v /root/hy4flash:/work -v "$IMG":/usr/local/lib/python3.12/dist-packages/vllm \
    -e PYTHONPATH="$SITE" <fork-image> -lc "
      vllm bench serve \
        --backend openai-chat --base-url $BASE --model $MODEL \
        --dataset-name random --random-input-len $in_len --random-output-len $out_len \
        --num-prompts $((conc * 4)) --max-concurrency $conc \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
    " 2>&1 | grep -E "Request throughput|Output token throughput|Mean TTFT|P95 TTFT|Mean TPOT|P95 TPOT|Peak concurrent"
}

run 2048 256 1
run 2048 256 8
run 32768 128 1
