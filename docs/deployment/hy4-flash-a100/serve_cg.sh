#!/usr/bin/env bash
# Multi-node Hy4-flash, same launch pattern as the production DS-V4.1 script
# (/root/v41_launch5.sh): mp backend + --nnodes/--node-rank, --headless on
# every non-zero node, long distributed timeouts, IB/RoCE on the 100G fabric.
set -uo pipefail
RANK="${1:?rank}"; HOST_IP="${2:?ip}"
MASTER="${MASTER:-29.165.44.30}"
MPORT="${MPORT:-40091}"
APORT="${APORT:-8000}"
NNODES="${NNODES:-2}"
TP="${TP:-8}"
SERVED="${SERVED:-hy4-flash}"

export VLLM_KV_CACHE_LAYOUT=BLNHC
export VLLM_HOST_IP="$HOST_IP"
export GLOO_SOCKET_IFNAME=enp194s0f0np0
export NCCL_SOCKET_IFNAME=enp194s0f0np0
export OMP_NUM_THREADS=8
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=22
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_CUMEM_ENABLE=0
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export VLLM_LOGGING_LEVEL=INFO
export TRITON_CACHE_DIR="/tmp/triton-cache-mp-${RANK}"
mkdir -p "$TRITON_CACHE_DIR"

CGJSON="{\"cudagraph_mode\": \"${CGMODE:-FULL_DECODE_ONLY}\", \"max_cudagraph_capture_size\": 32}"

COMMON=(
  --served-model-name "$SERVED"
  --tensor-parallel-size "${TP:-4}"
  --pipeline-parallel-size "${PP:-2}"
  --distributed-executor-backend mp
  --nnodes "$NNODES" --node-rank "$RANK"
  --master-addr "$MASTER" --master-port "$MPORT"
  --distributed-timeout-seconds 3600
  --cpu-distributed-timeout-seconds 3600
  --block-size "${BS:-128}"
  --kv-cache-dtype "${KVDTYPE:-bfloat16}"
  --engram-config '{"cpu_offload": true}'
  --max-model-len "${MAX_LEN:-4096}"
  --max-num-seqs "${MAXSEQS:-16}"
  --max-num-batched-tokens "${MAXBT:-2048}"
  --gpu-memory-utilization 0.90
  --compilation-config "$CGJSON"
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
  --disable-custom-all-reduce
  --reasoning-parser hy_v4
  --enable-auto-tool-choice
  --tool-call-parser hy_v4
  --chat-template /root/Models/Hy4-flash-FP8/chat_template.jinja
  --trust-remote-code
  --max-num-queued-tokens 1750000
  --max-num-queued-reqs 48
  --no-enable-flashinfer-autotune
  --prefix-match-unit 16
  --long-prefill-token-threshold 8192
  --enable-prefix-caching
  --disable-access-log-for-endpoints=/metrics
)

if [ "$RANK" = "0" ]; then
  exec vllm serve "${MODEL:-/root/Models/Hy4-flash-FP8}" --host 0.0.0.0 --port "$APORT" "${COMMON[@]}"
else
  exec vllm serve "${MODEL:-/root/Models/Hy4-flash-FP8}" --headless "${COMMON[@]}"
fi
