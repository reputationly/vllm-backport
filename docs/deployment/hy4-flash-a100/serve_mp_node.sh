#!/usr/bin/env bash
# Multi-node Hy4-flash, same launch pattern as the production DS-V4.1 script
# (/root/v41_launch5.sh): mp backend + --nnodes/--node-rank, --headless on
# every non-zero node, long distributed timeouts, IB/RoCE on the 100G fabric.
set -uo pipefail
RANK="${NODE_RANK:-${1:-}}"
HOST_IP="${HOST_IP:-}"
if [ -z "$RANK" ] || [ -z "$HOST_IP" ]; then
  echo "usage: NODE_RANK=0|1 HOST_IP=<100G address> bash serve_mp_node.sh" >&2
  exit 2
fi
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
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export VLLM_LOGGING_LEVEL=INFO
export TRITON_CACHE_DIR="/tmp/triton-cache-mp-${RANK}"
mkdir -p "$TRITON_CACHE_DIR"

COMMON=(
  --served-model-name "$SERVED"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "${PP:-1}"
  --distributed-executor-backend mp
  --nnodes "$NNODES" --node-rank "$RANK"
  --master-addr "$MASTER" --master-port "$MPORT"
  --distributed-timeout-seconds 3600
  --cpu-distributed-timeout-seconds 3600
  --block-size 64
  --kv-cache-dtype bfloat16
  --enable-expert-parallel
  --engram-config '{"cpu_offload": true}'
  --max-model-len "${MAX_LEN:-4096}"
  --max-num-seqs "${MAXSEQS:-4}"
  --max-num-batched-tokens "${MAXBT:-2048}"
  --gpu-memory-utilization 0.90
  --enforce-eager
  --limit-mm-per-prompt '{"image": 1, "video": 0}'
  --disable-custom-all-reduce
  --trust-remote-code
)

if [ "$RANK" = "0" ]; then
  exec vllm serve "${MODEL:-/root/Models/Hy4-flash-FP8}" --host 0.0.0.0 --port "$APORT" "${COMMON[@]}"
else
  exec vllm serve "${MODEL:-/root/Models/Hy4-flash-FP8}" --headless "${COMMON[@]}"
fi
