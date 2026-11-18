#!/usr/bin/env bash
# =============================================================================
# Hy4-flash on 2x4 A100-40G (sm80), overlay deployment, official FP8 weights.
#
#   WEIGHTS=/root/Models/Hy4-flash-FP8  bash run_tp8.sh
#
# Prereqs (see docs/deployment/hy4-flash-a100.md):
#   * branch hy4-flash-a100 checked out, `vllm/` rsynced over the image package
#     at /root/hy4flash/pkg (no image rebuild);
#   * transformers >= 5.23 at /root/hy4flash/site/tf (PYTHONPATH shadow);
#   * weights verified: /root/Models/Hy4-flash-FP8/.DOWNLOAD_COMPLETE present.
#
# Two nodes, 4 cards each, TP=8. Cross-node all-reduce every layer, so pin NCCL
# to Ring/Simple and disable the custom all-reduce (the fork's Ampere guidance).
# =============================================================================
set -uo pipefail

NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on the head node and 1 on the worker}"
NNODES="${NNODES:-2}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the head node IP}"
MASTER_PORT="${MASTER_PORT:-29501}"
WEIGHTS="${WEIGHTS:-/root/Models/Hy4-flash-FP8}"
SERVED_NAME="${SERVED_NAME:-hy4-flash}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"          # start short, raise to 131072/1048576 once stable
MAX_SEQS="${MAX_SEQS:-4}"
IMG=/root/hy4flash/pkg
SITE=/root/hy4flash/site/tf

export VLLM_KV_CACHE_LAYOUT=BLNHC      # required: mixed page sizes need packed layout
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1               # dev cluster has plain TCP between nodes
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export PYTHONPATH="$SITE"

docker run --rm --gpus all --ipc=host --network=host \
  -v /root/hy4flash:/work \
  -v "$IMG":/usr/local/lib/python3.12/dist-packages/vllm \
  -e PYTHONPATH="$SITE" \
  -e VLLM_KV_CACHE_LAYOUT -e NCCL_ALGO -e NCCL_PROTO -e NCCL_IB_DISABLE \
  -e GLOO_SOCKET_IFNAME \
  <fork-image> \
  bash -lc '
    python3 -m vllm.entrypoints.openai.api_server \
      --model '"$WEIGHTS"' \
      --served-model-name '"$SERVED_NAME"' \
      --host 0.0.0.0 --port '"$PORT"' \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 1 \
      --distributed-executor-backend mp \
      --nnodes '"$NNODES"' --node-rank '"$NODE_RANK"' \
      --master-addr '"$MASTER_ADDR"' --master-port '"$MASTER_PORT"' \
      --block-size 64 \
      --kv-cache-dtype fp8 \
      --enable-expert-parallel \
      --engram-config "{\"cpu_offload\": true}" \
      --max-model-len '"$MAX_LEN"' \
      --max-num-seqs '"$MAX_SEQS"' \
      --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 \
      --enforce-eager \
      --limit-mm-per-prompt "{\"image\": 1, \"video\": 0}" \
      --disable-custom-all-reduce \
      --trust-remote-code
  '
