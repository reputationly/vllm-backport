#!/usr/bin/env bash
# =============================================================================
# Hy4-flash on 2x4 A100-40G (sm80), overlay deployment, official FP8 weights.
#
#   NODE_RANK=0 HOST_IP=29.165.44.30 bash run_tp8.sh    # head node
#   NODE_RANK=1 HOST_IP=29.165.19.7  bash run_tp8.sh    # worker node
#
# Launch pattern copied from the production DS-V4.1 script
# (/root/v41_launch5.sh): `mp` backend with `--nnodes/--node-rank` and
# `--headless` on every non-zero node. Using torchrun/external_launcher instead
# makes every rank start its own API server and the TP8 group then crawls at
# ~0.1 layer/s; this pattern serves normally.
#
# Measured on 2x4 A100-40G (100G RoCE, mlx5_0, GID 3):
#   TTFT 0.49 s, TPOT 456 ms/token, 8 tok/s aggregate at c=4, KV 302k tokens,
#   vision 4/4 on synthetic images, coherent zh/math/code answers.
# =============================================================================
set -uo pipefail
NODE_RANK="${NODE_RANK:-}"
HOST_IP="${HOST_IP:-}"
if [ -z "$NODE_RANK" ] || [ -z "$HOST_IP" ]; then
  echo "usage: NODE_RANK=0|1 HOST_IP=<100G address of this node> bash $0" >&2
  exit 2
fi
IMG="${IMG:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport:arm64-sm80-20261030-1209-c14d2fbf}"

exec docker run --rm --name "hy4-r${NODE_RANK}" \
  --gpus all --ipc=host --network=host --pid=host --cap-add=SYS_PTRACE \
  -v /root/hy4flash:/work \
  -v /root/Models:/root/Models:ro \
  -v /root/hy4flash/pkg_branch:/usr/local/lib/python3.12/dist-packages/vllm \
  -e PYTHONPATH=/work/site/tf \
  -e NODE_RANK="$NODE_RANK" \
  -e HOST_IP="$HOST_IP" \
  -e MASTER="${MASTER:-29.165.44.30}" \
  -e MAX_LEN="${MAX_LEN:-32768}" \
  --entrypoint bash "$IMG" /work/serve_mp_node.sh
