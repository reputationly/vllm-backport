#!/usr/bin/env bash
# 在 A100(sm80)集群上跨节点起 vllm-backport。
#
# 拓扑:TP=节点内卡数(走 PCIe),PP=节点数(走 RoCE)。
#   为什么不用跨节点 TP:这些机器节点内是 PCIe 且无 P2P,fork 自己的实测
#   (commit 3b890f43e,4x sm80 64GB PCIe no-P2P)是 TP4 prefill 比 PP4
#   慢约 6.6 倍。PP 每个 stage 边界只过 hidden states,4x100G RoCE 绰绰有余。
#
# 网络(本集群实测):
#   4 张 mlx5 口全 ACTIVE / link_layer=Ethernet / 100 Gb/sec,无 IB fabric。
#   RoCE v2 的 GID index = 3(该项是 IPv4 映射地址,例如 ::ffff:1da5:9ed0
#   即 29.165.158.208)。NCCL 的 IB 传输支持 RoCE,不需要真 IB。
#   带外引导走 10.0.0.x 管理网(NCCL_SOCKET_IFNAME)。
#
# 用法:
#   # 先在每个节点(head 与 worker)各起一次容器
#   ROLE=head   HEAD_IP=10.0.0.191 NODES=2 ./docker/run_multinode_sm80.sh
#   ROLE=worker HEAD_IP=10.0.0.191 NODES=2 ./docker/run_multinode_sm80.sh
#   # head 容器起来后在里面 serve(脚本会打印命令)
set -euo pipefail

ROLE="${ROLE:?需要 ROLE=head|worker}"
HEAD_IP="${HEAD_IP:?需要 HEAD_IP(head 节点的管理网 IP,如 10.0.0.191)}"
NODES="${NODES:-2}"
IMAGE="${IMAGE:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport:arm64-sm80-base}"
MODEL="${MODEL:-/models/DeepSeek-V4-Flash-0731}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-/nfs-models/DeepSeek-V4-Flash-0731}"
CONTAINER="${CONTAINER:-vllm-backport-${ROLE}}"
RAY_PORT="${RAY_PORT:-6379}"
SOCKET_IFNAME="${SOCKET_IFNAME:-enp131s0f0}"
IB_HCA="${IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"

# JIT / autotune 缓存挂到宿主机,跨容器重启复用。
#
# 为什么重要:本 fork 重度依赖 Triton kernel(mqa_logits_triton、sparse MLA、
# Marlin MoE),首次使用要 autotune。实测冷态到热态吞吐差 1.7~2 倍
# (180~210 -> 约 356 tok/s),TTFT P99 从 13~15 秒降到 0.7~1 秒。
# 缓存默认落在容器内的 /root/.cache 与 /root/.triton,容器一删就没了 ——
# 每次重建都要重新付一遍热身代价。挂出来之后:
#   - 生产:重启服务不再退回冷态
#   - 调参:换配置重启后不必重新热身,A/B 对比也更干净
# 默认放共享 NFS:一个节点热身完,其余节点直接复用,不必各自再付一遍。
# 代价与风险(所以留了覆盖入口):
#   - Triton 用原子 rename 落缓存,多节点并发写在 NFS 上基本安全但非保证;
#     若怀疑缓存损坏,删掉该目录重建即可。
#   - 大量小文件读在 NFS 上比本地盘慢,极端情况下可能反而拖慢启动。
#     要退回本地盘:CACHE_HOST_DIR=/var/cache/vllm-backport
CACHE_HOST_DIR="${CACHE_HOST_DIR:-/nfs-models/vllm-backport-cache}"
mkdir -p "${CACHE_HOST_DIR}" "${CACHE_HOST_DIR}-triton"

GPUS_PER_NODE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
TP="${TP:-${GPUS_PER_NODE}}"
PP="${PP:-${NODES}}"

echo "role=${ROLE} head=${HEAD_IP} nodes=${NODES} gpus/node=${GPUS_PER_NODE} -> TP=${TP} PP=${PP}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

# --network host:NCCL 要直接看到 RoCE 网卡和管理网;
# --device /dev/infiniband + memlock 无限:RDMA 注册内存的硬性要求;
# --ipc host + 大 shm:vLLM 的多进程共享内存。
docker run -d --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=32g \
  --device /dev/infiniband \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MODEL_HOST_DIR}":/models/"$(basename "${MODEL_HOST_DIR}")":ro \
  -v "${CACHE_HOST_DIR}":/root/.cache \
  -v "${CACHE_HOST_DIR}-triton":/root/.triton \
  -e NCCL_IB_HCA="${IB_HCA}" \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}" \
  -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" \
  -e NCCL_ALGO=Ring \
  -e NCCL_PROTO=Simple \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e VLLM_HOST_IP="$(ip -4 -o addr show "${SOCKET_IFNAME}" | awk '{print $4}' | cut -d/ -f1)" \
  --entrypoint sleep \
  "${IMAGE}" infinity

if [ "${ROLE}" = "head" ]; then
  docker exec "${CONTAINER}" ray start --head --port="${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
  cat <<EOF

=== head 容器已就绪 ===
等 worker 全部 join 后,先确认集群规模:
  docker exec ${CONTAINER} ray status | grep -E 'GPU|node_'

然后 serve(第一轮故意不开 DSpark —— dspark/utils.py:58 保留了上游把 draft
强制 pipeline_parallel_size=1 的写法,DSpark+PP 是已知缺口,先确认能出 token):

docker exec -d ${CONTAINER} vllm serve ${MODEL} \\
  --served-model-name deepseek-v4-flash \\
  --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size ${TP} \\
  --pipeline-parallel-size ${PP} \\
  --max-model-len 131072 \\
  --gpu-memory-utilization 0.85 \\
  --kv-cache-dtype fp8_ds_mla \\
  --trust-remote-code \\
  --disable-custom-all-reduce \\
  --enforce-eager

冒烟:
  curl -s localhost:8000/v1/models
  curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \\
    -d '{"model":"deepseek-v4-flash","prompt":"你好,请自我介绍","max_tokens":64,"temperature":0}'
EOF
else
  docker exec "${CONTAINER}" ray start --address="${HEAD_IP}:${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
  echo "=== worker 已 join ${HEAD_IP}:${RAY_PORT} ==="
fi
