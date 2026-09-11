#!/usr/bin/env bash
# 在 ARM64 + A100 机器上全量编译 vllm-backport base 镜像,直推阿里云 ACR。
#
# 为什么不放 GitHub Actions:
#   本 fork 改了 csrc(custom_all_reduce.cuh / libtorch_stable/persistent_topk.cuh /
#   sampler.cu / topk.cu),VLLM_USE_PRECOMPILED 走不通,必须全量编 nvcc。
#   fork 自己的 docker-publish.yml 给这步留了 timeout-minutes: 2400,
#   GitHub 托管 runner 有 6h 作业上限和 ~14GB 磁盘,装不下。
#   所以 base 在自己的机器上低频构建(改 csrc / 依赖 / CUDA 版本时才跑),
#   然后 sync-base-to-dockerhub.yml 搬一份到 Docker Hub 供 app 流水线快速拉取。
#
# 注意:编译不需要 GPU(nvcc 是纯 CPU 工作),但需要 aarch64 Linux。
#   这台机器只要是 ARM64 + Docker 就行,有没有卡不影响出包;
#   有卡的好处是编完能立刻 `docker run --gpus all` 冒烟。
#
# 用法:
#   ACR_USERNAME=xxx ACR_PASSWORD=xxx ./docker/build_base_aarch64.sh
#   PUSH=0 ./docker/build_base_aarch64.sh                          # 只构建到本地,不推送
#   TORCH_CUDA_ARCH_LIST="8.0 8.6" ./docker/build_base_aarch64.sh   # 顺带带上 A6000
set -euo pipefail

# PUSH=0 时只把镜像留在本地 docker 镜像库,不需要任何 registry 凭据。
# 用于先验证 aarch64 编译本身(最贵最容易失败的一步),
# 之后 `docker login` 再 `docker push` 两个 tag 即可。
PUSH="${PUSH:-1}"

ACR_REGISTRY="${ACR_REGISTRY:-crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com}"
ACR_REPO="${ACR_REPO:-${ACR_REGISTRY}/reputationly/vllm-backport}"

# 只编 8.0 而不是官方镜像的 "8.0 8.6 8.9":单 arch 省掉大半编译时间,
# 而你们全是 A100。要兼容 A6000/4090 再加 8.6 / 8.9。
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
INSTALL_LMCACHE="${INSTALL_LMCACHE:-true}"

# ⚠ 必须显式换构建基座。docker/Dockerfile:44 的默认值
#   pytorch/manylinux2_28-builder:cuda13.0-78e737ad... 经 Docker Hub API 核实
#   是 **amd64 单架构**,在 arm64 上直接用会失败。
#   下面这个 pin 取自上游自己的 arm64 流水线
#   (.buildkite/image_build/image_build_arm64.sh:39、release-pipeline.yaml、
#    scripts/hardware_ci/run-gh200-test.sh:18),对应同一个 CUDA 13.0。
#   顺带一个好消息:上游 release-pipeline 的 CUDA_ARCH_AARCH64 是
#   "8.0 8.7 8.9 9.0 10.0 11.0 12.0" —— 含 8.0,即 aarch64 + sm80
#   本来就是上游在构建的组合,不是我们在开荒。
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042}"

# 隔离网(gpu4x / mgr 都不通 Docker Hub 和 pypi.org)需要把这三样换掉。
# 实测可达:mirrors.aliyun.com/pypi、download.pytorch.org、ACR(匿名可拉)。
# 用 mirror-build-base-to-acr.yml 把两个基座搬进 ACR 后:
#   BUILD_BASE_IMAGE=<ACR>/reputationly/manylinuxaarch64-builder:cuda13.0-b8b5f17a \
#   FINAL_BASE_IMAGE=<ACR>/reputationly/nvidia-cuda:13.0.3-base-ubuntu24.04 \
#   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ ./docker/build_base_aarch64.sh
# FINAL_BASE_IMAGE 留空则用 Dockerfile 的默认 nvidia/cuda:${CUDA_VERSION}-base-...
FINAL_BASE_IMAGE="${FINAL_BASE_IMAGE:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
# bootstrap.pypa.io 在隔离网内完全不通,而 Dockerfile 的 vllm-runtime-base
# 阶段会 `curl -sS ${GET_PIP_URL} | python3.12`。阿里云有一份:
#   GET_PIP_URL=https://mirrors.aliyun.com/pypi/get-pip.py   (实测 1.49MB/s)
GET_PIP_URL="${GET_PIP_URL:-}"
# ppa.launchpad.net 只有 8KB/s。把 DEADSNAKES_MIRROR_URL 设成非空而不设
# DEADSNAKES_GPGKEY_URL,Dockerfile 两个分支都不会走 —— 直接跳过
# add-apt-repository,python3.12 由 noble 的 main 仓提供(24.04 自带 3.12)。
DEADSNAKES_MIRROR_URL="${DEADSNAKES_MIRROR_URL:-}"

# 上游 arm64 CI 用 max_jobs=16 / nvcc_threads=4;fork 自己的 docker-publish.yml
# 用 nvcc_threads=1 配按内存算出的 max_jobs。默认跟 fork,内存富裕可调到 2~4。
NVCC_THREADS="${NVCC_THREADS:-1}"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: 需要在 aarch64 机器上跑(当前 $(uname -m))。" >&2
  echo "       x86 上用 --platform linux/arm64 会走 QEMU,编 CUDA 扩展会慢到不可用。" >&2
  exit 1
fi

# vllm 的 Dockerfile 通篇用 RUN --mount=type=cache/bind,必须 BuildKit;
# 而 Ubuntu 的 docker.io 包不带 buildx,没有它 DOCKER_BUILDKIT=1 也只会报
# "BuildKit is enabled but the buildx component is missing or broken"。
if ! docker buildx version >/dev/null 2>&1; then
  echo "ERROR: 缺少 docker buildx(vllm 的 Dockerfile 需要 BuildKit)。" >&2
  echo "       Ubuntu: apt-get install -y docker-buildx" >&2
  exit 1
fi

# 按内存定并发,和 fork 的 docker-publish.yml 一致:
# nvcc 单进程峰值约 4GB,超订会 OOM-kill 然后整个 build 白跑。
CORES="$(nproc)"
MEM_GB="$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))"
MAX_JOBS="${MAX_JOBS:-$(( MEM_GB / 4 ))}"
[ "${MAX_JOBS}" -gt "${CORES}" ] && MAX_JOBS="${CORES}"
[ "${MAX_JOBS}" -lt 2 ] && MAX_JOBS=2

SHORT_SHA="$(git rev-parse --short=8 HEAD)"
BUILD_TIME="$(date -u +%Y%m%d-%H%M)"
ARCH_SLUG="$(echo "${TORCH_CUDA_ARCH_LIST}" | tr -d ' .' | sed 's/^/sm/')"
VERSION_TAG="arm64-${ARCH_SLUG}-base-${BUILD_TIME}-${SHORT_SHA}"
FLOATING_TAG="arm64-${ARCH_SLUG}-base"

echo "=== base build ==="
echo "  build base: ${BUILD_BASE_IMAGE}"
echo "  arch list : ${TORCH_CUDA_ARCH_LIST}  (tag slug ${ARCH_SLUG})"
echo "  max_jobs  : ${MAX_JOBS}  (cores ${CORES}, mem ${MEM_GB}GB), nvcc_threads ${NVCC_THREADS}"
echo "  commit    : $(git rev-parse HEAD)"
echo "  tags      : ${VERSION_TAG} / ${FLOATING_TAG}"
echo

# 早失败优于编到一半才发现基座不对。
# 用 imagetools 而不是 `docker manifest inspect`:后者对单架构镜像返回的
# manifest 里根本没有 architecture 字段(那在 config 里),而经
# `crane --platform` 搬进 ACR 的镜像正是被拍平成单架构的,会误判。
if docker image inspect "${BUILD_BASE_IMAGE}" >/dev/null 2>&1; then
  # 本地已有(含 Dockerfile.cn-bases 派生出的 local/* 纯本地 tag)。
  # 这种镜像不在任何 registry 上,imagetools 查不到,所以先查本地镜像库。
  base_arch="$(docker image inspect "${BUILD_BASE_IMAGE}" --format '{{.Architecture}}')"
  if [ "${base_arch}" != "arm64" ]; then
    echo "ERROR: ${BUILD_BASE_IMAGE} 是 ${base_arch},不是 arm64。" >&2
    exit 1
  fi
elif ! docker buildx imagetools inspect "${BUILD_BASE_IMAGE}" 2>/dev/null | grep -qi "arm64"; then
  echo "ERROR: ${BUILD_BASE_IMAGE} 看不到 arm64(或镜像不可达)。" >&2
  echo "       别用 docker/Dockerfile 的默认 BUILD_BASE_IMAGE(manylinux2_28-builder 是 amd64 单架构)。" >&2
  echo "       隔离网下先跑 mirror-build-base-to-acr.yml 把基座搬进 ACR。" >&2
  exit 1
fi

OUTPUT_ARGS=()
if [ "${PUSH}" = "1" ]; then
  if [ -n "${ACR_PASSWORD:-}" ]; then
    echo "${ACR_PASSWORD}" | docker login "${ACR_REGISTRY}" \
      -u "${ACR_USERNAME:?需要 ACR_USERNAME}" --password-stdin
  fi
  # buildx 的默认 `docker` driver 不能导出到 registry(--push),
  # 所以推送走一个专用的 docker-container builder。
  # 清理:docker buildx rm "${BUILDER_NAME}"
  BUILDER_NAME="${BUILDER_NAME:-vllm-backport-builder}"
  docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1 \
    || docker buildx create --name "${BUILDER_NAME}" --driver docker-container >/dev/null
  OUTPUT_ARGS=(--builder "${BUILDER_NAME}" --push)
else
  # 默认 `docker` driver 直接把镜像建进本地镜像库,不必 --load。
  echo "PUSH=0:只构建到本地镜像库,不推送。"
  OUTPUT_ARGS=()
fi

# provenance/sbom 关掉:阿里云 ACR 个人版不认 buildx 的证明清单
# (空描述符 application/vnd.oci.empty.v1+json),推送报
# "denied: unknown manifest class"。踩过,见 LightX2V 的 build-arm64-docker.yml。
EXTRA_ARGS=()
[ -n "${FINAL_BASE_IMAGE}" ] && EXTRA_ARGS+=(--build-arg "FINAL_BASE_IMAGE=${FINAL_BASE_IMAGE}")
[ -n "${GET_PIP_URL}" ] && EXTRA_ARGS+=(--build-arg "GET_PIP_URL=${GET_PIP_URL}")
[ -n "${DEADSNAKES_MIRROR_URL}" ] \
  && EXTRA_ARGS+=(--build-arg "DEADSNAKES_MIRROR_URL=${DEADSNAKES_MIRROR_URL}")
if [ -n "${PIP_INDEX_URL}" ]; then
  # Dockerfile 里 UV_INDEX_URL 默认取 PIP_INDEX_URL,但那是 ARG 默认值,
  # 显式传了 PIP_INDEX_URL 时两个都给,免得某一层走到 uv 又回落到 pypi.org。
  EXTRA_ARGS+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}"
               --build-arg "UV_INDEX_URL=${PIP_INDEX_URL}")
fi

docker buildx build \
  --file docker/Dockerfile \
  --target vllm-openai \
  --provenance=false \
  --sbom=false \
  "${EXTRA_ARGS[@]}" \
  --build-arg "BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}" \
  --build-arg "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}" \
  --build-arg "max_jobs=${MAX_JOBS}" \
  --build-arg "nvcc_threads=${NVCC_THREADS}" \
  --build-arg "INSTALL_LMCACHE=${INSTALL_LMCACHE}" \
  --label "io.vllm-backport.base-commit=$(git rev-parse HEAD)" \
  --label "io.vllm-backport.torch-cuda-arch-list=${TORCH_CUDA_ARCH_LIST}" \
  --label "io.vllm-backport.build-base-image=${BUILD_BASE_IMAGE}" \
  --tag "${ACR_REPO}:${VERSION_TAG}" \
  --tag "${ACR_REPO}:${FLOATING_TAG}" \
  "${OUTPUT_ARGS[@]}" \
  .

echo
echo "=== 完成 ==="
echo "  ${ACR_REPO}:${VERSION_TAG}"
echo "  ${ACR_REPO}:${FLOATING_TAG}"
if [ "${PUSH}" != "1" ]; then
  echo
  echo "镜像只在本地。推送:"
  echo "  docker login ${ACR_REGISTRY}"
  echo "  docker push ${ACR_REPO}:${VERSION_TAG}"
  echo "  docker push ${ACR_REPO}:${FLOATING_TAG}"
fi
echo
echo "下一步:"
echo "  1) 跑 'Sync base image (ACR -> Docker Hub)' 流水线,base_tag=${FLOATING_TAG}"
echo "     (国外 runner 拉 Docker Hub 比拉上海 ACR 快一个量级)"
echo "  2) 之后改 Python/Triton 只需跑 build-arm64-app.yml"
echo "  3) 有卡的话先冒烟:"
echo "     docker run --rm --gpus all --entrypoint python3 ${ACR_REPO}:${VERSION_TAG} \\"
echo "       -c 'import torch; print(torch.cuda.get_arch_list())'"
