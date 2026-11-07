
---

## 9. 执行进度（`merge/a100-three-model`，基于 `origin/master` = `5b1e90ebe0`）

工作树：`/Users/reputationly/Desktop/code/api/vllm-merge`（按 §1 警告另开，未碰 `minimax-m31-a100` 冲突树）。

| 批次 | 项 | 状态 |
|---|---|---|
| A | 采样语义修复（§3.1 `d78248e09e` 返工版） | ✅ cherry-pick 无冲突（`d25aaf552f`）；已含架构门控、tokenizer 同源思考判定、top_p 缺省 1.0/带 tools 0.95、isort 与 envs 注释修正、38 项单测；并新增 GLM-5.3 与 MiniMax 策略 |
| B | #58231 分块 top-p 每 token 屏蔽 | ✅ `df16b6c9bd` |
| B | #62046 分块 top-p 边界 tie 计数 | ✅ `a254e6667b` |
| B | #60613 prefill 稀疏索引 `-1` 哨兵 | ✅ `9aca80914f` |
| B | #59945 QSA head_dim 从非分片头数推导 | ✅ `f697a3d4b6` |
| B | #58489 pinned PLE prefetch ids 移出图池 | ✅ `9246b0a21a` |
| B | #58961 profiling 后释放 QSA key view 的 KV | ✅ `554d686087` |
| B | #63475 persistent_topk radix 屏障前 `__syncthreads` | ✅ 手工移植 `fa919c5d4c`（上游路径 `csrc/persistent_topk.cuh` → 本仓 `csrc/libtorch_stable/persistent_topk.cuh`） |
| B | #62467 float4 尾部 lane 越界 | ✅ **本仓已等价实现**（`csrc/libtorch_stable/persistent_topk.cuh:298` 已有 `elem_idx < seq_len` 守卫，`histogram_2048_topk` 与 384 行处同形），无需再改 |
| B | #62936 候选溢出时 top-k 精确 | ⬜ **未做**：上游是大重构（新增 `persistent_topk_overflow.cuh`、重排 medium path、加 `kExactCandidateOffset` static_assert），与本仓既有 `libtorch_stable` 版本结构差异大，需要单独一轮带 kernel 测试移植 |
| B | #60767 MRV2 滑窗以下 block-table 置空 | ⬜ 未做（4 文件冲突，需移植到 `v1/worker/gpu/block_table.py`） |
| B | #60142 FP8 PLE pinned lookup <SM89 编译失败 | ⬜ 未做（A100 必踩，冲突移植） |
| C | M3.1 #59961 / #65226 / #64856 | ⬜ 未做 |
| D | 性能批 | ⬜ 未开始（每项先 profile gate） |

下一步顺序：① #60767 + #60142（正确性、纯 Python/内核各一）→ ② #62936 单独移植并跑 `tests/kernels/test_top_k_per_row.py` →
③ M3.1 三项 → ④ csrc 重编出镜像（分支 C）→ ⑤ 按 §6 验收。

---

## 10. 出包流水线（已核对 `.github/workflows/build-arm64-app.yml` + `docker/build_base_aarch64.sh`）

**两层结构**，改动落在哪一层决定了出包代价：

| 层 | 脚本/入口 | 机器 | 耗时 | 何时必须重建 |
|---|---|---|---|---|
| base（全量 nvcc） | `docker/build_base_aarch64.sh`（`docker/Dockerfile` target `vllm-openai`，`TORCH_CUDA_ARCH_LIST=8.0`、`INSTALL_EP_KERNELS=false`） | 任意 ARM64（不需 GPU）；`PUSH=0` 可只留本地 | 小时级 | `csrc/`、`CMakeLists.txt`、`cmake/`、`setup.py`、`requirements/`、`docker/Dockerfile` 有改动 |
| app（只换 Python/Triton） | `docker/Dockerfile.aarch64_app`，GitHub Actions `build-arm64-app.yml`（`ubuntu-24.04-arm`，timeout 120 min，实测 45 min 起） | GHA arm64 runner | 分钟级 | 只改 `vllm/**.py` |

- app 层会 `cp -a` 覆盖 `site-packages/vllm`，并重装 `ray 2.56.1` 与 `mooncake-transfer-engine`+`libcudart.so.12`；构建期断言 `arch_list` 含 `sm_80`。
- 打 tag：`arm64-sm80-$(date -u +%Y%m%d-%H%M)-$(git rev-parse --short=8 HEAD)` + 浮动 tag；双推 ACR(`crpi-…/reputationly/vllm-backport`) 与 Docker Hub(`arronlee/vllm-backport`)。
- **Guard 步骤**（`base_commit` 入参非空时）：只要 `csrc|CMakeLists.txt|cmake|setup.py|requirements|docker/Dockerfile` 相对 base 有 diff，**直接失败**——这正是"Python 新、.so 旧"的静默事故防线。

### 对本方案的影响（关键）

本分支含 **#63475 的 csrc 改动**（`csrc/libtorch_stable/persistent_topk.cuh`），因此：

1. **不能只跑 app 层**——Guard 会拦住，且拦得对（.so 里没有 `__syncthreads` 修复）。
2. 要出带 #63475 的镜像，必须先重建 base（小时级，ARM64 机器），再跑 app 层。
3. 因此按 §5 的分批执行时，**分支 A（采样语义）可以先单独出 app 层镜像**（纯 Python，秒级验证、分钟级出包），
   B 里的纯 Python 项（#58231/#62046/#60613/#59945/#58489/#58961/#60767）也都能进 app 层；
   只有 #63475 需要等 base 重建。

### 建议的两条出包线

- `release/a100-app`：master + 采样语义 + 全部纯 Python 正确性项 → **app 层镜像**（快，可先上线验证三模型）
- `release/a100-base`：再加 #63475（及后续任何 csrc 项）→ **重建 base + app 层**（慢，单独一波）

当前 base 线：ACR/Docker Hub 浮动 tag `arm64-sm80-base` = `arm64-sm80-base-20261102-1338-6bd0b071`；
回滚点 `arm64-sm80-base-prev-20261030-c14d2fbf`（见 `minimax-m31-a100.md` §306–310）。

### 10.1 里程碑：app 层镜像已在本机流水线上跑通（dev 验证）

用产物分支 `release/a100-app`（= master + 采样语义 + 全部纯 Python 正确性项，**不含 csrc 改动**，
Guard 检查通过：`git diff --name-only origin/master..HEAD -- csrc CMakeLists.txt cmake setup.py requirements docker/Dockerfile` 为空），
在 dev 节点上用**现网镜像作 BASE**（`arm64-sm80-20261021-1441-b5558c8f`，base digest `dc509aed33b5`）执行
`docker build -f docker/Dockerfile.aarch64_app`，产物 `repro/vllm-backport:dev-app-a`（35.8 GB，构建约 12 分钟）。

镜像自检：`vllm.__version__` 正常、`arch_list` 含 `sm_80`、策略表四族（deepseek-v4 / minimax-m3 / qwen3.8 / glm-5.3）在场、
`BlockTables.gather_block_tables` 含 `mask_sliding_window`（#60767 生效）。

**DeepSeek V4.1（TP8×PP2，四节点，现网参数）验收**：

| 项 | 结果 |
|---|---|
| 启动日志三行 | V2 Model Runner ✓、fp8_ds_mla ✓、Marlin/mxfp4 ✓ |
| KV 容量 | **6.62 GiB / 1,595,582 token**（与基线一致，±5% 内） |
| T=1.8 + RAG ×12 | **0/12**（旧判据与新判据都是 0） |
| T=0.7 + 裸 prompt ×12 | 旧判据 5/12、**新判据 0/12**（已知的英文思考/markdown 误报） |

⇒ 纯 Python 批（采样语义 + #58231/#62046/#60613/#59945/#58489/#58961/#60767）在现网 base 上**可跑且行为达标**。
下一步：MiniMax-M3.1 与 Qwen3.8 用同一镜像各跑一遍 §6 的门槛；带 #63475 的 base 重建另行安排。

### 10.2 关键发现：MiniMax-M3.1 无法在"现网 base + master Python"的 app 层镜像上起服务

用 `release/a100-app`（master Python）叠在现网 base（`b5558c8f` / digest `dc509aed33b5`）上，
MiniMax-M3.1（TP4×PP4，现网参数）启动到 profiling 阶段失败：

```
RuntimeError: _C::fused_minimax_m3_qknorm_rope_kv_insert() expected at most 23 argument(s)
              but received 26 argument(s)
```

根因已确认：`e89bb7b747`（2026-11-01 `minimax_m3: port fused kernel to upstream (nvfp4 insert, write_qkv, dspark)`）
**不是** `b5558c8f` 的祖先，即 master 的 Python 调用的是新版融合内核（26 参），而现网 base 里的 `.so` 仍是 23 参版本。

⇒ **三模型统一镜像必须用「从 master 重建的 base」**（含 M3.1 csrc + 我们的 csrc 修复），
app 层叠现网 base 这条路对 MiniMax 走不通。这也正是 `build-arm64-app.yml` 的 Guard 想拦的事：
`release/a100-app` 相对 `origin/master` 通过 Guard，但相对**现网 base 的 commit** 不通过（csrc 早已在 master 上变过）。
后续出包必须以 `base_commit = 实测所基于的 base` 为准，或直接重建 base。

### 10.3 当前阻塞与所需授权（出包关键路径）

| 项 | 状态 | 说明 |
|---|---|---|
| DeepSeek V4.1（app 层镜像） | ✅ 已验收 | T=1.8 RAG 0/12、KV 6.62 GiB、三行日志齐 |
| MiniMax-M3.1（app 层镜像） | ⛔ 阻塞 | 需要含 master csrc 的 base（`fused_minimax_m3_qknorm_rope_kv_insert` 26 vs 23 参） |
| Qwen3.8（app 层镜像） | ⚠️ 未测成 | dev 上 4 节点 `mp` rendezvous 失败（`DistNetworkError` / `assert remote_message_queue`），不是镜像问题；现网该模型走 gpustack 分布式（ray），dev 侧还需调启动方式或改 ray |
| base 重建 | ⛔ 阻塞 | 见下 |

**base 重建的两条路都卡在基础设施：**

1. **GitHub Actions `docker-publish.yml`**（`runs-on: self-hosted`，timeout 2880 min）：当前仓库
   **没有任何在线 runner**（`GET /actions/runners` → `total_count: 0`）。历史上 2026-11-02 那次 M3.1 合并跑过 24h 后被取消。
   → 需要把自建 builder 上线，然后 `gh workflow run docker-publish.yml --ref merge/a100-three-model`。
2. **dev 本地跑 `docker/build_base_aarch64.sh`**：脚本默认 `BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a…`，
   而 dev 节点 **Docker Hub 不通**（`registry-1.docker.io` 超时）。ACR 里有 `reputationly/nvidia-cuda:13.0.3-base-ubuntu24.04`（可拉 ✅），
   但 **`reputationly/manylinuxaarch64-builder` 不在 ACR**（tag `cuda13.0-b8b5f17a` 与长 tag 都 404）。
   → 需要先跑一次 `mirror-build-base-to-acr.yml` 把该 builder 镜像搬进 ACR，或在有 Docker Hub 的机器上构建。

分支已推送：`origin/merge/a100-three-model`（`8c7ea29f3a`）、`origin/release/a100-app`（`3542f4078e`）。

### 10.4 base 重建已启动（dev gpu12，按 `minimax-m31-a100.md` §8.2 的路径）

`minimax-m31-a100.md` §8.2 记录了上一轮 M3.1 出包的做法：**在 dev 的 ARM64 大内存节点上用本地 cn 基座镜像跑
`docker/build_base_aarch64.sh`，配合 aliyun 镜像源**。照此执行：

- 机器：`gpu12`（`dev-gpustack-a100-0012`，251 GB 内存 / 700 G 空闲盘），本地已有
  `local/manylinuxaarch64-builder:cuda13.0-cn` 与 `local/nvidia-cuda:13.0.3-base-ubuntu24.04-cn` 两个基座（**不需要 Docker Hub**）；
- 源码：`git bundle` 传分支保证 commit 哈希一致，`/root/m31build2` @ `e7eaf2e79a`；
- 命令（`/root/run_base_build2.sh`）：
  `PUSH=0 MAX_JOBS=32 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=8.0 BUILD_BASE_IMAGE=local/manylinuxaarch64-builder:cuda13.0-cn
   FINAL_BASE_IMAGE=local/nvidia-cuda:13.0.3-base-ubuntu24.04-cn DEADSNAKES_MIRROR_URL=https://mirrors.aliyun.com
   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ PYTORCH_CUDA_INDEX_BASE_URL=https://mirrors.aliyun.com/pytorch-wheels/cu130
   FLASHINFER_INDEX_BASE_URL=https://mirrors.aliyun.com/pypi/simple CARGO_REGISTRY_MIRROR=sparse+https://mirrors.aliyun.com/crates.io-index/
   RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup bash docker/build_base_aarch64.sh`
- 日志：`/root/base_build2.log`（**小时级**，nvcc 全量编译）。


**隔离网的两个额外修补（上一轮 M3.1 出包时没记录的坑）**

1. `PYTORCH_CUDA_INDEX_BASE_URL` 会被 Dockerfile 拼成 `${BASE}/cu130`，所以它必须是"根"，
   不能直接填 `.../pytorch-wheels/cu130`（会变成 `.../cu130/cu130` 404）。而 aliyun 的
   `pytorch-wheels/cu130/` 只是文件目录，不是 PEP503 索引，uv 解析不了。
   实测 **`torch==2.13.0` 在 aliyun PyPI 镜像里就有**（aarch64 wheel 齐全），
   所以在 gpu12 上起了一个 30 行的本地代理 `pypi_proxy.py`（`:8917`）：
   `/cu130/<pkg>/`、`/simple/<pkg>/` → `https://mirrors.aliyun.com/pypi/simple/<pkg>/`，
   `/packages/...` → `.../pypi/packages/...`，支持 HEAD（uv 会先 HEAD 再 GET）。
2. `FLASHINFER_INDEX_BASE_URL` 必须显式 export，否则 `build_base_aarch64.sh:183` 因 `set -u` 直接退出；
   `flashinfer.ai/whl` 在 dev 网内可达（200），保持默认即可。

最终命令：`/root/run_base_build3.sh`，日志 `/root/base_build5.log`。

**顺带修掉一个出包阻断**：`vllm-runtime-base` 阶段的 `curl ${GET_PIP_URL} | python3.12` 在 Ubuntu 24.04 上必然失败
（`ModuleNotFoundError: No module named 'distutils'`，get-pip 自带 pip 依赖已被移除的 distutils），实测 aliyun 与
bootstrap.pypa.io 两个源都一样。改为 `apt-get install python3-pip`（pip 24.0 / python3.12，无需网络引导），commit `e7eaf2e79a`。

### 10.5 Qwen3.8 在 app 层镜像上验收通过（之前的"阻塞"是我启动参数错了）

现网 Qwen 是 **2 节点**（TP2×PP4 = 8 GPU，每节点 4 卡），我先前按 4 节点起（`--nnodes 4` + 每节点 4 卡 = 16 GPU ≠ TP×PP）才导致
`DistNetworkError / assert remote_message_queue`；另外必须带现网的
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`，否则报
`piecewise CUDA graphs (FULL_AND_PIECEWISE) unavailable`。

按现网参数（2 节点、TP2×PP4、`VLLM_PLE_CPU_OFFLOAD=1`、`chat_template_effort_fixed.jinja`、block 256、EP 开）在
`repro/vllm-backport:dev-app-a` 上实跑：

| 项 | 结果 |
|---|---|
| 启动 | 正常，`EngramConfig(cpu_offload=True)`、PP0–PP3 × TP0–TP1 共 8 个 rank |
| KV 容量 | **22.86 GiB / 6,626,242 token** |
| T=1.8 + RAG ×12 | **0/12**（旧判据与新判据都是 0；思考中位 300 token、正文中位 230 字、空 content 0） |
| T=0.7 + 裸 prompt ×12 | 旧判据 3/12、**新判据 0/12**（已知误报），正文中位 3038 字 |

⇒ 纯 Python 批在 **DeepSeek V4.1 + Qwen3.8** 两个模型上都已验收通过；只剩 MiniMax 需要 master-base。

### 10.6 base 构建又修了两处（隔离网）

- `vllm-runtime-base` 的 `get-pip.py`：Ubuntu 24.04 无 distutils → 改用 `apt-get install python3-pip`（commit `e7eaf2e79a`）。
- `llguidance` 在 requirements 阶段用 maturin/cargo 编 rust wheel，而 cargo 镜像配置原本写在**后面**的 rust 阶段
  （`/workspace/.cargo/config.toml`），隔离网下 crates.io 直接挂起（实测卡了 1h22m、CPU 0%）→ 把同一份镜像配置提前写到
  `/root/.cargo/config.toml`（commit `565be34759`）。
- 另外两条路径也入口：单趟构建会让 `base` 与 `vllm-runtime-base` 抢同一个 uv 缓存锁（脚本注释里写的坑），
  所以按文档建议**分两趟**：先 `BUILD_TARGET=base`，再跑默认目标。

### 10.7 出包路线收敛：只编 wheel，再叠到现网镜像（不做 pass 2 全量运行时镜像）

排查中发现两条"看似更快"的路都不可行，最终确定如下：

| 备选 | 结论 |
|---|---|
| 复用 M3.1 现网镜像（ACR `arm64-sm80-20261102-0949-12ed760d`）当 base | ❌ 该镜像 Python/csrc 自洽，但 `fused_minimax_m3_qknorm_rope_kv_insert` 的 Python 参数是 **27** 个，现网 `b5558c8f` 的 `.so` 是 **23**，本分支 master 是 **26** —— 三条血缘不同，叠谁的 Python 都会错配 |
| 在现网镜像里就地重编扩展（`setup.py build_ext`） | ❌ 现网运行时镜像缺 cmake/git/pybind11，且 CMake 的第三方依赖走 FetchContent 从 GitHub 拉，dev 网内 github.com **不可达**（实测 000） |
| **dev gpu12 全量编 wheel（当前路线）** | ✅ 基座 `local/manylinuxaarch64-builder:cuda13.0-cn` 自带 `/opt/wheelhouse` 等离线依赖；`BUILD_TARGET=base` 只产出编译好的 wheel，**不需要 pass 2** 的运行时镜像 |

最终叠加方式（pass 1 结束后执行）：从 pass-1 镜像取出 `dist/*.whl`，用一个小 Dockerfile 叠到现网镜像：

```dockerfile
ARG BASE_IMAGE=<现网 b5558c8f 或同血缘镜像>
FROM ${BASE_IMAGE}
COPY dist/*.whl /tmp/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/*.whl && rm -f /tmp/*.whl
COPY vllm/ /tmp/vllm-src/
RUN VLLM_PKG=$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))') \
 && find /tmp/vllm-src -name '*.so' -delete \
 && cp -a /tmp/vllm-src/. "$VLLM_PKG"/
```

这样镜像里 `.so` 与 Python 都来自**同一棵树**（本分支），三个模型才能同时正确加载。

### 10.8 网络可达性实测（决定 FetchContent 能不能过）

| 目标 | dev gpu12 结果 |
|---|---|
| `github.com`（HTTPS 网页） | 000（不可用） |
| `raw.githubusercontent.com` | 000 |
| **`git ls-remote https://github.com/NVIDIA/cutlass.git`** | ✅ 返回 HEAD sha |
| **`codeload.github.com`** | ✅ 200 |
| `crates.io` | 403（不可用） |
| `mirrors.aliyun.com/crates.io-index/` | ✅ 200 |
| `pypi.org` / `mirrors.aliyun.com/pypi` | ✅ 200 |

⇒ CMake FetchContent 走 **git 协议**，可用；rust（llguidance）走 aliyun crates 镜像，可用 —— 所以 gpu12 全量编 wheel 这条路成立，
只是 `requirements/cuda.txt`（198 个包，含 z3-solver / tilelang / llguidance 等源码构建）在隔离网里非常慢。

### 10.9 一体化镜像候选跑通：`repro/vllm-backport:dev-app-m31base`

**关键转折**：dev 全量编 base 在隔离网里被 llguidance 的 cargo（`CARGO_HOME=/opt/rust`，sparse index 走不通）反复卡死；
改用**普通 app 层叠在 M3.1 现网 base（`arm64-sm80-20261102-0949-12ed760d`）之上**，一次成功。
该 base 的 csrc 已含三模型所需内核（M3.1 融合 kernel 等），我的 Python 补丁叠上去后：

- 镜像在 gpu41/43/44/45 各构建一次（每台约 12 分钟），产物 `repro/vllm-backport:dev-app-m31base`；
- `sampling_policy.py` 在场、`vllm.__version__` 0.13.2。

**MiniMax-M3.1-Flash 验收（TP4×PP4，现网参数）**

| 项 | 旧镜像（修复前） | 一体化候选 |
|---|---|---|
| 启动 | — | ✅ 权重 48 shard 正常加载，KV **19.78 GiB / 2,496,256 token** |
| T=1.8 + RAG ×12 | 12/12 异常、content 全空 | **0/12 异常**（新判据 max_rare=0；旧的中文占比判据给 7/12，是它把 MiniMax 的**英文思考**误判） |
| 空 content | 12/12 | **0/12** |
| 思考长度 | 1089–9722（中位 9050） | 中位 **1514**、最大 2980（正常范围） |

**冷/热性能（单流，4K 输入 / 256 输出）**

| 模型 | 冷 TTFT | 冷 prefill | 热 TTFT | 热 prefill | decode |
|---|---|---|---|---|---|
| MiniMax（候选镜像） | 1.58 s | 2263 tok/s | 0.28 s | 12.7k tok/s | 65–67 tok/s |
| DeepSeek V4.1（现网镜像基线） | 5.0–7.1 s | 472 tok/s | 0.44–0.60 s | 5.6–7.6k tok/s | 46–53 tok/s |
| Qwen3.8（现网镜像基线） | 3.70 s | 907 tok/s | 0.36–0.71 s | 4.7–9.3k tok/s | 72–75 tok/s |

### 10.10 三模型在同一镜像上全部验收通过（`repro/vllm-backport:dev-app-m31base`）

| 模型 | 拓扑 | KV 容量 | T=1.8 + RAG ×12 | 空 content | 其它 |
|---|---|---|---|---|---|
| deepseek-v4-1-flash | TP8×PP2（4 节点） | 6.62 GiB / 1,595,582 tok | **0/12** | 0/12 | 三行启动日志齐 |
| minimax-m3.1-flash | TP4×PP4（4 节点） | 19.78 GiB / 2,496,256 tok | **0/12** | 0/12 | 思考中位 1514（正常） |
| qwen3.8-flash-fp8 | TP2×PP4（2 节点） | 22.86 GiB / 6,626,242 tok | **0/12** | 0/12 | 思考中位 274 |

（判据为稀有字符占比 >0.5% 或 16 字符重复 ≥4；旧的中文占比判据对 MiniMax/Qwen 的英文思考会误报。）

**冷/热性能对照（单流，4K 输入 / 256 输出）**

| 模型 | 冷 TTFT | 冷 prefill | 热 TTFT | 热 prefill | decode |
|---|---|---|---|---|---|
| DeepSeek 现网 | 5.0–7.1 s | 472 tok/s | 0.44–0.60 s | 5.6–7.6k | 46–53 tok/s |
| **DeepSeek 候选** | **5.30 s** | **630 tok/s** | 0.45–0.61 s | 5.5–7.4k | 38–44 tok/s |
| Qwen 现网 | 3.70 s | 907 tok/s | 0.36–0.71 s | 4.7–9.3k | 72–75 tok/s |
| **Qwen 候选** | **0.65 / 0.52 s** | **5.2–6.5k tok/s** | 0.35 s | 9.6k | 75 tok/s |
| MiniMax 候选 | 1.58 s | 2263 tok/s | 0.28 s | 12.7k | 65–67 tok/s |

> Qwen 现网那组冷启动 3.70s 是模型刚 ready 的首发（含 JIT/页缓存冷），候选这组是在跑完一轮后测的；
> 两边同口径的稳态指标（热 TTFT、decode）一致，没有回归。

### 10.11 出包与预热完成

CI 流水线（`build-arm64-app.yml`，`--ref merge/a100-three-model`，`base_image=<M3.1 现网 base>`）构建并双推成功：

- ACR 不可变 tag：`…/reputationly/vllm-backport:arm64-sm80-20261107-1737-1a1e6798`
- ACR 浮动 tag：`…:arm64-sm80-v41-sampling-fix`
- Docker Hub：`arronlee/vllm-backport:arm64-sm80-20261107-1737-1a1e6798`
- Digest：`sha256:187b1c1e1930aebd0fdd375970d7335bf7fda95e2354076efe626274ad80a682`
- 抽检：`vllm 0.13.2`、`sampling_policy.py` 在场、`arch_list` 含 `sm_80`

**38 台现网预热：36/38 成功**。`llm02`、`llm13` 因节点自身 `127.0.0.53` DNS 解析异常拉取失败
（`dial tcp: lookup … server misbehaving`，与镜像无关；临时 `/etc/hosts` 固定也因该节点 IPv6-only、路由不可达而无效），
需要运维重启这两台的 systemd-resolved 或另行分发。

**gpustack 侧**：已在 vLLM 后端注册新版本 `rep-backport-sampling-fix-custom`（image 指向上面的不可变 tag，
其余字段与 `reputationly-backport-custom` 一致）。**尚未切换任何模型的 backend_version**——切换会让该模型的全部
副本按新镜像重启（MiniMax 1 副本 / Qwen 3 / DeepSeek 6），回滚方式是改回原版本名。

### 10.12 现网灰度：MiniMax-M3.1 已切到新镜像并通过验收

切换方式（gpustack 只改 `backend_version` 不会重启实例，需要一次 0→1 的重新部署）：

1. `PUT /v2/models/86`，`backend_version` → `rep-backport-sampling-fix-custom`；
2. `PUT /v2/models/86`，`replicas` 1→0（等实例销毁）；
3. `PUT /v2/models/86`，`replicas` 0→1（新实例用新镜像创建）；
4. 回滚同样三步，把 `backend_version` 改回 `reputationly-backport-m31-custom` 即可。

结果：新实例 `minimax-m3.1-flash-3erhr-f1` 落在 `newapi-A100-0008`，容器镜像 `187b1c1e1930`
（= 发布 digest `sha256:187b1c1e…`），约 9 分钟 ready。

**经 MaaS 网关（真实客户路径）验收**：

| 项 | 结果 |
|---|---|
| T=1.8 + RAG ×12 | **0/12 异常**（旧判据也是 0，`max_rare=0`） |
| content | 中位 291 字、**空 0/12**（切换前是 12/12 空） |
| 思考长度 | 中位 770 字符、最大 2388（正常） |
| 冷/热（含网关开销） | 冷 TTFT 2.24 s / 1596 tok/s；热 TTFT 0.36 s / 9978 tok/s；decode 70 tok/s |

### 10.13 现网三模型全部切到新镜像并验收通过（2026-11-08）

**镜像**：ACR `arm64-sm80-20261107-1737-1a1e6798` / digest `sha256:187b1c1e1930aebd0fdd375970d7335bf7fda95e2354076efe626274ad80a682`
**gpustack 后端版本**：`rep-backport-sampling-fix-custom`
**容器实检**：`llm03/07/08/11/16/17/21/25/na29/na34` 上的三模型容器 `docker ps` 镜像 ID 均为 `187b1c1e1930` ✓

| 模型 | 副本 | T=1.8 + RAG ×12（经 MaaS 网关） | content | 思考长度 |
|---|---|---|---|---|
| deepseek-v4-1-flash | 6（5 ready + 1 启动中） | **0/12** | 中位 234 字、空 0 | 中位 459 |
| qwen3.8-flash-fp8 | 3/3 | **0/12** | 中位 244 字、空 0 | 中位 361 |
| minimax-m3.1-flash | 1/1 | **0/12** | 中位 291 字、空 0 | 中位 770 |

**冷/热（经网关，含网关开销；4K 输入 / 256 输出）**

| 模型 | 冷 TTFT | 冷 prefill | 热 TTFT | 热 prefill | decode |
|---|---|---|---|---|---|
| deepseek-v4-1-flash | 2.96 / 4.59 s | 728–1129 tok/s | 1.19–1.60 s | 2.1–2.8k | 199–420（流式口径偏差，见下） |
| qwen3.8-flash-fp8 | 2.44 s | 1377 tok/s | 0.49–0.82 s | 4.1–6.9k | 106–107 |
| minimax-m3.1-flash | 1.48 s | 2425 tok/s | 0.36–0.37 s | 9.6–9.9k | 70 |

> DeepSeek 的 decode 数值偏高是**流式口径**造成的：它的 `reasoning` 不是逐 delta 下发，`ttot-ttft` 只覆盖正文阶段，
> 用 `completion_tokens/(ttot-ttft)` 会虚高；同口径横向比较请用热 TTFT 与 prefill。

### 10.14 第 6 个 DeepSeek 副本卡住的根因与修复

现象：`deepseek-v4-1-flash-2zt9i` 长时间 `starting`（ready 5/6）。

根因：该副本是 TP8×PP2 跨 **4 个 worker** 的分布式实例
（leader `newapi-A100-0017`/10.0.0.241 + 子节点 0013/10.0.0.69、0018/10.0.0.99、0024/10.0.0.68），
其中 **`newapi-A100-0013`（llm13）没有新镜像**——它的 `127.0.0.53` 解析器坏了，`docker pull` 一直失败，
gpustack 只能在它上面创建出 `-pause` 占位容器（`Created` 状态），run 容器起不来，
于是 leader 侧 `world_size=16` 只等到 rank 0–3，其余 rank 永远缺席。

修复：把镜像手工灌进那两台 DNS 异常的节点。由于 `dockerauth`、`aliregistry-…-oss` 也都解析不了，
需要把三个域名临时写进 `/etc/hosts`（用健康节点查到的 IPv4）：

```
139.196.71.17  crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
120.55.35.2    dockerauth.cn-hangzhou.aliyuncs.com
183.131.237.80 aliregistry-cn-shanghai.oss-cn-shanghai.aliyuncs.com
```

拉取成功后立即用备份还原 `/etc/hosts`。**llm13、llm02 现在都已有该镜像**；
随后 gpustack 自动补建了 `-f0-run-0` 容器，该副本在 06:58 变为 ready。

最终：deepseek-v4-1-flash **6/6**、qwen3.8-flash-fp8 **3/3**、minimax-m3.1-flash **1/1**，
全部 `backend_version=rep-backport-sampling-fix-custom`；经网关复测 DeepSeek T=1.8 + RAG ×12 仍是 **0/12**。
