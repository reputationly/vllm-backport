
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
