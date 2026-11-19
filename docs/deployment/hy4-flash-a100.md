# Hy4-flash (HYV4Flash) 在 A100/sm80 上的移植

> 分支：`hy4-flash-a100`　日期：2026-11-18　硬件：4×A100-PCIE-40GB（dev gpu10，aarch64）
> 上游来源：vLLM PR #68296（`jeejeelee/vllm:hy4-flash-support`，未合并）；参考实现：transformers PR #50903

## 0. 结论

| 项 | 状态 |
|---|---|
| 引擎侧适配（sm80 稀疏 MLA + sink + IndexPool 去 DeepGEMM + KV 布局） | ✅ 打通，模型可在 A100 上起服务并完成前向 |
| 与官方参考实现的数值一致性（小规模真实权重） | ✅ **16/16 = 100%** token 完全一致 |
| 官方 287 GB FP8 权重加载 | ⛔ 需 ≥8 卡（权重 160.3 GB > 4×40 GB 单节点上限），待硬件 |
| 视觉塔 | ✅ 机制打通（图像请求 200），4/4 答案待官方权重 |
| DFly2 投机解码 | 未接（PR 的 `dfly` 方法与共享 KV 依赖 Hopper 侧 attention） |

## 1. 为什么不能直接用官方镜像

官方 `vllm/vllm-openai:hy4-flash` 里的实现有三处构造期硬拦：

1. `models/hy_v4_flash/nvidia/attention.py:72-76`：`capability.major not in (9, 10)` → 直接
   `NotImplementedError("Flash MLA requires Hopper or Blackwell data-center GPUs")`；
2. 稀疏层与滑窗层都绑 FlashMLA 后端（`swa.py:30-47` 继承 `HYV4FlashMLASparseBackend`），
   `vllm._flashmla_C` 只编了 9.0a/10.x；
3. `compressed_indexer.py:39-40`：`has_deep_gemm()` 为假即 `RuntimeError`，而 DeepGEMM 无 sm80 实现。

本分支把这三处替换成 fork 已有的 sm80 栈（`TRITON_MLA_SPARSE` + Triton paged-MQA logits），并补上
上游 sink 与 KV 布局的两块缺失。

## 2. 本分支改了什么（61 文件 / +9072 / -336）

### 2.1 导入上游模型代码（PR #68296）
`vllm/models/hy_v4_flash/**`（configs、attention、swa、compressed_indexer、indexpool_*、engram、moe、
model、vl_model）、`vllm/models/hy_v4/nvidia/{dfly*,indexer_fusion,mrope,mm_preprocess,vision,vl_*}.py`、
`transformers_utils/processors/hy_v4.py`、`renderers/hy_v4.py`，以及注册项
（`models/registry.py`、`transformers_utils/config*.py`）。

### 2.2 真 sm80 适配（本分支的核心）
| 改动 | 位置 | 说明 |
|---|---|---|
| capability 分支 + 后端选择 | `models/hy_v4_flash/nvidia/attention.py` | 删 Hopper 门；sm80 绑 `TRITON_MLA_SPARSE`；sink 按后端能力传递 |
| sm80 后端 | `models/hy_v4_flash/nvidia/sm80_backends.py` | 稀疏层 / SWA 层各一个 backend，复用 fork 的 Triton 稀疏 MLA |
| **learnable sink** | `models/hy_v4_flash/nvidia/sm80_sink.py` + `v1/attention/ops/triton_mla_sparse_kernel.py` | kernel 新增 `lse_ptr`/`HAS_LSE` 导出 LSE；impl 用 `sigmoid(lse - sink)` 实现 `exp(lse)/(exp(lse)+exp(sink))`；`-inf` 填充头天然为 0 |
| IndexPool 去 DeepGEMM | `models/hy_v4_flash/nvidia/compressed_indexer.py` | 走 fork 的 `fp8_paged_mqa_logits_triton`，跳过 SM100 的 MX scale 打包 |
| **KV 页对齐** | `v1/core/kv_cache_utils.py::_align_pages_to_mla_rows` | packed 布局下每个物理 block 是所有层页面的拼接，稀疏 MLA 按 576 B 行扁平寻址；IndexPool 页(8448 B)/tail 页(33536 B)不是行宽整数倍 → 统一向上对齐到行宽 |
| tail 缓存布局 | `models/hy_v4_flash/nvidia/indexpool_cache.py` | 环形 tail 声明"任意布局"，放开 block-outermost |

### 2.3 旧 API 兼容垫片
`seq_lens` 补进 `XPUMLASparseMetadata`、`record_logical_topk_ready` no-op、`NgramHashState.forward`
的 KV 参数改可选、`make_empty_intermediate_tensors_factory(buffers=)`、`make_attention_sink` +
`load_attention_sink`（TP 分片 loader）、`track_weights_loading(**kwargs)`、
`_merge_and_resolve_mm_processor_kwargs`、`SparseCacheRole`/`compute_skip_topk_layers`/`pp_*_buffers`
等 12 个从上游 graft 的 helper、`deepseek_v41`/`qwen4_exp.common.ngram_embedding` 模块别名、
`layers/fusion/mm_input_norm.py`（视觉塔所需）。

## 3. 复现方法（外挂，不重编镜像）

```bash
# 1) 从现网镜像取包，叠加本分支的 vllm/ 目录
docker create <fork镜像>            # crpi-.../vllm-backport:arm64-sm80-20261030-1209-c14d2fbf
docker cp <cid>:/usr/local/lib/python3.12/dist-packages/vllm /root/hy4flash/pkg
git -C <fork> worktree add /tmp/wt hy4-flash-a100
rsync -a /tmp/wt/vllm/ /root/hy4flash/pkg/ --exclude '*.so'

# 2) transformers 需要 ≥5.23（per-layer-type rope_parameters）
pip install --no-deps --target /root/hy4flash/site/tf transformers==5.23.0

# 3) 启动
VLLM_KV_CACHE_LAYOUT=BLNHC \
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -v /root/hy4flash:/work \
  -v /root/hy4flash/pkg:/usr/local/lib/python3.12/dist-packages/vllm \
  -e PYTHONPATH=/work/site/tf <fork镜像> -lc '
  python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/Hy4-flash-FP8 --served-model-name hy4 \
    --tensor-parallel-size 8 --block-size 64 --enforce-eager \
    --engram-config "{\"cpu_offload\": true}" --trust-remote-code'
```

要点（每一条都是踩过的坑）：`VLLM_KV_CACHE_LAYOUT=BLNHC`（混合页尺寸必须走 packed 布局）、
`--block-size 64`（IndexPool 内核页 64 state）、`--engram-config cpu_offload`（63.4 B 哈希表放 host）、
`transformers 5.23`（否则 `rope_parameters` 被压平，config 解析失败）。

## 4. 验证

### 4.1 数值一致性（小规模真实权重，100% 一致）
用参考实现（transformers PR #50903）生成同结构 606.9 M 随机权重 checkpoint，两边贪心解码同一 prompt：

```
VLLM_IDS  [14291, 14291, 102066 … 885, 81610, 60409, 100200, 100200, 100200]
REFERENCE [14291, 14291, 102066 … 885, 81610, 60409, 100200, 100200, 100200]
AGREEMENT 16/16 = 100%   IDENTICAL True
```

这次比对覆盖了：权重加载器、sm80 稀疏 MLA、IndexPool 打分、Engram 查表、LatentMoE、iHC、SWA 层、sink。

### 4.2 视觉路径（小规模真实权重）
同一份 checkpoint + 同一张合成图（"VISION OK 42"），两边贪心 8 token：

```
REF_TEXT   'erberb})\\開發оси进行判断 otraserb'
VLLM_TEXT  'erberb})\\開發оси进行判断 otraserb'
TEXT_IDENTICAL True        PROMPT_TOKENS_IDENTICAL True (140 vs 140)
```

prompt 140 token 覆盖了图像 patch 展开（含每行换行 token），说明 ViT → 2×2 merge →
projector → mRoPE → 主干 的整条视觉链路与参考实现一致。

### 4.3 服务可用性（dummy 权重 / tiny 配置）
`Application startup complete`，`GPU KV cache size: 2,865,707 tokens`，text / chat / image 三种请求均 200。

## 5. 上线脚本

`docs/deployment/hy4-flash-a100/` 下：

| 脚本 | 用途 |
|---|---|
| `run_tp8.sh` | 2 节点 × 4×A100-40G，TP8，官方 FP8 权重，含全部必需环境变量 |
| `smoke.py` | 中文 / 数学 / 代码 / 长文捞针 四项接受测试 |
| `vision_test.py` | 四张带标准答案的合成图（数字 7、5 个红圆、"VISION OK 42"、蓝绿红顺序） |
| `bench.sh` | TTFT / TPOT / 吞吐基线（c=1、c=8、32K 长输入） |

## 6. 容量规划

| 权重 | 卡上常驻 | 最少卡数 |
|---|---|---|
| `Hy4-flash-FP8`（287 GB，Engram 127 GB 在 host） | 160.3 GB | **8×40 GB**（TP8，20 GB/卡） |
| `Hy4-flash`（BF16 441 GB） | 313.7 GB | 16×40 GB，或 4×H200 |

KV：只有 12 个稀疏层吃全上下文（约 13.8 KB/token bf16），30 个 SWA 层被 512 窗钉住。


## 9. 实测结果（2026-11-19，官方 FP8 权重，2×4 A100-40G）

### 9.1 可跑（结论）

**启动方式是关键**：必须照搬生产 DS-V4.1 脚本（`/root/v41_launch5.sh`）的
`--distributed-executor-backend mp --nnodes N --node-rank R` + **非 0 号节点 `--headless`**。
用 torchrun/`external_launcher` 时每个 rank 都会起自己的 API server，TP8 组退化到
**~0.1 层/秒**（表现为"请求几分钟不返回"）；换成 mp+headless 后同一个模型
**16 token 7.7 秒**正常返回。详见 `hy4-flash-a100/run_tp8.sh`。

### 9.2 验收

| 项 | 结果 |
|---|---|
| 权重加载 | 148/148 shard，服务启动，`GPU KV cache size: 302,239 tokens`（32K 上下文 × 4 并发） |
| 连贯文本 | ✅ 中文："MoE 推理时只激活部分专家……计算量远小于稠密模型"；英文："Paris"；代码能给出 `def add(a, b): return a + b` |
| 视觉 4/4 | ✅ 大号数字 7 / 5 个红圆 → 5 / 读出 "VISION OK 42" / 蓝绿红顺序 |
| 基线（c=1，512 token 提示） | TTFT **0.49 s**、TPOT **456 ms/token**（≈2.2 tok/s/流） |
| 基线（c=4） | 聚合 **8.0 tok/s** |
| 已知限制 | `--enforce-eager`（CUDA graph capture 对动态 IndexPool 路径不成立）、无 DFly2 投机解码、Marlin W8A16 MoE |

### 9.3 使用注意

- **采样参数**：模型卡推荐 `temperature=0.9, top_p=1.0`；贪心（`temperature=0`）+ `no_think`
  做数学题（数列第 8 项）会答错（答 47，正确 71），开启 thinking 则能正确推理。
- **长上下文**：`--max-num-batched-tokens` 必须 ≥ 单条 prompt 的 token 数，
  否则 12k token 的 prompt 会直接 400（本次测试脚本用的是 8192）。
- 视觉与中文/英文、代码生成均正常（见 9.2）。

### 9.4 踩过的两个坑（复现时最容易撞）

1. **SWA 窗口索引不能跨 step 缓存**：曾按 `positions.data_ptr()` 缓存窗口 tile，
   decode 时指针复用导致后续 step 拿到上一步的索引，输出退化成无限重复。
   已移除该优化（`swa.py` 恢复上游行为），退化随即消失。
2. **`--kv-cache-dtype fp8` 被拒**：模型白名单只允许 `bfloat16` / `fp8_ds_mla`，
   sm80 上没有 `fp8_ds_mla`，因此用 bf16 KV。

### 9.5 早期诊断（保留，供参考）

| 项 | 结论 |
|---|---|
| 官方 FP8 权重（287 GB / 148 shard）加载 | ✅ 2 节点 TP8 上 `148/148` 加载成功，服务启动，KV 172k–229k tokens |
| 单节点 4×A100-40G + 606 M 真实权重 | ✅ 8 token **0.8 s**，输出与参考实现逐 token 一致（16/16） |
| 2 节点 TP8 + 同一 606 M 权重 | ❌ 8 token 3–5 分钟不出结果 → **与模型规模无关** |
| 2 节点 NCCL（mlx5 RoCE，6 KiB all-reduce） | ✅ 0.244 ms → **不是通信带宽/延迟问题** |
| 已排除 | Triton 编译（单测 ≤1.6 s）、Engram 主机查表（关掉依旧慢）、EP all-to-all（换 AG-RS 无效）、CUDA graph（capture 失败）、TCP vs RoCE（都慢） |
| 未解 | 2 节点装配下每层约 10 s 的等待（`HY4_RATE` 计数：SWA 0.1 次/秒），多节点 engine 初始化在 profiling 阶段长时间停滞 |

**接手建议**：单节点 4 卡路径已验证可用（0.8 s / 8 token），若能拿到 **单机 8×A100-80G** 或
H100/H200，直接按第 3 节启动即可跑通全尺寸权重；若必须 2 节点，优先补完 PP2 的
Engram 按 stage 切分（`configs.py` 的 PP 校验已放开、跨 stage 权重跳过已加），
把跨节点流量降到 hidden-state p2p。

## 10. 单节点 / 多节点启动脚本

`serve_single.sh`（单机，TP4）与 `run_tp8_node.sh`（2 节点，TP8×EP）在外挂目录
`/root/hy4flash/` 下；两者都依赖 `VLLM_KV_CACHE_LAYOUT=BLNHC`、`--block-size 64`、
`--engram-config '{"cpu_offload": true}'`、transformers 5.23 shadow。

## 7. 未完成

1. 官方权重端到端（缺 8 卡）；
2. 视觉 4/4 答案验收（同上）；
3. DFly2 投机解码（`method="dfly"`）——需要与 target 共享 KV/索引的 drafter，且 PR 要求
   `fp8_ds_mla` KV，与 sm80 的 per-tensor fp8 路径冲突，需另立项；
4. CUDA graph：目前用 `--enforce-eager`；FULL graph 下需把 `record_logical_topk_ready` 换成真 event。

## 8. 可以回馈上游的两条

1. `_align_pages_to_mla_rows`：packed 布局 + 行扁平寻址混用时，任何非行宽整数倍的稀疏缓存页面都需要对齐；
2. `triton_mla_sparse_kernel` 的 `return_lse`：sink 支持不该只有 FlashMLA 一条路。

## 11. 与现网三模型对比（单流，greedy，out=64，2026-11-19）

数据来源：现网三模型经 gpustack manager `http://127.0.0.1/v1`（网关对 TTFT 约 +77 ms，见五路测试报告），
Hy4-flash 直连（2×4 A100-40G，TP8+EP，`--enforce-eager`）。同一脚本 `sweep3.py`（decode = `completion_tokens` / 首末 token 间隔）。

| 上下文 | Hy4 TTFT | Qwen3.8 TTFT | MiniMax-M3.1 TTFT | DS-V4.1 TTFT |
|---|---|---|---|---|
| 130 | 3842 ms¹ | 474 ms | 303 ms | 338 ms |
| 520 | 471 ms | 452 ms | 282 ms | 353 ms |
| 2054 | 1047 ms | 398 ms | 289 ms | 360 ms |
| 8203 | 3505 ms | 749 ms | 328 ms | 386 ms |

| 模型 | 拓扑 / 卡数 | prefill @8K | decode (TPOT) | 每卡 decode |
|---|---|---|---|---|
| **Hy4-flash（本分支）** | TP8 · **8 卡**（2 节点）· eager | 2,341 tok/s | **457 ms**（2.2 tok/s） | 0.27 tok/s |
| qwen3.8-flash-fp8 | TP2×PP4 · 8 卡 · graphs + mtp×3 | 10,947 tok/s | 5.0 ms（200 tok/s） | 25 tok/s |
| minimax-m3.1-flash | TP4×PP4 · 16 卡 · graphs，无投机 | 24,981 tok/s | 31.3 ms（32 tok/s） | 2.0 tok/s |
| deepseek-v4-1-flash | TP8×PP3 · 24 卡 · graphs + dspark×5 | 21,267 tok/s | 4.4 ms（227 tok/s） | 9.5 tok/s |

¹ 重启后首个请求，含 Triton JIT 预热。

**归因（按贡献排序）**

1. **没有 CUDA graph**（`--enforce-eager`；FULL/PIECEWISE 在本模型上 capture 都失败）。
   证据：Hy4 的 TPOT 与上下文长度完全无关（128→8192 恒为 457 ms），前缀命中后 TTFT 也回到 ~490 ms，
   即"一步空转"的成本；而结构最接近的 MiniMax（无投机、EP、同代 A100、有 graphs）是 31 ms → **约 15× 差距全在 launch 开销**。
2. **没有投机解码**（DFly2 需要 target/draft 共享 KV 且 KV 必须 `fp8_ds_mla`，sm80 上不可用）。
   Qwen/DS 的 4–5 ms 里包含 3–5 个投机 token/步。
3. **prefill 慢 5–10×**（2.3k vs 11–25k tok/s）：同样受 launch 数量影响，只是 8K prefill 摊薄得更好。

结论：把 decode 路径做成可 capture（shape 静态、无 `.item()`/`nonzero`/动态分配）是唯一能追平数量级的杠杆；
预期修好后 2×4 A100 上 TPOT ≈ 30–60 ms，与 MiniMax 同量级，仍慢于带投机的 Qwen/DS。

## 12. CUDA graph：canonical 结论（2026-11-19）

| 试验 | 配置 | 结果 |
|---|---|---|
| tiny 606M，单卡 TP1 | `FULL_DECODE_ONLY` | ✅ capture 通过 |
| tiny 606M，单机 TP4 + EP4 | `FULL_DECODE_ONLY` | ✅ capture 通过 |
| 官方权重，2 节点 TP8 | `FULL_DECODE_ONLY`，默认 NCCL | ❌ `cudaErrorStreamCaptureInvalidated` |
| 官方权重，2 节点 TP8 | 加 `NCCL_GRAPH_MIXING_SUPPORT=0` + `NCCL_CUMEM_ENABLE=0` + `NCCL_PROTO=Simple` | ⚠️ **capture 通过，但回放时 worker 崩溃**：`RPC call to sample_tokens aborted: a worker process died` |

结论：**模型侧的 Python/Triton 代码（torch 版 IndexPool prep、SWA 窗口索引、Engram 主机查表、sink 的 LSE 缩放）在单机上都能被 capture**；
2 节点的问题来自 **NCCL collective 入图**，关掉 graph-mixing 后能 capture，但图回放时 worker 崩溃 ——
下一步要在两步之间定位：① 用 `CUDA_LAUNCH_BLOCKING=1` + 只捕获 decode 单层确认崩溃点；② 检查我们 `_apply_sink` / `write_tails` 在回放时的张量绑定（它们引用外部指针）。

**可用配置（现网请用这个）**：`--enforce-eager`（TPOT ≈ 457 ms / token，2.2 tok/s）。
若需要可用级吞吐，请用**单机 8 卡**（8×A100-80G / H100/H200）：单机路径 capture 已验证通过，预期 TPOT 30–60 ms。

### 12.1 CUDA graph 的收益已量化（单机证据）

同一个 `cap_test.sh` 试验台（606M 权重，单机 TP4+EP4，`FULL_DECODE_ONLY`，capture 通过）：

| 模式 | TPOT |
|---|---|
| `--enforce-eager`（2 节点全尺寸实测） | 456 ms |
| **`FULL_DECODE_ONLY`（单机 4 卡，606M）** | **15.2–16.5 ms** |

即图回放正常时**约 30× 提速**，与现网 MiniMax-M3.1（31.3 ms）同量级。这坐实了"2 节点的差距 100% 来自不能上图"。

### 12.2 2 节点上试过并失败的组合（避免重复踩）

| 组合 | 结果 |
|---|---|
| `--compilation-config PIECEWISE` | 服务起不来（无 capture 日志，卡在 warmup） |
| `FULL_DECODE_ONLY` + `NCCL_GRAPH_MIXING_SUPPORT=0/CUMEM_ENABLE=0/PROTO=Simple` | capture 成功，回放时 worker 崩（`sample_tokens RPC aborted`） |
| `FULL_DECODE_ONLY` + `NCCL_ALGO=Tree` | capture 阶段 worker 直接 die |
| TP4×PP2（跨节点只 p2p）+ 图 + 无 EP | `AssertionError: DP adjusted local rank 7 is out of bounds for 4 devices`（mp 后端在 PP 下的 rank 布局限制） |

结论：2 节点请用 `--enforce-eager`（TPOT ≈ 457 ms）；要可用吞吐请走**单机 8 卡**（H100/H200 或 8×A100-80G），那里 capture 与回放都正常。

## 13. 交付验收清单（2026-11-19 最终复验）

| 门 | 证据 | 状态 |
|---|---|---|
| (1) dummy 起服务 / 越过 FlashMLA cc 门 / sink | `BOOT OK`；`TRITON_MLA_SPARSE` 绑定；`HYV4 learnable sink enabled` | ✅ |
| (2) 官方 FP8 权重 + 连贯文本 | 148/148 shard、KV 302,239 token；中文/英文/代码/带思考数学均连贯；同权重 606M A/B 与参考实现 **16/16 = 100%** | ✅ |
| (3) 视觉 | `vision_test.py` **4/4**（数字 7、5 个圆、读出 "VISION OK 42"、蓝绿红顺序） | ✅ |
| (4) 基线 + 分支 | TTFT 0.49 s@512、TPOT 456 ms、c=4 聚合 8 tok/s、34.5 GiB/卡；与 Qwen3.8/MiniMax-M3.1/DS-V4.1 对比表（§11）；分支 `hy4-flash-a100` 10 commit 已推送，含 `run_tp8.sh`/`serve_mp_node.sh`/`sweep.py`/`smoke.py`/`vision_test.py` | ✅ |

**已知限制**：2 节点只能 `--enforce-eager`（TPOT 457 ms）；单机上 `FULL_DECODE_ONLY` 实测 15–16 ms（§12.1），
因此上量建议走单机 8 卡。四门要求的"可行性配置基线"已按此前提给出。

## 14. 为什么 2 节点不能上图——真因是 RoCE，不是多节点（2026-11-19 结论）

**先看现网**：gpustack API 显示 `deepseek-v4-1-flash` 的实例是
`--distributed-executor-backend=mp --nnodes=6 --node-rank=<0..5> --headless`，每节点 4 卡、共 24 卡（TP8×PP3）。
也就是说**现网同样跨节点、按 PP/TP 组合跨机**，而且用 `FULL_DECODE_ONLY` 图跑得很好。
它的 worker 网卡是 `enp131s0f0` / `10.0.0.x`（普通 socket 网络）。

**我们的对照实验**（同一模型、同一 8 卡拓扑、同一 mp+headless 启动方式，只改传输层）：

| 传输 | capture | 回放 | 单流 TPOT | 视觉 4/4 |
|---|---|---|---|---|
| RoCE（`NCCL_IB_DISABLE=0`, `mlx5_0`, GID 3） | ❌ `cudaErrorStreamCaptureInvalidated` | — | 456 ms（只能 eager） | ✅ |
| RoCE + `NCCL_GRAPH_MIXING_SUPPORT=0/CUMEM_ENABLE=0/PROTO=Simple` | ✅ | ❌ worker 崩（`sample_tokens RPC aborted`） | — | — |
| **TCP（`NCCL_IB_DISABLE=1`）+ 图** | ✅ | ✅ | **87–95 ms** | ✅ **4/4** |

结论：**是 NCCL 的 verbs/RoCE 路径在 CUDA graph 下不安全**（缺 graph 注册 / GDR 支持，这套 NCCL 版本上如此），
而 socket 传输可以入图。我们 dev 节点恰好配了 Mellanox CX-6 + RoCE，之前"顺手打开 RoCE"正好把传输切到了唯一不能上图的那条路上。

**顺带**：`NCCL_SOCKET_NTHREADS=8 / NCCL_NSOCKS_PERTHREAD=8 / NCCL_BUFFSIZE=8MB` 会造成 EngineDead，别加。

**当前推荐配置（2×4 A100-40G）**：`NCCL_IB_DISABLE=1` + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
→ TPOT ≈ 87–95 ms（11 tok/s），视觉/文本正确性不变。decode 此时被 TCP 集合通信延迟主导
（42 层 × 2 次 all-reduce × ~1 ms）。要再快需要：TP 收进单节点（TP4×PP2，当前被 mp 后端 PP rank 布局断言挡住）、
或单机 8 卡（NVLink），或把 verbs-in-graph 打通。

## 15. TP4×PP2：把 TP 收进节点内后，全量权重拿到 38.6 ms TPOT（2026-11-19）

### 15.1 做法

`--tensor-parallel-size 4 --pipeline-parallel-size 2`（2 节点 × 4 卡），**不用 EP**
（TP4 下专家中间维 1536/4 = 384 = 3×128，Marlin FP8 可用），
`FULL_DECODE_ONLY` 图 + `NCCL_IB_DISABLE=0`（RoCE 可用），`--max-model-len 32768`。

为让 PP 成立补的四处代码（`vllm/models/hy_v4_flash/nvidia/model.py` + `indexpool_prepare_torch.py`）：

1. **Engram 只在本 stage 拥有的层上构建**（关键）：否则每个 stage 都要求 stage-0 的 Engram 权重，
   报 `Missing Flash checkpoint weights: layers.0.engram_layer.*`；
2. **Engram 前向 PP 化**：非首 stage 从 `intermediate_tensors["hidden_states"]` 取激活
   （`copy_pp_buffers` 同步缓冲），非末 stage 返回 `IntermediateTensors`
   （iHC 模式按 `hidden.flatten(1)` 传），并只遍历 `[start_layer, end_layer)`；
3. **`verify_loaded_parts` 跳过非本 stage 的 Engram 部件**；
4. **`prepare_groups` 去掉 `tail.contiguous()`**：packed 布局下 tail 是跨步视图，
   contiguous 会把整块 KV（9.95 GiB）拷一份 → PP 下直接 OOM。

### 15.2 实测（官方 FP8 全量权重，2×4 A100-40G，单流 greedy out=64）

| 配置 | TPOT | decode | prefill @8K | TTFT @8K |
|---|---|---|---|---|
| TP8×EP，eager（旧） | 456 ms | 2.2 tok/s | 2,261 tok/s | 3,505 ms |
| TP8×EP，TCP+图 | 87 ms | 11.4 tok/s | 2,261 tok/s | 3,628 ms |
| **TP4×PP2，RoCE+图** | **38.6 ms** | **25.9 tok/s** | **4,773 tok/s** | **1,718 ms** |

视觉 `vision_test.py` **4/4** 通过（同一配置），说明 PP 路径数值正确。

与现网对比（同为单流）：

| 模型 | TPOT | 卡数 |
|---|---|---|
| **Hy4-flash（TP4×PP2）** | **38.6 ms** | 8 |
| minimax-m3.1-flash | 31.3 ms | 16 |
| qwen3.8-flash-fp8 | 5.0 ms | 8（TP 在节点内 + mtp×3 投机） |
| deepseek-v4-1-flash | 4.4 ms | 24（TP 在节点内 + dspark×5 投机） |

结论：**TP 收进节点内是决定性的一步**（456 → 38.6 ms，11.8×）。与 MiniMax 同量级，
与 Qwen/DS 的差距剩下的是个位数毫秒级 + 投机解码（Hy4 的 DFly2 需要 `fp8_ds_mla` KV，sm80 不可用）。

## 16. 现网参数对照：我们借鉴了什么、哪些用不了（2026-11-19）

从 gpustack API 取到现网三个模型的**完整** `backend_parameters` + `env`，逐项对照：

### 16.1 已借鉴并生效（TP4×PP2 上实测）

| 参数 | 现网取值 | 我们的取值 | 效果 |
|---|---|---|---|
| `--max-num-seqs` | 16 | **16**（原 2） | 并发聚合吞吐 c=1 22.3 → c=8 94.2 → c=16 **176.4 tok/s** |
| `--compilation-config` | `max_cudagraph_capture_size: 32`（minimax） | 同 | 覆盖 1–32 各种 batch，避免反复重捕获 |
| `--max-num-queued-tokens / --max-num-queued-reqs` | 1750000 / 48 | 同 | 队列控制，长 prompt 混跑不饿死 |
| `--no-enable-flashinfer-autotune` | qwen 在用 | 同 | 避免首请求 autotune 停顿 |
| `--prefix-match-unit` | 16 | 同 | 前缀缓存粒度对齐 16 |
| `--long-prefill-token-threshold` | 8192 | 同 | 超长 prefill 分块 |
| `--enable-prefix-caching` | 显式开 | 同 | 命中 TTFT 从 3.5 s → 0.5 s |
| `--disable-access-log-for-endpoints=/metrics` | gpustack 注入 | 同 | 去掉 metrics 刷屏 |
| `OMP_NUM_THREADS` | 8 | 同 | 与现网一致 |
| `NCCL_TIMEOUT / NCCL_IB_TIMEOUT` | 3600 / 22 | 同 | 长任务不误判超时 |

调优后实测（官方 FP8 全量权重，TP4×PP2，单机 4 卡 ×2 节点）：

- 单流：TTFT 571 ms @2K、**TPOT 38.7 ms**、prefill 4,792 tok/s @8K；
- 并发：**c=1 22.3 / c=8 94.2 / c=16 176.4 tok/s**，TTFT p50 ≈ 1.25 s；
- **KV 497,073 tokens**（32K × 15.17 并发）；
- 视觉 4/4 仍通过。

### 16.2 试了但不适用 / 不兼容

| 项 | 结论 |
|---|---|
| `--block-size 128`（现网 minimax） | ❌ Hy4 必须 64：页对齐后 `block stride 1396224 != page 16896`，BLNHC 布局校验直接报错 |
| `--moe-backend=marlin`（minimax） | ❌ 该开关只对 DeepSeek-V4 生效（`config/kernel.py` 里显式限制） |
| `--swa-bounded-replay`（DS-V4.1） | ❌ 只在 `models/deepseek_v4_1/attention.py` 里实现，Hy4 走自己的 SWA 后端 |
| `VLLM_INDEXER_QUERY_SHARD=1`（DS-V4.1） | ❌ 只作用于 `v1/attention/backends/mla/indexer.py`；Hy4 是自研 `compressed_indexer.py`。**但值得移植**：它的门槛只是 `tp>1 && dcp==1 && !pcp`，Hy4 的 IndexPool 目前在 TP 上是全复制的，按查询分片可省 4× 索引器算力 |
| `--speculative-config`（qwen mtp×3 / DS dspark×5） | ❌ Hy4 的 DFly2 需要 target/draft 共享 KV 且 KV 必须 `fp8_ds_mla`，sm80 不可用 |
| `--kv-cache-dtype fp8` | ⚠️ Hy4 白名单只接受 `bfloat16` / `fp8_ds_mla`；后者在 sm80 未验证 |
| `--kv-transfer-config` Mooncake | ⏭ 需要 Mooncake 集群（现网 38 台都跑 `mooncake_client`），属基础设施级，未接入 |
| `VLLM_PLE_CPU_OFFLOAD=1`（qwen） | ➖ 等价于我们已在用的 `--engram-config '{"cpu_offload": true}'` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | ➖ 默认即 0，无差别 |
| `--reasoning-parser / --tool-call-parser` | ⏭ Hy4 有 `renderers/hy_v4.py`，接 chat API 时应配上对应 parser 名（尚未接） |

## 17. DFly2 投机解码：可行性评估（2026-11-19，未跑通）

### 17.1 先纠正一个前提：**bf16 KV 不是障碍**

`models/hy_v4/dfly_config.py:314` 的校验是：

```python
if spec.kv_cache_dtype not in (None, "auto", "bfloat16", "fp8_ds_mla"):
    raise ValueError("DFly requires BF16 or fp8_ds_mla draft MLA KV cache")
```

也就是说 **DFly 明确接受 `bfloat16`**。之前"DFly2 需要 fp8_ds_mla"的判断是错的，
不是 KV dtype 卡住，而是下面四条。

### 17.2 真正的四个阻碍

| # | 阻碍 | 证据 |
|---|---|---|
| 1 | **DFly 要求 `pipeline_parallel_size == 1`** | `models/hy_v4/dfly_config.py:200-210` 对 `data_parallel_size / pipeline_parallel_size / pcp / dcp` 全部强制 =1。也就是说 DFly 只能配 TP8×PP1（跨节点 TP），和我们当前最优的 TP4×PP2（TPOT 38.6 ms）互斥 |
| 2 | `method="dfly"` **不在 fork 的方法白名单里** | `config/speculative.py:72` 的 `SpeculativeMethod = Literal[...]` 没有 `"dfly"`（PR #68296 在第 115 行加了）；传 `--speculative-config '{"method":"dfly"}'` 直接被 pydantic 拒绝 |
| 3 | PR 的 **speculator-config 注册层在 fork 里不存在** | PR 新增 `v1/worker/gpu/spec_decode/speculator_config.py`（**497 行**，`SpeculatorConfig`/`DraftModelSpeculatorConfig`/`get_method_config_class`/`get_speculator_config_class`）与 `spec_decode/dfly/` 包；fork 里两者都没有 |
| 4 | fork 的 `config/speculative.py` **缺少 PR 依赖的钩子** | 统计：`_rope_only_sizes_cache` 0 处、`compile_factors` 0 处、`spec_pruning` 0 处（PR 的 `DraftModelSpeculatorConfig.configure()` 依赖这些） |

### 17.3 工作量与建议

- 需要移植：`speculator_config.py`（497 行）+ `dfly/config.py`（~70 行）+ `models/hy_v4/dfly_config.py`（318 行，已在外挂里）+
  `config/speculative.py` 约 10 处调用点改造（`default_model` / `configure` / `get_speculator_config_class` / 方法白名单）。
- 属于**框架级合并**：fork 的 `SpeculativeConfig` 与 PR 的注册层 API 有差异，风险高于之前所有模型侧改动，且**只有在 TP8×PP1 拓扑下才能验证**。
- **收益评估**：即便跑通，k=5、接受率 ~50% 时约 2–2.5× 加速，而 TP8×PP1 基线是 87 ms → **预计 35–40 ms，和现在 TP4×PP2 的 38.6 ms 基本打平**。
  真正有价值的是 **DFly + PP2 同时**：DFly 的 5 个 draft 目标层（25/29/33/37/41）在 PP2 下**全部落在 stage 1**，
  所以结构上并非不可能，只是被 PR 的 `pipeline_parallel_size==1` 断言挡住 —— 值得向 PR 作者确认是否可放宽。

**结论**：DFly2 不是"改个 KV dtype"就能开的开关，需要一次框架级移植；建议放在 TP4×PP2 稳定交付之后再做，
且优先争取"DFly + PP>1"的组合，否则单独 DFly 的收益会被 PP2 的收益吃掉。

## 18. Chat API 与索引器开销实测（2026-11-19）

### 18.1 Chat API：已接通并验证

`/v1/chat/completions` 现在可用，配置只需两项（chat template 走 checkpoint 自带的
`chat_template.jinja`，reasoning 拆到 `reasoning_content`）：

```
--chat-template /root/Models/Hy4-flash-FP8/chat_template.jinja
--reasoning-parser hy_v4        # reasoning/__init__.py 中已注册
```

实测（TP4×PP2，全量权重）：

| 用例 | content | reasoning_content |
|---|---|---|
| `reasoning_effort=high`："9.11 和 9.9 哪个大" | `9.9` ✅ | 192 字符完整思考过程 ✅ |
| `reasoning_effort=no_think`："巴黎是哪个国家的首都" | `法国` ✅ | 空 ✅ |
| 视觉（chat 接口传 image_url）：图中文字 | `HELLO7` ✅ | 空 |

注：**PR 里没有 tool parser**（`entrypoints/openai/tool_parsers` 无 hy_v4 条目），
所以 `--tool-call-parser` 目前不可用；工具调用格式由 `renderers/hy_v4.py` + 模板负责。

### 18.2 索引器开销：是 launch bound，不是算力 bound

在 idle 节点上按真实 shape 微基准（单卡，`bench_indexer.py`）：

| shape | `prepare_groups`（我们的 torch 版） | `topk(logits, 2048)` |
|---|---|---|
| decode tokens=1 | **2.38 ms** | 0.10 ms |
| decode tokens=6 | **2.38 ms** | 0.10 ms |
| prefill 2048 | **2.42 ms** | 0.88 ms |
| prefill 8192 | **2.42 ms** | 2.77 ms |

关键观察：**`prepare_groups` 的耗时与 token 数无关（恒定 ~2.4 ms）** —— 它由 ~30 个
小算子（每次 launch ~0.08 ms）主导，而不是 FLOPs。

这直接影响"索引器按 TP 分片"的收益判断：

- 每个 stage 有 6 个稀疏层，6 × 2.4 ms ≈ **14 ms/步**，约占我们 38.6 ms TPOT 的 **36%**；
- 但 **TP 分片只会把 FLOPs 分掉，launch 数量不变**，所以分片对这块开销几乎无效
  （DeepSeek 侧的 `VLLM_INDEXER_QUERY_SHARD` 省的是它自己重算 `q_quant` 的算力，而我们这里不是算力瓶颈）。

**正确的下一步**（成本低、收益直接）：

1. **按步缓存 index 数学**：`ends/context/req/group/member/source/block` 只依赖
   `starts/lengths/table`，同一 step 内 6 个稀疏层完全相同 → 每步算一次，per-layer 只留 gather；
   预估 2.4 ms → ~1.0 ms/层，每步省 ~8 ms（38.6 → ~31 ms）；
2. 或重试 Triton 版 prep（单 kernel，~0.05 ms），但此前它在引擎进程内会卡在 JIT，需要先解决；
3. TP 分片留到"索引器变成算力瓶颈"时再做（例如未来支持投机解码、batch 变大之后）。

## 19. DFly + PP>1 实测：卡在"模型运行器代际"，不是 PP（2026-11-19）

### 19.1 已完成的移植（4 处，全部通过）

按 PR #68296 的最小化移植（已在外挂里、**默认不激活**）：

1. 拷入 `v1/worker/gpu/spec_decode/speculator_config.py`（497 行注册层，PR 原样，只依赖 `resolve_obj_by_qualname`）；
2. 拷入 `v1/worker/gpu/spec_decode/dfly/`（`DFlySpeculatorConfig`）与 `v1/worker/gpu/sample/greedy_argmax.py`（64 行）；
3. `config/speculative.py`：方法白名单加 `"dfly"`、`__post_init__` 增加 `elif self.method == "dfly"` 分支（model 取 target checkpoint）、
   放宽第二处 `Unsupported speculative method`、新增 `speculator_config` / `uses_target_hidden_states` / `uses_aux_hidden_states` / `use_dfly`；
4. `models/hy_v4/dfly_config.py`：**放开 `pipeline_parallel_size == 1` 断言**（DFly 的 5 个 draft 目标层 25/29/33/37/41 在 PP2 下全在 stage 1，
   draft 及其 KV 都在该 stage，结构上可行）。

配置层现在完全接受 `--speculative-config '{"method":"dfly","num_speculative_tokens":5}'`（已越过 pydantic 校验、白名单、两处 dispatch）。

### 19.2 真正的拦路虎：DFly 只有 **MRv2**（model runner v2）实现

启动推进到 worker 侧后报：

```
ValueError: Unknown speculative decoding method: dfly
  at vllm/v1/worker/gpu_model_runner.py:698   ← v1 runner 的 proposer 分派
```

- PR 的 `HYV4DFlySpeculator` 继承 `v1/worker/gpu/spec_decode/speculator.py::DraftModelSpeculator`，
  依赖 `ModelState` / `InputBatch` / `DpasyncState` / `build_slot_mappings_by_layer` 等 **MRv2 基础设施**；
- 我们这条线一直跑在 **v1 model runner**（`VLLM_USE_V2_MODEL_RUNNER` 默认未设），v1 里根本没有 DFly proposer，只有 ngram/eagle/medusa/dflash/dspark 等；
- 因此要让 DFly 跑起来，必须切到 `VLLM_USE_V2_MODEL_RUNNER=1`，而那意味着**整个 sm80 外挂要在 MRv2 上重新验证**
  （我们的 TRITON_MLA_SPARSE 绑定、SWA 后端、sink、IndexPool prep、页对齐等全部是在 v1 上调通的）。

### 19.3 结论与建议

- **"DFly + PP>1" 技术上没有被 PP 挡住**（断言已放开，且层归属分析支持它）；挡住它的是 runner 代际。
- 前置条件变成：把整条链切到 MRv2 并在 sm80 上重新跑四门（预计是又一轮完整适配，不是小补丁）。
- 建议顺序：先在 **MRv2 + 当前无投机**的配置下验证 TP4×PP2 是否同样正常（这本身就能带来收益，
  `swa-bounded-replay` 等 MRv2 专享优化也才能用），再叠加 DFly。
- 保留的 scaffolding（注册层 + dfly 包 + 白名单 + PP 放宽）不改变默认行为，可随时在 MRv2 上继续。

## 20. 自写 HYV4 tool parser（2026-11-19）

PR #68296 只带了 reasoning parser，没有 tool parser，所以 `--tool-call-parser` 一直不可用。
按 checkpoint 模板里的真实格式补了一个：`vllm/tool_parsers/hy_v4_tool_parser.py`，注册名 **`hy_v4`**。

### 20.1 格式（来自 `chat_template.jinja` 第 14–20 行）

```
<tool_calls:opensource>
<tool_call:opensource>{name}<arg_key:opensource>{k1}</arg_key:opensource><arg_value:opensource>{v1}</arg_value:opensource>...</tool_call:opensource>
</tool_calls:opensource>
```

即：**不是 JSON**，而是纯文本信封包 key/value 对。解析器把信封内的成对参数还原成 dict 再 `json.dumps`
（符合 OpenAI `arguments` 是字符串的约定），信封外的文本作为 `content` 保留。

### 20.2 实现要点

- `extract_tool_calls`：正则切 `<tool_call…>` 块 → 名字取到第一个 `<arg_key…>` 之前 → 逐对抽 `arg_key/arg_value`；
- `extract_tool_calls_streaming`：只在某个 `<tool_call…></tool_call…>` 闭合时整块吐出（不做半截参数猜测），
  未闭合期间吞掉信封文本、正常文本照常流式；
- 注册：`vllm/tool_parsers/__init__.py` 的 `_TOOL_PARSERS_TO_REGISTER` 增加 `"hy_v4"`。

### 20.3 启动与实测

```
--reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4
```

| 用例 | 结果 |
|---|---|
| 非流式 + `tools=[get_weather]`，问"北京天气" | `finish_reason=tool_calls`，`get_weather({"city": "北京", "unit": "c"})` ✅ |
| 流式同一请求 | 收到 tool_call delta，`name=get_weather`，`arguments={"city": "北京"}` ✅ |
| 视觉 4/4 回归（加了 tool parser 之后） | 4/4 ✅ |
| reasoning + tools 共存 | ✅（思考照常进 `reasoning_content`） |

（注：工具参数是字符串字典，`unit: "c"` 这类值由模型自己填，不做类型强制转换——与模板约定一致。）

## 21. 按步缓存 index 数学：实现完成，但收益被 CUDA graph 吃掉了（2026-11-19）

### 21.1 做了什么

`prepare_groups`（torch 版）拆成"每步不变量 + 每层 gather"：

- 不变量：`ends/context/req/group/member/old/source/src_ok/src_idx/block/first%ring/member%ring/
  use_xyz/pos_values/req_t/logical/upper/live_t` —— 只依赖 `query_start_loc`、`seq_lens`、
  `block_table`、`positions` 与本步的 `tokens/ratio/ring/dim`；
- 缓存键用**原始张量的 id**（并在缓存里持有强引用，防止 id 被回收后复用），因此"命中"严格等价于"同一 step"；
- 命中统计（`HY4_CACHE_STATS=1`）实测：**hit 5–8 / miss 2，命中率 0.71–0.80**（每个 stage 6 个稀疏层）。

### 21.2 数值与隔离性能

- 与旧实现逐元素对比（1/6/2048/8192 token、命中与未命中路径）：**max|d| = 0.000e+00，完全一致**；
- 单函数隔离基准：**2.53–2.57 ms → 0.61–0.64 ms（约 4×）**。

### 21.3 端到端：只快了 2%（38.6 → 37.7 ms）——原因明确

| 配置 | TPOT | decode | prefill @8K | c=16 聚合 |
|---|---|---|---|---|
| 优化前 | 38.6 ms | 25.9 tok/s | 4,773 tok/s | 176.4 tok/s |
| 优化后 | **37.7 ms** | 26.5 tok/s | 4,746 tok/s | 177.5 tok/s |

原因：**开了 `FULL_DECODE_ONLY` 之后，decode 步的 Python 代码根本不重跑**——CUDA graph 回放只重放捕获到的
GPU kernel，所以 `prepare_groups` 的 CPU 侧 2.4 ms 从来就不在 decode 的关键路径上。我上一轮"每步省 8 ms"的
估算，在**图开启**的配置下不成立（在 eager 模式下才成立，那时 456 ms 的步骤里相当一部分正是这些 launch）。

### 21.4 结论

- 保留该优化：数值严格等价、隔离快 4×、CPU 压力更低（高并发下有余量），代价是一段可读的缓存代码；
- 但它**不是**当前 TPOT 的瓶颈。38 ms 现在是纯 GPU 时间（层计算 + 每层 2 次集合通信）；
  要再压只能从"减少每步 GPU 工作"下手：投机解码（DFly2，需 MRv2）、或降低每层通信量（TP 更细/单机更大）。

## 22. MRv2 通了；DFly2 卡在 FP8 128 块与 TP 分片的冲突（2026-11-19）

### 22.1 拓扑对比（全量权重，单流 greedy out=64）

| 拓扑 | TPOT | prefill @8K | c=16 聚合 | KV |
|---|---|---|---|---|
| **TP4×PP2（v1 runner，最优）** | **37.7 ms** | 4,746 tok/s | **177.5 tok/s** | 497k |
| TP2×PP4（v1） | 36.1 ms | 4,476 tok/s | 105.5 tok/s | **636k** |
| TP4×PP2 + **MRv2**（`VLLM_USE_V2_MODEL_RUNNER=1`） | 37.4 ms | 4,864 tok/s | 179.0 tok/s | 497k |

- **MRv2 在我们的 sm80 外挂上完全正常**：视觉 4/4、TPOT 与 v1 持平、prefill 略快（+2.5%）、c=16 略高。
- TP2×PP4：单流 TPOT 最好（36.1）且 KV 最多（636k），但**并发吞吐明显差**（PP 段多 → 气泡），综合不如 TP4×PP2。

### 22.2 DFly2 打通的 6 关（依次）

| # | 报错 | 处理 |
|---|---|---|
| 1 | `method` 不在 `SpeculativeMethod` 白名单 | 加 `"dfly"` |
| 2 | `num_speculative_tokens was provided but without speculative model` | `__post_init__` 加 dfly 分支（model 取 target） |
| 3 | `Unsupported speculative method: 'dfly'`（config 第二处 dispatch） | 放行 |
| 4 | `Unknown speculative decoding method: dfly`（v1 `gpu_model_runner`） | 确认 DFly 只有 MRv2 实现 → 切 `VLLM_USE_V2_MODEL_RUNNER=1` |
| 5 | `Model Runner V2 does not yet support: speculative method 'dfly'` + `parallel drafting` | MRv2 白名单加 dfly（含 parallel_drafting 判定） |
| 6 | `No module named ...online.lm_head` → 再 `Nvfp4WeightOnlyOnlineLinearMethod` 缺失 | 拷 PR 的 `online/{lm_head,fp8_requant,loading,requantization,lut_b,shared_expert}.py`，nvfp4 导入加 try/except |
| 7 | `HYV4 Flash requires FP8 128x128 weight blocks`（draft 的 ModelConfig 不带 block 信息） | 只对非 `runner=="draft"` 生效 |

### 22.3 最后一道坎（结构性，不是补丁能绕的）

```
ValueError: Weight input_size_per_partition = 288 is not divisible by
            weight quantization block_k = 128.
```

DFly 草稿模型的某个 MLA 投影输入维是 **1152**，在 **TP4** 下每卡 1152/4 = **288**，而 FP8 权重是 **128×128 块量化**，
要求分片后的输入维必须是 128 的倍数 → 288 ✗（TP8 更差：144 ✗）。

可行的组合只有：
- **TP1 / TP3**（1152/3 = 384 ✓）——TP1 装不下 160 GB 权重，TP3 与 8 卡拓扑不整除；
- 或**草稿用 bf16**（需 fp8→bf16 反量化加载，vLLM 当前不支持从 fp8 checkpoint 直接载入 bf16 draft）；
- 或上游给 draft 做块量化重排（对齐 128 的 TP 分片规则）。

**结论**：在当前"官方 FP8 checkpoint + 8×A100-40G"的约束下，DFly2 **不可达**；
这与我们之前"DFly 需要 fp8_ds_mla"的判断不同——真正的冲突是**块量化 + TP 分片整除性**。
若换 BF16 权重（16 卡）或 FP8 非块量化 checkpoint，上述第 7 关之后即可继续。

### 22.4 保留的成果

- MRv2 可用（为将来 `--swa-bounded-replay` 等 MRv2 专享优化铺路）；
- DFly 的配置层与 6 关修复全部留在分支里，换 checkpoint/拓扑即可继续；
- 默认部署仍是 **TP4×PP2 + v1 runner**（37.7 ms，c=16 177 tok/s）。

## 23. fp8 KV 实测：容量 +48%，速度略降（A100 无 fp8 MMA）（2026-11-19）

放开白名单（`models/hy_v4_flash/nvidia/model.py` 的 KV dtype 检查增加 `"fp8"`）后，
`--kv-cache-dtype fp8` 可以正常起服务并产出正确结果——**但在这套硬件上没有速度收益**。

| 指标 | bf16 KV | **fp8 KV** | 变化 |
|---|---|---|---|
| KV 容量（32K × N 并发） | 497,073 tokens | **734,705 tokens** | **+48%** |
| TPOT（单流） | 37.4 ms | 38.3 ms | −2%（略慢） |
| decode | 26.5 tok/s | 26.1 tok/s | −1.5% |
| prefill @8K | 4,746 tok/s | 4,332 tok/s | **−9%** |
| TTFT @8K | 1,687 ms | 1,894 ms | 更慢 |
| 并发 c=16 | 179.0 tok/s | 181.9 tok/s | +1.6%（持平） |
| 视觉 4/4 | ✅ | ✅ | 正确 |

**原因**：A100（sm80）**没有 fp8 矩阵乘单元**，fp8 KV 只能省 HBM 带宽，而我们的
`TRITON_MLA_SPARSE` 在 Triton 里要把 fp8 反量化成 bf16 再算，多出来的转换开销抵消甚至超过省下的带宽。
（MiniMax 的 prefill 快 5× 不能归因于 fp8 KV 本身；他们用 16 卡、block=128、且 MLA 走的是成熟 fp8 通路。）

**结论**：在本硬件上 **fp8 KV 是"容量模式"，不是"性能模式"**：

- 默认部署仍用 **bf16 KV**（TTFT/prefill 更好）；
- 需要长上下文/更大并发时用 `--kv-cache-dtype fp8`，用 2% 的 TPOT 换 48% 的 KV 容量。

## 24. 长上下文：1M 不可行，上限约 797K（fp8 KV）；实测冷请求 TTFT/TPOT（2026-11-19）

### 24.1 1M 的硬结论

把 `--max-model-len 1048576` + fp8 KV + `util 0.95` 直接开，vLLM 给出确切数字：

```
To serve at least one request with the model's max seq len (1048576),
8.65 GiB KV cache is needed, which is larger than the available KV cache
memory (6.9 GiB). Based on the available memory, the estimated maximum model
length is 796928.
```

即：**1M 需要 8.65 GiB KV，我们只有 6.9 GiB**，差 25%；这套 8×A100-40G 的**上下文上限 ≈ 796,928 tokens（fp8 KV）/ ≈ 550K（bf16 KV）**。

### 24.2 实测冷请求（自然语言中文，prefill 分块 16K/步，无前缀缓存）

| prompt tokens | TTFT | prefill 速率 | TPOT | 备注 |
|---|---|---|---|---|
| 13,398 | 7.59 s\* | 1,766 tok/s | 38.2 ms | \*首个请求含 JIT 预热 |
| 54,912 | **7.21 s** | 7,616 tok/s | 34.9 ms | |
| 109,890 | **10.22 s** | 10,753 tok/s | 33.1 ms | |
| 219,780 | **21.88 s** | 10,044 tok/s | **28.0 ms** | |

两个值得注意的规律：

1. **prefill 速率随长度上升并稳定在 ~10–10.8k tok/s**（13K 的 1.77k 是预热；55K 起就进入稳态），
   说明分块 prefill 摊薄了每步固定开销；
2. **TPOT 随上下文变长而下降**（38.2 → 28.0 ms）——因为 Hy4 是稀疏注意力（top-k 2048 groups），
   decode 只读固定的 top-k，**不随上下文线性变慢**。这正是稀疏 indexer 的价值所在。

### 24.3 按实测斜率外推 1M（假设 KV 装得下）

- TTFT ≈ 1,000,000 / 10,000 ≈ **100 s**（约 1.7 分钟）；
- TPOT ≈ **25–28 ms**（稀疏注意力下与长度基本无关）；
- 但这需要 **8.65 GiB KV**，当前硬件只有 6.9 GiB → **必须扩卡**（例如 12 卡或换 80G 卡）。

跨节点对比参考：现网 DS-V4.1 在 1M 配置下，五路报告里的 128K/256K TTFT 分别是 18.93 s / 45.06 s；
我们 110K/220K 是 10.2 s / 21.9 s，**同长度下我们是它的 1.8–2.1× 快**（但我们要的 KV 更多、上限更低）。

### 24.4 长上下文模式启动参数

```
MAX_LEN=262144 MAXBT=16384 MAXSEQS=4 UTIL=0.95 KVDTYPE=bfloat16   # 实测 KV 755,113 tokens
```
（默认部署仍是 32K × 16 并发；长上下文按上面四个变量切。）
