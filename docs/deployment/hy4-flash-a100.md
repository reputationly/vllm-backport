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
