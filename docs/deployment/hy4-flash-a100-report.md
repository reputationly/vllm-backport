# Hy4-flash 在 A100/sm80 上的适配、验证与性能报告

> 汇总日期：2026-11-20 ｜ 分支：`hy4-flash-a100` ｜ 结论版本：commit `9a55cd3575`
>
> 本文是**结论型**文档（按主题组织，便于查阅）；按时间顺序的实验流水账另见
> `docs/deployment/hy4-flash-a100.md`（§1–§39）。

---

## 0. 一句话结论与本次决策

**Hy4-flash 已经能在本 fork 的 A100/sm80 栈上以"外挂"方式正确跑起来**（官方 268 GB FP8 权重、
chat/推理/工具调用/视觉全部通过、与参考实现逐 token 一致），但**本次不上线**：

| 判断维度 | 结论 |
|---|---|
| 功能 | ✅ 全通（文本、视觉、函数调用、思考内容、长上下文） |
| 正确性 | ✅ 视觉 4/4；贪心输出与 transformers 参考实现 / eager 基线逐字一致 |
| TPOT（decode） | ✅ **与现网 MiniMax-M3.1 持平**（31.0 ms vs 31.3 ms，同为 16 卡） |
| **短上下文 TTFT / prefill** | ❌ **落后 MiniMax-M3.1 约 5×**（prefill@8K 4,923 vs 24,981 tok/s） |
| 结论 | 效率差距集中在短上下文 prefill 的**每请求固定开销**，未解决前不上线 |

差距的定量来源见 §7.3；消除它的两条路（PIECEWISE 图捕获 / 核心路径重写）见 §9.4、§10。

---

## 1. 交付物清单

| 类别 | 内容 |
|---|---|
| 代码分支 | `origin/hy4-flash-a100`（commit 序列见 §11.1），基线为 PR #68296 的模型代码 + 本 fork 的 sm80 适配 |
| 镜像 | `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport:arm64-sm80-20261030-1209-c14d2fbf`（**未重编**，走外挂） |
| 权重 | `/root/Models/Hy4-flash-FP8`（NFS，268 GB，148 shard，全部 sha256 校验通过） |
| 启动脚本 | `docs/deployment/hy4-flash-a100/serve_compiled.sh`（编译版，当前推荐）、`serve_cg.sh`（eager 版）、`serve_kvdtype.sh`（KV dtype 可切）、`serve_pp2.sh`、`run_tp8.sh` |
| 评测脚本 | `sweep.py`/`sweep3.py`（上下文阶梯）、`ladder.py`（TTFT 线性拟合）、`floor.py`（每请求地板）、`conc.py`（并发）、`longctx.py`/`longctx2.py`（长文冷请求）、`vision_test.py`（视觉 4/4）、`smoke.py`、`chat_test.py`、`tool_test.py`、`bench_indexer.py`、`verify_cache.py` |
| 外挂目录（机器上） | `/root/hy4flash/`：`pkg_branch`（vllm overlay）、`site/tf`（transformers 5.23 shadow）、各类脚本与日志 |

---

## 2. 运行环境与部署形态

### 2.1 外挂（不重编镜像）原理

```
docker run <官方镜像> \
  -v /root/hy4flash/pkg_branch:/usr/local/lib/python3.12/dist-packages/vllm \   # overlay 整个 vllm 包
  -v /root/hy4flash:/work \
  -v /root/Models:/root/Models:ro \
  -e PYTHONPATH=/work/site/tf \                                                  # 用 5.23 的 transformers 盖掉镜像内置 5.17
  ... /work/serve_compiled.sh
```

- `pkg_branch` 是"从镜像里 `docker cp` 出来的 vllm 包 + 我们的 patch 覆盖"，因此**自带编译好的 `.so`**，
  只需同步 `.py`（`rsync --exclude='*.so'`）即可。
- 镜像内置 transformers 5.17 会把 `rope_parameters` 拍平成 float，导致 `AttributeError: 'float' object has no attribute 'get'`；
  外挂 `site/tf`（transformers 5.23.0）解决。**这是硬性前置条件**。

### 2.2 硬件与拓扑

| 项 | 值 |
|---|---|
| 卡 | A100-PCIE-40GB（sm80），每节点 4 卡 |
| 节点（本次实测用） | `gpu10`(29.165.44.30)、`gpu44`(29.165.19.7)、`gpu21`(29.165.102.196)、`gpu29`(29.165.2.118) |
| 互联 | Mellanox ConnectX-6 Dx，100 Gb/s，RoCE v2（GID index 3），`enp194s0f0np0` |
| 权重驻留 | 160.3 GB 在卡上（TP8 时 20.4 GB/卡）+ Engram 63.4B 参数在**主机内存**（~127 GB/节点） |

### 2.3 启动方式（**最容易踩的坑**）

必须照搬现网 DS-V4.1 的写法：

```bash
--distributed-executor-backend mp --nnodes N --node-rank R \
--master-addr <rank0 的 100G IP> --master-port 40091 \
--headless            # 非 0 号节点必须加
--distributed-timeout-seconds 3600 --cpu-distributed-timeout-seconds 3600
```

用 `torchrun`/`external_launcher` 会让**每个 rank 都起自己的 API server**，TP8 组退化到
**~0.1 层/秒**（表现为"请求几分钟不返回"）。这一条曾让我们误判为"模型跑不动"。

---

## 3. 模型与容量事实

### 3.1 模型结构（来自 checkpoint config）

| 项 | 值 |
|---|---|
| 规模 | 220B 总参 / ~12B 激活（MoE） |
| 层 | 42 层（1 dense + 41 MoE）；**12 层稀疏注意力**（`indexed_attention`）+ **30 层滑动窗**（`sliding_attention`, window 512） |
| hidden / expert | hidden 3072，512 routed experts + 1 shared，top-16，LatentMoE `routed_expert_hidden_size=1536` |
| 注意力 | MLA：latent 512 + rope 64（head_size 576），32/64 heads |
| IndexPool | ratio 4、16 打分头、head_dim 128、`group_topk 512`、`index_topk 2048`、页 64 |
| Engram | n-gram 哈希表挂在层 [2,12,22,32]，63.4B 参数，`cpu_offload=true`（UVA 访问） |
| 视觉 | SigLIP2 tower + `mm_input_norm` + interleaved MRoPE |
| DFly2 草稿 | 5 层，目标层 [25,29,33,37,41]（投机解码，见 §10.1） |
| 特殊 | learnable attention sink（**架构必需**，不能关）、iHC 残差流 |

### 3.2 KV 成本与容量上限（实测）

| 配置 | KV 池 | 单请求上限 |
|---|---|---|
| 8 卡 TP4×PP2，bf16 KV，util 0.95 | 755,113 tokens | ~550K |
| 8 卡 TP4×PP2，**fp8 KV**，util 0.95 | ~797K（vLLM 报"最大 796,928"） | ~797K |
| 16 卡 TP4×PP4，bf16 KV，util 0.92 | **1,160,941 tokens** | >1M（1M 配置可起服务） |
| 16 卡 TP4×PP4，fp8 KV | 1,347,995 tokens | 1M ✅ |

**1M 冷请求的实测**（16 卡，fp8 KV）：prompt 838,464 → **TTFT 92.45 s**，prefill 9,069 tok/s，
TPOT 30.5 ms 量级；按斜率外推真正 1M ≈ **100–110 s TTFT**。
8 卡上 1M **装不下**：`To serve ... max seq len (1048576), 8.65 GiB KV cache is needed, which is larger
than the available KV cache memory (6.9 GiB)`。

---

## 4. 四门验收结果

| 门 | 要求 | 结果 | 证据 |
|---|---|---|---|
| **1** | 官方 config + `--load-format dummy` 起服务、越过 FlashMLA 的 `cc∈{9,10}` 门、补 learnable sink | ✅ | 改绑 `TRITON_MLA_SPARSE`；日志 `HYV4 learnable sink enabled`（无 "sink disabled" 警告）；KV 正常分配 |
| **2** | 官方 FP8 权重加载 + 产出连贯文本（与参考 top-1 一致率 ≥90%） | ✅ | 148/148 shard；中文/英文/代码/带思考数学均连贯；606M 同权重 A/B 与 transformers 参考 **16/16 = 100%**；`seq` 提示 `' 6, 7, 8, 9, 10, '` 与 eager 逐字一致 |
| **3** | 视觉路径（tower + mm_input_norm + interleaved MRoPE + processor + VL 模型） | ✅ | `vision_test.py` **4/4**：数字 7、5 个圆、读出 "VISION OK 42"、蓝绿红顺序 |
| **4** | 单节点 4 卡 / 2 节点 TP8 实测基线 + 分支 | ✅（有取舍） | 见 §7；分支 `hy4-flash-a100` 已推送。单节点 4 卡跑全量权重不可行（160 GB 权重），用 606M 权重测通（8 token / 0.8 s） |

---

## 5. sm80 适配清单（为什么改、怎么改）

| # | 上游假设 | sm80 现实 | 我们的处理 | 验证 |
|---|---|---|---|---|
| 1 | FlashMLA 要求 `cc ∈ {9,10}` | sm80 = 8.0 | 改绑 `AttentionBackendEnum.TRITON_MLA_SPARSE`，并把 sink 计算搬进 `sm80_sink.py` | 服务能起、视觉 4/4 |
| 2 | 稀疏 MLA 只输出 attention（无 LSE） | learnable sink 需要 `sigmoid(lse − sink)` | 给 `triton_mla_sparse_kernel` 两个 kernel 加 `lse_ptr/HAS_LSE/return_lse=`，dummy lse 必须是 2D | sink 生效 |
| 3 | IndexPool 打分走 DeepGEMM `fp8_paged_mqa_logits` | Ampere 无 DeepGEMM | 用 fork 的 Triton 版 `fp8_paged_mqa_logits_triton` | 数值与参考一致 |
| 4 | 页大小按 block 对齐 | 混合页大小（MLA 1152 B/行、indexer 页、tail 页）导致 `page size is not divisible` | 新增 `_align_pages_to_mla_rows()`（`v1/core/kv_cache_utils.py`，在 `get_kv_cache_groups` 入口调用）**只允许调用一次**（重复调用会把 KV 压成 3 token） | KV 恢复正常 |
| 5 | IndexPool 用 64-state kernel 页 | 默认 block 校验拒绝 | `get_supported_kernel_block_sizes` 返回 `POOL_PAGE_SIZE=64` | 启动通过 |
| 6 | KV dtype 允许 `fp8_ds_mla` | sm80 无该后端；白名单只到 bf16 | 用 `--kv-cache-dtype bfloat16`；后续放开 per-tensor `fp8`（见 §7.4） | 通过 |
| 7 | IndexPool prep 用 Triton 内核 | 该内核在引擎进程内**卡死在 `triton _init_handles`**（独立进程 1.6 s 能编译） | 按 kernel 逐行移植成 **torch 版**（`indexpool_prepare_torch.py`），`VLLM_HY4_TRITON_PREP=1` 可切回 | tiny 权重 A/B 16/16 |
| 8 | 30 个 SWA 层各自算窗口索引 | 纯开销 | 曾做"按步缓存"，**但跨步指针复用导致输出退化**，最终**移除**（恢复上游行为） | 退化消失 |
| 9 | Engram 是全局模块 | PP 下每个 stage 都要求 stage-0 的 Engram 权重 | Engram 只在本 stage 拥有的层构建；前向 PP 化（非首 stage 取 `intermediate_tensors`，非末 stage 返回 `IntermediateTensors`）；`verify_loaded_parts` 跳过非本 stage 部件 | PP2/PP4 启动通过 |
| 10 | `tail.contiguous()` | packed 布局下 Tail 是跨大步视图，contiguous 会拷整块 KV（**9.95 GiB**） | 去掉 | PP 下不再 OOM |
| 11 | 跨节点集合通信可入图 | **RoCE/verbs 路径不能进 CUDA graph** | 要么 TCP（`NCCL_IB_DISABLE=1`），要么把 TP 收进节点内（TP4×PP2/PP4，跨节点只剩 p2p） | TP8+TCP+图 = TPOT 87 ms；TP4×PP2+RoCE+图 = 37.7 ms |
| 12 | 权重加载无需分片感知 | PP 下 stage 只见部分权重 | `load_weights` 加 PP stage 守卫 + 非本 stage/停用 Engram 权重跳过 | PP 通过 |

**两个值得回贡上游的发现**：
1. `_align_pages_to_mla_rows()`（混合页大小对齐）；
2. 稀疏 MLA 的 `return_lse`（sink 需要 LSE）。

---

## 6. 问题台账（症状 → 根因 → 修复）

### 6.1 启动/加载类

| 症状 | 根因 | 修复 |
|---|---|---|
| `AttributeError: 'float' object has no attribute 'get'` | 镜像内 transformers 5.17 拍平了 per-layer `rope_parameters` | 外挂 transformers 5.23（`PYTHONPATH=/work/site/tf`） |
| `NotImplementedError: Flash MLA requires Hopper or Blackwell` | 上游 kernel 的 cc 门 | 改绑 `TRITON_MLA_SPARSE` |
| `RuntimeError: Flash IndexPool requires DeepGEMM paged MQA` | Ampere 无 DeepGEMM | Triton 版 paged MQA logits |
| `page size is not divisible by the maximum page size` | 混合页大小 | `_align_pages_to_mla_rows`（**只能调用一次**） |
| `IndexPool must be bound as 64-state kernel pages` | block 校验 | `get_supported_kernel_block_sizes → 64` |
| `HYV4 Flash requires BF16 or fp8_ds_mla KV storage` | 模型白名单 | `--kv-cache-dtype bfloat16`（后续放开 `fp8`） |
| 一串 `ImportError/AttributeError`（`CompileFactors`、`supports_hc_sp`、`fuse_indexer_wk_weights_proj`、`SparseCacheRole`、循环 import 等） | 上游 PR 依赖若干 fork 里没有的共享符号 | 逐步 graft ~18 个 shim（记录在分支 commit 里） |
| `Marlin` 不可用：`No FP8 MoE backend supports the deployment configuration` | TP8 下专家中间维 1536/8=192，不是 128 的倍数 | 必须开 `--enable-expert-parallel`（EP8）；或改用 TP4（1536/4=384 ✓，不需要 EP） |
| PP2 报 `Missing Flash checkpoint weights: layers.21.engram_layer...` / `'PPMissingLayer' object has no attribute 'engram_layer'` | Engram 是全局模块 + 前向遍历了所有层 | 见 §5 第 9 条（stage 守卫 + PP 化前向） |
| PP 下 OOM `Tried to allocate 9.95 GiB` | `tail.contiguous()` 复制整块 KV | 去掉 contiguous |
| 请求几分钟不返回、日志每层 ~10 s | 用了 `torchrun/external_launcher`，每个 rank 都起 API server | 改 `mp + --nnodes/--node-rank + --headless` |
| `Address already in use` / 显存只剩 2.99 GiB / 起服务卡在权重加载后 | 上一轮容器未完全退出（GPU 显存未释放） | `docker rm -f` 后等 20+s 并 `nvidia-smi` 确认 0 MiB 再起 |

### 6.2 数值/正确性类

| 症状 | 根因 | 修复 |
|---|---|---|
| 输出无限重复（`的不的不 Malay Malay…`） | 我们给 SWA 窗口索引做了"按张量指针"跨步缓存，decode 时分配器复用同一地址 → 拿到上一步的索引 | **移除缓存**（恢复上游每层重算） |
| 数学题答错（47 vs 71） | 贪心 + `no_think` | 用模型卡推荐采样（`temperature=0.9, top_p=1.0`）+ thinking；已写入文档 |
| 编译后输出乱码 / 长生成退化 / `IndexKernel` device assert | 见 §8（compile 专题） | 逐个把侧效应算子做成 custom op |

### 6.3 性能类（详见 §7）

| 症状 | 根因 | 处理 |
|---|---|---|
| eager 下 TPOT 456 ms（2.2 tok/s） | 无 CUDA graph，42 层 × ~1800 次 Python launch（ARM CPU ~0.3 ms/次） | TP4×PP2 + `FULL_DECODE_ONLY` 图 → 37.7 ms；再加 compile → 30 ms |
| TP8 跨节点 decode 87 ms | 每层 2 次跨机 all-reduce（TCP ~1 ms） | 把 TP 收进节点内（TP4×PP2/PP4） |
| prefill 只有 4–5k tok/s（MiniMax 25k） | 每请求固定开销（1 token 也要 350–515 ms） | 未解决，见 §9.4 |
| `prepare_groups` 每层 2.4 ms | ~30 个小算子 launch-bound（与 token 数无关） | 按步缓存 index 数学（隔离 4×，端到端仅 2%，因为图回放不跑 Python） |
| fp8 KV 没有提速 | A100 无 fp8 MMA，Triton 里要反量化回 bf16 | 仅作为**容量模式**使用 |

### 6.4 度量口径类（教训）

| 坑 | 说明 |
|---|---|
| 并发对比必须同 `--max-num-seqs` | 曾用 `MAXSEQS=8` 测 c=16，被排队截断（TTFT 6.7 s、聚合 123 tok/s），误判"编译后变差"；同口径下编译版反而全面更优 |
| 请求 token 数要用 `usage.completion_tokens` | 用 SSE chunk 计数会在服务端合批时严重高估/低估 decode 速率 |
| 现网数据经网关 | 网关对 TTFT 约 +77 ms（见五路报告），且随生产负载波动（同一脚本两次测得 MiniMax c=16 302.8 / 176.4 tok/s） |
| 首个请求含 JIT/图捕获预热 | 测 TTFT 阶梯时要丢弃第一个点，或每组重复两次取第二次 |

---

## 7. 性能基线（全部实测）

### 7.1 8 卡（2 节点 ×4，TP4×PP2）— 主推配置

| 指标 | eager（无图） | +CUDA graph | **+CUDA graph + torch.compile** |
|---|---|---|---|
| TPOT | 456 ms →（早期 TP8×EP） | 37.7 ms | **30.0–30.6 ms** |
| decode | 2.2 tok/s | 26.5 | **32.7** |
| TTFT 地板（1 token 命中） | 515 ms | 347–357 ms | **347 ms** |
| TTFT 拟合 | 596 ms + 91.6 µs/tok | — | **352 ms + 145 µs/tok** |
| TTFT @2K | 557 ms | 433–480 ms | 433–480 ms |
| prefill @8K | 4,746 tok/s | — | **5,263 tok/s** |
| 并发 c=1 / c=8 / c=16 | 23.1 / 96.4 / 179.0 | — | **28.4 / 123.8 / 235.4** |
| TTFT p50 @c=16 | 1,225 ms | — | **789 ms** |
| KV | 497,073 | — | 484,779 tokens |

### 7.2 16 卡（4 节点 ×4，TP4×PP4，编译版）

| 指标 | 值 | 对比 8 卡 |
|---|---|---|
| TPOT | 31.0–31.6 ms | 基本不变（略差） |
| decode | 32.2 tok/s | 持平 |
| TTFT 地板 | 370 ms | 略差 |
| TTFT @8K | 1,666 ms | 略差 |
| prefill @8K | 4,923 tok/s | 略差 |
| 并发 c=1 / c=8 / c=16 | 28.1 / 87.3 / 162.0 | **明显更差** |
| KV | **1,160,941 tokens** | **2.4×** |
| 长文冷请求（eager 版实测） | 77K→9.08 s；154K→10.48 s；294K→20.26 s；703K→73.66 s | — |

**结论：加卡只换来 KV 容量，延迟与吞吐都变差**（PP4 流水更深、气泡更多，TP 宽度不变）。
要高吞吐应当用"**多个 8 卡副本做 DP**"，而不是一个大 PP4 实例。

### 7.3 其它拓扑（供选型）

| 拓扑 | TPOT | prefill | 并发 c=16 | 备注 |
|---|---|---|---|---|
| TP8×EP（2 节点）+ eager | 456 ms | 2,261 tok/s | — | 最初能跑的形态 |
| TP8×EP + TCP + 图 | 87 ms | 2,261 tok/s | — | RoCE 无法入图，只能 TCP |
| **TP4×PP2 + 图** | 37.7 ms | 4,746 tok/s | 179.0 | 好 |
| **TP4×PP2 + 图 + compile** | **30.0 ms** | 5,263 tok/s | **235.4** | **最优** |
| TP2×PP4 + 图 | 36.1 ms | 4,476 tok/s | 105.5 | 单流略好、吞吐差；KV 635K |
| TP4×PP4（16 卡）+ compile | 31.0 ms | 4,923 tok/s | 162.0 | KV 1.16M |

### 7.4 KV dtype 对照（8 卡 bf16 vs fp8）

| 指标 | bf16 | fp8 | 变化 |
|---|---|---|---|
| KV 池 | 497,073（util 0.92） | 734,705 | **+48%** |
| TPOT | 37.4 ms | 38.3 ms | −2% |
| prefill @8K | 4,746 | 4,332 | **−9%** |
| 并发 c=16 | 179.0 | 181.9 | 持平 |

**A100 无 fp8 张量核 ⇒ fp8 KV 只是容量模式**（Triton 里需反量化回 bf16，把省下的带宽吃掉）。

---

## 8. torch.compile 专题

### 8.1 为什么要做

eager 下每步 ~1800 次 Python launch，是 TTFT 地板（515 ms）与 TPOT（37 ms）的主要来源。

### 8.2 做了哪些工作（全部保留在分支）

| 项 | 内容 |
|---|---|
| 开启编译 | 给 `HYV4FlashModel` 加 `@support_torch_compile(dynamic_arg_dims=...)`。**注意**：默认的 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` 本身就会触发 `VLLM_COMPILE`，所以加了装饰器就会真的编译 |
| `hy4_engram_lookup_*` | Engram host-pinned 表 gather。**host 表必须用闭包持有**，op 只收 CUDA 张量——否则 breakable 捕获的 `weak_ref_tensor` 不支持 CPU 张量 |
| `hy4_engram_prepare_hashes` | n-gram 哈希（读 host token_map + Triton）。**必须在 `__init__` 里注册**，懒注册会撞 `torch.library.infer_schema`（dynamo MOD_SKIPLIST） |
| `hy4_indexpool_write_pool` | indexer KV 分页 scatter（`mutates_args=["cache"]`） |
| `hy4_indexpool_write_tails` | tail ring 刷新（`mutates_args=["tail"]`） |
| `hy4_prepare_groups_into` | IndexPool 索引数学（9 个预分配输出 + 精确 `mutates_args`） |
| `hy4_indexer_and_attn_*`（每层一个） | indexer 写 KV/top-k → MLA attention 读 KV 的完整序列（`mutates_args=["out"]`，闭包捕获 module，`q_c=None` 用空张量哨兵） |
| 其它 | `_STEP_CACHE` 在 `torch.compiler.is_compiling()` 时必须关闭（否则 trace 期把 fake tensor 塞进缓存）；KV 写入路径加防御性 clamp |

### 8.3 图捕获组合矩阵（全部实测）

| 组合 | 服务 | 数值 | TTFT 地板 |
|---|---|---|---|
| eager（无装饰器） | ✅ | ✅ 基准 | 515 ms |
| **compile + `FULL_DECODE_ONLY`** | ✅ | ✅ 视觉 4/4、`seq` 逐字一致 | **347–357 ms（当前默认）** |
| compile + `FULL` | ✅ | ✅ | 347–362 ms |
| compile + `PIECEWISE` | 起得来 | ❌ device assert（`torch.cuda.graphs.replay` 内） | 曾测得 **66 ms** |
| PIECEWISE + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + 5 个 op 加 `@eager_break_during_capture` | 起得来 | ❌ 同一 assert | — |
| FULL + breakable CG | ✅ | ✅（`frac` 与 eager 逐字一致） | 440 ms（更慢） |
| `mode=2`（DYNAMO_TRACE_ONCE）+ 无图 | ✅ | ✅ | 347 ms，但 decode 仅 **3.4 tok/s**，不可用 |
| `cudagraph_mode=NONE` + `mode=3` | ✅ | ✅（= eager） | 515 ms（vLLM 降级为不编译） |

### 8.4 为什么拿不到 66 ms（结论）

- 地板 = **"1 token 的 prefill"**（前缀缓存命中也必须重算最后一个 token）；
- decode 已被 FULL 图捕获（TPOT 30 ms），地板不是它；
- 66 ms 只在 **PIECEWISE 捕获了 prefill** 时出现；
- PIECEWISE 图**回放期不执行 Python**，而 Hy4 的 indexer/attention/Engram 全用 Python 侧尺寸
  （`num_actual_tokens`、`slots`）编排——捕获期被冻结、padding 行内容不受约束 → 回放时必然
  触发 `IndexKernel` 越界断言。

**要拿到 66 ms，必须把这三条路径重写成 padding-safe 的单 kernel / C++ op（回放期零 Python）。**
这是核心路径重写，不是再包几个 custom op 能解决的；本 fork 提供的图捕获开关已全部试过。

---

## 9. 与现网三模型对比（同一套脚本）

### 9.1 单流（greedy，out=64）

| 上下文 | Hy4（8 卡编译） | Hy4（16 卡编译） | Qwen3.8-flash-fp8 | MiniMax-M3.1 | DS-V4.1-flash |
|---|---|---|---|---|---|
| 520 | 511 ms | 416 ms | 474 ms | 282 ms | 338 ms |
| 2048 | 488 ms | 523 ms | 398 ms | 289 ms | 360 ms |
| 8192 | 1,596 ms | 1,666 ms | 749 ms | 328 ms | 386 ms |

| 模型 | 拓扑 / 卡数 | prefill @8K | TPOT | decode |
|---|---|---|---|---|
| **Hy4（8 卡编译）** | TP4×PP2 · 8 | 5,263 tok/s | **30.0 ms** | 32.7 tok/s |
| **Hy4（16 卡编译）** | TP4×PP4 · 16 | 4,923 tok/s | **31.0 ms** | 32.2 tok/s |
| qwen3.8-flash-fp8 | TP2×PP4 · 8 · 图 + mtp×3 | 24,981 tok/s | 5.0 ms | 200 tok/s |
| minimax-m3.1-flash | TP4×PP4 · 16 · 图，无投机 | 24,981 tok/s | 31.3 ms | 31.9 tok/s |
| deepseek-v4-1-flash | TP8×PP3 · 24 · 图 + dspark×5 | 21,267 tok/s | 4.4 ms | 227 tok/s |

### 9.2 并发（out=96，同日同脚本）

| 并发 | Hy4（8 卡编译） | Hy4（16 卡编译） | MiniMax-M3.1 |
|---|---|---|---|
| c=1 | 28.4 tok/s | 28.1 | 32.7 |
| c=8 | **123.8** | 87.3 | 109.1 |
| c=16 | **235.4** | 162.0 | 176.4 |
| TTFT p50 @c=16 | **789 ms** | 751 ms | 798 ms |
| 每卡 @c=16 | **29.4 tok/s/卡** | 10.1 tok/s/卡 | 11.0 tok/s/卡 |

### 9.3 长上下文（冷请求，中文自然语言）

| prompt | Hy4（16 卡 eager） | MiniMax-M3.1 |
|---|---|---|
| 72–77K | **9.08 s** | 9.20 s |
| 145–154K | 10.48 s | **9.84 s** |
| 277–294K | 20.26 s | **16.65 s** |
| 703K | 73.66 s | 未测 |

### 9.4 差距归因（结论）

1. **TPOT/decode 已持平**（31.0 vs 31.3 ms）——TP 收进节点内 + CUDA graph 的功劳；
2. **并发吞吐同量级**（16 卡 162 vs 176，8 卡 235 反超）；
3. **长上下文 prefill 基本持平**（77K）到略差（290K 1.2×）；
4. **短上下文 TTFT/prefill 落后 5×**（8K：1,666 ms vs 328 ms；4,923 vs 24,981 tok/s）
   —— 根因是**每请求固定开销（1 token 也要 350–370 ms）**，即 §8.4 那个只能靠 PIECEWISE 消除的部分。
   此外 MiniMax 用 16 卡（我们 8 卡时 prefill 反而更高）、`block-size=128`、以及成熟的 fp8 MLA 通路。

---

## 10. 未解决问题与路线图

### 10.1 DFly2 投机解码（阻塞）

- 打通的 6 关：方法白名单 → `__post_init__` 分支 → config 第二处 dispatch → MRv2 限定 →
  MRv2 白名单（含 `parallel_drafting`）→ 缺失的 `online/{lm_head,...}` 模块 + draft runner 的 FP8 块校验豁免；
- **最终阻塞（结构性）**：DFly 草稿的 MLA 投影输入维 **1152**，TP4 分片后 **288**，不是 FP8 **128×128** 块的倍数
  （TP8 更差：144）→ `ValueError: input_size_per_partition = 288 is not divisible by block_k = 128`；
- 可行组合只有 **TP1/TP3**（装不下 160 GB）或 **bf16 草稿**（需 fp8→bf16 反量化加载，vLLM 暂不支持）；
- 另：DFly 要求 `pipeline_parallel_size==1`（我们已放开该校验，但当前最优拓扑是 PP2/PP4）。

### 10.2 PIECEWISE / 66 ms 地板（阻塞，见 §8.4）

需要 padding-safe 的核心路径重写：`compressed_indexer`（含 `prepare_groups`、`fwht_quantize`、
`gated_pool`、`rotate_indexer`）、`_indexer_and_attn`、Engram 哈希/查表，全部要在回放期零 Python。

### 10.3 其它遗留

| 项 | 状态 |
|---|---|
| 编译版 PP4 首次起服务"卡在权重加载后" | 实为上一容器未释放显存（脏 GPU），清理后正常；**非** PPD4 死锁 |
| 并发口径 | 所有吞吐对比必须固定 `max-num-seqs`/`max-num-batched-tokens` |
| 长上下文 | 32K 是默认；262K 需 `MAX_LEN=262144 MAXBT=16384 MAXSEQS=4 UTIL=0.95`；1M 需 16 卡 + fp8 KV |
| `--block-size` | Hy4 只能 64（页对齐后 BLNHC 布局校验会拒绝 128） |
| `--moe-backend=marlin`、`--swa-bounded-replay` | 均为 DeepSeek 专属开关，Hy4 不适用 |
| `VLLM_INDEXER_QUERY_SHARD` | 只作用于 fork 的 `mla/indexer.py`，我们的 `compressed_indexer` 未接；且实测索引器是 launch-bound（非算力），分片收益低 |
| 工具调用 | 已自写 `hy_v4` tool parser（`tool_parsers/hy_v4_tool_parser.py`），非流式/流式均验证 |
| 待办建议 | ① 修 padding-safe 以拿 66 ms；② 或换 H100/H200（piecewise 与 fp8 原生支持）；③ 或换 BF16/非块量化 checkpoint 以叠加 DFly2 |

---

## 11. 复现手册

### 11.1 分支提交（`hy4-flash-a100`）

```
9a55cd3575 docs: exhaustive capture-mode matrix
f21aa36bad docs: correct the concurrency claim (compiled beats eager everywhere)
0c9692ef88 docs: five rounds of PIECEWISE attempts
2478194a32 feat: torch.compile runs correctly (1.45x TTFT floor, 1.25x TPOT)
84c1a39776 feat: prepare_groups as a pre-allocated custom op
6e381ee735 docs: OOB index under compile
0a755a81e3 feat: indexer+attention as an opaque custom op (vision 2/4)
533f91f4f5 feat: opaque KV writers
0a755a81e3 …（更早的 stage A/B、launch 修复、对比数据等）
```

### 11.2 起服务（8 卡 / 2 节点，编译版=当前最优）

```bash
# 节点 0
NODE_RANK=0 HOST_IP=29.165.44.30 \
  TP=4 PP=2 MAX_LEN=32768 MAXSEQS=16 MAXBT=8192 UTIL=0.92 BS=64 \
  KVDTYPE=bfloat16 CGMODE=FULL_DECODE_ONLY COMPILE_MODE=3 \
  bash docs/deployment/hy4-flash-a100/serve_compiled.sh
# 节点 1 同理，NODE_RANK=1 HOST_IP=<本节点 100G IP>
```

- 高吞吐/eager 版：去掉装饰器（或 `COMPILE_MODE=0`），其余不变；
- 16 卡：`NNODES=4 TP=4 PP=4`；
- 长上下文：`MAX_LEN=262144 MAXBT=16384 MAXSEQS=4`；1M 需 16 卡 + `KVDTYPE=fp8`；
- **起服务前务必确认 4 张卡 `0 MiB`**，否则 worker 会因显存不足退出、其余节点挂起等待。

### 11.3 验收与基线

```bash
python3 sweep3.py <base>/v1 hy4-flash "" 512,2048,8192 64     # 上下文阶梯：TTFT/prefill/TPOT
python3 ladder.py  <base>/v1 hy4-flash "" 512,2048,8192       # 拟合 TTFT = a + b·tokens
python3 floor.py                                              # 每请求地板（1/16/520 token，冷+命中）
python3 conc.py                                               # c=1/8/16 聚合吞吐与 TTFT
python3 longctx2.py <base>/v1 hy4-flash "" 92136,184272 64    # 长文冷请求
python3 vision_test.py --base <base> --model hy4-flash        # 视觉 4/4
python3 chat_test.py                                          # 思考拆分 + 视觉（chat 接口）
python3 tool_test.py                                          # 函数调用（流式/非流式）
```

### 11.4 排障关键词

| 日志关键字 | 含义 |
|---|---|
| `learnable sink enabled` | sink 生效（**必须有**） |
| `sink disabled` / `does not apply` | sink 没生效 → 输出会错 |
| `HY4_CACHE hit=/miss=` | 按步索引缓存命中率（需 `HY4_CACHE_STATS=1`） |
| `Compiling a graph for compile range` | torch.compile 真的在工作 |
| `saved AOT compiled function` | AOT 缓存写入成功 |
| `model is not torch-compiled` | 装饰器缺失（编译没生效） |
| `GPU KV cache size: N tokens` | KV 池容量（异常小 = 页对齐/内存问题） |
| `IndexKernel.cu ... index out of bounds` | 图回放期索引越界（padding-safe 问题） |
| `To serve at least one request with the model's max seq len` | KV 装不下该上下文 |

---

## 12. 附：一句话经验总结

1. **多节点必须 `mp + --headless`**，否则每 rank 一个 API server，慢到像"跑不动"；
2. **TP 收进节点内**（TP4 + PP）是 decode 提速的关键；RoCE 不能进 CUDA graph，TCP 可以；
3. **A100 没有 fp8 算力**：fp8 KV 只增容量不提速；fp8 权重必须用 Marlin(W8A16) 并注意 128 整除；
4. **Hy4 的 per-request 固定开销**（≈350 ms，1 token 也要）是短上下文落后的唯一根因，
   而它只能靠 PIECEWISE 图捕获消除，PIECEWISE 又要求 padding-safe 的核心路径——这是剩余工作；
5. **任何吞吐对比都要锁定 `max-num-seqs` / `max-num-batched-tokens`**，否则结论会反转。
