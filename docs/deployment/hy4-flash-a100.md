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


## 9. 实测状态（2026-11-19）

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
