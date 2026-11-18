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

### 4.2 服务可用性（dummy 权重 / tiny 配置）
`Application startup complete`，`GPU KV cache size: 2,865,707 tokens`，text / chat / image 三种请求均 200。

## 5. 容量规划

| 权重 | 卡上常驻 | 最少卡数 |
|---|---|---|
| `Hy4-flash-FP8`（287 GB，Engram 127 GB 在 host） | 160.3 GB | **8×40 GB**（TP8，20 GB/卡） |
| `Hy4-flash`（BF16 441 GB） | 313.7 GB | 16×40 GB，或 4×H200 |

KV：只有 12 个稀疏层吃全上下文（约 13.8 KB/token bf16），30 个 SWA 层被 512 窗钉住。

## 6. 未完成

1. 官方权重端到端（缺 8 卡）；
2. 视觉 4/4 答案验收（同上）；
3. DFly2 投机解码（`method="dfly"`）——需要与 target 共享 KV/索引的 drafter，且 PR 要求
   `fp8_ds_mla` KV，与 sm80 的 per-tensor fp8 路径冲突，需另立项；
4. CUDA graph：目前用 `--enforce-eager`；FULL graph 下需把 `record_logical_topk_ready` 换成真 event。

## 7. 可以回馈上游的两条

1. `_align_pages_to_mla_rows`：packed 布局 + 行扁平寻址混用时，任何非行宽整数倍的稀疏缓存页面都需要对齐；
2. `triton_mla_sparse_kernel` 的 `return_lse`：sink 支持不该只有 FlashMLA 一条路。
