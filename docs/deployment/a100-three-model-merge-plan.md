
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
