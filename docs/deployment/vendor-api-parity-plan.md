# 自部署三模型对齐官方接口语义：收尾计划

> 分支 `feat/vendor-api-parity`（基线 `origin/master`）
> 规范：`vendor-api-parity-design.md`（464 行，§4 三张表是唯一判据）
> 进度：`vendor-api-parity-progress.md`
> 建立：2026-11-09

## 0. 目标

以**官方渠道的实际行为**为唯一标准，把三个自部署模型的接口语义对齐到"同一份请求打过去，状态码 / 生效参数 / 输出形态一致"。

官方基线取 MaaS 网关（`https://maas.ovaijisuan.com/v1`）：

| 我们的模型名 | 官方渠道名 |
|---|---|
| `deepseek-v4-1-flash` | `deepseek-v4-1-flash-261017` |
| `qwen3.8-flash-fp8` | `Qwen3.8-Flash` |
| `minimax-m3.1-flash` | `MiniMax-M3.1-Flash` |

2026-11-09 复验三个渠道均可达（`max_tokens=32` 短请求全 200）。网关会吞掉部分字段（`thinking_budget`、`preserve_thinking`、`repetition_penalty`、`min_p`），这几项只能对**自部署直连**验收。

## 1. 已完成（有证据）

| WP | 内容 | 证据 |
|---|---|---|
| WP1 | `VendorProfile`/`VendorState`/`SamplingInputs` 框架、wrap validator 闸门、模板与采样接线、三个开关、`VLLM_MAX_STOP_STRINGS>=64` 启动检查 | 4 条冒烟单测；缺省 4 时启动即报错 |
| WP2 | DeepSeek 画像 + B1（`tool_choice=none` 删 tools）+ B3 | 代码就绪 |
| WP3 | MiniMax 画像 + B1（`tool_choice` 改写成 `auto`） | 代码就绪 |
| WP4 | Qwen 画像 + B1 + B2 + `preserve_thinking` | dev 直连 **§2 表 12/12 与官方一致** |
| WP5a | `reasoning_config.enabled` / `thinking_token_budget` 对 qwen4_exp | `thinking_budget=64` → `reasoning_tokens=62` |
| WP5b-1 | Qwen `max_tokens` 只限正文，引擎上限加思考预算 | `max_tokens=64` → completion 470 / reasoning 154 / answer 316 |
| WP5b-2 | 达正文预算时 `engine_client.abort`，`finish_reason="length"`（流式 + 非流式） | **待验收**，见 §2.1 |

实施期抓出并修掉的 4 个只有端到端才暴露的 bug：`ACTIVE_PROFILE` 按值导入导致闸门失效；`_require_range` 缺 `high_open` 导致 Qwen 全量 400；`self.thinking_budget` 非 pydantic 字段导致全量 500；`top_p=0` 通过闸门但被 `SamplingParams` 拒绝。

## 2. 剩余工作包

### 2.1 WP5b 验收（P0，唯一阻塞项）

集群 `pv6`（gpu43 = r0，gpu44 = r1，Qwen TP2×PP4）已用改后文件重建。

**验收判据（设计 §7.2）**：Qwen 设 `max_tokens=32`，问一个需要长回答的问题，**流式与非流式各一次**：

- `finish_reason == "length"`；
- 正文 token 数 ≤ 32+10；
- `usage.completion_tokens - reasoning_tokens ≤ 42`。

最后一条是关键：正文被截了但 `completion_tokens` 远大于"思考+42"，说明引擎没停，判失败。

**同批顺带验**：
- 非思考模式 `max_tokens=32` → 引擎上限应恰好 32（本轮修掉了"非思考也加思考预算"的漏洞）；
- `max_completion_tokens=32` → 不启用正文预算（D2 规定它限思考+正文）；
- 不传 `max_tokens` → 行为与改动前一致。

### 2.2 §4 三张表逐行单测（P1）

现在只有 4 条冒烟。按设计 §7.1 扩展 `tests/entrypoints/openai/test_vendor_sampling_policy.py`：

- 用 fixture 设置/还原 `ACTIVE_PROFILE`，走 `ChatCompletionRequest.model_validate()` + `to_sampling_params()` / `resolved_template_kwargs()`，不起服务；
- §4.1 / §4.2 / §4.3 **每张表每一行至少一条**：400 的断言抛 `VLLMValidationError` 且 `parameter` 正确；忽略的断言生效值为表中值；映射的断言映射结果；
- 思考判定优先级：Qwen `thinking.type=enabled` + `enable_thinking=false` → 开；DeepSeek `thinking.type=disabled` + `reasoning_effort=high` → 关；
- 无画像模型完全不变；`VLLM_VENDOR_API_PROFILE=0` 时闸门不报错；白名单生效。

### 2.3 三模型逐参数对拍（P0，主验收）

对每个模型跑 §4 对应表，**官方渠道 与 自部署直连 并排**，单并发串行。

断言粒度：HTTP 状态一致；400 时 `message` 含参数名；200 时看可观测量（`n_choices`、有无 logprobs 及条数、思考是否为 0、`tool_calls` 数与 `finish_reason`、content 是否合法 JSON、是否以 `\n` 开头、`stop` 是否截断）。

状态：

| 模型 | 状态 |
|---|---|
| Qwen3.8-Flash | **已完成**，§2 表 12/12 一致 |
| DeepSeek-V4.1-Flash | 未做 |
| MiniMax-M3.1-Flash | 未做 |

另需一条 HTTP 层断言锁住 400 通路：DeepSeek 发 `temperature=2.5` → `status==400` 且 `error.param=="temperature"`、`error.type=="BadRequestError"`。闸门在 pydantic 阶段抛 `VLLMValidationError`（不是 `ValueError`，不会被包成 `ValidationError`），由 `serve/exception_handling/` 映射成带 `param` 的 400，这条只能 e2e 锁。

### 2.4 回归电池（P0）

| 电池 | 门槛 |
|---|---|
| 客户 RAG，T=1.0/1.8，各 12 次，三模型 | 异常 0/12，空 content 0 |
| DeepSeek 关思考惩罚参数（`nt_freq0.5_P4` 等 3 臂，6 次） | 格式合格 ≥5/6（修复前 0/6） |
| Qwen 困难题（`count200`/`hop4`/`kv12800`） | 截断 ≤1/30（修复前 10/30），正确率 ≥ 官方−2 |
| Qwen 思考 T=1.8 RAG | 5 类全覆盖 ≥11/12 |
| 基础能力 28 题 | 与修复前持平（84/84 量级） |
| DeepSeek 缺省 `max_tokens` | 关思考长文在 8192 处 `length`（预期行为变化，记录在案）；开思考不提前截断 |
| 单测全量 | `pytest tests/entrypoints/openai/ -k "sampling or protocol or chat"` 全过 |

硬性约束：对自部署端点只能 `WORKERS=1` 单并发串行；只在 dev 机测，现网 `llm01–llm26` 只读。

### 2.5 现网滚动升级（P0）

镜像沿用 `arm64-sm80-20261108-0340-f1a14bb7-csrc`（digest `sha256:8695df29c91b…`），gpustack 版本 `rep-backport-csrc-custom`。

顺序：**MiniMax 直接升**（1 副本）→ **DeepSeek 删一个重建，起来一个再删剩下的**（6 副本）→ **Qwen 全量升**（3 副本）。

gpustack 注意：只改 `backend_version` **不会**重启实例，必须 `replicas 0 → N`。版本名必须 `-custom` 结尾。

回滚：`backend_version` 改回 `rep-backport-sampling-fix-custom`（或 `reputationly-backport-custom` / `reputationly-backport-m31-custom`）+ 0→N 重新部署。

### 2.6 合入 master + 收尾（P1）

- `feat/vendor-api-parity` → `master`；
- 更新 `vendor-api-parity-progress.md` 为终态；
- 出验收报告（对齐结果表、回归数据、上线记录、回滚手册）；
- 设计文档从 `vllm-backport` 树搬进 `vllm-merge` 的 `docs/deployment/` 一并提交。

## 3. 明确放弃的项

| 项 | 现象 | 处置 |
|---|---|---|
| **B4** | MiniMax `reasoning_effort=medium/xhigh` 时思考 0 token | 单个样本，两次尝试未复现。**再给一次机会，仍不复现则记为"未复现，不阻塞"并关闭**。§4.3 表按官方文档（缺省 max、minimal→low）实现，不为此改动代码 |
| **B5** | Qwen `json_schema` 答错（`{"v":2}` vs `391`） | 单个样本，属模型能力问题而非接口语义问题。**同上，一次机会后关闭** |

判据：这两项都不属于"同一份请求两边返回不同状态码 / 不同生效参数"，不影响接口对齐目标。若复现且表现为接口层差异，再单独立项。

## 4. 执行顺序

```
2.1 WP5b 验收  ──┐
                 ├─→ 2.3 三模型对拍 ──→ 2.4 回归电池 ──→ 2.5 现网滚动升级 ──→ 2.6 合入 master
2.2 §4 逐行单测 ─┘        （B4/B5 复现尝试穿插在此，各一次机会）
```

2.2 可与 2.1 并行（纯单测，不占 GPU）。2.3 里 DeepSeek 与 MiniMax 的对拍需要另起 dev 集群（`pv7`/`pv8`），注意 dev 空闲机只有 gpu41–gpu48。

## 5. 风险

- **GPU 排队**：dev 空闲机有限，Qwen 占 2 节点（gpu43/44）。DeepSeek TP8×PP2 要 4 节点，MiniMax 单节点。对拍阶段要串行起停，每次起停约 10–15 分钟。
- **网关字段丢失**：`thinking_budget` / `preserve_thinking` / `repetition_penalty` / `min_p` 经网关不可见，只能直连验；报告里要标注哪些行是"直连验、网关不可比"。
- **单样本误判**：B4/B5 的教训。凡"只跑了一次就下结论"的差异，一律先复现 3 次再定性。
- **现网不可回退点**：DeepSeek 6 副本滚动期间必须保证 ≥1 个存活；任何时刻不要同时清空。
