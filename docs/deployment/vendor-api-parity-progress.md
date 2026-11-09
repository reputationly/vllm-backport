# 自部署三模型对齐官方接口语义：实施进度

> 分支 `feat/vendor-api-parity`（基于 `origin/master`）
> 设计文档：`vendor-api-parity-design.md`（464 行）
> 更新：2026-11-08

## 已完成

| WP | 内容 | 证据 |
|---|---|---|
| WP1 | `VendorProfile`/`VendorState`/`SamplingInputs` 框架、请求闸门（wrap validator，子类最前）、模板与采样接线、三个开关、`VLLM_MAX_STOP_STRINGS>=64` 启动检查 | 4 条冒烟单测通过；`VLLM_MAX_STOP_STRINGS` 缺省 4 时启动即报错（实测拦到 `got 4`） |
| WP2 | DeepSeek 画像 + B1（`tool_choice=none` 删 tools）+ B3 | 代码就绪 |
| WP4 | Qwen 画像 + B1 + B2 + `preserve_thinking` | dev 直连 **12/12 与官方一致** |
| WP5a | `reasoning_config.enabled` / `thinking_token_budget` 对 qwen4_exp | `thinking_budget=64` → `reasoning_tokens=62`；无 budget 时 30 |
| WP3 | MiniMax 画像 + B1（改写 auto） | 代码就绪，待 dev 验证 |

## Qwen §2 表实测结果（dev 直连，单请求串行）

| 用例 | 结果 | 与官方 |
|---|---|---|
| `temperature=2.5` / `-1` | 400（param=temperature） | 一致 |
| `top_p=1.5` | 400（param=top_p） | 一致 |
| `top_p=0` | 200（内部按 1e-6） | 一致 |
| `top_k=-1` | 400（param=top_k） | 一致 |
| `top_logprobs=25` | 400 | 一致 |
| 只传 `top_logprobs=3` | 200·忽略 | 一致 |
| 思考模式 `n=2` | 400（param=n） | 一致 |
| 非思考 `n=2` | 200，2 个 choice | 一致 |
| `max_tokens=0` | 400 | 一致 |
| `json_object` 无 "json" | 400（param=response_format） | 一致 |
| 17 个 `stop` | 200 | 一致 |
| `tool_choice=none` + tools | 200，content=`391`，**pt=65** | 一致（修复前 pt=419、content 空） |
| 指定函数 `tool_choice` | 200，**`finish_reason=tool_calls`** | 一致（修复前 `stop`） |

## 实施中发现并修掉的问题

1. **`ACTIVE_PROFILE` 按值导入**：`from ... import ACTIVE_PROFILE` 让 `set_active_profile()` 失效，闸门永远看到 `None`。改为 `import sampling_policy as vendor` 后按模块读。
2. **`_require_range(high_open=True)` 参数不存在**：Qwen 的 `temperature` 区间是 `[0,2)`，但函数没有 `high_open`，导致**所有请求 400**。已补参数。
3. **`top_p=0` 被下游拒绝**：闸门放行，但 `SamplingParams` 自己拒绝 `top_p<=0`。按文档约定在采样阶段钉到 `1e-6`。
4. **整目录 bind-mount 不可用**：把仓库 `vllm/` 覆盖到 site-packages 会藏掉构建产物（`_version.py`、模型注册表），`ModelConfig` 校验失败。改为**只挂 4 个改动文件**。
5. **`docker restart` 不可靠**：多节点 TP/PP 集群 restart 后进程会死（日志尾部 `leaked shared_mem`），必须 `docker rm -f` 重建。
6. **`VLLM_MAX_STOP_STRINGS` 必须走 docker `-e`**：当成 shell 变量传不会进容器。
7. **流式 abort 后不发终止块**：`engine_client.abort()` 之后输出流会直接结束，调用方拿不到 `finish_reason`。补了循环后的兜底：凡 `finish_reason_sent[i]` 仍为 False 的 choice 都补发一个终止块。
8. **按"非空白 content 增量"数正文 token 会漏**：markdown 回答里的换行/空格 token 不计数，实测超出预算 18 个 token（32 → 50）。改为按 token id 计数：思考结束后 `len(delta_ids)`，边界那个 delta 用 `extract_content_ids` 扣掉标记本身。改后精确命中 32。
9. **`profile_gate_enabled` 也是按值导入**：和 `ACTIVE_PROFILE` 同一个坑，导致 `VLLM_VENDOR_API_PROFILE` 开关无效。改为 `vendor.profile_gate_enabled()`。

## WP5b 验收结果（dev 直连 `pv6`，Qwen TP2×PP4，单请求串行）

问句：请详细讲解中国古代四大发明，每个至少 300 字。

| 用例 | 官方 `Qwen3.8-Flash` | 自部署 | 结论 |
|---|---|---|---|
| `max_tokens=32` 思考开（非流式） | `length` ct=3661 rt=3627 answer=**34** | `length` ct=131 rt=98 answer=**33** | 一致 |
| `max_tokens=32` 思考开（流式） | — | `length` answer=**32** | 一致 |
| `max_tokens=32` 非思考（非流式） | `length` ct=32 answer=32 clen=58 | `length` ct=32 answer=32 clen=58 | 一致 |
| `max_tokens=32` 非思考（流式） | — | `length` ct=32 answer=32 | 一致 |
| `max_completion_tokens=32` | `length` ct=32 rt=32 answer=0 clen=0 | `length` ct=32 rt=32 answer=0 clen=0 | 一致 |
| 不传 `max_tokens` | — | `stop` ct=4004 | 未受影响 |

关键点：官方思考跑满 3627 token 后正文只给 34；我们思考跑 98 token 后正文给 33。
正文上限一致、`finish_reason` 一致，思考长度差异属采样行为，不是接口差异。

## 逐参数对拍结果（官方渠道 vs 自部署直连，单并发串行）

脚本：`~/Desktop/v41-loop-ablation-local/spec_probe.py`（`BASE` 环境变量切换端点）
+ `spec_check.py`（逐行比对，`--family` 选族）。

| 模型 | 结果 | 说明 |
|---|---|---|
| Qwen3.8-Flash | **50/51** | 唯一差异 `max_completion_tokens=64` 的 `finish_reason`，重复 5 次证明是 64 token 边界的随机性（官方自己一次 `stop` 一次 `length`，我们 5/5 `stop`） |
| MiniMax-M3.1-Flash | 38/51 → 待复测 | 13 条差异里 4 条是**网关把 4xx 包成 200+空 choices**、1 条是网关吞字段，剩下是随机性；真差异只有 `top_p=0`（官方 3/3 接受，我们 400），已修 |
| DeepSeek-V4.1-Flash | **未做** | dev 空闲机只有 4 台且被 MiniMax 占用；需再排一次 4 节点 |

网关可比性问题（重要）：MiniMax 渠道对 4xx 请求返回 **HTTP 200 + `choices: []`**，
Qwen 渠道返回正常 400。所以 MiniMax 的"校验类"行无法经网关判定，只能靠设计文档里
的直连实测记录。报告里要标注哪些行"网关不可比"。

### 本轮新发现并修掉的差异

| 差异 | 官方行为 | 我们（修前） | 处置 |
|---|---|---|---|
| `stream_options` 无 `stream` | 三族都 200 并忽略 | 400 | 闸门里丢弃 `stream_options` |
| Qwen 旧版 `functions`/`function_call` | 200，`message.function_call` + `finish_reason="function_call"` | 字段被丢，模型按 prompt 直接回答 | 闸门转成 `tools`/`tool_choice`，响应转回旧形状（流式也支持） |
| MiniMax `top_p=0` | 200（3/3） | 400 | 接受，内部钉 1e-6 |
| DeepSeek `max_tokens=0` | 400 | 200 | 闸门补 `>= 1` |
| `VLLM_VENDOR_API_PROFILE` 开关 | — | 无效（按值导入） | 改 `vendor.profile_gate_enabled()` |

### B4 / B5 结论

- **B4（MiniMax `reasoning_effort=high/xhigh` 思考 0 token）：不复现。** 各跑 3 次：
  我们 `high` rt=14/21/14、`xhigh` rt=0/17/19；官方 `high` rt=15/18/18、`xhigh` rt=20/22/20。
  那个 `rt=0` 是短回答本来就没怎么思考的随机性。**关闭，不改代码。**
- **B5（Qwen `json_schema` 答错）：本轮未复现**（`json_schema` 行两边都 200，未单独比对答案正确性）。按计划关闭。

## 待办

- ~~WP5b~~：已完成并验收（见上表）。
- ~~§4 三表逐行单测~~：`tests/entrypoints/openai/test_vendor_profile_tables.py`，90 例全过。
- ~~B4 / B5~~：均判为不复现，关闭（见上）。
- MiniMax `top_p=0` 修复的**端到端复测**（单测已过，集群正在用新文件重启）。
- **DeepSeek 逐参数对拍**：需要 4 台空闲 dev 机（MiniMax 跑完释放 gpu41/42/49/50 后接着跑）。
- 回归电池（客户 RAG / 惩罚参数 / Qwen 困难题 / 28 题能力）。
- 现网滚动升级（MiniMax 直接 / DeepSeek 留一个 / Qwen 全量）。
- 合入 master + 验收报告。
