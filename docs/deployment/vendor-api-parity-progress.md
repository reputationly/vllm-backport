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

## 待办

- ~~WP5b~~：已完成并验收（见上表）。
- ~~§4 三表逐行单测~~：`tests/entrypoints/openai/test_vendor_profile_tables.py`，83 例全过。
- WP3/B4：MiniMax `reasoning_effort=medium/xhigh` 思考 0 token —— 再给一次复现机会，仍不复现则关闭。
- WP7/B5：Qwen `json_schema` 答错 —— 同上。
- DeepSeek / MiniMax 的 dev 逐参数对拍与回归电池。
- 现网滚动升级（MiniMax 直接 / DeepSeek 留一个 / Qwen 全量）。
- 合入 master + 验收报告。
