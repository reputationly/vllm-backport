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

## 待办

- WP5b：Qwen 正文上限 A（`max_tokens` 只限正文）+ 达到 A 时 `engine_client.abort`、`finish_reason=length`。
- WP3/B4：MiniMax `reasoning_effort=medium/xhigh` 思考 0 token 的复现与修复。
- WP7/B5：Qwen `json_schema` 答错复现。
- §4 三表逐行单测（当前只有 4 条冒烟）。
- DeepSeek / MiniMax 的 dev 逐参数对拍与回归电池。
- 现网滚动升级（MiniMax 直接 / DeepSeek 留一个 / Qwen 全量）。
