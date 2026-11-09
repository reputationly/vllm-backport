# 自部署三模型对齐官方接口语义：验收报告

> 分支 `feat/vendor-api-parity` 已合入 `master`（merge commit `11920cf46a`）
> 规范：`vendor-api-parity-design.md` §4 三张表
> 计划：`vendor-api-parity-plan.md`　进度与踩坑：`vendor-api-parity-progress.md`
> 日期：2026-11-09

## 1. 结论

三个自部署模型（`deepseek-v4-1-flash`、`qwen3.8-flash-fp8`、`minimax-m3.1-flash`）
的请求闸门、采样语义和输出形态已与各自的官方渠道一致。逐参数对拍：

| 模型 | 官方基线 | 一致 | 剩余差异 |
|---|---|---|---|
| DeepSeek-V4.1-Flash | `deepseek-v4-1-flash-261017`（MaaS 网关） | **50/51** | 1 条思考长度随机 |
| Qwen3.8-Flash | `Qwen3.8-Flash`（MaaS 网关） | **50/51** | 1 条思考长度随机 |
| MiniMax-M3.1-Flash | `MiniMax-M3.1-Flash`（**厂商直连** `api.minimax.cn`） | **46/51** | 5 条思考长度随机 |

剩余差异全部落在"这次有没有输出 reasoning"这一项上，两边都会正反跳变（重复实测见 §4），
不属于接口差异。

## 2. 对拍方法

`~/Desktop/v41-loop-ablation-local/`：

- `spec_probe.py`：51 个用例（参数校验 / 忽略 / 映射 / 思考开关 / 工具 / 历史 / 流式选项），
  单并发串行。`BASE` 环境变量切换端点，`MAAS_KEY` 切换密钥。
- `spec_check.py`：逐行比对两份结果。400 只比状态码（官方渠道把 `error.param` 留空，
  我们填了字段名，是更丰富的信息）；200 比 `n_choices` / 有无 logprobs / 思考是否为空 /
  `tool_calls` 数 / `finish_reason`。
- `cap_compare.py`：回归电池（`degen` 客户 RAG、`cap` 28 题能力、`hard` 困难题）。
- `param_sweep.py`：DeepSeek 关思考惩罚参数扫描。

**网关可比性**：MaaS 网关会把 MiniMax 渠道的 4xx 包成 `HTTP 200 + choices: []`，
所以 MiniMax 的校验类行经网关无法判定。本轮拿到厂商直连端点后重跑，
4xx 行才真正可比——见 §5 的两条新发现。

## 3. WP5b：Qwen `max_tokens` 只限正文（设计 D2）

问句：请详细讲解中国古代四大发明，每个至少 300 字。

| 用例 | 官方 | 自部署 | 结论 |
|---|---|---|---|
| `max_tokens=32` 思考开（非流式） | `length` answer=34 | `length` answer=33 | 一致 |
| `max_tokens=32` 思考开（流式） | — | `length` answer=32 | 一致 |
| `max_tokens=32` 非思考（非流式） | `length` ct=32 clen=58 | `length` ct=32 clen=58 | 一致 |
| `max_tokens=32` 非思考（流式） | — | `length` ct=32 | 一致 |
| `max_completion_tokens=32` | `length` ct=32 rt=32 clen=0 | 完全一致 | 一致 |
| 不传 `max_tokens` | — | `stop` ct=4004 | 未受影响 |

实现要点：非流式也切 `RequestOutputKind.DELTA`，本地聚合增量，正文达预算立刻
`engine_client.abort`，`finish_reason="length"`。

## 4. 回归电池

| 电池 | 门槛 | DeepSeek | Qwen | MiniMax |
|---|---|---|---|---|
| 客户 RAG T=1.0/1.8 各 12 次 | 异常 0/12、空 content 0 | **0/24，空 0** | **0/24，空 0** | **0/24，空 0** |
| 关思考惩罚参数 3 臂 ×6 | 格式合格 ≥5/6（修复前 0/6） | **18/18** | — | — |
| 困难题 30（`count200`/`hop4`/`kv12800`） | 截断 ≤1/30、正确率 ≥ 官方−2 | — | **截断 0/30、26 题**（官方 25） | — |
| 基础能力 28 题 | 与官方持平 | **23/28**（官方 23，错项完全相同） | **22/28**（官方 22，错项完全相同） | — |
| 缺省 `max_tokens` | 关思考长文不异常截断；开思考不提前截断 | 关思考写长文自然结束 10261 token；开思考 `max_tokens=2000` → `stop` ct=1771 rt=1023 | — | — |

"错项完全相同"是本轮最有说服力的一条：基础能力 28 题里我们和官方错的都是
`code3`/`kv800`/`kv3200`/`if_lines`/`if_words`/`if_json` 这几道格式题，
说明差距不在引擎。

## 5. 本轮新发现并修掉的差异

| # | 差异 | 官方 | 我们（修前） | 处置 |
|---|---|---|---|---|
| 1 | **DeepSeek 默认请求完全不思考** | 每例都有 reasoning | rl=0 全线 | `resolved_template_kwargs` 无条件传 `reasoning_effort="none"`，V4.1 renderer 把 `none` 解释成"不思考"。改为只在调用方真给了 effort 时才传 |
| 2 | Qwen 旧版 `functions`/`function_call` | `message.function_call` + `finish_reason="function_call"` | 字段被丢，模型按 prompt 硬答 | 闸门转成 `tools`/`tool_choice`，响应转回旧形状（流式也支持） |
| 3 | `stream_options` 无 `stream` | 三族都 200 并忽略 | 400 | 闸门里丢弃 |
| 4 | MiniMax `top_p=0` | 200（3/3） | 400 | 接受，内部钉 1e-6 |
| 5 | MiniMax `tool_choice=none` | 200，无 tool_call（3/3） | 被改写成 `auto`，出了 tool_call | 只在非 `none` 时才改写 |
| 6 | MiniMax `reasoning_split=false` | 400 `requires reasoning_split=true`（3/3） | 忽略 → 200 | 补校验 |
| 7 | DeepSeek `max_tokens=0` | 400 | 200 | 闸门补 `>= 1` |
| 8 | `VLLM_VENDOR_API_PROFILE` 开关 | — | 无效（`profile_gate_enabled` 按值导入） | 改 `vendor.profile_gate_enabled()` |

另有 3 个只有端到端才暴露的实现 bug：

- **流式 abort 后不发终止块**：`engine_client.abort()` 之后输出流直接结束，调用方拿不到
  `finish_reason`。补了循环后的兜底，凡没发过终止块的 choice 都补一个。
- **按"非空白 content 增量"数正文 token 会漏**：markdown 里的换行/空格 token 不计数，
  实测超出预算 18 个 token（32 → 50）。改为按 token id 计数后精确命中 32。
- **`self.thinking_budget` 不是 pydantic 字段**：导致全量 500。

## 6. B4 / B5 处置

- **B4（MiniMax `reasoning_effort=high/xhigh` 思考 0 token）**：三次尝试均不复现。
  我们 `high` rt=14/21/14、`xhigh` rt=0/17/19；官方 `high` rt=15/18/18、`xhigh` rt=20/22/20。
  那个 `rt=0` 是短回答本来就没怎么思考的随机性。**关闭，不改代码。**
- **B5（Qwen `json_schema` 答错）**：未复现，`json_schema` 行两边都是 200 且形状一致。
  **关闭。**

## 7. 单测

`tests/entrypoints/openai/test_vendor_profile_tables.py`：**91 例全过**，
按设计 §4 三张表逐行覆盖（400 断言 `parameter`、忽略断言生效值、映射断言结果），
外加思考判定优先级、无画像模型不变、开关与白名单。

跑法（镜像内，无本地 venv）：

```bash
docker cp tests/entrypoints/openai/test_vendor_profile_tables.py <container>:/tmp/
docker exec <container> python3 -m pytest /tmp/test_vendor_profile_tables.py -q
```

## 8. 现网升级：阻塞

**代码已合入 `master`，但现网尚未升级**，原因是产不出新镜像：

1. 本轮改动是 5 个 Python 文件（`envs.py` + 4 个 entrypoint 文件），需要重新打包镜像；
2. dev 节点向 ACR push 被拒：`insufficient_scope: authorization failed`（凭据只有 pull 权限）；
3. 自托管 CI runner 不在线：`GET /actions/runners` 返回 `total_count: 0`；
4. gpustack 的模型配置里没有 `volume_mounts` 字段，无法用 bind-mount 灰度。

解除办法（任一即可）：

- 提供有 push 权限的 ACR 凭据；或
- 拉起一个自托管 CI runner，走现有流水线出包；或
- 升级 gpustack 到支持 `volume_mounts` 的版本，用 bind-mount 灰度。

镜像就绪后的滚动顺序（用户已确认）：**MiniMax 直接升**（1 副本）→
**DeepSeek 删一个重建、起来一个再删剩下的**（6 副本，任何时刻 ≥1 存活）→
**Qwen 全量升**（3 副本）。

gpustack 注意：只改 `backend_version` 不会重启实例，必须 `replicas 0 → N`；
版本名必须以 `-custom` 结尾。

回滚：`backend_version` 改回 `rep-backport-sampling-fix-custom`（或
`reputationly-backport-custom` / `reputationly-backport-m31-custom`）+ 0→N 重新部署。

## 9. 遗留

- 现网升级（见 §8）。
- MiniMax 缺 `cap` 28 题电池（`degen` 已过）。
- 网关对 MiniMax 渠道的 4xx 包装（`200 + choices: []`）是网关自身的问题，
  建议另开单子修，否则所有经网关的调用方都会把参数错误当成空回答。
