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
| 1 | DeepSeek 默认请求不思考（**分支内自伤，未进过现网**） | 每例都有 reasoning | rl=0 全线 | `resolved_template_kwargs` 无条件传 `reasoning_effort="none"`，V4.1 renderer 把 `none` 解释成"不思考"。这个 bug 由 WP1 提交 `be133fd8d5` 自己引入，合并前在 `5538610b20` 修掉，**现网从未出现**，因此不计入"修掉的真实差异"，实际为 7 个 |
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

`tests/entrypoints/openai/test_vendor_profile_tables.py` + `test_vendor_sampling_policy.py`：
**138 例通过、1 例跳过**，按设计 §4 三张表逐行覆盖（400 断言 `parameter`、忽略断言生效值、
映射断言结果），外加思考判定优先级、无画像模型不变、开关与白名单。

两点前提必须说明：

- 初版报告写的"91 例全过"是在容器里设了 `VLLM_MAX_STOP_STRINGS=64` 时跑出来的。该上限在
  导入期被固化进 pydantic 字段（`Annotated[..., Field(max_length=envs.VLLM_MAX_STOP_STRINGS)]`），
  运行时改环境变量无效，缺省值 4 时 16 个 stop 的用例必然失败。现在拆成两条：gate 规则直接
  断言（与环境无关），请求级用例在 `VLLM_MAX_STOP_STRINGS < 16` 时 `skip` 并给出原因。
- 原有的 `test_vendor_sampling_policy.py` 在初版里**整个文件导入失败**（`GLM5` 被删），
  报告没有如实反映这一点。

跑法（镜像内，无本地 venv）：

```bash
docker cp tests/entrypoints/openai/test_vendor_profile_tables.py <container>:/tmp/
docker exec <container> python3 -m pytest /tmp/test_vendor_profile_tables.py -q
```

## 8. 现网升级（进行中）

镜像：`swr.cn-central-221.ovaijisuan.com/reputationly/vllm-backport:arm64-sm80-20261109-1600-9497ff2f`
（digest `sha256:c20bee8162cd0a1bf1dde8ddc88e22652d742de150b76890189c9425d64ed591`）。
在 prod 基础镜像 `…f1a14bb7-csrc` 上叠一层，只替换 5 个 Python 文件；镜像内 91 条单测全过。

gpustack 版本名：`rep-backport-parity-custom`（已注册到 vLLM backend 的 `version_configs`）。

### 8.1 出包路径（踩过的坑）

- ACR（`crpi-…personal.cr.aliyuncs.com`）从 dev 节点 push 被拒：
  `insufficient_scope: authorization failed`（凭据只有 pull 权限）；自托管 CI runner 不在线
  （`/actions/runners` → `total_count: 0`）。
- **可行路径**：dev 节点上有 `swr.cn-central-221.ovaijisuan.com` 的 push 凭据。
  本地 `docker build` 出小层 → `docker push` 到 SWR → prod 节点可直接 pull。
- **必须先在每个 prod 节点预 pull**：gpustack 在部分节点上拉不到镜像时，只会留一个
  `-pause` 占位容器、实例状态停在 `starting` 且不报错（本轮 MiniMax 卡了 22 分钟，
  Qwen/DeepSeek 各报过一次 `Failed to pull image`）。预 pull 后立刻恢复。
  建议把这一步固化到升级流程里。
- **预 pull 要逐个确认成功**：本轮 `llm07` 静默失败——它的 `systemd-resolved` 两个
  网卡都是 `Current Scopes: none`、没有上游 DNS，`swr.cn-central-221...` 解析不了，
  `docker pull` 报 `lookup ... server misbehaving`。这个节点正好是 Qwen 一台实例的
  主节点，导致那台实例反复加载-退出（每轮约 24 分钟），卡了 1 小时。
  修法与其他节点一致：写 `/etc/systemd/resolved.conf.d/zz-upstream-dns.conf`
  （`DNS=100.125.0.25`、`FallbackDNS=223.5.5.5`、`Domains=~.`）后
  `systemctl restart systemd-resolved`，再 `getent hosts <registry>` 验证。
  升级前建议在所有节点跑一次 `getent hosts swr.cn-central-221.ovaijisuan.com`。

### 8.2 必配环境变量

新镜像对命中 vendor profile 的模型有启动硬校验：`VLLM_MAX_STOP_STRINGS >= 64`
（pydantic 默认只有 4，而 Qwen 官方允许不限个数的 `stop`）。三个模型都必须加：

```
VLLM_MAX_STOP_STRINGS=64
```

本轮 `deepseek-v4-1-flash` / `qwen3.8-flash-fp8` 起初没配，重建出来的实例一起来就
`ValueError` 退出，表现为"一直 error"。

### 8.3 进度

全部三个模型的所有副本都已换成新镜像（`c20bee8162cd`），旧副本（`8695df29c91b`）无残留。

| 模型 | 副本 | 状态 |
|---|---|---|
| `minimax-m3.1-flash` | 1/1 | **已升级**。网关 200、`content=391`、`rt=18~20`；`top_p=0` 200（修复生效） |
| `deepseek-v4-1-flash` | 6/6 | **已升级**。新副本直连 `fin=stop pt=40 rt=17 content=391`（修复前 `pt=14 rt=0`） |
| `qwen3.8-flash-fp8` | 3/3 | **已升级**。新副本直连 `max_tokens=32` → `fin=length ct=109 rt=75 answer=34`（WP5b 门槛 ≤42） |

升级后终验（经 MaaS 网关，客户 RAG 问句，`max_tokens=2000`，每模型每温度 12 次）：

| 模型 | T=1.0 | T=1.8 |
|---|---|---|
| `deepseek-v4-1-flash-261017`（→ 我们） | 异常 0/12、空 0 | 异常 0/12、空 0 |
| `Qwen3.8-Flash`（→ 我们） | 异常 0/12、空 0 | 异常 0/12、空 0 |
| `MiniMax-M3.1-Flash`（→ 我们） | 异常 0/12、空 0 | 异常 0/12、空 0 |

gpustack 改 `env` 或删实例后会按副本逐个重建，所以整个过程 DeepSeek 始终有 ≥3 台在线。

### 8.4 回滚

`backend_version` 改回 `rep-backport-csrc-custom`（或 `rep-backport-sampling-fix-custom` /
`reputationly-backport-m31-custom`）+ `replicas 0 → N`。gpustack 的 PUT 需要带
`name` + `source` 全量字段，只传 `replicas` 会 422。

## 8.5 评审后的修复（2026-11-09 第二轮）

代码评审跑了两份对照（`c2f4707c29` 合并前 / `ffe96511aa` 合并后）后提出 3 个必修 + 一批建议，
本轮全部处理：

| # | 问题 | 处置 |
|---|---|---|
| 1 | **GLM-5.3 的采样策略在 WP1 重写时被删**（`GLM5` 及 `glm5_next`/`Glm5NextForCausalLM` 映射），下次给 `glm-5.3-flash-w4a8` 出包会失去「思考时温度 ≤1.0 / top_p 0.95」，高温退化可能重现 | 以 `VendorProfile` 形式恢复：`_glm5_sampling` 做 `min(T, 1.0)` + `top_p=0.95`，`_glm5_state` 恒思考，`_glm5_gate` 原样放行（未实测过该族的请求规则，不臆造）。并在 `resolved_template_kwargs` 的 `"always"` 分支保留调用方 `chat_template_kwargs`（MiniMax 的闸门已把它们删掉，因此对 MiniMax 无影响） |
| 2 | **`finish_reason="tool_calls"` 覆盖没有限定只对画像模型生效**，违反"无规则表模型不受影响" | 流式与非流式两条路径都加上 `self.sampling_policy is not None` 条件 |
| 3 | **`VLLM_VENDOR_API_PROFILE=0` 语义错误**：闸门关闭后不再判定思考，一律按"开思考"采样，而模板可能渲染成不思考 | `to_sampling_params` 在闸门关闭时退回 `thinking_enabled(resolved_template_kwargs(), style)` |

建议项处理情况：

- **Qwen 流式提前停止两处漏洞**：达预算后立即给该 choice 打 `finish_reason_sent` 并发终止块，
  引擎 abort 前已生成的那段不再外推；撞上限的那个 delta 正常发出并计入 `usage`。
- **非流式未判断是否开思考**：补上 `or not thinking`，与流式一致。
- **DeepSeek 思考态 `top_p` 缺省**：不再依赖模型 `generation_config`，未显式传时固定 1.0
  （现网 `DeepSeek-V4.1-Flash-1017` 没有 `generation_config.json`，本来也是 1.0，属加固）。
- **4 个没人读的配置字段**（`default_max_tokens_thinking` / `default_max_tokens_chat` /
  `drop_tools_when_none` / `force_tool_choice_auto`）：删除。设计里 DeepSeek 不传 `max_tokens`
  时的缺省上限（65536 / 8192）**未实现**，此处明确记录。
- **`thinking_budget` 类型校验**：补上，非负整数之外一律 400。
- **DeepSeek `thinking.type=enabled` + `reasoning_effort=none`**：`thinking.type` 优先，
  不再把 `none` 透给 renderer（否则模板按不思考渲染、采样按思考处理）。Qwen 同类问题一并修。
- **流式去开头换行可能吞掉 tool_calls**：现在只有 content 为空且既无 reasoning 也无 tool_calls
  时才置空整个 delta。
- **`test_gate_smoke.py`**：删除（无 SPDX 头、行超长、内容与 profile tables 重复），
  其中"无画像模型不变"与"开关"两条已并入 profile tables。
- **Qwen 缺省值**：显式实现设计 §4.2 的 `top_p` 0.95/0.8、`top_k` 20、`presence_penalty` 0.0/1.5、
  非思考 `temperature` 0.7，且**只对调用方没传的字段生效**（`SamplingInputs.specified`）。
- **隐藏的全局依赖**：`to_sampling_params` 原先无条件读 `_vendor_state`，而它由模块级
  `ACTIVE_PROFILE` 的闸门产生；传一个非激活的 policy 时会用错状态。现在只要
  `ACTIVE_PROFILE is not policy` 就用传入的 policy 重新推导。

仍未做：**服务层新逻辑没有单测**（提前停止、终止块补发、`functions` 旧形状、
去开头换行、`finish_reason` 覆盖）——这些只能端到端验证，补单元测试需要先把
async 生成器里的逻辑抽成纯函数，留作后续。

## 9. 遗留

- 现网升级（见 §8）。
- MiniMax 缺 `cap` 28 题电池（`degen` 已过）。
- 网关对 MiniMax 渠道的 4xx 包装（`200 + choices: []`）是网关自身的问题，
  建议另开单子修，否则所有经网关的调用方都会把参数错误当成空回答。
