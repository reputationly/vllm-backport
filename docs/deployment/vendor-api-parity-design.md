# 自部署三模型对齐官方接口语义：设计文档

> 日期：2026-11-08
> 适用模型：`deepseek-v4-1-flash`（DeepSeek-V4.1-Flash）、`qwen3.8-flash-fp8`（Qwen3.8-Flash）、`minimax-m3.1-flash`（MiniMax-M3.1-Flash）
> 对照的官方接口（同一 MaaS 网关上的供应商渠道）：`deepseek-v4-1-flash-261017`、`Qwen3.8-Flash`、`MiniMax-M3.1-Flash`
> 代码基线：`origin/master`（含 `sampling_policy.py`，见 `a100-three-model-rollout-report.md`）
> 目标读者：按本文档实现的开发 agent。**本文所有"目标行为"均已定稿，不需要再做选择。**

---

## 0. 目标与结论

**目标**：同一个请求打到我们自部署的模型，和打到官方接口，**参数语义一致**：

- 官方支持的参数：效果与官方一致；
- 官方不支持、静默忽略的参数：我们也静默忽略（不报错、不生效）；
- 官方报 400 的参数/取值：我们也报 400（HTTP 状态码一致，报错信息指明参数名）。

**现状**（2026-11-08 逐参数实测，§2）：三个模型都存在大量差异，主要三类：

1. **我们放行、官方 400**：越界值（`temperature=2.5`、`top_p=0`/`1.5`、`presence_penalty=3`）、`json_object` 缺少 "json" 关键词、MiniMax `reasoning_effort=none` 等；
2. **我们生效、官方忽略**：DeepSeek 的 `top_k`/`n`/`seed`/`chat_template_kwargs`，MiniMax 的 `logprobs`/`stop`/`response_format`/`n` 等；
3. **我们不认、官方生效**：DeepSeek/Qwen 的 `thinking: {"type": "disabled"}`、Qwen 顶层 `enable_thinking`、Qwen 的 `max_tokens` 只限正文等。

另有 5 个**输出层面的 bug**（§5），其中 `tool_choice: "none"` 时 DeepSeek/Qwen 返回空回答、MiniMax 把原始 tool-call 标记漏进 content，属于必须修的正确性问题。

**已定的三个决策**（用户确认）：

| # | 决策 | 结论 |
|---|---|---|
| D1 | Qwen 思考模式 temperature | **保留 ≤1.0 上限**（不跟官方的 [0,2)）；同时实现官方的 0.6 下限；≥2 或 <0 报 400 |
| D2 | Qwen `max_tokens` 语义 | **按官方**：`max_tokens` 只限正文，不含思考；`max_completion_tokens` 限思考+正文 |
| D3 | vLLM 扩展字段（`chat_template_kwargs`/`top_k`/`min_p`/`repetition_penalty`/`n`/…） | **严格按官方**：官方不认的就忽略或报 400；保留一个环境变量白名单给内部工具用 |

---

## 1. 规范来源与判定规则

按优先级：

1. **实测的官方接口行为**（`results40-spec/`，§2 表格的 "官方" 列）——用户明确"给你的都是官方的接口"，所以实测行为是第一规范；
2. **官方文档**（只用于实测拿不到的情况）：
   - DeepSeek：<https://api-docs.deepseek.com/zh-cn/api/create-chat-completion>、<https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/>
   - Qwen（阿里云百炼）：<https://docs.bailian.console.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>
   - MiniMax：<https://platform.minimax.io/docs/api-reference/text-openai-api>
3. 实测与文档冲突时，**以实测为准**（例：DeepSeek 文档说 `tool_choice=required` 在思考模式报 400，实测官方返回 200 并调用工具 → 我们按 200 实现）。
   例外：MiniMax 的 `thinking.disabled` 实测时而 200 时而报错，**按文档报 400**。

**网关会丢的字段**：new-api 的 `GeneralOpenAIRequest`（`new-api/dto/openai_request.go:29-110`）没有 `thinking_budget`、`preserve_thinking`、`min_p`、`repetition_penalty`、`user_id`，`service_tier` 默认被过滤（`relay/common/relay_info.go:891`）。这些字段**经网关时任何后端都收不到**，规范只能来自文档；验收时必须直连 vLLM 测（§7.3）。

---

## 2. 实测差异总表（目标行为 = "官方" 列，除非另注）

测试方法：`~/Desktop/v41-loop-ablation-local/spec_probe.py`，问题 "17*23 等于多少？只回答数字。"，每个用例单请求串行；
原始数据 `results40-spec/*.json`，并排对照 `results40-spec/compare.txt`。
简写：`200` 正常；`200·忽略` 接受但不生效；`400` 拒绝；`rl` 思考字符数；`pt` prompt token。

### 2.1 DeepSeek-V4.1-Flash（我们 `deepseek-v4-1-flash` vs 官方 `deepseek-v4-1-flash-261017`）

| 用例 | 我们（现状） | 官方（目标） |
|---|---|---|
| `temperature=2.0` | 200 | 200 |
| `temperature=2.5` / `-1` | 200 | **400** `The parameter \`temperature\` specified in the request are not valid` |
| `top_p=0` | 200 | **400** `top_p must be in (0, 1]` |
| `top_p=1.5` | 200 | **400**（同上类信息） |
| `top_k=5` / `-1` | 200·**生效** | 200·忽略 |
| `presence_penalty=3` / `frequency_penalty=-3` | 200 | **400**（范围 [-2,2]；在范围内则一律忽略，见 §4.1） |
| `n=2` / `n=5` | 200，返回 2/5 个 choice | 200，**只返回 1 个 choice** |
| `seed` | 200·生效 | 200·忽略（官方文档无此参数） |
| `logprobs=true, top_logprobs=3` | 200，有 logprobs | 200，有 logprobs |
| `top_logprobs=25` | 400 | 400 |
| 只传 `top_logprobs`、不传 `logprobs` | **400** | 200·忽略 |
| `stop` 17 个 | 400（pydantic 类型错误） | 400 `stop ... array too long`（上限 16） |
| `max_tokens=0` | 400 | 400 |
| `response_format=json_object`，提示词无 "json" | **200** | **400** `Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'` |
| `response_format=json_schema` | 200 | 200 |
| `logit_bias` | 400（不同信息） | 400 `Reasoning model does not support n > 1, logit_bias` |
| 旧版 `functions` | 200·忽略 | 200·忽略 |
| `tools` + `tool_choice=none` | **200，content 为空（`"\n\n"`），pt=291** | 200，正常回答，**pt=35（工具被移出 prompt）** |
| `tool_choice=required`（思考模式） | 200，调用工具 | 200，调用工具 |
| `tool_choice` 指定函数 | 200，调用工具，**`finish_reason=stop`** | 200，调用工具，`finish_reason=tool_calls` |
| `chat_template_kwargs.thinking=false` | **关闭思考** | 忽略（仍思考） |
| 顶层 `enable_thinking=false` | 忽略 | 忽略 |
| `thinking: {"type":"disabled"}` | **忽略（仍思考）** | **关闭思考** |
| `thinking: {"type":"adaptive"}` / `"foo"` | 200 | **400** `The parameter \`type\` specified in the request are not valid: invalid value` |
| `reasoning_effort=none` | 关闭思考 | 关闭思考 |
| `reasoning_effort` minimal/low/medium/high/xhigh/max | 200 | 200 |
| `reasoning_effort=ultra` | **400**（pydantic Literal） | **200**（按 max） |
| `reasoning_effort=foo` | 400 | 400 |
| 历史消息带 `reasoning_content` | 丢弃（pt=54，与不带时相同） | 丢弃（pt=54） |

另：上一轮的惩罚参数实验（`results30-pen`）——关思考时 `frequency_penalty=0.3` 就让我们的 40 条长回答 0/6 合格，官方 5–6/6；官方文档明确两个惩罚参数"已不再支持，传入不会产生任何效果"。

### 2.2 Qwen3.8-Flash（我们 `qwen3.8-flash-fp8` vs 官方 `Qwen3.8-Flash`）

| 用例 | 我们（现状） | 官方（目标） |
|---|---|---|
| `temperature=2.0` / `2.5` | 200 | **400** `Temperature should be in [0.0, 2.0)` |
| `temperature=-1` | 400 | 400 |
| 思考模式 `temperature=0` | 生效（4 次中 2 种输出） | 按 0.6 处理（文档："0.6 以下的 temperature 值会默认改为 0.6"；实测 4/4 不同） |
| 思考模式 `temperature=1.8` | 被我们钳到 1.0 | 生效（RAG 5 类全覆盖 4/12）——**D1：保留我们的 1.0 上限** |
| `top_p=0` | 200 | 200（接受） |
| `top_p=1.5` | **200** | **400** `'top_p' must be in (0.0, 1.0]` |
| `top_k=-1` | **200** | **400** `top_k must be a non-negative integer` |
| `presence_penalty=3` / `frequency_penalty=-3` | 400 | 400（范围 [-2,2]，范围内**生效**） |
| 思考模式 `n=2` | **200，返回 2 个** | **400** `The n parameter must be 1 when enable_thinking is true` |
| `n=5` | 200 | **400** `Range of n should be [1, 4]` |
| 非思考 `n=2` | 200，2 个 | 200，2 个 |
| `top_logprobs=25` | 400（上限 20） | 400 `Range of top_logprobs should be [0, 5]` |
| 只传 `top_logprobs` | **400** | 200·忽略 |
| `stop` 17 个 | **400** | 200（`stop=["7"]` 实测生效：输出截在 "1,2,3,4,5,6,"） |
| `max_tokens=0` | 400 | 400 `Range of max_tokens should be [1, 131072]` |
| `max_tokens` 语义 | **含思考**（长推理截断成空回答） | **只限正文**（实测 `max_tokens=8000` 时 `completion_tokens` 达 17200；文档同） |
| `json_object` 无 "json" | **200** | **400** `'messages' must contain the word 'json' in some form` |
| `json_schema` | 200，但答案错（`{"v": 2}`，单样本） | 200，`{"v": 391}` |
| `logit_bias` | **400** | 200（效果未验证） |
| 旧版 `functions` | 200·忽略 | **200，`finish_reason=function_call`**（支持） |
| `tool_choice=none` | **200，content 为空，pt=419** | 200，正常回答，pt=66（工具被移出） |
| `tool_choice` 指定函数 | 调用，**`finish_reason=stop`** | 调用，`finish_reason=tool_calls` |
| 顶层 `enable_thinking=false` | **忽略** | 关闭思考 |
| `chat_template_kwargs.enable_thinking=false` | 关闭思考 | 关闭思考 |
| `thinking: {"type":"disabled"}` | **忽略** | 关闭思考 |
| `thinking: {"type":"adaptive"}` / `"foo"` | 200 | **400** `'type' must be in ["enabled", "disabled", "auto"]` |
| `reasoning_effort=none` | 关闭思考 | 关闭思考 |
| `reasoning_effort=ultra` / `foo` | 400 | 400 `'reasoning_effort' must be one of: ...` |
| 历史消息带 `reasoning_content` | 拼入（pt 365 vs 88） | 拼入（pt 374 vs 97，`preserve_thinking` 默认 true） |
| content 开头 | **`"\n\n391"`** | `"391"` |

### 2.3 MiniMax-M3.1-Flash（我们 `minimax-m3.1-flash` vs 官方 `MiniMax-M3.1-Flash`）

| 用例 | 我们（现状） | 官方（目标） |
|---|---|---|
| `temperature`（含 2.5、-1） | 200 | 200·忽略（恒为 1） |
| `top_p=0` | 200 | 200·忽略 |
| `top_p=1.5` | **200** | **400** `invalid params, param 'top_p' should be in (0,1] (2013)` |
| `top_k`、`presence_penalty=3`、`frequency_penalty=-3`、`logit_bias`、`seed` | 200 | 200·忽略（不校验范围） |
| `n=2` / `n=5` | **200，返回 2/5 个** | **400** `invalid params, model[MiniMax-M3.1-Flash] does not support n > 1 (2013)` |
| `logprobs=true` | **返回 logprobs** | 200·忽略（不返回） |
| `top_logprobs=25` / 只传 `top_logprobs` | **400** | 200·忽略 |
| `stop` | **生效**（17 个时 400） | 200·**忽略**（`stop=["7"]` 实测输出完整的 1..20） |
| `max_tokens=0` | **400** | 200（按未设置处理） |
| `max_tokens` 语义 | 含思考 | 含思考（文档） |
| `response_format` json_object / json_schema | **生效（输出 JSON）** | 200·**忽略**（输出 "391"） |
| 旧版 `functions` | 200·忽略 | 200·忽略 |
| `tool_choice=none` | **content 漏出 `]<]minimax[>[<tool_call>`** | 200，**仍调用工具**（官方只支持 auto） |
| `tool_choice=required` / 指定函数 | 调用；指定函数时 `finish_reason=stop` | 调用，`finish_reason=tool_calls` |
| `parallel_tool_calls=false` | 200 | 200·忽略 |
| `thinking: {"type":"disabled"}` | **200** | **400** `model "MiniMax-M3.1-Flash" requires adaptive thinking; thinking.type="disabled" (including reasoning.effort=none) is not allowed (2013)` |
| `thinking: {"type":"foo"}` | **200** | **400**（2013） |
| `thinking: {"type":"enabled"/"adaptive"}` | 200 | 200 |
| `reasoning_effort=none` | **200，仍思考** | **400** `... requires adaptive thinking; reasoning effort none is not allowed (2013)` |
| `reasoning_effort=foo` / `ultra` | 400（pydantic） | 400 `invalid params, invalid reasoning_effort: "foo" (allowed: low, medium, high, xhigh, max) (2013)` |
| `reasoning_effort=minimal` | 200 | 200（按 low） |
| `reasoning_effort=medium` / `xhigh` | **200，思考为 0 token** | 200，正常思考 |
| `chat_template_kwargs.thinking=false`、顶层 `enable_thinking=false` | 200（仍思考） | 200·忽略 |
| `reasoning_split=false`、`service_tier` | 200 | 200·忽略 |
| 历史消息带 `reasoning_content` | 拼入 | 拼入 |

---

## 3. 设计

### 3.1 总体结构

把现有 `vllm/entrypoints/openai/chat_completion/sampling_policy.py` 扩展为"**官方接口画像（vendor API profile）**"。一个画像描述一个模型族的全部参数语义，分四个阶段生效：

```
HTTP JSON ──► ① 请求闸门（raw dict 阶段）──► pydantic 解析 ──► ② 模板参数 ──► ③ 采样参数 ──► ④ 输出后处理
               校验/报 400、丢弃字段、            ChatCompletionRequest     思考开关、      temperature/top_p/   content 去前导换行、
               别名映射、思考开关归一化                                     effort         惩罚、n、max_tokens  finish_reason、正文长度上限
```

- 一个 vLLM 进程只服务一个模型，所以画像在进程启动时确定一次（`OpenAIServingChat.__init__` 里已有 `self.sampling_policy = policy_for(self.model_config)`），并写到模块级变量 `ACTIVE_PROFILE`，供 ① 使用。
- **没有画像的模型完全不受影响**（与现有 `policy_for` 返回 `None` 的语义一致）。

### 3.2 数据结构（放在 `sampling_policy.py`，替换现有 `SamplingPolicy`）

```python
@dataclass(frozen=True)
class VendorProfile:
    name: str
    thinking_style: Literal["deepseek", "qwen", "always"]
    # ① 闸门
    validate: Callable[[dict], None]          # 违反官方规则时 raise VLLMValidationError（→400）
    normalize: Callable[[dict], dict]         # 丢弃/映射字段，返回新的 raw dict
    # ③ 采样
    sampling: Callable[[SamplingInputs, bool], SamplingInputs]   # (原值, thinking) -> 生效值
    # ④ 输出
    answer_only_max_tokens: bool = False      # Qwen=True（D2）
    strip_leading_newlines: bool = False      # Qwen=True（§5 B2）
```

每个族一个模块级实例：`DEEPSEEK_V4`、`QWEN3_8`、`MINIMAX_M3`（`GLM5` 保持现有采样行为不动，本次不扩展）。
`policy_for()` 的查表逻辑（`_POLICIES` / `_ARCHITECTURE_POLICIES`）保留，返回值类型改为 `VendorProfile`。
`validate`/`normalize`/`sampling` 写成每族一组普通函数，**不要**做通用规则引擎；规则即 §4 的表。

### 3.3 ① 请求闸门：位置与顺序

- 在 `ChatCompletionRequest`（`protocol.py:219`）上加一个 `@model_validator(mode="wrap")` 的 `_vendor_profile_gate`，**定义在子类所有 validator 的最前面**：拿到 raw dict 后，若 `ACTIVE_PROFILE` 存在且开关打开，先 `validate(data)` 再 `data = normalize(data)`，然后 `return handler(data)`。
- **必须先于其它校验运行**：现有的 `check_logprobs`（只传 `top_logprobs` 时报 400）、`StopParam` 的 `max_length`、`reasoning_effort` 的 `Literal` 都会在官方放行的用例上报错。
  - 注意基类 `OpenAIBaseModel`（`serve/engine/protocol.py:37-47`）已经有一个 wrap validator `__log_extra_fields__`，我们的是第二个 wrap，而子类还有 `mode="before"` 的 `check_cache_salt_support`、`_normalize_messages_before`、`check_logprobs`。
  - 嵌套顺序不要凭记忆，**用三条单测锁住**。DeepSeek 画像下：
    - `{"top_logprobs": 3}`（无 `logprobs`）解析成功；
    - `reasoning_effort="ultra"` 解析成功；
    - `{"temperature": 2.5}` 抛 `VLLMValidationError` 且 `parameter == "temperature"`。
  - 基类是 `extra="allow"`，所以 `thinking`、`thinking_budget`、`preserve_thinking` 本来就不会让解析失败。`normalize` 删它们是为了行为，不是为了让解析通过。
- `normalize` 的产出必须能通过后续所有 pydantic 校验：
  - `reasoning_effort` 映射到 `Literal` 允许的值（ultra→max）；
  - 被忽略的字段直接从 dict 删除。
- `stop` 个数（**硬性部署要求**）：`StopParam` 的 `max_length` 取自 `VLLM_MAX_STOP_STRINGS`（`generate/base/protocol.py:28`），**缺省是 4**，并且在 import 时就固定了。闸门之后 pydantic 仍会按它校验，所以：
  - 有画像的实例必须设 `VLLM_MAX_STOP_STRINGS=64`。否则 DeepSeek 5–16 个 stop（官方接受）和 Qwen 17 个 stop（官方接受）都会被 pydantic 拒掉。
  - `OpenAIServingChat.__init__` 在有画像时检查这个值，小于 64 就启动失败并给出明确报错。
  - 各族的个数规则只在闸门里执行：DeepSeek 超过 16 报 400（官方实测 17 个即 400，**不能改成静默丢弃**）；Qwen 不限；MiniMax 直接删掉 `stop`。
- `n` 被静默改为 1（DeepSeek）时，闸门打一行 `logger.info`，带 request_id 和原始 `n`，方便排查"为什么少了 choice"。同理，MiniMax 把 `tool_choice` 改写为 `auto` 时也打一行。
- 报错统一抛 `VLLMValidationError(message, parameter=...)`，走 vLLM 现有的 400 错误响应（`{"error": {"message", "type", "param", "code": 400}}`）。
  `message` 用 §2 表中官方的原文（不要求逐字一致，但必须包含参数名和取值范围），方便客户端日志对照。
- 思考开关在闸门里**一次性归一化**成两个私有字段，后续阶段只读它们，不再各自解析：
  - `_profile_thinking: bool`
  - `_profile_effort: str | None`

  `ChatCompletionRequest` 用 `PrivateAttr` 存，闸门把它们放进 dict 的保留键（如 `"__vendor_thinking"`）并在 after-validator 里搬进 PrivateAttr，再从 extra 里删掉。

### 3.4 ② 模板参数

- `resolved_template_kwargs()`（`protocol.py:586`）在有画像时**不再合并用户的 `chat_template_kwargs`**（D3），只用闸门算出的值。DeepSeek 和 MiniMax 完全忽略；Qwen 只认 `enable_thinking` 一个键，因为官方也认，由闸门读取。
  - DeepSeek：`{"thinking": _profile_thinking, "reasoning_effort": _profile_effort 或 "none"}`
  - Qwen：`{"enable_thinking": _profile_thinking}`，`reasoning_effort` 按 §4.2 映射后透传
  - MiniMax：`{"reasoning_effort": _profile_effort}`
- `sampling_policy.thinking_enabled()` 改为直接返回 `_profile_thinking`。现有按 `template_kwargs` 推断的逻辑保留，只作为无画像模型的回退。
- 白名单逃生口：环境变量 `VLLM_VENDOR_API_PROFILE_ALLOW`（逗号分隔，默认空）里列出的字段，闸门不删、阶段②/③照旧生效。
  例：内部压测脚本需要 `chat_template_kwargs` 时，在**内部专用实例**上设 `VLLM_VENDOR_API_PROFILE_ALLOW=chat_template_kwargs`。**现网对外实例不设。**

### 3.5 ③ 采样参数

在现有 `to_sampling_params()` 的 policy 分支（`protocol.py:716-747`）里调用 `profile.sampling(...)`，按 §4 生效值表覆盖：

- `temperature`、`top_p`、`top_k`、`min_p`、`repetition_penalty`
- `presence_penalty`、`frequency_penalty`
- `seed`、`n`（DeepSeek 静默改成 1）
- 缺省值（官方缺省与我们的 `generation_config` 不同时，以官方为准，见 §4 "缺省"行）

### 3.6 max_tokens（全部在 `serving.py:314-335`）

- **DeepSeek**：含思考（与现状同）。请求未设置时用官方缺省：非思考 8192；思考 65536（`reasoning_effort=max` 时 131072）。超过 393216 的值接受，并按上下文剩余量截断（官方实测 `999999` 返回 200）。
  - 缺省值作为 `fallback_max_tokens` 传入 `get_max_tokens`（`serve/utils/api_utils.py:169`），这个函数本身就会对"上下文剩余"取 min，不需要另外处理。
  - 真正的行为变化：以前不传 `max_tokens` 时输出上限是上下文剩余量，现在**关思考时只有 8192**。客户的超长输出（例如关思考写长文档）会被截断。这是官方行为，但要进回归（§7.3）和上线通知（§7.4）。
- **MiniMax**：含思考。`max_tokens=0` 按未设置处理。
- **Qwen（D2）**：
  - 有 `max_completion_tokens` 时：它就是思考+正文总上限，与现状同。
  - 否则 `max_tokens` = **正文上限 A**（缺省 = 模型最大输出）。另有思考预算 B：`thinking_budget`，或按 effort 映射：low 4096 / medium 16384 / xhigh 262144；都没设时 131072。
  - 引擎总上限 = `min(上下文剩余, A + B)`。用 vLLM 已有的 `SamplingParams.thinking_token_budget = B` 强制在 B 处结束思考（`vllm/v1/sample/thinking_budget_state.py`）。
  - **不需要新增启动参数**：只要有 `--reasoning-parser`，`arg_utils.py` 的 `_set_default_reasoning_config_args` 就会自动构造 `ReasoningConfig`；`config/reasoning.py:initialize_token_ids` 从 parser 取起止字符串（qwen3 parser 是 `<think>` / `</think>`，见 `parser/qwen3.py:41-42`），两者都能 tokenize 就 `enabled=True`。现网 Qwen 已有 `--reasoning-parser=qwen3`。**不要**另加 `--reasoning-config`，以免覆盖从 parser 推导出的字符串。
  - **WP5a 只做验证**：启动时打印 `reasoning_config.enabled`，有画像且 `answer_only_max_tokens=True` 时断言它为 True。然后在 dev 上实测 `thinking_token_budget=64` 对 qwen4_exp 是否在约 64 个思考 token 处结束思考。要特别确认 Qwen3.8 模板把 `<think>` 放在 prompt 里的情况下，预算是否仍从生成开始计数。验证通过才做 WP5b。
  - **WP5b 正文上限 A 必须提前停，不能事后截断**：在 serving 的输出循环里（流式和非流式都要），统计 reasoning parser 判定为正文的 token 数，达到 A 时**立即** `engine_client.abort(request_id)` 停止生成，`finish_reason="length"`，返回已生成的部分。
    只截断返回文本而让引擎继续生成，会把 A 之后的算力全部浪费，这种做法不合格。允许多出几个 token（官方文档也说有 ±10 token 误差）。
  - 范围：`max_tokens` / `max_completion_tokens` 只接受 [1, 131072]，否则 400。

### 3.7 ④ 输出后处理

见 §5 的 B1–B3：去掉 Qwen content 的前导换行；指定函数时 `finish_reason` 改为 `tool_calls`；MiniMax 去掉 `tool_choice` 的 none/required/named 语义。

### 3.8 开关

| 环境变量 | 缺省 | 作用 |
|---|---|---|
| `VLLM_VENDOR_SAMPLING_POLICY` | `1`（已存在） | 总开关，`0` 时①②③④全部关闭，回到原生 vLLM 行为 |
| `VLLM_VENDOR_API_PROFILE` | `1`（新增） | 只关①闸门和②模板，保留③采样语义（用于排障时区分是闸门还是采样的问题） |
| `VLLM_VENDOR_API_PROFILE_ALLOW` | 空（新增） | 字段白名单，见 §3.4 |
| `VLLM_MAX_STOP_STRINGS` | `4`（已存在） | **有画像的实例必须设为 64**（硬性部署要求，启动时检查，见 §3.3） |

---

## 4. 各族规范（实现时逐行对照）

"校验"列：违反就 400。"效果"列：校验通过后的生效方式。

### 4.1 DeepSeek-V4.1-Flash（`DEEPSEEK_V4`）

| 参数 | 校验 | 效果 |
|---|---|---|
| `temperature` | [0, 2] | 思考：忽略（=1.0）；非思考：生效。缺省 1.0 |
| `top_p` | (0, 1] | 思考：缺省 1.0，<0.95 按 0.95（**带 tools 时缺省也是 1.0**，现有 `top_p_default_with_tools=0.95` 是 v4-pro 的规则，改掉）；非思考：恒 1.0 |
| `presence_penalty` / `frequency_penalty` | [-2, 2] | **两种模式都忽略**（官方已废弃）——这是修复 `results30-pen` 的关键 |
| `top_k` / `min_p` / `repetition_penalty` / `seed` | 不校验 | 忽略（=-1 / 0 / 1.0 / None） |
| `n` | 不校验 | 恒为 1 |
| `logprobs` / `top_logprobs` | `top_logprobs` ∈ [0, 20] | 生效；只传 `top_logprobs` 时忽略它 |
| `logit_bias` | 出现即 400 `Reasoning model does not support n > 1, logit_bias` | — |
| `stop` | ≤ 16 个 | 生效 |
| `max_tokens` / `max_completion_tokens` | ≥ 1 | 见 §3.6 |
| `response_format` | `json_object` 时 messages 里必须出现 "json"（不区分大小写）。**只扫 `messages` 的文本**（string content 和 `type=text` 的 content part，包括 system 和 tool 结果），**不扫 `tools` 定义**，避免 tool schema 让检查假通过 | `json_object` / `json_schema` 生效 |
| `tools` / `tool_choice` | — | `none`：从 prompt 里移除 tools（闸门删 `tools`）；`auto`/`required`/指定函数：生效（思考模式也允许）|
| `functions` / `function_call` | — | 忽略 |
| `thinking` | `type` ∈ {enabled, disabled} | 优先级最高 |
| `reasoning_effort` | ∈ {none, minimal, low, medium, high, xhigh, max, ultra} | none → 关思考；ultra → max；其余交给模板（模板已有映射，`deepseek_v41_encoding.py:190`） |
| `enable_thinking`、`chat_template_kwargs` | — | 忽略 |
| 思考判定 | — | `thinking.type` 有值用它；否则 `reasoning_effort=="none"` → 关；否则开 |

### 4.2 Qwen3.8-Flash（`QWEN3_8`）

| 参数 | 校验 | 效果 |
|---|---|---|
| `temperature` | [0, 2) | 思考：`clamp(T, 0.6, 1.0)`（**D1**），缺省 1.0；非思考：生效，缺省 **0.7** |
| `top_p` | [0, 1]（官方接受 0） | 生效；0 按 1e-6 处理（官方实测接受，效果是推断）。缺省：思考 0.95 / 非思考 **0.8** |
| `top_k` | 整数 ≥ 0 | 0、null 或 >100 → 禁用；其余生效。缺省 20 |
| `presence_penalty` | [-2, 2] | 生效。缺省：思考 0.0 / 非思考 **1.5**（官方文档对 qwen3.8-flash 非思考模式的缺省值） |
| `frequency_penalty` | [-2, 2] | 生效（官方实测生效） |
| `repetition_penalty` | > 0 | 生效，缺省 1.0（网关会丢，直连才可见） |
| `min_p`、`logit_bias` | — | 忽略（官方对 `logit_bias` 返回 200，效果未验证，按忽略处理） |
| `seed` | 不校验 | 生效 |
| `n` | 思考：必须 1，否则 400 `The n parameter must be 1 when enable_thinking is true`；非思考：[1, 4]，否则 400 `Range of n should be [1, 4]`；0 → 1 | 生效 |
| `logprobs` / `top_logprobs` | `top_logprobs` ∈ [0, 5] | 生效，**只返回正文 token 的 logprobs**（文档：思考部分不返回）；只传 `top_logprobs` 时忽略它 |
| `stop` | 不限个数（部署设 `VLLM_MAX_STOP_STRINGS=64`） | 生效 |
| `max_tokens` / `max_completion_tokens` | [1, 131072] | 见 §3.6（D2） |
| `response_format` | `json_object` 时 messages 必须出现 "json"（扫描范围同 §4.1，不扫 `tools`），否则 400 `'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'` | 生效 |
| `tools` / `tool_choice` | — | `none`：移除 tools；其余生效 |
| `functions` / `function_call`（旧版） | — | **支持**：闸门把 `functions` 转成 `tools`、把 `function_call` 转成 `tool_choice`；响应里把第一个 tool_call 转回 `message.function_call`，并设 `finish_reason="function_call"`（P2，见 §6） |
| `enable_thinking`（顶层） | bool | 生效 |
| `chat_template_kwargs` | — | 只读 `enable_thinking` 一个键，其余忽略 |
| `thinking` | `type` ∈ {enabled, disabled, auto}，否则 400 `'type' must be in ["enabled", "disabled", "auto"]` | auto 等同 enabled |
| `reasoning_effort` | ∈ {none, minimal, low, medium, high, xhigh, max}，否则 400 `'reasoning_effort' must be one of: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'`；与 `thinking_budget` 同时出现 → 400 | max/high → xhigh；minimal → low；none → 关思考 |
| `thinking_budget` | 整数 ≥ 0 | 思考预算（§3.6） |
| `preserve_thinking` | bool，缺省 true | false 时闸门删掉历史 assistant 消息的 `reasoning_content` |
| 思考判定（冲突时按此顺序取第一个有值的） | — | `thinking.type` → 顶层 `enable_thinking` → `chat_template_kwargs.enable_thinking` → `reasoning_effort=="none"` → 缺省开 |

### 4.3 MiniMax-M3.1-Flash（`MINIMAX_M3`）

| 参数 | 校验 | 效果 |
|---|---|---|
| `temperature` | 不校验 | 恒 1.0 |
| `top_p` | ≤ 1 且 ≥ 0，否则 400 `invalid params, param 'top_p' should be in (0,1] (2013)` | 恒 0.95 |
| `top_k`/`min_p`/`repetition_penalty`/`presence_penalty`/`frequency_penalty`/`logit_bias`/`seed` | 不校验 | 忽略 |
| `n` | > 1 → 400 `invalid params, model[MiniMax-M3.1-Flash] does not support n > 1 (2013)` | 1 |
| `logprobs` / `top_logprobs` | 不校验 | 忽略（响应 `logprobs: null`） |
| `stop` | 不校验 | **忽略** |
| `max_tokens` | 0 按未设置处理 | 含思考 |
| `response_format` | 不校验 | **忽略** |
| `tools` / `tool_choice` / `parallel_tool_calls` | — | `tool_choice` 任意值都按 `auto` 处理（闸门改写）；`parallel_tool_calls` 忽略 |
| `functions` / `function_call` | — | 忽略 |
| `thinking` | `type=disabled` → 400 `model "MiniMax-M3.1-Flash" requires adaptive thinking; thinking.type="disabled" (including reasoning.effort=none) is not allowed (2013)`；`type` ∉ {adaptive, enabled, disabled} → 400（官方返回 2013 错误但原文未取到，信息自拟，须含 `thinking.type`） | 恒思考；`enabled` 按 `adaptive` 处理 |
| `reasoning_effort` | none → 400 `invalid params, model "MiniMax-M3.1-Flash" requires adaptive thinking; reasoning effort none is not allowed (2013)`；∉ {minimal, low, medium, high, xhigh, max} → 400 `invalid params, invalid reasoning_effort: "<v>" (allowed: low, medium, high, xhigh, max) (2013)` | minimal → low（官方接受 minimal，但该次实测思考为 0 token，只有一个样本，按文档就近映射到 low）；**缺省 max** |
| `enable_thinking`、`chat_template_kwargs`、`reasoning_split`、`service_tier` | — | 忽略 |

---

## 5. 必须一起修的 bug

| # | 现象（证据） | 定位线索 | 修法 | 状态 |
|---|---|---|---|---|
| B1 | `tool_choice="none"` 时 DeepSeek/Qwen 返回空 content；MiniMax 把 `]<]minimax[>[<tool_call>` 漏进 content（§2 各表） | tools 仍留在 prompt（pt=291/419），模型照样输出工具调用；而 parser 在 `tool_choice=="none" and tools` 时主动压制 tool_calls（`parser/engine/parser_engine.py:441-448`，`_suppress_tool_calls=True`），于是工具调用文本既进不了 `tool_calls` 也进不了 `content`。`serving.py:1024` 只是响应侧按 `tool_choice` 选择构造方式的分支，不是根因 | DeepSeek/Qwen：闸门删 `tools`（等价于 `--exclude-tools-when-tool-choice-none`，`online_renderer.py:229`，但不依赖启动参数）；MiniMax：改写成 `auto` | 已定位 |
| B2 | Qwen content 总以 `"\n\n"` 开头 | 模板在 `</think>` 后输出 `\n\n`，`qwen3` reasoning parser（`vllm/reasoning/qwen3_engine_reasoning_parser.py`）没有剥掉 | 画像 `strip_leading_newlines=True`：正文开头的连续 `\n` 去掉。流式：正文第一个非空 delta 之前的换行不发 | 待实现时确认是模板还是 parser |
| B3 | 指定函数的 `tool_choice` 返回 `finish_reason="stop"`（三个模型） | 推测：vLLM 的 named tool choice 走 guided JSON 分支，构造响应时没有把 finish_reason 改成 `tool_calls` | 有 tool_calls 时 `finish_reason` 一律设为 `tool_calls`（流式和非流式都改） | 行为已确认；代码位置需在 `serving.py` 的 named tool_choice 分支（搜 `ChatCompletionNamedToolChoiceParam`）里确认 |
| B4 | MiniMax `reasoning_effort=medium` / `xhigh` 思考 0 token（官方正常思考） | 模板来自权重目录的 `chat_template.jinja`（不在仓库里）；`reasoning_effort` 注入后 pt +9，但 medium/xhigh 走了"不思考"分支 | 在 dev 机打开 `MiniMax-M3.1-Flash-w8a8/chat_template.jinja` 核对 effort 分支，修模板或在闸门里改成模板认识的值 | **假设，需复现** |
| B5 | Qwen `json_schema` 返回 `{"v": 2}`（应为 391，官方正确） | 单样本；怀疑 guided decoding 在思考阶段就开始约束 | 先用 10 次复现；如果确实是在思考阶段就开始约束，改为只在 `</think>` 之后约束 | **假设，需复现** |

不在本次范围：MiniMax 思考长度约为官方 2 倍、单请求慢 4 倍（性能问题，另开）；官方 prompt 比我们多 9（Qwen）/24（MiniMax）个 token（官方注入了内容未知的系统提示，不复刻）。

---

## 6. 工作包与顺序

每个工作包单独一个 commit。全部是纯 Python 改动，不需要重编内核。

| WP | 内容 | 依赖 | 优先级 |
|---|---|---|---|
按下表顺序执行：

| 顺序 | WP | 内容 | 依赖 | 优先级 |
|---|---|---|---|---|
| 1 | WP1 | 框架：`VendorProfile`、`ACTIVE_PROFILE`、闸门 wrap validator（§3.3 的三条顺序单测）、模板与采样接入、三个开关、`VLLM_MAX_STOP_STRINGS` 启动检查、"无画像模型不变"单测 | — | P0 |
| 2 | WP2 | DeepSeek 画像（§4.1）+ B1 + B3 | WP1 | P0 |
| 3 | WP4 | Qwen 画像（§4.2，不含旧版 functions）+ B1 + B2 + `preserve_thinking` | WP1 | P0 |
| 4 | WP5a | **只做验证**：`reasoning_config.enabled` 断言和日志；`thinking_token_budget` 对 qwen4_exp 是否生效（§3.6） | WP4 | P0 |
| 5 | WP5b | Qwen 正文上限 A（提前 abort）+ `thinking_budget` / effort 预算映射 | WP5a 通过 | P0 |
| 6 | WP3 | MiniMax 画像（§4.3）+ B1（改写成 auto）+ B4 | WP1 | P0 |
| 7 | WP7 | B5 复现与修复 | WP4 | P1 |
| 8 | WP6 | Qwen 旧版 `functions` / `function_call` 兼容 | WP4 | P2 |
| 9 | WP8 | 部署：dev 验收 → 现网灰度 | 1–6 | — |

如果 WP5a 证实 `thinking_token_budget` 对 qwen4_exp 不生效，停下来回报，不要自行换方案。

现有单测 `tests/entrypoints/openai/test_vendor_sampling_policy.py` 和 `sampling_policy.py` 需要同步改：
- DeepSeek 带 tools 时 `top_p` 缺省从 0.95 改为 1.0，并**删除 `SamplingPolicy.top_p_default_with_tools` 字段**，不留死字段；
- 现有 `clear_penalties` 只在思考分支里执行（`protocol.py:721-747` 的 `if thinking_enabled(...)` 内）。DeepSeek 改为两种模式都忽略惩罚参数，"非思考模式保留惩罚参数"的断言要改；MiniMax 恒思考，不受影响；Qwen 不清零；
- `chat_template_kwargs` 不再参与 DeepSeek/MiniMax 的思考判定，Qwen 只认其中的 `enable_thinking`。

---

## 7. 验收

### 7.1 单元测试（`tests/entrypoints/openai/test_vendor_sampling_policy.py`，扩展现有文件）

用 fixture 设置/还原 `ACTIVE_PROFILE`。通过 `ChatCompletionRequest.model_validate(...)` 加 `to_sampling_params(...)` / `resolved_template_kwargs()` 断言，不起服务：

- **闸门顺序**：DeepSeek 下 `{"top_logprobs": 3}` 解析成功且 `logprobs` 为 False；`reasoning_effort="ultra"` 解析成功并映射为 max；Qwen 下 17 个 stop 解析成功。
- **§4 每张表每一行至少一条**：400 的用例断言抛 `VLLMValidationError` 且 `parameter` 正确；忽略的用例断言生效值为表中的值；映射的用例断言映射结果。
- **思考判定优先级**：Qwen 用 `thinking.type=enabled` + `enable_thinking=false` → 开；DeepSeek 用 `thinking.type=disabled` + `reasoning_effort=high` → 关。
- **无画像模型完全不变**：`policy_for(qwen3)` 为 None 时，`temperature=2.5` 等行为与改动前一致。
- **开关**：`VLLM_VENDOR_API_PROFILE=0` 时闸门不报错；`VLLM_VENDOR_API_PROFILE_ALLOW=chat_template_kwargs` 时 DeepSeek 的 `chat_template_kwargs.thinking=false` 生效。
- 运行：`.venv/bin/python -m pytest tests/entrypoints/openai/test_vendor_sampling_policy.py -v`；lint：`pre-commit run --files <改动文件>`。

### 7.2 端到端：逐参数对拍（主验收）

1. 在 dev 集群按现网配置起三个模型（启动参数见 `a100-three-model-rollout-report.md` §6，不新增 `--reasoning-config`；三个模型都加环境变量 `VLLM_MAX_STOP_STRINGS=64`）。
2. 改 `spec_probe.py`：用环境变量 `BASE` 覆盖 `URL`；对 dev 端点**直连**跑三族，单并发串行。
3. 写 `spec_check.py`：读新结果，按 §2 各表"官方（目标）"列逐行断言。断言粒度：
   - HTTP 状态一致；
   - 400 时 `message` 包含参数名；
   - 200 时检查"可观测量"：
     - `n_choices`；
     - 是否有 logprobs、top_logprobs 条数；
     - 思考是否为 0（`rl==0` 或 `rl>0`）；
     - `tool_calls` 数和 `finish_reason`；
     - content 是否为合法 JSON；
     - content 是否以 `\n` 开头；
     - `stop=["7"]` 时是否截断。
4. **通过标准**：§2 三张表所有行 100% 匹配（B4、B5 若复现失败，在验收报告里写明，不算失败）。
   - HTTP 层的 400 至少有一条单独断言锁住：DeepSeek 发 `temperature=2.5`，断言 `status == 400`，且响应体里 `error.param == "temperature"`、`error.type == "BadRequestError"`。
     闸门在 pydantic 解析阶段抛 `VLLMValidationError`；它是 `VLLMClientError`，不是 `ValueError`，所以不会被包成 `ValidationError`，而是由 `serve/exception_handling/` 映射成带 `param` 的 400。这条路径必须用 e2e 锁住，不能只靠单测。
   - WP5b 单独验收"提前停"：Qwen 设 `max_tokens=32`，问一个需要长回答的问题。断言 `finish_reason == "length"`、正文 token 数 ≤ 32+10，且 `usage.completion_tokens - reasoning_tokens ≤ 42`。如果正文被截了但 `completion_tokens` 远大于思考+42，说明引擎没停，判失败。
5. 经网关的字段丢失问题：`thinking_budget`、`preserve_thinking`、`repetition_penalty`、`min_p` 只在直连时验收。

### 7.3 回归（防止改坏已修好的东西）

| 电池 | 脚本 | 门槛 |
|---|---|---|
| 客户 RAG，T=1.0/1.8，各 12 次，三模型 | `cap_compare.py <out> degen`（`DEGEN_N=12 WORKERS=1`） | 异常 0/12，空 content 0 |
| DeepSeek 关思考惩罚参数（`nt_freq0.5_P4`、`nt_pen0.5_T0.7`、`nt_freqneg1_P4`） | `param_sweep.py`（`ARMS=...`，6 次，`WORKERS=1`） | 格式合格 ≥5/6（达到官方水平；修复前 0/6） |
| Qwen 困难题（`count200`/`hop4`/`kv12800`，`max_tokens=16000`） | `cap_compare.py <out> hard`（`REPS=3 WORKERS=1`） | 截断 ≤1/30（修复前 10/30），正确率 ≥ 官方（26/30）−2 |
| Qwen 思考 T=1.8 RAG | 同第一行 | 5 类全覆盖 ≥11/12（D1 保持） |
| 基础能力 28 题 | `cap_compare.py <out> cap`（`REPS=1`） | 与修复前持平（84/84 量级） |
| DeepSeek 缺省 `max_tokens` | 不传 `max_tokens`：①关思考，要求写约 12k token 的长文；②开思考，长文档摘要 | ①按官方在 8192 处 `finish_reason=length`（预期中的行为变化，记录在案）；②不提前截断 |
| 单测全量 | `pytest tests/entrypoints/openai/ -k "sampling or protocol or chat"` | 全过 |

**负载约束**（硬性）：我们的副本很少，对自部署端点只能单并发串行（`WORKERS=1`）。只在 dev 机测，现网 `llm01–llm26` 只读，不要动 `gpustack-worker`。

### 7.4 上线与回滚

- 纯 Python 改动，按 `a100-three-model-rollout-report.md` §6 "按改动类型选择出包方式"走 app 层出包，或在现网实例上 bind-mount 灰度。
- 灰度顺序：DeepSeek（6 副本，先切 1 个，**观察 24h**）→ Qwen → MiniMax（只有 1 个副本，放在低峰切）。
  - 观察项：网关日志里该模型 HTTP 400 的占比，以及按 `error.param` 的分布；
  - 关思考请求的 `finish_reason=length` 占比（缺省 8192 带来的行为变化）。
  - 400 占比明显上升时，先确认是否来自客户本来就不合法的参数，再决定是否继续全量。
- 回滚：设 `VLLM_VENDOR_SAMPLING_POLICY=0` 重启（全部回到原生行为），或回滚 gpustack 版本。
- **对外影响需要提前通知客户**：以前静默接受的请求现在会返回 400。重点：
  - `temperature>2`；
  - `top_p=0`（DeepSeek）；
  - MiniMax 的 `reasoning_effort=none` 和 `n>1`；
  - `json_object` 时提示词里没有 "json"。

  这些都是官方接口本来就会报的错，客户切换到官方渠道也会遇到。另一项要通知的变化是：DeepSeek 关思考时不传 `max_tokens`，输出上限变为 8192。

---

## 8. 附录：证据与脚本（`~/Desktop/v41-loop-ablation-local/`）

| 文件 | 内容 |
|---|---|
| `spec_probe.py`、`results40-spec/*.json`、`results40-spec/compare.txt` | 逐参数探测（§2 全部数据） |
| `passthrough_probe.py`、`results27-pass-*`、`results32-qwen-pass-*` | top_k / min_p / T=0 等是否生效 |
| `param_sweep.py`、`results29-sweep/`、`results30-pen/`、`results33-qwen-sweep/`、`results36/38-mm-sweep*` | 参数扫描（含惩罚参数剂量-反应） |
| `cap_compare.py`、`results23-xchan*`、`results31-qwen/`、`results35-mm/`、`results37-mm-seq/` | 能力与客户 case 对照 |
| `qwen_switch_probe.py` | 关思考开关对照 |
| `new-api/parity-report-20261108/report.md` §8 | 网关侧另一份越界值对照（与本文一致） |
