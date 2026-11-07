# V4.1 文本字面哨兵拼写转义方案（评审稿）

## 1. 问题背景

上游 vLLM 在 `_normalize_messages` 中对消息文本做 `_reject_image_placeholder`
检查：任何 str content / text part / `reasoning_content` 里出现字面量
`<｜deepseek_image｜>` 就抛 `ValueError`，OpenAI 兼容层转成
`400 {"type":"BadRequestError","code":400}`。

- 上游出处：`vllm-project/vllm` PR #62199（commit `ddefe40e16`），本仓
  `5132bea899` 手工合入，逻辑逐字一致。`ddefe40e16` 不是 HEAD 祖先。
- 误伤机制：agent 类客户端每轮全量回放消息历史；工具输出（`cat`/`grep`
  本仓源码、`mm_preprocess.py` 等）、用户粘贴的报错原文中都可能含该字面量。
  一旦进入历史，之后**每轮请求**都被拒——会话永久中毒；报错文本本身
  含该拼写，粘贴报错会再次污染新会话。
- 公开佐证：NousResearch/hermes-agent#149807（同日发生，复现路径一致）。

guard 的原始动机是真实的：vLLM 对整串 prompt 做 `encode()`，HF tokenizer
的 added-token 匹配会把文本中的字面拼写编成**真的图片 token id**
（129264），随后 mm 展开按 `PromptReplacement(target=[image_token_id])`
计数占位符，多出的占位符导致图文槽位不匹配或静默错位。

## 2. 设计

DeepSeek 官方参考实现（deepseek-recipe）语义：**文本中打出的特殊 token
拼写是普通文本**。参考实现 `encode_ids(text, special_ranges)`：渲染输出的
特殊区间按拼写映射 token id，区间之间的文本用**不带 added vocabulary**
的 tokenizer 编码（`deepseek-recipe-encoding/src/tokenizer.rs`；
`docs/tokenizer.md` 明确此语义；`v4/mod.rs:573` 有对应测试）。

本方案复刻该语义，不引入 special_ranges 结构，改用**标记包裹**：

```
文本中的字面拼写  →  \ue000{原拼写}\ue001      （normalize 阶段，随文本流动）
渲染后 prompt     →  ...模板文本...\ue000<｜deepseek_image｜>\ue001...
编码时            →  普通段照常编码；被标记拼写 → 剥离 added vocab 后的
                     tokenizer 对拼写的纯文本 ids，拼回原位
```

等价性：参考实现在 special-range 边界切开纯文本区分别编码；本方案在标记
边界切开，两侧编码结果一致（同一边界、同一无 added vocab 编码器）。

## 3. 实现细节

全部改动集中在 `vllm/tokenizers/deepseek_v41.py`（+ 测试文件）。

### 3.1 标记与转义（`deepseek_v41.py:146-264`）

- `_MARK_OPEN = "\ue000"` / `_MARK_CLOSE = "\ue001"`（PUA 私用区字符；
  包裹内保留原拼写，编码时还原拼写后按拼写取纯文本 ids）。
- `_LITERAL_SPELLINGS` 保底集合 = `IMAGE_PLACEHOLDER` +
  `"<|place_holder_mm_span_0436|>"`（129265 压缩对齐 pad；
  `image_sentinel_mask` 会把该 id 当图片 span 做 embedding 掩码——
  此前的漏洞是**静默损坏、无报错**，且该 ASCII 拼写就明文写在
  `mm_preprocess.py:73`，同一条 `cat` 污染路径可达）。
- `_escape_literal_spellings`：regex 单趟替换（`re.escape` + 按长度
  降序排列的 alternation），避免长拼写被内部子串匹配嵌套破坏；
  `re.sub` 不回扫已插入文本。
- `_escape_json_value`：递归处理 dict/list/str，用于 `tools` function
  定义与 dict 形式的 `arguments`（返回新对象，不改动入参）。
- `_escape_tool_arguments`：`arguments` 为 JSON 字符串时先 `json.loads`
  再递归转义、`ensure_ascii=False` 重新序列化——`ensure_ascii` 会把
  `｜` 编码成 `\uff5c`，对原始字符串直接 replace 会漏；非 JSON 字符串
  退化为直接转义。
- `_normalize_messages(messages, literal_spellings)`：转义覆盖——
  所有 role 的 str content、text/`input_text`/`output_text` part、
  `reasoning`/`reasoning_content`、`tool_calls.function.arguments`。
  image part 仍插入**未标记**的 `IMAGE_PLACEHOLDER` → 照常编码为图片
  token 并计入 `_image_indices`/`image_order`。
- `apply_chat_template` 里 `system["tools"]` 同样过 `_escape_json_value`。

### 3.2 保护拼写集合的构造（`get_deepseek_v41_tokenizer`，`:267-282`）

wrap 时读取 `added_vocab`：`image_id = added_vocab.get(IMAGE_PLACEHOLDER)`；
模型侧把 `[image_id, image_id + 5)` 整段 id 当图片 token 路由
（`mm_preprocess.py:64-66` 注释），因此把 added vocab 中 id 落在
`(image_id, image_id+5)` 的拼写全部并入保护集合（例如部署 tokenizer
里可能存在的 `<｜image｜>`、`<｜rl_image_*｜>`）。找不到 `IMAGE_PLACEHOLDER`
时只用保底集合。

### 3.3 编码期拼接（`:337-446`）

- `_plain_tokenizer()`：`backend_tokenizer.to_str()` → 删 `added_tokens`
  → `Tokenizer.from_str` 重建，等价于 deepseek-recipe 的
  `PromptTokenizer`（"a copy without the added vocabulary"）。懒构造并缓存。
  无 `backend_tokenizer` 时返回 `None`。
- `_literal_token_ids(spelling)`：有 plain tokenizer → `plain.encode(
  spelling)`；无 → 逐字符 `self.encode` 兜底（added token 只匹配完整
  拼写）。按拼写 dict 缓存。
- `_marked_parts(text)`：按 `_MARK_OPEN` 切开，每段以 `_MARK_CLOSE`
  取还原拼写；未闭合的 `_MARK_OPEN` 按普通文本回吐（防御有机 PUA）。
- `encode()`：无标记直通 `super().encode`；有标记逐段拼接。
- `__call__()`：serving 主路径（`tokenize=False` → 下游
  `tokenizer(prompt)`）。分块 `super().__call__` 后合并 BatchEncoding：
  `input_ids` 拼 plain ids、`attention_mask` 补 1、`offset_mapping`
  按累计字符偏移重基、其余字段补 0；`truncation`/`max_length` 从分块
  kwargs 剥离、合并后按 `truncation_side` 对全量字段统一切片；
  `return_tensors` 经 `convert_to_tensors` 处理。
- `_dsv41_encode()`：入口先按 `_MARK_OPEN` 切，普通段递归走原有分段/
  缓存/`backend.encode_batch` 快路径——保证 batch 编码永远看不到标记。

## 4. 覆盖与未覆盖

已覆盖的消息字段：所有 role 的 str content、content list 的 text 类
part、`reasoning`/`reasoning_content`、`tool_calls.function.arguments`
（str/dict）、`tools` function 定义、system content。

有意未覆盖（与改动前行为一致，均非本次 400 成因）：

- `tool_calls.function.name` 中的哨兵拼写不转义（模型可输出工具名的
  任意字符串，属病理场景）。

- 控制类拼写（`<｜User｜>`/`<｜Assistant｜>`/`｜DSML｜`/eos 等）在文本中
  被引用仍会编成控制 token → 角色边界污染的**静默**语义问题。彻底修复
  需全量 special-range 编码（约 1300 个 added 拼写全部按纯文本处理），
  每条消息跑大 alternation 正则的性能成本高，超出本次范围。
- `tools` 之外请求级字段（`response_format`、sampling 文本等）中的拼写：
  同上游，不处理。
- Rust 前端 `rust/src/chat/src/renderer/deepseek.rs:411`
  `reject_image_placeholder_text` 未动：只查 `IMAGE_PLACEHOLDER` 一个
  拼写、仍抛 `UnsupportedMultimodalContent`。若部署链路走 Rust frontend
  需镜像此逻辑；当前观察到的报错为 Python 文案，说明线上走 Python 前端。
- `deepseek_v4`/`deepseek_v32` encoding 同样有 `IMAGE_PLACEHOLDER`
  （v4 路径连 guard 都没有，字面量直接静默变图片 token）；部署为 v41，
  未改。

## 5. 注意事项 / 边界

- `tokenize=False` 返回的 prompt **保留标记**（必需：serving 主路径是
  prompt 字符串到下游再 `tokenizer(prompt)`）。标记会出现在 prompt 日志
  中（PUA 不可见字符）；`_text_len_check` 字符计数含标记字符，量级可忽略。
- 标记机制对"误标记"容忍：非 added token 的拼写经 plain tokenizer 编码
  与正常编码 ids 相同；有机出现的 `\ue000`/`\ue001` 不会破坏解析
  （未闭合 OPEN 按文本回吐，孤立 CLOSE 是普通字符）。
- 已标记文本回流（用户把日志里的 marked prompt 粘回会话）→ normalize
  会把标记内的拼写再转义成嵌套标记 → 编码时表现为标记附近少量异常
  token，不报错。可接受的病理场景。
- `encode`/`__call__` 只处理 `str` 输入；`list[str]` 批量调用不走标记
  路径（chat prompt 均为单条编码）。
- `__call__` 合并按"所有分块返回相同字段集合"假设工作；对 BatchEncoding
  中非 list 字段（如 `num_truncated_tokens`）跳过不合并。
- 若部署 tokenizer 的 `IMAGE_PLACEHOLDER` 同时存在于 base vocab（非仅
  added vocab），plain ids 仍会产出图片 id——`mm_preprocess.validate_
  image_sentinel_ids` 的校验前提保证不会出现此情形。
- `_dsv41_encode` 的分段安全边界（换行/空白）不会落在标记内部：标记
  不含空白字符。
- `apply_chat_template` 的 `tokenize=True` 路径（`enable_prompt_embeds`）
  与 `tokenize=False`→`__call__` 路径产出一致的 ids。
- 无哨兵的 `arguments` 经 `_escape_tool_arguments` 逐字节透传（P2 修复），
  正常路径 prompt 在已验证范围内与上游一致；含哨写字段按设计改写。

## 6. 测试方案

### 已完成

- `tests/tokenizers_/test_deepseek_v41.py`（47 个用例全过）：
  - `test_escapes_image_placeholder_in_text`（role×as_part 参数化）/
    `_in_reasoning`/`_in_tool_arguments`/`_pad_spelling`：prompt 中仅存在
    `\ue000…\ue001` 包裹形式、无未包裹裸拼写；
  - `test_literal_image_placeholder_encodes_as_plain_text`：
    FakeHfTokenizer（`encode→[len(text)]`，无 backend → 逐字符兜底）
    验证 tokenize=True 路径拼接；
  - `test_reference_encoder_fixtures`：5 组 1017 参考 fixture 逐字一致，
    证明无字面量的正常路径 prompt 不变（已验证范围内：消息文本字段；无哨兵的 arguments 亦逐字节透传）。
- 真实 tokenizer 端到端（deepseek-recipe bundled v41 tokenizer.json，
  把 129264 的 added 拼写改成 `<｜deepseek_image｜>` 模拟部署态）：
  - 含字面 SPELL+PAD 的 user/tool 消息：`tokenize=False` 与
    `tokenize=True` 两路径 ids 一致，**129264/129265 均不出现**，
    decode 后两种拼写完整可读；
  - 真实 image part 仍精确产出 1 个 129264（槽位计数正确）。
- `ruff check` / `ruff format` 干净。

### 建议补做（部署前）

- GPU 侧：gpu42 `/root/v41-fix/appsrc` 部署后，复跑"贴含拼写文本→多轮
  对话"场景：不再 400、输出正常；
- 带图请求回归：text+image part 混合消息确认 mm 展开槽位对齐
  （`image_order` 计数不受文本字面量影响）；
- `enable_prompt_embeds` 部署形态下 `tokenize=True` 路径实测；
- 长 prompt（>64k）分段编码路径含标记时的 ids 与整串 `__call__` 一致；
- 若 Rust frontend 也在服务，需另行评估 `reject_image_placeholder_text`
  的镜像改造。

## 7. 评审结论（已处理）

- **P1 双重编码 arguments 漏网（已修复）**：初版 `_escape_tool_arguments`
  只 `json.loads` 一轮，`ensure_ascii` 内层 JSON 中的 `\uff5c` 转义序列
  使正则失配；`encode_arguments_to_dsml` 容忍两层编码，第二轮解出裸拼写
  进 prompt（静默编成 129264）。修复：镜像 dsml 的两层解码循环，转义
  最深层值后按相同层数 re-dump，编码层数不变。复测：双重编码 +
  ensure_ascii 的 arguments → 标记 1 处、无裸拼写。
- **P2 无哨兵 arguments 规范化漂移（已修复）**：`"[1,2,3]"` 等可 decode
  但无需转义的字符串曾被 loads→dumps 规范化成 `"[1, 2, 3]"` 落进
  prompt。修复：`escaped == decoded` 时原样返回，无变化时逐字节透传。
  另注：`_normalize_messages` 作用于 `copy.deepcopy(messages)`，不改写
  调用方 request 对象（评审所提改写问题经复测不成立）。
- **`__call__` 空 merged 丢 ids（不成立，坐实）**：`_marked_parts` 首段
  必为 plain（可为 `""`），先于任何 marked 段编码并播种 `merged` 字段。
- **P3 `__call__` 合并路径零覆盖（已补）**：新增
  `test_call_splices_marked_spellings_through_backend`（tiny BPE 真实
  backend：added token 不出现在 ids、attention_mask 全 1、与
  `tokenize=True` 路径 ids 一致）与
  `test_call_rebases_offsets_and_honours_truncation`（offset_mapping
  重基 + truncation 重放）。
- `re` alternation 性能：~4-10 个拼写 vs 消息体，每轮全量回放会跑；
  拼写集合保持在小规模（哨兵区间），暂不作为问题。
