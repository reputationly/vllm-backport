# A100-40G 集群 Token 产能:实测、推导与被证伪的结论

> 硬件:2 节点 × 4×A100-PCIE-40GB / 鲲鹏 ARM64 / 节点内 PCIe 无 P2P / 节点间 RoCE / 每节点 251 GB 内存
> 模型:`DeepSeek-V4-Flash-0731`(167 GB,路由专家 FP4,注意力与共享专家 FP8)
> 需求:1000 亿 token/天(**用户确认**);售价 1.35 / 4.05 元/百万(**用户确认**);成本 2.00 元/卡时(**用户确认**)
>
> 标注:`[实测]` 本文实验所得 · `[推导]` 由数据推算 · `[证伪]` 曾经的结论,已被实测推翻
>
> ⚠️ **本文所有 TPOT 数字都是「每 SSE chunk」,不是「每 token」。** 投机解码下
> 一个 chunk 携带约 2.7 个 token(DSpark 接受长度),所以真实的每 token 间隔约为
> 所列值的 1/2.7。TTFT 与 tokens/s/卡 不受影响(前者是首 chunk 时刻,后者取自
> 服务端 `usage.completion_tokens`)。该口径问题由外部仓的「测量陷阱」清单点出。

---

## 0. 已被实测推翻的结论(先看这个)

| 曾经的判断 | 实测结果 |
|---|---|
| 输入:输出 = 8:1(需求文档所拍) | **计费 100:1,算力 22.8:1**,差一个数量级 |
| 提高 `--max-num-seqs` 到 64/128 能 +14~23% | **b=64 比 b=32 慢 31%**;b=128 起不来。32 已是最优 |
| `VLLM_PP_LAYER_PARTITION=24,19` 有 +5.7% | 真实流量形状下**反而慢**(370.3s vs 353.0s)。那 +5.7% 只在假形状 8288:1024 上成立 |
| 批预算 2048→8192 能救冷 prefill | 只快 2~5%。瓶颈是 kernel,不是批预算(§4) |
| 「V4.1 把 KV 压缩了很多」 | **反的** —— 0731 的 `compress_ratios` 是 4/128,V4.1 是 2/1(§8) |
| 「用 4 节点跑 V4.1 就不用分流了」 | V4.1 在本硬件**根本跑不起来**(三个结构性约束,§8.1) |
| 冷启动 110s 是个普遍问题 | 那是最坏情况。真实流量**中位冷 TTFT 仅 0.1s**,只有 8.4% 的请求 >20s(§7.1) |
| TP4 节点内是唯一选择 | **TP2×PP4 吞吐 +51.6%**,冷 prefill +30~66%(§3.6) |
| 外部仓的 `ROW_CHUNK=64` 修复必须采纳 | 那个崩溃在本 fork **不复现**;64 还慢 6%(§3.7) |
| 外部仓的 EP 否定结论(72σ)适用于我们 | **反的** —— 我们开 EP 快 17.2%(§3.8) |
| 900k 深度召回崩到 30%/0% | **我的 `max_tokens=64` 截断造成的假象。给足预算后 862k 深度 10/10 全中**(§3.9) |
| 随机文本会虚高吞吐 ~35% | 方向反了,自然文本还略快(§3.9) |
| 「8 台机器 rail2 坏了 / 只有 2 条轨可用 / metric 决定可达性 / SDN 没下发端口-IP 绑定」 | **全部作废。RoCE 从没坏过 —— 55 台 × 4 轨 RDMA 全部线速 11.4~11.7 GB/s。是我拿 ping 当健康判据,而 ICMP 与 RoCEv2(UDP/4791)是两条策略路径(§7.4.1)** |
| 合成基准上 TP4×PP2 的 decode 每卡快 36%,该换回去 | **真实形状下反而慢 22.1%**(计费 2409.9 vs 3092.5 tokens/s/卡),轮0 TTFT +49.6s。合成基准的输入:输出是 2.5:1,线上是 100:1(§3.6.1) |
| `--max-num-batched-tokens` 提到 8192 能提吞吐 | 1M 下**装不下**(需 5.07 GiB > 可用 5.06 GiB,启动失败);4096 能装但 −1.7% 且吃掉 41% KV 池(§3.6.1) |
| 卸载缓冲区实用上限是 150 GiB | **上限是 `cudaHostRegister` 的注册次数(约 4.5 万次/进程),不是固定 GiB 数。** 它随 chunk 大小变,而 chunk 大小随并行配置变。本配置下 150 GiB 需 70,595 次,实测失败(§3.3) |
| 投机解码 `num_speculative_tokens` 从 5 降到 3 有 +5.8% | 真实形状下 **−0.2%**,噪声下限的 1/27。那 +5.8% 来自合成基准(§3.6.1) |
| GPU KV 池越大命中率越高 | **无关。** KV 池小 20% 时命中率完全相同(79.5% vs 79.4%)—— 淘汰的前缀落到宿主内存那层,重入 2.79s vs GPU 热态 2.70s(§3.10) |

两条方法论教训:

1. **任何在 8288:1024 这种合成形状上得到的结论,都必须在真实形状上重验。**
2. **合成测试要用真实分位点采样,不要只测最坏情况** —— 我用 490k 全新上下文
   得出「冷启动 110s」,而线上纯输入的中位只有 892 token,两者差 550 倍。
3. **不要照搬别人在不同硬件/不同区间上的结论。** 本轮四条错误全部源于此:
   外部仓是 **8×A100 NVLink + 8K prefill**,我们是 **PCIe 无 P2P + 长上下文**。
   他们的 EP 否定、`ROW_CHUNK` 修复、kernel 开关收益,在我们这儿三条不成立。
4. **推理模型的评测必须给足输出预算。** `max_tokens=64` 让 862k 深度的召回从
   100% 读成 0%(思维链还没写完就被截断)。对齐各组预算,否则测的是预算。
5. **判据必须走要用的那条路。** 同一类错误已犯三次:`max_tokens=64` 当召回判据、
   批量 ping 超时当可达性判据、ICMP 当 RoCE 健康判据。前两次浪费时间,
   第三次更严重 —— 为一个不存在的故障重启了 6 台生产机、改了 7 台的路由。
   **要测 RDMA 就用 `ib_write_bw`,要测召回就给足预算,不要用替代信号。**
6. **合成基准的 workload 形状不匹配线上时,结论可能符号相反 —— 不是偏差,是反的。**
   同一对配置(TP2×PP4 vs TP4×PP2),合成基准(输入:输出 2.5:1、强制 cache miss)
   得出 TP4×PP2 **赢 36%**;真实会话形状(输入:输出 ~100:1、命中率 79.4%)得出
   TP4×PP2 **输 22.1%**。原因是线上的 prompt 九成被缓存省掉,真实开销集中在
   轮0/轮1 的冷 prefill,而合成基准把 decode 的权重放大了约 40 倍。
   **任何拓扑 / 调度 / 投机解码参数的决策,必须用 §3.2 那套会话形状回放复核。
   单请求或短输出的合成基准只能用来定位现象,不能用来做决策。**
7. **一个端点上只能有一个压测客户端。** 实例启动失败后我在同一端口重开,
   而上一轮的 poller 还在轮询,于是两个客户端同时发压 —— 并发翻倍、TTFT 从
   4.75s 读成 103.44s,整组数据作废。重开必须换端口,或先确认旧 poller 已退出。
   并行测多臂时**每臂配一个独立客户端节点**:一个 Python 进程驱动 24 路
   19 万 token 的 SSE 长流,客户端自己就是瓶颈,测出来的 TTFT 全是排队时间。

---

## 1. 真实流量(线上一个用户 17,198 条请求,2026-08-14 ~ 09-12)

### 1.1 比例与命中率

| | 实测 |
|---|---|
| 计费输入 : 输出 | **100.2 : 1** |
| **算力**输入 : 输出(扣掉缓存命中) | **22.8 : 1** |
| **前缀命中率** | **77.3%** |
| 计费 token : 算力 token | **4.26 : 1** |

`1000亿 计费token/天` → 算力只需 **271,767 token/s**,不是 1,157,407。缓存命中直接把算力需求砍掉 4.26 倍。

### 1.2 上下文分布 —— 用户真的在用满 1M

| | 中位 | P75 | P90 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| prompt tokens | 39,543 | 271,849 | 490,194 | 632,362 | 895,214 | **1,045,198** |
| **纯输入**(需真算) | **892** | — | 114,523 | 249,279 | 620,687 | 1,036,067 |

**32% 的请求上下文 >200k,却占了 83.6% 的计费量。** 收入几乎全部来自超长上下文。

注意纯输入中位只有 **892 token** —— 典型请求是「十万级缓存前缀 + 一两千新增」。

### 1.3 覆盖率(决定分池阈值)

| max-model-len | 覆盖请求 | 覆盖计费量 |
|---|---|---|
| 128k | 61.5% | 10.4% |
| 256k | 73.9% | 24.7% |
| **512k** | **90.9%** | **62.3%** |
| 600k | 93.9% | 72.5% |
| 1M | 99.7% | 98.3% |

### 1.4 业务线差异极大

| 业务线 | 请求 | 计费入:出 | 算力入:出 | 命中率 |
|---|---|---|---|---|
| 问数测试* | 17,093 (99.4%) | ~100:1 | 19~25:1 | 75~80% |
| 阿米加智能助手* | 102 (0.6%) | 2.5~3:1 | **2.1~2.6:1** | 15% |

按 §5 的敏感性模型,**阿米加那条线是亏钱的**。定价与调度必须分池。

---

## 2. 与供应商(maas.ovaijisuan.com,同为 0731)的正面对比

### 2.1 延迟

| 上下文 | 我们冷 | 供应商冷 | | 我们热 | 供应商热 | |
|---|---|---|---|---|---|---|
| 40k | 5.36s | 1.73s | 3.2× 慢 | **0.29s** | 1.10s | **4× 快** |
| 140k | 19.32s | 3.41s | 6.0× 慢 | **0.68s** | 2.99s | **4× 快** |
| 490k | 104.7s | 12.36s | 8.5× 慢 | **2.42s** | 5.52s | **2.3× 快** |
| 900k | 286.4s | 27.62s | 10.4× 慢 | **4.26s** | 8.34s | **2× 快** |

- 我们的冷 prefill 速率随上下文**衰减**(7.5k→7.3k→4.7k→3.2k tok/s);供应商**平**(23k→41k→40k→33k)
- 热路径我们 150~220k tok/s,全程碾压

**结论:不要去追冷 prefill,要让冷 prefill 不发生(见 §3)。**

### 2.2 行为一致性(无感知切换)

| 项 | 状态 |
|---|---|
| 默认 reasoning 输出长度 | ✅ 我们中位 420 / 均值 397;供应商 379 / 445。**一致** |
| `reasoning_effort` 参数 | ⚠️ 供应商实现了(默认 405 → max 996);**我们的模板里根本没有这个字段**,传了无效 |
| 流式字段名 | 🔴 **我们 `delta.reasoning`,供应商 `delta.reasoning_content`** —— 客户端按后者解析会丢思维链 |
| `usage.completion_tokens_details.reasoning_tokens` | ✅ 两边都有 |
| 输出可复现性 | ⚠️ 供应商自己 temperature=0 也不可复现。**无感知不能定义成输出一致**,只能是质量/延迟/格式对齐 |

`--default-chat-template-kwargs {"reasoning_effort":"max",...}` 是**死配置**(`tokenizer_config.json` 里零匹配),可以删掉。

### 2.3 供应商实际计价(从费用字段最小二乘反推)

| | 元/百万 |
|---|---|
| 纯输入 | 0.861 |
| **缓存读** | **0.027(纯输入价的 3.1%)** |
| 输出 | 1.646 |

**我们要不要跟这个折扣,直接决定毛利是 85% 还是 55%。这是待定的定价决策。**

---

## 3. KV 卸载:本项目最大的单项优化 `[实测]`

### 3.1 淘汰重入测试

GPU KV 池 638,912 token,一个 490k 上下文即可挤掉另一个。

| 步骤 | TTFT |
|---|---|
| A 首次(冷) | 106.34s |
| A 再次(GPU 热) | 2.70s |
| B、C 各 490k 把 A 挤出 GPU | 106.72s / 107.61s |
| **A 被挤掉后重来** | **2.79s** |

**被彻底淘汰的会话重新命中只要 2.79s,比重算快 38 倍。**

原因:我们 prefill 慢(4.5~7.5k tok/s),而 KV 搬运走 host→GPU 的 PCIe(不受无 P2P 影响,~20 GB/s)。**搬 KV 比重算 KV 快两个数量级。**

### 3.2 真实流量形状(8 会话 × 5 轮累积,文档大小按线上分位点采样)

**同一套硬件(gpu43+44)的对照**,排除机器差异:

| 配置 | 墙钟 | 计费 tokens/s/卡 |
|---|---|---|
| 基线(无卸载) | **1332.3s** | — |
| **开卸载 100 GiB** | **357.0s** | **2380.1** |

**3.7 倍。** 另外两台机器的交叉验证:gpu49+50 卸载 100 GiB → 353.0s / 2407.2;
gpu45+47 卸载 + `PP_LAYER_PARTITION=24,19` → 370.3s / 2295.1(PP 分区**变慢**)。

逐轮 TTFT:

| 轮 | TTFT 中位 | 最大 | 平均 prompt |
|---|---|---|---|
| 0 | 82.46s | 258.47s | 166,401 |
| 1 | 33.68s | 113.71s | 167,921 |
| **2** | **2.27s** | 4.11s | 169,457 |
| 3 | 3.60s | 5.17s | 171,004 |
| **4** | **1.27s** | 3.46s | 172,553 |

**第 2 轮起,17 万 token 上下文只要 1.3~3.6 秒。** 真实流量正是多轮累积,绝大多数请求落在第 2 轮之后。

### 3.3 卸载缓冲区大小

默认后端是 `/dev/shm`,上限为宿主内存的一半(实测 134.6 GB free)。超过就报:

```
/dev/shm has 134.6 GB free but the KV offload region needs 193.3 GB
... or set VLLM_KV_OFFLOAD_REGION_BACKEND=memfd
```

`memfd` 后端可突破 `/dev/shm`,但会在 `cudaHostRegister` 处撞墙。

**墙是注册次数,不是字节数,也不是剩余内存 `[2026-09-15 修正]`**

原先这里写的是「实用上限约 150 GiB」。那个数字只在当时那组 chunk 大小下成立,
不是通用上限。三次实测:

| flag | chunk 大小 | 总 chunk 数 | 结果 |
|---|---|---|---|
| **75**(现网) | 2.28 MB | 35,297 | ✅ `pinned 35297 chunks in 35297 registrations (10.06 GB)` |
| 150 | 2.28 MB | 70,595 | ❌ rank0 死在 59365、rank1 死在 45452 |
| 300 | 2.28 MB | 141,190 | ❌ 死在 49419 |
| 200(旧配置) | 4.19 MB | 51,200 | ❌ 死在 41270(≈161 GiB) |

失败点全部落在 **4.1~5.9 万次注册**,而 chunk 大小差 1.8 倍、失败时的已注册字节
从 56 GB 到 173 GB 不等。失败时宿主机 251 GiB 里还空着 176 GiB —— **和内存余量无关**。

```python
# vllm/v1/kv_offload/cpu/spec.py:104-120
num_copies = 1 if replicated_layout else world_size
kv_bytes_per_chunk = worker_kv_bytes_per_block * num_copies * blocks_per_chunk
num_chunks = int(cpu_bytes_to_use) // round_up(kv_bytes_per_chunk, BLOCK_SIZE_ALIGNMENT)
```

`chunk` 大小由 `world_size`、`blocks_per_chunk`、每 block 的 KV 字节共同决定,
**换并行配置就会变**。所以可填的最大 GiB 也跟着变:

```
可填上限 ≈ 45,000 × chunk 大小
本配置(TP2×PP4,chunk 2.28 MB)→ 约 100 GB ≈ 95 GiB
```

**现网填 75(35,297 次注册)已经贴着天花板,没有加的空间。** 想榨到 90 也可以
(约 42,000 次),但按下表 8 会话量级的收益只有 1.4%(噪声),而撞墙的代价是
实例 pending 十几分钟后才失败,比直接报错更难排查。不值得。

> 这也解释了 §6 那句「TP4×PP2 用 150,TP2×PP4 必须减半到 75」**是对的**。
> 我曾按代码算式推断「每节点只 pin flag/2,所以能填 300」,实测 300 和 150 都失败。
> 推导输给实测。

| 卸载大小 | 8 会话 | 24 会话 |
|---|---|---|
| 100 GiB(`/dev/shm`) | 353.0s / 2407.2 | 1777.8s / **1433.9** |
| 150 GiB(`memfd`) | 348.0s / 2441.6 | 1561.0s / **1632.6** |

**低并发下没区别;24 会话时 150 GiB 快 13.9%。** 容量只在超订严重时才兑现。

### 3.4 并发会话数:约束是延迟,不是吞吐 `[实测]`

| 会话数 | 平均上下文 | 活跃上下文总量 | 超订率 | tokens/s/卡 | 第2轮(热)TTFT 中位 |
|---|---|---|---|---|---|
| **8** | 172k | 1.38M | **2.2×** | **2441.6** | **3.2s** |
| **12** | 130k | 1.56M | **2.4×** | 2132.6 | **1.2~2.4s** |
| 16 | 166k | 2.76M | 4.3× | 1796.7 | **56~73s** |
| 24 | 172k | 4.14M | 6.5× | 1632.6 | **157.3s** |

超订率 = 活跃上下文总量 ÷ GPU KV 池(638,912)。**以上全部在 TP4×PP2 下测得。**

换成 TP2×PP4 后悬崖大幅后移(自然文本语料):

| 拓扑 | 会话 | tokens/s/卡 | 后段(热)TTFT 中位 |
|---|---|---|---|
| TP4×PP2 | 16 | 1796.7 | **56~73s** |
| **TP2×PP4** | **16** | **3271.2** | **2.2~2.5s** |
| **TP2×PP4** | **8** | **3773.2** | **1.4~2.2s** |

**同样 16 会话,换拓扑后 TTFT 从 56~73 秒降到 2.2~2.5 秒,吞吐还高 82%。**
所以「16 会话就过悬崖」只对 TP4×PP2 成立;**TP2×PP4 的甜点是 8~16 会话**。

两条仍然成立的结论:

1. **吞吐随会话数单调下降**(8 会话最高)。**加会话只买到延迟,买不到吞吐。**
2. **约束是延迟,不是吞吐。** 定运行点要看 TTFT 的 SLO,不是看 tokens/s。

`--max-num-seqs 32` 在长上下文下是虚的:32 个 170k 会话需要 540 万 token 的 KV,
调度器实际只放行约 3~4 个,其余排队 —— 所以必须在**网关**侧限流,靠引擎参数拦不住。

`[推导]` **网关准入规则**(用 token 总量而非会话数,因为上下文跨度极大):

```
TP4×PP2: 活跃上下文总量 ≤ 2.2 × KV池 ≈ 1.4M token  → 8~9 个长会话
TP2×PP4: 实测 16 会话(约 2.2M token)仍然健康 → 上限至少 16 个长会话
```

中位 39.5k 的小会话可到 32 个(被 `max-num-seqs` 封顶)。

### 3.5 机队规模(首次基于真实形状实测) `[推导]`

```
单实例(TP2×PP4,2 节点 8 卡,8 会话运行点)= 30,186 计费 tokens/s
需求 1000 亿/天                            =  1,157,407 计费 tokens/s
→ 38.3 个实例 = 77 个节点 = 307 张卡
```

| 计价策略 | 日成本 | 日收入 | 毛利 |
|---|---|---|---|
| 缓存读收全价 | 14,736 元 | 164,700 元 | **91.1%** |
| 缓存读按供应商折扣(3.1%) | 14,736 元 | 53,470 元 | **72.4%** |

**307 张卡**,而不是 §5 用合成形状外推的 1,560~2,744 张。三个来源:
真实流量算力比 22.8:1(非文档所拍的 8:1)、77.3% 前缀命中把计费量放大 4.26 倍、
以及 TP2×PP4 拓扑(+51.6%)。

规模演进记录(同一需求,认知逐步纠正):

| 依据 | 卡数 |
|---|---|
| 合成形状 8288:1024 + 假设 8:1 | 1,560~2,744 |
| 真实流量形状 + TP4×PP2 | 474 |
| **真实流量形状 + TP2×PP4 + 自然文本语料** | **307** |

> §5 的敏感性模型仍可用于判断「什么样的流量结构会亏钱」,
> 但**规模测算应以本节为准** —— 它是真实多轮累积形状下的实测。

---

## 3.6 拓扑:减少 TP 是冷 prefill 的大杠杆 `[实测]`

线索来自外部仓 [`allover326/deepseek-v4-cmp170hx`](https://github.com/allover326/deepseek-v4-cmp170hx):
在无 P2P 的 PCIe 上,TP 每次前向要做 `2 × 43 = 86` 次 all-reduce,而 PP 每个 stage
边界只搬一次 hidden states。他们在 **PCIe Gen2 x4** 上测到 PP4 比 TP4 快 **6.6 倍**。

我们此前**一直是 TP4 节点内**,从未试过减少 TP。同样 8 张卡:

| 拓扑 | 40k 冷 | 140k 冷 | 490k 冷 | KV 池 |
|---|---|---|---|---|
| TP4×PP2(原配置) | 5.36s / 7,516 | 19.32s / 7,297 | 104.7s / 4,710 | 638,912 |
| **TP2×PP4** | **4.12s / 9,793** | **12.41s / 11,370** | **63.25s / 7,798** | 674,060 |
| 提升 | **+30%** | **+56%** | **+66%** |

**冷 prefill 快 30~66%,490k 从 105 秒降到 63 秒。** 幅度小于他们的 6.6× —— 因为
我们是 PCIe Gen4 x16(带宽约 10 倍),TP 的通信惩罚本来就轻得多。

`TP1×PP8` 起不来:`No available memory for the cache blocks`。TP1 时每卡要独自承担
全部 8192 token 的激活(无 TP 分片),显存不够。用批 2048 可重试,但 TP2×PP4 已够用。

### 3.6.1 六个候选的全量对照:全部否掉 `[实测 2026-09-15]`

起因是和同事的线上实例(同为 8 卡跨 2 节点、A100-PCIE-40GB)正面对比,发现
**decode 每卡我们 40.2、他们 57.9(−31%)**。为定位原因,在 gpu31-40 上手工起了
五个隔离对照臂(各 8 卡、独占 2 台机器),与现网配置逐变量对比。

**配置差异(同事 vs 我们)**

| | 我们(现网) | 同事 |
|---|---|---|
| TP × PP | **2 × 4** | **4 × 2** |
| `--max-num-batched-tokens` | **2048** | **8192** |
| `--max-model-len` | **1,048,576** | 400,000 |
| `--kv-offloading-size` | **75** | 无 |
| `--disable-custom-all-reduce` | 有 | 无 |
| KV 池 | **3,611,775**(并发 3.44×) | 539,151(并发 1.35×) |
| 累计前缀命中率 | — | **89.3%**(生产流量积累) |
| 思维链字段 | `reasoning_content` | `reasoning` |

**第一阶段:合成基准(并发 32,prompt≈1k,out≤1024,唯一前缀强制 miss)**

| 臂 | 上下文 | 批 | 拓扑 | spec | decode 每卡(热) | TTFT | KV 池 |
|---|---|---|---|---|---|---|---|
| 现网 | 1M | 2048 | TP2×PP4 | 5 | 40.2 | 4.20s | 3,611,775 |
| **D** 基线复现 | 1M | 2048 | TP2×PP4 | 5 | **41.2** | 4.16s | 3,611,775 |
| A' | 400K | 8192 | TP2×PP4 | 5 | 40.8 | 4.75s | 528,476 |
| **C** | 1M | 2048 | **TP4×PP2** | 5 | **55.9** | 6.03s | 2,890,000 |
| E | 1M | **4096** | TP2×PP4 | 5 | 40.5 | 4.31s | 2,126,396 |
| F | 1M | 2048 | TP2×PP4 | **3** | 43.6 | 4.43s | 3,607,724 |
| 同事参照 | 400K | 8192 | TP4×PP2 | 5 | 57.9 | 5.96s | 539,151 |

D 复现出现网的 41.2 vs 40.2(差 2.5%),基线可信。合成基准的结论是
**「差距全在拓扑」**:A' 把上下文和批预算一起改只动了 +1.5%,而 C 换拓扑 +36%。

**第二阶段:真实会话形状回放(8 会话 × 5 轮,文档按 §1.2 分位点采样,均值 19 万 prompt)**

这一步推翻了第一阶段的结论。三臂并行、**每臂独立客户端节点**:

| 臂 | 计费 tokens/s/卡 | 均值 | vs 现网 | 轮0 TTFT | 轮1 | 轮2-4 | 命中率 |
|---|---|---|---|---|---|---|---|
| **D** 现网配置 | 3052.9 / 3028.3 / 3196.4 | **3092.5** | — | 69.7s | 2.91s | 2.4~2.8s | 79.4% |
| **F** spec 3 | 3056.3 / 3116.3 | **3086.3** | **−0.2%** | 72.8s | 2.96s | 2.4~2.7s | 79.4% |
| **C** TP4×PP2 | 2293.3 / 2526.4 | **2409.9** | **−22.1%** | **119.3s** | **34.3s** | 2.5~2.6s | 79.5% |

D 三次跨度 3028.3~3196.4 = **5.5%,这是本 harness 的噪声下限**。

harness 的形状与线上对得上,所以它的结论可采信:

| | 本 harness | 线上真实流量(§1.1) |
|---|---|---|
| 前缀命中率 | 79.4% | 77.3% |
| 计费:算力 | 4.80:1 | 4.26:1 |
| 轮2 起 TTFT | 2.4~2.8s | §3.2 实测 1.3~3.6s |

**结论表**

| 候选 | 判据 | 结论 |
|---|---|---|
| 批预算 8192 + 1M | 需 5.07 GiB > 可用 5.06 GiB,`_check_enough_kv_cache_memory` 启动失败 | **物理不可能** |
| 批预算 4096 | 合成 −1.7%,代价 41% KV 池(3.61M→2.13M) | 否 |
| 上下文 400K | decode 40.8 vs 41.2,和批预算一起改也只 +1.5% | **不是杠杆** |
| **拓扑 TP4×PP2** | **真实形状计费 −22.1%(两次都输),轮0 TTFT +49.6s、轮1 +31.4s** | **否** |
| **spec 5→3** | **真实形状 −0.2%**(合成基准的 +5.8% 是假信号) | **否** |
| 卸载 150 / 300 | `cudaHostRegister` 注册次数撞墙(§3.3) | **物理不可能** |

**`--max-num-batched-tokens` 这条线彻底关闭:1M 下 2048 是唯一可行值,而且提高它
对 decode 毫无帮助。** §6 原先那句「批预算要降到 2048…只影响冷 prefill 2~5%」的
前半句是硬约束(不是偏好),后半句的适用范围只限冷 prefill —— 但补测表明它对并发
decode 也没有影响,所以这个取舍比原先描述的更便宜。

**投机解码的接受率与 spec 深度**

| spec | 接受率 | 每步接受 tok(上限 spec+1) |
|---|---|---|
| 5 | 28.1~28.9% | 2.41~2.44 |
| 3 | 40.8~41.7% | 2.22~2.25 |

画 5 个 draft 只中 28%,减到 3 个能中 42% —— 后两个位置基本浪费。但每步接受量
从 2.43 降到 2.25,两者抵消,真实形状下净效果 −0.2%。**保持 5。**
同事那台在同一批请求下的增量接受率是 27.3%,与我们的 28.9% 一致,
**排除了「draft 质量差异」这个怀疑对象**。

---

## 3.7 1M 多轮累积:已验证 `[实测]`

此前只验证过**一次性** 1M prefill。多轮累积是不同的代码路径(前缀缓存复用历史 KV),
外部仓在 `ROW_CHUNK=128` 下于 **718~733k 崩溃**(复现两次,`CUDA illegal memory access`)。

配置 TP2×PP4 / `max-model-len 1048576` / 批 2048,KV 池 **3,406,162 token**(并发 3.25×),
每轮追加 2 万 token,连续 50 轮:

> `[2026-09-15 更新]` 这个 KV 池数字来自 09-12 的构建。同一套参数在 `85d0e70c`
> (09-14)上实测是 **3,611,775 token / 并发 3.44×**,多 6%。现网 gpustack 实例与
> 手工起的隔离对照臂都是这个数,两台不同机器上完全一致。

| `VLLM_DSV4_LOGITS_ROW_CHUNK` | 最终深度 | 第50轮 TTFT | 结果 |
|---|---|---|---|
| **128(默认)** | **1,006,903** | **12.28s** | ✅ 全程无错 |
| 64 | 1,006,893 | 13.02s | ✅ 但慢 6% |

**外部仓的崩溃在本 fork 上不复现,不要照搬他们的 `ROW_CHUNK=64` 修复。**
本 fork(2026-09-12)比他们的基线(2026-08 的 `f8ea5bb`/`c3046d1`)新一个月。

> 已知的质量边界(外部仓实测,我们未复核):检索召回 150k 时 ~100%,**900k 时 ~30%**。
> **能装进 1M ≠ 1M 全程可用** —— 对外承诺长上下文时要讲清这条。

## 3.8 Expert-parallel:我们该开,外部仓该关 `[实测]`

外部仓在 **8×A100 NVLink、TP=8** 上测到 `--enable-expert-parallel`
**TTFT +50.8 ms(72σ)**,判为负收益。我们在 TP2×PP4 上实测相反:

| 配置(自然文本 8 会话) | 墙钟 | tokens/s/卡 |
|---|---|---|
| **TP2×PP4 开 EP** | **187.6s** | **3773.2** |
| TP2×PP4 关 EP | 220.0s | 3219.5 |
| TP4×PP2 开 EP | 284.5s | 2488.6 |

**开 EP 快 17.2%。** 机制上说得通:TP2 时 all2all 只跨 2 个 rank(节点内),
而更好的 GEMM 形状照样受益;TP=8 时 all2all 跨 8 个 rank,代价盖过收益。

## 3.9 上下文深度的真实能力边界 `[实测]`

大海捞针:每个深度种 **10 根针**,均匀分布在 5%~95% 位置(遵循「单探针不算验证」),
语料为容器内 41 MB 的真实 Python 源码,每会话独占区间。

| 深度 | `max_tokens=64` | `max_tokens=4000` |
|---|---|---|
| 38,305 | 50% | **100%** |
| 132,401 | 100% | **100%** |
| 223,306 | 90% | **100%** |
| 407,611 | 80% | **100%** |
| 664,899 | **0%** | **100%** |
| 862,604 | **0%** | **100%** |

**全深度 10/10。** 左列那条「断崖」完全是输出预算造成的:`finish_reason=length`,
模型的思维链里已经正确列出了答案,只是没来得及写进 `content`。

> 外部仓报告的「900k 召回 ~30%」在本配置上**不成立**。但请注意本测试是
> 带显式标记(`【重要记录】`)的事实检索 —— **100% 捞针 ≠ 900k 深度的完整理解能力**。

自然文本 vs 随机文本(同形状、同会话数):自然文本 **2488.6** vs 随机 2441(TP4×PP2),
**略快**。所以此前用随机中文字符得到的吞吐数字没有被高估。
压测实测前缀命中率 **79.4%**,与线上 77.3% 吻合,说明形状造对了。

---

## 3.10 GPU KV 池的真实作用:是并发天花板,不是命中率 `[实测 2026-09-15]`

开了 `--kv-offloading-size` 之后,KV 变成两层:GPU 是一级,宿主内存是二级。
两层的容量关系要按**全局**口径算,别按每节点:

```
一级(GPU)   = 8 卡 × 11.46 GiB = 91.7 GiB 全局
二级(宿主)  = cpu_bytes_to_use = 75 GiB 全局
              每节点 mmap 一份 80.53 GB 的 region(sparse),
              但每个 rank 只 pin 自己的 1/8 切片:10.06 GB × 8 = 80.5 GB = 75 GiB
              每节点实际 pin = 4 个本地 rank × 10.06 = 40.24 GB
```

**所以二级是一级的 0.82 倍,略小,不是更大。** 曾误算成「每节点 75 GiB × 2 节点
= 150 GiB、比一级大 1.6 倍」—— 那是把全局量当成了每节点量。

§3.6.1 的三臂给出了直接证据 —— **KV 池差 20%,命中率完全相同**:

| 臂 | GPU KV 池 | 1M 并发 | 真实形状命中率 |
|---|---|---|---|
| D | 3,611,775 | 3.44× | **79.4%** |
| C | 2,890,000(**−20%**) | 2.76× | **79.5%** |
| E | 2,126,396(**−41%**) | 2.03× | (合成臂,未跑形状) |

配合 §3.1 的淘汰重入实测(GPU 热态 **2.70s** vs 被挤出后重入 **2.79s**,只差 3%,
比重算的 106.34s 快 38 倍),结论是:

> **被挤出 GPU 的前缀不会丢,落到宿主内存那层,命中照样算命中。
> 所以 GPU KV 池的大小不决定命中率。**

那它决定什么?**高并发时的人均上下文上限。** 每个在跑的请求,上下文里每个 token
的 KV 都要驻留到请求结束:

```
3,611,775 ÷ max_num_seqs(32) = 每条请求平均 112,868 token
```

**32 并发时人均上下文超过约 11.3 万就开始抢占。** 对照 §1.2 的线上分布:

| | prompt tokens | 32 并发时 |
|---|---|---|
| 中位 | 39,543 | 安全(用 35%) |
| **P75** | **271,849** | **超 2.4 倍** |
| P90 | 490,194 | 超 4.3 倍 |

§3.6.1 那几轮(8 会话 × 19 万 = 1.59M,占池子 44%)**抢占次数全部为 0**,
说明当前并发水平下这个天花板还没碰到。

**评估「用 KV 容量换别的东西」这类选项时,不要按 KV token 数的降幅算损失。**
要看两件事:二级缓存是否装得下被挤出的部分(现在余量充足),以及并发是否高到
让容量真正成为约束(§3.3:8 会话下 100/150 GiB 无差别,24 会话才差 13.9%)。
C 臂输掉 22.1% 与它少 20% 的 KV 池**无关** —— 它输在冷 prefill(轮0 TTFT
119.3s vs 69.7s)。

---

## 4. 无效与失败的尝试 `[实测]`

| 尝试 | 结果 |
|---|---|
| `--max-num-batched-tokens` 2048 → 8192 | 冷 prefill 只快 2~5%。**批预算不是瓶颈** |
| `VLLM_SPARSE_RAGGED_FAST_SCAN=1` | 无效(7,109 vs 基线 7,297 tok/s),尽管文档描述完全对上我们的 8192 形状 |
| `VLLM_MARLIN_FP8_DEQUANT_BF16=1` | 无效(6,728) |
| 上面两个同时开 | 无效(7,064) |
| `VLLM_MARLIN_INPUT_DTYPE=int8` | **引擎起不来** |
| `--max-num-seqs` 64 | **−31%**(290 vs 421.8 tokens/s/卡),TPOT 34→236ms |
| `VLLM_INDEXER_QUERY_SHARD=1` | −3% |
| `--gpu-memory-utilization 0.90` | 运行期 OOM |
| PP3 / PP4 | −25% / −65% |
| **`VLLM_UNREPLICATE_ATTN_GEMMS=1`** | **发布态直接崩引擎**(已修,见 `9a53df78b`);修好后在 TP=2 上冷 prefill **慢 1.3~3.6%** |
| **`--kv-cache-dtype fp8_ds_mla`**(对照通用 `fp8`) | **打平**(+0.5%,噪声内),但冷 TTFT 更差(81.5s vs 73.7s) |
| **`VLLM_DISABLE_MULTI_STREAM_PARALLEL=1`** | **−6.6%** —— 多流重叠是有效的,别关 |
| **`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`** | **−1.8%** —— 同上,别关 |
| **拓扑 TP4×PP2**(2026-09-15 复测) | **真实形状计费 −22.1%**,轮0 TTFT +49.6s。合成基准上它 decode +36%,是假信号(§3.6.1) |
| **`--max-num-batched-tokens` 4096 / 8192**(1M 下) | 4096 **−1.7%** 且吃掉 41% KV 池;8192 **启动失败**(需 5.07 GiB > 可用 5.06 GiB)(§3.6.1) |
| **`num_speculative_tokens` 5→3** | 真实形状 **−0.2%**。接受率从 28% 涨到 42%,但每步接受量从 2.43 降到 2.25,抵消(§3.6.1) |
| **`--max-model-len` 降到 400K** | decode 无变化(40.8 vs 41.2)。上下文长度不是产能杠杆,而 400K 只覆盖 ~46% 计费量(§1.3) |
| **`--kv-offloading-size` 150 / 300** | `cudaHostRegister` 注册次数撞墙(约 4.5 万次/进程),75 已贴天花板(§3.3) |
| `VLLM_SPARSE_PREFILL_EXACT_TILE=1` | **读代码即排除**:生效条件是 `num_heads == BLOCK_H`,注释注明只在 TP=8 成立,TP=2 下是空操作 |
| `VLLM_INDEXER_QUERY_SHARD_QPATH=1` | 依赖 `VLLM_INDEXER_QUERY_SHARD`(已实测 −3%),不再单测 |
| **`VLLM_MHC_POST_FUSE_SQRSUM=1`** | **引擎起不来**:`tilelang.py:867` 导入的 `mhc_post_sqrsum_tilelang` 在 `tilelang_kernels.py` 里**根本没有定义**,且无 Triton 回退。本硬件无 DeepGEMM,必然走进这条坏分支 |
| **`VLLM_MHC_PRENORM_SHARD=1`**(单开) | **空转**:`triton.py:158-162` 的门控要求 `sqrsum is None`,而那恰好等价于「`POST_FUSE_SQRSUM` 已开」;注释里写明这个配对是强制的。所以单开永不生效 —— 两个 mHC 优化被同一个缺失 kernel 一起堵死 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0`(想借此打开 torch.compile) | **由构造即排除**:`@support_torch_compile` 只加在 `deepseek_v4/cpu/model.py`,CUDA 版模型没有装饰器,GPU 上 torch.compile 永不生效。关掉 breakable cudagraph 只会两头落空,引擎直接报 `piecewise CUDA graphs unavailable, model is not torch-compiled` |
| **`VLLM_MARLIN_USE_ATOMIC_ADD=1`** | 确认已生效(开启后那条 "consider set ...ATOMIC_ADD to 1" 的建议日志消失),但**无可测差异** |
| **`VLLM_DSPARK_VOCAB_SHARD=1`** | 无可测差异(配置为 greedy 草稿,条件满足) |
| **`VLLM_SPARSE_DECODE_MAXNREG=128`** | 无可测差异 |

> **这一轮的方法论结论比结果更重要:并发压测分辨不了 kernel 开关。**
> 同配置两次重复(不同 salt、全新文档)的波动:
>
> | arm | rep1 | rep2 | 均值 |
> |---|---|---|---|
> | 基线 | 2724.6 | 2613.9 | 2669.2 |
> | `MHC_PRENORM_SHARD` | 2650.8 | 2659.3 | 2655.1 |
> | `MARLIN_USE_ATOMIC_ADD` | 2744.0 | 2631.1 | 2687.6 |
> | `DSPARK_VOCAB_SHARD` | 2575.6 | 2676.6 | 2626.1 |
> | `SPARSE_DECODE_MAXNREG` | 2618.5 | 2711.2 | 2664.8 |
>
> **重复间波动约 4%,而五个均值全在 2669 ± 30(±1.1%)内。** 单次读到的
> 「−5.5%」「−3.9%」在第二次重复里直接翻了符号。
> 换成串行隔离探针后分辨率是 0.2~1.0%,五个 arm 在每个尺寸上都无差异:
>
> | 尺寸 | 基线 | PRENORM | ATOMIC | VOCAB | MAXNREG | 极差 |
> |---|---|---|---|---|---|---|
> | 20k | 7967 | 7968 | 8000 | 7952 | 7920 | 1.0% |
> | 60k | 11186 | 11189 | 11216 | 11196 | 11215 | 0.3% |
> | 140k | 11180 | 11186 | 11206 | 11196 | 11201 | 0.2% |
> | 260k | 9795 | 9823 | 9792 | 9808 | 9813 | 0.3% |
>
> 另外:**「无差异」必须区分「生效了但没用」和「根本没生效」。**
> `MHC_PRENORM_SHARD` 属于后者(门控未满足),这是读代码才发现的。

冷 prefill 撬不动。用隔离探针(串行单请求、每次全新文档、无缓存无排队、
各 2 次重复,两次相差 ~1%)测得的真实速率曲线是**非单调**的:

| prompt | 冷 prefill 速率 | TTFT |
|---|---|---|
| 21k | 7,891 tok/s | 2.7s |
| **63k** | **11,035 tok/s** | 5.7s |
| **148k** | **11,028 tok/s** | 13.4s |
| 274k | 9,701 tok/s | 28.3s |

比原先记录的「稳定 ~7,300」要好,峰值在 63k~148k,两端都衰减。
**测冷 prefill 必须用这种隔离探针** —— 并发压测里的第 0 轮 TTFT 中位
主要由排队顺序决定:同一组 arm,并发下读到「73.65s → 22.8s,快 3 倍」,
隔离探针下方向是反的(慢 1.3~3.6%)。

日志里的结构性原因:

```
custom_all_reduce.py:237  Custom allreduce is disabled because it's not
                          supported on more than two PCIe-only GPUs
sparse_attn_indexer.py:1082  DeepGEMM not supported on this platform;
                          using Triton fallback for sparse attention indexer
```

**TP4 all-reduce 只能走 NCCL 绕 PCIe,稀疏 indexer 只能走 Triton 回退 —— sm80 + 无 P2P 的硬伤,没有开关能绕。**

### 4.1 Triton JIT 缓存**不能**放共享 NFS `[实测]`

`docker/run_multinode_sm80.sh` 默认把 `/root/.cache` 与 `/root/.triton`
挂到 `/nfs-models/vllm-backport-cache*`,本意是「一台热身完其余节点直接复用」。
**多实例并发下这会把引擎打死**,而且它在两个维度上都更差。

5 个 TP2×PP4 实例(40 个 rank 进程)并发压测时,其中 **2 个**在第一轮就挂:

```
OSError: [Errno 116] Stale file handle
  在 deepseek_v4/.../combine_topk_swa_indices 的 Triton kernel 启动处
→ EngineCore 挂掉 → 之后所有请求 HTTP 500
```

ESTALE 是 NFS 的错误码。缓存规模决定了这是个元数据密集负载,不是带宽负载 ——
**1.9 GB / 40,080 个文件**。实测各介质(2000 个 50 KB 文件,写用临时文件 + rename,
与 Triton 的落盘方式一致):

| 介质 | 写 | 读 |
|---|---|---|
| **本地盘** | 2.9s(**690 文件/s**) | **1.0s** |
| NFS | 24.8s(81 文件/s)**慢 8.5×** | 10.2s **慢 10×** |
| tmpfs | 2.6s(769 文件/s) | 1.0s |

也就是说共享 NFS 缓存既会打死引擎,读 4 万个文件还慢 10 倍
(本地约 40s vs NFS 约 200s)——「一台热身完其余复用」实际上在**拖慢**启动。

**结论:用本地盘。** `CACHE_HOST_DIR=/var/cache/vllm-backport`。
空间不是问题(需要 1.9 GB,节点本地盘空闲 673 GB)。

顺带否掉「NFS 只读种子 + 本地写」这个折中:`cp -r` 那 1.9 GB / 4 万文件
实测 **191s**,比本地重新 JIT 还贵。

> 机队规模上这是硬约束:35 个实例 × 8 rank = 280 个进程共写一份 NFS 缓存。

---

## 5. 经济性敏感模型

标定(从 r=8.09、b=32、零复用的 421.8 tokens/s/卡 与 prefill 占 30% 反推):

```
纯 prefill 速率 p = 1,251 tokens/s/卡
纯 decode  速率 d =    66.3 tokens/s/卡      (差 19 倍)
T(r) = (r+1) / (r/p + 1/d)
混合单价 P(r) = (1.35r + 4.05)/(r+1)
```

独立校验:r=4 预测 273 / 实测 222 → **模型偏乐观 23%**,输出重的一侧真实更差。

| 入:出 | 混合单价 | 盈亏平衡 | 可达(b=32 零复用) | 毛利 |
|---|---|---|---|---|
| 1:1 | 2.700 | 206 | 126 | −64% |
| 2:1 | 2.250 | 247 | 180 | −37% |
| 4:1 | 1.890 | 294 | 273 | −7.5% |
| **4.8:1** | 1.816 | 306 | 306 | **0%** |
| 8:1 | 1.650 | 337 | 419 | +20% |
| 16:1 | 1.509 | 368 | 610 | +40% |
| 32:1 | 1.432 | 388 | 812 | +52% |

**真正的盈亏平衡不是「337 tokens/s/卡」,是「输入:输出 ≥ 4.8:1」。** 真实流量算力比 22.8:1,远在安全区。

> 需求文档里的 **680 tokens/s/卡是作者自选的 50% 毛利档位**(`HANDOFF:544`),不是用户需求;且它从 1125 下调而来(`HANDOFF:548`)。**不应作为目标对待** —— 毛利是结果,不是输入。

---

## 6. 推荐配置

```bash
--tensor-parallel-size 2 --pipeline-parallel-size 4   # ★ 不是 TP4×PP2,见 §3.6
--max-model-len 524288           # 覆盖 90.9% 请求 / 62.3% 计费量;超出的走长池或供应商
--max-num-batched-tokens 8192
--max-num-seqs 32                # 实测最优;64 反而 −31%(TP4×PP2 下测)
--gpu-memory-utilization 0.85    # 0.90 会 OOM
--block-size 256
--kv-cache-dtype fp8
--enable-prefix-caching
--kv-offloading-size 75 --kv-offloading-backend native    # ★ 同硬件对照 3.7×
                                 # 该值不按 PP 摊分:每节点用量 = (PP/节点数) × 本值。
                                 # TP4×PP2 用 150,TP2×PP4 必须减半到 75,否则
                                 # cudaHostRegister 失败。>125 GiB 另需 memfd 后端。
--enable-expert-parallel         # 保持开:我们 +17.2%(外部仓在 TP=8 下相反,见 §3.8)
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4
--enable-auto-tool-choice --chat-template-content-format string
# 不要传 --enforce-eager;不要传 --default-chat-template-kwargs(死配置)

环境变量:
VLLM_DETERMINISTIC_MOE_ALIGN=0                 # +9%
VLLM_KV_OFFLOAD_REGION_BACKEND=memfd           # 卸载 >125 GiB 时必需
VLLM_REASONING_OUTPUT_AS_REASONING_CONTENT=1   # 无感知切换必需,见 §2.2
NCCL_IB_GID_INDEX=3  NCCL_ALGO=Ring  NCCL_PROTO=Simple
# NCCL_IB_HCA 不必再裁成 mlx5_0,mlx5_2 —— 4 轨都是线速(§7.4.1)
# 不要设 VLLM_PP_LAYER_PARTITION(真实形状下变慢)
```

> **`--max-model-len` 填 524288 还是 1048576?**
> 1M 已实测可用(§3.7:50 轮累积到 1,006,903,针召回 100%),所以**能填 1M**,
> 而且填了就不必在网关做长度分流(1M 覆盖 99.7% 请求 / 98.3% 计费量)。
> 代价:批预算**必须**降到 2048(§3.7 那组就是 `max-model-len 1048576` + 批 2048)。
> 这是硬约束不是偏好 —— 1M + 批 8192 启动就失败(需 5.07 GiB KV > 可用 5.06 GiB),
> 4096 能起但吃掉 41% 的 KV 池且 −1.7%(§3.6.1)。
> 而这个代价比原先以为的更小:批预算 8192→2048 不仅只影响冷 prefill 2~5%(§4),
> 对**并发 decode 也没有影响**(400K 下 8192 与 2048 分别是 40.8 / 41.2 每卡,§3.6.1)。
> 贵的是 `max-model-len` 本身:400K→1M 在批 8192 下要多花 5.57 GiB/卡,
> 因为 sparse indexer 元数据与 CUDA graph 地址空间都随它线性增长。
> 512K 的唯一理由是「只想覆盖 90.9% 请求 / 62.3% 计费量、把长尾推给供应商」。
> **默认建议 1M**,除非明确要做长度分流。

> **上面这份配置不需要改引擎源码**,只是启动参数与环境变量。
>
> 但本轮调优确实产生了两个引擎改动,都与上面的配置无关(默认不生效):
>
> | commit | 内容 | 上线是否需要 |
> |---|---|---|
> | `f1c429fd8` | `VLLM_REASONING_OUTPUT_AS_REASONING_CONTENT`:思维链按 `reasoning_content` 输出 | **需要**(无感知切换,见 §2.2) |
> | `9a53df78b` | 修 `VLLM_UNREPLICATE_ATTN_GEMMS` 会崩引擎的 bug | 不需要(该开关已否,见 §4) |

### 6.1 在 GPUStack 上怎么填这些参数 `[已核对源码]`

GPUStack 托管的实例不走 `docker/run_multinode_sm80.sh`,参数填在**模型的两个字段**里。

**① 后端参数 → 模型的 `backend_parameters`(UI 里"后端参数"那一栏)**

填法就是把上面那串命令行参数**逐个 token** 填进去,GPUStack 会自己规范化成
`--k=v`(`utils/command.py:170 format_backend_parameters`):

```
--tensor-parallel-size=2
--pipeline-parallel-size=4
--max-model-len=524288
--max-num-batched-tokens=8192
--max-num-seqs=32
--gpu-memory-utilization=0.85
--block-size=256
--kv-cache-dtype=fp8
--enable-prefix-caching
--kv-offloading-size=75
--kv-offloading-backend=native
--enable-expert-parallel
--speculative-config={"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}
--tokenizer-mode=deepseek_v4
--reasoning-parser=deepseek_v4
--tool-call-parser=deepseek_v4
--enable-auto-tool-choice
--chat-template-content-format=string
--enable-prompt-tokens-details
--trust-remote-code
--disable-custom-all-reduce
```

**TP/PP 必须自己填。** GPUStack 会按显存自己算一组
(`worker/backends/base.py:1270 cal_distributed_parallelism_arguments`),但注入时走
`extend_args_no_exist`(`utils/command.py:137`)—— **只在用户没填时才补**,所以我们填了
就以我们的为准。不填的话它大概率给出 TP4×PP2,而那比 TP2×PP4 慢 51.6%(§3.6)。

**`--enable-prompt-tokens-details` 必须显式加。** 它默认关
(`cli_args.py:132`),不开的话 `usage.prompt_tokens_details.cached_tokens` 恒为空,
new-api 会把全部 prompt 按 ¥3.00 计价 —— **客户账单高 3.81 倍**(§1.1)。

**② 环境变量 → 模型的 `env`(UI 里"环境变量")**

```
VLLM_DETERMINISTIC_MOE_ALIGN=0
VLLM_KV_OFFLOAD_REGION_BACKEND=memfd
VLLM_REASONING_OUTPUT_AS_REASONING_CONTENT=1
NCCL_IB_GID_INDEX=3
NCCL_ALGO=Ring
NCCL_PROTO=Simple
```

注入点在 `worker/backends/base.py:540`(`env.update(self._model.env)`),会覆盖
worker 自身继承的同名变量。

**`NCCL_IB_HCA` 可以不填。** 55 台 × 4 轨 RDMA 都是线速(§7.4.1),不需要像原来那样
裁成 `mlx5_0,mlx5_2`。唯一的例外是 §7.4.2 那种单轨故障未修复的机器,那时按坏轨裁剪。

**不要填的**:`--enforce-eager`(会关掉 CUDA graph)、`VLLM_PP_LAYER_PARTITION`
(真实形状下变慢,§0)、`--default-chat-template-kwargs`(死配置,`tokenizer_config.json`
里零匹配)。

**③ JIT 缓存不用管**

§4.1 说的「Triton 缓存不能放共享 NFS」**不适用于 GPUStack 托管的实例**:它把
`VLLM_CACHE_ROOT` 指向 `<data_dir>/cache/vllm`(`worker/backends/vllm.py:416`),而
worker 的 data_dir 是 `/var/lib/gpustack` → docker volume → **本地盘**。那个 ESTALE
问题只出在我自己那个把 `/root/.cache` 挂到 NFS 的多节点脚本上。

> 只有一处值得显式设:`VLLM_CACHE_ROOT` 管的是 vLLM 自己的缓存,Triton 的 JIT 缓存
> 是否也落在同一路径取决于版本。若在日志里看到 `Stale file handle`,就在 `env` 里加
> `TRITON_CACHE_DIR=/var/lib/gpustack/cache/triton` 显式钉到本地盘。

**④ 一个实例 = 2 个节点 × 4 卡**

TP2 × PP4 = 8 卡。GPUStack 的调度器按 `computed_resource_claim` 分配,确认它把
8 张卡分在**两台**机器上(每台 4 张),而不是别的切法。

---

## 7. 网关设计(机队层面,比任何单实例调参都重要)

### 7.1 分流键不能用「总上下文长度」

我们的耗时由**未命中的新增 token** 决定,不由总长度决定:

- 490k 上下文、489k 已缓存 → **2.42s**(供应商 5.52s,我们快 2.3×)
- 250k 上下文、全是新的 → **~53s**(供应商 ~6s,我们慢 9×)

后者短一半却慢 22 倍。按总长度分流会把「长但全热」(我们最赚)推给供应商,把「短但全冷」(我们最亏)留给自己。

### 7.2 应该按会话分流

1. **会话首轮**按 prompt 大小决定去向(超大文档首轮 → 供应商)
2. 一旦落到我们这边,**该会话后续所有轮次锁定同一实例**,不管上下文涨多大
3. 配合预热(§7.3),首轮也能抢回来

### 7.3 让任何请求都变成热请求

| 层 | 手段 | 状态 |
|---|---|---|
| 1 | 会话亲和路由 | 网关待做,**前提** |
| 2 | `--kv-offloading-size` 卸载到宿主内存 | ✅ 已验证 3.8× |
| 3 | **预热**:收到文档立即发 `max_tokens=1` 预热请求,用户思考时间是免费算力 | 网关待做,**唯一能救首轮的手段** |
| 4 | 跨实例 KV 共享 | 见 §7.4 |

### 7.4 KV 分层的判据:传输必须比重算快 `[实测]`

**每 token 的 KV ≈ 75 KB**(实测:8 卡 × 9.2 GiB 可用 KV 显存 ÷ 1,012,659 token;
与报错「524,288 token 需要 36.6 GiB」算出的 75 KB/token 吻合)。
所以一个平均会话(166.5k token)约 **12.5 GB**,满 1M 的会话约 **75 GB**。

各介质实测带宽,以及搬 12.5 GB 与重算的对比:

| 层 | 实测带宽 | 加载 12.5 GB | vs 重算 |
|---|---|---|---|
| **RoCE RDMA(4×100G 齐上)** | **50 GB/s** | **0.25s** | **50× 快** ✅ |
| **RoCE RDMA(单张 100G,实测 12.2)** | **12.5 GB/s** | **1.0s** | **12× 快** ✅ |
| 宿主内存(PCIe) | ~20 GB/s | **0.6s** | **20× 快** ✅ 已采纳 |
| **NFS**(多流聚合,实测 942~952 MB/s) | **0.95 GB/s** | **13.2s** | **打平/略慢** ❌ |
| NFS(单流,实测) | 478 MB/s | 26s | 2× 慢 ❌ |
| **本地磁盘**(实测) | **109 MB/s** | **115s** | **9× 慢** ❌ |
| 重算 prefill(140k) | 11,370 tok/s | 12.3s | 基准 |

**判据:传输必须比重算快,否则分层是负收益。** 我们重算 140k 要 12.3s,
所以约 1 GB/s 是生死线。这一条排除:

- `ExampleConnector` + 共享文件系统(NFS 打平,无收益)
- LMCache 的 `local_disk` 后端(不论指向 NFS 还是本地盘)
- 任何「磁盘 L3」设想

### 网络拓扑:RoCE 那 50 GB/s 完全没用上 `[实测]`

| 路径 | 带宽 | 当前用途 |
|---|---|---|
| 管理网 `enp131s0f0`(NFS 走这条) | ~950 MB/s | 模型权重、NFS |
| **RoCE `enp194s0f0/f1`+`enp226s0f0/f1`** | **4×100G = 50 GB/s** | 仅传 PP 的 hidden states(每步几十 MB) |

`ip route get <nfs-server>` 确认 NFS 走管理网。**所以跨实例 KV 共享走 RDMA 时
有 50 GB/s 可用,比宿主内存的 PCIe 还快** —— 这使 **Mooncake / NIXL 的离线安装
成为最高价值的工程任务**,而不是可选项。

> 注:这些节点的本地盘(109 MB/s)比 NFS(478 MB/s)还慢,不要想当然。

### 7.4.1 RoCE 健康判据:**不要用 ping** `[实测,一次代价很大的误判]`

**ping 不通 ≠ RoCE 坏。** 本集群每台有 4 个 RoCE IP(全在 `29.165.0.0/16`),
云侧 SDN 只让其中一个回 ICMP,其余三个的 ICMP 在 fabric 里就被丢掉 ——
`tcpdump -i any icmp` 在目标机上**一个包都抓不到**,看起来像是网络坏了。
但 RoCEv2 走 UDP/4791,与 ICMP 是两条完全不同的策略路径。

实测对照(gpu46 ↔ gpu43,以及 gpu46 ↔ 8 台「ping 不通」的机器):

| 轨 | `ping -I <iface>` | `ib_write_bw -d mlx5_N -x 3` |
|---|---|---|
| rail0 `mlx5_0` | ✅ 通 | **11,685 MiB/s**(线速) |
| rail1 `mlx5_1` | ❌ 100% loss(全集群双向) | **11,675 MiB/s**(线速) |
| rail2 `mlx5_2` | ❌ 8 台 100% loss | **11,421~11,686 MiB/s**(线速) |
| rail3 `mlx5_3` | ❌ 100% loss(全集群双向) | **11,679 MiB/s**(线速) |

之前基于 ping 得出的「8 台坏 rail2」「只有 2 条轨可用」「metric 决定可达性」
「SDN 未下发端口-IP 绑定」全部作废 —— 那些是 ICMP 策略的假象。
**那 8 台(gpu2/4/6/7/11/15/16/27)四轨 RDMA 全部线速,完全健康。**

正确的检查方式:

```bash
# 1) 廉价预检:GID 3(RoCEv2 IPv4)存在 + 端口 ACTIVE
for d in mlx5_0 mlx5_1 mlx5_2 mlx5_3; do
  printf "%s state=%s gid3=%s\n" $d \
    "$(cat /sys/class/infiniband/$d/ports/1/state)" \
    "$(cat /sys/class/infiniband/$d/ports/1/gids/3)"
done   # gid3 全 0 = 该轨没地址,这才是真故障

# 2) 权威判据:实打 RDMA(带外握手走管理网,数据面走指定轨)
#    服务端
ib_write_bw -d mlx5_2 -x 3 -F -D 5
#    客户端(<mgmt-ip> 是服务端的 10.0.0.x)
ib_write_bw -d mlx5_2 -x 3 -F -D 5 <mgmt-ip>
```

> 唯一真实存在过的故障是**地址完全缺失**(netplan/NetworkManager 竞态,
> 曾在 gpu1/gpu48 出现 0 个地址)。那种情况下 GID 3 全零,RDMA 真的会断,
> 用上面的预检一眼能看出来。修复:`nmcli device reapply <iface>`(零中断)。
> 另外开机后云侧配置有几分钟延迟,**boot 后至少等 10 分钟再判定**。

### 7.4.2 全量 RDMA 巡检:发现 3 台 5 条轨真坏,重启修复 `[实测]`

> **当前状态(2026-09-14):55 台 × 4 轨 RDMA 全部线速,零故障。**
> 全量复测 112 次(28 对 × 4 轨,覆盖全部 220 个机器-轨组合)全部 ≥11 GB/s。
> 下文保留巡检过程与故障签名,供下次复现时对照。

用 `ib_write_bw` 对 **55 台 × 4 轨 = 220 个组合**做了全量实打(28 对配对,
每台至少作一端)。GID 3 预检 220/220 通过;RDMA 实测结果:

| | 台数 |
|---|---|
| 4 轨全部线速(11.4~11.69 GB/s) | **52** |
| 有轨道 RDMA 不通 | **3** |

坏的具体是:

| 机器 | 坏轨 | 网卡 | 好轨 |
|---|---|---|---|
| **gpu10** | rail1、rail2 | `mlx5_1`/`enp194s0f1np1`、`mlx5_2`/`enp226s0f0np0` | rail0、rail3 线速 |
| **gpu35** | rail3 | `mlx5_3`/`enp226s0f1np1` | 其余三轨线速 |
| **gpu37** | rail2、rail3 | `mlx5_2`、`mlx5_3` | rail0、rail1 线速 |

故障特征(和「ping 不通」那类假象完全不同):

- QP 建得起来、GID 正常交换,**但数据零字节**,`BW average = 0.00` 后挂住
- **双向都不通**(嫌疑机作发端、作收端都是 0.00)
- **换 3 个不同对端复现**(gpu20 / gpu50 / gpu26 / gpu44 全部 0.00),
  而同一台机器的好轨打同一个对端是线速 → 排除对端与配对因素
- 计数器只有 `hw_counters/local_ack_timeout_err`(发出去收不到 ACK)
  和 `req_cqe_error`;`port_rcv_errors`、`port_xmit_discards`、`symbol_error`
  **全为 0**,链路 100 Gb/sec ACTIVE、MTU 8888
- 55 台的 `ecn/roce_np`、`traffic_class`、MTU 配置完全一致 → 不是主机配置差异

**修复:重启,3/3 全部恢复。** 三台重启后 4 轨双向全部线速
(gpu10 rail1/rail2、gpu35 rail3、gpu37 rail2/rail3 均 11.6~11.68 GB/s)。
所以这不是交换机硬件损坏,而是某种会被重启清掉的状态 —— 具体在主机驱动侧
还是云侧端口配置,从主机上无法区分,不做推测。

操作要点:

- **重启后至少等 10 分钟再判定**(开机后云侧配置有数分钟延迟,
  boot+5min 时测会得到假故障)
- gpustack 纳管的业务实例是多节点部署,单机重启不影响服务,
  重启后 gpustack 会自动重建实例(实例 ID 会变)
- 若暂时不能重启,可按坏轨裁剪该机的 `NCCL_IB_HCA`
  (如 `NCCL_IB_HCA=mlx5_0,mlx5_3`),否则 NCCL 会在坏轨上挂死

### 7.5 RDMA 通道实测:可用,但 GPU-Direct 被硬件挡住 `[实测]`

用 mooncake 自带的 `transfer_engine_bench` 在 gpu45 ↔ gpu47 之间实测
(`--protocol=rdma`,metadata 走自带的 HTTP server):

| 模式 | 结果 |
|---|---|
| 主机内存 ↔ 主机内存,单张 100G(`mlx5_2`) | **12.23 GB/s**(线速 12.5,已到顶) |
| 主机内存,`--auto_discovery`(多卡) | **19.61 GB/s** |
| **GPU 显存直传(GDR,`--use_vram=true`)** | **失败:`transport retry counter exceeded`** |

**RoCE fabric 本身健康,坏的专是 GDR。** 原因在硬件拓扑:

```
nvidia-smi topo -m:  GPU0..3 ↔ NIC0..3 全是 SYS
                     (要穿 PCIe + 跨 NUMA 的 CPU 互联)
ACS:                 19 个 PCI 桥中 11 个已启用 —— ACS 会阻断 peer-to-peer DMA
```

`nvidia-peermem.ko`(580.65.06)本来没加载,`modprobe nvidia_peermem` 后
`/sys/kernel/mm/memory_peers/nv_mem` 出现、内存注册那一步过了,但数据仍传不过去
—— 剩下的是 ACS + 跨 NUMA,只能在 BIOS/启动参数层面解(`pcie_acs_override`),
属运维决策。

> **已推翻**:此处原写「`mlx5_1`/`mlx5_3` 全部报 transport retry,只有 2 张卡
> 真正通」。见下面 §7.5.1 —— 那是**跨轨配对**失败被误读成轨道故障。
> 4 张卡都能跑满线速。

**所以跨实例 KV 共享要经主机内存中转**,搬一个平均会话(12.5 GB):

```
GPU → 主机内存 (PCIe ~20 GB/s)   0.6s
    → RDMA (单轨 12.21 GB/s)     1.02s
    → 主机内存 → GPU              0.6s
                       合计 ≈ 2.2s   vs 重算 12.3s  →  约 5.6× 快
```

比 GDR 理想值(~0.8s)差,但仍远优于重算,方案成立。

### 7.5.1 这是 rail-optimized 拓扑:只能同轨对同轨 `[实测]`

用 mooncake 的 `transfer_engine_bench` 在 gpu31 ↔ gpu32 之间逐轨实测
(带外 metadata 走管理网的 `mooncake_http_metadata_server`,
`--use_vram=false`):

| 配对方式 | 结果 |
|---|---|
| `--device_name=mlx5_0` 两端同轨 | **12.21 GB/s** |
| `--device_name=mlx5_1` 两端同轨 | **12.21 GB/s** |
| `--device_name=mlx5_2` 两端同轨 | **12.21 GB/s** |
| `--device_name=mlx5_3` 两端同轨 | **12.21 GB/s** |
| `--auto_discovery`(mooncake 自行配对) | **失败**,读写皆然 |

`auto_discovery` 的报错点明了原因 —— 它尝试的是跨轨路径:

```
local_nic: mlx5_2, peer_nic: ...@mlx5_1 : transport retry counter exceeded
local_nic: mlx5_3, peer_nic: ...@mlx5_2 : transport retry counter exceeded
local_nic: mlx5_3, peer_nic: ...@mlx5_0 : transport retry counter exceeded
Rail paused: peer=...@mlx5_1 error_count=5 pause_ms=30000
```

**第 N 条轨只与对端第 N 条轨连通,轨间无路径** —— 标准的 rail-optimized
组网(每张卡接不同的 leaf 平面,平面之间不互联)。

工程结论:

- **不要开 `--auto_discovery`**,它假设 NIC 之间任意可达
- MooncakeStore 必须显式指定单轨(或给出只产生同轨配对的
  `--nic_priority_matrix`),单轨 12.21 GB/s 已满足 §7.4 的判据
  (搬 12.5 GB 约 1.0s vs 重算 12.3s)
- 之前记录的 `--auto_discovery` 19.61 GB/s 是只有部分轨恰好同轨配对时的读数,
  不是可复现的聚合带宽
- 同理,`NCCL_IB_HCA` 列出多卡本身没问题(NCCL 按 rank 同序配对),
  但任何假设任意 NIC 互通的工具都会在这个 fabric 上失败

### 7.6 Mooncake 的安装:三个坑 `[实测]`

已烘入 `docker/Dockerfile.aarch64_app`。踩过的坑:

1. **不能带依赖装。** `nixl` 的 `Requires-Dist` 含 `torch`,pip 重新解析后开始拉
   `nvidia-nccl-cu13`(206 MB)等,有覆盖 base 里 `2.13.0+cu130`(带 sm_80)的
   风险。**已去掉 nixl**(它是 P2P/PD 分离用的,不是共享缓存);mooncake 改
   `--no-deps`(其依赖 aiohttp/requests/msgpack 在 base 里已满足)。
2. **轮子是 CUDA 12 的,镜像是 CUDA 13。** `ldd mooncake/engine.so` →
   `libcudart.so.12 => not found`,PyPI 无 cu13 变体。解法:并存装
   `nvidia-cuda-runtime-cu12`,用 `ld.so.conf.d` 注册(不用 `LD_LIBRARY_PATH`,
   那会污染全镜像的库搜索顺序)。soname 不同所以两个运行时互不干扰,已实测
   torch 建张量 → import mooncake → torch 再建张量全部正常、`arch_list` 仍含 sm_80。
3. **`nvidia.cuda_runtime` 是命名空间包**,`__file__` 为 `None`,取路径必须用
   `list(m.__path__)[0]`。

**不需要外部 etcd/redis** —— 轮子自带 `mooncake_master`、
`mooncake_http_metadata_server`、`transfer_engine_bench`、
`transfer_engine_topology_dump`。部署就是在一台起 master + metadata server,
其余实例指过去。

容器还需 `--cap-add=SYS_NICE`,否则 NUMA 绑定报 `mbind: Operation not permitted`。

### 7.7 MooncakeStoreConnector:接通了,但写不进去 `[实测]`

> **已结案,见 §7.7.1** —— 根因是这套硬件上 GPU-Direct RDMA 不可用,
> 不是连接器 bug。下文保留当时的排查过程与被排除的候选。

已完整搭起两实例共享一个 Store 的环境并实测,结论是**目前拿不到收益**。

环境(均已验证):
- `mooncake_master --enable_http_metadata_server=true`(master 与元数据服务同进程,
  **不需要外部 etcd/redis**)
- 两个 TP2×PP4 实例 + `--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'`
- `MOONCAKE_CONFIG_PATH` 指向 `{metadata_server, master_server_address, protocol=rdma,
  device_name=mlx5_2, mode=embedded, global_segment_size=8GiB}`
- 容器需 `--cap-add=SYS_NICE`(否则 `mbind: Operation not permitted`)

启动侧一切正常:

```
master:  Clients: 16                        (8 rank × 2 实例全部注册)
         master_total_capacity_bytes 68719476736   (64 GiB 池)
worker:  Mooncake mode=embedded (global_segment_size=8589934592, ...)
         Mounting segment: 8589934592 bytes
         Started 1 Mooncake KV-load receive thread(s)
```

**读路径通,写路径不通:**

```
KV Transfer metrics: lookup_exists_count=7, lookup_exists_total_keys=24564   ← 在查
master_put_start_requests_total 0                                            ← 从不写
```

跨实例测试(固定 90k 文档,A 冷算后 B 打同一份):

| 步骤 | TTFT |
|---|---|
| A 首次(冷) | 16.73s |
| Store `key_count` | **0** |
| B 跨实例 | **17.28s**(零收益) |

对照:同实例同请求连打三次 **16.69s → 1.12s → 0.88s**,
说明**本地前缀缓存不受影响**,问题只在 Store 的写路径。

已逐一排除的拦截点(读代码 + 算数,非猜测):

| 候选 | 排除依据 |
|---|---|
| `kv_role` | `scheduler.py:208` `is_consumer = kv_role == "kv_consumer"`,`kv_both` → False → `skip_save=False` |
| chunk 门槛 | `data.py:779-783`:首 chunk 8192 token,`num_tokens_to_save=8192 ≥ chunk_boundary=256` |
| `block_hashes` 为 None | `data.py:775` 会降级为 `[]`,不提前返回 |
| `transfer_group_ids` 为空 | `kv_cache_interface.py:1278` `enable_kv_transfer` 默认 True,核心代码无处置 False |

### 7.7.1 结案:写路径**一直在触发**,是每个 key 都失败 `[实测]`

上面「写路径不触发」的判断是错的 —— 我只看了 master 侧的
`master_put_start_requests_total`,而失败发生在**客户端、RPC 发出之前**,
所以 master 什么也没看到。

vLLM 自己的 `/metrics` 里写得很清楚:

```
vllm:mooncake_store_operation_total{operation="save_exists",status="ok"}              8
vllm:mooncake_store_operation_total{operation="save_put",status="partial_failure"}    8
vllm:mooncake_store_operation_keys_total{operation="save_put"}                      116
vllm:mooncake_store_operation_failed_keys_total{operation="save_put"}                116   ← 全失败
vllm:mooncake_store_operation_bytes_total{operation="save_put"}                29,064,960
```

失败码在 Ray worker 的日志里(**不在 driver 的 serve.log 里**,这是之前漏掉的原因,
要去 `/tmp/ray/session_*/logs/` 找):

```
WARNING [worker.py:1160] batch_put failed: 15/15 keys failed (codes={-800},
        batch_bytes=3559680), first_key=...@pp_rank:0@group:0@875ba951...
```

8 个 rank 全是 `codes={-800}`。

**根因:`batch_put_from_multi_buffers` 传入的是 KV cache 的 GPU 显存地址,
而这套硬件上 GPU-Direct RDMA 不可用。** 逐步实测(gpu41 ↔ gpu42,同轨 mlx5_0):

| 条件 | 结果 |
|---|---|
| `--use_vram=false`(主机内存) | **12.20 GB/s** ✅ |
| `--use_vram=true`,`nvidia_peermem` 未加载 | `Failed to register memory: Bad address [14]` —— 注册就失败 |
| `--use_vram=true`,`nvidia_peermem` 已加载 | 注册通过、QP 连上,**但传输 `transport retry counter exceeded`** |

也就是说 peermem 只解决注册,数据仍然过不去。硬件侧证据:

```
/proc/cmdline           无 pcie_acs_override / iommu 相关参数
lspci -vvv              19 个 PCI 桥,11 个 ACSCtl SrcValid+   ← ACS 阻断 P2P DMA
nvidia-smi topo -m      GPU↔NIC 全为 SYS;GPU0/1 在 NUMA 0,GPU2/3 在 NUMA 2
```

**所以这不是连接器的 bug,是硬件能力缺失。** 三条出路:

1. **BIOS / 内核加 `pcie_acs_override`** 后重测 —— 唯一可能真正打开 GDR 的办法,
   属运维决策(要改启动参数并重启)。跨 NUMA 那一段仍在,收益未知。
2. **改连接器走主机内存中转**(GPU→主机 PCIe,再从主机 RDMA)。
   `--use_vram=false` 已实测 12.20 GB/s,原理成立,但连接器目前不做这一步,
   属引擎开发任务。
3. **放弃跨实例共享,靠网关会话亲和**(§7.2)把同一会话钉在同一实例上。
   本地 KV 卸载到主机内存已有 3.7× 收益(§3),而会话亲和做到之后,
   跨实例共享的价值本来就大幅下降。**这是当前的实际方案。**

> 附带修正:§7.7 表格里「`device_name=mlx5_2`」不是问题所在,
> 换成 `mlx5_0` 并确保同轨配对后,`save_exists` 正常、`save_put` 依旧全失败。

### 7.8 可用的跨实例 KV connector

| connector | 来源 | 镜像内 |
|---|---|---|
| `lmcache_*` | LMCache | ⚠️ 包已装(0.5.5.dev135)但**两条接口都起不来**:`--kv-offloading-backend lmcache` 报 `device.py:56 Failed to infer device type`;`--kv-transfer-config LMCacheConnectorV1` 把 KV 显存账改写成「524288 需 36.6 GiB」超出可用 |
| `ExampleConnector` | vLLM 内置 | ⚠️ 机制对(按 input_ids 哈希写共享目录),但只能走文件系统 —— 见 §7.4,原理上不成立 |
| **`mooncake`** | Moonshot | ✅ **已烘入镜像**(见 §7.6)。RDMA 通道已实测可用(§7.5) |
| `nixl` | NVIDIA(PD 分离) | ❌ 故意不装:`Requires-Dist: torch`,有覆盖 base 的风险;且用途是 PD 分离而非共享缓存 |
| `hf3fs` | DeepSeek 3FS | ❌ 未装 |

---

## 8. DeepSeek-V4.1-Flash 评估(已下载并校验,510.3 GB)

| | 0731 | V4.1-Flash |
|---|---|---|
| 权重 | 167 GB | **510.3 GB** |
| 层 / hidden / 路由专家 | 43 / 4096 / 256 | 40 / 5120 / **384** |
| `compress_ratios` | **4 和 128**(压得狠) | **2 和 1**(几乎不压) |
| Engram | 无 | `engram_layer_ids=[1,14]` |
| 模态 | 纯文本 | 带 vision |

**「V4.1 把 KV 压缩了很多」这个说法是反的 —— 0731 压得更狠。**

但权重构成给了一条出路:

```
routed_experts   296.0 GB  58.0%
engram           203.1 GB  39.8%   ← --engram-config.cpu_offload 可放 pinned CPU 内存
attention/其他      8.7 GB   1.7%
```

开 Engram CPU 卸载后 GPU 只需 **307 GB** —— 这一步**实测可行**:

```
[实测] engram.py:693  Engram table offloaded to pinned host memory:
                      48002473 rows x 256, 11.80 GiB   (每 rank,×16 = 189 GiB)
[实测] model_runner.py:425  Model loading took 19.2 GiB   (每卡,307GB/16 = 19.2 ✓)
```

### 8.1 但 V4.1 在 A100-40G 上**跑不起来** `[实测]`

三个相互独立的硬约束:

| # | 约束 | 证据 | 后果 |
|---|---|---|---|
| 1 | **PP 不支持**(其一):KV 共享组不可切分 | `kv_source_layer_ids=[2,8,14,20]` → 层 20-39 是一个 20 层的组;`attention.py:486` `NotImplementedError: PP splits inside a v4.1 kv-sharing group are not supported` | PP4 最均衡只能 `8,6,6,20` |
| 2 | **PP 不支持**(其二):vision MoE 每层都要 `input_ids`,而 PP 跨级只传 hidden_states | `deepseek_v4/nvidia/model.py:1027` `ValueError: DeepSeek V4 vision MoE routing requires input_ids`,报错来自 `Worker_PP1_*` | **只能 PP=1** |
| 3 | **`o_groups=8` 限死 TP≤8** | TP16 时 `8//16=0` → `RuntimeError: shape '[8192, 0, -1]' is invalid for input of size 16777216` | **最多 8 卡** |

8 卡 = 2 节点 = **272 GB 可用**,而 Engram 全卸载后仍需 **307 GB** —— **差 35 GB**。
唯一补法是再开 `--cpu-offload-gb` 把 35 GB 权重也放 CPU,但那要每次前向过 PCIe
流式读 35 GB(~1.75s/步),不可用。

**结论:不是调参问题,是结构性的。V4.1-Flash 在本硬件上无法运行。**

### 8.2 下载完整性

首次下载有一个分片 `model-00014-of-00048.safetensors` **多出 12,582,912 字节
(正好 12 MiB)**,是断点续传重复追加所致,表现为
`SafetensorError: incomplete metadata, file not fully covered`。
已删除重下并校验全部 48 个分片(510.3 GB,逐个比对头部声明大小)。

> 注意 `du -sh` 报 476G 是 GiB,与索引的 510.3 GB 并不矛盾 —— 不要用它判断完整性。

---

## 9. 4 节点为什么慢,以及为什么不必修

PP4 实测 −65%。三个原因:

1. **DSpark draft 被硬编码成 PP=1**(`vllm/config/speculative.py:1748`、`vllm/v1/worker/gpu/spec_decode/dspark/utils.py:58`)。每个接受周期的 draft 前向只有一个 stage 在工作,PP4 时 **75% 的卡空转**。修它需要改代码。
2. **层切分未为 PP4 调过**。默认 `[11,11,11,10]`,但最后一级还扛着 `lm_head`(5.3 亿参数)和 `dspark_target_layer_ids=[40,41,42]`。推导最优约 `[12,12,11,8]`。
3. 流水线气泡本身是 PP 的固有代价。

**但不必修** —— 想上 4 节点的动机是 KV 空间,而 KV 卸载用宿主内存拿到同样的容量,代价是 0 张卡:

| 扩容方式 | KV 增量 | 代价 |
|---|---|---|
| 2 → 4 节点 | 3.6× | 每卡吞吐 −65%,多占 8 张卡 |
| `--kv-offloading-size` | ~6× | **0 张卡**,淘汰重入 2.79s |

---

## 10. 待办

单实例调参已到边际收益递减 —— **剩余收益集中在网关侧**:

1. **会话亲和路由**(§7.2)。不做这一项,77.3% 的命中率会被负载均衡摊薄,
   算力需求 ×4.26,毛利转负。**优先级最高。**
2. **准入控制**(§3.4):按活跃上下文总量限流到 1.4M token/实例。
   靠 `--max-num-seqs` 拦不住,必须在网关做。
3. **预热**(§7.3):收到文档立即发 `max_tokens=1`,把冷 prefill 移出关键路径。
   唯一能救「会话首轮 + 大文档」那 8.4% 请求的手段。
4. **缓存读的计价策略**:跟不跟供应商的 3.1% 折扣 —— 86% vs 57% 毛利的分界,
   需业务决策。
5. 改 `delta.reasoning` → `delta.reasoning_content` 以兼容客户端(唯一需要改代码的项)。
6. 未完成:LMCache 后端 vs native 的对比(尝试时被 Ray 陈旧 actor 句柄打断,
   重启 Ray 后未重跑)。
