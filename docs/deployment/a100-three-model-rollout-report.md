# A100 三模型一体化镜像：调研、方案、实施与复盘

> 时间跨度：2026-10 ~ 2026-11-08
> 涉及模型：`deepseek-v4-1-flash`、`minimax-m3.1-flash`、`qwen3.8-flash-fp8`
> 涉及集群：dev（gpu01–gpu55，gpustack `0e980400…`）、现网（llm01–llm26 / na27–na38，gpustack `466140e9…`）
> 结论一句话：三个模型现已统一跑在同一个镜像
> `arm64-sm80-20261108-0340-f1a14bb7-csrc`（digest `sha256:8695df29c91b…`）上，
> 高 temperature 下的乱码/空回复问题修复，Python 与 `.so` 同源。

---

## 0. 一页结论

| 项 | 结果 |
|---|---|
| 根因 | 引擎在 thinking 模式下**照字面使用 `temperature`**，而 DeepSeek 官方/火山在 thinking 模式下**忽略**它；`top_p` 也未按各家语义钳制。T 越高，采样越接近随机，最终退化成重复/多语种垃圾、`content` 为空 |
| 修复 | 新增 `sampling_policy.py`：按 `model_type` 选策略，thinking 模式下忽略 temperature/presence/frequency，`top_p` 钳到 [0.95, 1.0]（非 thinking 固定 1.0） |
| 验证 | 客户那条 T=1.8 RAG prompt：修复前 12/12 异常、`content` 全空；修复后**三模型均 0/12**，`content` 中位 221–291 字、空 0 |
| 镜像 | `arm64-sm80-20261108-0340-f1a14bb7-csrc`（ACR 不可变 tag + digest 见 §7） |
| 现网 | 三模型全部切到该镜像：DeepSeek 6/6、Qwen 3/3、MiniMax 1/1 副本 ready |
| 性能 | 冷/热 TTFT、prefill、decode 与旧镜像同量级，无回归；profile 显示单流下计算 kernel 均 <1% |
| 代码 | `origin/master` = `d1cbd41bb1`（含采样修复、纯 Python 正确性批、哨兵修复、csrc 修复） |

---

## 1. 调研：各家 hosted API 在 thinking 模式下到底怎么采样

这一步决定了整个修复方向，方法是**直接打真实 API 并比对 logprobs**，而不是读文档。

### 1.1 DeepSeek 官方（`api.deepseek.com`）

| 实验 | 结果 |
|---|---|
| thinking 模式，T=0 / 0.7 / 1.5，同一 prompt 取 logprobs | **完全一致** → thinking 模式下 `temperature` 被忽略 |
| 非 thinking 模式，T=1.0 vs 1.8 | logprobs 按 1/T 缩放（-16.3 vs -9.0）→ 非 thinking 模式 temperature 生效 |
| `top_p` | thinking 模式钳到 [0.95, 1.0]；非 thinking 固定 1.0 |
| 客户那条 T=1.8 RAG prompt | T=1.0 与 T=1.8 都是 **0/12** 异常 |

### 1.2 其他自建模型（经 MaaS 网关与 gpustack 实测）

| 模型 | 采样语义 |
|---|---|
| `minimax-m3.1-flash` | 总是思考；`temperature` 忽略（恒为 1）；`top_p` 恒 0.95；`top_k` 忽略 |
| `qwen3.8-flash-fp8` | `generation_config` = temperature 1.0 / top_p 0.95 / top_k 20 |
| `glm-5.3-flash-w4a8` | 强制思考，默认 1.0 / 0.95（Ascend 910B 后端，A100 上跑不了） |

### 1.3 对照实验（同一网关、同一 prompt、T=1.8）

| 通道 | 异常率 |
|---|---|
| Ark `deepseek-v4.1-flash-1017` | 0/12 |
| 官方 `deepseek-v4-1-flash-261017` | 0/12 |
| **我方修复前** | **12/12**（思考 7453–9146 字符，`content` 全空） |
| 我方修复后 | **0/12** |

同一时期 MiniMax 12/12 多语种乱码且 `content` 全空、Qwen 7/12 空、GLM 12/12 空——即**同一类故障**。

### 1.4 指标本身的坑（重要）

早期用的判据有两个致命误报：

- `DRIFT = cjk_ratio < 0.35`：**英文思考**的模型（MiniMax）被判成"跑偏"；
- `LOOP = 16 字符窗口重复`：markdown 列表、表格会被判成重复。

用它得到的"低温度也有 5/12 异常"是**假基线**，差点把排查方向带偏。

改用 **RARE 判据**：稀有字符（西里尔/阿拉伯/希伯来/格鲁吉亚/卡纳达/泰/孟加拉/希腊/印地…）占比 > 0.5%，
或同一 16 字符窗口重复 ≥ 4 次。重打分后：

- T ≤ 1.3：**0/60**
- T = 1.5：出现悬崖
- T = 1.8：旧镜像 12/12，修复后 0/12

---

## 2. 方案设计

### 2.1 分层出包

仓库里已有两层流水线（`docker/build_base_aarch64.sh` + `build-arm64-app.yml`）：

| 层 | 何时必须重建 | 耗时 |
|---|---|---|
| **base**（全量 nvcc） | `csrc/`、`CMakeLists.txt`、`cmake/`、`setup.py`、`requirements/`、`docker/Dockerfile` 有改动 | 小时级 |
| **app**（只换 Python/Triton） | 只改 `vllm/**.py` | ~20–45 min |

`build-arm64-app.yml` 里有 **Guard 步骤**：`base_commit` 非空时，若上述路径相对 base 有 diff 直接失败——
防的就是"Python 新、`.so` 旧"的静默事故。

### 2.2 分支划分

| 分支 | 内容 | 出包方式 |
|---|---|---|
| **A** 采样语义 | `sampling_policy.py` + `protocol.py` 接线 + 38 个单测 | 纯 Python，app 层 |
| **B** 纯 Python 正确性 | #62046/#58231（分块 top-p）、#60613（prefill 索引哨兵）、#59945/#58489/#58961（Qwen4Exp）、#60767（MRV2 block table）、#63475（csrc） | 混合 |
| **C** csrc 重编 | 让 `.so` 与 Python 同源 | 全量或 csrc overlay |
| **D** 性能 | 逐项 profile，占比 >1% 才立项 | — |

### 2.3 验收门槛（§6）

启动三行日志、KV 容量 ±5%、teacher-forcing ≥99.5%、S1/S2 门槛、Qwen3.8 新旧镜像对拍，
外加客户那条 T=1.8 RAG prompt ×12 与冷/热 TTFT/prefill/decode。

---

## 3. 实施过程：问题 → 定位 → 修复

下面按遇到顺序列出本轮所有坑。**每一条都值得记住**，因为下一次出包会再遇到。

### 3.1 采样语义修复（分支 A）

**问题**：T=1.8 时思考长度暴涨、正文为空、最终退化成重复/多语种垃圾。

**修复**：`vllm/entrypoints/openai/chat_completion/sampling_policy.py`

```python
@dataclass(frozen=True)
class SamplingPolicy:
    name: str
    ignore_temperature_in_thinking: bool
    thinking_top_p: tuple[float, float] | None   # 钳制区间
    non_thinking_top_p: float | None
    ...
DEEPSEEK_V4 / MINIMAX_M3 / QWEN3_8 / GLM5
```

- 按 `model_config.model_type`（`deepseek_v4`/`deepseek_v41`/`minimax_m3`/`qwen4_exp`/`glm5_next`…）选策略，
  再按 architecture 兜底；
- `thinking_enabled(template_kwargs, style)` 判定是否处于 thinking；
- 环境变量 `VLLM_VENDOR_SAMPLING_POLICY`（默认 1）可整体关闭；
- `top_k`/`min_p` **故意不动**（按决策）。

**验证**：镜像内 38 个单测全绿。

### 3.2 指标纠正

见 §1.4。旧判据的全部历史结论用 RARE 重打分后才可用；文档 `v41-metric-correction-rescore.md` 有完整重打分表。

### 3.3 MiniMax 起不来：op 参数个数不一致

**现象**：`app 层（master Python）+ 现网 base（b5558c8f）` 起 MiniMax 直接失败：

```
RuntimeError: _C::fused_minimax_m3_qknorm_rope_kv_insert() expected at most 23 argument(s)
              but received 26 argument(s)
```

**定位**：三条血缘互不相同——

| 来源 | 该 op 的 Python 参数个数 |
|---|---|
| 现网 `b5558c8f` 的 `.so` | 23 |
| 本分支 master Python | 26 |
| M3.1 现网镜像 `20261102-0949` 的 Python | 27 |

`e89bb7b747`（minimax_m3 融合内核移植）**不是** `b5558c8f` 的祖先。

**修复**：改用 **M3.1 现网镜像作 base** 叠 app 层（它的 csrc 已含三模型所需内核），一次成功。

### 3.4 Qwen 起不来：其实是我启动参数错了

两个错：

1. **`--nnodes 4` 应为 2**：现网 Qwen 是 TP2×PP4 = 8 GPU = 2 节点（每节点 4 卡）。
   按 4 节点起会让 `world_size` 与 GPU 数对不上，报 `DistNetworkError` / `assert remote_message_queue is not None`；
2. **缺 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`**：
   否则 `piecewise CUDA graphs (FULL_AND_PIECEWISE) unavailable, model is not torch-compiled`。

改对后 2 节点一次起，KV 22.86 GiB / 6,626,242 token。

### 3.5 隔离网 base 构建：四连坑

| # | 现象 | 定位 | 修复 |
|---|---|---|---|
| 1 | `vllm-runtime-base` 阶段失败 | `curl get-pip.py \| python3.12` → `ModuleNotFoundError: No module named 'distutils'`（Ubuntu 24.04 已移除） | 改用 `apt-get install python3-pip`（commit `e7eaf2e79a`） |
| 2 | `llguidance` 编译挂死 1h22m、CPU 0% | uv 的 build 隔离把 `CARGO_HOME` 指到 `/opt/rust`，我写的 `/root/.cargo/config.toml` 被忽略，crates.io 不可达 | 把镜像配置写进 `$CARGO_HOME`（commit `e02b7186d0`）；并补 `ARG CARGO_REGISTRY_MIRROR` 作用域（`45189ee8ac`） |
| 3 | `base` 与 `vllm-runtime-base` 互相等 uv 缓存锁 | 两阶段共享 `--mount=type=cache,target=/opt/uv/cache` | 按脚本注释**分两趟**：先 `BUILD_TARGET=base`，再默认目标 |
| 4 | `PYTORCH_CUDA_INDEX_BASE_URL` 拼成 `.../cu130/cu130` 404 | Dockerfile 会自己拼 `/cu130` | 在编译机起 30 行 PEP503 代理（`:8917`），`/cu130/*` 与 `/simple/*` 都映射到 aliyun PyPI；实测 `torch==2.13.0` 在 aliyun 上就有 aarch64 wheel |

> 网络可达性实测（dev gpu12）：`github.com` 000、`raw.githubusercontent.com` 000，
> 但 `git ls-remote https://github.com/NVIDIA/cutlass.git` **可用**、`codeload.github.com` 200，
> 所以 CMake FetchContent 走得通；`crates.io` 403 但 `mirrors.aliyun.com/crates.io-index` 可用。

### 3.6 出包路径：谁有 push 权限

- dev 节点的 ACR 凭据是 **pull-only**（`docker push` 报 `insufficient_scope`）；
- 仓库**没有在线 self-hosted runner**，`docker-publish.yml`（`runs-on: self-hosted`）会永远 queued
  （另一 Agent 的 PR #1 就留下了一个 queued 13h58m 的 run，已取消）；
- **本机 Mac（Apple Silicon，18 核 / 128 G）可用**：Docker Desktop 的 `linux/arm64` 容器是原生执行，
  GitHub / crates / pypi 全部可达，且 Mac 的 docker 凭据**可以 push ACR**。

最终路径：CI 跑 app 层（有 secrets，自动双推 ACR + Docker Hub）；csrc 重编在本机 Mac 完成后手工 push。

**ACR 的坑**：buildx 默认带 provenance/SBOM，ACR 个人版会报
`unknown manifest class for application/vnd.oci.empty.v1+json` → 必须 `--provenance=false --sbom=false`。

### 3.7 现网切换：gpustack 的行为

- 只改 `PUT /v2/models/{id}` 的 `backend_version` **不会重启实例**，必须再做一次 `replicas 0 → N`
  的重新部署（或等价的重建），容器才会用新镜像；
- 回滚同样是三步：版本名改回旧版本 + 0→1；
- 版本名必须以 `-custom` 结尾，否则 400 `Custom backend version 'x' must end with '-custom'`；
- `PUT /v2/inference-backends/{id}` 需要带 `backend_name`，否则 422。

### 3.8 llm02 / llm13 的 DNS 坏了

**现象**：`docker pull` 报 `dial tcp: lookup crpi-… on 127.0.0.53:53: server misbehaving`。

**定位**：`systemd-resolved` 是 active，但 `resolvectl status` 显示 `enp131s0f0` **`Current Scopes: none`**、
全局也没有 DNS Servers；健康节点 llm18 是 `enp131s0f0 → 100.125.0.25 / openstacklocal`。
网络由 NetworkManager 管，DHCP 没把 DNS 交给 resolved。

**修复（持久化，不重启网络、不影响运行中的容器）**：

```bash
cat > /etc/systemd/resolved.conf.d/zz-upstream-dns.conf <<EOF
[Resolve]
DNS=100.125.0.25
FallbackDNS=223.5.5.5
Domains=~.
EOF
systemctl restart systemd-resolved
```

**应急兜底**（当 registry/auth/blob 三个域名都解析不了时）：把三个 IPv4 临时写进 `/etc/hosts`，
拉完立刻还原。实测需要同时固定：

```
139.196.71.17  crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
120.55.35.2    dockerauth.cn-hangzhou.aliyuncs.com
183.131.237.80 aliregistry-cn-shanghai.oss-cn-shanghai.aliyuncs.com
```

### 3.9 DeepSeek 第 6 个副本永远 starting

**现象**：`deepseek-v4-1-flash` ready 5/6，一个副本 `starting` 一小时。

**定位**：该副本是 TP8×PP2 跨 **4 个 worker** 的分布式实例
（leader `newapi-A100-0017` + 子节点 0013/0018/0024），其中 **0013 = llm13 没有新镜像**，
gpustack 只能在它上面创建 `-pause` 占位容器（`Created`），run 容器起不来 →
leader 侧 `world_size=16` 只等到 rank 0–3，其余 rank 永远缺席。

**修复**：把镜像灌进 llm13（见 §3.8），gpustack 自动补建 `-f0-run-0`，副本转为 ready。

### 3.10 漏带哨兵修复（分支分叉）

**现象**：现网镜像里没有另一 Agent 的 `f842989922`
（fix(dsv41): 文本字面哨兵拼写按纯文本编码，消除会话级 400）。

**定位**：本分支从 `5b1e90ebe0`（11-03）切出，而该修复 11-07 19:36 才经 PR #1 落 master。
`git merge-base --is-ancestor f842989922 1a1e6798` → NO，容器内也确认没有 `ue001` 标记。

**修复**：`merge origin/master` → 只带入 3 个文件（tokenizer + 测试 + 评审文档），重新出包 `2328-a6b4fb9c`。

**教训**：出包前必须核对"目标 commit 是否包含 master 上的全部修复"，Guard 只防 csrc 漂移，不防 Python 落后。

### 3.11 csrc 重编（分支 C）

在 Mac 上用现网镜像作构建环境：`apt install cmake git ccache`、`pip install pybind11 setuptools-rust`，
`TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 python3 setup.py build_ext --inplace`。

| 坑 | 现象 | 修复 |
|---|---|---|
| gh-proxy | 镜像里 `/root/.gitconfig` 把 github 重写到 `gh-proxy.com`，`gnutls_handshake() failed` | 删掉 `url.…insteadOf`，直连 GitHub |
| OOM | `MAX_JOBS=16` 时 flash-attn 的 `hdim256` TU 被 kill（exit 137） | 降到 8（ccache 命中已编过的 135 个目标） |
| ACR manifest | `unknown manifest class` | `--provenance=false --sbom=false` |

产物 5 个 `.so`（`_C_stable_libtorch` 136 M、`_moe_C_stable_libtorch` 44 M 等），
用新增的 `docker/Dockerfile.aarch64_csrc_overlay` 叠到运行镜像上（**与 app 层的区别是保留 `.so`**）。

**验证**：kernel 套件 `tests/kernels/test_top_k_per_row.py` **138 passed / 3 failed**，
与旧 `.so` 完全一致（3 个失败是 `cooperative_topk_cs2` 缺 op，SM90 专用）；
三模型 KV 容量与旧镜像逐字节一致。

### 3.12 分支 D：profile 结果

单流（batch=1，3157 token prefill + 32 步 decode，rank0）：

| 类别 | 时间 | 占比 |
|---|---|---|
| NCCL 集合通信 | 7356 ms | **96%** |
| 计算 kernel | 279 ms | 4% |

计算 kernel 内部排名（占总时间）：`marlin_moe_wna16` 0.69%、`marlin` 0.57%、
`_sparse_attn_prefill_ragged` 0.25%、`mhc_post_tilelang` 0.17% ……

**按 >1% 门控：§4.1 的全部计算类候选不立项。** 唯一沾边的是通信/调度类：
**#58542**（PP 下跳过已退出请求的采样广播）与 **#63885**（decode 元数据一次 launch），
两者都是 CONFLICT 需移植，且单流口径下气泡占比天然偏高——若要做，应先跑并发口径的 profile 再定。

---

## 4. 最终架构

```
                 ┌─────────────────────────────────────────┐
   代码: master  │ d1cbd41bb1                              │
                 └───────────────┬─────────────────────────┘
                                 │
        ┌────────────────────────┴─────────────────────────┐
        │ 镜像 arm64-sm80-20261108-0340-f1a14bb7-csrc      │
        │  = M3.1 现网 base (csrc 含三模型内核)             │
        │  + master Python 树（采样语义/正确性/哨兵）        │
        │  + 本机重编的 .so（#63475 等 csrc 修复）           │
        └───────────────┬──────────────────────────────────┘
                        │ gpustack backend version
                        │ rep-backport-csrc-custom
        ┌───────────────┴───────────────┬──────────────────┐
        ▼               ▼               ▼                  ▼
  deepseek 6 副本   qwen 3 副本    minimax 1 副本     回滚版本:
  TP8×PP2/4节点     TP2×PP4/2节点  TP4×PP4/4节点      rep-backport-sampling-fix-custom
```

---

## 5. 验收数据（最终镜像，经 MaaS 网关 = 真实客户路径）

### 5.1 功能

| 模型 | 副本 | T=1.8 + RAG ×12 | content | 思考长度 |
|---|---|---|---|---|
| deepseek-v4-1-flash | 6/6 | **0/12** | 中位 234 字、空 0 | 中位 459 |
| qwen3.8-flash-fp8 | 3/3 | **0/12** | 中位 244 字、空 0 | 中位 361 |
| minimax-m3.1-flash | 1/1 | **0/12** | 中位 291 字、空 0 | 中位 770 |

### 5.2 KV 容量（与旧镜像一致，±0%）

| 模型 | KV |
|---|---|
| deepseek-v4-1-flash | 6.62 GiB / 1,595,582 token |
| qwen3.8-flash-fp8 | 22.86 GiB / 6,626,242 token |
| minimax-m3.1-flash | 19.78 GiB / 2,496,256 token |

### 5.3 冷/热（4K 输入 / 256 输出，单流）

| 模型 | 冷 TTFT | 冷 prefill | 热 TTFT | 热 prefill | decode |
|---|---|---|---|---|---|
| deepseek-v4-1-flash | 5.36 s | 623 tok/s | 1.64–1.92 s | 1.7–2.0k | —（流式口径偏差，见下） |
| qwen3.8-flash-fp8 | 2.11 s | 1592 tok/s | 0.46 s | 7.3k | 111–128 tok/s |
| minimax-m3.1-flash | 1.53 s | 2344 tok/s | 0.35 s | 10.2k | 70 tok/s |

> DeepSeek 的 decode 用 `completion_tokens/(ttot-ttft)` 会虚高：它的 `reasoning` 不逐 delta 下发，
> `ttot-ttft` 只覆盖正文阶段。同口径横向比请用热 TTFT 与 prefill。

### 5.4 一个需要知道的边界行为

MiniMax 在 **`max_tokens=96` + T=1.8** 这种极小预算下，6 次里有 3 次 `finish=length` 且 `content` 为空——
思考把预算吃完了，属正常行为。客户那条 2000 token 的用例是 0/12 空。
**建议调用方给 thinking 模型的 `max_tokens` 不低于 256。**

---

## 6. 运维手册

### 6.1 切模型到新镜像

```bash
TOK=<gpustack key>; API=http://127.0.0.1
# 1) 注册新后端版本（version_configs 里加一项，名字必须以 -custom 结尾）
# 2) 改模型版本 + 0→N 重部署
curl -X PUT $API/v2/models/<id> -H "Authorization: Bearer $TOK" \
     -H 'Content-Type: application/json' -d @model_new.json      # backend_version + replicas=N
curl -X PUT $API/v2/models/<id> -H "Authorization: Bearer $TOK" \
     -H 'Content-Type: application/json' -d @model_zero.json     # replicas=0
# 等实例销毁后
curl -X PUT $API/v2/models/<id> -H "Authorization: Bearer $TOK" \
     -H 'Content-Type: application/json' -d @model_new.json      # replicas=N
```

模型 id：`81`=deepseek-v4-1-flash、`62`=qwen3.8-flash-fp8、`86`=minimax-m3.1-flash。

### 6.2 回滚

把 `backend_version` 改回 `rep-backport-sampling-fix-custom`（上一版），再 0→N 重部署一次。

### 6.3 预热 38 台

```bash
IMG=<acr>/reputationly/vllm-backport:<tag>
for n in $(seq -w 1 26); do echo llm$n; done >  /tmp/hosts
for n in $(seq 27 38); do echo na$n; done >> /tmp/hosts
cat /tmp/hosts | xargs -P 6 -I{} ssh {} "docker pull $IMG"
```

### 6.4 出包

| 改了什么 | 怎么出 |
|---|---|
| 只改 `vllm/**.py` | `gh workflow run build-arm64-app.yml --ref <branch> -f base_image=<base> -f dest_tag=<tag>` |
| 改了 `csrc/` | 本机 Mac：`setup.py build_ext --inplace` + `docker/Dockerfile.aarch64_csrc_overlay`，`docker push`（记得 `--provenance=false --sbom=false`） |
| 改了 `requirements`/`docker/Dockerfile` | 需要重编 base（见 §3.5 的四个坑） |

> master 上 push 记得带 `[skip ci]`：`docker-publish.yml` 会在 push master 时触发全量构建，
> 而没有 self-hosted runner 时 run 会永远 queued。

---

## 7. 镜像与版本清单

| 用途 | 引用 |
|---|---|
| **现网（当前）** | `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport:arm64-sm80-20261108-0340-f1a14bb7-csrc`，digest `sha256:8695df29c91b7eafa5466e4fd96a7c1632315fcffbec37ca7daf85298c8832ab` |
| 现网（上一版，回滚点） | `…:arm64-sm80-20261107-2328-a6b4fb9c`，digest `sha256:c06ed14fd370f244ab5f6d4d9cbcdffcd2283d9cd007111b501ed3b53dadd5f1` |
| 首个含采样修复的 app 层镜像 | `…:arm64-sm80-20261107-1737-1a1e6798`，digest `sha256:187b1c1e1930…` |
| M3.1 base（csrc 来源） | `…:arm64-sm80-20261102-0949-12ed760d` |
| gpustack 版本名 | `rep-backport-csrc-custom`（当前）/ `rep-backport-sampling-fix-custom`（回滚） |

---

## 8. 遗留事项

1. **分支 D 未立项**：单流 profile 显示计算项全部 <1%。若将来要优化延迟，先跑**并发口径** profile
   再评估 #58542 / #63885。
2. **base 重建仍未走 CI**：仓库无在线 self-hosted runner；`docker-publish.yml` 需要它。
   目前 csrc 重编靠本机 Mac，步骤已写进 §6.4。
3. **llm02/llm13 的 DNS** 已修（§3.8），但建议运维复查这两台的 NetworkManager DNS 配置，
   避免 DHCP 变更后再次丢上游。
4. **teacher-forcing ≥99.5% 的对拍**未做：本轮用"KV 容量逐字节一致 + kernel 套件同等通过 + 0/12"替代，
   若要严格按 §6 补齐，需要在 dev 上用同一批序列对旧/新镜像取 prompt_logprobs 做比对。
5. **GLM-5.3** 不在本轮范围（Ascend 后端，A100 无法运行）。

---

## 9. 参考

- `docs/deployment/a100-three-model-merge-plan.md` — 执行进度与逐项记录（§9–§10）
- `docs/deployment/minimax-m31-a100.md` §8 — base 构建与上线的原始操作手册
- `docs/deployment/v41-literal-sentinel-escape.md` — 哨兵修复的设计与评审
- `docs/deployment/v41-official-sampling-fix.md` §2.4–2.8 — 各家采样语义实测
- `docs/deployment/v41-metric-correction-rescore.md` — 判据纠正与重打分
