# MiniMax-M3.1-Flash 在 A100(sm80)上的移植、上线与出包实录

本文记录 2026-11 把 MiniMax-M3.1-Flash 搬进本 fork、在 A100 集群上线、并全量
重编 base 镜像的全过程。重点是踩过的坑、根因和解决办法，以及下次可以照着做的
操作指南（第 8 节）。

合入 master、Qwen3.8 回归、base 浮动 tag 切换、影响面分析和 na20 排查，见
[M3.1 合入 master、Qwen3.8 回归与 base 浮动 tag 切换记录](minimax-m31-master-merge-and-base-switch.md)。

**结论**

- 分支 `minimax-m31-a100`：从 `master`（`c14d2fbf9c`）切出，**没有整体 merge
  上游**，只按模型做了选择性移植。M3.1 在 4 台 × 4×A100-40G 上以 TP4×PP4、1M
  上下文服务。
- 现网 M3.1 用的镜像：`swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-backport:arm64-sm80-20261102-0949-12ed760d`，
  对应 gpustack 后端版本 `reputationly-backport-m31-custom`。
- 正式全量包：`arm64-sm80-20261102-1338-6bd0b071-full`（SWR、ACR 都有，另有同
  一镜像的 `arm64-sm80-base-20261102-1338-6bd0b071` tag；digest
  `sha256:5359bfadc7d4…`）。现网暂不替换。
- DSv4.1-Flash、Qwen3.8-Flash-Next 仍在原后端版本 `reputationly-backport-custom`
  （镜像 `arm64-sm80-20261030-1209-c14d2fbf`），**没有**被这次改动影响。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 线上节点 | 38 台 `newapi-A100-0001..0038`（ssh 别名 `na1..na38`，部分有 `llmNN` 别名），每台 4×A100-PCIE-40GB、251 GB 内存、Kunpeng ARM64 |
| M3.1 节点 | gpustack 标签 `m31=1`：`0001`(llm01) `0008`(llm08) `0023`(llm23) `0036`(na36) |
| 编译/测试机 | dev 集群 `gpu1..gpu55`（同为 ARM64 + A100，由 dev gpustack 托管） |
| 权重 | NFS `100.125.40.2:/MaaS_Models` → `/root/Models/MiniMax-M3.1-Flash`（48 个 shard，约 236 GB） |
| 网络 | 节点能访问阿里云镜像站、download.pytorch.org、ACR/SWR；访问 pypi.org、GitHub、crates.io、static.rust-lang.org 很慢或不通 |

> **规定：gpustack 托管的生产节点上不允许起任何非 gpustack 的服务**（测试容器、
> 编译、临时脚本服务都不行）。需要做 A/B 测试或编译，一律用 dev 集群；确需占用
> 生产空闲节点时，先征得负责人同意，测完立即清理。

---

## 2. 代码移植：问题与方案

分支上的提交（按顺序）：

| 提交 | 内容 |
|---|---|
| `42b08c310a` | 原样导入上游 `vllm/models/minimax_m3/`（上游 `e0d00e4b8e`） |
| `90cb874da5` | 补齐 M3.1 需要的共享符号（只增不改） |
| `c04ac0d8c1` | 模型内 shim，适配 fork 较旧的共享 API |
| `7f0d8ed06f` | MXFP8 checkpoint 里的 NVFP4 packed 专家改走 CT W4A4 |
| `35228e6169` | sm80 上用查表解码 e4m3fn 的 KV 字节 |
| `e89bb7b747` | 移植 M3 融合 kernel（nvfp4 insert、`write_qkv`、dspark） |
| `716c47d333` | PP 下不给其它 stage 的层分配 KV（上游 `29e66dad46`） |
| `35d77d9201` / `9dc8a5d50d` | 格式化；修 fp8-KV 测试的 shape 和 scale 参考值 |
| `05690e1235` | 移植上游 aiter `fused_qknorm_idxrqknorm` 开关与 runner |
| `aa80f1ca99` | `attention_config.minimax_m3_indexer_cp` + sm80 安全的初始化 |
| `12ed760d60` | `triton_mrope` 按显式步长寻址，去掉 `.contiguous()`（视觉修复） |
| `6bd0b07116` | Rust 前端编译修复（同时已合入 master：`961396034c`） |
| `77a79363a0` / `28b23142e5` | 构建流程支持镜像源/离线索引、版本号只认 `v*` tag、`docker/cn/mkindex.py`（master 上为 `a163f78cb7` / `5975ed0fac`） |

### 2.1 不做整体 merge

fork 从 `93911fcf68` 分叉，当时比上游多约 299 个提交、少约 3,187 个。试合并有约
202 个冲突文件，涉及 A100、Mooncake、DeepSeek、Qwen 和 worker 等核心路径。所以
只移植 M3.1 相关的部分：模型目录原样导入，共享代码只增不改，接口差异放在模型
内部用 shim 处理。M3 和 M3.1 共用 `minimax_m3` 目录（架构名都是
`MiniMaxM3SparseForConditionalGeneration`），差异由 config 决定。

### 2.2 权重加载报 `192 vs 384`

- 现象：`RuntimeError: The size of tensor a (192) must match the size of tensor b (384)`。
- 根因：checkpoint 的普通线性层是 `quant_method: mxfp8`，但路由专家是
  `moe_quant_algo: NVFP4`、`nvfp4-pack-quantized`、`group_size: 16`。专家
  `w1/w3.weight_scale` 是 `[3072, 384]`（6144/16），老代码按 MXFP8（6144/32=192）
  去解析。
- 方案：`moe_quant_algo == "NVFP4"` 时把 packed 专家路由到 compressed-tensors
  W4A4/NVFP4，在 A100 上由 Marlin W4A16 执行（`--moe-backend marlin`，`auto`
  也会选它）。
- 教训：先读 `config.json` 和 safetensors 头，不要只信上游的假设。

### 2.3 `--moe-backend emulation` 不存在

这是 p18 镜像的私有补丁，不在上游，也不在本 fork。A100 上用 `marlin`。

### 2.4 sm80 上不支持 FP8 E4M3 的 Triton 转换

KV / index cache 改用 `uint8` 视图存储，读取时用 `get_e4m3fn_bf16_lut()` 查表
解码；sm89+ 的原有路径不变。

### 2.5 sm100 专用模块在 A100 上导入失败

MegaMoE、M3 MSA warmup 依赖的 fmha 等模块改为在用到时才导入；融合 finalize
只在 sm100 上初始化。

### 2.6 C++ 融合 kernel

只移植 M3 的 kernel，不引入上游大规模的 CMake/源码重组。算子 schema 只在末尾追加
带默认值的 `kv_k_scale`、`kv_v_scale`、`write_qkv` 参数，保证向后兼容。DSpark 的
context-KV 算子也一并加入（尚未启用）。

### 2.7 视觉理解全错（上线后才发现）

- 现象：文本、工具调用都正常，但把文字密集的表格图描述成"旋转的数学公式"。
  第一次验证只用了红蓝色块图，颜色能认出来，所以没发现。
- 根因：fork 的 `triton_mrope` 先对 q/k 调用 `.contiguous()`。M3 视觉塔传入的是
  打包 qkv 上的非连续视图，并依赖原地旋转、丢弃返回值。结果 kernel 旋转的是拷贝，
  视觉注意力等于没有位置编码。上游在 `16070ed7e9` 里修过这个问题。
- 方案（`12ed760d60`）：kernel 接收 `q_token_stride`/`k_token_stride`，行偏移用
  int64 计算，去掉 `.contiguous()`，并断言输入是二维、最后一维连续。
- 影响面：Qwen3.8（`qwen4_exp`）的 `MRotaryEmbedding` 和 indexer 也调用这个 kernel。
  连续输入时结果不变，但 **Qwen3.8 升级到这个分支的镜像之前，必须先做一轮新旧
  镜像对比**（见 8.6）。
- 教训：验证视觉一律用文字密集的图（表格、截图），并问具体的单元格内容。

---

## 3. 上线（gpustack）

### 3.1 后端版本

gpustack 内置后端的自定义版本名**必须以 `-custom` 结尾**。新增版本时不要改动
`reputationly-backport-custom`，DSv4.1 和 Qwen3.8 都在用它：

```yaml
  reputationly-backport-m31-custom:
    image_name: >-
      swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-backport:arm64-sm80-20261102-0949-12ed760d
    run_command: ''
    entrypoint: ''
    custom_framework: cuda
    env: {}
```

推理后端配置是整体保存的。**不要用 API 整体覆盖**，以免误伤其它版本；改镜像地址
请在控制台里只改这一项。

### 3.2 模型 86（`minimax-m3.1-flash`）当前配置

- `backend_version=reputationly-backport-m31-custom`，`worker_selector={m31: 1}`，
  副本数 1，跨节点分布式。
- 参数：

```
--tensor-parallel-size=4 --pipeline-parallel-size=4 --max-model-len=1048576
--block-size=128 --kv-cache-dtype=fp8 --gpu-memory-utilization=0.90
--max-num-seqs=16 --max-num-batched-tokens=8192
--max-num-queued-tokens=1000000 --max-num-queued-reqs=32
--moe-backend=marlin --disable-custom-all-reduce --enable-prefix-caching
--enable-auto-tool-choice --tool-call-parser=minimax_m3 --reasoning-parser=minimax_m3
--distributed-timeout-seconds=3600 --cpu-distributed-timeout-seconds=3600
--compilation-config={"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":32}
--kv-transfer-config={"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"save_decode_cache":true,"cache_prefix":"minimax-m31","host_staging":true,"host_staging_bytes":"1GB"}}
--enable-log-requests
```

- 环境变量：`NCCL_IB_GID_INDEX=3`，`NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME=enp131s0f0`，
  `NCCL_CROSS_NIC=0`，`NCCL_IB_TIMEOUT=22`，`NCCL_ALGO=Ring`，`NCCL_TIMEOUT=3600`，
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`，`OMP_NUM_THREADS=8`，
  `MOONCAKE_CONFIG_PATH=/root/Models/_gpustack/mooncake-minimax.json`，
  `VLLM_MOONCAKE_LOAD_RECV_THREADS=8`，`VLLM_USE_BREAKABLE_CUDAGRAPH=0`。
- 与 p18 镜像原配置相比去掉了：`--moe-backend=emulation`、`--trust-remote-code`
  （vLLM 自带 M3 config）、`--skip-mm-profiling`（否则不给视觉编码器预留显存，
  40GB 卡会 OOM）、`VLLM_M31_W4A16_EMU`（p18 私有开关）。
- 每次改模型 86 之前的完整配置都备份在 `llm01:/root/m31/m86-before-*.json`。

### 3.3 上线时踩的坑

| 问题 | 根因 | 解决 |
|---|---|---|
| SWR 推送报 `fail to parse 'manifest.json'` | docker 29（containerd 存储）默认推多平台索引清单，SWR 不认 | `docker push --platform linux/arm64 …` 推单个 arm64 清单 |
| 前两次实例启动即失败 | na27–38 的 gpustack-worker 找 `gpustack/runtime:pause`，节点上只有 `quay.io/gpustack/runtime:pause`，又上不了外网 | 38 台都补打 `gpustack/runtime:pause` tag |
| 调度到新节点要现场拉 20+GB 镜像 | 镜像只在少数节点上有 | 38 台预拉 M3.1 镜像（只需拉增量层，很快） |
| 推理请求报 `--moe-backend emulation` 无效 | p18 私有值 | 改 `marlin` |
| 图片理解错误 | 见 2.7 | 修复前用 `--limit-mm-per-prompt={"image":0,"video":0}` 让图片请求明确返回 400；修复后去掉 |
| `restart_on_error=false` 时报错实例不会自愈 | 模型配置 | 删掉报错实例，gpustack 按副本数重新调度 |

### 3.4 Mooncake 大池子

- 38 台都跑着 `mooncake_client`，注册在同一个 master `10.0.0.218:50061` 上，每台
  贡献 16 GB，合计约 608 GB。M3.1 的 `mooncake-minimax.json` 也指向它。
- M3.1 有主 KV 和稀疏 indexer 两组 cache，两组都要能存进池子、读回来。验证方法：
  实例刚重启、本地缓存为空时，同一个长请求的 24,192 个 token 从池子读回，答案
  仍然正确。
- 池子是共享的，冷门前缀会被其它模型挤掉，所以重启后第一次请求命中很少是正常的。

---

## 4. 验证结果

### 4.1 M3.1（16×A100，1M 上下文）

| 项 | 结果 |
|---|---|
| KV 容量 | 约 255.6–261 万 token（1M 请求约可并发 2.4–2.5 个） |
| GSM8K（500 题） | 95.6%（478/500，0 无效，22 题在 `max_tokens=1024` 截断） |
| 工具调用 / 推理解析 | 格式正确 |
| 长上下文 | 10 万、20 万/50 万 token 大海捞针均找到 |
| 视觉（修复后） | 表格表头和单元格、基准测试图、色块+文字、自然照片均正确 |
| Mooncake | 远端读回 24,192 token，答案正确 |

### 4.2 对现网模型的回归（新镜像 vs 线上镜像，同参数）

- **Qwen3.8-Flash-Next**（视觉修复之前的镜像）：KV 容量相同（4,391,235），
  GSM8K 93.2% vs 92.8%，MTP 接受长度 2.172 vs 2.180，工具和大海捞针通过，没有
  新增告警。最初看到 4–6% 的吞吐差距，在两组节点上交叉对调、每组跑 3 遍后确认是
  浮动：差距在 ±1.5% 以内且方向相反；并发 16 的第一遍普遍偏低，是没在该并发下
  预热造成的。
- **DSv4.1-Flash**：功能、KV 容量（1,251,508）、dspark 接受长度、大海捞针、告警
  都一致；GSM8K 95.8% vs 94.4%。新镜像吞吐低约 2.6–2.9%，没到 3% 的门槛，但方向
  一致，**未复核**（用户决定不做）。
- **Qwen3.8-Flash-Next，正式全量包 `6bd0b071`（含 mrope 修复）**：第一次在 na20+na38
  上用 docker 直接起服务时，na20 整机宕机（见 7.1）。后来改用 gpustack 临时测试模型
  （`qwen3.8-ab-test`，线上配置去掉 Mooncake），在同一对节点上先测线上镜像、再测
  新镜像，两轮都很稳定：

  | 项 | 线上 `c14d2fbf` | 新 `6bd0b071` |
  |---|---|---|
  | KV | 21.91 GiB | 21.91 GiB |
  | 20 条固定提问逐字对比（温度 0） | 自己跑两遍有 12/20 不一致 | 与线上 11/20 不一致，在同一范围内 |
  | 工具调用 | 2/2 | 2/2 |
  | GSM8K 500 | 92.8% | 92.6% |
  | MTP 接受长度 | 约 3.0–3.7 | 约 3.0 |
  | 并发 1 吞吐 | 90.4 tok/s | 92.8 tok/s（+2.65%） |
  | 并发 16 吞吐 | 287.9 tok/s | 280.3 tok/s（−2.65%，3 遍 273.8→281.1→285.9） |
  | 20 万 token 大海捞针 | 找到 | 找到 |
  | 文字密集图片 / 照片 | 读对 | 与线上读出的内容完全一致 |
  | 日志告警 | 基线 | 无新增 |

  两种并发下的差距方向相反，且都不到 3%，判定通过。并发 16 是需要留意的一项。

### 4.3 `reasoning_effort` 的行为（M3.1）

- M3.1 官方 chat template 把思考模式**写死为 adaptive**，没有关闭思考的开关。
  `reasoning_effort` 只会在系统提示末尾插入一行 `<effort>xxx</effort>`。官方支持的
  值只有 `low/medium/high/xhigh/max`。
- vLLM 额外传入的 `enable_thinking` 不被这个模板读取，所以 `none` 等同于不传，
  `minimal` 是官方未定义的值。
- 实测（温度 0，中等难度题，思考 token 数）：low 57 → medium 204 → high 191 →
  xhigh 237 → max 345，基本单调；none 195、不传 191。
- 建议客户端只给 M3.1 提供 `low/high/max` 这几档，不要提供 off。如果确实需要关闭
  思考，要定制 chat template，把 M3 时代的 `thinking_mode=disabled` 加回去（回答
  开头预填 `</mm:think>`），上线前需验证。

---

## 5. 全量重编 base 镜像（出正式包）

### 5.1 为什么不能只用 GitHub 流水线

- `build-arm64-app.yml`（GitHub 托管 4 核 arm64，限时 120 分钟）只把 Python/Triton
  叠加到已有的 base 上，**不编 C++**。这次改了 `csrc`，只跑它会发出带旧 `.so` 的
  镜像，M3.1 一调用新的算子参数就会报错。
- 全量 nvcc 编译超出 GitHub 托管 runner 的 6 小时上限和约 14GB 磁盘
  （`docker/build_base_aarch64.sh` 文件头已说明）。`docker-publish.yml` 要求
  self-hosted runner，而仓库一台都没注册，所以一直是 cancelled。
- 所以 base 按 fork 的流程在自己的 ARM64 机器上编：编完推 ACR，再用
  `sync-base-to-dockerhub.yml` 同步到 Docker Hub，供 app 流水线使用。

### 5.2 一路踩过的坑（按出现顺序）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | gpu7 上 7 次构建都死在 `vllm-runtime-base 15/19` / `base 11/13` | 外网慢：uv 安装器、flashinfer（实际从 GitHub release 下载）、pypi 包超时；`z3-solver` 等现场从源码编也超时 | 依赖全部离线化，见下 |
| 2 | 第 1 次超时后，之后几次一直失败 | 第一次超时在 uv 缓存里留下了一个损坏的 flashinfer 半包 | 清理 buildkit 缓存 |
| 3 | uv 报等锁 1800 秒超时 | 两个构建同时跑，抢同一个 `/opt/uv/cache` | 永远只跑一个构建；慢网下先 `BUILD_TARGET=base` 单独跑一趟 |
| 4 | `base` 阶段解析依赖要 17 分 42 秒 | `base` 阶段**没有声明 `ARG PIP_INDEX_URL UV_INDEX_URL`**，脚本传了镜像源也没用，仍走 pypi.org | 补上 ARG（已提交） |
| 5 | flashinfer-jit-cache 安装会失败（在 nvcc 之后） | 下载地址写死为 flashinfer.ai（跳转 GitHub）；临时改成本地后 IP 又写成了旧机器 | 下载地址参数化（已提交） |
| 6 | `llguidance`/`z3-solver`/`tilelang` 在 `base` 阶段从源码编 | builder 是 manylinux_2_28（glibc 2.28），官方 aarch64 wheel 要求 glibc 2.31/2.34 | z3/tilelang 编一次后进 uv 缓存；llguidance 用 Mac 上 vendor 的 351 个 crate 离线编出 `manylinux_2_28` wheel 放进本地索引 |
| 7 | 最后打包阶段还要装 accelerate、modelscope、LMCache、nixl、mooncake 的依赖 | 这些不在 `requirements/cuda.txt` 里 | 一并下载进本地索引；LMCache 要求更低版本的 numpy/prometheus/opentelemetry，照样保留 |
| 8 | rustup 下载工具链 44 分钟、自更新 14 分钟，随后 crates.io SSL 失败 | 外网 | rustup 用 `rsproxy.cn`（实测 29 MB/s），crates 用阿里云 sparse 镜像；cargo 会按 `Cargo.lock` 校验和逐个核对 |
| 9 | `error[E0599]: no variant named 'Custom' found for enum 'ChatMessage'` | master 上的 `5132bea899` 从上游搬了一个 `ChatMessage::Custom` 分支，fork 的枚举没有这一项。现网 base 编于此提交之前，一直没暴露 | `6bd0b07116` 删掉这一行（永远匹配不到，行为不变），已合入 master（`961396034c`） |
| 10 | `ValueError: Can't parse version from tag 'm31-tip'` / `'arm64-sm80-…'` | setuptools_scm 取最近的 tag，而 app 流水线每次出包都打一个 `arm64-sm80-*` tag，解析不了 | 版本号只匹配 `v*` tag（已提交）；当次构建在 gpu12 的本地仓库副本里临时删掉这些 tag |

最终在 gpu12 上从启动到出镜像只用了 26 分钟（大部分 C++ 已在编译缓存里）。

第 4、5、8、10 条的临时绕过方式，之后已做成正式提交（master `a163f78cb7`）：新增的
构建参数都是可选的，不设置时构建行为与原来完全一致。为了验证，在 gpu12 上把构建
目录恢复成干净代码、恢复全部发布 tag，只靠这些参数完整构建了一次，结果成功，版本号
为 `0.13.2.dev101+ga163f78cb`，不带 `.d` 后缀。

### 5.3 本地离线索引的做法

1. 在 Mac 上用 `uv pip compile` 解析出全部依赖的确切版本（目标平台
   `aarch64-manylinux_2_28` 和 `_2_39`，Python 3.12）。Mac 上 17 秒，服务器上要
   17 分钟。
2. 在 Mac 上下载 Linux aarch64 的 wheel（`pip download --only-binary=:all:
   --platform … --no-deps`，逐包并行；`-r` 一个失败就会整体退出）。逐个校验
   sha256，确认能当 zip 打开。
3. `rsync --partial` 传到编译机，用 `docker/cn/mkindex.py --cuda-suffix cu130`
   生成 PEP 503 索引（`simple/` 和 `simple/cu130/` 两份），再用
   `python3 -m http.server` 在内网提供服务。
4. 构建时把 PyPI、PyTorch、flashinfer 三个下载源都指向这个索引，并在构建前离线
   验证：只用本地索引 `uv pip compile` 所有依赖集合，确认都能解析成功。

---

## 6. 产物与清理

- 正式包：SWR 和 ACR 各两个 tag，均为 `sha256:5359bfadc7d4b86f5d29e6fd7f620a415d1e7f29114c9b3c659eb446179f90fc`，36.8 GB。
  - `…/vllm-backport:arm64-sm80-20261102-1338-6bd0b071-full`
  - `…/vllm-backport:arm64-sm80-base-20261102-1338-6bd0b071`
- 已同步到 Docker Hub：`arronlee/vllm-backport:arm64-sm80-base-20261102-1338-6bd0b071`。
- 校验：镜像里 2542 个 `.py` 文件与 `6bd0b07116` 逐个比对一致；`import vllm`
  正常；M3.1 融合算子带 `write_qkv`；DSpark 算子存在。
- M3.1 分支已合入 master（`92a339955e`），合并后的代码与分支完全一致。
- Qwen3.8 对比通过后，**ACR 和 Docker Hub 的浮动 tag `arm64-sm80-base` 已切到
  `sha256:5359bfadc7d4…`**（即上面这个 base）。之后 master 的 app 流水线默认基于它
  出包。`arm64-sm80-latest` 只在 app 流水线出包时移动，这次没动。
- 回滚：旧的 base（`sha256:69bf05711b31…`）保留为固定 tag
  `arm64-sm80-base-prev-20261030-c14d2fbf`（ACR 和 Docker Hub 都有）。回滚时把
  `arm64-sm80-base` 复制回这个 tag，再跑一次 `sync-base-to-dockerhub.yml`：
  ```bash
  # 在 devm 上（已登录 ACR）
  A=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport
  skopeo copy --authfile ~/.docker/config.json \
    docker://$A:arm64-sm80-base-prev-20261030-c14d2fbf docker://$A:arm64-sm80-base
  gh workflow run sync-base-to-dockerhub.yml -f base_tag=arm64-sm80-base
  ```
- 现网 DSv4.1、Qwen3.8、M3.1 的 gpustack 后端版本都没有改动，切 tag 不影响正在运行
  的服务。
- 编译机文件保留在 `gpu12:/root/m31/`（本地 wheel 索引、llguidance vendor 包、
  日志）；临时文件服务和容器已停。

---

## 7. 事故与隐患

### 7.1 na20 宕机（原因未确认）

- 在 na20+na38 上以 Qwen3.8 现网配置做对比测试时，na20 在 13:31（北京时间）整机
  失联，需要从 BMC 重启。
- 系统日志在 13:27:55 停止写入，没有 OOM、panic、Xid 记录；vLLM 日志一直写到
  13:31:19（QSA 预热完成，下一步是 CUDA graph 捕获）。na20 在 10 月 18 日有过
  `NVRM: Assertion failed: pIOVAS != NULL` 的驱动报错。更倾向于 GPU、驱动或 PCIe
  问题，但没有确证。
- 建议运维查 BMC SEL；查清之前不要把重要服务调度到 na20（可打开 gpustack 维护
  模式）。

### 7.2 Qwen3.8 leader 节点内存余量小

线上三个 Qwen3.8 leader 容器各占 158–163 GB 内存（其中约 138 GB 是 PLE 卸载用的
共享内存），整机可用只剩 64–67 GB。

### 7.3 凭据

本次沟通中出现过 SWR 密码和 gpustack 管理 token，均未写入仓库和日志，建议轮换。

---

## 8. 后续操作指南

### 8.1 只改了 Python / Triton

用 GitHub 的 `build-arm64-app.yml`：

- `base_image`：默认的 `arronlee/vllm-backport:arm64-sm80-base` 现在就是
  `6bd0b071` 的 base；也可以写成带时间戳的固定 tag
  `arronlee/vllm-backport:arm64-sm80-base-20261102-1338-6bd0b071`。
- `base_commit`：填 base 对应的提交 `6bd0b07116`。流水线会检查之后 `csrc`、`cmake`、
  `setup.py`、`requirements`、`docker/Dockerfile` 有没有被改动，改了就要求重编 base。
  注意：`a163f78cb7` 改过 `docker/Dockerfile` 和 `setup.py`，但只是构建参数和版本号
  逻辑，不影响编出来的 C++，所以这道检查会报警。在下一次重编 base 之前，可以把
  `base_commit` 留空跳过检查，前提是确认之后确实没有人改过 `csrc`、`cmake` 和依赖。
- `dest_tag`：如果不想移动 `arm64-sm80-latest`，填一个专用的浮动 tag。

### 8.2 改了 C++ / CMake / 依赖：重编 base

1. **选机器**：dev 集群上内存最空的 ARM64 节点（编译只用 CPU 和内存，不占 GPU）。
   这次 gpu12 约有 224 GB 可用内存，跑 48 个并行任务没有问题；内存更少的机器
   要相应调低 `MAX_JOBS`。**不要用 gpustack 生产节点。**
2. **准备离线依赖**（服务器能直接下就跳过）：
   ```bash
   # Mac：解析版本（按需加 requirements/build/cuda.txt、accelerate、modelscope、
   # requirements/kv_connectors.txt、LMCache requirements 等集合）
   uv pip compile requirements/cuda.txt --python-platform aarch64-manylinux_2_28 \
     --python-version 3.12 --index-url https://pypi.org/simple \
     --extra-index-url https://download.pytorch.org/whl/cu130 \
     --index-strategy unsafe-best-match --no-header --no-annotate -o pins.txt
   # 逐包下载 aarch64 wheel（失败的包放宽 --platform 到 manylinux_2_39 再试）
   uv run --no-project --with pip python -m pip download --only-binary=:all: --no-deps \
     --python-version 3.12 --implementation cp --abi cp312 --abi abi3 --abi none \
     --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 --platform any \
     --extra-index-url https://download.pytorch.org/whl/cu130 \
     --extra-index-url https://flashinfer.ai/whl/ -d wh "<name>==<version>"
   rsync -a --partial wh/ <编译机>:/root/m31/findlinks/
   ```
   flashinfer-jit-cache 要从 `https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/`
   单独下载。在 glibc 2.28 上没有 wheel 的包（llguidance、tilelang、z3-solver）
   要么按 5.2 第 6 条提前编好 `manylinux_2_28` wheel 放进去，要么把它们的源码包
   **连同 `build-system.requires` 里的构建依赖**一起放进索引（tilelang 需要
   cython、scikit-build-core、pathspec、patchelf 等，llguidance 需要 maturin）。
   换了下载源之后，uv 缓存里之前编好的 wheel 不会复用，会重新从源码编。
3. **生成并提供索引**，然后离线验证：
   ```bash
   python3 docker/cn/mkindex.py /root/m31/findlinks --cuda-suffix cu130
   cd /root/m31/findlinks && nohup python3 -m http.server 8917 &
   # 在 builder 镜像里只用本地索引 uv pip compile 各依赖集合，必须全部成功
   ```
4. **构建**（一次只跑一个；镜像源通过环境变量传入，不需要改任何文件）：
   ```bash
   PUSH=0 MAX_JOBS=48 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=8.0 \
   BUILD_BASE_IMAGE=<builder 镜像> FINAL_BASE_IMAGE=<cuda base 镜像> \
   GET_PIP_URL=https://mirrors.aliyun.com/pypi/get-pip.py \
   DEADSNAKES_MIRROR_URL=https://mirrors.aliyun.com \
   PIP_INDEX_URL=http://<编译机IP>:8917/simple/ \
   PYTORCH_CUDA_INDEX_BASE_URL=http://<编译机IP>:8917/simple \
   FLASHINFER_INDEX_BASE_URL=http://<编译机IP>:8917/simple \
   CARGO_REGISTRY_MIRROR=sparse+https://mirrors.aliyun.com/crates.io-index/ \
   RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
   nice -n 19 bash docker/build_base_aarch64.sh
   ```
   - 编译机连不上 GitHub 时，用 `git bundle` 把提交传过去，保持 commit 哈希一致。
   - 每 10 分钟检查一次日志；可用内存低于 25 GB 要立即处理，不要拖垮同机的服务。
   - 不要在编译机上留未提交的改动，否则版本号会带 `.dYYYYMMDD`。
5. **校验**：`import vllm` 和 `vllm.__version__`；关键算子的 schema；镜像里的
   `.py` 文件与 `git archive <commit>` 逐个比对 md5。
6. **推送**（只用带时间戳的不可变 tag）：
   ```bash
   docker push --platform linux/arm64 swr.cn-central-221.ovaijisuan.com/whcs-fae/vllm-backport:<tag>
   # ACR：在 devm（已登录 ACR）上 pull --platform linux/arm64 → tag → push --platform linux/arm64
   ```
7. **同步 Docker Hub**：
   `gh workflow run sync-base-to-dockerhub.yml -f base_tag=<带时间戳的 base tag>`。
8. **浮动 tag**：只有在所有现网模型都用新 base 回归通过后，才切换
   `arm64-sm80-base`。

### 8.3 上线一个新镜像到 gpustack

1. 先在全部 38 台上预拉镜像，并确认 `gpustack/runtime:pause` tag 存在。
2. 推理后端里**新增**一个 `*-custom` 版本，不要改别的模型在用的版本。
3. 只改目标模型的 `backend_version` 和参数；改之前备份模型 JSON。
4. 实例就绪后的验证清单：数学题；中文；工具调用；推理内容解析；2 万 token 以上的
   大海捞针；**文字密集的图片**（读具体单元格）；同一长请求重启后从 Mooncake 读回
   并核对答案；检查请求日志，确认没有异常。

### 8.4 回归测试方法

- 参数、环境变量照搬线上实例（`docker inspect` 和 `pgrep -af "vllm serve"` 实时读）。
  A/B 时可以去掉 Mooncake；如要验证，用独立的 `cache_prefix`。
- 同一组节点上先跑线上镜像，再跑新镜像。性能有差距时，**交叉对调节点**再测一轮。
- 压测：每种并发先单独预热一遍，并发 1 跑 32 条、并发 16 跑 128 条，各重复 3 遍。
- 对比项：启动、KV 容量、20 条固定提问逐字对比（线上镜像自己先跑两遍，作为基线）、
  GSM8K 500 题、投机接受率、吞吐、20 万 token 大海捞针、日志告警，视觉模型加上文字
  密集图片。
- 门槛：启动失败、精度或接受率下降、两种并发下都稳定慢 3% 以上，就停下来分析，
  不要直接改共享代码。

### 8.5 待办

- [x] Qwen3.8 在正式全量包 `6bd0b071` 上做新旧对比：通过（见 4.2）。
- [x] 浮动 tag `arm64-sm80-base` 切到新 base：已完成（见第 6 节）。
- [ ] 现网 Qwen3.8 / DSv4.1 是否迁移到基于新 base 的镜像：要迁移时，先出 master 的 app
      镜像，新增一个 `*-custom` 后端版本，再逐个切换并观察。
- [ ] DSv4.1 约 3% 的吞吐差距：如需复核，在同一组节点上按"线上 → 新 → 线上"交替
      压测，并用"新镜像 + 线上 C++ 扩展"的混合镜像区分 C++ 和 Python 的影响。
- [ ] na20 的硬件排查（BMC SEL）。
- [ ] 轮换本次暴露过的 SWR 密码和 gpustack token。
- [ ] 如需让 M3.1 支持关闭思考，评估定制 chat template（4.3）。
