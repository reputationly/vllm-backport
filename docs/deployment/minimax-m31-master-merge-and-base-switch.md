# M3.1 合入 master、Qwen3.8 回归与 base 浮动 tag 切换记录

本文接着 [MiniMax-M3.1-Flash 在 A100 上的移植、上线与出包实录](minimax-m31-a100.md)，
记录 2026-11-02 至 11-03 的收尾工作：把 M3.1 合入 master、用 gpustack 在生产空闲
节点上做 Qwen3.8 新旧镜像对比、切换 base 浮动 tag，以及影响面分析、na20 排查和之后
需要注意的事项。

## 结论

| 事项 | 结果 |
|---|---|
| M3.1 代码和文档合入 master | 已完成，合并提交 `92a339955e`，合并后的代码与 `minimax-m31-a100` 分支完全一致 |
| Qwen3.8 在正式全量包 `6bd0b071` 上的回归 | 通过（na20+na38，gpustack 临时模型，线上配置去掉 Mooncake） |
| base 浮动 tag `arm64-sm80-base` | ACR 和 Docker Hub 都已切到 `sha256:5359bfadc7d4…`；旧 base 备份为 `arm64-sm80-base-prev-20261030-c14d2fbf` |
| 现网 Qwen3.8 / DSv4.1 | **不迁移**，继续用 `reputationly-backport-custom`（镜像 `arm64-sm80-20261030-1209-c14d2fbf`） |
| 现网 M3.1 | 不替换，继续用 `reputationly-backport-m31-custom`（镜像 `arm64-sm80-20261102-0949-12ed760d`） |
| 视觉修复（`triton_mrope`）的影响面 | 只影响 M3.1 视觉塔和 Qwen3.8（`qwen4_exp`），**不影响 DSv4.1** |
| na20 | 当前健康；11-02 宕机原因仍未确认 |
| DSv4.1 约 3% 吞吐差距 | 不急，暂不复核 |

---

## 1. 合入 master

### 1.1 master 上相关提交的顺序

| 提交 | 内容 |
|---|---|
| `961396034c` | Rust 前端编译修复（cherry-pick 自分支 `6bd0b07116`）：删掉永远匹配不到的 `ChatMessage::Custom` 分支 |
| `a163f78cb7` | 构建流程支持镜像源和离线索引；版本号只认 `v*` tag；新增 `docker/cn/mkindex.py` |
| `5975ed0fac` | `mkindex.py` 打印路径的小修 |
| `92a339955e` | `Merge branch 'minimax-m31-a100'`（`--no-ff`） |
| `0923279361` | 总文档补充 Qwen3.8 结果、切 tag 和回滚方法 |

### 1.2 做法

- 在独立的 worktree 里基于 `origin/master` 执行 `git merge --no-ff origin/minimax-m31-a100`。
  分支上已经 cherry-pick 了 master 的那三个提交，内容相同，所以没有冲突。
- 合并后执行 `git diff HEAD origin/minimax-m31-a100`，结果为空，确认 master 和分支完全
  一致后再推送。
- 推送 master 会触发 `docker-publish.yml`。它要求 self-hosted runner，而仓库没有注册
  任何 runner，所以和以往一样会被取消，不会产生任何镜像。只改 `docs/**` 的提交不会触发
  它（`paths-ignore`）。

### 1.3 合入后 master 上被改到的共享代码（M3.1 目录之外）

`CMakeLists.txt`、`csrc/libtorch_stable/{ops.h,torch_bindings.cpp}`、`setup.py`、
`vllm/_aiter_ops.py`、`vllm/_custom_ops.py`、`vllm/config/attention.py`、
`vllm/model_executor/layers/fused_moe/{all2all_utils,config,layer}.py`、
`fused_moe/oracle/nvfp4.py`、`fusion/quant_activation.py`、
`quantization/compressed_tensors/.../compressed_tensors_moe_w4a4_nvfp4.py`、
`quantization/modelopt.py`、`quantization/utils/quant_utils.py`、
`rotary_embedding/{mrope,mrope_vit_setup}.py`、`warmup/attention_warmup.py`、
`vllm/utils/flashinfer.py`、`vllm/v1/attention/backend.py`、`vllm/v1/attention/ops/fp8_sm80.py`。

这些改动的性质见第 4 节。

---

## 2. Qwen3.8 新旧镜像回归

### 2.1 为什么用 na20+na38，而且必须通过 gpustack

- dev 集群 gpu1–55 里没有内存和显存同时空闲的机器。Qwen3.8 线上配置每台需要 4 张全空的卡，
  leader 还要约 170 GB 内存（容器约 160 GB，其中约 138 GB 是 PLE 卸载用的共享内存）。
  - 内存够的机器（gpu1/2/3/11/12/13/15/16/51–54，可用 216–228 GB）GPU 上都跑着 gpustack 服务。
  - 显卡空闲的机器（gpu7/8/18/29 等）可用内存只有 60–115 GB。
- 生产集群当时只有 na20、na38 两台没有实例。
- **规定：gpustack 托管的生产节点上不允许起非 gpustack 的服务。** 所以这次没有用
  `docker run`，而是在 gpustack 里建临时模型来跑测试。

### 2.2 步骤

1. 给 `newapi-A100-0020`、`newapi-A100-0038` 合并加上标签 `qwen-ab=1`，保留原有标签，
   并记下原值，测完恢复。
2. 新建模型 `qwen3.8-ab-test`，从模型 62（`qwen3.8-flash-fp8`）复制：
   - 权重、`backend_parameters`、`env`、跨节点分布式设置都相同；
   - **去掉 `--kv-transfer-config`**，不碰线上 Mooncake 缓存；
   - `replicas=1`、`worker_selector={"qwen-ab": "1"}`、`restart_on_error=false`。
3. 第 1 轮：`backend_version=reputationly-backport-custom`（线上镜像 `c14d2fbf`）。
4. 第 2 轮：由用户在控制台新增后端版本 `reputationly-backport-qwentest-custom`，指向
   `…/vllm-backport:arm64-sm80-20261102-1338-6bd0b071-full`。
   - 先通过 API 核对版本存在、镜像地址一致；
   - 在两台节点上预拉镜像；
   - 把测试模型切到新版本，删掉实例让 gpustack 重建；
   - 到节点上确认容器跑的确实是新镜像。
5. 测试脚本都从 llm01 和 gpu7 发起，不在 na20/na38 上跑任何额外进程。期间每 4–5 分钟
   从外部检查一次 na20：ping、ssh、gpustack 心跳。
6. 清理：
   - 删掉测试模型，恢复两台节点的标签；
   - 通过 API 确认没有残留实例，`nvidia-smi` 确认显存回到约 4 MiB；
   - na20 上残留了一个 follower 容器，手动 `docker rm` 掉了。gpustack 删除模型后不一定能
     清干净，**以后清理时要逐台检查**。

### 2.3 结果

| 项 | 线上 `c14d2fbf` | 新 `6bd0b071` | 判定 |
|---|---|---|---|
| 启动 / KV | 约 11–12 分钟，21.91 GiB，1M 上下文最多并发 4.19 个 | 约 10 分钟，21.91 GiB，最多并发 4.28 个 | ✅ |
| 20 条固定提问逐字对比（温度 0） | 自己跑两遍有 12/20 不一致 | 与线上 11/20 不一致 | ✅ 在同一范围内 |
| 工具调用 | 2/2 | 2/2 | ✅ |
| GSM8K 500 | 92.8%（464/500） | 92.6%（463/500） | ✅ 在误差范围内 |
| MTP 接受长度 | 约 3.0–3.7 | 约 3.0 | ✅ |
| 吞吐，并发 1（32 条 ×3） | 90.4 tok/s（90.2/90.8/90.2） | 92.8 tok/s（92.8/92.3/93.2），+2.65% | ✅ |
| 吞吐，并发 16（128 条 ×3） | 287.9 tok/s（288.2/287.2/288.2） | 280.3 tok/s（273.8/281.1/285.9），−2.65% | ✅ 需留意 |
| 20 万 token 大海捞针 | 找到，10.2 秒 | 找到，15.0 秒 | ✅ |
| 文字密集图片 / 照片 | 读对（如 LOME 9.36、Ctrl-World 22.99、Cosmos3-Super 26.0） | 与线上读出的内容完全一致 | ✅ |
| 日志告警 | 基线 | 无新增 | ✅ |

- **吞吐的判定**：两种并发下的差距方向相反，且都不到 3%，所以不算"同方向持续慢 3% 以上"。
  新镜像并发 16 的 3 遍一路上升（273.8 → 281.1 → 285.9），第 3 遍和线上只差 0.8%，更像
  预热没完全稳定。
- **大海捞针多出约 5 秒**：只测了一次，在单次波动范围内。
- **图片**：新旧镜像的输出完全一致，说明 mrope 修复对 Qwen3.8 的连续输入是等价的，符合预期。
- 结果和日志：`llm01:/root/m31/qwen-ab2/`（`det-*.json`、`gsm8k-*.json`、`needle-*.log`、
  `inst1895-full.log`、`inst1897-full.log`）；压测日志在 `gpu7:/root/m31/qwen-ab2/`。

---

## 3. base 浮动 tag 切换

### 3.1 切换前后

| tag | 切换前 | 切换后 |
|---|---|---|
| ACR `…/vllm-backport:arm64-sm80-base` | `sha256:69bf05711b3199177f829a16d60a8e786fa1982acf63aafd75ae1b578a7d878d` | `sha256:5359bfadc7d4b86f5d29e6fd7f620a415d1e7f29114c9b3c659eb446179f90fc` |
| ACR `…:arm64-sm80-base-prev-20261030-c14d2fbf` | 不存在 | `sha256:69bf0571…`（旧 base 的备份） |
| Docker Hub `arronlee/vllm-backport:arm64-sm80-base` | 旧 base | 新 base（同步流水线 run `41561999783`，成功） |
| Docker Hub `…:arm64-sm80-base-prev-20261030-c14d2fbf` | 不存在 | 旧 base（run `41562013246`，成功） |
| `arm64-sm80-latest` | 未动 | 未动（只在 app 流水线出包时才会移动） |

### 3.2 操作方法

devm（`dev-gpustack-manager`，x86_64，已登录 ACR）上没有 buildx，但有 skopeo，可以直接
在镜像仓库里复制 tag，不用把 36 GB 的镜像拉到本地：

```bash
A=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-backport
AF=~/.docker/config.json
# 1) 先备份旧的浮动 tag
skopeo copy --authfile $AF docker://$A@sha256:69bf0571… docker://$A:arm64-sm80-base-prev-20261030-c14d2fbf
# 2) 移动浮动 tag
skopeo copy --authfile $AF docker://$A:arm64-sm80-base-20261102-1338-6bd0b071 docker://$A:arm64-sm80-base
# 3) 核对 digest（manifest 原文的 sha256）
skopeo inspect --authfile $AF --raw docker://$A:arm64-sm80-base | sha256sum
# 4) 同步 Docker Hub
gh workflow run sync-base-to-dockerhub.yml -f base_tag=arm64-sm80-base
gh workflow run sync-base-to-dockerhub.yml -f base_tag=arm64-sm80-base-prev-20261030-c14d2fbf
```

- devm 上的 skopeo 版本较旧，**不支持 `--preserve-digests`**。不加这个参数时，同一个
  仓库内、格式不变的复制会保持 digest 不变，但复制完必须用第 3 步核对一遍。
- 回滚：把第 2 步的源换成 `arm64-sm80-base-prev-20261030-c14d2fbf`，再同步一次 Docker Hub。

### 3.3 切 tag 的影响

- 现网三个模型的 gpustack 后端版本都写死了带时间戳的镜像地址，切 tag 不影响正在运行的服务。
- 之后用 `build-arm64-app.yml` 出包时，默认的 `base_image`
  （`arronlee/vllm-backport:arm64-sm80-base`）就是新 base，出来的镜像会带上 M3.1 的全部
  改动（包括共享代码）。

---

## 4. 影响面分析：视觉修复只影响 Qwen3.8，不影响 DSv4.1

### 4.1 `triton_mrope` 的调用方（合并后的 master）

- `vllm/models/minimax_m3/common/vision_tower.py`：M3.1 视觉塔，修复的就是它。
- `vllm/models/qwen4_exp/{nvidia,amd}/indexer_qsa.py`：Qwen3.8 的 indexer。
- `MRotaryEmbedding.forward_cuda`（`rotary_embedding/mrope.py`）：只有 `rope_parameters`
  里带 `mrope_section` 的模型才会构造它（`rotary_embedding/__init__.py`），Qwen3.8 属于这种情况。

DSv4.1 的 RoPE 由 `vllm/models/deepseek_v4_1/common/rope.py` 调 `get_rope(...)` 创建，
`rope_type="deepseek_yarn"`，没有 `mrope_section`，**走不到 `triton_mrope`**。

### 4.2 其它共享改动和 DSv4.1 的关系

| 改动 | 对 DSv4.1 |
|---|---|
| `csrc` / `CMakeLists.txt`：新增 M3 融合算子和 DSpark 算子，schema 只在末尾追加带默认值的参数 | 只增不改，不调用 |
| `modelopt.py` / compressed-tensors NVFP4：`moe_quant_algo == "NVFP4"` 时才生效 | 不触发 |
| `fused_moe/*`、`quant_utils.py`、`flashinfer.py`、`attention/backend.py`、`config/attention.py` | 只新增符号、配置项和分支；DSv4.1 的执行路径上有其中几个文件，已由下面的对比覆盖 |
| `fp8_sm80.py` | 只多了一个"没有 Triton 时"的判断，有 Triton 时行为完全不变 |
| `_aiter_ops.py` | 只在 AMD 上使用 |
| `warmup/attention_warmup.py` | sm100 专用的 import 改为用到时才导入 |
| Rust 前端修复 | 现网没有开 `VLLM_USE_RUST_FRONTEND` |

- 之前 DSv4.1 的新旧镜像对比用的是 `aa80f1ca` 镜像，已经包含上表所有改动（只差 mrope 修复
  和 Rust 修复）。结果：功能、KV 容量、dspark 接受长度、大海捞针、告警都一致；吞吐低约
  2.6–2.9%，**未复核**。
- 现网 DSv4.1 不迁移，所以目前没有任何影响。

---

## 5. na20 排查

### 5.1 11-02 宕机回顾

- 第一次在 na20+na38 上以 Qwen3.8 线上配置测试时（当时直接用 `docker run`），na20 在
  13:31（北京时间）整机失联，需要从 BMC 重启。
- 系统日志 13:27:55 就停止写入，没有 OOM、panic、Xid 记录；vLLM 日志写到 13:31:19，
  当时 QSA 预热刚完成。pstore 和 `/var/crash` 都是空的。
- 10 月 18 日的日志里有 `NVRM: Assertion failed: pIOVAS != NULL` 和 IOVAS 残留映射。

### 5.2 当前状态（11-03 09:54 检查）

| 检查项 | 结果 |
|---|---|
| 连续运行时间 | 19 小时 40 分钟 |
| 本次启动以来的内核日志 | 只有 AER 初始化信息，没有 Xid、NVRM 报错、PCIe 错误、MCE、卡死或 OOM |
| GPU 不可纠正 ECC（当前 / 累计） | 4 张卡都是 0 |
| 显存行重映射 | 没有待重映射，也没有失败 |
| PCIe AER 错误计数（fatal / nonfatal） | 全部为 0 |
| 温度 | 31–32°C |
| 负载验证 | 宕机后以 Qwen3.8 线上配置跑完两轮完整测试，上次出事的预热阶段两次都正常通过 |

检查用的命令：

```bash
journalctl -k -b 0 | grep -iE "xid|nvrm.*(error|assert|fail)|aer|pcie bus error|mce|hardware error|soft lockup|hung_task|oom"
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,remapped_rows.pending,remapped_rows.failure,temperature.gpu --format=csv
cat /sys/bus/pci/devices/*/aer_dev_fatal /sys/bus/pci/devices/*/aer_dev_nonfatal
```

### 5.3 结论与建议

- 当前可以正常使用。宕机原因没有确证；宕机那次是直接 `docker run` 起的容器，后面两次成功的
  是 gpustack 起的容器，这一点不足以说明原因。
- 如果要查清，需要运维从 BMC 看 11-02 13:27–13:32 前后的硬件事件日志（SEL）。节点上没装
  ipmitool，无法在系统内查看。
- 如果以后再出现类似的整机卡死，按硬件问题处理，报修或打开 gpustack 维护模式。

---

## 6. 之后的注意事项

### 6.1 现网

- 三个现网模型的后端版本都不要改：
  - DSv4.1、Qwen3.8 用 `reputationly-backport-custom`；
  - M3.1 用 `reputationly-backport-m31-custom`。
- gpustack 推理后端的配置是整体保存的。**不要用 API 整体覆盖**；新增版本在控制台里加，
  名字必须以 `-custom` 结尾。
- 如果以后要把 Qwen3.8 或 DSv4.1 迁到基于新 base 的镜像：
  1. 用 master 出 app 镜像；
  2. 新增一个 `*-custom` 后端版本；
  3. 按总文档 8.4 的方法做新旧对比，DSv4.1 要先复核那约 3% 的吞吐差距；
  4. 逐个副本切换并观察。
- 生产节点上**只能通过 gpustack 做测试**，例如临时模型加临时标签。测完要做到：
  - 删掉模型、恢复标签；
  - 逐台确认没有残留容器、显存已释放；
  - 保留测试数据，但不要在节点上留下常驻进程。

### 6.2 出包

- `build-arm64-app.yml`：
  - `base_image` 默认就是新 base；
  - `base_commit` 填 `6bd0b07116` 时检查会报警，因为 `a163f78cb7` 改过 `docker/Dockerfile`
    和 `setup.py`，但只涉及构建参数和版本号，不影响编出来的 C++。下次重编 base 之前，可以把
    `base_commit` 留空跳过检查，前提是确认之后没人改过 `csrc`、`cmake` 和依赖；
  - `dest_tag` 默认会移动 `arm64-sm80-latest`，不想移动时填一个专用 tag。
- 重编 base 的完整步骤见总文档 8.2。要点：
  - 编译用 dev 机器，不用生产节点；
  - 依赖先在 Mac 上下好，构建期间不访问外网；
  - 一次只跑一个构建；
  - 只推带时间戳的固定 tag，浮动 tag 等回归通过后再切，切之前先备份旧 tag。
- 版本号只认 `v*` tag（`git describe --match "v[0-9]*"`）。不要再打 `v` 开头、但不是
  版本号的 tag。

### 6.3 代码

- `triton_mrope` 现在会**原地**修改传入的 q/k，并且要求输入是二维、最后一维连续的张量。
  以后新增调用方时：
  - 传入 `[num_tokens, num_heads * head_dim]` 形状；
  - 依赖原地修改或者接收返回值都可以；
  - 不要在外面先 `.contiguous()` 再丢掉结果。
- 视觉相关的改动，一律用文字密集的图片验证，问具体的单元格内容。
- Rust 前端在 master 上已经能编译，但现网都没有开启。要开启的话需要单独测试。

### 6.4 回归测试

- 参数照搬线上，从容器里实时读取，不要凭记忆写。
- 每种并发先单独预热，每项跑 3 遍。并发 16 的第一遍经常偏低。
- 吞吐差距只有在两种并发下都"同方向、超过 3%"才算退化；差距接近门槛时，交叉对调节点再测一轮。
- 测试期间从外部监控节点健康（ping、ssh、gpustack 心跳）。

### 6.5 待办

- [ ] DSv4.1 约 3% 吞吐差距的复核（不急；迁移 DSv4.1 之前必须做）。
- [ ] na20 的 BMC SEL 排查（可选）。
- [ ] 轮换本次沟通中出现过的 SWR 密码和 gpustack 管理 token。
- [ ] 如需让 M3.1 支持关闭思考，评估定制 chat template（见总文档 4.3）。
