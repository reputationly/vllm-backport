# A100(sm80)ARM64 多节点部署实录

本文记录在**鲲鹏 ARM64 + 4×A100-PCIE-40GB 裸金属 + 云 RoCE** 上把
vllm-backport 从零跑起来的全过程,重点是那些只有真跑才会暴露、且下次还会
再撞一遍的坑。环境是隔离网(不通 Docker Hub / pypi.org / github https)。

验证结论:`DeepSeek-V4-Flash-0731` 在 **2 节点 × 4×A100-40G,TP4×PP2** 下
正常服务,KV cache 1,141,155 token,chat 输出正确。

---

## 1. 硬件与拓扑

| | |
|---|---|
| 单节点 | 2×Kunpeng 920 64C / 256GB / 4×A100-PCIE-40GB(`physical.kp1.2xlarge.h`) |
| 节点内互联 | PCIe,**无 P2P** |
| 节点间 | 云 RoCE,规格标称 `20GE + 4×100G` |
| 权重 | DSv4-Flash-0731 155.4 GiB(fp8 / e4m3 / ue8m0 / block `[128,128]`) |

**拓扑选择:TP=节点内卡数,PP=节点数。** 依据是 fork 自己的实测
(commit `3b890f43e`,4×sm80 64GB PCIe no-P2P):TP4 prefill 比 PP4 慢约
6.6 倍。PP 每个 stage 边界只过 hidden states,单条 100G 就够。

**单节点装不下任何 DSv4-Flash 变体**:最小的 `DeepSeek-V4-Flash` 148.6 GiB,
单节点裸容量 160 GiB(4×40960 MiB),占 93%,没有余量给 activation、KV、
CUDA graph 池(FULL capture 每卡可能 >800MB)和 NCCL buffer。所以 ≥2 节点。

---

## 2. RoCE:4 个口只有 2 个可用(未解决,需云厂商)

### 现象

四个 RoCE 口链路全部 up / 100Gb/s / 控制台"激活" / 同一个网络 ID 与子网 ID,
但只有每张双口卡的**第 0 口**(`np0`,即 `mlx5_0` / `mlx5_2`)能转发流量。

### 定位过程与排除项

| 假设 | 结论 |
|---|---|
| 四口同在一个 `/16`,内核只用最低 metric 的口 | 是事实但**不是主因**;NCCL 的 IB 传输走 ibverbs 直接绑 GID,数据路径不依赖内核路由表 |
| `rp_filter` 丢弃非对称应答 | **排除**:两台都是 `rp_filter=2`(loose),只要源地址任意网卡可达就不丢 |
| 安全组拦截 | **排除**:安全组只挂在云网卡(`NIC1: 10.0.0.x`),RoCE 网卡不在安全组管辖范围 |
| 云侧对 `np1` 整类不放行 | **排除**:gpu43 的 rail1 → gpu41 的 np1 ERI 实测 0% 丢包 |
| **特定 ERI 不转发** | **确认** |

### 决定性测试(自环)

```bash
# 从 np1 口 ping 本机 np0 口的 IP —— 目标在本机,排除全部主机侧与对端因素
ping -I enp194s0f1np1 -c 3 29.165.19.7
# 预期 0% 丢包;故障节点实测 100% 丢包
```

包从 `np1` 出去进了云 fabric 就再也没回来。这一条把路由 metric、rp_filter、
安全组、对端配置全部排除。

### 给云厂商的描述模板

> 裸金属 X 的以下弹性网卡控制台显示"激活",但无法发出任何流量,
> **连本机同子网的另一张 RoCE 网卡都 ping 不通**:<网卡 ID / IP>。
> 同机另两张正常:<网卡 ID / IP>。
> 对照:另一台裸金属的同类(`np1`)网卡可以正常收发。
> 主机侧已排除:链路 up/100Gb、IP 已配、metric 正常、
> `rp_filter=2`(loose)、RoCE 网卡不受安全组管辖。

### 另一个自己能修的问题

云侧给每台机器分了 4 个 RoCE 网卡,但**部分节点的 OS 只配了 2 个的 IP**
(控制台能看到 4 个 IP,主机 `ip addr` 只有 2 个)。补的时候必须在 netplan 里
**显式写 metric**:

```bash
# 反例:这样加的连接路由 metric=0,会抢走整个 29.165.0.0/16 的默认出口,
#       导致原本正常的 rail0/rail2 全部不通
ip addr add 29.165.123.0/16 dev enp194s0f1np1
```

这也正是多轨 RoCE 应当**每轨一个独立子网**的原因 —— 四个口挤在一个 `/16` 里,
路由表无法区分,只能靠 metric 这种脆弱机制。

### 当前可用配置

```bash
NCCL_IB_HCA=mlx5_0,mlx5_2      # 已验证双向可通的两条轨 = 200 Gb/s
NCCL_IB_GID_INDEX=3            # RoCE v2 over IPv4(该 GID 项是 IPv4 映射地址)
NCCL_SOCKET_IFNAME=enp131s0f0  # 带外引导走 10.0.0.x 管理网
NCCL_ALGO=Ring NCCL_PROTO=Simple   # 本 fork 的硬要求,见 README
```

修好后改成四个全列即可。对 TP4×PP2 而言 2 条轨已远远够用;4 轨的价值在
跨节点 TP、专家并行 all-to-all、以及 LMCache 的 KV 传输。

### 排查用的一组命令

```bash
# 设备 <-> 网卡 <-> IP 映射
for d in mlx5_0 mlx5_1 mlx5_2 mlx5_3; do
  pci=$(basename $(readlink -f /sys/class/infiniband/$d/device))
  nic=$(for n in /sys/class/net/*; do
          [ "$(basename $(readlink -f $n/device) 2>/dev/null)" = "$pci" ] && basename $n
        done)
  echo "$d $nic $(ip -4 -o addr show $nic | awk '{print $4}')"
done

# 端口状态与链路层(RoCE 应为 Ethernet)
cat /sys/class/infiniband/mlx5_0/ports/1/{state,link_layer,rate}

# RoCE v2 的 GID index
for i in 0 1 2 3; do
  echo "$i $(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$i)"
done

# 逐轨连通性(-I 绕开路由 metric 选择)
ping -I <本地网卡> -c 3 <对端同轨 IP>
```

> 提示:ARP 表里邻居 MAC 全是 `00:00:00:00:00:01` 是云 SDN 的伪 MAC,正常现象,
> 不代表故障。

---

## 3. 隔离网构建:9 个坑

镜像构建侧的全部改动见 `docker/Dockerfile.cn-bases`、`docker/build_base_aarch64.sh`
和 `docker/cn/wheelhouse/`,这里只列清单备查:

| # | 问题 | 解法 |
|---|---|---|
| 1 | Ubuntu 的 `docker.io` 包不带 buildx,而本仓 Dockerfile 通篇 `RUN --mount` 必须 BuildKit | `apt-get install -y docker-buildx` |
| 2 | `docker/Dockerfile` 默认 `BUILD_BASE_IMAGE` 是 **amd64 单架构** | 换上游 arm64 CI 的 pin(`pytorch/manylinuxaarch64-builder`) |
| 3 | `repo.almalinux.org` 不通(110 B/s) | 换阿里云,891 kB/s |
| 4 | EPEL metalink 不通(`ccache` 来自 EPEL) | 自带 `epel*` 全部 `enabled=0` + 另写阿里云源。只注掉 metalink 不够,`[epel]` 仍 enabled 却无可用 baseurl,dnf 报 `Cannot find a valid baseurl` |
| 5 | `bootstrap.pypa.io` 不通;阿里云那份 `get-pip.py` 是旧版,内嵌 pip 还 `import distutils`(3.12 已删);Ubuntu 又剥了 ensurepip 的 wheel | 放一个 `file:///opt/get-pip-cn.py` shim,内容是用 apt 装 `python3-pip` |
| 6 | `flashinfer-cubin` / `flashinfer-python` 的 pin 只在 GitHub Releases 上(cubin 1.5GB),阿里云镜像最高只到 `0.6.13` | 有外网的机器下载后放进 `docker/cn/wheelhouse/`,基座里设 `UV_FIND_LINKS` |
| 7 | `RUN python3 -m pip install uv` 是裸 pip,该阶段没有 `PIP_INDEX_URL` 这个 ARG | 在基座里设 `ENV PIP_INDEX_URL/UV_INDEX_URL`(ENV 优先于同名 ARG) |
| 8 | base 阶段与 vllm-base 阶段**并行**装依赖却共享同一个 uv cache mount,锁等待 300s 超时 | 分两趟(`BUILD_TARGET=base` 先跑)+ `ENV UV_LOCK_TIMEOUT=1800` |
| 9 | EP kernels(DeepEP)那步固定 `TORCH_CUDA_ARCH_LIST='9.0a 10.0a'`,对 A100 无用,却要 `git clone github.com/deepseek-ai/DeepEP` | 新增 `ARG INSTALL_EP_KERNELS`(默认 true 保持上游行为),sm80 传 false;并给三处消费 `dist/*.whl` 的地方加空目录兜底 |

### 传输经验

| 路径 | 实测 |
|---|---|
| 节点 → GitHub(git clone) | ~60 KB/s(不可用) |
| Mac → 节点(scp) | ~10 MB/s |
| **节点 → Mac(scp)** | **~43 MB/s** |
| 节点 ↔ 节点(`docker save \| ssh docker load`,23GB) | ~4.5 分钟 |
| ModelScope → 管理节点 | ~46 MB/s |

结论:**镜像与大文件在节点间直传,不要让节点去外网拉**。管理节点
(`10.0.0.238`)能通 ModelScope、hf-mirror 和国内 Docker Hub 镜像源,
适合当中转;GPU 节点只通阿里云 PyPI、`download.pytorch.org`、`flashinfer.ai`、
ACR、华为 SWR、`developer.download.nvidia.cn`。

---

## 4. 模型格式:NFS 上哪些 vLLM 读不了

带 `-w8a8` / `-w4a8` 后缀的检查点是**昇腾 ModelSlim**
(`apiversion: modelslim_v1`,quarot + flex_smooth_quant),给 MindIE 用的,
vLLM 一份都读不了。判别方法:目录里有 `quant_model_description.json`
且 `config.json` 的 `quantization_config` 为 `None`。

DSv4-Flash 三个 vLLM 格式变体的区别:

| | 量化 | index total_size |
|---|---|---|
| `DeepSeek-V4-Flash` | `fp8` / e4m3 / ue8m0 / block `[128,128]` | 148.6 GiB |
| `DeepSeek-V4-Flash-0731` | 同上(后来的快照) | 155.4 GiB |
| `DeepSeek-V4-Flash-NVFP4` | `MIXED_PRECISION`:FP8 底座 + MoE experts NVFP4(modelopt) | 156.7 GiB |

**sm80 上别选 NVFP4**:FP4 是 Blackwell 原生格式,A100 没有 FP4 tensor core;
而且 `group_size=16` 的 scale 开销把 4-bit 省下的量吃光了还更大,一点好处没有。
`0731` 的 fp8 block-quant 正好走 fork 里 `fp8_utils.py`(E8M0 upcast)和
Marlin 覆盖的主路。

---

## 5. 起服务

镜像里**不带 ray**,而跨节点 TP/PP 需要它(`distributed_executor_backend`
只有 `ray` 和 `mp`,`mp` 不跨节点)。版本按 `requirements/test/cuda.txt` 的 pin:

```bash
pip install "ray[cgraph,default]==2.56.1"
```

容器与启动见 `docker/run_multinode_sm80.sh`。容器侧 RDMA 必需项:
`--network host`、`--device /dev/infiniband`、`--ulimit memlock=-1`,
另配 `--ipc host` 与足够大的 shm。

启动成功时的关键日志:

```
Worker_PP0_TP0 / Worker_PP1_TP0        <- TP4 x PP2 分片生效
Available KV cache memory: 12.0~12.9 GiB   (每卡)
GPU KV cache size: 1,141,155 tokens
Maximum concurrency for 131,072 tokens per request: 8.71x
Application startup complete.
```

### 验证时的一个陷阱

**不要用 `/v1/completions` 裸补全来判断模型是否正常。** DSv4 是 chat 模型,
不套 chat template 时会输出训练数据残影(实测吐出过带 `"tokenizer":
"Qwen/..."` 字样的 JSON 片段),很容易被误判成"模型坏了"。用
`/v1/chat/completions`:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash",
       "messages":[{"role":"user","content":"用一句话解释什么是张量并行。"}],
       "max_tokens":200,"temperature":0}'
```

另外 0731 的 thinking 默认开启(见 README 的 0731 契约),不加
`--reasoning-parser` 时思考内容会内联在 `content` 里带着 `</think>`,
不会被拆到 `reasoning_content`。
