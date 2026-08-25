# EP16 v1_ll 小 token 场景测试基线（MI300X × 2 节点）

本文档记录 EP16 `InterNodeV1LL` dispatch/combine 在**小 token（4 / 8 / 16）**场景下的
测试方法、固定命令和当前最佳性能。所有数字都钉在同一套机器和同一组参数上，保证跨次可比。

---

## 1. 机器 —— 不要换

这里的性能对节点组合很敏感（NIC 拓扑、机架位置都会影响）。**必须固定用下面这两台**，
换机器会导致数字有 gap，对比就没意义了。

| | node_rank 0（master） | node_rank 1 |
|---|---|---|
| 主机名 | `useocpm2m-097-032` | `useocpm2m-097-086` |
| `eth0`（bootstrap 用） | `10.158.214.156/22` | `10.158.214.85/22` |
| GPU | 8 × AMD Instinct MI300X（gfx942，304 CU） | 同左 |
| RDMA 网卡 | `mlx5_0/2/3/4/5/7/8/9` → `rdma0..7` | 同左 |
| `mlx5_1` / `mlx5_6` | 挂在 `eth0` / `eth1` 上，**不是** fabric 网卡 | 同左 |

`EP16 = 2 节点 × 8 GPU`。两台共享同一个 NFS `/shared_inference`，所以代码和日志是共用的，
容器则是每台一个。

### 容器（两台完全一致）

```bash
docker run -d --name yutong-mori-ep16 \
  --network host --ipc host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined \
  --group-add video --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared_inference:/shared_inference \
  -w /shared_inference/yutongwu/store/dev/mori \
  lmsysorg/sglang-rocm:v0.5.16-rocm720-mi30x-20260730 \
  sleep infinity
```

容器内软件栈：

| | |
|---|---|
| 镜像 | `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi30x-20260730`（官方） |
| HIP | 7.2.26015 |
| torch | 2.9.1+rocm7.2.0 |
| mori | 源码安装，`pip install --no-build-isolation .` |
| 额外 | `pip install prettytable`（镜像里没有，缺了会在 tuning 打印结果表时崩） |

---

## 2. 固定场景

**`--hidden-dim 6144` 是我们要优化的目标场景，不是可调参数。** 它是我们关心的模型维度，
不要为了让数字好看去改它 —— 换一个 hidden dim 就是另一个问题了。

每次跑都固定：

```
--kernel-type v1_ll --num-qp 2 --hidden-dim 6144
--dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none
EP16（nnodes=2, GPU_PER_NODE=8），topk=8，每 rank 16 experts
```

只有 `--max-tokens` 在 **4 / 8 / 16** 之间变。

---

## 3. 命令

以下命令都在**两台机器的容器内**执行，`$R` 在 `useocpm2m-097-032` 上填 `0`，
在 `useocpm2m-097-086` 上填 `1`。**先起 node 1，再起 node 0。**

`IFNAME=eth0`，`master_addr=10.158.214.156`（032 的 eth0）。

### 3.1 Tuning

```bash
MORI_RDMA_TC=41 GPU_PER_NODE=8 MORI_TUNING_SCOPE=full \
GLOO_SOCKET_IFNAME=eth0 MORI_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --node_rank=$R --nproc_per_node=1 \
  --master_addr=10.158.214.156 --master_port=1234 \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --cmd tuning --kernel-type v1_ll --num-qp 2 \
  --max-tokens 4 --hidden-dim 6144 \
  --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none \
  --save-tuning-config auto
```

`auto` 会写到 `python/mori/ops/tuning_configs/gfx942_mi300x_InterNodeV1LL_ep16_{dispatch,combine}.json`。

### 3.2 Bench（指定固定 launch 配置）

```bash
MORI_RDMA_TC=41 GPU_PER_NODE=8 \
GLOO_SOCKET_IFNAME=eth0 MORI_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --node_rank=$R --nproc_per_node=1 \
  --master_addr=10.158.214.156 --master_port=1234 \
  examples/ops/dispatch_combine/test_dispatch_combine_internode.py \
  --cmd bench --kernel-type v1_ll --num-qp 2 \
  --max-tokens 4 --hidden-dim 6144 \
  --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none \
  --block-num 32 --warp-per-block 8 --rdma-block-num 16
```

> `--cmd bench` 只有**一组** block/warp 覆盖参数，dispatch 和 combine 共用。两者最优配置
> 不同，需要分别指定时用 `bench_matrix.py`（§3.5）。

### 3.3 正确性

```bash
... --cmd test --kernel-type v1_ll --num-qp 2 \
    --max-tokens 4 --hidden-dim 6144 \
    --dtype fp8_e4m3_fnuz --combine-dtype bf16 --quant-type none
```

16 个 rank 应全部输出 `error times: 0`。

### 3.4 DeepEP 风格 bench（`test_low_latency.py`）

对齐 DeepEP `tests/test_low_latency.py` 的测试方式。参数写死在脚本
`test_loop()` 里，本文档所有 DeepEP 数据都用
**`num_tokens=4, hidden=7168, topk=8, num_experts=288`**：

```python
# examples/ops/dispatch_combine/test_low_latency.py:660
num_tokens, hidden, num_topk, num_experts = 4, 7168, 8, 288
```

> 仓库里这一行的默认值是 `128`，要测小 token 场景需要手工改成 `4`（或 8 / 16）。
> 注意 hidden 是 **7168** 不是 6144 —— 这组是 DeepEP 的低延迟档，用于横向对比，
> 和 §2 的 6144 场景是两回事。

多节点时脚本自动走 `InterNodeV1LL`，`block=64, warp=8, rdma_block=32`。

```bash
MORI_RDMA_TC=41 MORI_NUM_QP_PER_PE=1 \
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 MORI_SOCKET_IFNAME=eth0 \
WORLD_SIZE=2 RANK=$R MASTER_ADDR=10.158.214.156 MASTER_PORT=8361 \
PYTHONPATH=$(pwd) python examples/ops/dispatch_combine/test_low_latency.py
```

> **多机必须加 `MORI_RDMA_TC=41`。**
> 原始命令里的 `NCCL_SOCKET_IFNAME=bond0` / `MASTER_ADDR=skyriver07` 在我们这套机器上
> 分别替换成 `eth0` / `10.158.214.156`。脚本自己 spawn 8 个进程，不需要 torchrun。

这个 bench 会打印**两组**指标，含义完全不同，看的时候别混：

| 输出行 | 测的是什么 | 用 |
|---|---|---|
| `Dispatch + combine bandwidth: ... avg_t=` | `bench()`：cuda event 包住 50 次连续调用，**包含 host 提交开销**，端到端延迟 | 看真实延迟 |
| `Dispatch bandwidth: ... / Combine bandwidth: ...` | `bench_kineto()`：从 CUDA profile 里累加 `Ep*` kernel 时长，**纯 GPU kernel 时间** | 看 kernel 本身 |

`bench_kineto` 里有一句注释写得很明白 —— 它特意用大 kernel + barrier 来
"eliminate the unbalanced CPU launch overhead"。也就是说**它在设计上就看不见 host 侧开销**，
这正是 §5.3 那个问题长期没被发现的原因。

### 3.5 `bench_matrix.py` —— 多次重复测量框架

`examples/ops/dispatch_combine/bench_matrix.py`。标配的 `--cmd bench` 只给一个 10 轮均值，
在这个量级上区分不出真实效果和噪声。这个框架跑 N 次独立 trial，报**最慢 rank 的中位数 +
离散度**，支持 **dispatch 和 combine 用不同配置**，并且会报**host 侧 enqueue 耗时**。

```bash
M='[{"name":"tuned","disp":[32,8,16],"comb":[64,4,32]}]'   # [block_num, warp_per_block, rdma_block_num]
MORI_RDMA_TC=41 GPU_PER_NODE=8 \
GLOO_SOCKET_IFNAME=eth0 MORI_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --node_rank=$R --nproc_per_node=1 \
  --master_addr=10.158.214.156 --master_port=1234 \
  examples/ops/dispatch_combine/bench_matrix.py --matrix "$M" \
  --trials 7 --rounds 10 --max-tokens 4 --hidden-dim 6144
```

其他有用的 flag：`--profile-kernels`（逐 kernel 计时）、`--cpu-profile`（cProfile host
提交循环）、`--barrier-per-round`、`--no-batch-launch`。

辅助脚本都在 `/shared_inference/yutongwu/store/ep16_tuning/`：
`run_matrix.sh`、`sweep.sh`、`verify.sh`、`tune_full.sh`、`run_deepep.sh`、
`cleanup.sh`、`deploy_kernel.sh`。

### 3.6 测量注意事项（踩过的坑）

1. **一定要先 warm up。** JIT 缓存被清掉之后、或者机器闲置一段时间之后的第一次跑，会
   **虚高 40–80%**（JIT 编译 + GPU 降频未恢复）。对比之前先丢弃一轮。这个坑让我两次
   A/B 得出错误结论。
2. **每次跑之间清残留进程**（`cleanup.sh`）。任一节点上残留的 `torchrun` 会占住 rendezvous，
   下一次跑会卡在 `dist.barrier()`。
3. **run-to-run 离散度有 10–20%。** 小于这个幅度的差异不算结果。用 `--trials 7` 比中位数。
4. **A/B 必须在同一 session 内做。** 跨 session 数字会漂，不要和几小时前的表格比。

---

## 4. 当前最佳配置与性能

用修复后的 tuner（`MORI_TUNING_SCOPE=full`，75 组配置）在 §1 两台机器上、
`hidden_dim=6144` 下重新调优。dispatch 和 combine 最优配置**不同**，所以分阶段给出：

| max-tokens | dispatch（block/warp/rdma） | combine（block/warp/rdma） |
|---|---|---|
| 4 | `32 / 8 / 16` | `64 / 4 / 32` |
| 8 | `32 / 6 / 16` | `32 / 4 / 8` |
| 16 | `32 / 8 / 16` | `32 / 4 / 16` |

该配置下实测延迟（`bench_matrix.py --trials 7 --rounds 10`，取 trial 中位数、最慢 rank）：

| max-tokens | dispatch | combine | dispatch+combine | host enqueue/轮 |
|---|---|---|---|---|
| 4 | **76.0 us** | **93.2 us** | 169 us | 156 us |
| 8 | **79.9 us** | **88.2 us** | 168 us | 150 us |
| 16 | **75.7 us** | **93.9 us** | 170 us | 152 us |

DeepEP 风格 bench（§3.4，`num_tokens=4, hidden=7168, topk=8, experts=288`）：

| 指标 | 数值 |
|---|---|
| 端到端 `bench()` avg_t | **156.8 us** |
| kineto dispatch kernel | 70.4 us |
| kineto combine kernel | 113.7 us |

几点说明：

- 延迟在 **4 / 8 / 16 token 之间基本持平** —— 这个区间是 overhead-bound，不是 data-bound
  （见 §5.2）。
- host enqueue（~150 us/轮）已经很接近端到端 wall time（~170 us/轮），
  **节奏是 host 定的，不是 GPU。**
- 分阶段调优只在 16 token 有实质收益（dispatch 96.6 → 75.7 us，−22%）；4 / 8 token 下和
  旧的 `32/4/8` 打平，因为 launch 配置根本不是耗时所在。

要让 op 真正用上这些 JSON，需要设 `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`
（默认 `MANUAL`，完全不读 JSON —— 见 §5.5）。

---

## 5. 2026-08-25 —— margin 修复与小 token 瓶颈定位

### 5.1 Tuning margin 修复

Tuner 对任何 shape 都原样返回第 `[1/75]` 组候选。根因：`_BW_NOISE_MARGIN = 1.0` 是**绝对**
GB/s 阈值，而 4 token 下整个扫描区间只有 2.0–3.2 GB/s，`bw > best + 1.0` 永远不可能成立。
dispatch 属于蒙对（第 1 组本来就接近它的最优），combine 想要的是配置空间的另一端，损失约 13%。

四处改动：

| 改动 | 文件 |
|---|---|
| `_BW_NOISE_MARGIN=1.0`（绝对）→ `_BW_REL_MARGIN=0.02`（相对） | `examples/ops/dispatch_combine/test_dispatch_combine_internode.py` |
| 每配置 9 轮，用**中位数**而非均值评分（`_TUNING_ROUNDS`） | 同上 |
| margin 内确定性 tie-break —— block 少的胜出（`_beats()`） | 同上 |
| 重新调优需超过已有 rule 2% 才覆盖（`_SAVE_REL_MARGIN`） | `python/mori/ops/tuning_config.py` |

最后一条针对 JSON 频繁变动：仅靠噪声跑高一点的重新调优，不会再改写已提交的 JSON。

验证有效：tuner 现在会给 dispatch 和 combine 选**不同的配置**（§4），这在修复前结构上
就不可能发生。

### 5.2 时间到底花在哪

实现并验证正确性的 4 个 kernel 级优化，**实测全部在噪声内**。patch 保留在
`/shared_inference/yutongwu/store/ep16_tuning/kernel_changes.patch`，均未合入。

1. `MultiWarpIter` 切分下限 —— 4 token 时 304-block grid 得到 `warpsPerItem=304`，
   每个 warp 只分到 21 个元素（1216 个 warp 去搬 48 KB）。
2. 跳过 combine 的 padding slot —— 每 chunk 256 次 `atomicAdd` 打同一地址，只有 16 次有用。
3. v1_ll poll 循环加 `s_sleep` 退避（`intranode.hpp` 里已有同样的先例）。
4. dispatch send barrier 只统计有工作的 block（4 token 时 456 warp → 1）。

第 3 条把 `CombineInterNodeV1KernelLowLatency` 从 90 → 67 us，但 `EpCombineSyncBarrier`
从 57 → 79 us：**barrier 流水线里等待只会挪位置，总量守恒。**

三个测量定位了真正的瓶颈：

| 测试 | 结果 | 结论 |
|---|---|---|
| `hidden_dim` 1024 vs 6144（6 倍数据量） | 93→85 us / 147→145 us，没变 | 不是 data-bound |
| grid 304 → 24 block | 12.13 → 12.05 us | 不是 grid-bound |
| host enqueue vs wall | 每轮 211 us vs 222 us | **host-bound** |

同样地，延迟在 4/8/16 token 间持平，`--num-qp` 取 1/2/4 也没有区别。

### 5.3 真正有效的优化 —— 缓存 GPU 指针视图

对提交循环做 `cProfile`，发现 `from_gpu_ptr` / `_torch_from_ptr` 每轮被调用 6 次
（约 60 us）。它把**固定的** shmem 缓冲区按**固定的** shape 包成 torch tensor，却每次
都重建一遍，每次都要走 `__cuda_array_interface__` 并查询 `torch.cuda.current_device()`。

改为 memoize：`python/mori/ops/dispatch_combine.py` 里的
`EpDispatchCombineOp._cached_view()`。设 `MORI_VIEW_CACHE=0` 可关闭（用于 A/B）。

同 session A/B，各 5 trial，两阶段都用 `32/4/8`：

| tokens | 指标 | 关闭 | 开启 | 变化 |
|---|---|---|---|---|
| 4 | combine | 150.1 us | 88.3 us | **−41%** |
| 4 | dispatch | 84.9 us | 72.7 us | −14% |
| 4 | wall/轮 | 222.0 us | 164.2 us | −26% |
| 4 | host enqueue/轮 | 210.6 us | 146.1 us | −31% |
| 8 | combine | 146.5 us | 85.9 us | **−41%** |
| 8 | wall/轮 | 229.1 us | 162.1 us | −29% |
| 16 | combine | 139.7 us | 93.2 us | **−33%** |
| 16 | wall/轮 | 232.4 us | 184.2 us | −21% |

把 4 个 kernel 改动全部回退、只留这一个改动，复现出同样的数字（combine 91.2 / 96.7 us），
**说明收益 100% 来自 host 侧。**

正确性：`--cmd test` 在 4 和 16 token 下各 500 轮，16 个 rank 全部 0 error。

另外试过并已回退（无可测量收益）：给 `mp` 大小的 launch 按工作量裁剪 grid；
给 `hipModuleLaunchKernel` 声明 `argtypes` 以省掉每次 launch 的 `c_uint` 构造。

### 5.4 用 DeepEP 风格 bench 交叉验证

用 §3.4 的 `test_low_latency.py` 独立复验，各跑 2 次取中位数：

| 指标 | `MORI_VIEW_CACHE=0` | `MORI_VIEW_CACHE=1` | 变化 |
|---|---|---|---|
| 端到端 `bench()` avg_t | 216.3 / 212.1 → **214.2 us** | 156.3 / 157.3 → **156.8 us** | **−27%** |
| kineto dispatch kernel | 80.0 / 72.6 → 76.3 us | 76.8 / 64.0 → 70.4 us | −8%（噪声内） |
| kineto combine kernel | 130.9 / 108.8 → 119.9 us | 120.7 / 106.7 → 113.7 us | −5%（噪声内） |

这组结果同时确认了三件事：

1. 优化在另一套测试方式下同样成立，端到端 **−27%**，和 §5.3 的量级一致。
2. **纯 GPU kernel 时间没变** —— 再次证明收益全在 host 侧，不是 kernel 变快了。
3. **`bench_kineto` 这类指标结构上看不到这个问题。** 它设计上就是要排除 CPU launch
   开销，所以只盯 kernel 带宽的话，这 27% 永远不会暴露。**评估小 token 性能时必须看
   端到端指标。**

### 5.5 Tuning config 加载的 4 个坑

op **确实**是分阶段读 JSON 的（`get_launch_config(is_dispatch=...)`），但实际大概率不生效：

1. `MORI_EP_LAUNCH_CONFIG_MODE` 默认 `MANUAL`，只有 `AUTO` 才加载 JSON。
2. tuning 写到 `<repo>/python/mori/ops/tuning_configs/`，而 op 读的是**安装后的**
   `site-packages/.../tuning_configs/`。非 editable 安装时这是两个目录。
3. miss 时 `_find_fallback_config` 会静默换用其他型号的配置 —— 实测出现过在 MI300X 上
   跑 **mi308x** 的配置，没有任何警告。
4. combine 查表用的是 `self.config.data_type`（**dispatch** 的 dtype），而 tuning 存
   combine rule 用的是 **combine** 的 dtype。`--dtype fp8 --combine-dtype bf16` 组合下
   combine 永远 miss。

另外 `MORI_EP_TUNING_CONFIG` 忽略 phase，一个路径没法同时覆盖 dispatch 和 combine。

以上 4 个坑本次都**没有**修。

### 5.6 后续方向

剩余 host 开销约 150 us/轮，已经没有单点热点了 —— 摊在 `launch_multi` 的 ctypes FFI
（~40 us）、`torch.cuda.current_stream()`（~20 us）以及一堆几微秒的调用上。

1. **HIP graph capture** 整个 dispatch+combine 序列。剩下最大的杠杆：把每轮 6 次 kernel
   launch 换成一次 graph replay，应该能吃掉大部分 150 us。需要处理 args 缓冲区稳定性和
   replay 语义。
2. 把 `EpCombineSyncBarrier`（grid=1、单 warp）用 last-block-arrives 模式并进
   `EpCombineSync` 尾部，6 次 launch 省 1 次。
3. 修 §5.5 的 4 个配置加载问题。

### 5.7 逐 kernel 详细分析（GPU 侧）

前面几节的结论都基于 wall time，而 wall time 在这个区间被 host 节流，看不清 GPU 侧。
`bench_matrix.py --kineto` 直接从 CUDA profiler 读 kernel 时长，不插 event、不破坏 launch
批处理，**这是唯一能评价 kernel 改动的指标**。

4 tokens / hidden 6144 / EP16，20 轮，跨 rank 取平均：

| kernel | grid | mean (us) | 占比 | 性质 |
|---|---|---|---|---|
| `EpDispatchCopyToStaging` | mp | 4.74 | 3.5% | 与 payload 无关，基本是 launch 底噪 |
| **`EpDispatchInterNodeV1KernelLowLatency`** | bn | **55.5** | **41%** | ~50us 固定 + 0.5us/token |
| `EpCombineSync` | mp | 6.8 | 5% | 随 payload 线性（1K→16K：4.7→13.8） |
| `EpCombineSyncBarrier` | 1 | 7.7 | 5.7% | node 内 8-GPU barrier |
| **`EpCombineInterNodeV1KernelLowLatency`** | bn | **56.3** | **41%** | ~46us 固定 + 0.48us/token |
| `EpCombineAll` | mp | 4.96 | 3.6% | 与 payload 无关，launch 底噪 |
| **合计** | | **136.1** | | |

#### 固定成本的确认

两个 LL kernel 占 **82%** 的 GPU 时间，其中绝大部分是与工作量无关的固定成本。三组扫描：

| 变量 | 范围 | dispatch LL | combine LL |
|---|---|---|---|
| hidden_dim（4 tokens） | 1024 / 6144 / 16384 | 55.1 / 53.4 / 53.4 —— **完全不变** | 54.8 / 66.5 / 79.8 |
| max-tokens（hidden 6144） | 4 / 8 / 16 | 52.2 ~ 57.2 —— **不变** | 56.4 ~ 66.5 |
| max-tokens 拉大 | 4 / 64 / 128 / 256 | 51.7 / 82.8 / 124.0 / 180.1 | 58.2 / 76.7 / 107.8 / 169.4 |

对第三行线性拟合：**dispatch LL ≈ 50us + 0.5us/token**，**combine LL ≈ 46us + 0.48us/token**。
4 token 时 98% 的开销是那个固定截距。dispatch 连 payload 都完全不敏感（16 倍数据量零变化），
说明它的 XGMI 拷贝全部被等待掩盖了。

#### 这个固定成本合理吗？—— 不合理

用 `ib_write_lat` 在同一对网卡（mlx5_0，`10.224.2.156` ↔ `10.224.2.85`）上测裸 RDMA：

| 消息大小 | t_typical |
|---|---|
| 8 B | 8.1 us |
| 4 KB | 9.9 us |
| 24 KB（≈4 token × 6144 fp8） | **10.7 us** |

线路只要 ~11us，mori 每个 phase 花 46–50us，**每个 phase 有约 40us 高于线路的软件开销，
两个 phase 合计约 80us，占 GPU 总时间的 59%。**

而且这不是硬件下限：同一 fabric 上 mori 自己的 async_ll，其接收侧 kernel
（`EpDispatchLowLatencyAsyncRecvTransfer`）最快能到 **18.6us**，
`EpCombineLowLatencyAsyncRecvTransfer` 到 **16.6us**。也就是说接收路径本身可以做到接近线路，
v1_ll 的 50us 是协议/实现开销。

（注意 async_ll 整体并不可用：端到端 1387us，比 v1_ll 慢 8.5 倍，它的 SendTransfer 是常驻
 kernel，只有 RecvTransfer 的下限有参考价值。）

#### 协议横向对比（端到端 wall/round，4 tokens）

| kernel-type | dispatch | combine | wall/round |
|---|---|---|---|
| **`v1_ll`** | 72.8 / 73.1 | 93.3 / 92.2 | **161.2 / 166.6** |
| `v1` | 121.7 / 123.0 | 123.1 / 166.3 | 222.3 / 272.6 |
| `async_ll` | 500.6 / 502.7 | 884.6 / 883.8 | 1386.7 / 1386.9 |

**`v1_ll` 是当前正确选择**，比 v1 快 27–39%，async_ll 在这个配置下完全不适用。

#### 可优化点（按收益排序）

1. **两个 LL kernel 的 ~96us 固定成本**（GPU 时间的 71%）。最大的一块，且已证明不是 fabric
   限制。方向是 GPU 发起 RDMA 的 WQE 构建/doorbell 开销、轮询检测延迟、以及
   send→signal→poll 这条协议链路。**需要协议层改动，参数调优已经榨干了。**
2. **合并 kernel**：6 个 kernel，每个 launch 底噪约 5us，合计 20–25us（GPU 侧 15–18%）。
   `EpCombineSyncBarrier`（grid=1、单 warp）可用 last-block-arrives 并进 `EpCombineSync` 尾部；
   `EpDispatchCopyToStaging` 可并进 dispatch LL 头部。
3. **host 侧**：cpu-enqueue ~150us 仍然 ≥ GPU 的 136us，**端到端依旧是 host 定的节奏**。
   即使把上面两条全做完，不解决 host 侧也看不到端到端收益。HIP graph 仍是最大杠杆。

#### 已用 kineto 复测确认无效的改动

| 尝试 | 结果 |
|---|---|
| §5.2 那 4 个 kernel 改动 | stock 136.0 / 136.2 vs 打补丁 135.5 / 135.5 —— 无差异 |
| `--num-qp` 1 / 2 / 3 / 4 | wall 161–169us，全在噪声内（2 微弱最优，维持不变） |
| `MORI_RDMA_DEVICES`（显式 8 张 fabric 卡 / `^mlx5_1,mlx5_6`） | 与默认完全一致 |

`MORI_RDMA_DEVICES` 无效的原因是**默认选卡本来就是对的**。`MORI_APP_LOG_LEVEL=info` 可见
`TopoSystem::MatchGpuAndNic()` 按 PCIe 拓扑给出：

```
rank 0 → mlx5_0   rank 2 → mlx5_3   rank 4 → mlx5_5   rank 6 → mlx5_8
rank 1 → mlx5_2   rank 3 → mlx5_4   rank 5 → mlx5_7   rank 7 → mlx5_9
```

8 个 rank 各自拿到对应 fabric 网卡，以太网的 `mlx5_1` / `mlx5_6` 已自动跳过，没有改进空间。

> 另外澄清：`MORI_NUM_QP_PER_PE`（transport 层 QP 数，默认 4）和 `--num-qp`
> （`config.numQpPerPe`，kernel 选 qpId 用）是两个不同的东西。DeepEP 那条命令里
> env=1 而脚本内部 config 写死 4，其实是不匹配的。

### 5.8 HIP graph：把 host 从关键路径上拿掉（有效）

§5.7 第 3 条指出即使 GPU 侧优化完，端到端仍被 host 的 ~150us/轮卡住。用
`torch.cuda.CUDAGraph` 把一轮 dispatch+combine 捕获成图再 replay，`bench_matrix.py --cuda-graph`：

| 指标 | eager（3 次） | HIP graph（3 次） | 变化 |
|---|---|---|---|
| cpu-enqueue/轮 | 152.7 / 164.5 / 149.2 | **29.0 / 28.4 / 28.6** | **−82%** |
| wall/轮 | 166.7 / 168.7 / 159.8 | 156.9 / 157.0 / 174.6 | −6% |

**正确性：graph replay 与 eager 结果 16/16 rank 逐位相同（max abs diff = 0）。**

wall 只降 6%，是因为 GPU 侧（136us）本来就贴着 host（150us），拿掉 host 后 GPU 立刻成为新的
下限。但这一步的意义不在这 6%：

> **捕获后 cpu-enqueue 29us ≪ wall 157us，瓶颈第一次完全落在 GPU 上。
> 也就是说，从现在起 kernel 层面的任何优化才会真正反映到端到端。**

在此之前所有 kernel 改动看不到收益，正是因为省下的 GPU 时间被 host 吸收掉了。

注意 view 缓存（§5.3）是 graph 捕获能成功的前提之一：捕获期间每轮重建 tensor 视图会引入
无法录制的 host 逻辑。

### 5.9 LL kernel 那 ~50us 到底花在哪（消融实验）

对 `EpDispatchInterNodeV1KernelLowLatency` 逐层砍掉功能，用 kineto 量（**这些改动会破坏
正确性，只用于测量，已全部回退**）：

| 变体 | dispatch LL mean | fastest | 该层成本 |
|---|---|---|---|
| 基线 | 55.5 us | 44.7 | — |
| 去掉 recv 轮询等待 | 50.5 us | 41.5 | **等待 ≈ 5 us** |
| 再去掉 RDMA put | 37.1 us | 34.9 | **put 发起 ≈ 13 us** |
| 剩余 | **37.1 us** | 34.9 | **??? ≈ 32 us** |

（对照：`EpDispatchCopyToStaging` 这种简单 kernel 是 4.7us，可视为空 kernel 底噪。）

**这个结果推翻了"LL kernel 在等对端"的直觉：**

1. **等待只占 5us。** 把轮询循环整个换成一次非阻塞探测，kernel 时间几乎不变。这也解释了
   为什么 §5.2 的 `s_sleep` 退避完全无效——轮询根本不是瓶颈。
2. **RDMA put 发起占 13us。** 对比裸 RDMA 全程 10.7us，GPU 侧发 WQE + doorbell 的代价确实
   偏高，但不是主要矛盾。
3. **剩下 ~32us 既不是等待也不是发送，也不是数据搬运**（dispatch 对 payload 完全不敏感）。
   这是最大的一块，且目前**没有定位到**。嫌疑：kernel ramp（这个 kernel 体积远大于
   CopyToStaging，i-cache 压力）、grid barrier 的竞争原子、recv 侧的记账原子
   （`destPeTokenCounter`）。

想进一步切分需要"kernel 开头直接 return"的消融，但那会让 combine 侧永久等待而死锁，
这条路走不通。**下一步应该用 mori 内置的 `MORI_TRACE_SPAN` span profiler**（需要带
`ENABLE_PROFILER` 重新编译 mori，当前安装的 pybind 没有该支持：
`hasattr(mori.cpp, "get_debug_time_buf") == False`）。

### 5.10 双峰抖动（未解决）

约 **1/3** 的 run 会整体慢 1.7 倍，干净双峰、没有中间值：

| | wall/round | combine |
|---|---|---|
| 好模式（~2/3） | ~165 us | ~88 us |
| 坏模式（~1/3） | ~275 us | ~170 us |

两个 bench 都会出现。已排除：`--num-qp`、RDMA 设备选择、其他任务抢 GPU（两台均空闲，
无第三方进程）。关键线索是坏模式下 **cpu-enqueue 仍是 ~150us 但 wall 到 275us**，
即 GPU 侧真的变慢了（好模式是 wall≈enqueue，host-bound）。怀疑 GPU 没升频
（`rocm-smi` 空闲 131MHz 并警告 `AMD GPU device(s) is/are in a low-power state`），
但本机 `rocm-smi --setperflevel high` 返回 `Not supported on the given system`，无法直接验证。

**实践影响：任何对比都要跑 ≥3 次取好模式，单次结果不可信。**

### 5.11 改动清单

修改：

- `examples/ops/dispatch_combine/test_dispatch_combine_internode.py` —— tuning margin
- `python/mori/ops/tuning_config.py` —— 写回时的 no-regress margin
- `python/mori/ops/dispatch_combine.py` —— `_cached_view`

新增：

- `examples/ops/dispatch_combine/bench_matrix.py` —— 测量框架
- `docs/EP16-LL-SMALL-TOKEN.md` —— 本文档
