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

### 5.7 改动清单

修改：

- `examples/ops/dispatch_combine/test_dispatch_combine_internode.py` —— tuning margin
- `python/mori/ops/tuning_config.py` —— 写回时的 no-regress margin
- `python/mori/ops/dispatch_combine.py` —— `_cached_view`

新增：

- `examples/ops/dispatch_combine/bench_matrix.py` —— 测量框架
- `docs/EP16-LL-SMALL-TOKEN.md` —— 本文档
