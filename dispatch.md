# InterNode V1 Dispatch —— 同步点清单、gfx950 ISA 实测、剩余优化点

清点自 `src/ops/dispatch_combine/internode_v1.cpp`（dispatch 段 45–701 行）
与 `src/ops/dispatch_combine/launch.cpp`。表中「行」为 `internode_v1.cpp` 行号。
**只覆盖 `KernelType::InterNodeV1`**，不含 `InterNodeV1LL`。

文档最初只清点同步点；实测下来同步层面几乎没有可回收量，
**剩下最大的一块是 launch geometry**，所以后半部分扩到了这个范围。
先看下一节的优先级排序，再按需往下读。

**口径**

- 「站点数」按**代码位置**计，不按动态执行次数；一行里的 `__shfl ×2` 记作一个站点。
- 未包含 `ENABLE_PROFILER` 下才编译的 `MORI_TRACE_SPAN` / profiler 桩。
- 符号：`B`=blockNum, `R`=rdmaBlockNum, `X`=B−R, `W`=warpNumPerBlock,
  `N`=nNodes, `G`=gpuPerNode, `T`=curRankNumToken, `K`=numExpertPerToken。

**ISA 实测环境**

- GPU：**gfx950**（MI350 系列，8 卡 / 每卡 256 CU / 8 个 XCD），
  由 `/sys/class/kfd/kfd/topology/nodes/*/properties` 的 `gfx_target_version=90500` 确认。
- NIC：ionic（8 个 ionic + 2 个 bnxt，检测逻辑取多数）→ `-DMORI_DEVICE_NIC_IONIC`。
- 工具链：`/shared/apps/ubuntu/opt/rocm-10.0.0`（HIP 7.15，AMD clang 23.0.0git）。
- 复现命令见文末「复现」一节。所有 ISA 数字均为**静态指令数**，非运行时开销。

---

## 剩余优化点：优先级排序

四轮实测（ISA 静态归因 / 同步微基准 / geometry 扫描 / intra 带宽）之后的全景。
「收益」一律是 gfx950 实测，除注明外。

### 值得做

| # | 项 | 收益 | 风险 | 详见 |
|---:|---|---|---|---|
| G1 | **按 token 数调 launch geometry** | **3 – 10%** | 低（纯 host 侧查表） | 「launch geometry」节 |
| G3 | **RDMA / XGMI 两半的 block 分配**（`xgmiBlockNum = blockNum − rdmaBlockNum` 是余数，不是调出来的） | **+9%**（intra payload：417 → 455 GB/s） | 低 | 「DispatchIntraNode 审查」节 |
| G2 | 调 geometry 时 dispatch / combine 必须**一起**调 | 避免 18 µs 的净亏 | —— | 「launch geometry」节 |
| S1 | 619 fence 冗余 + 606/621/622 降 relaxed | 0.7 – 1.0 µs（0.2%） | 零 | 「二次测量 T1」 |
| C1 | **`weightsBuf` 空指针没有守卫**（85 / 685 行） | 修 bug，非性能 | 零 | 「DispatchIntraNode 审查」节 |

### 已测并否决

| 项 | 实测结果 |
|---|---|
| per-token 远程原子批量化（移植 `intranode.hpp` 三段式） | **负收益**，阶段是带宽瓶颈 |
| `destPeTokenCounter` 加 `cnt != 0` 守卫 | 0.3 µs |
| 自旋加 `s_sleep` | 带宽 **0**（只剩防 livelock 的理由） |
| ~~大 token 加 block 数~~ | **此条已撤销** —— 原结论来自 `reserve_vs_payload` 的 `ceil` 循环，该基准不能扫 block 数（见「复现」节末）。用正确的 grid-stride 重测后 block 数**有显著收益**，见 G1 / G3 |
| intra payload 的 `WarpCopy` 从 `Unroll=1` 提到 4 | **−0.1%**，且在 6 种 grid 下逐一验过 |
| intra metadata 三次分开拷 → 打包一次 | +0.7% |
| intra 先发 load 盖住远程原子（load-first） | +1.3% |

### 盲区：跨节点 AMO / doorbell 排队（本文全部基准都测不到）

本文所有实测都是**单节点**（XGMI + 本地内存），跨节点 NIC 站点（226 / 282 / 633 行）
一个都没覆盖。而 PR [#630](https://github.com/ROCm/mori/pull/630) 的数据显示
**这里才是同步开销的大头**：光 282 行那一次 AMO 就值约 3.3 µs 的 send 阶段，
比本文在能测范围内找到的全部（约 1 µs）还多 3 倍。

| 项 | 量级 | 状态 |
|---|---|---|
| 282 行 `nodeRecvTokenNum` AMO | **~3.3 µs**（send 阶段的一段 barrier） | #630 已在 LL 路径上做，非 LL 未做，见「PR #630」节 |
| 226 行 put 融合的 signal | 未测 | 与 payload 同一个 WQE，大概拆不开 |
| 633 行 `ShmemQuietThread` | 未测 | 每节点 1 次，必需 |

**所以「同步层面只剩 1 µs」这个结论要限定在「我能测的范围内」。**
要覆盖这一档需要 2 节点环境。

### 结构性机会（未测，有理由怀疑）

| 项 | 说明 | 风险 |
|---|---|---|
| 去掉 `EpDispatchCopyToStaging` | 一整趟额外 HBM 往返（T × xferBytes 读+写）。它存在是因为 RDMA 需要注册内存；若 `inpTokenBuf` 本身已注册即可直接发。**最大的结构性机会** | 高，要动内存管理 |
| `numRecvBlock = 8` 硬编码 | 在 `DispatchInterNodeRecv`（367 行），与 `rdmaBlockNum` / chunk 划分交织，完全不随 config 变，会限制 G1 的搜索空间 | 中 |
| recv 侧 4 次独立 `WarpCopy` | 源是交错的 staging，目标是 4 个分开的 buffer（`dispatchOut` / `shmemOutIndices` / `shmemDispatchOutWeights` / `shmemOutScales`），每对 token-expert 做 4 次 scatter。目标端也交错就能合成一次 | 中，改布局牵连 combine |

### 根本约束

大 token 下这个阶段撞在 **XGMI 写带宽墙**上：**最优 grid 下约 455 GB/s**
（intra 形状 128×16），而当前配置只跑到 417 GB/s —— 差的那 9% 就是 G3。
到了平台之后，改拷贝循环的形状（`Unroll`、metadata 打包、load-first）都动不了它，
三项实测分别是 −0.1% / +0.7% / +1.3%。
所以 G1 + G3 拿完之后，**单节点侧的同步没有更多可拿**
（跨节点侧见上面的盲区一节）。
再上一个数量级只有两条路：换传输（TDM / SDMA，即 gfx1250 那条路），
或减少搬运量（即上表的 `CopyToStaging`）。

### 前置条件：先把测量搞对

PR #625 的 `330edcb` / `a97ba1a` 记录了两件必须先处理的事，v1 同样适用：

- CCO/GDA 路径在 3 warmup + 10 rounds 下均值 run-to-run 摆动约 **20%** 且系统性偏高
  （shmem 第 3 轮就稳）。改用 20 warmup + 30 timed rounds。
- `dist.barrier()` 一次性放行所有 rank 造成 **thundering herd**：前几轮 collective
  同时开火撞上峰值 fabric 争用，第一轮约 1.6 – 2.5× 稳态。**不是** warmup/cache 效应。
  用 `MORI_EP_DROP_ROUNDS` 排除这个爬坡。

20% 的噪声会把 G1 那 3 – 10% 完全淹没。

### 另有两条非性能项

- **60 / 448 行 scope 不一致**：写 peer 的 `dispTokIdToSrcTokId`，intra 侧用
  `AtomicStoreRelaxedSystem`，recv 侧是普通 store。ISA 上两者都不发缓存维护，
  所以是一致性问题。
- **`dispatchGridBarrier` / `interNodeBlocksBarrier` 命名误导**：语义是 last-one-out
  计数器，没有任何 block 在上面等待，容易被误读成需要全 block 驻留的真 barrier。

---

## 同步层级阶梯

共 34 个同步站点，按代价从低到高。
另有两个 ISA 确认的全局事实：`s_barrier` = **0**、`s_sleep` = **0**。

| 层级 | 机制 / API | 站点 | 代价与用量 |
|---|---|---:|---|
| wave 内 | `__shfl` / `__ballot` / `__any` —— SIMD lane 交换，不访存 | 8 | 基本免费 |
| workgroup | `s_barrier`（`__syncthreads()`）硬件单元 | **0** | 整条路径一次都没用到（ISA 确认） |
| block 间 (agent) | L2/TCC 上的 agent-scope 原子计数器 | 10 | 全是 last-one-out 或 slot 分配，无人自旋 |
| 本地 · system scope | 对 `hipMalloc` 本地 buffer 用 SEQ_CST + SYSTEM / `__threadfence_system` | 4 | 语义强于需要，见下方实测 |
| 跨 GPU (XGMI) | symm uncached buffer 上的远程原子 / system-scope load-store | 6 | 含唯一一处等 peer 的自旋 |
| 跨节点 (NIC) | RDMA put 融合 AMO signal / 独立 AMO / `ShmemQuiet` | 5 | SYSTEM scope 必需，**非**过度同步 |
| kernel 边界 | runtime 隐式 grid barrier | 1 | dispatch 共 2 个 kernel |

---

## 逐阶段清单

标记：⚠ = scope 或自旋形态强于语义需要；🔥 = 路由相关的同步热点。

### Kernel 1 — `EpDispatchCopyToStaging`（grid = multiProcessorCount）

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 658–701 | 无同步原语 —— 仅 `WarpCopy` + lane 0 普通 store | — | — | 0 | ISA 确认：668 条指令里 0 缓存维护、0 原子、0 `s_barrier` |

### Kernel 边界

| 位置 | API | 层级 | Scope | 次数 | 备注 |
|---|---|---|---|---|---|
| `launch.cpp` 480–484 | kernel 边界（CopyToStaging → V1Kernel） | kernel 边界 | runtime 隐式 grid barrier | 1 | 唯一一次真正的全 grid 同步；不可能死锁 |

### `DispatchInterNodeSend`（RDMA blocks，`blockId < rdmaBlockNum`）

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 181 | `__ballot` / `__activemask` | wave 内 | 不访存 | 每 (warp, node, 64-token chunk) | dedup 同 destPe；`num == 0` 时整块跳过后面的原子 |
| 190 | `atomicAdd(blockFlagCounter + node, 1)` | block 间 (agent) | AGENT · 本地 `hipMalloc` | ≤ R × (N−1) × blockChunkNum | slot 分配器，**不是** barrier |
| 193–200 | `__shfl ×2`（replay 路径再 +1） | wave 内 | 不访存 | 同上 | 广播 `flagSlotId` / `flag` |
| 226 | `ShmemPutMemNbiSignalThread(..., AMO_ADD, proxyPe, qpId)` | 跨节点 (NIC) | RDMA write + remote AMO | 每 chunk 内每段连续 sender 1 次 | 数据搬运与 flag signal 融合进同一个 WQE |
| 274 | `atomicAdd(interNodeBlocksBarrier[0], 1)` | block 间 (agent) | AGENT · 本地 | R × W | last-one-out：先到的直接退出，无人自旋 |
| 281 | `core::AtomicLoadRelaxed(blockFlagCounter + lane)` | block 间 (agent) | RELAXED + AGENT · 本地 | 最后 1 warp × N lane | 整条 dispatch 路径唯一一处显式 agent-scope load |
| 282 | `ShmemAtomicTypeNonFetchThread<u64>(nodeRecvTokenNum, AMO_ADD)` | 跨节点 (NIC) | remote AMO | 最后 1 warp × (N−1) lane | 通报本节点发送总量；未按 QP 展开（combine 侧会） |

### `DispatchInterNodeRecv`（RDMA blocks）

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 390 ⚠ | `core::AtomicLoadRelaxedSystem(chunkFlag[node][k])` | 跨节点 (NIC) | RELAXED + SYSTEM | 无界自旋，每圈 1 次 | NIC 写的 flag，SYSTEM 必需；但循环**无退避** |
| 393 ⚠ | `core::AtomicLoadRelaxedSystem(nodeRecvTokenNum[node])` | 跨节点 (NIC) | RELAXED + SYSTEM | 同一自旋内每圈 1 次 | 作为「该节点已发完」的终止条件 |
| 400–401 | `__shfl ×2` | wave 内 | 不访存 | 每 bid 迭代 | 广播 chunk token 数 / nodeFlag |
| 433 | `__any`（跨 lane dedup） | wave 内 | 不访存 | 每 (recv token, expert) | 与越界 destPe 的守卫合并成一个 `shouldSkip` |
| 443 🔥 | `atomicAdd(dispTokOffsetMemObj->GetAs(destPe), 1)` | 跨 GPU (XGMI) | symm + `hipDeviceMallocUncached` | 每个落本节点的 (recv token, expert) | 结果经 `__shfl` 后决定其后 4 次 `WarpCopy` 的地址 —— 全路径最热的同步点 |
| 448 | `dispTokIdToSrcTokId->GetAs(destPe)[i] = srcTokId` | 跨 GPU (XGMI) | 普通 remote store（无 fence） | 同上 | 与 60 行同一语义，但那边用了 SYSTEM 原子 store |
| 450 | `__shfl` | wave 内 | 不访存 | 同上 | 广播 `destTokId` |
| 479 | `atomicAdd(destPeTokenCounter + destPe, cnt)` | block 间 (agent) | AGENT · 本地 | 每 warp × G lane | `cnt == 0` 也无条件发。返回值未使用 —— ISA 确认编译器已降级为 no-return 形式 |

### `DispatchIntraNode` + `DispatchIntraNodeBlock`（XGMI blocks，`blockId >= rdmaBlockNum`）

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 134 | `__any`（跨 lane dedup） | wave 内 | 不访存 | 每 (token, expert) | 哨兵 lane 用不可能的 destPe 避免误配 |
| 55 🔥 | `atomicAdd(dispTokOffsetMemObj->GetAs(destPe), 1)` | 跨 GPU (XGMI) | symm + `hipDeviceMallocUncached` | 每 (token, 本节点 expert) | 与 443 行争用同一批 per-destPe 计数器 |
| 60 | `core::AtomicStoreRelaxedSystem(dispTokIdToSrcTokId->GetAs(destPe) + id, …)` | 跨 GPU (XGMI) | RELAXED + SYSTEM | 同上 | recv 侧同一写法是普通 store（448 行），两侧不一致。RELAXED 不产生缓存维护指令 |
| 64 | `__shfl` | wave 内 | 不访存 | 同上 | 广播 `destTokId` |
| 146 | `atomicAdd(destPeTokenCounter + destPe, cnt)` | block 间 (agent) | AGENT · 本地 | 每 warp × G lane | 同 479 行 |

### `DispatchSync`（所有 B 个 block）

| 行 | API | 层级 | Scope | 次数 | 生成的指令（gfx950 实测） |
|---:|---|---|---|---|---|
| 601 | `atomicAdd(dispatchGridBarrier, 1)` | block 间 (agent) | AGENT · 本地 | B × W | `global_atomic_add`，无缓存维护 |
| 602 | `__shfl` | wave 内 | 不访存 | B × W | — |
| 606 ⚠ | `core::AtomicLoadSeqCstSystem(destPeTokenCounter + destPe)` | 本地 · system scope | **SEQ_CST + SYSTEM** | 最后 warp × G lane | `1× buffer_inv sc0 sc1` + `2× s_waitcnt vmcnt(0)` |
| 608 | `core::AtomicStoreSeqCstSystem(peer recvTokenNum + myPe, n)` | 跨 GPU (XGMI) | SEQ_CST + SYSTEM | 最后 warp × G lane | 写 peer 的 symm buffer —— 这一处 SYSTEM **必需** |
| 611 | `__hip_atomic_store(dispatchGridBarrier, 0, RELAXED, AGENT)` | block 间 (agent) | RELAXED + AGENT | 1 | `global_store … sc1`，1 条指令 |
| 617 ⚠ | `shmem::ShmemInt32WaitUntilGreaterThan(signal, 0)` | 跨 GPU (XGMI) | 内部 RELAXED + SYSTEM 裸自旋 | 最后 warp × G lane | 循环体 = `global_load sc0 sc1` + `s_waitcnt vmcnt(0)`，无 `s_sleep` |
| 618 | `atomicAdd(totalRecvTokenNum, n)` | block 间 (agent) | AGENT · 本地 | G | `global_atomic_add`，无缓存维护 |
| 619 ⚠ | `__threadfence_system()` | 本地 · system scope | **SYSTEM 全屏障** | G lane | `1× buffer_wbl2 sc0 sc1` + `1× s_waitcnt vmcnt(0)` + `1× buffer_inv sc0 sc1` |
| 621 ⚠ | `core::AtomicStoreSeqCstSystem(signal, 0)` | 本地 · system scope | **SEQ_CST + SYSTEM** | G | `1× buffer_wbl2 sc0 sc1` + `1× s_waitcnt vmcnt(0)` |
| 622 ⚠ | `core::AtomicStoreSeqCstSystem(destPeTokenCounter + destPe, 0)` | 本地 · system scope | **SEQ_CST + SYSTEM** | G | 同上 |
| 627 | `atomicAdd(crossDeviceBarrierFlag, 1)` | block 间 (agent) | AGENT · 本地 | 1 | epoch++，combine 侧的 barrier 目标值靠它 |
| 628 | `__hip_atomic_store(combineGridBarrier + 1, 0, RELAXED, AGENT)` | block 间 (agent) | RELAXED + AGENT | 1 | 替 combine 预清槽位 |
| 633–636 | `shmem::ShmemQuietThread(proxyPe)` | 跨节点 (NIC) | NIC CQ 排空 | warp 0..N−1 各 1 次 → N | 保证前面的 RDMA put 真正落地 |

---

## 示例配置下的绝对次数

`B` / `R` / `W` 取自 `tests/python/ops/test_dispatch_combine_internode_v1.py:45-51`
的真实测试配置：`block_num=96, rdma_block_num=64, warp_num_per_block=8`。

下表用 `N=2, G=8, T=4096, K=8`，
则 `blockChunkNum = ceil(ceil(T/64) / R) = ceil(64/64) = 1`。

只列公式确定、与路由无关的站点：

| 行 | 同步对象 | 公式 | 次数 / dispatch |
|---:|---|---|---:|
| 146 + 479 | `destPeTokenCounter` atomicAdd | `B × W × G` | **6,144** |
| 601 | `dispatchGridBarrier` atomicAdd | `B × W` | 768 |
| 274 | `interNodeBlocksBarrier` atomicAdd | `R × W` | 512 |
| 190 | `blockFlagCounter` atomicAdd | `R × (N−1) × blockChunkNum` | 64 |
| 226 | RDMA put + 融合 signal（下界） | `R × (N−1) × blockChunkNum` | 64 |
| 606 / 608 / 621 / 622 | SEQ_CST + SYSTEM 原子 | `4 × G` | 32 |
| 619 | `__threadfence_system()` | `G` | 8 |
| 633 | `ShmemQuietThread` | `N` | 2 |
| 282 | 节点级 `nodeRecvTokenNum` AMO | `N − 1` | 1 |

`destPeTokenCounter` 的 6,144 次超过其他所有站点之和（且多数在加零），
但 ISA 显示它是一条不带缓存维护的 `global_atomic_add`，单次很便宜。

**唯一与路由相关、且量级最大的一项不在表内**：55 / 443 行的
`dispTokOffsetMemObj` 跨 XGMI 远程原子，次数 = 落到本节点的 `(token, expert)` 对数，
上界 `T × K` = 32,768。

---

## gfx950 ISA 实测

### 1. 决定代价的是 ordering，不是 scope

用 `tools/isa_probe/scope_probe.hip` 隔离每个构造，gfx950 实测结果：

| 构造 | 生成的指令 |
|---|---|
| relaxed + **agent** load | `global_load … sc1` |
| relaxed + **system** load | `global_load … sc0 sc1` |
| acquire + agent load | `global_load sc1` + `s_waitcnt vmcnt(0)` + `buffer_inv sc1` |
| **seq_cst + system load**（606 行现状） | `global_load sc0 sc1` + `s_waitcnt vmcnt(0)` + `buffer_inv sc0 sc1` |
| relaxed + agent store | `global_store … sc1` |
| relaxed + system store | `global_store … sc0 sc1` |
| **seq_cst + system store**（621/622 现状） | `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0) lgkmcnt(0)` + `global_store sc0 sc1` |
| `__threadfence()` | `buffer_wbl2 sc1` + `s_waitcnt` + `buffer_inv sc1` |
| **`__threadfence_system()`**（619 行） | `buffer_wbl2 sc0 sc1` + `s_waitcnt vmcnt(0) lgkmcnt(0)` + `buffer_inv sc0 sc1` |

关键：**relaxed 无论 agent 还是 system 都只有一条指令，差别仅在 sc0 这一个位。**
所以把 scope 从 system 降到 agent 几乎不省东西；真正省的是把 ordering
从 seq_cst 降到 relaxed —— 那会让整条 `buffer_inv` / `buffer_wbl2`
连同全排空 `s_waitcnt` 一起消失。

与之对应的编译器逻辑在 `llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp`：
`expandLoad` 只在 ordering ∈ {Acquire, SeqCst} 时才调用 `insertAcquire()`
（即发 `buffer_inv`），而 scope 只被 `enableLoadCacheBypass()` 用来选 sc 位。

### 2. 619–622 那段序列的实际展开

探针 `probe_fence_then_seqcst_store` 复现了 619→621→622 的原样序列：

```
buffer_wbl2 sc0 sc1
s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_inv sc0 sc1          <- 619 __threadfence_system()
buffer_wbl2 sc0 sc1         <- 621 的 release，与上面 fence 的回写重复
s_waitcnt vmcnt(0)
global_store_dword … sc0 sc1
buffer_wbl2 sc0 sc1         <- 622，第三次全 L2 回写
s_waitcnt vmcnt(0)
global_store_dword … sc0 sc1
```

**9 条指令，其中 3 次整个 L2 回写、1 次 L2+L1 失效、3 次全排空。**
等价语义的 relaxed 版本（`probe_relaxed_store_pair`）是 **2 条指令**，
两个 store 之间连 `s_waitcnt` 都没有，可以重叠。

编译器不会合并 fence 的回写与 store 自带的 release —— LLVM 源码里留着 TODO
明确说明这一点（"If both release and invalidate are happening they could be
combined to use the single BUFFER_WBINV* instruction"）。

### 3. dispatch 主 kernel 的缓存维护指令总量与归因

`EpDispatchInterNodeV1Kernel_bf16`（5502 条指令）：

| 指令 | 条数 |
|---|---:|
| `buffer_wbl2 sc0 sc1` | 21 |
| `buffer_inv sc0 sc1` | 10 |
| `buffer_inv sc1` | 5 |
| `buffer_wbl2 sc1` | 3 |
| `s_waitcnt vmcnt(0)` | 192 |
| 原子（其中 12 条带 sc0 = 返回值形式） | 18 |
| `s_barrier` | **0** |
| `s_sleep` | **0** |

`EpDispatchCopyToStaging_bf16`（668 条指令）：缓存维护 0、原子 0、`s_barrier` 0。

按源码行归因（`-gline-tables-only` + `.loc`）：

| 来源 | 指令 | 是否必需 |
|---|---|---|
| `amd_device_functions.h:695`（`__threadfence_system` 展开） | 9× `buffer_inv sc0 sc1` + 9× `buffer_wbl2 sc0 sc1` | 大部分来自 shmem RDMA 原语，必需 |
| `utils.hpp:185`（`AtomicStoreSeqCstSystem`） | 5× `buffer_wbl2 sc0 sc1` | 部分必需（写 peer） |
| `ionic_device_primitives.hpp:218/395/563` | 6× `buffer_wbl2 sc0 sc1` | **必需** —— NIC doorbell 必须对设备可见 |
| `utils.hpp:292` / `utils.hpp:314` / `utils.hpp:140` / `utils.hpp:175` | 5× `buffer_inv sc1` + 3× `buffer_wbl2 sc1` + 1× `buffer_wbl2 sc0 sc1` | agent scope，shmem AMO 模拟路径 |
| `utils.hpp:145`（`AtomicLoadSeqCstSystem`） | 1× `buffer_inv sc0 sc1` | ← 就是 606 行 |

### 4. 消融归因：本文标记的 4 处到底占多少

在 `/tmp` overlay 里逐项降级后重新编译（仓库源码未改动），只看
`EpDispatchInterNodeV1Kernel_bf16`：

| 变体 | `inv sc0 sc1` | `inv sc1` | `wbl2 sc0 sc1` | `wbl2 sc1` | `vmcnt(0)` |
|---|---:|---:|---:|---:|---:|
| baseline | 10 | 5 | 21 | 3 | 192 |
| 删掉 619 的 fence | 9 | 5 | 20 | 3 | 191 |
| 606 → `AtomicLoadRelaxed` | 9 | 5 | 21 | 3 | 190 |
| 621+622 → `AtomicStoreRelaxed` | 10 | 5 | 19 | 3 | 190 |
| **四处全降级** | **8** | 5 | **18** | 3 | **187** |

即这 4 处合计贡献 **2 条 system-scope 失效 + 3 条 system-scope 回写 + 5 次全排空**，
是 dispatch kernel 31 条 system-scope 缓存维护指令中的 5 条（约 16%，静态计数）。

---

## ISA 一节推翻的三处判断

全文一共有 **6 处**被实测推翻的判断，分散在各自的测量节里：本节 3 处，
「launch geometry」节 2 处（`ceil` 基准导致的两条假结论），
「per-token 远程原子」C 节 1 处（slot 别名造成的假「局部性效应」）。

1. **「未使用的 `atomicAdd` 返回值」这一条作废。** 146 / 479 行 `int counter =
   atomicAdd(...)` 里 `counter` 从未使用。探针实测 unused-return 与显式 no-return
   生成**逐字节相同**的 `global_atomic_add`，真实 kernel 的归因也确认落在 no-return
   形式上。编译器已经处理了。

2. **「scope 用得过宽」这个归因是错的，应该说「ordering 用得过强」。**
   relaxed 无论 agent 还是 system 都只有一条指令；省下来的全部来自 seq_cst → relaxed。

3. **「打穿全卡 L2」说重了。** `buffer_inv` 失效的是该 wave 所在 **XCD** 的 L2。
   MI350 是 8 个 XCD × 32 CU，连带损失约 32 个 CU，不是全部 256 个。

一条仍然站得住的反向考虑：那 4 处位于 `DispatchSync`，执行在 dispatch 的**最末尾**、
所有 warp 都已收敛的时刻，即并行度最低、最无事可重叠的位置；而 RDMA 路径的 fence
分布在有大量其他工作在飞的阶段。所以**单次**代价可能不同，静态指令数看不出来。

---

## launch geometry：gfx950 实测（G1 / G2）

### 起因

PR #625 的 `5a7050d` 发现 cco 路径的 grid 是通过 launch redirect 从
`dispatch_combine.py` 的 shmem resolve 拿的，**不是 CU-aware**：
80 CU 的 MI308X 上 `InterNodeV1LL` 的 AUTO 默认 256/128/8 是 CU 数的 3.2 倍，
grid 尾部要跑第二个 wave。那个 resolve 就是 v1 自己的，所以问题对 v1 同样存在 ——
而 256 CU 的 gfx950 上测试配置的 96/64/8 只用了四分之一的部件。

他为 cco 路径建了 `internode_tuning_configs.py`（per-device × per-shape × per-token 查表，
每 phase 独立 `rdma_block_num`，`lookup()` 把 `block_num` 夹到 ≤ CU 数）。
**v1 的 AOT 路径没有等价物。**

### 扫描结果

`tools/isa_probe/grid_sweep.hip`，`DispatchInterNodeRecv` 的 payload 形状
（每对 token-expert：远程 `fetch_add` 取 slot → `__shfl` 广播 → payload `WarpCopy` 到 peer），
topk=8、hidden=7168 bf16、8 个 peer。循环是**按精确对数的 grid-stride**，
和真实 kernel 的 `for (i = globalWarpId; i < N; i += globalWarpNum)` 一致。

| tokens | 64×8（当前配置） | 最优 | 差 |
|---:|---:|---:|---:|
| 8 | 7.1 µs | 6.9 µs @ 32×4 | 3% |
| 32 | 14.7 µs | 14.7 µs @ 192×4 | 0% |
| 128 | 42.3 µs | 39.0 µs @ 256×4 | 8% |
| 512 | 143.1 µs | 138.0 µs @ 192×8 | 3.6% |
| 4096 | 1159.2 µs | 1045.0 µs @ 128×16 | **10%** |

**哪个旋钮更重要，随 token 数换手。**

- **小 token 看 warp 数。** t=8 时 4 warp 比 16 warp 快约 25%
  （7.0 vs 9.0 µs @ 64 block），而 block 数从 16 到 256 基本不动。
- **大 token 看 block 数。** t=4096、warps=8 时，block 32 → 128 是
  1321.2 → 1064.0 µs，**+24%**；相比之下同一点上 warp 8 → 16 只有 +9%
  （1159.2 → 1056.0 @ 64 block）。

block 数的形状是「有膝点」而不是「全平」：warps=8 时 32 → 96 陡升、96 以上趋平、
到 256 反而回退（1058.2 @ 192 → 1225.4 @ 256）；warps=16 时膝点更早，64 之后就平了。
**当前 XGMI 半边只有 32 block，正好落在陡坡上** —— 这就是 G3。

当前固定的 8 warp 是个两头都差 3 – 10% 的折中。

### G2：dispatch 与 combine 必须一起调

这一条我们没有独立复现，但 PR #625 的 `ddf9c9f` 有硬数据，而且机制对 v1 同样成立
（两个 phase 共享 QP 和 CUDA-graph replay）：

> tok4/8 的 dispatch 几何（32/16/6、64/16/4）单独调 dispatch 延迟是最优的，
> 但 `rdma_block_num=16` 的 dispatch 会让紧跟其后的 combine 慢约 18 µs
> （combine ~77 µs vs ~56 µs）。孤立的 dispatch 收益（~46 vs ~50 µs）被完全吃掉。

`de39b18` 又验证了一次：sweep 里各 phase 独立求 argmin 的结果，
在 dispatch / combine 用不同几何时**复现不出来**。tok16 最后改成两个 phase
共用一个 80/rdma40/warp4，比旧的 80/48/8 + 80/40/8 快约 4 µs。

**所以 G1 的调优必须以「(dispatch 几何, combine 几何)」为一个整体作为搜索单元**，
而不是各自求最优。

### 两处自我修正

**一、** 本节初版的 block 扫描显示「小 token 用 16 block 快 2.2×」，那是假的：
当时用 `pairs_per_warp = ceil(pairs / (blocks × warps))`，
在 `pairs < blocks × warps` 时总工作量随 block 数增长
（T=8 时 64 block 做了 16 block 的 4 倍工作）。
改成上面那个精确 grid-stride 后，真实差距是 3 – 10%。

**二、** 同一个 `ceil` 基准还产出过一条「大 token 下 block 数 32 → 256 带宽全平」，
被写进过「已测并否决」表和「根本约束」两处。**那条也是假的** ——
它的工作量随 block 数增长，正好抵消了真实收益。
用 grid-stride 重测后 block 数在膝点前有 +24%，两处引用已撤销并改正。
教训：一个基准被判定为某个维度不可用之后，要回头检查它之前产出的结论有没有
被别处引用 —— 我漏了一轮。

### 保留意见

`grid_sweep` 是**单卡向 8 个 peer 写**，没有 RDMA、没有 flag 轮询、没有真实路由倾斜。
所以这些数字说明的是「geometry 值 3 – 10%」这个量级，
真实 kernel 的最优点还受 `numRecvBlock = 8` 和 chunk 按 `blockId` 划分的影响，
必须在真实 kernel 上重测。

---

## 二次测量：gfx950

`tools/isa_probe/sync_round2.hip`，独立程序。grid 用真实的 96 × 8。

### T1. `DispatchSync` 尾部（606 + 617–622）

关键点：`buffer_wbl2` 只为**脏行**付钱，而 dispatch 结束时 L2 是脏的，
所以必须先把 L2 弄脏再测。768 个 warp 各写一段，经同一个 last-one-out 计数器汇合后，
由最后那个 warp 用 `wall_clock64` 计时整段尾部。
`:608` 那个写 peer 的 store 在两个变体里都保持 seq_cst + system（它是必需的）。

| 先脏化的 L2 | seq_cst + system（现状） | relaxed + agent | 差 |
|---:|---:|---:|---:|
| 0 KB | 1.539 µs | 0.774 µs | 0.765 µs |
| 64 KB | 3.828 µs | 3.122 µs | 0.706 µs |
| 256 KB | 3.616 µs | 2.770 µs | 0.846 µs |
| 1024 KB | 3.876 µs | 2.914 µs | 0.962 µs |

**降级省 0.7 – 1.0 µs / 次 dispatch。** 参照约 450 µs 的迭代，是 0.2%。
绝对量小，但改动只有 4 行、语义无风险、且 ISA 已证明那对 `buffer_wbl2` 确实重复。
剩下的约 2.9 µs 下不去 —— 那是 `:608` 必需的 peer store。

### T2. `destPeTokenCounter` 的守卫

768 warp × 8 lane = 6,144 次 agent-scope 原子：

| 非零比例 | 无守卫 | 加 `cnt != 0` | 差 |
|---|---:|---:|---:|
| 1/1 | 10.01 µs | 9.71 µs | 0.30 µs |
| 1/4 | 10.00 µs | 9.73 µs | 0.27 µs |
| 1/8 | 10.03 µs | 9.74 µs | 0.29 µs |
| 1/32 | 9.99 µs | 6.71 µs | 3.28 µs |

只有在极度稀疏时才有意义，真实密度下省 0.3 µs。**否决。**

### T3. 自旋是否抢带宽

copier 在 stream A，spinner 在 stream B 并发，spinner 用 `wall_clock64` **按时间封顶**
而不是按迭代数（按迭代数只会测出「自旋自己跑多久」，`s_sleep` 因为自旋次数少反而显得更慢
—— 第一版就是这么错的）：

| spinner blocks | 无 spinner | 热自旋 | `s_sleep` 自旋 |
|---:|---:|---:|---:|
| 8 | 147.6 µs | 156.8 µs (+6.2%) | 155.0 µs (+5.0%) |
| 32 | 147.6 µs | 158.0 µs (+7.0%) | 157.9 µs (+6.9%) |
| 64 | 147.6 µs | 157.2 µs (+6.5%) | 157.1 µs (+6.4%) |

自旋 block 确实让并发拷贝慢约 6.5%，但**热自旋与 `s_sleep` 自旋几乎完全相同**。
干扰来自 spinner 占着 CU，不是它的访存流量。
**所以加 `s_sleep` 换不回带宽**，它只值防 livelock 那份保险。

---

## DispatchIntraNode 审查（45–148 行）与 intra 带宽实测

### C1. `weightsBuf` 空指针没有守卫 —— 潜在 crash

85–87 行无条件从 `args.weightsBuf` 读，而紧接着的 scales 却守卫了
（`if (args.scalesBuf && (scaleBytes > 0))`）：

```84:87:src/ops/dispatch_combine/internode_v1.cpp
  float* remoteWeightPtr = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe);
  const float* localWeightPtr = args.weightsBuf;
  core::WarpCopy(remoteWeightPtr + destTokId * config.numExpertPerToken,
                 localWeightPtr + tokenId * config.numExpertPerToken, config.numExpertPerToken);
```

`WeightBytes()` 是无条件的 `numExpertPerToken * sizeof(float)`（`dispatch_combine.hpp:191`），
永不为 0，所以这次拷贝一定执行。

**空 weights 是显式支持的输入** —— `dispatch_combine.py:1229`：
`weight_ptr = weights.data_ptr() if weights is not None else 0`。
同文件的 `DEF_COMMON_VARS` 也在判它（`combXferBytes = (args.weightsBuf == nullptr) ? …`），
v1 自己的 **combine** 侧在 756 / 854 / 909 / 1012 / 1165 行处处守卫。

整个 kernel 家族里只有 v1 dispatch 不守卫：

| 位置 | 处理方式 |
|---|---|
| `internode.hpp:422`（v0） | `weightSize = args.weightsBuf ? K * sizeof(float) : 0`，把长度算成 0 |
| `intranode_ll.hpp:293`、`intranode_1250x.hpp:902/924/…` | `if (args.weightsBuf)` 守卫 |
| v1 **combine**（756 / 854 / 909 / 1012 / 1165） | 守卫 |
| **v1 dispatch（85、685）** | **无守卫** |
| `low_latency_async.cpp:107 / 223` | 无守卫 |

严重程度：`tokenId = 0` 时地址就是 0，必然 HSA page fault，所以是**崩溃而非静默错误**。
目前是**潜在**的 —— tests / examples / benchmark 里没有任何调用给 dispatch 传 `None`
（`test_dispatch_combine_internode.py:890` 那个 `dispatch_weights = None` 是给 combine 用的）。
属于「API 承诺了但 kernel 没兑现」。

### G3. XGMI block 数是余数，不是调出来的 —— 值 9%

```104:104:src/ops/dispatch_combine/internode_v1.cpp
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
```

intra 路径的 grid 不是独立参数。测试配置 96 / 64 → XGMI 侧只拿到 **32 个 block**。
`tools/isa_probe/intra_bw.hip` 复刻了 `DispatchIntraNodeBlock` 的完整序列
（远程 `fetch_add` → `__shfl` → payload + metadata `WarpCopy` 到 peer），
T=4096、topk=8、hidden=7168 bf16：

| grid（XGMI block × warp） | Unroll=1 | Unroll=4 |
|---|---:|---:|
| **32×8 ← 当前** | **417.7 GB/s** | 416.9 |
| 64×8 | 441.3 | 441.0 |
| 128×8 | 451.0 | 451.2 |
| 64×16 | 450.0 | 449.6 |
| **128×16 ← 最优** | **455.2 GB/s** | 453.9 |
| 256×16 | 451.6 | 450.3 |

**32×8 → 128×16 是 +9%。** 而 gfx950 有 256 CU，`blockNum=96` 只用了 37.5%；
要让 XGMI 侧拿到 128 block 同时保住 `rdmaBlockNum=64`，`blockNum` 提到 192 即可。

**这里没有取舍**：`grid_sweep` 里 recv 形状也是多 block 更好（64×8 = 1159 µs，
128×8 = 1086 µs），所以在 256 CU 的卡上提 `blockNum` 两半都受益。
取舍只存在于 CU 紧张的卡上（MI308X 80 CU）—— 那正是 PR #625 `5a7050d` 的情形，
他要把 `block_num` **夹到** ≤ CU 数；这台卡是反过来，**给得太少**。

### 三处实现不一致，实测都不值钱

`core::WarpCopy` 的第二个模板参数是 **`Unroll`，不是向量宽度** ——
`WarpCopyImpl` 内部恒用 16B（`device_primitives.hpp:277`）。
recv 路径写 `core::WarpCopy<uint8_t, 4>`，intra 路径用默认 `Unroll=1`，
即访存并行度只有四分之一。但上表逐 grid 验过，**每一行 Unroll=1 与 4 都一样**，
不是被 grid 限流掩盖 —— 在带宽墙下这个并行度已经够了。

| 项 | 现状 vs 改后 | 实测 |
|---|---|---:|
| payload `Unroll` | 1 → 4 | −0.1% |
| metadata | 3 次分开拷 → 打包 1 次 | +0.7% |
| 原子与 load 的顺序 | 先原子 → 先 load | +1.3% |

metadata 那条的机制：`K × 4 = 32 B` 低于一个向量步长，所以三次拷贝**全部落进
`WarpCopy` 的标量尾巴**，64 个 lane 里只有 8 个在动、各存 4 B。
字节占比只有 0.5%，所以改了也看不出来。

load-first 那条在 git 历史里有据 —— 被删掉的 LEGACY intranode kernel 注释写着
"LOAD first ... so the atomic's ~us cross-GPU round-trip overlaps the load"，
v1 intra 没继承。实测 1.3%。

### 两处代码卫生问题（非性能）

- **`laneNode` 是死变量，且藏了负数除法的坑。** 123 / 129 行算出 `laneNode` 后
  全函数再没读过。而哨兵 lane 的 `lanePe` 是负数（`-1 - laneId`），
  C++ 负数除正数向零截断，`-1/8 … -7/8` 全是 **0** —— 哨兵会看起来属于 node 0。
  recv 侧对同一件事是显式守卫的（424 行 `isSentinelSlot ? -1 : destPe / gpuPerNode`）。
  现在无害，但将来有人拿 `laneNode` 做节点级 dedup 就会中招。130 行还有个多余的分号。
- **105 行在 `rdmaBlockNum == blockNum` 时除零**，只因调用点在 `else` 分支里到不了。

### 一处值得记的正面设计

`i = warpId + j * warpNum` 配 `tokenId = i / K`，在 `warpNum == K` 时
（测试配置正是 8 == 8）让**所有 warp 同时处在同一个 token 上** ——
于是那 K 次 `inpTokenBuf` 读和 K 次 index 读都落在 L2 上，读放大被吸收。
代价是 warp w 恒定只处理 expert slot w，dedup 的 `inTokenExpertId = warpId`
使各 warp 的 dedup 工作量按 slot 序号倾斜（warp 0 恒不 dedup）。
净下来是好的权衡，别照直觉去「修」。

---

## PR #630：跳掉 282 / 353 行那次跨节点 AMO

PR [#630](https://github.com/ROCm/mori/pull/630)（isytwu，draft）优化的正是本文
清单里的 282 行（非 LL）/ 353 行（LL）站点 —— 也是本文唯一无法实测的一类。

### 思路

send 尾部在 RDMA put 之后，同一个 warp 再发一次跨节点 AMO 公布
「我发了多少 chunk」（`nodeRecvTokenNum`）。接收侧用它做两件事：
轮询时区分「这个 slot 永远不会被 signal」和「它还没到」；以及告诉 combine 迭代到哪。

**当每个 slot 都已被 put 自带的 signal 覆盖时，这两件事都不需要它。**
不变量：`chunksSent <= maxChunkNum` 恒成立 —— `chunksSent` 是每 `warpSize` 个 token
一次 `atomicAdd`，而 `maxChunkNum = ceil(maxNumInpTokenPerRank / warpSize)`
是同一个上取整。所以**相等即全覆盖**，轮询必然在 `chunkFlag` 上终止，
那条「永不到达」的捷径成了死代码。于是加一个
`chunksSent == maxChunkNum` 判断把 AMO 跳掉。

核对过 slot 分配：`flagSlotId` 来自共享的 `atomicAdd(blockFlagCounter + node, 1)`，
所以 slot 号连续占满 0..chunksSent−1，相等确实等价于 0..maxChunkNum−1 全被 signal。
推理成立。

### 数据

设备侧计时器测出那段 barrier segment 从 540 降到 208 个 tick，约 **3.3 µs**
（100 MHz wall clock，1 tick = 10 ns，差 332 tick 正好对上）。

端到端（2×8 MI300X，AUTO，num-qp 1，hidden 6144，3 次取中位，dispatch µs）：

| tokens/rank | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| before | 49.53 | 47.99 | 49.83 | 52.18 |
| after | 47.29 | 47.66 | 48.89 | 51.75 |

作者自己标注：方向一致但幅度小，只有 tok16 / tok32 的 per-run 区间不重叠，
4 和 8 是重叠的。combine 不变，符合预期。

### 3.3 µs 的来源（本文推断，PR 未言明）

`ShmemAtomicTypeNonFetchThread` 是 **non-fetch**、fire-and-forget 的，不等返回值，
一次 WQE 不该要 3.3 µs。但这个 AMO 由**最后一个 warp** 发出
（在 `(finishedWarp + 1) == rdmaBlockNum * warpNum` 分支里），
此时其他所有 warp 的 put 已经在 doorbell 链上占了位，
所以它的 `ringDoorbellOrdered` 要排在全部 put 之后等自己的号。
**3.3 µs 是 doorbell 排队延迟，不是 AMO 本身。**

这与 PR #625 分析 `ringDoorbellOrdered` / `dbTouchIdx` 死锁时的那条链路是同一处。

### 会限制它落地的条件

触发条件是 `chunksSent == maxChunkNum`，而分母 `maxNumInpTokenPerRank`
（`MaxNumTokensToSendPerRank()`，`dispatch_combine.hpp:154`）是**声明的容量**，
不是实时 token 数。

benchmark 通常把 `max_num_inp_token_per_rank` 设成接近实际 token 数，所以条件总成立
（作者说 4–32 tokens 时一个 chunk 覆盖全部）。但生产里它是个宽松上界
（struct 默认 128，实际配置更大），decode 步只有 8 个 token 时
`chunksSent = 1` 而 `maxChunkNum = 64` —— **条件不成立，零收益**。

所以那张表的 0.4 – 2.2 µs 能否搬到生产，取决于容量声明有多贴。

### 非 LL 路径也适用，而且更简单

282 行与 353 行结构完全一样，但两条路径消费 `nodeRecvTokenNum` 的方式不同：

| | 消费方式 | 需要的 combine 改动 |
|---|---|---|
| **LL** combine（1113 行） | `nodeCount` 直接当外层循环界，读到 0 就 `continue` 跳过整个节点 | **必须**加静态 `maxChunkNum` 回退 —— PR 做了 |
| **非 LL** combine（970 行） | 只在 `chunkFlag == 0` 分支里查，且有 `nodeFlag > 0` 守卫；外层循环界本来就是静态 `maxChunkNum` | **不需要** —— AMO 不发时守卫为假，chunk 自然被跳过，已是正确行为 |

所以非 LL 侧只需要 send 端加一个判断，是比 LL 更干净的一次改动。

### 风险

recv 的轮询（389–398 行）是只有两个出口的无界 `while (1)`。
守卫一旦判错，症状是**挂死**而不是算错。
不变量核对下来是成立的，但这是 hang 类改动，值得比常规 perf 改动更小心。

---

## per-token 远程原子：gfx950 实测

`tools/isa_probe/atomic_latency.hip`，独立程序，只需 hipcc。
counter 按真实布局摆放：`dispTokOffsetMemObj` 是每个 PE 各自
`MallocSymm(sizeof(index_t), hipDeviceMallocUncached)`，所以每个 destPe 的 counter
在**它自己那张 GPU** 上、彼此相隔 4 KB（若把 8 个 counter 挤在一个 peer 的同一条
cacheline 上，会凭空造出约 8 倍争用外加 false sharing —— 第一版就踩了这个坑）。

### A. 依赖链延迟（warp 在拿到 `destTokId` 前真正停的时间）

lane 0 连发原子，每次的地址依赖上一次的返回值，强制每轮 `s_waitcnt`：

| 目标 | scope | 延迟 / 次 |
|---|---|---:|
| 本地 `hipMalloc` | AGENT | 0.21 – 0.27 µs |
| 本地 `hipMalloc` | SYSTEM | 0.25 µs |
| 本地 `hipDeviceMallocUncached` | SYSTEM | 0.34 µs |
| **远程 peer `hipDeviceMallocUncached`**（55 / 443 行的原样） | SYSTEM | **0.40 – 0.64 µs** |

**对比 gfx1250 的 ~3.5 µs**（该数字在本仓库 git 历史里有据：
`6a4d74756^` 版 `intranode.hpp` 的 `MORI_DISP_TIMING` 注释
"If this ~matches the asm-barrier number, it proves the ~3.5us completion latency"）。
即 gfx950 上这条往返比 gfx1250 快 6–9 倍 —— 当年逼出 batch 改造的压力在这台机器上
弱得多，但没有消失。

另外注意 uncached 本身要花 0.34/0.25 ≈ 1.35 倍，跨卡再叠一层。

### B. 孤立聚合开销（**已被 C 节推翻，仅作参考**）

只测原子、不带 payload 时，96 × 8 grid、npes=8、T=4096：

| topk | (token,expert) 对 | per-token | batched | 省下 |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 9.82 µs | 4.39 µs | 5.4 µs |
| 4 | 16,384 | 30.54 µs | 7.23 µs | 23.3 µs |
| **8** | **32,768** | **57.72 µs** | **10.18 µs** | **47.5 µs** |

远程原子条数从 33,024 降到 768（43 倍）。per-token 一列对 topk 完全线性，
边际成本 `(57.72 − 9.82) / (32768 − 4096)` = **1.67 ns 每对** ——
单次延迟 0.4 µs 而边际只有 1.67 ns，说明 768 个 warp 在飞已经把延迟摊掉了，
**问题是吞吐/争用，不是暴露的延迟**。

这个 47.5 µs 是「暴露出来的部分」的**上界**：真实 kernel 里 payload copy 会进一步
遮蔽它。下一节把 payload 加回来后它基本消失。

### C. 把 payload 放回去之后：**不值得改**

`tools/isa_probe/reserve_vs_payload.hip`：三种取 slot 的方案
搬运**逐字节相同**的负载（hidden 14336 B + meta 80 B，写到 peer 显存），
grid 用真实的 `rdmaBlockNum=64 × 8 warps`，唯一差别就是 slot 怎么来的。

T=4096, hidden=7168 bf16, gfx950, 单卡向 8 个 peer 写：

| 变体 | topk=4 | topk=8 |
|---|---:|---:|
| 1. no_reserve（连续 slot，零原子） | 569.0 µs / 414 GB/s | 1113.0 µs / 424 GB/s |
| 2. per_token（当前 internode_v1） | 549.0 µs / 429 GB/s | 1100.1 µs / 429 GB/s |
| 3. batched（`intranode.hpp` 方案） | 558.0 µs / 422 GB/s | 1128.1 µs / 419 GB/s |
| 4. batched + per-token 空转原子 | 566.7 µs / 416 GB/s | 1139.2 µs / 415 GB/s |
| 5. batched，warp 连续派号 | 539.6 µs / 437 GB/s | 1097.4 µs / 430 GB/s |

分解（topk=8）：

| 项 | 值 | 含义 |
|---|---:|---|
| `per_token − batched` | **−28.1 µs** | 两种方案直接 A/B：**batched 反而略慢** |
| `dummy − batched` | **+11.1 µs** | 同等局部性下 per-token 原子的净成本 ≈ **1%** |
| `batched − warpcontig` | +30.7 µs | 派号**顺序**的影响，约 3% |
| `warpcontig − no_reserve` | −15.5 µs | 残差，噪声量级 |

**五个变体全部落在 1100 µs / 420–430 GB/s 的 ±3% 以内。**
取 slot 的方案换成什么都改变不了它 —— 512 个 warp 在飞，
把 0.4 µs 的原子往返完全藏住了。

一处口径提醒：这里的 420–430 GB/s 是在 **64×8** 这个当前配置下测的，
而 G3 表明最优 grid（intra 形状 128×16）能到 **455 GB/s**。
所以这个数字不是带宽墙本身，是「当前 grid 下的平台」。
本节是同 grid 下的 A/B，结论不受影响；但别拿 420–430 当天花板引用。

结论：**这个移植不值得做。** batched 省下的远程往返次数（43 倍）
换不回它自己 Phase 1 全量计数 + 多一次 `__syncthreads()` 的开销；
per-token 原子在真实负载下只占约 1%。

这也解释了它与 gfx1250 的差异：那边原子要 3.5 µs 且 payload 走 TDM 更快，
原子是**暴露**的；gfx950 上原子只有 0.4 µs 而 payload 撞在带宽墙上，原子被完全遮蔽。

**一处自我修正**：本节初版有一个 `batched − no_reserve = +920 µs` 的「局部性效应」，
那是假的 —— 当时 `no_reserve` 的 slot 用全局对索引取模，导致 8 个 warp 写同一个 slot，
反复命中同一批 cacheline，并没有搬运真实 footprint。
改成不重叠的连续派号（变体 1 现在的样子）后，它和其余变体一样是 424 GB/s。

### D. 若将来换到原子更慢的 arch 仍要移植

`intranode.hpp:116-216` 是直接模板。两个要点：recv 侧循环是
`(node, chunk, recvBlock)` 而非平坦 `Npair`，「计数域」要重新定义；
以及 `intranode.hpp:167-173` 那条不变量 —— Phase 1 与 Phase 3 必须用完全相同的
循环边界、步长和过滤条件，否则路由静默错乱。

---

## 复现

```bash
export ROCM=/shared/apps/ubuntu/opt/rocm-10.0.0
export PATH=$ROCM/bin:$ROCM/llvm/bin:$PATH

# 1. 隔离探针 —— 确认 ordering/scope 到指令的映射
cd tools/isa_probe
hipcc -O2 --offload-arch=gfx950 --cuda-device-only -S \
      -o scope_probe.gfx950.s scope_probe.hip

# 2. 真实 v1 kernel（编译参数照抄 src/ops/CMakeLists.txt:75-85）
cd <repo root>
hipcc --genco --offload-arch=gfx950 -std=c++17 -O2 -S -gline-tables-only \
  -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS -DMORI_DEVICE_NIC_IONIC \
  -I./include -I. -I./3rdparty/spdlog/include \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  src/ops/kernels/ep_internode_v1.hip -o /tmp/isa/v1_lines.s

# 3. 按 kernel 统计（EpDispatchInterNodeV1Kernel_bf16 / EpDispatchCopyToStaging_bf16）
grep -nE 'buffer_inv|buffer_wbl2|s_sleep|s_barrier' /tmp/isa/v1_lines.s

# 4. 消融归因 —— 用 overlay 目录覆盖，不动仓库源码
#    mkdir -p /tmp/ov/src/ops/dispatch_combine
#    cp src/ops/dispatch_combine/internode_v1.cpp /tmp/ov/src/ops/dispatch_combine/
#    sed -i '619s|__threadfence_system();|/*ablate*/;|' /tmp/ov/src/.../internode_v1.cpp
#    然后在上面第 2 步的命令里把 -I/tmp/ov 放在 -I. 之前
```

`-gline-tables-only` 会把汇编从 155k 行放大到 288k 行，但 `.loc` 指令让每条
缓存维护指令都能归因到源码行，这是整套分析里最关键的一步。

```bash
cd tools/isa_probe

# 5. per-token 原子（需要 >=2 张空闲 GPU）
hipcc -O2 --offload-arch=gfx950 -o atomic_latency atomic_latency.hip
./atomic_latency 0 1 4096 8          # src_gpu dst_gpu tokens topk
for k in 1 2 4 8; do ./atomic_latency 0 1 4096 $k; done

# 6. 把 payload 放回去后的 A/B（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o reserve_vs_payload reserve_vs_payload.hip
for k in 1 2 4 8; do ./reserve_vs_payload 4096 $k 7168; done

# 7. 二次测量：DispatchSync 尾部 / counter 守卫 / 自旋干扰（单卡）
hipcc -O2 --offload-arch=gfx950 -o sync_round2 sync_round2.hip
./sync_round2

# 8. launch geometry 扫描（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o grid_sweep grid_sweep.hip
./grid_sweep 8 7168                  # topk hidden

# 9. intra 带宽：Unroll / metadata 打包 / load-first / XGMI block 分配（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o intra_bw intra_bw.hip
./intra_bw 4096 8 7168               # tokens topk hidden [xgmi_blocks] [warps]
for cfg in "32 8" "64 8" "128 8" "64 16" "128 16" "256 16"; do
  ./intra_bw 4096 8 7168 $cfg
done
```

**一个用错就会得出假结论的坑**：`reserve_vs_payload.hip` 的循环用的是
`pairs_per_warp = ceil(pairs / (blocks × warps))`，在 `pairs < blocks × warps` 时
**总工作量随 block 数增长**。它对三个变体的 A/B 仍然有效（工作量相同），
但**不能用它扫 block 数**。扫 grid 用 `grid_sweep.hip` 或 `intra_bw.hip`，
两者都是按精确对数的 grid-stride。

## 参考

- PR [#630](https://github.com/ROCm/mori/pull/630) —— 跳掉 282/353 行那次跨节点 AMO。
  本文唯一测不到的那一档的实测数据来源，见「PR #630」节。
- PR [#625](https://github.com/ROCm/mori/pull/625) —— v1 internode 移植到 CCO/GDA。
  本文引用的 geometry（`5a7050d`）、dispatch↔combine 耦合（`ddf9c9f`、`de39b18`）、
  launch 批量化（`e21c3d3`）、测量方法（`330edcb`、`a97ba1a`）都来自这个 PR 的
  `jhchouuu` 部分。注意那些改动落在 **cco / v2 路径**（`ep_internode_kernel.hpp`），
  不在本文分析的 AOT v1 路径上。
- `llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp` —— gfx940/950 分支决定了
  ordering / scope 到缓存维护指令的映射，是「ISA 实测」一节的上游依据。
- `docs/rdma_bandwidth_utilization.md` —— v1 的阶段划分图与 RDMA 带宽核算方法。
  注意其中提到的 `analyze_trace_internode.py` 并不存在，实际叫
  `tools/profiler/analyze_ep_kernel_trace.py`。
