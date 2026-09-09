# InterNode V1 Dispatch —— 同步点清单、gfx950 ISA 实测、剩余优化点

清点自 `src/ops/dispatch_combine/internode_v1.cpp`（dispatch 段 45–701 行）与 `launch.cpp`。
表中「行」为 `internode_v1.cpp` 行号。**只覆盖 `KernelType::InterNodeV1`**，不含 `InterNodeV1LL`。

> **行号基准：`867cbdb6`（2026-09-08）。** 全文行号一律对这个 commit。
> 工作区若有未提交改动，行号会漂 —— 核对时先 `git stash` 或用 `git show 867cbdb6:<file>`。
> （写这版时工作区在 382 行多一行注释，其后全部 +1；本节「复现」第 10 步产出的 ISA
> `.loc` 值就带着这个 +1，正文已折算回基准行号。）

本文最初只清点同步点，实测下来**同步层面几乎没有可回收量**，于是逐步扩到 launch geometry、
再扩到三个阶段的逐行审查。现在最大的几块是 **launch geometry（G1/G3）**、
**`CopyToStaging` 的 kernel 边界（F1）** 和 **recv 的冗余搬运（P1/P2）**。
先看下一节的优先级排序，再按需往下读。

**口径**：「站点数」按代码位置计（一行里的 `__shfl ×2` 算一个站点），不含 `ENABLE_PROFILER`
才编译的 profiler 桩；ISA 数字一律是**静态指令数**，非运行时开销。
符号：`B`=blockNum, `R`=rdmaBlockNum, `X`=B−R, `W`=warpNumPerBlock,
`N`=nNodes, `G`=gpuPerNode, `T`=curRankNumToken, `K`=numExpertPerToken。

**环境**：gfx950（MI350，8 卡 / 每卡 256 CU / 8 个 XCD，由 `gfx_target_version=90500` 确认）；
NIC ionic（8 ionic + 2 bnxt 取多数）→ `-DMORI_DEVICE_NIC_IONIC`。
工具链**有两套**：ISA 静态归因与全部运行时基准用 **ROCm 10.0.0**（HIP 7.15，clang 23.0.0git）；
「复现」第 10 步那次 recv 反汇编（R3 / C2 / P1 的依据）用的是 **ROCm 7.1.1** ——
循环结构、依赖链、寄存器数这类结论跨版本大概率一致，但严格说是另一个工具链的观测。

---

## 剩余优化点：优先级排序

四轮实测（ISA 静态归因 / 同步微基准 / geometry 扫描 / intra 带宽）之后的全景。
「收益」一律是 gfx950 实测，除注明外。

### 值得做

| # | 项 | 收益 | 风险 | 详见 |
|---:|---|---|---|---|
| G1 | **按 token 数调 launch geometry** | **3 – 10%** | 低（纯 host 侧查表） | 「launch geometry」 |
| G3 | **RDMA / XGMI 两半的 block 分配**（`xgmiBlockNum = blockNum − rdmaBlockNum` 是余数，不是调出来的） | **+9%**（intra payload 417 → 455 GB/s） | 低 | 「intra 审查」 |
| G2 | 调 geometry 时 dispatch / combine 必须**一起**调 | 避免 18 µs 净亏 | —— | 「launch geometry」 |
| F1 | **就地融合 `EpDispatchCopyToStaging`**，去掉那次 kernel 边界 | 未测；省一整趟 HBM 往返 + NIC 空闲窗口 | 低到中 | 「send 审查」 |
| S1 | 619 fence 冗余 + 606/621/622 降 relaxed | 0.7 – 1.0 µs（0.2%） | 零 | 「二次测量 T1」 |
| P2 | **recv 的 payload 改 tile-broadcast**：同一 token 的源被重复读 `K_d ≈ 3.3` 次，且 `dispatchInp` 是 **uncached**（读不走 L2） | 未测；该阶段流量 **−35%**（94.6 → 61.6 KB/token） | 中 | 「recv 审查 / P2」 |
| P1 | **recv 的 K 个远程原子并行发**，依赖链 8 → 1（ISA 确认现在是严格串行且每轮全排空） | 未测；grid 只有 2 wave/SIMD，这条链盖不住 | 低 | 「recv 审查 / P1」 |
| C1 | **`weightsBuf` 空指针没有守卫**（85 / 685 行） | 修 bug，非性能 | 零 | 「intra 审查」 |
| C2 | **`destTokId` 溢出没有守卫**（444 行）——`assert` 在这条路径上从不产生陷阱，与 NDEBUG 无关 | 修 bug，非性能 | 零 | 「recv 审查」 |
| C3 | **G1 的搜索空间有两条没写下来的硬约束**，违反会静默算错 | 防止调 geometry 时踩雷 | 零 | 「recv 审查」 |
| R1 | **recv 外层循环界是声明容量而非实际 token 数** | 小 token 区间；T=8/cap=4096 时 512 次迭代只有 8 次干活 | 低 | 「recv 审查」 |

### 已测并否决

| 项 | 实测结果 |
|---|---|
| per-token 远程原子批量化（移植 `intranode.hpp` 三段式） | **负收益**，该阶段是带宽瓶颈。**注意适用范围**：这里比的是「原子怎么取 slot」（条数差 43 倍），五个变体的**依赖链深度都一样**，所以它不覆盖 P1（链深）也不覆盖 P2（冗余读） |
| `destPeTokenCounter` 加 `cnt != 0` 守卫 | 0.3 µs |
| 自旋加 `s_sleep` | 带宽 **0**（只剩防 livelock 的理由）。**但这个结论只覆盖「各转各地址」的形态**，不适用于 390 行，见 T3 的适用范围 |
| ~~把 226 行的 per-thread put 合并成 warp 级发送~~ | **已经是了。** `ShmemPutMemNbiSignalThread` 内部就是 warp 协作的，见「send 审查 / S3」。改用 `...SignalWarp` 反而会算错 |
| intra payload 的 `WarpCopy` 从 `Unroll=1` 提到 4 | **−0.1%**，6 种 grid 下逐一验过 |
| intra metadata 三次分开拷 → 打包一次 | +0.7% |
| intra 先发 load 盖住远程原子（load-first） | +1.3% |
| ~~大 token 加 block 数~~ | **已撤销**，原结论来自坏基准，见「修正记录」#4/#5 |

### 结构性机会（未测，有理由怀疑）

| 项 | 说明 | 风险 |
|---|---|---|
| **就地融合 `CopyToStaging`**（= F1） | 一整趟额外 HBM 往返（`T × xferBytes` 读+写），且**在 NIC 完全空闲的窗口里串行跑完**。4 次 `WarpCopy` 搬进 send 循环即可，buffer 布局不变 | 低到中 |
| ~~换成 ring buffer 省 footprint~~ | 与 F1 不是同一件事。门槛 `T > 64 × R × D`，测试配置要 **T > 8192** 才开始省，T=4096 时反而大一倍 | 高，当前 T 下负收益 |
| `numRecvBlock = 8` 硬编码 | 在 `DispatchInterNodeRecv`（368 行），与 `rdmaBlockNum` / chunk 划分交织，不随 config 变。**它不只是限制 G1 搜索空间，还是两条硬正确性约束的来源** —— 见 C3 | 中 |
| **390 行自旋：512 个 lane 打同一条 cacheline** | `k`/`node` 只由 `blockId` 推出，一个 block 的 8 个 warp 轮询同一地址；再 ×8 个共享 chunk 的 block ×8 个并发 k = 512 lane 打 64 字节 uncached 区，无退避，而 NIC 正要往那里写 | 未测，先要能测到自旋时长 |
| recv 侧 4 次独立 `WarpCopy` | 源在 staging 里是连续的一条 14404 B 记录，目标却是 4 个分开的 buffer，每对 token-expert 做 4 次 scatter。目标端也交错就能合成一次。**与 P2 正交且互补** —— 目标端一旦交错，P2 的 tile-broadcast 就能一趟覆盖整条记录而不只是 `hiddenBytes` | 中，改布局牵连 combine |
| **`proxyPe` 是同 local index 的静态映射**（167 行） | forwarder 恒定选「同一 local index 的那张卡」，倾斜时造成 NIC 热点。FUSCO 实测值 **8.7%（真实）/ 16.6%（倾斜）**，v1 现状逐字对应它的消融对照组 | 中，且与 rail 亲和性冲突，见「FUSCO」 |
| send 阶段 `nNodes−1` 卡死 warp 并行度 | N=2 时 8 个 warp 只有 1 个在发，见 S2 | 未测，可能不值钱 |

### 盲区：跨节点 AMO / doorbell 排队（本文全部基准都测不到）

本文所有实测都是**单节点**（XGMI + 本地内存），跨节点 NIC 站点一个都没覆盖。
而 PR [#630](https://github.com/ROCm/mori/pull/630) 的数据显示**这里才是同步开销的大头**：
光 282 行那一次 AMO 就值约 3.3 µs，比本文能测范围内找到的全部（约 1 µs）还多 3 倍。

| 项 | 量级 | 状态 |
|---|---|---|
| 282 行 `nodeRecvTokenNum` AMO | **~3.3 µs**（send 阶段的一段 barrier） | #630 已在 LL 路径做，非 LL 未做 |
| 226 行 put 融合的 signal | 未测 | 与 payload 同一个 WQE，大概拆不开 |
| 633 行 `ShmemQuietThread` | 未测 | 每节点 1 次，必需 |

**「同步层面只剩 1 µs」要限定在「我能测的范围内」**，覆盖这一档需要 2 节点环境。

### 根本约束

**`DispatchIntraNode`（XGMI 半边）**撞在**写带宽墙**上：最优 grid 约 **455 GB/s**（128×16），
当前配置只跑到 417 —— 差的 9% 就是 G3。到平台之后改拷贝循环的形状动不了它
（`Unroll` −0.1% / metadata 打包 +0.7% / load-first +1.3%）。**这个阶段确实没有更多可拿**；
再上一个数量级只有两条路：换传输（TDM / SDMA，即 gfx1250 那条路），或减少搬运量。

**这条只对 intra 成立，另外两个阶段都还没撞墙：**

- **send**：连 warp 都没用满（N=2 时 8 个里只有 1 个在发），瓶颈是 doorbell 排队，见 S2 / S3。
- **recv**：每个 token 串行拷 K 次、源被重复读 `K_d ≈ 3.3` 次且 uncached，
  而 grid 只有 2 wave/SIMD 盖不住那条 8 深的依赖链，见 R3 / P1 / P2。

**所以「G1 + G3 拿完就没了」是错的** —— 那句话曾经写在这里，来自「只看 intra」的视角。
recv 那两条的量级还没测，但它是唯一一个「已知有结构性浪费、且改动不牵连任务划分」的阶段。

### 前置条件：先把测量搞对

PR #625 的 `330edcb` / `a97ba1a` 记录了两件事，v1 同样适用：

- CCO/GDA 在 3 warmup + 10 rounds 下均值 run-to-run 摆动约 **20%** 且系统性偏高
  （shmem 第 3 轮就稳）。改用 20 warmup + 30 timed rounds。
- `dist.barrier()` 一次性放行所有 rank 造成 **thundering herd**，第一轮约 1.6 – 2.5× 稳态。
  **不是** warmup/cache 效应。用 `MORI_EP_DROP_ROUNDS` 排除这个爬坡。

**20% 的噪声会把 G1 那 3 – 10% 完全淹没。**

### 另有两条非性能项

- **60 / 448 行 scope 不一致**：写 peer 的 `dispTokIdToSrcTokId`，intra 侧用
  `AtomicStoreRelaxedSystem`，recv 侧是普通 store。ISA 上两者都不发缓存维护，是一致性问题。
- **`dispatchGridBarrier` / `interNodeBlocksBarrier` 命名误导**：语义是 last-one-out
  计数器，没有任何 block 在上面等待，容易被误读成需要全 block 驻留的真 barrier。

---

## 同步层级阶梯

共 34 个同步站点。另有两个 ISA 确认的全局事实：`s_barrier` = **0**、`s_sleep` = **0**。

| 层级 | 机制 / API | 站点 | 代价与用量 |
|---|---|---:|---|
| wave 内 | `__shfl` / `__ballot` / `__any` —— SIMD lane 交换，不访存 | 8 | 基本免费 |
| workgroup | `s_barrier`（`__syncthreads()`） | **0** | 整条路径一次都没用到（ISA 确认） |
| block 间 (agent) | L2/TCC 上的 agent-scope 原子计数器 | 10 | 全是 last-one-out 或 slot 分配，无人自旋 |
| 本地 · system scope | 对 `hipMalloc` 本地 buffer 用 SEQ_CST + SYSTEM / `__threadfence_system` | 4 | 语义强于需要 |
| 跨 GPU (XGMI) | symm uncached buffer 上的远程原子 / system-scope load-store | 6 | 含唯一一处等 peer 的自旋 |
| 跨节点 (NIC) | RDMA put 融合 AMO signal / 独立 AMO / `ShmemQuiet` | 5 | SYSTEM scope 必需，**非**过度同步 |
| kernel 边界 | runtime 隐式 grid barrier | 1 | dispatch 共 2 个 kernel |

---

## 逐阶段清单

标记：⚠ = scope 或自旋形态强于语义需要；🔥 = 路由相关的同步热点。

### Kernel 1 `EpDispatchCopyToStaging`（grid = multiProcessorCount）与 kernel 边界

| 位置 | 说明 |
|---|---|
| 658–701 | 无同步原语 —— 仅 `WarpCopy` + lane 0 普通 store。ISA 确认：668 条指令里 0 缓存维护、0 原子、0 `s_barrier` |
| `launch.cpp` 480–484 | kernel 边界（CopyToStaging → V1Kernel），runtime 隐式 grid barrier，**唯一一次真正的全 grid 同步**；不可能死锁 |

### `DispatchInterNodeSend`（RDMA blocks，`blockId < rdmaBlockNum`）

⚠ 下表 181 / 190 / 193 / 226 行的「次数」有个隐含前提：节点循环按 warpId 切分（165 行），
**只有 `warpId < nNodes` 的 warp 会执行它们**。N=2 时每 block 的 8 个 warp 只有 warp 1 干活，
而 274 行仍是全部 `R × W` 都要走。见 S2。

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 181 | `__ballot` / `__activemask` | wave 内 | 不访存 | 每 (warp, node, 64-token chunk) | dedup 同 destPe；`num == 0` 时整块跳过后面的原子 |
| 190 | `atomicAdd(blockFlagCounter + node, 1)` | block 间 | AGENT · 本地 | ≤ R × (N−1) × blockChunkNum | slot 分配器，**不是** barrier |
| 193–200 | `__shfl ×2`（replay 再 +1） | wave 内 | 不访存 | 同上 | 广播 `flagSlotId` / `flag` |
| 226 | `ShmemPutMemNbiSignalThread(..., AMO_ADD, proxyPe, qpId)` | 跨节点 | RDMA write + remote AMO | 每 chunk 内每段连续 sender 1 次 | 数据搬运与 flag signal 融合进同一个 WQE |
| 275 | `atomicAdd(interNodeBlocksBarrier[0], 1)` | block 间 | AGENT · 本地 | R × W | last-one-out，无人自旋 |
| 281 | `core::AtomicLoadRelaxed(blockFlagCounter + lane)` | block 间 | RELAXED + AGENT | 最后 1 warp × N lane | 整条 dispatch 路径唯一一处显式 agent-scope load |
| 282 | `ShmemAtomicTypeNonFetchThread<u64>(nodeRecvTokenNum, AMO_ADD)` | 跨节点 | remote AMO | 最后 1 warp × (N−1) lane | 通报本节点发送总量；未按 QP 展开（combine 侧会） |

### `DispatchInterNodeRecv`（RDMA blocks）

⚠ 下表 390 / 393 行的「次数」要乘一个隐藏系数：`k` / `node` 只由 `blockId` 推出，**`warpId` 没参与**，
所以一个 block 的 8 个 warp 轮询的是**同一个地址**。测试配置下总计 512 个 lane 打同一条 cacheline，见 R2。

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 390 ⚠ | `AtomicLoadRelaxedSystem(chunkFlag[node][k])` | 跨节点 | RELAXED + SYSTEM | 无界自旋，每圈 1 次 × **每个 warp** | NIC 写的 flag，SYSTEM 必需；循环**无退避** |
| 393 ⚠ | `AtomicLoadRelaxedSystem(nodeRecvTokenNum[node])` | 跨节点 | RELAXED + SYSTEM | 同一自旋内每圈 1 次 | 「该节点已发完」的终止条件 |
| 400–401 | `__shfl ×2` | wave 内 | 不访存 | 每 bid 迭代 | 广播 chunk token 数 / nodeFlag |
| 433 | `__any`（跨 lane dedup） | wave 内 | 不访存 | 每 (recv token, expert) | 与越界 destPe 守卫合并成一个 `shouldSkip` |
| 443 🔥 | `atomicAdd(dispTokOffsetMemObj->GetAs(destPe), 1)` | 跨 GPU | symm + uncached | 每个落本节点的 (recv token, expert) | 结果经 `__shfl` 后决定其后 4 次 `WarpCopy` 的地址 —— **全路径最热的同步点** |
| 448 | `dispTokIdToSrcTokId->GetAs(destPe)[i] = srcTokId` | 跨 GPU | 普通 remote store（无 fence） | 同上 | 与 60 行同一语义，那边却用了 SYSTEM 原子 store |
| 450 | `__shfl` | wave 内 | 不访存 | 同上 | 广播 `destTokId` |
| 479 | `atomicAdd(destPeTokenCounter + destPe, cnt)` | block 间 | AGENT · 本地 | 每 warp × G lane | `cnt == 0` 也无条件发。返回值未使用 —— ISA 确认已降级为 no-return |

### `DispatchIntraNode` + `DispatchIntraNodeBlock`（XGMI blocks，`blockId >= rdmaBlockNum`）

| 行 | API | 层级 | Scope | 次数 | 备注 |
|---:|---|---|---|---|---|
| 135 | `__any`（跨 lane dedup） | wave 内 | 不访存 | 每 (token, expert) | 哨兵 lane 用不可能的 destPe 避免误配 |
| 55 🔥 | `atomicAdd(dispTokOffsetMemObj->GetAs(destPe), 1)` | 跨 GPU | symm + uncached | 每 (token, 本节点 expert) | 与 443 行争用同一批 per-destPe 计数器 |
| 60 | `AtomicStoreRelaxedSystem(dispTokIdToSrcTokId->GetAs(destPe) + id, …)` | 跨 GPU | RELAXED + SYSTEM | 同上 | 与 448 行不一致。RELAXED 不产生缓存维护指令 |
| 64 | `__shfl` | wave 内 | 不访存 | 同上 | 广播 `destTokId` |
| 146 | `atomicAdd(destPeTokenCounter + destPe, cnt)` | block 间 | AGENT · 本地 | 每 warp × G lane | 同 479 行 |

### `DispatchSync`（所有 B 个 block）

| 行 | API | 层级 | Scope | 次数 | 生成的指令（gfx950 实测） |
|---:|---|---|---|---|---|
| 601 | `atomicAdd(dispatchGridBarrier, 1)` | block 间 | AGENT · 本地 | B × W | `global_atomic_add`，无缓存维护 |
| 602 | `__shfl` | wave 内 | 不访存 | B × W | — |
| 606 ⚠ | `AtomicLoadSeqCstSystem(destPeTokenCounter + destPe)` | 本地 · system | **SEQ_CST + SYSTEM** | 最后 warp × G lane | `1× buffer_inv sc0 sc1` + `2× s_waitcnt vmcnt(0)` |
| 608 | `AtomicStoreSeqCstSystem(peer recvTokenNum + myPe, n)` | 跨 GPU | SEQ_CST + SYSTEM | 最后 warp × G lane | 写 peer 的 symm buffer —— 这一处 SYSTEM **必需** |
| 611 | `__hip_atomic_store(dispatchGridBarrier, 0, RELAXED, AGENT)` | block 间 | RELAXED + AGENT | 1 | `global_store … sc1`，1 条指令 |
| 617 ⚠ | `ShmemInt32WaitUntilGreaterThan(signal, 0)` | 跨 GPU | 内部 RELAXED + SYSTEM 裸自旋 | 最后 warp × G lane | 循环体 = `global_load sc0 sc1` + `s_waitcnt vmcnt(0)`，无 `s_sleep` |
| 618 | `atomicAdd(totalRecvTokenNum, n)` | block 间 | AGENT · 本地 | G | `global_atomic_add`，无缓存维护 |
| 619 ⚠ | `__threadfence_system()` | 本地 · system | **SYSTEM 全屏障** | G lane | `1× buffer_wbl2 sc0 sc1` + `1× s_waitcnt` + `1× buffer_inv sc0 sc1` |
| 621 ⚠ | `AtomicStoreSeqCstSystem(signal, 0)` | 本地 · system | **SEQ_CST + SYSTEM** | G | `1× buffer_wbl2 sc0 sc1` + `1× s_waitcnt vmcnt(0)` |
| 622 ⚠ | `AtomicStoreSeqCstSystem(destPeTokenCounter + destPe, 0)` | 本地 · system | **SEQ_CST + SYSTEM** | G | 同上 |
| 627 | `atomicAdd(crossDeviceBarrierFlag, 1)` | block 间 | AGENT · 本地 | 1 | epoch++，combine 侧 barrier 目标值靠它 |
| 628 | `__hip_atomic_store(combineGridBarrier + 1, 0, RELAXED, AGENT)` | block 间 | RELAXED + AGENT | 1 | 替 combine 预清槽位 |
| 633–636 | `ShmemQuietThread(proxyPe)` | 跨节点 | NIC CQ 排空 | warp 0..N−1 各 1 次 → N | 保证前面的 RDMA put 真正落地 |

---

## 示例配置下的绝对次数

`B=96, R=64, W=8`（取自 `tests/python/ops/test_dispatch_combine_internode_v1.py:45-51`），
`N=2, G=8, T=4096, K=8` → `blockChunkNum = ceil(ceil(T/64)/R) = 1`。只列与路由无关的站点：

| 行 | 同步对象 | 公式 | 次数 / dispatch |
|---:|---|---|---:|
| 146 + 479 | `destPeTokenCounter` atomicAdd | `B × W × G` | **6,144** |
| 601 | `dispatchGridBarrier` atomicAdd | `B × W` | 768 |
| 275 | `interNodeBlocksBarrier` atomicAdd | `R × W` | 512（其中 **448 次来自从未进过节点循环的 warp**，见 S2） |
| 190 | `blockFlagCounter` atomicAdd | `R × (N−1) × blockChunkNum` | 64 |
| 226 | RDMA put + 融合 signal（下界） | 同上 | 64 |
| 606/608/621/622 | SEQ_CST + SYSTEM 原子 | `4 × G` | 32 |
| 619 | `__threadfence_system()` | `G` | 8 |
| 633 | `ShmemQuietThread` | `N` | 2 |
| 282 | 节点级 `nodeRecvTokenNum` AMO | `N − 1` | 1 |

`destPeTokenCounter` 的 6,144 次超过其他所有站点之和（且多数在加零），
但 ISA 显示它是一条不带缓存维护的 `global_atomic_add`，单次很便宜。

**唯一与路由相关、量级最大的一项不在表内**：55 / 443 行的 `dispTokOffsetMemObj`
跨 XGMI 远程原子，次数 = 落到本节点的 `(token, expert)` 对数，上界 `T × K` = 32,768。

---

## gfx950 ISA 实测

### 1. 决定代价的是 ordering，不是 scope

`tools/isa_probe/scope_probe.hip` 隔离每个构造：

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

**relaxed 无论 agent 还是 system 都只有一条指令，差别仅在 sc0 这一个位。**
所以 scope 从 system 降到 agent 几乎不省东西；真正省的是 ordering 从 seq_cst 降到 relaxed ——
那会让 `buffer_inv` / `buffer_wbl2` 连同全排空 `s_waitcnt` 一起消失。
编译器依据在 `SIMemoryLegalizer.cpp`：`expandLoad` 只在 ordering ∈ {Acquire, SeqCst}
时才调 `insertAcquire()`（发 `buffer_inv`），scope 只被 `enableLoadCacheBypass()` 用来选 sc 位。

### 2. 619–622 那段序列的实际展开

探针 `probe_fence_then_seqcst_store` 复现原样序列：

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
等价语义的 relaxed 版本（`probe_relaxed_store_pair`）是 **2 条指令**，两个 store 之间
连 `s_waitcnt` 都没有，可以重叠。编译器不会合并 fence 的回写与 store 自带的 release ——
LLVM 源码留着 TODO 明确说明这一点。

### 3. 缓存维护指令总量与归因

`EpDispatchInterNodeV1Kernel_bf16`（5502 条指令）：`buffer_wbl2 sc0 sc1` 21、
`buffer_inv sc0 sc1` 10、`buffer_inv sc1` 5、`buffer_wbl2 sc1` 3、`s_waitcnt vmcnt(0)` 192、
原子 18（其中 12 条带 sc0 = 返回值形式）、`s_barrier` **0**、`s_sleep` **0**。
`EpDispatchCopyToStaging_bf16`（668 条指令）：缓存维护 0、原子 0、`s_barrier` 0。

按源码行归因（`-gline-tables-only` + `.loc`）：

| 来源 | 指令 | 是否必需 |
|---|---|---|
| `amd_device_functions.h:695`（`__threadfence_system` 展开） | 9× `buffer_inv sc0 sc1` + 9× `buffer_wbl2 sc0 sc1` | 大部分来自 shmem RDMA 原语，必需 |
| `utils.hpp:185`（`AtomicStoreSeqCstSystem`） | 5× `buffer_wbl2 sc0 sc1` | 部分必需（写 peer） |
| `ionic_device_primitives.hpp:218/395/563` | 6× `buffer_wbl2 sc0 sc1` | **必需** —— NIC doorbell 必须对设备可见 |
| `utils.hpp:292/314/140/175` | 5× `buffer_inv sc1` + 3× `buffer_wbl2 sc1` + 1× `buffer_wbl2 sc0 sc1` | agent scope，shmem AMO 模拟路径 |
| `utils.hpp:145`（`AtomicLoadSeqCstSystem`） | 1× `buffer_inv sc0 sc1` | ← 就是 606 行 |

### 4. 消融归因：本文标记的 4 处占多少

`/tmp` overlay 里逐项降级后重新编译（仓库源码未动），只看主 kernel：

| 变体 | `inv sc0 sc1` | `inv sc1` | `wbl2 sc0 sc1` | `wbl2 sc1` | `vmcnt(0)` |
|---|---:|---:|---:|---:|---:|
| baseline | 10 | 5 | 21 | 3 | 192 |
| 删掉 619 的 fence | 9 | 5 | 20 | 3 | 191 |
| 606 → `AtomicLoadRelaxed` | 9 | 5 | 21 | 3 | 190 |
| 621+622 → `AtomicStoreRelaxed` | 10 | 5 | 19 | 3 | 190 |
| **四处全降级** | **8** | 5 | **18** | 3 | **187** |

这 4 处合计贡献 **2 条 system-scope 失效 + 3 条回写 + 5 次全排空**，
是 dispatch kernel 31 条 system-scope 缓存维护指令中的 5 条（约 16%，静态计数）。

**一条静态计数看不出的反向考虑**：这 4 处位于 `DispatchSync`，执行在 dispatch 最末尾、
所有 warp 已收敛的时刻，即并行度最低、最无事可重叠的位置；而 RDMA 路径的 fence
分布在有大量其他工作在飞的阶段。**单次**代价可能不同。

---

## 修正记录：10 处被推翻的判断

#1–6 被实测推翻，#7 被源码阅读推翻，#8–10 被 ISA 反汇编推翻。

1. **「未使用的 `atomicAdd` 返回值」作废。** 146 / 479 行 `int counter = atomicAdd(...)`
   里 `counter` 从未使用。探针实测 unused-return 与显式 no-return 生成**逐字节相同**的
   `global_atomic_add`。编译器已经处理了。
2. **「scope 用得过宽」归因错误，应说「ordering 用得过强」。** relaxed 无论 agent 还是
   system 都只有一条指令；省下来的全部来自 seq_cst → relaxed。
3. **「打穿全卡 L2」说重了。** `buffer_inv` 失效的是该 wave 所在 **XCD** 的 L2。
   MI350 是 8 XCD × 32 CU，连带损失约 32 个 CU，不是全部 256 个。
4. **「小 token 用 16 block 快 2.2×」是假的。** 当时用
   `pairs_per_warp = ceil(pairs / (blocks × warps))`，在 `pairs < blocks × warps` 时总工作量
   随 block 数增长（T=8 时 64 block 做了 16 block 的 4 倍工作）。改精确 grid-stride 后真实差距 3–10%。
5. **「大 token 下 block 数 32 → 256 带宽全平」也是假的**，同一个坏基准，工作量增长正好抵消
   真实收益。重测后膝点前有 +24%。它曾被「已测并否决」表和「根本约束」两处引用，均已撤销。
   **教训：一个基准被判定为某维度不可用后，要回头检查它之前产出的结论有没有被别处引用 —— 我漏了一轮。**
6. **`batched − no_reserve = +920 µs` 的「局部性效应」是假的。** 当时 `no_reserve` 的 slot
   用全局对索引取模，8 个 warp 写同一个 slot，反复命中同一批 cacheline，没搬运真实 footprint。
   改成不重叠连续派号后它和其余变体一样是 424 GB/s。
7. **「staging 存在是因为 RDMA 需要注册内存，若 `inpTokenBuf` 已注册即可直接发」是错的。**
   漏掉了 SoA → AoS 重打包这个理由，据此给出的「高，要动内存管理」风险评级也一并修正为「低到中」。
8. **「R3 大概率被 512 warp 在飞摊掉，不推荐改」是错的。** 那句话拿 B/C 节的结论去盖 R3，
   但 C 节比的是「原子怎么取 slot」，五个变体链深相同，根本不覆盖链深这个轴。
   ISA metadata 显示 `vgpr_count=90` 允许 5 wave/SIMD，**而 96 个 block 铺在 256 CU 上实际只有
   2 wave/SIMD** —— 延迟藏不住。R3 已升级为 P1，进「值得做」。
9. **「`assert` 在 release 下被 NDEBUG 剥掉」是错的**（C2 原来的理由）。AOT kernel 的编译命令
   从头到尾没有 `-DNDEBUG`，但产物里 `__assertfail` / `s_trap` 计数为 **0** ——
   这条路径上 `assert` **从来不产生陷阱**，与 NDEBUG 无关。结论（越界无守卫）不变，理由全错。
10. **「G1 + G3 拿完之后单节点侧没有更多可拿」是错的**（曾写在「根本约束」）。
    那是「只看 intra」的视角：intra 确实撞在 455 GB/s 的写带宽墙上，但 send 连 warp 都没用满，
    recv 有 `K_d ≈ 3.3` 倍的冗余 uncached 读。已改写并按阶段拆开。

---

## launch geometry：gfx950 实测（G1 / G2）

### 起因

PR #625 `5a7050d` 发现 cco 路径的 grid 是通过 launch redirect 从 `dispatch_combine.py`
的 shmem resolve 拿的，**不是 CU-aware**：80 CU 的 MI308X 上 `InterNodeV1LL` 的 AUTO 默认
256/128/8 是 CU 数的 3.2 倍。那个 resolve 就是 v1 自己的，所以问题对 v1 同样存在 ——
而 256 CU 的 gfx950 上测试配置的 96/64/8 只用了四分之一。他为 cco 建了
`internode_tuning_configs.py`（per-device × per-shape × per-token 查表，每 phase 独立
`rdma_block_num`，`lookup()` 把 `block_num` 夹到 ≤ CU 数）。**v1 的 AOT 路径没有等价物。**

### 扫描结果

`tools/isa_probe/grid_sweep.hip`，复刻 `DispatchInterNodeRecv` 的 payload 形状
（每对 token-expert：远程 `fetch_add` 取 slot → `__shfl` → payload `WarpCopy` 到 peer），
topk=8、hidden=7168 bf16、8 个 peer，按精确对数的 grid-stride。

| tokens | 64×8（当前） | 最优 | 差 |
|---:|---:|---:|---:|
| 8 | 7.1 µs | 6.9 µs @ 32×4 | 3% |
| 32 | 14.7 µs | 14.7 µs @ 192×4 | 0% |
| 128 | 42.3 µs | 39.0 µs @ 256×4 | 8% |
| 512 | 143.1 µs | 138.0 µs @ 192×8 | 3.6% |
| 4096 | 1159.2 µs | 1045.0 µs @ 128×16 | **10%** |

**哪个旋钮更重要，随 token 数换手**：小 token 看 warp 数（t=8 时 4 warp 比 16 warp 快约 25%，
7.0 vs 9.0 µs @ 64 block，而 block 16→256 基本不动）；大 token 看 block 数
（t=4096、warps=8 时 block 32→128 是 1321.2 → 1064.0 µs，**+24%**；同点上 warp 8→16 只有 +9%）。

block 数是「有膝点」而非「全平」：warps=8 时 32→96 陡升、96 以上趋平、256 反而回退
（1058.2 @192 → 1225.4 @256）；warps=16 时膝点更早，64 之后就平。
**当前 XGMI 半边只有 32 block，正好落在陡坡上** —— 这就是 G3。
当前固定的 8 warp 是个两头都差 3–10% 的折中。

### G2：dispatch 与 combine 必须一起调

没有独立复现，但 PR #625 `ddf9c9f` 有硬数据，机制对 v1 同样成立（两 phase 共享 QP 和
CUDA-graph replay）：

> tok4/8 的 dispatch 几何（32/16/6、64/16/4）单独调 dispatch 延迟是最优的，
> 但 `rdma_block_num=16` 的 dispatch 会让紧跟其后的 combine 慢约 18 µs
> （combine ~77 µs vs ~56 µs）。孤立的 dispatch 收益（~46 vs ~50 µs）被完全吃掉。

`de39b18` 又验一次：各 phase 独立求 argmin 的结果，在两 phase 用不同几何时**复现不出来**。
tok16 最后改成两 phase 共用 80/rdma40/warp4，比旧的 80/48/8 + 80/40/8 快约 4 µs。
**所以 G1 必须以「(dispatch 几何, combine 几何)」为一个整体作为搜索单元。**

### 保留意见

`grid_sweep` 是**单卡向 8 个 peer 写**，没有 RDMA、没有 flag 轮询、没有真实路由倾斜。
这些数字说明的是「geometry 值 3–10%」这个量级，真实最优点还受 `numRecvBlock = 8` 和
chunk 按 `blockId` 划分的影响，必须在真实 kernel 上重测。

**搜索空间不是连续的（C3）**：`rdma_block_num` **必须是 8 的倍数**，`warp_num_per_block`
**必须恰好是 8** —— 否则 recv 的任务划分会重复处理或漏掉 token，结果是**算错而不是变慢**。
推导见「recv 审查 / C3」。这两条把 G1 的可行域砍掉了一大块，尤其第二条直接封死了
「小 token 用 4 warp / 大 token 用 16 warp」那条路 —— 想动 warp 数就得先改 `numRecvBlock`。

**更要紧的一条**：它复刻的是 **recv** 的形状，本节「warp 8→16 有 +9%」之类的结论**只对 recv 成立**。
`DispatchInterNodeSend` 的节点循环按 warpId 切分，N=2 时 8 个 warp 只有 1 个干活（S2），
**给它加 warp 一点用都没有**。两个阶段跑在同一批 RDMA block 上却对 geometry 偏好相反 ——
G1 调参必须把 send / recv 分开归因，否则会把 send 的空转记到 recv 的账上。

---

## 二次测量：gfx950

`tools/isa_probe/sync_round2.hip`，独立程序，grid 用真实的 96 × 8。

### T1. `DispatchSync` 尾部（606 + 617–622）

`buffer_wbl2` 只为**脏行**付钱，而 dispatch 结束时 L2 是脏的，所以必须先把 L2 弄脏再测。
768 个 warp 各写一段，经 last-one-out 计数器汇合后由最后那个 warp 用 `wall_clock64` 计时。
`:608` 那个写 peer 的 store 在两个变体里都保持 seq_cst + system（它必需）。

| 先脏化的 L2 | seq_cst + system（现状） | relaxed + agent | 差 |
|---:|---:|---:|---:|
| 0 KB | 1.539 µs | 0.774 µs | 0.765 µs |
| 64 KB | 3.828 µs | 3.122 µs | 0.706 µs |
| 256 KB | 3.616 µs | 2.770 µs | 0.846 µs |
| 1024 KB | 3.876 µs | 2.914 µs | 0.962 µs |

**降级省 0.7 – 1.0 µs / 次 dispatch**，参照约 450 µs 的迭代是 0.2%。绝对量小，
但改动只有 4 行、语义无风险、ISA 已证明那对 `buffer_wbl2` 确实重复。
剩下约 2.9 µs 下不去 —— 那是 `:608` 必需的 peer store。

### T2. `destPeTokenCounter` 的守卫

768 warp × 8 lane = 6,144 次 agent-scope 原子：

| 非零比例 | 无守卫 | 加 `cnt != 0` | 差 |
|---|---:|---:|---:|
| 1/1 | 10.01 µs | 9.71 µs | 0.30 µs |
| 1/4 | 10.00 µs | 9.73 µs | 0.27 µs |
| 1/8 | 10.03 µs | 9.74 µs | 0.29 µs |
| 1/32 | 9.99 µs | 6.71 µs | 3.28 µs |

只在极度稀疏时才有意义，真实密度下省 0.3 µs。**否决。**

### T3. 自旋是否抢带宽

copier 在 stream A，spinner 在 stream B 并发，spinner 用 `wall_clock64` **按时间封顶**而非
按迭代数（按迭代数只会测出「自旋自己跑多久」，`s_sleep` 因自旋次数少反而显得更慢 —— 第一版就错在这）：

| spinner blocks | 无 spinner | 热自旋 | `s_sleep` 自旋 |
|---:|---:|---:|---:|
| 8 | 147.6 µs | 156.8 µs (+6.2%) | 155.0 µs (+5.0%) |
| 32 | 147.6 µs | 158.0 µs (+7.0%) | 157.9 µs (+6.9%) |
| 64 | 147.6 µs | 157.2 µs (+6.5%) | 157.1 µs (+6.4%) |

自旋 block 确实让并发拷贝慢约 6.5%，但**热自旋与 `s_sleep` 自旋几乎完全相同** ——
干扰来自 spinner 占着 CU，不是访存流量。**加 `s_sleep` 换不回带宽**，只值防 livelock 那份保险。

**适用范围（务必看）**：本测里 spinner 是**各转各的地址**，所以才能得出「瓶颈是 CU 占用而非访存」。
390 行那个自旋是 **512 个 lane 打同一条 uncached cacheline，而且 NIC 正要往那里写** ——
是完全不同的机制，本节结论**不覆盖**它。要判断那里要不要退避，得重新设计一个同址争用的基准。见 R2。

---

## intra 审查（45–148 行）与 intra 带宽实测

### C1. `weightsBuf` 空指针没有守卫 —— 潜在 crash

85–87 行无条件从 `args.weightsBuf` 读，而紧接着的 scales 却守卫了
（`if (args.scalesBuf && (scaleBytes > 0))`）。`WeightBytes()` 是无条件的
`K * sizeof(float)`（`dispatch_combine.hpp:191`），永不为 0，所以这次拷贝一定执行。

**空 weights 是显式支持的输入** —— `dispatch_combine.py:1229`：
`weight_ptr = weights.data_ptr() if weights is not None else 0`；
`DEF_COMMON_VARS` 也在判它（`combXferBytes = (args.weightsBuf == nullptr) ? …`）。
整个 kernel 家族里只有 v1 dispatch 不守卫：

| 位置 | 处理方式 |
|---|---|
| `internode.hpp:422`（v0） | `weightSize = args.weightsBuf ? K * sizeof(float) : 0`，长度算成 0 |
| `intranode_ll.hpp:293`、`intranode_1250x.hpp:902/924/…` | `if (args.weightsBuf)` 守卫 |
| v1 **combine**（756 / 854 / 909 / 1012 / 1165） | 守卫 |
| **v1 dispatch（85、685）** | **无守卫** |
| `low_latency_async.cpp:107 / 223` | 无守卫 |

`tokenId = 0` 时地址就是 0，必然 HSA page fault —— **崩溃而非静默错误**。目前是**潜在**的：
tests / examples / benchmark 里没有任何调用给 dispatch 传 `None`
（`test_dispatch_combine_internode.py:890` 那个 `dispatch_weights = None` 是给 combine 用的）。
属于「API 承诺了但 kernel 没兑现」。

### G3. XGMI block 数是余数，不是调出来的 —— 值 9%

```104:104:src/ops/dispatch_combine/internode_v1.cpp
  int xgmiBlockNum = blockNum - args.rdmaBlockNum;
```

intra 的 grid 不是独立参数。测试配置 96 / 64 → XGMI 侧只拿到 **32 个 block**。
`tools/isa_probe/intra_bw.hip` 复刻 `DispatchIntraNodeBlock` 的完整序列，T=4096、topk=8、hidden=7168 bf16：

| grid（XGMI block × warp） | Unroll=1 | Unroll=4 |
|---|---:|---:|
| **32×8 ← 当前** | **417.7 GB/s** | 416.9 |
| 64×8 | 441.3 | 441.0 |
| 128×8 | 451.0 | 451.2 |
| 64×16 | 450.0 | 449.6 |
| **128×16 ← 最优** | **455.2 GB/s** | 453.9 |
| 256×16 | 451.6 | 450.3 |

**32×8 → 128×16 是 +9%。** gfx950 有 256 CU 而 `blockNum=96` 只用了 37.5%；
要让 XGMI 侧拿到 128 block 同时保住 `rdmaBlockNum=64`，`blockNum` 提到 192 即可。

**这里没有取舍**：`grid_sweep` 里 recv 形状也是多 block 更好（64×8 = 1159 µs，128×8 = 1086 µs），
256 CU 的卡上提 `blockNum` 两半都受益。取舍只存在于 CU 紧张的卡上（MI308X 80 CU）——
那正是 PR #625 `5a7050d` 的情形，他要把 `block_num` **夹到** ≤ CU 数；这台卡是反过来，**给得太少**。

### 三处实现不一致，实测都不值钱

`core::WarpCopy` 的第二个模板参数是 **`Unroll`，不是向量宽度** —— `WarpCopyImpl` 内部恒用 16B
（`device_primitives.hpp:277`）。recv 路径写 `WarpCopy<uint8_t, 4>`，intra 用默认 `Unroll=1`，
访存并行度只有四分之一。但上表逐 grid 验过，**每一行 Unroll=1 与 4 都一样** ——
不是被 grid 限流掩盖，而是带宽墙下这个并行度已经够了。

| 项 | 现状 vs 改后 | 实测 |
|---|---|---:|
| payload `Unroll` | 1 → 4 | −0.1% |
| metadata | 3 次分开拷 → 打包 1 次 | +0.7% |
| 原子与 load 的顺序 | 先原子 → 先 load | +1.3% |

metadata 那条的机制：`K × 4 = 32 B` 低于一个向量步长，三次拷贝**全部落进 `WarpCopy` 的标量尾巴**，
64 lane 里只有 8 个在动、各存 4 B；字节占比只有 0.5%。
load-first 在 git 历史里有据 —— 被删的 LEGACY intranode kernel 注释写着
"LOAD first ... so the atomic's ~us cross-GPU round-trip overlaps the load"，v1 intra 没继承。

### 两处代码卫生问题（非性能）

- **`laneNode` 是死变量，且藏了负数除法的坑。** 123 / 129 行算出后全函数再没读过。
  哨兵 lane 的 `lanePe` 是负数（`-1 - laneId`），C++ 负数除正数向零截断，`-1/8 … -7/8` 全是 **0**
  —— 哨兵会看起来属于 node 0。recv 侧对同一件事是显式守卫的
  （424 行 `isSentinelSlot ? -1 : destPe / gpuPerNode`）。现在无害，但将来有人拿它做节点级 dedup 就会中招。
  130 行还有个多余的分号。
- **105 行在 `rdmaBlockNum == blockNum` 时除零**，只因调用点在 `else` 分支里到不了。

### 一处值得记的正面设计

`i = warpId + j * warpNum` 配 `tokenId = i / K`，在 `warpNum == K` 时（测试配置正是 8 == 8）
让**所有 warp 同时处在同一个 token 上** —— K 次 `inpTokenBuf` 读和 K 次 index 读都落在 L2 上，
读放大被吸收。代价是 warp w 恒定只处理 expert slot w，dedup 工作量按 slot 序号倾斜
（warp 0 恒不 dedup）。净下来是好的权衡，**别照直觉去「修」**。

**recv 侧完全没有这个结构** —— 它一 warp 一 token、内层串行走 K 个 expert，而且源 buffer
是 uncached、L2 吸收不了那 K 次重复读。R3 / P1 / P2 讲的就是这个不对称。

---

## send 审查（150–290 行）与 `CopyToStaging` 的去留

**本节全部结论来自源码阅读与算术，不是实测。**

### S2. N=2 时每个 RDMA block 的 8 个 warp 只有 1 个在发

```165:168:src/ops/dispatch_combine/internode_v1.cpp
  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
```

节点循环**按 warpId 切分**。测试配置 `worldSize=16, gpuPerNode=8` → `nNodes = 2`，而 `warpNum = 8`：
warp 0 拿到 `i=0 == myNode` 被 `continue`，warp 1 是**唯一干活的**，warp 2–7 的 `i >= nNodes`
连循环体都不进。即 send 阶段 512 个 warp 里**只有 64 个在发**，其余 87.5% 落空；
而 275 行的 `interNodeBlocksBarrier` 仍是全部 512 个都要 `atomicAdd`（其中 448 次是纯开销）。

三条推论：

1. **send 阶段加 `warpNumPerBlock` 完全无效**，warp 级并行度被 `nNodes − 1` 硬卡死。
   `rdmaBlockNum` 要给到 64，本质是在用 block 数补 warp 数补不上的并行度。
2. **与「launch geometry」的结论表面冲突、实则不矛盾**：那节扫的是 recv。
   **send 和 recv 跑在同一批 block 上却对 geometry 偏好相反**，G1 必须分开归因。
3. `nNodes` 越大问题越轻（N=8 时 warp 0–7 全有活）。**这是个「小规模部署才暴露」的问题**，
   2 节点恰好是最坏情况之一。

未测：把节点循环改成按 `(node, chunk)` 二维展开让 8 个 warp 都参与，值多少完全没量过。
考虑到 send 本来就不是带宽瓶颈，可能不值钱。

### staging 为什么在：不只是「RDMA 需要注册内存」

`XferBytesPerToken`（`dispatch_combine.hpp:203`）把**四个 stride 各不相同的源数组**拼成一条连续记录：

```
staging[tokenId]  ← 偏移 tokenId * 14404
┌──────────────────┬────────┬────────┬─────────┬──────────┐
│ hidden  14336 B  │ idx 32 │ wgt 32 │ scale 0 │ srcIdx 4 │
└──────────────────┴────────┴────────┴─────────┴──────────┘
   inpTokenBuf      tokenIndices weightsBuf  scalesBuf   现算
```

`xferBytes = 14336 + 32 + 32 + 4 + 0 = 14,404 B`，T=4096 时 staging = **56.3 MiB**。

注册只是理由之一；理由之二是**这个 SoA → AoS 重打包**：没有它，一个 WQE 只能带走 hidden，
indices / weights / srcIdx 各自还要再发一个 —— WQE 数 ×4。后者比注册难绕得多（见「修正记录」#7）。

两个此前没记下来的性质：

- **staging 按 `tokenId` 索引，是原序稠密数组，一次 permute 都没做。** 稀疏→稠密的压实是 RDMA
  的 scatter 干的（`remoteIdx = SendBufSlotOffset(...)`，222 行），不是 GPU 干的。
  所以 v1 本来就没有 FUSCO 要消除的那个 permute。
- **`EpDispatchCopyToStaging` 对全部 T 个 token 无条件执行**（669 行循环界是
  `curRankNumToken * warpsPerItem`），包括 8 个 expert 全落本节点、不需要跨节点的那些。
  测试配置下占比很小（`P(8 个全在本节点) = (8/16)^8 ≈ 0.4%`），但 `nNodes` 大、topk 小时会显著。

### 走一遍 token #1000

rank 3（node 0），8 个 expert 落在 destPe `{2,3,6,6,9,11,11,13}` —— node 0 占 4 个、node 1 占 4 个。

**Kernel 1**（grid = `multiProcessorCount` = 256，**与主 kernel 的 96 不是同一个几何**）：
4 次 `WarpCopy` 填满 `staging + 1000*14404`。

**⟶ kernel 边界（`launch.cpp:480-484`）⟵ NIC 在整个 kernel 1 期间完全空闲。**

**Kernel 2**：token 1000 属于 chunk 15，由 block 15 的 warp 1 处理 tokens 960–1023 对 node 1。
`__ballot` 得到约 63 个置位；设 lane 17 为空则切成两段连续 run，每段首 lane 发一个 WQE：

```
put(src = staging + 960*14404, len = 17*14404 = 244,868 B, → proxyPe 11)
put(src = staging + 978*14404, len = 46*14404 = 662,584 B, → proxyPe 11)
```

**staging 的原序稠密布局正是 run coalescing 能成立的前提** —— 源地址连续，17 个 token 才能
合成一个 WQE。这在考虑替换 staging 时是硬约束。

### fuse 与 ring 是两件事

| 想要的 | 靠什么拿到 | 代价 |
|---|---|---|
| **消除 NIC 空闲的 kernel 1 窗口**（= F1） | **就地融合**：4 次 `WarpCopy` 搬进 send 循环，仍写 `staging[tokenId]`，删掉独立 kernel 和 grid barrier | 低到中。布局不变，run coalescing 不变，顺带跳过本地-only token |
| 省 56 MiB footprint | 真 ring buffer | 高，且当前 T 下是负收益 |

**就地融合不需要额外加 fence**：ISA 第 3 节已把 6× `buffer_wbl2 sc0 sc1` 归因到
`ionic_device_primitives.hpp:218/395/563`，doorbell 路径本来就保证前面的写对设备可见。

一处要小心：一个 token 可能发往多个远端节点，而节点按 warpId 分给不同 warp（S2 那个循环）。
融合后同一个 `staging[tokenId]` 会被多个 warp 各 gather 一次 —— 内容相同所以幂等，但浪费。
**N=2 时不存在这个问题**；`nNodes > 2` 要专门处理。

**ring 的门槛**：省内存的条件是 `R × D × warpSize × xferBytes < T × xferBytes`
（`D`=在飞 slice 深度），即 `T > warpSize × R × D`。测试配置 `R=64`、双缓冲 `D=2` →
**T > 8192 才开始省**。而 T=4096 时 `blockChunkNum = 1`，**每个 RDMA block 恰好处理一个 chunk，
从头到尾没有任何复用**，ring 退化成 staging：双缓冲 ring = `64 × 2 × 64 × 14404` = **118 MB**，
比 staging 的 56 MiB **大一倍**。T=32768 才是 450 MB → 118 MB 的 4× win，
而 PR #630 那张端到端表跑的是 **4–32 tokens/rank**，那量级 ring 是纯亏。

ring 还会**新增一个当前不存在的同步点**：slice 回收必须等 CQ 确认，而今天整条 send 路径
只有末尾 633 行一次 `ShmemQuietThread`。参照 T3 —— 自旋 block 靠占 CU 就能让并发拷贝慢 6.5%。

**结论：ring 在当前 token 规模下不要走；要走的是就地融合。**

### S3. 226 行的 put 已经是 warp 合并的 —— 别再想这件事

`ShmemPutMemNbiSignalThread` 的 `Thread` 后缀含义是「**每个线程提供自己的 (src, dst, len) 描述符**」，
不是「每个线程独立 post」。`ShmemPutMemNbiSignalThreadKernelImpl`
（`shmem_ibgda_kernels.hpp:888`）从头到尾是 warp 协作的：

| 行 | 做的事 |
|---:|---|
| 912 | `activemask = __ballot(has_remaining)` —— 没调用的 lane 天然不在 mask 里 |
| 969 | `num_wqes = onlyOneSignal ? num_active_lanes + 1 : num_active_lanes * 2` |
| 981 | leader **一次** `atomicAdd(&wq->postIdx, num_wqes)`，整个 warp 的份 |
| 998 | `__shfl` 广播基址，每 lane `my_sq_counter = warp_sq_counter + my_logical_lane_id` |
| 1031 | K 个 lane **并行**各写自己那条 WQE |
| 1060 / 1079 | `should_signal = onlyOneSignal ? is_leader : true` —— 整个 warp **只发一条** signal WQE |
| 1104–1119 | leader **敲一次** doorbell，一次 `needConsIdx++`，一次 `dbTouchIdx` 更新 |

`onlyOneSignal` 默认 `true`（`shmem_device_kernels.hpp:132`），v1 用的正是默认值 ——
这也是必须的：`flag = num + 1` 是 warp-uniform 的，K 个 run 各发一次 AMO_ADD 会让接收侧
收到 `K × flag`，直接算错。

所以调用点的 K 个 run → **1 次 postIdx 原子 + K 条并行 WQE + 1 条 signal WQE + 1 次 doorbell**，
已经是最小 doorbell 数。

**换成 `ShmemPutMemNbiSignalWarp` 是倒退且会算错**：RDMA 路径的 Warp / Block 变体就是
`if (laneId == 0) ThreadImpl(...)`（`shmem_ibgda_kernels.hpp:1177-1181`），只有 lane 0 那个 run
会被发出去，其余 K−1 个直接丢失。它是给「整个 warp 参数一致」的调用点用的（如
`all_gather.hpp:397`），不是给 per-lane 描述符用的。

真正的串行点在 doorbell 排序（1104–1107 行）：leader 自旋等 `dbTouchIdx == warp_sq_counter`，
同一 QP 上严格串行 —— 就是 PR #630 那 3.3 µs 的来源。**doorbell 次数 = 调用 API 的 warp-次数**，
warp 内合并已经把它压到理论最小。剩下的合并空间只在「跨 chunk」（`blockChunkNum > 1` 时才有）
和「减少 run 分裂」（大 `nNodes` 下才明显，且 F1 顺手就能做）。

---

## recv 审查（361–481 行）

**证据分级**：R3 / C2 **已用 ISA 反汇编验证**（「复现」第 10 步）；
C3 / R1 / R2 是源码阅读与算术推导；P1 / P2 是据此提出的方案，**收益一个都没实测过**。

### `bid` 是一次三维展平 —— 最容易读反的地方

```380:383:src/ops/dispatch_combine/internode_v1.cpp
    int k = bid / (numRecvBlock * (nNodes - 1));
    int i = (bid / numRecvBlock) % (nNodes - 1);

    int node = (myNode + 1 + i) % nNodes;
```

`bid` 按混合基数解码，**低位是「8 个 block 分担同一个 chunk」，不是「不同 chunk」**：

```
bid = k × (numRecvBlock × (nNodes−1))  +  i × numRecvBlock  +  s
      └───── 第几个 chunk（高位）────┘    └ 第几个远端节点 ┘   └ 组内第几个 block（低位）┘
```

`i` 是**远端节点的相对序号**，`i ∈ [0, nNodes−1)`，不是节点号；`+1` 保证永远算不出 `myNode`，
`myNode` 偏移让不同 rank 从不同节点开始轮询。`nNodes = 2` 时 `i ≡ 0`，中位被压扁。

测试配置（`R=64, W=8, nNodes=2, maxChunkNum=64`）下：上界 512、步长 64，
`k = ⌊blockId/8⌋ + 8m`，每 block 跑 8 轮。**`⌊blockId/8⌋` 决定处理哪个 chunk，
`blockId%8` 决定在 chunk 内负责哪 8 个 token**：blocks 0–7 合力做 k=0，blocks 8–15 做 k=1……
内层 `j = startTokenIdx + (blockId%8)*8 + warpId` 步长 64，8 block × 8 warp = 64 warp
正好铺满一个 64-token chunk，**一 warp 一 token**。

### C3. 两条没写下来的硬约束，违反会静默算错

**一、`numRecvBlock × warpNum == warpSize`。** 上面那个「一 warp 一 token」靠的就是
`8 × 8 == 64`。`warpNumPerBlock` 一旦调成 16，`8 × 16 = 128 ≠ 64`，内层步长变成 128 而 chunk 只有
64 个 token —— 一半 warp 空转。**G1 想调 `warpNumPerBlock` 会直接撞上这条。**

**二、`rdmaBlockNum % numRecvBlock == 0`。** 内层循环用的是 `blockId % numRecvBlock`
而不是 `bid % numRecvBlock`（406 行）。两者相等的前提是 `rdmaBlockNum` 是 8 的倍数。
取 `rdmaBlockNum = 60` 时：

```
k=7 那一组的 8 个 bid 是 56…63
  bid 56,57,58,59 ← blockId 56,57,58,59 (m=0)   blockId%8 = 0,1,2,3
  bid 60,61,62,63 ← blockId  0, 1, 2, 3 (m=1)   blockId%8 = 0,1,2,3  ← 撞车
```

block 56 与 block 0 用同一个 `s`，**同一批 token 被处理两次**（两次 `fetch_add` 抢两个槽位、
拷两份），另外 4 段无人认领。症状是**输出重复 + 丢失，不是挂死**。

仓库里没有任何地方校验这两条。PR #625 的 tuning config 出现过的 `rdma_block_num` 是
16 / 40 / 48，碰巧都是 8 的倍数。**扫 geometry 时必须把 `rdma_block_num` 限制成 8 的倍数、
`warp_num_per_block` 固定为 8**，否则不是慢，是错。

### R1. 外层循环界是声明容量，不是实际 token 数

`maxChunkNum = CeilDiv(MaxNumTokensToSendPerRank(), warpSize)`，而那个函数返回的是
`maxNumInpTokenPerRank` —— **声明容量**。以 `cap = 4096`、实际 `T = 8`（decode）为例：

发送侧只发出 1 个 chunk（`flagSlotId = 0`），并置 `nodeRecvTokenNum[1] = 1×64 + 1 = 65`。
接收侧循环上界仍是 512，每 block 仍跑 8 轮：

| k | startTokenIdx | 出口 | 有效工作 |
|---:|---:|---|---|
| 0 | 0 | `chunkFlag[64] > 0` | ✅ 8 个 token |
| 1 | 64 | `64 >= nodeFlag−1 = 64` ✓ | ❌ 空 |
| 2…63 | 128…4032 | 同上 | ❌ 空 |

**512 次 bid 迭代里只有 8 次在干活**，其余 504 次每次仍付一次 `chunkFlag` + 一次
`nodeRecvTokenNum` 的 uncached SYSTEM 读。

这正是 PR #630 那条局限的**镜像**：容量声明宽松时 #630 一分钱不省，而 recv 在同一场景下白付。
`cap` 降到 128（测试用值）时 `maxChunkNum = 2`，上界 16，浪费自然消失。

最小改法：`nodeFlag` 一旦读到 > 0 就缓存进寄存器，后续 k 直接用 `startTokenIdx >= nodeFlag−1`
判断，**一次访存都不用**（`startTokenIdx = k×64` 对 k 单调）。想直接 `break` 要先核对 `i` 是否随
轮次变 —— `i = (⌊b/8⌋ + m) mod (nNodes−1)`，`nNodes−1 = 2` 时恒定，一般 `nNodes` 不成立。

### R2. 390 行的自旋：512 个 lane 打同一条 cacheline

`k` / `node` / `startTokenIdx` 全部只由 `bid` ← `blockId` 推出，**`warpId` 完全没参与** ——
一个 block 的 8 个 warp 算出完全相同的 `k`，它们的 lane 0 轮询**同一个地址**：

| 层级 | 倍数 | 累计 |
|---|---:|---:|
| 每 warp 的 lane 0 | 1 | 1 |
| 一个 block 的 8 个 warp | ×8 | 8 |
| 共享同一 chunk 的 8 个 block | ×8 | **64 个 lane 盯同一个 `uint64`** |
| 同时推进的 8 个 k | ×8 | **512 个 lane** |

而这 8 个 k 的地址是 `chunkFlag[64+0] … chunkFlag[64+7]` —— 8 个相邻 uint64 = **64 字节，一条 cacheline**。
buffer 是 `hipDeviceMallocUncached` 的（`dispatch_combine.cpp:563`），每次 load 都真走 fabric，
L2 一点都吸收不了。而 lane 1–63 在 `if (laneId == 0)` 外发散空等到 `__shfl` 才汇合 ——
**wave slot 占着，只有 1/64 的 lane 在做事**。

**T3「`s_sleep` 换不回带宽」这条不能照搬到这里**：T3 的 spinner 是各转各的地址，
结论是「干扰来自占 CU 而非访存流量」；这里是 512 lane 打同一个字，而且 NIC 正要往那条线上写。
两种机制不同，T3 没有覆盖。

三个改法：

| 方案 | 效果 | 代价 |
|---|---|---|
| 只让 warp 0 转，`__syncthreads()` 广播 | 512 → 64 lane | 引入**整条 dispatch 路径上的第一个 `s_barrier`**（现在是 0 个）。安全性没问题 —— bid 循环次数只依赖 `blockId`，8 个 warp 一致，不会挂 |
| 加退避 | 未知 | 按上面的理由值得单独量一次，别引 T3 |
| 减少共享同一 flag 的 block 数 | —— | 就是 `numRecvBlock`，牵连整个划分且受 C3 约束，不划算 |

**但先别急着改**：自旋多久完全未知。`Slot::DispatchInterNodeRecv` 是整段计时的，没拆出等待部分。
**先在自旋段单独开一个 profiler slot（或 lane 0 上夹 `wall_clock64`），能测到再谈改。**

### R3. 每个 token 串行拷 K 次，源还完全相同 —— ISA 确认

**这一条已用 gfx950 ISA 验证过**（ROCm 7.1.1，命令见「复现」第 10 步），不再是源码推测。

两边循环形状是反的：

| | 循环在什么上 | 远程原子依赖链深度 | 源 buffer |
|---|---|---:|---|
| intra（110 行） | `i < tokens × K`，**(token, expert) 对** | 1（K 个 expert 分给 K 个 warp，并发） | `inpTokenBuf`，普通 cached |
| recv（406 + 421 行） | 外层只循环 token，内层 `for (e = 0; e < K; e++)` | **K = 8** | `dispatchInp`，**uncached** |

**intra 两个条件都是好的，recv 两个都反着。**

#### ISA 证据

```
.LBB6_350:      ; Parent Loop BB6_330 Depth=1      ← bid 循环
                ;   Parent Loop BB6_340 Depth=2    ← j (token) 循环
                ; =>  This Loop Header: Depth=3    ← e (expert) 循环
                ;      Child Loop BB6_376 Depth 4  ┐ 12 个 depth-4 的
                ;      ...                          ┘ WarpCopy 字节循环
```

| 检查项 | 结果 |
|---|---|
| **e-loop 体内的原子指令** | **1** 条 `flat_atomic_add … sc0` ← 决定性证据 |
| `if (laneId == 0)` 那个谓词的 `.loc` 全 kernel 出现次数 | **1**（旁证：分支只生成一次） |
| 回边 | BB6_349 排在 header 之前、fall-through 回去（rotated layout） |

`numExpertPerToken` 是运行时值、体内还有 `continue`，**编译器一点都没展开**。

注意 `atomicAdd` 本身的 `.loc` 归到了被内联的 `amd_hip_atomic.h:218`，不是 `internode_v1.cpp`
的行 —— 所以判「没展开」要数**指令**，不能数 `.loc`。

#### 真实依赖链是 3 跳，不是 1 跳

```asm
global_load_dwordx2 v[50:51], v1, s[34:35] offset:16   ; 取 SymmMemObj 的 peerPtrs 基址
s_waitcnt vmcnt(0)
flat_load_dwordx2   v[50:51], v[50:51]                 ; 取 peerPtrs[destPe]
s_waitcnt vmcnt(0) lgkmcnt(0)
flat_atomic_add     v50, v[50:51], v65 sc0             ; 真正的跨 XGMI 原子
s_waitcnt vmcnt(0)                                     ; 等返回值
```

前两跳是 `GetAs<index_t*>(destPe)` 展开的指针解析（本地，大概率 L2 命中），第三跳才是远程原子。

**而且 `vmcnt(0)` 是全排空**：gfx9 系没有独立的 `vscnt`，store 也计入 `vmcnt` 且计数器按发射顺序
递减，所以第 e+1 轮的第一个 `s_waitcnt vmcnt(0)` **会把第 e 轮 WarpCopy 还在飞的 store 一并等完**。
相邻两轮之间**没有任何重叠** —— 不是「原子延迟被下一轮拷贝盖住」，而是严格的
`原子 → 等 → 拷 → 全排空 → 下一轮`。原子之后到循环尾还有 23 个 `s_waitcnt vmcnt(0)`。

#### 源被重复读 K_d 次，而且 uncached

`stagingPtr + tokIdx * xferBytes` **不含 `e`** —— K 次拷贝的源地址完全相同，只有目标不同。
而 `dispatchInp` 是 `hipDeviceMallocUncached`（`dispatch_combine.cpp:349`），
**读不走 L2，每次都真打到 HBM/fabric**。

设 `K_d` = 去重后落本节点的目标卡数。N=2/G=8/K=8、路由均匀时
`K_d = 8(1−(7/8)^4) ≈ 3.3`（落本节点的 expert 期望 4 个，从 8 张卡里去重）：

| | 现在 | 理想 | |
|---|---:|---:|---|
| uncached 读 | 3.3 × 14336 = **47.3 KB** | 14.3 KB | **3.3× 冗余** |
| 跨 XGMI 写 | 3.3 × 14336 = 47.3 KB | 47.3 KB | 固有，省不掉 |
| 合计 / token | **94.6 KB** | 61.6 KB | **−35%** |

4096 个收到的 token 就是 388 MB → 252 MB。

#### 为什么现在藏不住：grid 太小，不是寄存器不够

`vgpr_count = 90`、`sgpr_count = 106`（ISA metadata）。gfx950 每 SIMD 512 VGPR →
寄存器允许 5 wave/SIMD。**但整个 grid 只有 96 个 block 铺在 256 个 CU 上** ——
一个 CU 一个 block、8 个 wave、4 个 SIMD → **实际只有 2 wave/SIMD**。

所以那条 8 深的依赖链**没有别的 wave 可以拿来盖**。这也说明 G3（提 `blockNum`）和本条是
**互补**的：G3 给更多 wave 来盖延迟，P1 减少要盖的延迟。

#### 「已测并否决」里那条不覆盖这里

B 节测到边际成本 1.67 ns/对、C 节五种取 slot 方案全在 ±3% 内 —— 但 C 节比的是
**「原子怎么取 slot」（per-token vs batched，条数差 43 倍）**，五个变体的**链深度都一样**。
本条说的是**链深度**和**冗余读**，是正交的两个轴，C 节没有覆盖。

### P1 / P2：两个可以合成一次的改动

两者都发生在**单个 token 的处理体内部**，不动 `bid` 三维展平、不动 `numRecvBlock`、
不动任务划分 —— **和 G1 / G3 正交，也不碰 C3 的两条硬约束**。

#### P1. K 个原子并行发（链深 8 → 1）

dedup 可以纯用 wave op 一次算完，不访存：

```cpp
uint64_t dupMask = 0;
for (int e = 0; e < numExpertPerToken; e++) {
  int pe_e = __shfl(lanePe, e);
  dupMask |= __ballot((laneId > e) && (lanePe == pe_e));   // 纯 ALU
}
bool mine = (laneId < numExpertPerToken);
bool skip = !mine || (lanePe < 0) || (lanePe >= worldSize)
         || (lanePe / gpuPerNode != myNode) || ((dupMask >> laneId) & 1);

int      myDestTokId = -1;
uint8_t* myOutBase   = nullptr;
if (!skip) {
  myOutBase   = args.interNodeV1TokBufs.dispatchOut->GetAs<uint8_t*>(lanePe);  // 2 跳，K 路并行
  myDestTokId = atomicAdd(args.dispTokOffsetMemObj->GetAs<index_t*>(lanePe), 1);
}
// ← 到这里只有一次 s_waitcnt，覆盖全部 K 条原子
if (myDestTokId >= config.MaxNumTokensToRecv()) myDestTokId = -1;   // 顺手堵掉 C2
```

**语义等价性**：原来是 `__any((laneId < e) && (destPe == lanePe))`，问「有没有序号更小的 lane
撞同一个 pe」；新版对每个 `e` 标记 `l > e` 且相同的 lane 再 OR —— lane `l` 被标记
⟺ `∃ e < l` 撞车，**完全一致**。哨兵拿的是互不相同的 `-1 − laneId`，不会自相匹配，
且被 `lanePe < 0` 先筛掉。

**为什么能并行**：K 条原子打的是 K 个**不同**的 destPe（去重保证互异），
彼此本来就没有顺序依赖。

#### P2. tile-broadcast（消除 K_d× 冗余读）

把「按目标外层、按字节内层」翻过来：

```cpp
const uint8_t* src = stagingPtr + tokIdx * xferBytes;
for (size_t off = 0; off < hiddenBytes; off += 1024) {        // 64 lane × 16 B
  uint4 v = load16(src + off + laneId * 16);                  // ← 唯一一次 uncached 读
  for (int e = 0; e < numExpertPerToken; e++) {
    int d = __shfl(myDestTokId, e);   if (d < 0) continue;
    uint8_t* base = __shfl(myOutBase, e);                     // P1 已解析好
    store16(base + (size_t)d * hiddenBytes + off + laneId * 16, v);
  }
}
```

代价只有 **4 个 VGPR**（每 lane 16 B）。当前 `vgpr_count=90`，离 5 wave/SIMD 的 102 上限
还有余量，而且现在是 grid 限制（2 wave/SIMD），寄存器根本不吃紧。

**必须在 P1 里把 K 个目标基址一并解析好并广播**，否则内层 14 × K 次重做 2 跳指针解析会比
省下来的还贵。K 个基址是 warp-uniform 的，落 SGPR（8 × 2 = 16 个，`sgpr_count=106` 有余量）。

`hiddenBytes = 14336`、`14336 / 1024 = 14` 整除，但通用实现要留标量尾巴。
metadata 那三段（32 + 32 + 4 B）保持原样即可 —— 打包只值 +0.7%。

#### 要先核对的一件事

P1 让槽位按 **lane 序**而不是 **e 序**分配，`destTokId` 的具体取值会变（每个 destPe 的相对顺序不变，
但跨 destPe 的交错变了）。下游全部经 `interNodeDispDestTokIdMap` 反查，理应无影响 ——
**但要确认 combine 侧没有任何地方假设 `destTokId` 的单调性或分组连续性**。这是唯一的正确性风险。

#### 一条更激进但不建议先做的

把 `dispatchInp` 改成普通 cached 内存，recv 观测到 `chunkFlag` 后做一次 invalidate，
让 K 次读全部命中 L2 —— 比 P2 简单得多。但 `buffer_inv sc0 sc1` 会**打掉整个 XCD 的 L2**
（「修正记录」#3：约 32 个 CU 连带受损），每 chunk 每 block 来一次大概率得不偿失。
RDMA 注册只要求 pin、不要求 uncached，所以技术上可行，风险在一致性不在可行性。

### C2. `destTokId` 溢出没有守卫（原因不是 NDEBUG）

425–431 行有一段注释讲这个失败模式：

> HSA-RCA Signature 1 guard: in Release builds NDEBUG strips the assert at :387, so an
> out-of-range expert id ... yields `destPe >= worldSize` and an OOB GetAs/WarpCopy/atomicAdd
> -> HSA page fault.

他们据此给 `destPe` 加了 `peOutOfRange` 守卫。但**往下 13 行**：

```444:445:src/ops/dispatch_combine/internode_v1.cpp
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
```

同一套逻辑，紧接着就拿 `destTokId` 做远程写，**却没有对应守卫**。
而 `maxTotalRecvTokens` 是用户可设的（`dispatch_combine.hpp:141`），设小了就是静默 OOB。

**ISA 查出来的实际情况比那条注释说的更糟**：

1. AOT kernel 的编译命令里**根本没有 `-DNDEBUG`**（`src/ops/CMakeLists.txt:79-80`，
   `_extra_defs` 只加 `ENABLE_PROFILER` / `ENABLE_STANDARD_MOE_ADAPT`）；
2. 但编译产物里 **`__assertfail` 和 `s_trap` 一条都没有**（全 kernel grep 计数 0）。

assert 的条件被降级成了一个 exec mask，而且**只罩住 446–449 行**（两次映射表写入），
payload 拷贝在 `s_or_b64 exec` 恢复之后：

```
444 行 assert 的条件  → v_cmp + s_and_saveexec
446–447 行 interNodeDispDestTokIdMap 写入   ┐ 被罩住
448 行    dispTokIdToSrcTokId 写入           ┘
s_or_b64 exec, exec, s[48:49]                ← 恢复
457 行    WarpCopy → dispatchOut[destPe] + destTokId*hiddenBytes   ← 不在保护内
```

所以越界时**两次小的映射写入反而被意外挡住了，而 14 KB 的 payload 照写不误**。
`assert` 在这条路径上**从来不产生陷阱**，与 NDEBUG 无关。（为什么不生成 `__assertfail` 未深究。）

修法见 P1 里那一行 `if (myDestTokId >= MaxNumTokensToRecv()) myDestTokId = -1;`，零额外成本。

### 一处无害但糊涂的账

`chunkFlag` 按 `MaxNumTokensToSendPerRank()`（4096）分配（`dispatch_combine.cpp:561-563`），
但索引只用到 `maxChunkNum`（= 4096/64 = 64）—— **多分配了 64 倍**，cap=4096 时 64 KB 只用 1 KB。
量小无害，越界也不会发生（索引 stride 小于分配 stride），但对不上。

---

## PR #630：跳掉 282 / 353 行那次跨节点 AMO

PR [#630](https://github.com/ROCm/mori/pull/630)（isytwu，draft）优化的正是 282 行（非 LL）/
353 行（LL）—— 本文唯一无法实测的一类。

**思路**：send 尾部在 RDMA put 之后再发一次跨节点 AMO 公布「我发了多少 chunk」
（`nodeRecvTokenNum`）。接收侧用它做两件事：轮询时区分「这个 slot 永远不会被 signal」和
「它还没到」；以及告诉 combine 迭代到哪。**当每个 slot 都已被 put 自带的 signal 覆盖时，
这两件事都不需要它。** 不变量 `chunksSent <= maxChunkNum` 恒成立（两边是同一个上取整），
所以**相等即全覆盖**，轮询必然在 `chunkFlag` 上终止，那条「永不到达」的捷径成了死代码。
核对过 slot 分配：`flagSlotId` 来自共享的 `atomicAdd(blockFlagCounter + node, 1)`，
slot 号连续占满 `0..chunksSent−1`，推理成立。

**数据**：设备侧计时器测出那段 barrier segment 从 540 降到 208 tick，约 **3.3 µs**
（100 MHz wall clock，1 tick = 10 ns，差 332 tick 正好对上）。
端到端（2×8 MI300X，AUTO，num-qp 1，hidden 6144，3 次取中位，dispatch µs）：

| tokens/rank | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| before | 49.53 | 47.99 | 49.83 | 52.18 |
| after | 47.29 | 47.66 | 48.89 | 51.75 |

作者自己标注：方向一致但幅度小，只有 tok16 / tok32 的 per-run 区间不重叠。combine 不变，符合预期。

**3.3 µs 的来源（本文推断，PR 未言明）**：`ShmemAtomicTypeNonFetchThread` 是 non-fetch、
fire-and-forget 的，一次 WQE 不该要 3.3 µs。但这个 AMO 由**最后一个 warp** 发出
（`(finishedWarp + 1) == rdmaBlockNum * warpNum` 分支），此时其他所有 warp 的 put 已在
doorbell 链上占位，所以它的 `ringDoorbellOrdered` 要排在全部 put 之后等自己的号。
**3.3 µs 是 doorbell 排队延迟，不是 AMO 本身。** 这与 PR #625 分析
`ringDoorbellOrdered` / `dbTouchIdx` 死锁时是同一处链路。

**会限制它落地的条件**：触发条件是 `chunksSent == maxChunkNum`，而分母
`maxNumInpTokenPerRank`（`dispatch_combine.hpp:154`）是**声明的容量**，不是实时 token 数。
benchmark 通常把它设得接近实际 token 数所以条件总成立；但生产里它是宽松上界
（struct 默认 128，实际更大），decode 步只有 8 个 token 时 `chunksSent = 1` 而
`maxChunkNum = 64` —— **条件不成立，零收益**。那张表的 0.4–2.2 µs 能否搬到生产，
取决于容量声明有多贴。

**非 LL 路径也适用，而且更简单**：

| | 消费方式 | 需要的 combine 改动 |
|---|---|---|
| **LL** combine（1113 行） | `nodeCount` 直接当外层循环界，读到 0 就 `continue` 跳过整个节点 | **必须**加静态 `maxChunkNum` 回退 —— PR 做了 |
| **非 LL** combine（970 行） | 只在 `chunkFlag == 0` 分支里查，且有 `nodeFlag > 0` 守卫；外层循环界本来就是静态 `maxChunkNum` | **不需要** —— AMO 不发时守卫为假，chunk 自然被跳过 |

**风险**：recv 的轮询（389–398 行）是只有两个出口的无界 `while (1)`。守卫一旦判错，
症状是**挂死**而不是算错。不变量核对下来成立，但这是 hang 类改动，值得比常规 perf 改动更小心。

---

## FUSCO：forwarder GPU 的选法是静态的

论文 [FUSCO (arXiv:2512.22036)](https://arxiv.org/abs/2512.22036)（Infinigence AI + 清华，2025-12-26）
提三个机制，其中两个 v1 已经有了，第三个正好是 v1 的空白。

### 前两个：v1 已经在做，所以论文的大头数字对 v1 不成立

| FUSCO 机制 | v1 的对应物 |
|---|---|
| **dComm** —— segment descriptor 驱动的融合拷贝，消除 all-to-all 前后各一次全量 permute | v1 dispatch 是远程 `fetch_add` 取 slot → `__shfl` → `WarpCopy` **直接写进 peer 的最终位置**（55–64 / 443–450 行），从来没有独立的 permute kernel |
| **Planner** —— 每个远端节点只指定一个 forwarder GPU，同 token 命中该节点多个 expert 只跨网络发一份 | 167 行的 `proxyPe` 就是 forwarder；181 行 `__ballot` dedup 同 destPe；recv 侧经 XGMI 分发（443 行） |

论文关掉 dComm 的 −27% ~ −33% 是打在 **NCCL + PyTorch `index_select`** 基线上的
（它 Table 1 量到重排占 shuffle 总时间：节点内 68.8%、跨节点 25%），v1 结构上不付这笔钱。
相应地 FUSCO 对 **DeepEP** 的真实流量优势只有 **1.13 – 1.34×**；那个 3.84× 是构造场景 ——
一个 token 的所有 expert 恰好落同一节点。
**引用这篇论文时别拿 3.84× / 30% 说事，对 v1 有意义的只有第三个机制。**

### 第三个：Online Load Balancer —— v1 的现状正好是它的消融对照组

```167:167:src/ops/dispatch_combine/internode_v1.cpp
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
```

即「节点 `i` 上**同一 local index** 的那张卡」。论文描述 Balancer-off 时的原话是
"a commonly used static placement scheme that **clusters GPUs with the same local index
across nodes**" —— 逐字对应。

FUSCO 定义 **communication group**（每节点各出一个 GPU，组内互为 forwarding endpoint），
目标 minimize max group load，贪心三步：① 每节点按跨节点发送量 `L` 降序排得 `P_n`；
② 节点 `n` 把 `P_n` 循环右移 `n` 位得 `S_n`；③ `group_i` = 各节点 `S_n` 的第 `i` 位。
移位量取节点号，保证各节点的最高负载卡落进**不同**的组。`O(M log M)`，M=8 时可忽略。

消融数据（seqlen 16k，EP=64，hidden 7168，topk 8 —— 与 v1 测试配置同一形状族）：
真实流量 **−8.7%**、跨节点负载倾斜（双峰）**−16.6%**、所有 expert 同节点 −3.2%。

### 三个会挡住它落地的问题

1. **与 rail 亲和性直接冲突。** v1 的 `proxyPe` 是 **same-rail** 的，
   `dispatch_combine.cpp:153-156` 明确说 V1/V1LL 靠这一点才能在 `MORI_ENABLE_RAIL_ONLY` 下工作
   —— rail-only 模式下跨 rail 的 QP **根本不存在**，改了直接挂在无连接的 QP 上。
   而贪心置换必然打破 same-local-index，在 rail-optimized 网络上要多走一跳 spine。
   **论文全文没出现过 "rail" 这个词**，它默认跨节点任意 GPU 对之间带宽均匀。
   所以搬过来必须先选一边：在 rail 约束内做置换（搜索空间从 `O((M!)^N)` 塌缩到几乎没有），
   还是接受多一跳换 NIC 负载均衡。**这个取舍论文没有回答。**
2. **「无需跨节点协调」站不住（本文推断，论文未言明）。** 论文称该算法
   "can be executed fully locally without creating any centralized bottleneck"。但节点 A 上位于
   `S_A[i]` 的发送方要知道自己在节点 B 的对端是谁，就得知道 `S_B[i]`，而 `S_B` 取决于节点 B
   各卡的负载。所以要么先交换各节点排列，要么先 allgather 路由元数据 —— 都是一次额外的跨节点小消息。
   可能能搭在 v1 已有的 token 计数交换上，但需要核对，不是白拿的。
3. **只在倾斜时有收益。** single-node 那行只有 3.2% 已经说明负载均匀时这套机制什么都不产生。
   要不要做，取决于目标负载的倾斜程度 —— 这是个先量一量 per-GPU 跨节点发送量分布的问题，
   而不是先改代码。

### 改动面与不适用的部分

`proxyPe` 有 **10 个定义点**：dispatch 侧 167 / 279 / 306 / 634 行，combine 侧
1033 / 1073 / 1084 / 1185 / 1217 / 1230 行。两个 phase 必须用**同一个**映射，
否则 combine 找不到 dispatch 的对端 —— 与 G2 是同一类耦合。

FUSCO 的 Planner 是 **1000 行 Python/PyTorch**，每个 MoE 层跑一次。论文自己承认 4k seqlen 时
优势缩水（固定开销与输入规模无关），而**全文只评测 training 和 prefill TTFT，没有一个 decode 数字**。
decode 步只有几十个 token 时这份 host 侧规划开销大概率吃掉全部收益。
v1 的路由决策在 kernel 里 on-device 做，低延迟场景下反而更合适。
**所以要搬的只是 balancer 那个 `O(M log M)` 的选组逻辑（可放 device 上做），不是整套 planner。**

---

## per-token 远程原子：gfx950 实测

`tools/isa_probe/atomic_latency.hip`。counter 按真实布局摆放：`dispTokOffsetMemObj` 是每个 PE
各自 `MallocSymm(sizeof(index_t), hipDeviceMallocUncached)`，所以每个 destPe 的 counter
在**它自己那张 GPU** 上、彼此相隔 4 KB（把 8 个 counter 挤在一个 peer 的同一条 cacheline 上，
会凭空造出约 8 倍争用外加 false sharing —— 第一版就踩了这个坑）。

### A. 依赖链延迟（warp 拿到 `destTokId` 前真正停的时间）

lane 0 连发原子，每次地址依赖上一次返回值，强制每轮 `s_waitcnt`：

| 目标 | scope | 延迟 / 次 |
|---|---|---:|
| 本地 `hipMalloc` | AGENT | 0.21 – 0.27 µs |
| 本地 `hipMalloc` | SYSTEM | 0.25 µs |
| 本地 `hipDeviceMallocUncached` | SYSTEM | 0.34 µs |
| **远程 peer `hipDeviceMallocUncached`**（55 / 443 行原样） | SYSTEM | **0.40 – 0.64 µs** |

**对比 gfx1250 的 ~3.5 µs**（本仓库 git 历史有据：`6a4d74756^` 版 `intranode.hpp` 的
`MORI_DISP_TIMING` 注释 "If this ~matches the asm-barrier number, it proves the ~3.5us
completion latency"）。即 gfx950 上这条往返比 gfx1250 快 6–9 倍 —— 当年逼出 batch 改造的
压力在这台机器上弱得多，但没有消失。另外 uncached 本身要花 0.34/0.25 ≈ 1.35 倍，跨卡再叠一层。

### B. 孤立聚合开销（**已被 C 节推翻，仅作参考**）

只测原子、不带 payload，96 × 8 grid、npes=8、T=4096：

| topk | (token,expert) 对 | per-token | batched | 省下 |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 9.82 µs | 4.39 µs | 5.4 µs |
| 4 | 16,384 | 30.54 µs | 7.23 µs | 23.3 µs |
| **8** | **32,768** | **57.72 µs** | **10.18 µs** | **47.5 µs** |

远程原子条数从 33,024 降到 768（43 倍）。per-token 一列对 topk 完全线性，边际成本
`(57.72 − 9.82) / (32768 − 4096)` = **1.67 ns 每对** —— 单次延迟 0.4 µs 而边际只有 1.67 ns，
说明 768 个 warp 在飞已经把延迟摊掉了，**问题是吞吐/争用，不是暴露的延迟**。
这个 47.5 µs 是「暴露部分」的**上界**，下一节把 payload 加回来后它基本消失。

### C. 把 payload 放回去之后：**不值得改**

`tools/isa_probe/reserve_vs_payload.hip`：三种取 slot 的方案搬运**逐字节相同**的负载
（hidden 14336 B + meta 80 B，写到 peer 显存），grid 用真实的 `rdmaBlockNum=64 × 8 warps`，
唯一差别是 slot 怎么来的。T=4096, hidden=7168 bf16，单卡向 8 个 peer 写：

| 变体 | topk=4 | topk=8 |
|---|---:|---:|
| 1. no_reserve（连续 slot，零原子） | 569.0 µs / 414 GB/s | 1113.0 µs / 424 GB/s |
| 2. per_token（当前 internode_v1） | 549.0 µs / 429 GB/s | 1100.1 µs / 429 GB/s |
| 3. batched（`intranode.hpp` 方案） | 558.0 µs / 422 GB/s | 1128.1 µs / 419 GB/s |
| 4. batched + per-token 空转原子 | 566.7 µs / 416 GB/s | 1139.2 µs / 415 GB/s |
| 5. batched，warp 连续派号 | 539.6 µs / 437 GB/s | 1097.4 µs / 430 GB/s |

分解（topk=8）：`per_token − batched` = **−28.1 µs**（直接 A/B，**batched 反而略慢**）；
`dummy − batched` = **+11.1 µs**（同等局部性下 per-token 原子的净成本 ≈ **1%**）；
`batched − warpcontig` = +30.7 µs（派号**顺序**约 3%）；`warpcontig − no_reserve` = −15.5 µs（噪声）。

**五个变体全部落在 1100 µs / 420–430 GB/s 的 ±3% 以内。** 取 slot 的方案换成什么都改变不了它 ——
512 个 warp 在飞，把 0.4 µs 的原子往返完全藏住了。

**结论：这个移植不值得做。** batched 省下的远程往返次数（43 倍）换不回它自己 Phase 1
全量计数 + 多一次 `__syncthreads()` 的开销。这也解释了与 gfx1250 的差异：那边原子要 3.5 µs
且 payload 走 TDM 更快，原子是**暴露**的；gfx950 上原子只有 0.4 µs 而 payload 撞在带宽墙上，
原子被完全遮蔽。

**两处口径提醒**：

1. 这里的 420–430 GB/s 是在 **64×8** 当前配置下测的，而 G3 表明最优 grid（128×16）能到
   **455 GB/s**。所以这个数字不是带宽墙本身，是「当前 grid 下的平台」。
   本节是同 grid 下的 A/B，结论不受影响；但别拿 420–430 当天花板引用。
2. **「原子被完全遮蔽」只对本节比较的那个轴成立。** 五个变体差的是「原子怎么取 slot」
   （条数、派号顺序），**链深度全都一样**。所以本节**不能**用来否定 P1（把 recv 里 K 条串行
   原子改成并行发）或 P2（消除 K_d 倍冗余 uncached 读）—— 那是另外两个轴。
   曾经拿这一节去盖 R3，见「修正记录」#8。

### D. 若将来换到原子更慢的 arch 仍要移植

`intranode.hpp:116-216` 是直接模板。两个要点：recv 侧循环是 `(node, chunk, recvBlock)`
而非平坦 `Npair`，「计数域」要重新定义；以及 `intranode.hpp:167-173` 那条不变量 ——
Phase 1 与 Phase 3 必须用完全相同的循环边界、步长和过滤条件，否则路由静默错乱。

---

## 复现

```bash
export ROCM=/shared/apps/ubuntu/opt/rocm-10.0.0
export PATH=$ROCM/bin:$ROCM/llvm/bin:$PATH

# 1. 隔离探针 —— 确认 ordering/scope 到指令的映射
cd tools/isa_probe
hipcc -O2 --offload-arch=gfx950 --cuda-device-only -S -o scope_probe.gfx950.s scope_probe.hip

# 2. 真实 v1 kernel（编译参数照抄 src/ops/CMakeLists.txt:75-85）
cd <repo root>
hipcc --genco --offload-arch=gfx950 -std=c++17 -O2 -S -gline-tables-only \
  -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS -DMORI_DEVICE_NIC_IONIC \
  -I./include -I. -I./3rdparty/spdlog/include \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  src/ops/kernels/ep_internode_v1.hip -o /tmp/isa/v1_lines.s

# 3. 按 kernel 统计
grep -nE 'buffer_inv|buffer_wbl2|s_sleep|s_barrier' /tmp/isa/v1_lines.s

# 4. 消融归因 —— 用 overlay 目录覆盖，不动仓库源码
#    mkdir -p /tmp/ov/src/ops/dispatch_combine
#    cp src/ops/dispatch_combine/internode_v1.cpp /tmp/ov/src/ops/dispatch_combine/
#    sed -i '619s|__threadfence_system();|/*ablate*/;|' /tmp/ov/src/.../internode_v1.cpp
#    然后在第 2 步命令里把 -I/tmp/ov 放在 -I. 之前

cd tools/isa_probe
# 5. per-token 原子（>=2 张空闲 GPU）
hipcc -O2 --offload-arch=gfx950 -o atomic_latency atomic_latency.hip
for k in 1 2 4 8; do ./atomic_latency 0 1 4096 $k; done   # src_gpu dst_gpu tokens topk

# 6. 把 payload 放回去后的 A/B（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o reserve_vs_payload reserve_vs_payload.hip
for k in 1 2 4 8; do ./reserve_vs_payload 4096 $k 7168; done

# 7. 二次测量：DispatchSync 尾部 / counter 守卫 / 自旋干扰（单卡）
hipcc -O2 --offload-arch=gfx950 -o sync_round2 sync_round2.hip && ./sync_round2

# 8. launch geometry 扫描（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o grid_sweep grid_sweep.hip
./grid_sweep 8 7168                  # topk hidden

# 9. intra 带宽：Unroll / metadata 打包 / load-first / XGMI block 分配（8 张 GPU）
hipcc -O2 --offload-arch=gfx950 -o intra_bw intra_bw.hip
for cfg in "32 8" "64 8" "128 8" "64 16" "128 16" "256 16"; do
  ./intra_bw 4096 8 7168 $cfg        # tokens topk hidden [xgmi_blocks] [warps]
done
```

`-gline-tables-only` 会把汇编从 155k 行放大到 288k 行，但 `.loc` 让每条缓存维护指令
都能归因到源码行，这是整套分析里最关键的一步。

**一个用错就会得出假结论的坑**：`reserve_vs_payload.hip` 的循环用的是
`pairs_per_warp = ceil(pairs / (blocks × warps))`，在 `pairs < blocks × warps` 时
**总工作量随 block 数增长**。它对三个变体的 A/B 仍然有效（工作量相同），
但**不能用它扫 block 数**（见「修正记录」#4/#5）。扫 grid 用 `grid_sweep.hip` 或
`intra_bw.hip`，两者都是按精确对数的 grid-stride。

### 10. 验证 recv 的 e-loop 是否串行（R3 / C2 / P1 的依据）

不需要 GPU，只要能编译。**注意这一步是在 ROCm 7.1.1 上做的**，而本文其余 ISA 数字来自 10.0.0 ——
循环结构和依赖链这类结论跨版本大概率一致，但严格说是另一个工具链的观测。

```bash
# 用第 2 步同样的命令产出 v1_lines.s，然后：
sed -n '/^EpDispatchInterNodeV1Kernel_bf16:/,/^\s*\.size\s*EpDispatchInterNodeV1Kernel_bf16/p' \
    v1_lines.s > recv_kernel.s

# a) 循环层次：确认 e-loop 是 Depth=3 的真循环、没被展开
grep -n "This Loop Header\|Header=BB6_350" recv_kernel.s | head

# b) 原子只发一条 → 没展开
grep -c "loc	2 443" recv_kernel.s          # 期望 1
grep -cE "flat_atomic|global_atomic" recv_kernel.s

# c) 依赖链：原子前后各一次全排空
grep -B6 -A4 "flat_atomic_add.*sc0" recv_kernel.s

# d) assert 是否真的产生陷阱（C2）
grep -cE "__assertfail|s_trap" recv_kernel.s   # 实测 0

# e) 占用率：寄存器 vs grid 谁是瓶颈
grep -E "vgpr_count|sgpr_count" v1_lines.s | head -2   # 实测 90 / 106
```

`vgpr_count=90` → 512/90 = 5 wave/SIMD 是**寄存器**上限；而 `blockNum=96` 铺在 256 CU 上
每 SIMD 只有 **2 wave**，所以真正的限制是 grid 太小。这是 P1 值得做的核心理由。

## 参考

- PR [#630](https://github.com/ROCm/mori/pull/630) —— 跳掉 282/353 行那次跨节点 AMO，
  本文唯一测不到的那一档的实测数据来源。
- PR [#625](https://github.com/ROCm/mori/pull/625) —— v1 internode 移植到 CCO/GDA。
  本文引用的 geometry（`5a7050d`）、dispatch↔combine 耦合（`ddf9c9f`、`de39b18`）、
  launch 批量化（`e21c3d3`）、测量方法（`330edcb`、`a97ba1a`）都来自这个 PR 的 `jhchouuu` 部分。
  注意那些改动落在 **cco / v2 路径**（`ep_internode_kernel.hpp`），不在本文分析的 AOT v1 路径上。
- 论文 [FUSCO (arXiv:2512.22036)](https://arxiv.org/abs/2512.22036) —— MoE 通信库，三个机制里
  dComm / Planner 与 v1 现有结构重合，只有 Online Load Balancer 是 v1 的空白。
- `llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp` —— gfx940/950 分支决定了 ordering / scope
  到缓存维护指令的映射，是「ISA 实测」的上游依据。
- `docs/rdma_bandwidth_utilization.md` —— v1 的阶段划分图与 RDMA 带宽核算方法。
  注意其中提到的 `analyze_trace_internode.py` 并不存在，实际叫
  `tools/profiler/analyze_ep_kernel_trace.py`。
