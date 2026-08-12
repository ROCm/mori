# MORI EP intranode dispatch/combine 挂起问题排查报告

**环境**：ROCm 7.14 / gfx950（MI355X ×8，XGMI P2P）
**现象**：`tests/python/ops/test_dispatch_combine_intranode.py` 整套跑一遍，约 1/6 的概率挂死。ROCm 7.2.4 下不复现。
**当前状态**：**未修复。** 已用 rocgdb 抓到挂起现场，锁定卡死位置在 dispatch 收尾的信号握手。本轮据此改了跨设备屏障的语义，但**改动生效后问题依旧复现**。

同时查明一件影响全局的事：此前所有内核级实验的改动**从未进入过编译产物**，容器 JIT 编译的是 site-packages 里的另一份源码拷贝（详见 §6.1）。因此旧版报告里的验证数据全部作废，"三处缺陷修补无效"这一结论也不成立——它们根本没被测过。

---

## 1. 结论摘要

1. **挂起是一个对称的多 rank 死锁，不是某个 rank 抛异常后的下游后果。** rocgdb 在挂起现场抓到 8 个 rank 的 GPU kernel **全部仍然驻留**，没有任何 rank 的 Python 侧抛过异常。
2. **卡死位置已精确定位**：dispatch 收尾段的**接收方信号等待**（`ShmemInt32WaitUntilGreaterThan(signal, 0)`），而不是原先以为的 combine 跨设备屏障。
3. **现场形态是 4/4 分裂**：4 个 rank 已经走完 MORI 的 dispatch/combine 进到 NCCL 集合操作，另外 4 个仍卡在 dispatch 里等前 4 个的信号，而读到的信号值是 0。
4. **最可能的根因**：跨迭代的轮次隔离失效——本轮的 `dispTokOffset` 复位与下一轮发送方的远程分配发生重叠，导致多个发送方拿到**同一个槽位**、后写覆盖先写。这与更早抓到的正确性失败特征（见 §3）完全吻合。
5. **轮次隔离唯一的保障就是 combine 的跨设备屏障**，而该屏障此前被改成了宽松比较，不再具备真正的屏障语义（见 §5.1）。

---

## 2. 挂起现场（rocgdb 硬证据）

用自动探测脚手架在 pytest 输出停滞 90 秒后触发 rocgdb 抓取，命中于第 2 轮，测试用例 `test_dispatch_combine_ll[none-False-8-32-1-4-0-4096-data_type2-8]`。

| rank | 现场 |
|---|---|
| 0、1、5、6 | 已在 `ncclDevKernel`（`runTreeSplit` / `readLL` / `barrier`）——MORI 部分已走完 |
| 2、3、4、7 | 卡在 `EpDispatchIntraNodeLLKernel_fp8_ocp()` 地址 `0x218c0` |

卡住 PC 处的反汇编是 `flat_load_dword` + `s_waitcnt` + 比较 + 回跳，与信号等待循环一致。按 exec 掩码逐 lane 读寄存器（lane 号即 peer 号，`$v2` 为读到的信号值）：

```
agent 3  exec=0x03   等待 peer 0,1          读到 0
agent 4  exec=0x47   等待 peer 0,1,2,3,6,7  读到 0
agent 5  exec=0xcf   等待 peer 0,1,2,3,6,7  读到 0
agent 8  exec=0x47   等待 peer 0,1,2,3,6,7  读到 0
```

**关键矛盾**：卡住的 rank 在等 0、1、5、6 的信号，而这 4 个 rank 早已走完整个 dispatch——它们的发送循环排在接收循环之前，所以信号**一定写出去过**。收方却读到 0。要么写丢了，要么被谁清零了。

同时，8 个 worker 的 `MORI_WORKER_TRACE` 里**没有任何 Python 异常**。这直接推翻了旧版报告"挂起是正确性失败的下游"的说法。

---

## 3. 正确性失败的损坏特征（早期硬证据，仍然成立）

在 `check_dispatch_result` 里加详细诊断后抓到的典型现场：

```
rank 5: 1 of 2 received tokens decode to a source slot that was never sent.
  recv_num_token=2   send_stride=8
  tokens_sent_per_rank=[0, 1, 0, 0, 1, 0, 0, 1]
  src_token_pos=[56, 0]
  slots[0:18] (past recv_num_token)=[56, 0, 0, 0, 0, 0, ...]
  expected_recv=2 (per source rank [0, 0, 0, 0, 1, 0, 0, 1])
```

`expected_recv` 是在 host 端**从路由索引独立重算**的应收数，三次独立抓取中与 `recv_num_token` 每次都吻合。读法：计数正确；槽位 0 = 56 合法；槽位 1 应为 rank 4 的条目却是零初始值，**这一格从未被写入**；越界槽位全零。

最有价值的一次：同一轮里 rank 5 和 rank 7 同时失败，**丢失的条目全部来自 rank 4，且同时丢向两个不同的 peer**。单条 store 的偶发竞态不会产生这种模式，更像 rank 4 在两个 peer 上都撞进了别人已占用的槽位。

---

## 4. 已排除的假设

| 假设 | 排除依据 |
|---|---|
| 挂起是某 rank 抛异常后其余 rank 空等 | rocgdb 抓到 8 个 kernel 全部驻留；8 份 worker trace 无任何异常 |
| 接收计数虚高（多算了 token） | `expected_recv` 独立重算，三次均与 `recv_num_token` 吻合 |
| 槽位分配基址未复位（从 1 开始） | 越界槽位全为零，没有有效条目落在计数范围之外 |
| 槽位分配的 `atomicAdd` 是 agent scope，跨 GPU 不保证原子 | 写了 8 卡 × 4096 次的硬件探针，agent 与 system 两种 scope **均零重复**，跑三轮全过 |
| 7.14 与 7.2.4 的编译器把 `atomicAdd` 降级得不一样 | 对比两版 ISA，输出完全相同（agent 为 `sc0`，system 为 `sc0 sc1`）。差异在时序而非代码生成 |
| CCO 竞技场未清零 | `MORI_EP_COMM` 未设置，走 shmem 路径；且 `MallocSymm` 每次分配都 `hipMemset` |
| 屏障槽位跨 handle 残留旧值 | 同上，`MallocSymm` 逐次清零，且 `crossDeviceBarrierFlag` 每个 handle 从 1 重新开始 |
| combine 屏障前有提前 return | 全文件唯一的 `return` 在屏障**之后**（`curRankNumToken == 0`） |
| host 端跳过了某个 rank 的 kernel 启动 | `dispatch_combine.cpp` 中无任何按 token 数跳过启动的分支 |

### 排查过程中踩到的坑（避免重走）

- **改仓库源码不等于改了跑的内核**：JIT 编译的是 site-packages 里的 `_jit-sources` 拷贝，详见 §6.1。这是本次排查中代价最大的一个坑，它让之前所有内核级实验的数据全部失效。
- **设备端 printf 看门狗无效**：printf 只在 kernel 正常结束时刷出，挂死的 kernel 永远不刷，看门狗必然沉默；且插桩本身扰动时序，装上后 11 轮一次都不复现。
- **`MORI_WORKER_TRACE` 路径必须在容器内先建好**：`tests/python/utils.py` 的 `_trace` 把 `open()` 写在了 try 之外，路径不存在会静默打死全部 8 个 worker，现象和真挂死一模一样。
- **改完 kernel 源码的第一轮会重编所有 JIT 变体**（4 分钟以上），极易被停滞检测器误判为挂起。

---

## 5. 修复

改动位于 `src/ops/dispatch_combine/` 下的 `intranode.hpp`、`intranode_ll.hpp`。

**前提**：`EpDispatchIntraNodeKernel`（非 LL）与 `EpDispatchIntraNodeLLKernel`（LL）共用同一个 combine —— `intranode.hpp` 的 `EpCombineIntraNodeKernel`（见 `src/ops/kernels/ep_intranode.hip`）。`intranode_ll.hpp` 里只有 dispatch，本身不含任何跨设备同步。所以下面对屏障的改动对两条路径同时生效。

### 5.1 跨设备屏障恢复真正的屏障语义（本轮核心改动）

轮次隔离的完整链条是这样的：

- 接收方在自己 dispatch 的末尾把 `dispTokOffset` 清零；
- 发送方在自己下一轮 dispatch 的开头，用远程原子在**对端**的 `dispTokOffset` 上分配槽位；
- 两者之间唯一的顺序保证，就是 combine 里的跨设备屏障——它必须挡住所有 rank，直到最慢的那个也走完上一轮 dispatch。

此前为了消除 ABA，把屏障的等待条件从严格相等改成了单调比较（`peer_generation >= mine`）。**这一步把屏障降级成了单向的"对端至少和我一样新"检测**：代数落后的 rank 会立即通过，没有任何人为它这一轮到达过。轮次隔离随之失效，发送方就能在接收方清零之前抢先分配，于是多个发送方拿到同一个槽位——正是 §3 的损坏特征。

本轮改为**按代数奇偶分两组槽位（parity banking）+ 恢复严格相等**：

```cpp
const int barrierBank = static_cast<int>(crossDeviceBarrierFlag & 1) * args.config.worldSize;
...
while (core::AtomicLoadRelaxedSystem(localBarrierPtr + barrierBank + thdId) !=
       crossDeviceBarrierFlag) {
  __builtin_amdgcn_s_sleep(1);
}
```

一组槽位隔一代才复用，因此残留值恒为 `当前代数 - 2`，永远不可能等于当前代数——原来促使我们放弃严格相等的 ABA 消失了，而严格相等重新保证"所有 peer 都到齐了这一代"。槽位数组本身容量足够（`InitializeBarrier` 分配的是 `worldSize × 8` 个 `uint64_t`，只需要 `2 × worldSize`），不需要改分配。

### 5.2 `dispTokOffset` 复位改为相干 store

复位原本是普通赋值，而所有发送方是用远程原子在这个地址上分配的。裸 store 与远程原子路径混用不保证相干，改为 `core::AtomicStoreRelaxedSystem`。两条路径（`intranode.hpp`、`intranode_ll.hpp`）各一处。

### 5.3 早前已提交、仍然保留的两处修补

- **release fence 只覆盖了 block 0**：dispatch 的 Phase 3 中每个 block 都在对 peer 做裸 P2P 写，但 system-scope release fence 只由 block 0 的 warp 0 执行，fence 只约束执行它的那个 wave。改为每个 block 上报到达之前先由每个线程 `__threadfence_system()`。
- **元数据裸 store**：`dispTokIdToSrcTokId` 的远程写改为系统作用域相干 store，与代码库信号路径的既有做法一致。

这两处缺陷本身成立。旧版报告称它们"无法消除挂起"，该结论不成立——受 §6.1 的问题影响，它们从未真正被编译进去测过。

---

## 6. 验证数据

### 6.1 先说一个把此前全部数据作废的发现

容器里 `import mori` 解析到的是 **`/opt/venv/lib/python3.12/site-packages/mori`**，而 JIT 编译与缓存哈希用的是它自带的一份**源码拷贝** `mori/_jit-sources`（见 `python/mori/jit/config.py` 的 `get_mori_source_root()` 与 `cache.py` 的 `_hash_tree()`）。仓库 `src/` 里改什么都不会进入 GPU 上跑的代码，而且因为 content hash 也不动，**旧内核会被静默复用，连重编都不会发生**。

发现经过：改完源码后第一轮只用了 57 秒（和热缓存一样快），本该有的重编没出现；查缓存目录，只有一份 02:50 的旧产物，没有新的 content_hash 目录。把三个文件同步进 `_jit-sources` 后立刻生成了新目录 `be815b644571`，确认改动这才真正生效。

后果：

- **§6.2 表中原有的全部配置跑的其实是同一个二进制**（`_jit-sources` 停留在 8 月 11 日 13:52 那份拷贝）。"没有任何配置能和基线区分开"不是统计结论，是它们本来就是同一份代码。这些数字已全部删除。
- `[MORI SLOTCHK]` 探针从未被编译进去，所以此前每轮 `slotchk=0` 不代表槽位没有重复。
- `ci_repro.sh` 已加入同步步骤，每轮运行前把三个 device 源码推进 `_jit-sources`，防止再次跑偏。

**任何后续实验，第一件事都是确认 `_jit-sources` 与仓库一致、并且出现了新的 content_hash 目录。**

### 6.2 同步之后的真实数据

每轮 = 完整跑一遍 `test_dispatch_combine_intranode.py`（335 passed / 640 skipped，约 56 秒）。挂起判定：`timeout 360` 触发。

| 配置 | 挂起/总轮次 |
|---|---|
| 严格相等 + 奇偶分组 + fence + 相干 store | 验证中 |

已知的单点观察：同步后第一次跑（`-x -k 'data_type0 and 4096'` 子集）在约 55% 处出现 FAILED 并随即挂死，GPU 保持满载。**所以本轮修复没有消除问题。**

判定标准：需要连续 25 轮无挂起才谈得上有效——若真实基线仍是 17% 量级，25 轮全过的概率约 1%。

### 6.3 测量环境的噪声

宿主上有一个每隔几分钟重启一次的 `torchrun --nnodes=2 --node_rank=1 ... test_dispatch_combine_internode.py --cmd stress`，在等 `10.2.80.22` 的主节点、每次 600 秒超时退出（疑似本机的 nightly CI）。它会周期性占用 GPU，测量时需要考虑这一噪声源。

---

## 7. 若本轮修复无效，下一步

证据仍然收敛到**多个发送方拿到了同一个槽位**。要直接证实，需要看发送方自己认为写到了哪个槽位：

1. 在 `ComputeTokenRoute`（LL，`intranode_ll.hpp`）和 Phase 2（非 LL，`intranode.hpp`）里，把 `atomicAdd` 返回的槽位号连同 `(源 rank, 目标 rank, token)` 写进一个调试缓冲区。拿到各 rank 的分配序列后，重复槽位会一眼可见。
2. 现有的 `[MORI SLOTCHK]` 探针（两个 dispatch 收尾各一处）已经在做弱化版的检查：比较本轮实际发出的槽位数与各 peer 声称发来的 token 数，不等就打印。它只在 kernel 能正常结束时才刷得出来，所以对挂死的那一轮无效，但对**先于挂起发生的正确性失败**有效。
3. 若确认重复，重点仍是 `dispTokOffset` 的复位时序；彻底的做法是给它也做按轮次的双缓冲，而不是依赖屏障提供的外部顺序。

---

## 8. 留下的调试设施

- **`MORI_WORKER_TRACE=<前缀>`**（`tests/python/utils.py`）：每个 rank 在异常发生的**当下**把完整 traceback 写入 `<前缀>_<rank>.log`，不依赖 pytest 的 FAILURES 汇总，挂起时仍能拿到现场。**使用前务必确认该目录在容器内存在**（见 §4 的坑）。
- **`check_dispatch_result` 增强诊断**：打印接收计数、host 端独立重算的期望值与逐源 rank 分解、越界槽位内容、每 rank 发送数、非法解码槽位清单。
- **`[MORI SLOTCHK]` 设备端探针**：dispatch 收尾比对已分配槽位数与已接收 token 数。
- **`assert_worker_results` 加超时与去重校验**：某个 rank 失联时明确报出缺哪几个 rank，而不是无限阻塞成裸超时。
- **测试框架异常推迟**：`run_test_once` 把校验异常暂存，等 `combine()` 和所有同步点走完再抛。
- **`ci_repro.sh`**：按 nightly 的 "MORI-EP (intranode)" 步骤逐字复现（同样的 `timeout 360` 与 `-v`），逐轮报告挂起与 SLOTCHK 计数。
- **`hang_capture.sh`**：监控 pytest 输出，停滞超阈值即自动用 rocgdb 抓取各 rank 的 wave 现场、exec 掩码与寄存器。§2 的证据即由它取得。

---

## 9. 改动文件清单

```
src/ops/dispatch_combine/intranode.hpp          屏障奇偶分组+严格相等、每 block fence、相干 store、SLOTCHK 探针
src/ops/dispatch_combine/intranode_ll.hpp       每 block fence、相干 store、信号收发 fence、SLOTCHK 探针
src/ops/dispatch_combine/intranode_1250x.hpp    每 block fence、相干 store（gfx1250 变体，未在本机验证；
                                                §5.1 的屏障改动尚未同步过去）
tests/python/ops/dispatch_combine_test_utils.py 诊断、异常推迟、结果超时
tests/python/utils.py                           per-rank 异常 trace
```

内核为纯 JIT 路径（`intranode_entry.hpp` 包含），源码哈希变化会自动触发重编，无需重建 `.so`。
