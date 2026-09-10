# InterNode V1 Dispatch：优化顺序与验证计划

**先修正确性并建立可信基线；第一个性能实验只调 B/R/W。** 后续保留六项局部优化，统一按“优化 1～6”引用。每项先单独验证，只组合有稳定收益的版本。

本文只覆盖 `KernelType::InterNodeV1`，保留现有 kernel 边界、buffer 布局和同 rail 的 proxy 映射。源码基准为 `25e64aa8`；下文行号均按这一版本，不再混用旧行号。文档整理日期：2026-09-10。

证据范围：已有源码检查、CPU 索引枚举，以及此前 ROCm 7.1.1 的真实 kernel 离线编译/ISA 记录；**尚无本轮优化的两节点 GPU 正确性或性能结果**。本文描述的是实验计划，不代表改动已经落地。

## 先做哪个

| 顺序 | 优化 | 何时值得试 | 第一版范围 |
|---|---|---|---|
| **1，先做** | [调 B/R/W](#opt-1) | 所有目标负载 | 使用现有参数接口，不改算法 |
| **2，随后做** | [调 recv 分片数 S](#opt-2) | 尤其是 W 较大、每 chunk 有空闲 wave 时 | 只改 dispatch 的局部任务划分 |
| **3，场景优先** | [按 node 跳过空 chunk](#opt-3) | 小 token、声明容量明显偏大 | 缓存各远端 node 的最终范围 |
| **4，独立实验** | [并行预取目标和取 slot](#opt-4) | 一个接收 token 有多个去重后的本节点目标 | 保留原来的 payload 拷贝 |
| **5，独立实验** | [复用 payload tile](#opt-5) | payload 大，且同一 token 写往多个目标 | 保留串行取 slot，先单独验证读复用 |
| **6，命中率决定** | [条件省略发送结束 AMO](#opt-6) | 对某远端 node，所有容量内 chunk slot 都实际发送 | 只省略满足条件的 node 的结束通告 |

执行顺序按负载收敛：

- **第一轮统一只做优化 1**，得到真实 V1 的基线和少量候选配置。
- **大 token、多目标**：1 → 2 → 分别做 4、5 → 比较是否组合 4+5。
- **小 token、大 cap**：1 → 3 → 2；先处理空循环，再判断是否需要改 payload。
- **小 token、cap 紧且 chunk 全覆盖率高**：1 → 6；命中率低就跳过 6。

其余局部微调放在[后续观察项](#later)，不与上述实验同时开展。任何阶段出现错误或挂起，都先停止性能比较。

## 调参前必须完成的正确性工作

这些问题不计入六项性能优化。性能 A/B 的两端必须包含同一组正确性修复，避免把少处理 token 或状态错误当成加速。

| 问题 | 源码依据与触发条件 | 完成条件 |
|---|---|---|
| **recv 分片使用了错误的索引** | `internode_v1.cpp:407` 用 `blockId % 8`。R 非 8 倍数、外层 bid 跨轮时，可能重复/遗漏 token | 使用 `bid % S`；审计已有 R=21/42/85/170 的 AUTO 规则，覆盖跨轮和不同 W |
| **发送容量没有按 chunk slot 布局留足空间** | `internode_v1.cpp:190–207` 动态分配 64-token slot；`common.hpp:49–50` 的节点分区 stride 却是原始 cap | 统一内部 slot stride、buffer 分配和所有相关索引；验证尾 chunk 先拿 slot 的顺序 |
| **`weights=None` 路径无保护** | intra 的 `internode_v1.cpp:84–87` 和 staging 的 `685–687` 无条件读取 weights | 明确支持语义并覆盖 staging、recv、返回值；若暂不支持，应在 launch 前明确拒绝 |
| **输出容量错误处理不完整** | `internode_v1.cpp:55–62、444–449` 有容量 assert；NDEBUG 构建需另核对 | 越界必须明确报错并阻止无效写；同时处理 maps、计数、replay 与 combine，不能只丢弃 payload |

两个已经用 CPU 模型确认的反例：

- `R=60,W=8,N=2`、64 个完整 chunk：原 recv 分片遗漏 128 个位置、额外重复 128 个位置；换成 `bid % 8` 后精确覆盖。**W=8 不是正确性约束。**
- `cap=T=65,R≥2`：若 1-token 尾 chunk 先拿 slot 0，完整 chunk 后拿 slot 1，后者使用位置 `64..127`，超出节点分区 63 个位置。当前 Python/C++ 构造路径没有自动对齐 cap；仅在整个 buffer 尾部加空间不能解决跨节点分区重叠。

尾 chunk 布局修好前，64 对齐的 cap 可规避这类容量问题，但不能代替非对齐容量的回归验证。此前 ROCm 7.1.1 的编译记录确认容量 assert 会调用 `__assert_fail` 并触发 trap，不能再称其“永不生效”。

## 当前数据路径与参数

入口见 [launch.cpp](src/ops/dispatch_combine/launch.cpp) 和 [internode_v1.cpp](src/ops/dispatch_combine/internode_v1.cpp)：

```text
EpDispatchCopyToStaging：将输入与 metadata 打包到 staging
    ↓ 同一 stream 上的 kernel 边界
EpDispatchInterNodeV1Kernel
    RDMA blocks：各 wave 执行 send → recv
    XGMI blocks：执行 intra
    所有 blocks：执行 DispatchSync
```

| 符号 | 含义 | 当前关系 |
|---|---|---|
| B | 主 kernel 的 blockNum | R + X |
| R | rdmaBlockNum | 同一批 blocks 负责 send 和 recv |
| X | intra 的 block 数 | B − R |
| W | warpNumPerBlock | gfx950 上一个 wave 为 64 个 lane |
| S | numRecvBlock | recv 中分担同一 chunk 的分片数，当前为 8 |
| T / cap | 当前 rank 的实际输入 token 数 / maxNumInpTokenPerRank | 实际 token 数与声明容量要分别记录 |
| N / G / K | 节点数 / 每节点 GPU 数 / 每 token expert 数 | 不用 K 代替去重后的目标数 |
| K_d | 接收 token 的有效、去重后本节点目标 PE 数 | 含 self 目标；普通路径按此数量取 slot |

分析性能时保留三个事实：

- 固定 R 增加 B 只增加 X；不能把它描述成给 recv 增加 blocks。
- N=2 且 W 足够时，每个 RDMA block 只有一个 wave 执行远端 send；其余 wave 可以先进入 recv。两者之间没有 block barrier，“未执行 send”不等于整个阶段空闲。
- `CopyToStaging` 的 grid 是设备 CU 数，主 kernel 的 grid 是 B；W 同时影响这两个 launch，调 W 的收益不能全部归给 recv。

## 如何建立可信基线

### 使用真实 InterNode V1 入口

使用 [跨节点测试与 benchmark](examples/ops/dispatch_combine/test_dispatch_combine_internode.py) 的 `--kernel-type v1`，以及 [V1 正确性测试](tests/python/ops/test_dispatch_combine_internode_v1.py)。`tests/python/ops/bench_dispatch_combine.py` 是另一套节点内 benchmark，不能用其成绩替代 V1 跨节点结果。

首轮手动扫描使用 **`--cmd bench --kernel-type v1`**，并设置 `MORI_EP_LAUNCH_CONFIG_MODE=MANUAL`。该模式支持分开的 `--dispatch-block-num`、`--dispatch-rdma-block-num`、`--dispatch-warp-per-block` 与对应的 combine 参数；固定 combine 后先改 dispatch，再复测最终两阶段组合。

这些参数只在 `--cmd bench` 下转发，test、test_sentinel、tuning 等模式会忽略并警告。`--cmd tuning` 的公共扫描会同时改变两阶段，不能用它代替“固定 combine、只改 dispatch”的首轮实验。

已有全量 tuning 会分别保存 dispatch 和 combine 的优胜配置。**不能把两个独立优胜点直接拼起来当作验收结果**，应将选出的参数对放回连续执行的 dispatch+combine 中重测。

### 确认实际测到了哪一个版本

每组结果至少记录：

- 源码提交与补丁、实际加载的 kernel 产物、编译器版本、NIC/backend、GPU 型号和节点数。
- MANUAL/AUTO、匹配的设备配置文件、命中规则或 fallback，以及最终生效的 dispatch/combine B/R/W 和 S。
- 各 rank 的 T、cap、hidden、dtype、top-k、numQpPerPe、路由分布、是否 replay。
- dispatch、combine 和连续两阶段总延迟；dtype 转换或其他中间操作的时间单独记录，明确是否计入总延迟。

改 kernel 后确认新产物被加载，不在持有旧 module 的同一进程里直接比较。profiler 用于定位，最终延迟用一致的非 profiler 配置复测。

### 处理预热和样本波动

当前跨节点入口读取 `MORI_EP_ROUNDS`（默认 10），bench 与 tuning 都丢弃 round 0；`run_bench_once` 内还有固定 3 轮预热。增加到 `MORI_EP_ROUNDS=31` 可得到 30 个保留轮次，但这不是“预热已经充分”的证明。

`MORI_EP_DROP_ROUNDS` **当前没有实现**。若丢掉首轮后仍有 barrier 后的爬坡，应增加逐轮观测和可配置的预热/样本排除机制，不能靠设置不存在的变量。

对 baseline 与候选做多次交错测量，比较波动区间、最慢 rank 和总延迟；噪声与收益同量级时先改测量方法。正式比较前固定采样规则，不根据结果临时挑选要保留的轮次。

### 控制实验数量

先选择三类有代表性的输入，不立即穷举所有 shape：

| 输入 | 起始用例 | 要回答的问题 |
|---|---|---|
| 小 token、大 cap | T=8/32，cap=4096 | 空 chunk 遍历是否占主要时间 |
| 小 token、紧 cap | T=32，cap=32 或 64，分别用远端覆盖和本地-only 路由 | 结束 AMO 的省略条件实际命中多少 |
| 大 token、多目标 | T=4096，cap=4096，实际使用的 hidden/dtype/top-k | recv 的 slot 依赖与重复读是否暴露 |

T 与 cap 需要能独立构造；现有 bench 的满 token 路径不能直接代表“大 cap、小 T”。先补齐该输入构造，再评价优化 3。真实业务的 shape 和倾斜程度优先于上面的示例值。

**候选参数需要单独通过正确性检查。** 当前跨节点 benchmark 的 `run_test_once` 不接收候选 B/R/W，bench/tuning 的预热校验使用 op 默认配置，候选参数随后只进入计时路径。因此不传 `--skip-verify` 也不代表验证过候选。首轮采纳前应补齐校验路径的参数透传，或用显式构造候选配置的测试覆盖同一组实际 launch 参数。

通用验收：payload、indices、weights/scales、接收计数及 combine 结果正确；接收 slot 顺序可以变化，应通过源 token 映射核对，不能要求输出逐行顺序不变。覆盖重复 PE、`-1` expert、空/不均衡 rank、尾 chunk、routing cache/replay 和连续多轮。索引或协议变化另测多节点、多 QP；目标使用图重放时再覆盖同样的执行模式。

<a id="opt-1"></a>

## 优化 1：先调 B/R/W

**判断：最先做。** 参数接口与查表机制已经存在，先确定真实 kernel 是否缺少合适的资源分配。

源码：[dispatch_combine.py](python/mori/ops/dispatch_combine.py) 的 `_resolve_launch_params`、[tuning_config.py](python/mori/ops/tuning_config.py) 的 `lookup`，以及 `internode_v1.cpp:103–107、157–162、378–408`。

V1 的 AUTO fallback 是 `(B,R,W)=(96,64,8)`，它不代表每台设备的实际配置。查表会放宽 hidden/top-k 匹配，并对 token 数使用上界匹配或最大规则；不能把“没有精确 shape”当成必然 fallback。AUTO 的命中规则和非零 fallback 都可能覆盖调用参数。

**首个实验：**固定算法和 S=8，从实际基线出发做三条小扫描。若以 `(96,64,8)` 为起点，可用：

1. 固定 B=96、R=64，比较 W=4/8/16。
2. 固定选中的 W 和 R=64，比较 X=16/32/64，即 B=80/96/128。
3. 固定 X=32 和选中的 W，比较 R=32/64/96，即 B=64/96/128。

从结果中只留一两个候选，再小范围交叉，不把三条扫描的收益直接相加。检查 `0<R<B`、线程数与资源限制；调参前先完成 recv 分片修复。

**继续条件：**同一负载下，真实 dispatch+combine 总延迟稳定改善，且没有正确性或其他代表性输入的明显回退。只有 dispatch 变快而 combine 抵消收益的点不采纳。最后再评估是否需要独立调整 combine，并验证选出的参数对。

**交付物：**每类目标负载的一两个参数候选及其原始测量记录。完整验证后才更新相应设备/shape 的 tuning 规则。

<a id="opt-2"></a>

## 优化 2：联合调整 recv 分片数 S 与 W

**判断：值得做小原型，不能预先判为“不划算”。** 这是局部任务划分改动，先只实验 dispatch；combine 可以保留自己的分片数。

源码：`internode_v1.cpp:368、378–408`。修复后的任务编号为：

```text
bid = k * S * (N−1) + i * S + s
token_in_chunk = s * W + warpId + m * S * W
s = bid % S
```

`S*W=64` 不是正确性条件。W=16、S=8 时，每个完整 chunk 有 128 个 wave 读取其 flag，只有 64 个 wave 搬 payload；S=4 时，每个 chunk 的 64 个 wave 都有 payload 工作。同样 R 下也能先覆盖更多 chunk。

这不等于全设备的同时轮询量按比例下降：固定 R/W、所有 blocks 都在工作时，wave 总数没有减少。收益来自任务分摊、空工作和 flag 访问分布的变化，必须实测净效果。

**首个实验：**固定优化 1 的候选配置，对比 S=8 与 `S=ceil(64/W)`；相同则补一个较小的 S，例如 W=8 时比较 S=4/8。有方向后才增加少量 S 候选，不立即铺大网格。

先用文末 CPU 模型确认覆盖，再测真实 kernel 的 recv 等待、payload 时间和总延迟。S 较小也可能让每个 wave 多搬 token，不能只统计 polling 次数。若以后修改 combine 的 S，应同步核对 `internode_v1.cpp:1025` 的完成阈值。

**继续条件：**覆盖和多轮状态正确，总延迟稳定改善；只减少静态迭代数而没有端到端收益时停止扩大搜索。

<a id="opt-3"></a>

## 优化 3：按远端 node 跳过空 chunk

**判断：小 token、大 cap 时优先于 payload 改造。** 在该场景下，可以把本项提前到优化 2 之前。

源码：`internode_v1.cpp:369、378–405`。接收循环上界来自 cap，发送量来自远端实际路由。cap=4096、T=8、N=2、S=8 时，即使远端只发送一个 chunk，仍有 `8*64=512` 个 bid；其中只有 8 个 bid 对应非空 chunk，而且并非每个 wave 都有 payload。

`nodeRecvTokenNum[node]` 的终态编码是 `chunksSent[node]*64+1`，不是精确的有效 token 数。它允许判断某个 chunk slot 永远不会有数据。

**首个实验：**保留现有 flag 轮询协议；某个 node 的最终计数首次被读到后，将其范围保存在本轮该 wave 的状态中。后续遇到该 node 时，在读取 chunkFlag 前跳过已经确认超出范围的 k。

约束：

- 不用本 rank 的 T 推断远端数量；不同 rank/node 可以不均衡。
- 不跨 node 共享一个终止标志；bid 后续轮次可能切换 node，不能直接退出整个循环。
- 未收到最终计数时仍走原来的等待路径；缓存只在本次 dispatch 内有效，不能沿用到下一轮。

首轮只测 T=8/32、cap=128/4096，再加入空 rank 和各 node 发送量不同的输入。记录实际空 bid 数、flag load 次数和总延迟，避免为了少量读取引入过大的每 wave 缓存。

**继续条件：**空迭代/flag 读取下降且总延迟改善。若生产容量通常贴近实际发送量，或缓存开销抵消收益，则限制启用场景或停止。

<a id="opt-4"></a>

## 优化 4：并行预取目标和取 slot

**判断：源码支持这个方向，但未证明延迟一定暴露。** 第一版只改 slot 的准备方式，保留原 payload 拷贝，便于归因。

源码：`internode_v1.cpp:410–468`。recv 按 token 遍历，内层串行处理 expert；对每个有效去重目标，先解析 peer 指针，再取 slot，之后才拷贝。此前真实 kernel ISA 也保留了 expert 内循环及循环内的 AMO/等待链。

**首个实验：**用 wave 操作筛掉异节点、无效 expert 和重复 PE；剩余 K_d 个目标由不同 lane 并行解析指针和取 slot，再沿用原逐目标拷贝。普通路径与 replay 分开处理，replay 必须读取缓存 slot。

必须保持：

- `interNodeDispDestTokIdMap`、`dispTokIdToSrcTokId` 的映射语义及无效项处理。
- 每目的 PE 的计数、源 token 对应关系、容量错误处理和连续多轮状态。
- combine 与 routing handle 的消费方式；不能因输出顺序变化而破坏 replay。

先比较 K_d=1/2/4/8 的代表性路由。检查新产物是否减少串行等待，以及 VGPR、spill 和实际 active waves 的变化。不同 lane 的目标指针不等于天然可放在 SGPR 中的常量数组。

**继续条件：**等待减少能转成真实总延迟收益，且寄存器/溢出代价没有抵消它。K_d=1 无明显收益是合理结果；不能只凭 AMO 数量或单次原子探针的延迟决定是否采用。

<a id="opt-5"></a>

## 优化 5：payload tile 读一次，写往多个目标

**判断：重复读取确实存在，值得与优化 4 独立 A/B。** 不要求先实现并行取 slot。

源码：`internode_v1.cpp:458–472` 和 [WarpCopy](include/mori/core/transport/p2p/device_primitives.hpp)。当前同一 token 对每个有效目标重复读取同一份 `dispatchInp`；该 buffer 以 `hipDeviceMallocUncached` 分配。

**首个实验：**先按原顺序串行取得所有目标 slot，再将 payload 循环改为“协作读一个 tile → 将该 tile 写向各有效目标”。目标地址预先准备，避免每个 tile 都重复查表。indices、weights、scales 与源 token 映射保持原有语义。

只按 payload 的逻辑字节数估算，H 为每 token 的 hidden 字节数：

```text
现状：源读 K_d*H 字节 + 目标写 K_d*H 字节
复用：源读     H 字节 + 目标写 K_d*H 字节
减少比例：(K_d−1)/(2*K_d)
```

在 N=2、G=8、K=8、独立均匀选 PE 的模型中，以收到该 token 为条件，`E[K_d]≈3.2389`，对应约 34.56% 的局部 payload 读写字节减少。**这不是加速比，也不是 XGMI 写量减少。** 目标写包含 self 和 peer 写；输入 RDMA、metadata 及其他阶段不在这个估算内，实际 K_d 应从路由统计。

对齐不能省略：hidden=7168、bf16、K=8、无 scales 时，记录 stride 为 `14336+32+32+4=14404` 字节，token 起点模 16 为 `0/4/8/12`。16-byte load 必须处理未对齐和尾部；不能仅凭 hidden 长度整除 tile 大小就认定安全。

先固定 geometry 比较基线与 tile 版本，记录源读流量、store 带宽、寄存器/spill 和总延迟。优化 4、5 分别完成后，再比较基线、仅 4、仅 5、4+5 四种版本；合并版需要重新做正确性验证，并小范围复查 geometry。

**继续条件：**减少读取带来稳定总延迟收益。若主要限制仍是目标写带宽，或地址保存/spill 吃掉收益，保留简单版本，不以理论字节减少作为采纳理由。

<a id="opt-6"></a>

## 优化 6：条件省略发送结束 AMO

**判断：先量命中率，再决定是否实现。** 满足条件的低 token 场景可以提前尝试；宽 cap 场景通常不优先。

源码：`internode_v1.cpp:181–207、277–286、389–398、968–975`。send 尾部通告本 node 一共发送多少 chunk slot；recv 用它退出永远等不到 signal 的空 slot。若某远端 node 的所有 slot 都会被 put 自带的 signal 覆盖，可考虑省掉该 node 的结束通告。

准确条件是逐目的 node 的：

```text
chunksSent[node] <= ceil(T/64) <= maxChunkNum = ceil(cap/64)
只有 chunksSent[node] == maxChunkNum 时才考虑省略
```

当前非 LL send 会跳过 `mask==0` 的 chunk。因此 cap=T 不代表条件成立：cap=T=64、全部 expert 落本节点时，远端 `chunksSent=0`，仍必须发结束通告，否则 recv 会永久等待。

**首个实验：**在独立诊断运行中统计条件命中率，不把统计开销算进性能结果。确认有足够多命中后，只对满足条件的远端 node 跳过 AMO，其余通告与本地 bookkeeping 保持原样。

同时验证连续 slot 分配、replay 的缓存 slot、combine 的终止逻辑，以及清零/下一轮状态。对非 LL combine，只有在全覆盖不变量成立时，才无需依赖结束计数识别空 chunk；不能把“计数为零”泛化成可跳过整个 node。

**继续条件：**两节点、多 QP、稀疏路由、不同 cap/T、replay 和连续多轮均正确且无挂起，总延迟稳定改善。外部 [PR #630](https://github.com/ROCm/mori/pull/630) 只提供方向参考，不能把其 barrier segment 差值解释成当前版本一次 AMO 或全部 doorbell 排队的耗时。

<a id="later"></a>

## 后续观察项：有瓶颈证据再做

以下不占用六项优化的编号，也不与首轮实验并行叠加。

| 项目 | 当前处置 | 重新考虑的条件 |
|---|---|---|
| 同址轮询加退避 | 先分离 recv 等待时间，保留为小实验 | 确认 flag 等待或争用显著，比较等待延迟与总延迟 |
| 集中到一个 wave 轮询 | 后置 | 新 block barrier 可能让原本先进入 recv 的 wave 等待 send wave，必须计入丢失的重叠 |
| `DispatchSync` 的 fence/ordering 放宽 | 先证明发布、等待、清零、下一轮协议 | 单独确认可放宽的位置，再逐项消融；peer 共享 signal 不能直接改成 agent store |
| slot 原子三段式批量化 | 优先完成优化 4/5 的真实 recv A/B | 原子/争用仍是暴露瓶颈，且批量统计成本有机会被覆盖 |
| counter 加零守卫、Unroll、metadata 小拷贝、intra load-first | 后置 | 真实阶段计时表明其成本值得处理；旧局部测量不足以永久否决，也不足以承诺收益 |

轮询计数要准确：示例 S=8、W=8 时，一个 flag 可对应 64 个轮询 lane；初始 8 个相邻 flag 共对应 512 个 lane。它们不是同时读取“同一个字”，实际同时争用程度取决于调度与进度。各自轮询不同地址的旧探针不能替代同址争用实验。

`ShmemPutMemNbiSignalThread` 在当前 IBGDA RDMA 路径已经聚合活跃 lane 的描述符，不能机械换成 Warp 版本。只由 lane 0 执行会丢失其他 lane 的 run；实际 WQE/doorbell 数还受 VMM 分段影响。SQ 空间不足时，put 内部也会等待 CQ，不能只把显式 `ShmemQuietThread` 算成全部网络等待。

## 证据如何使用

| 证据 | 可以支持 | 不能支持 |
|---|---|---|
| 源码与 CPU 覆盖枚举 | 索引反例、分片覆盖、重复源读和循环依赖的存在 | GPU 同步正确性、调度和最终加速比 |
| 真实 kernel 的离线 ISA | 编译器是否展开循环、实际等待链、assert 调用及资源元数据 | 实际 occupancy、瓶颈暴露程度和可回收时间 |
| 原文 ROCm 10.0.0 微基准 | 选择少量实验候选 | 本分支收益、整机带宽上限或“永远不值得做” |
| 外部 PR 的结果 | 设计假设和验证方法的参考 | 当前 backend、设备和负载上的收益保证 |

原文依赖的 `tools/isa_probe/` 及六份探针源码目前缺失，本文不再保留不可直接复现的长表和运行命令。旧报告中的 geometry 3–10%、intra 约 9%、同步尾部 0.7–1.0 µs 都尚未独立复现，不用于本计划的收益承诺或接受门槛。

还需统一两个技术口径：

- **scope 与地址位置分开。** 当前源码用普通 `int atomicAdd` 取 slot；本机 ROCm 7.1.1 的 `amd_hip_atomic.h:217–218` 将其实现为 RELAXED + AGENT。远端地址或 ISA 的 sc0 不能证明 SYSTEM。原 ROCm 10 的 SYSTEM 原子探针未经等价性核对，不能称为“源码原样”；这也不单独证明当前跨 GPU 原子有正确性故障。
- **原子按去重目标和全部来源计数。** 一个 rank 的 intra 加 recv，会处理本 rank 和各远端同 rail 来源。各来源均为 T 时，slot AMO 的宽松上界为 `N*T*min(K,G)`，含 self；不能统一写成 `T*K`。跨 GPU 远程原子需再排除 self。

此前 bf16 dispatch 产物记录了 VGPR=90、SGPR=106、SGPR spill count=208、private segment=72 B。这些数字只属于该次编译，不能拿它们证明优化 4/5 必有寄存器余量。复编译后应按目标 kernel 符号检查，而不是取整个产物的第一组 metadata。

## 最小复核方法

### CPU 分片覆盖模型

只验证每个 `(node,chunk,token)` 是否恰好被分配一次，不模拟 GPU 调度或同步。可同时复核原 recv 问题和优化 2 的 S/W 候选。

```python
from collections import Counter

def coverage(R, W, N, chunks, length, S=8, use_bid=True):
    visits = Counter()
    for block in range(R):
        for bid in range(block, S * chunks * (N - 1), R):
            k = bid // (S * (N - 1))
            node = (bid // S) % (N - 1)
            shard = (bid if use_bid else block) % S
            for warp in range(W):
                for token in range(shard * W + warp, length, S * W):
                    visits[node, k, token] += 1
    return all(visits[node, k, token] == 1
               for node in range(N - 1)
               for k in range(chunks)
               for token in range(length))

assert not coverage(60, 8, 2, 64, 64, use_bid=False)
assert coverage(60, 8, 2, 64, 64)

for W in (4, 8, 12, 16):
    for S in sorted({4, 8, (64 + W - 1) // W}):
        for R in (1, 7, 21, 60, 64, 85):
            for N in (2, 3, 5):
                for length in range(65):
                    assert coverage(R, W, N, 3, length, S=S)
```

### 离线编译与 ISA 检查

以下是本机 ROCm 7.1.1、gfx950、ionic 的复编译方法，需从仓库根执行；其他环境调整 ROCm、MPI 头路径和 NIC define。它不运行 GPU，不代替两节点验收。

```bash
/opt/rocm/bin/hipcc --genco --offload-arch=gfx950 -std=c++17 -O2 -S \
  -gline-tables-only -D__HIP_PLATFORM_AMD__ -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -DMORI_DEVICE_NIC_IONIC -I./include -I. -I./3rdparty/spdlog/include \
  -I/usr/include/x86_64-linux-gnu/mpich \
  src/ops/kernels/ep_internode_v1.hip -o /tmp/mori-dispatch-v1.s

sed -n '/^EpDispatchInterNodeV1Kernel_bf16:/,/^[[:space:]]*\.size[[:space:]]*EpDispatchInterNodeV1Kernel_bf16/p' \
  /tmp/mori-dispatch-v1.s > /tmp/mori-dispatch-bf16.s

rg -n 'This Loop Header|flat_atomic|global_atomic|s_waitcnt' /tmp/mori-dispatch-bf16.s
rg -n '__assert_fail|s_swappc_b64' /tmp/mori-dispatch-bf16.s
rg -n '__assert_fail|s_trap|vgpr_count|sgpr_count|sgpr_spill_count|private_segment_fixed_size' \
  /tmp/mori-dispatch-v1.s
```

真实阶段分析可使用 [analyze_ep_kernel_trace.py](tools/profiler/analyze_ep_kernel_trace.py)。优先拆出 recv 等待与 payload、send 提交与 SQ/CQ 等待，再决定是否推进后续观察项。
