# InterNode V1 Dispatch：优化顺序与验证计划

**先修正确性并建立可信基线；第一个性能实验只调 B/R/W。** 后续保留六项局部优化，统一按“优化 1～6”引用。每项先单独验证，只组合有稳定收益的版本。

本文只覆盖 `KernelType::InterNodeV1`，保留现有 kernel 边界、buffer 布局和同 rail 的 proxy 映射。源码基准为 `25e64aa8`；下文行号均按这一版本，不再混用旧行号。文档整理日期：2026-09-10。

**行号提示：** recv 分片修复在原 407 行处插入了 6 行注释，因此当前工作树里 407 行之后的引用要加 6（例如原 `407` → 现 `413`，原 `410–468` → 现 `416–474`）。407 行之前的引用不变。

证据范围：源码检查、CPU 索引枚举、此前 ROCm 7.1.1 的离线编译/ISA 记录，**以及 2026-09-09/10 在 mi355-gpu-49 + mi355-gpu-51 两节点上实测的正确性与性能结果**（见[实测结果](#results)）。已完成的部分：recv 分片索引修复、候选参数的校验透传、`--bench-tokens`（T 与 cap 独立构造）、优化 1，以及优化 3 的收益量化。优化 2、4、5、6 仍只是实验计划。

## 先做哪个

| 顺序 | 优化 | 何时值得试 | 第一版范围 |
|---|---|---|---|
| **1，已完成** | [调 B/R/W](#opt-1) | 所有目标负载 | 使用现有参数接口，不改算法 |
| **2，随后做** | [调 recv 分片数 S](#opt-2) | 尤其是 W 较大、每 chunk 有空闲 wave 时 | 只改 dispatch 的局部任务划分 |
| **3，下一个做** | [按 node 跳过空 chunk](#opt-3) | 小 token、声明容量明显偏大 | 缓存各远端 node 的最终范围 |
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
| **recv 分片使用了错误的索引**（**已修复**） | `internode_v1.cpp:407` 用 `blockId % 8`。R 非 8 倍数、外层 bid 跨轮时，可能重复/遗漏 token | 已改为 `bid % S`，与 `CombineInterNodeTyped` 一致；CPU 模型与两节点 A/B 均已验证，见[实测结果](#results) |
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

这些参数现在在 `--cmd bench` 和 `--cmd test` 下转发（`test` 的轮数用 `MORI_EP_TEST_ROUNDS` 控制，默认 500）；test_sentinel、tuning、stress、profile、sweep 仍会忽略并警告。`--cmd tuning` 的公共扫描会同时改变两阶段，不能用它代替“固定 combine、只改 dispatch”的首轮实验。

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

实测到的两点：**这一对机器上两个节点会朝相反方向漂移**——一轮一轮看下去，node 0 的 dispatch 从 142 µs 降到 114 µs，同时 node 1 从 160 µs 升到 184 µs，而 16 个 rank 的总平均几乎不动。所以单看某一侧的 rank，或只看几轮，会读出一个纯属两节点相对进度的"趋势"；要比较的量是全 rank × 全保留轮的总平均。另一点是**噪声地板本身很低**（0.2%～0.3%），所以 2 µs 以上的差异是真的；把候选和基线交错跑比加长单次运行更有用。

### 控制实验数量

先选择三类有代表性的输入，不立即穷举所有 shape：

| 输入 | 起始用例 | 要回答的问题 |
|---|---|---|
| 小 token、大 cap | T=8/32，cap=4096 | 空 chunk 遍历是否占主要时间 |
| 小 token、紧 cap | T=32，cap=32 或 64，分别用远端覆盖和本地-only 路由 | 结束 AMO 的省略条件实际命中多少 |
| 大 token、多目标 | T=4096，cap=4096，实际使用的 hidden/dtype/top-k | recv 的 slot 依赖与重复读是否暴露 |

T 与 cap 需要能独立构造；现有 bench 的满 token 路径不能直接代表“大 cap、小 T”。**已补齐**：`--bench-tokens` 设定实际 T，cap 仍由 `--max-tokens` 决定，两者默认相等以保持原行为。这一条不是可选的收尾工作——只测 T == cap 会得出一条在 T < cap 上把性能改坏 8%～66% 的调参结论，见[实测结果](#results)。真实业务的 shape 和倾斜程度优先于上面的示例值。

**候选参数需要单独通过正确性检查。**（**已补齐**）原先 `run_test_once` 不接收候选 B/R/W，bench 的预热校验使用 op 默认配置，候选参数只进入计时路径，因此不传 `--skip-verify` 也不代表验证过候选。现在 `run_test_once` 接收两阶段的 B/R/W，`--cmd bench` 的预热校验和 `--cmd test` 都按候选几何执行：

- `--cmd bench` 每次运行的预热轮用候选几何做一次全量校验，但 `use_max_token_num=True`，只覆盖 T == cap。
- `--cmd test` 用 `use_max_token_num=False` 逐轮改变 T，覆盖尾 chunk 与不均衡 rank；轮数由 `MORI_EP_TEST_ROUNDS` 控制（默认 500）。**采纳任何候选前跑这一条**，只靠 bench 的单轮预热不够：本轮的 recv 分片反例正是在 T == cap 且 `maxChunkNum == 1` 时无法触发。
- 仍未补齐：`--cmd tuning` 只在整轮扫描前用默认几何校验一次，没有逐候选校验；`--cmd test_sentinel`、`stress`、`profile`、`sweep` 仍忽略这些参数并给出警告。

通用验收：payload、indices、weights/scales、接收计数及 combine 结果正确；接收 slot 顺序可以变化，应通过源 token 映射核对，不能要求输出逐行顺序不变。覆盖重复 PE、`-1` expert、空/不均衡 rank、尾 chunk、routing cache/replay 和连续多轮。索引或协议变化另测多节点、多 QP；目标使用图重放时再覆盖同样的执行模式。

<a id="results"></a>

## 实测结果（2026-09-09，两节点）

环境：mi355-gpu-49（node_rank 0）+ mi355-gpu-51，各 8 GPU，EP=16；gfx950 / mi355x，ROCm 7.14.60850，AMD clang 23.0.0git，NIC ionic；`--kernel-type v1 --num-qp 2 --dtype bf16`，hidden=7168，topk=8，`numExpertPerRank=16`，非 replay。源码 `ac1634db` 加下述两处改动。V1 kernel 由 JIT 从安装包的 `_jit-sources` 编译，cache key 是源码树内容哈希，所以改一行就换一个 hsaco 目录——A/B 的两端可以确认不是同一份产物。

驱动脚本：`run2n.sh`（两节点一次运行）、`tools/isa_probe/sweep2n.sh`（交错扫描并汇总）、`tools/isa_probe/parse_ep_bench.py`、`tools/isa_probe/audit_runs.py`。原始日志在 `data/run2n/`。

### recv 分片索引：CPU 模型与真机 A/B 一致

`tools/isa_probe/recv_shard_coverage.py` 复现了本文的反例，并给出修复后的覆盖结论：

| 形式 | R=60,W=8,N=2，64 个完整 chunk | 网格 (W,S,R,N,length) 19500 组 |
|---|---|---|
| `shard = blockId % S`（原） | 遗漏 128、重复 128 | 246 组不是划分 |
| `shard = bid % S`（现） | 精确覆盖 | 全部为划分 |

真机 A/B 只改这一行（改安装包里的 JIT 源码，其余完全相同）：

| 配置 | 原 `blockId % S` | 现 `bid % S` |
|---|---|---|
| B=96,R=60,W=8，cap=T=512 | **失败**：各 rank 接收计数全错，例如 rank 9 期望 3352 实得 3361、rank 10 期望 3459 实得 3446（有多有少，与"部分位置两次、部分零次"一致） | 通过 |
| B=96,R=60,W=8，cap=T=64 | 通过 | 通过 |

**cap=64 不能用来验证这个修复。** `maxChunkNum = ceil(cap/64) = 1` 时 bid 的取值范围只有 `S*1*(N-1) = 8`，R>8 时每个 block 只进一轮，`bid` 恒等于 `blockId`，两种写法生成同样的行为。要触发需要 `S*maxChunkNum*(N-1) > R`，即 `cap > 64*R/8`。这也解释了为什么默认配置从没暴露它：R=64 是 8 的倍数，两种写法本来就一致。

### 优化 1：W 和 X 已是最优，R 是唯一的杠杆且随 shape 变

每个候选与基线交错测量（pass 1 全部配置、pass 2 全部配置……），`MORI_EP_ROUNDS` 给 20～30 个保留轮次，取全 16 rank × 全保留轮的总平均。同一配置用两个不同名字各跑 3 个 pass，得到噪声地板：cap=64 时 dispatch 相差 0.14 µs、两阶段总延迟相差 0.74 µs（0.24%）；cap=4096 时 dispatch 相差 6.0 µs（0.34%）。所有数字由 `tools/isa_probe/report_opt1.py` 从日志重算。

下面所有增量都是**相对该组最优值的 dispatch（或 combine）平均延迟增量**，单位 µs。

固定 R 和 X 扫 W、固定 W=8 扫 X，在两个 shape 上都没有跑赢现状——`warp_num_per_block=8`、X=32 已经是最优：

| 扫描 | cap=64（R=64 / R=64） | cap=4096（R=128 / R=128） |
|---|---|---|
| W = 4 / **8** / 12 / 16 | +37.1 / **0** / 未测 / +20.6 | +845.8 / **0** / +35.6 / +61.5 |
| X = 16 / **32** / 48 / 64 / 96 | +0.2 / **0** / +0.4 / +2.8 / 未测 | +1200.8 / **0** / 未测 / +2.2 / +19.2 |

R 则两边相反：cap=64 越小越好，cap=4096 越大越好。dispatch 与 combine 的最优 R 也不同——combine 在 cap≥512 就饱和在 32，dispatch 到 128 才饱和。

| cap=T | dispatch 最优 R（其余候选的增量） | combine 最优 R（其余候选的增量） |
|---|---|---|
| 64 | **8**：R=16 +0.2，R=32 +0.9，R=64 +2.8，R=96 +5.5 | **8**：R=16 +2.5，R=32 +3.1，R=64 +8.4 |
| 128 | **16**：R=8 +54.8，R=32 +1.0，R=64 +2.9，R=128 +8.0 | **16**：R=8 +59.0，R=32 +2.3，R=64 +7.6 |
| 512 | **64**：R=32 +42.3，R=48 +6.6，R=96 +2.4，R=128 +4.5 | **32**：R=48 +7.0，R=64 +20.5，R=96 +27.1 |
| 4096 | **128**：R=32 +501.8，R=64 +176.7，R=96 +14.2，R=160 +5.2，R=192 +5.2，R=256 +28.9 | **32**：R=8 +3075.0，R=16 +940.0，R=48 +17.3，R=64 +44.6，R=96 +121.4，R=128 +199.5 |

两侧都不是平台：偏小时代价很陡（cap=4096 的 combine 用 R=8 慢 3 ms，dispatch 用 R=32 慢 0.5 ms），偏大时代价平缓（R 超过最优 1～2 倍只贵几 µs）。**猜不准的时候往大猜。**

一个能解释全部四行的规律：`8*ceil(cap/64)` 恰好是 recv 的 bid 总数 `numRecvBlock * maxChunkNum * (nNodes-1)`。最优 R 让每个 RDMA block 大约领到一个 chunk 分片；小于它，block 要多轮迭代；大于它，多出来的 block 没有 recv 工作，却仍要参加 `interNodeBlocksBarrier`（`R*warpNum`）和 `dispatchGridBarrier`（`B*warpNum`）。

```text
仅在 T == cap 时成立：
R_dispatch = min(8*ceil(cap/64), 128)
R_combine  = min(8*ceil(cap/64), 32)
X = 32（即 B = R + 32），W = 8
```

**这只是对四个测点的最简描述，不是从 kernel 推出来的定律。** 两个饱和上界（128 与 32）各只由一次扫描确定，为什么 combine 比 dispatch 早饱和四倍也没有解释。而且上面四个测点全都是 T == cap —— 后来补测 T < cap 发现**这条式子在 T < cap 时会把性能改坏**，见下一节。

### 选出的参数对回到连续 dispatch+combine

按上式给两阶段各自设参，与完全不加覆盖的默认（两阶段都是 96/64/8）交错对比，每个 shape 4～5 个 pass：

| cap=T | 调参后几何 B/R/W（dispatch → combine） | 默认总延迟 | 调参后 | 差 | dispatch | combine |
|---|---|---|---|---|---|---|
| 64 | 40/8/8 → 40/8/8 | 315.07 µs | 302.67 µs | **−12.40（−3.94%）** | −2.59% | −5.17% |
| 128 | 48/16/8 → 48/16/8 | 325.62 µs | 312.47 µs | **−13.15（−4.04%）** | −2.65% | −5.42% |
| 512 | 96/64/8 → 64/32/8 | 598.44 µs | 577.39 µs | **−21.05（−3.52%）** | −0.10% | −6.38% |
| 4096 | 160/128/8 → 64/32/8 | 3434.24 µs | 3209.10 µs | **−225.14（−6.56%）** | −10.16% | −2.74% |

四个 shape 的每一个 pass 都更快，两组取值区间不重叠：

| cap=T | pass 数 | 默认区间 | 调参后区间 |
|---|---|---|---|
| 64 | 5 | 314.17～316.08 | 301.79～303.96 |
| 128 | 4 | 325.01～326.26 | 312.30～312.78 |
| 512 | 4 | 597.58～599.85 | 576.75～577.90 |
| 4096 | 5 | 3425.03～3441.15 | 3206.46～3211.50 |

收益来自哪一阶段随 shape 变，两端各有一个纯粹的例子：cap=512 选出的 dispatch 几何**就是**默认的 96/64/8，−21 µs 全部来自 combine 换到 64/32/8；cap=4096 反过来，dispatch 贡献 −179.5 µs、combine 只有 −45.6 µs。所以不能只调一个阶段就宣布结束——把 dispatch 调完再看 combine，两次都拿到了收益。

正确性：每个 shape 用 `--cmd test` 在调参后的几何上跑 25～60 轮变 T，全 16 rank `error times: 0`；bench 的每次运行也在同一几何上做了一次全量校验。

**尚未做的事：** 只测了 bf16 / hidden=7168 / topk=8 / numQpPerPe=2 / 两节点。其他 hidden/topk/dtype 和 N>2 仍未覆盖。

### T ≠ cap：上面那条式子不能照搬，默认值反而已经接近最优

`--bench-tokens` 让 bench 可以在 cap 不变的前提下只改实际 T（cap 仍由 `--max-tokens` 决定），这是评价优化 3 的前提，也是分辨"代价随 cap 涨"还是"随 T 涨"的唯一办法。补测的结论推翻了上一节规律的适用范围。

cap=4096、T=64（四组交错测量，3～4 个 pass，区间互不重叠）：

| dispatch / combine 几何 | dispatch | combine | 总延迟 | 相对默认 |
|---|---|---|---|---|
| 默认 96/64/8 + 96/64/8 | 156.77 | 161.54 | 318.31 µs | — |
| 本节 T==cap 式子按 T=4096 取值：160/128/8 + 64/32/8 | 168.00 | 175.66 | 343.66 µs | **+25.35（+7.96%）** |
| 本节 T==cap 式子按 T=64 取值：40/8/8 + 40/8/8 | 201.36 | 327.90 | 529.26 µs | **+210.95（+66.27%）** |
| 实测最优 80/48/8 + 96/64/8 | 155.99 | 161.22 | 317.21 µs | −1.10（噪声量级） |

三件事：

1. **cap=4096、T=64 时默认的 `(96,64,8)` 已经在最优值的 1 µs 内。** 它不是一个偷懒的常数，而是"cap 固定偏大、T 逐批变化"这个真实场景下的合理折中。
2. **最优 R 同时取决于 cap 和 T，不是只取决于其中一个。** cap=4096 时 T=4096 要 R=128、T=64 要 R≈48；cap=64、T=64 要 R=8。按 T 猜（40/8/8）比按 cap 猜（160/128/8）错得更狠，因为 combine 用 R=8 在 512 个 bid 上要跑 64 轮。
3. **因此这组结果不能写成 tuning 规则。** `lookup` 的键是 `num_tokens`，而调用点传的是 `input.size(0)`，即**实际 T**（`dispatch_combine.py:1236`），键里**没有 cap**。把 T==cap 的扫描结果按 T 存进去，部署时（cap 固定 4096、T 逐批变化）在 T=64 上命中的就是第三行那个 +66% 的配置。这是没提交 tuning 配置的真正原因——不只是覆盖面不够，而是**这张表的键表达不了这条规律**。

优化 3 的意义因此不止是省那几 µs：**它如果能把 recv 的循环上界从 cap 改成实际发送量，几何选择就与 cap 解耦，`lookup` 现有的按 T 索引才成立。**

### 空 chunk 遍历的代价有多大

固定 T=64、固定 96/64/8，只改 cap（即只改 `maxChunkNum`）。dispatch 的 bid 空间是 `8*ceil(cap/64)`，每个 block 要走 `bid空间/R` 个：

| cap | ⌈cap/64⌉ | bid 空间 | dispatch | combine |
|---|---|---|---|---|
| 64 | 1 | 8 | 151.36 | 163.69 |
| 128 | 2 | 16 | 150.37 | 163.66 |
| 512 | 8 | 64 | 154.34 | 162.55 |
| 1024 | 16 | 128 | 154.45 | 161.74 |
| 4096 | 64 | 512 | 157.39 | 162.17 |

**同样的活，cap 从 128 涨到 4096 让 dispatch 贵 7.0 µs，而 combine 基本不动（−1.5 µs）。** 这个不对称能从代码上讲通：dispatch 的 recv 对每个 bid **自旋**，空 chunk 要一直转到发送端的最终计数从远端到达才能退出（`internode_v1.cpp:390–399`），而且同一 block 的这些 bid 是串行的；combine 的判断是非阻塞的一次读，`processedMask` 无论 flag 在不在都会把该 bid 标记为已处理（`internode_v1.cpp:1044`），所以只剩循环开销——combine 跑在 dispatch 之后、跨设备 barrier 之后，flag 保证已经到齐。

所以优化 3 只需要动 dispatch 的 recv，combine 那侧没有这个问题。7 µs 是 R=64 这一点上的量；换成小 R 会更贵（cap=4096、T=64、dispatch R=8 是 188.42 µs，比 R=48 贵 32 µs），这部分同样属于优化 3 能回收的范围。

### 测量环境本身的两个坑

两台机器同时也跑 MORI 自己的 CI，CI 会在同一批 GPU、同一张网卡上起同一个 EP harness。重叠时不会报错，只会把结果测错：一次重叠让某个几何的 dispatch 在 1635～3372 µs 之间乱跳，而二十分钟前同一几何是 1593 ± 6。

由于扫描只改一个阶段，另一个阶段的延迟不可能随候选变化，可以当成这次运行的健康指标而不是测量对象。按它剔除，规则与被比较的量无关、对所有候选一视同仁，也不是看到结果之后再挑轮次。`audit_runs.py` 按此复核了全部 142 次运行：cap=4096 的 dispatch 扫描（53 次运行，22:52–23:13）combine 全程落在 1665.68～1675.29 µs（0.6% 带宽内），是干净窗口；23:17 之后 combine 跳到 675～2418 µs，与 CI 容器启动时刻吻合，那批数据已作废重测。`run2n.sh` 现在在测量前等 CI 空闲（按容器内是否有忙碌 python/torchrun 判断，不按容器是否存在——其中一个 CI 容器已空转数天）。

第二个坑：一端失败后另一端的 worker 会活下来，占住 rendezvous 端口和 GPU。下一次运行要么 `EADDRINUSE`，要么更糟——和这个残留进程配上对，跑出一份"没人配置过的组合"的成绩。`run2n.sh` 现在在开始前和结束后都清理两台机器，并把 `EADDRINUSE` / gloo recv 超时单独标成"未配对"而不是结果错误。

<a id="opt-1"></a>

## 优化 1：先调 B/R/W

**判断：最先做。已完成，结果见[实测结果](#results)。** 参数接口与查表机制已经存在，先确定真实 kernel 是否缺少合适的资源分配。

结论（gfx950 / EP=16 / bf16 / hidden=7168 / topk=8 / 两节点）：W=8 和 X=32 已经最优，唯一的杠杆是 R。**T == cap 时**四个 shape 上两阶段总延迟改善 3.5%～6.6%；**T < cap 时默认 `(96,64,8)` 已经在最优值 1 µs 内**，把 T==cap 的结论搬过去会慢 8%～66%。所以这一项的产出是"知道该调什么、以及什么时候不该调"，不是一组可以直接落库的常数。下文保留原始实验设计，因为它是这些结果的来路。

源码：[dispatch_combine.py](python/mori/ops/dispatch_combine.py) 的 `_resolve_launch_params`、[tuning_config.py](python/mori/ops/tuning_config.py) 的 `lookup`，以及 `internode_v1.cpp:103–107、157–162、378–408`。

V1 的 AUTO fallback 是 `(B,R,W)=(96,64,8)`，它不代表每台设备的实际配置。查表会放宽 hidden/top-k 匹配，并对 token 数使用上界匹配或最大规则；不能把“没有精确 shape”当成必然 fallback。AUTO 的命中规则和非零 fallback 都可能覆盖调用参数。

**首个实验：**固定算法和 S=8，从实际基线出发做三条小扫描。若以 `(96,64,8)` 为起点，可用：

1. 固定 B=96、R=64，比较 W=4/8/16。
2. 固定选中的 W 和 R=64，比较 X=16/32/64，即 B=80/96/128。
3. 固定 X=32 和选中的 W，比较 R=32/64/96，即 B=64/96/128。

从结果中只留一两个候选，再小范围交叉，不把三条扫描的收益直接相加。检查 `0<R<B`、线程数与资源限制；调参前先完成 recv 分片修复。

**实际执行时对这条计划的两处修正：**

- **W 和 X 的扫描范围要按 shape 定，不能只用 cap=64。** cap=64 上 W 和 X 的整条曲线都在 3 µs 内（唯一的例外是 W=4），看起来像"这两个参数无关紧要"；同一组扫描在 cap=4096 上 X=16 慢 1.2 ms、W=4 慢 0.85 ms。小 token 下 X 只是没有工作可做，不是不敏感。
- **第 3 条扫描的 R 范围太窄。** `R=32/64/96` 在 cap=64 上会漏掉最优值（R=8），在 cap=4096 上会漏掉最优值（R=128）。R 的合理范围是 `8*ceil(cap/64)` 附近，跨 shape 有 64 倍的跨度，不是一个固定的三点。

**继续条件：**同一负载下，真实 dispatch+combine 总延迟稳定改善，且没有正确性或其他代表性输入的明显回退。只有 dispatch 变快而 combine 抵消收益的点不采纳。最后再评估是否需要独立调整 combine，并验证选出的参数对。**这一条已满足**：四个 shape 均稳定改善、区间不重叠，且没有一个 shape 出现回退。

**交付物：**每类目标负载的一两个参数候选及其原始测量记录。完整验证后才更新相应设备/shape 的 tuning 规则。**已交付候选与记录（`data/run2n/`），tuning 规则尚未更新**，原因见[实测结果](#results)的"尚未做的事"。

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

**判断：小 token、大 cap 时优先于 payload 改造，现在有实测支撑。** 在该场景下，可以把本项提前到优化 2 之前。**只需要动 dispatch 的 recv**，combine 侧没有这个问题（原因见[空 chunk 遍历的代价](#results)）。

已量化的收益上限有两块，都在 cap=4096、T=64 上测得：

- **直接的遍历开销 7.0 µs。** 固定 T=64、固定 96/64/8，cap 从 128 涨到 4096 让 dispatch 从 150.37 涨到 157.39 µs。
- **被迫抬高 R 的间接开销。** 因为空 bid 要串行自旋，R 不能按实际工作量取小：cap=4096、T=64 的最优 dispatch R 是 48，而同样工作量在 cap=64 上最优 R 是 8，用 R=8 在 cap=4096 上要付 188.42 µs（比 R=48 贵 32.4 µs）。空 bid 变便宜之后这个约束才会松开。

比这两个数字更重要的是**它决定了几何参数能不能进 tuning 表**：`lookup` 的键里只有实际 T、没有 cap，只要 recv 的上界还来自 cap，按 T 索引的规则就必然在某些 (T, cap) 组合上取到错的值（实测最坏 +66%）。

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
