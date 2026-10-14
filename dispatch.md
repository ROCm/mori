# InterNode V2：Dispatch / Combine 优化结论

本文按 Dispatch、Combine 两个 OP 记录优化尝试、效果和是否采用；带宽、延迟及前后对照见 [bw.md](bw.md)。除特别说明外，测试均在 MI355X/gfx950 双机（g17 + g09）上进行：Ionic、EP16、H7168、每 rank 16 个专家、QP2、uniform routing、关闭 CCQE，显式使用普通 `v2`。

## 当前进度（2026-10-08）

按新统计口径（见下文“测量方法”）对 MI355X 的重扫已于 10 月 8 日按要求停止。**目前没有独立确认的新最优配置，默认 JSON 的 geometry 未更新**；JSON 里的指标已于 10 月 13 日在 mi355-gpu-49 + mi355-gpu-50 上按新口径刷新，见 [bw.md](bw.md#epv2-json-refresh-20261013)。

- 数值门槛 12/12、原配置 baseline 36/36 已完成，baseline 数据见 [bw.md](bw.md#epv2-rescan-20260930)。数值门槛不等于最终 strict 验收。
- 全量扫描完成 22/36 个形状、67/108 个阶段（每个形状含 Dispatch 扫描、Combine 扫描和检查三个阶段）。停止时正在跑的 FP8 E4M3 → BF16 / K8 / T256 Combine 不计入，续跑时重做。

| 组合（总专家数 256） | 已扫完的 tokens | 剩余 |
|---|---|---|
| BF16 → BF16，TopK 6 | 16–4096 | 无 |
| BF16 → BF16，TopK 8 | 16–4096 | 无 |
| FP8 E4M3 → BF16，TopK 8 | 16、32、64、128 | T256 的 Combine 和检查；T512–4096 |
| FP8 E4M3 → FP8 E4M3，TopK 8 | 无 | T16–4096 |

扫完后还要依次做 shortlist 筛选、每个候选三组独立 A0/B/A1 确认、共享配置合并，以及全部 36 个形状的新指标和 strict 验收，之后才能回填默认 JSON。

扫描状态、控制脚本和原始证据在 g17 的 `/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930`（实验归档为同级的 `mori_epv2_recheck_20260917`），两机的 benchmark 容器 `mori_epv2_recheck_0917` 均保留，这些都不在仓库内。续跑时在 g17 的控制目录执行 `taskset -c 4-7 python3 -B continue_after_baseline.py --resume --stop-after audit_after_scans`（可先加 `--dry-run` 查看计划），它会跳过已有的 67 个阶段；后续步骤见该目录的 `post_scan_handoff_commands_20261008.md`。不要删除 state/lock/evidence 或另起队列，停止时在前台 Ctrl-C。更换 GPU 节点时须新建实验目录，重跑正确性和同机 baseline，不能把新机器的测量混入旧的配对数据。

## 当前实现

改动集中在 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp)：

- **Dispatch**：`DispatchInterNodeRecv` 先按目标 PE 去重，再由各 lane 并行分配接收 slot；payload 拷贝仍按 peer 顺序执行。
- **Combine**：普通 gather 在 gfx950 上、声明 capacity ≤ 256，且行和 staging stride 都是 4 字节整数倍时调用 `WarpAccumLF<T,4,2>`，否则使用原 `WarpAccum<T,4>`。没有 dtype、topk、拓扑条件，也没有向量尾部 helper；其它架构不启用 LF。LL、`CombineAll` 和公共 gather primitive 没有优化改动。
- **同步修复**：普通 V2 和 V2LL 的 Combine 在发布 chunk 完成标志前补了 warp 同步和 release/acquire fence；LL 还会等远端节点的 token 计数发布后再读取。这项修复在 9 月 21 日的 A/B 之后加入，9 月 30 日的重扫已包含。

Geometry 由 [hip_tuning_configs.py](python/mori/ops/dispatch_combine_v2/hip_tuning_configs.py) 从 JSON 读取：

- 每个 (kernel, EP, 总专家数, topk, 阶段) 一个文件，命名为 `{arch}_{model}_{kernel}_ep{n}_expert{总专家数}_top{k}_{phase}.json`，例如 `gfx950_mi355x_InterNodeV2_ep16_expert256_top6_dispatch.json`。查表直接打开对应文件；本型号没有该文件时，用同 arch 其它型号的同名形状文件。`--tuning-save` 也按这个规则写入对应文件。旧的 `{arch}_{model}_{kernel}_ep{n}_{phase}.json` 不再读取。
- Dispatch 和 Combine 分别查表，各自匹配本阶段的 dtype。FP8 → BF16 与 BF16 → BF16 共用 BF16 Combine 行，修改该行要验证两种输入。MI355X 目前没有 FP8 Combine 记录，FP8 → FP8 的 Combine 使用默认 96/64/8。
- `num_tokens` 是含上界的 ceiling，按实际 token 数选档，超过最大档时沿用最大档；capacity、QP 和 CQ 模式不是匹配键。
- 总专家数（`EP × experts_per_rank`）和 topk 决定读哪个文件：EP16 下 256/8（`--experts-per-rank 16 --topk 8`）读 `expert256_top8`，384/6（`--experts-per-rank 24 --topk 6`）读 `expert384_top6`。规则里记录 `topk`，缺这个字段的规则不当 wildcard；专家数由文件名确定，不逐条记录。现有 MI355X 表有三种形状：256 专家（H7168，topk 6/8），以及 384 专家 topk 6 的 DeepSeek-V4.1（H5120，见 [bw.md](bw.md#epv2-dsv41-tuning-20261013)）和 DeepSeek-V4-Pro（H7168，见 [bw.md](bw.md#epv2-dsv4pro-tuning-20261013)），后两者在同一个 `expert384_top6` 文件里按 hidden 区分；未覆盖的组合使用默认 geometry，补调优后才能入表。
- `MORI_EP_V2_TUNING_DIR` 指定表目录。设置 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM` 中任一个都会绕过两个 OP 的查表，测试自动选档前要清除。公开 API 的 family 默认是 `auto`，小 token 可能切到 `v2_ll`。
- `v2_ll` 查 `InterNodeV2LL` 表，规则同上。MI355X 的 V2LL EP16 表有 256 专家（topk 6/8）和 384 专家（topk 6，H5120 与 H7168）的规则、T64–4096，见 [bw.md](bw.md#epv2-ll-tuning-20261013)、[bw.md](bw.md#epv2-dsv41-tuning-20261013) 和 [bw.md](bw.md#epv2-dsv4pro-tuning-20261013)；`auto` 模式下 T ≤ 512 走 `v2_ll`、用这张表，T > 512 走普通 V2 表。
- 单机（如 EP8）由 `lookup()` 先读 `IntraNodeV2` JSON，匹配规则同上；某个阶段在 JSON 里没有规则时，该阶段回退到原来的 Python 表，两边都没有时用默认 geometry。`quant_type` 不为 `none` 时整份 schedule 只走 Python 表。MI355X EP8 有 256 专家（topk 6/8）和 384 专家（topk 6，H5120 与 H7168）的规则，见 [bw.md](bw.md#epv2-ep8-tuning-20261013)、[bw.md](bw.md#epv2-dsv41-tuning-20261013) 和 [bw.md](bw.md#epv2-dsv4pro-tuning-20261013)。

## 测量方法（2026-09-30）

适用于 [test_dispatch_combine_v2_internode.py](tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py) 的 `--cmd tuning --tuning-save`。最终指标由 `_row_metrics()` 采样、[_grand_mean_metrics.py](tests/python/ops/dispatch_combine_v2/_grand_mean_metrics.py) 汇总，与 V1 的 grand mean 口径对齐。搜索选优与最终入表测量是两个独立步骤。9 月及更早的数据按当时的口径记录，不按本节改写。

### 计时与采样

1. 选定配置后，用将要写入表的 Dispatch/Combine 配对重新建 op 测量，不复用搜索分数。
2. 每个 `--tuning-reps` pass 独立 warmup，然后 `torch.cuda.synchronize()`、`dist.barrier()`，再记录首个 GPU event；循环内不加逐轮屏障。每个 pass 丢弃前 `--drop-rounds` 轮。默认 `--warmup 20 --rounds 30 --drop-rounds 1 --tuning-reps 3`，即每个 rank、每阶段保留 87 个样本。
3. 每轮分别记录 Dispatch、显式 dtype 转换、Combine 的 event 边界。两阶段延迟及其和不含转换区间；它们是 GPU event 区间，不等于纯 kernel 时间，也不等于端到端 wall time。
4. 所有 pass 结束后做一次 CPU `all_gather_rows`，收集各 rank 的字节数依据和原始延迟；读取接收 token 数和跨 rank 汇总都不在计时循环内。

### Grand Mean 与字节口径

先对每个 rank、每个保留轮次计算带宽，再对所有 pass、rank、round 求算术平均，不再用“字节数除以平均延迟”。设 `t[r,p,i]` 为 rank `r`、pass `p`、保留轮次 `i` 的阶段延迟（µs），`recv[r]` 为该 rank 接收的 token 数，`rdma[r]` 为 `_rdma_algo_token_count()` 按所测 family 计算的算法 token 数，`width[r]` 为该阶段每 token 实际存储的 payload 字节数：

```text
avg_latency_us          = mean(t[r,p,i])
avg_rdma_bandwidth_gbps = mean(rdma[r] * width[r] / (1000 * t[r,p,i]))
avg_xgmi_bandwidth_gbps = mean(recv[r] * width[r] / (1000 * t[r,p,i]))
ll_scale_rank0          = max_tokens * topk / (recv[0] + 1)
avg_ll_bandwidth_gbps   = avg_xgmi_bandwidth_gbps * ll_scale_rank0
```

`mean` 覆盖全部保留样本。LL 与 V1 保存的 Average 行一致：先求全局 XGMI 带宽均值，再乘 rank 0 的缩放比。写入时保留两位小数。

| 入表字段 | 定义 |
|---|---|
| `bandwidth_metric` | `grand_mean` |
| `bandwidth_gbps`，普通 `v2` | `avg_rdma_bandwidth_gbps` |
| `bandwidth_gbps`，`v2_ll` | `avg_ll_bandwidth_gbps` |
| `bandwidth_gbps`，单机 V2 | `avg_xgmi_bandwidth_gbps` |
| `avg_rdma_bandwidth_gbps` / `avg_xgmi_bandwidth_gbps` / `avg_ll_bandwidth_gbps` | 分别保存上式三个指标；单机行没有 RDMA 和 LL，只保存 XGMI |

- `width` 按实际存储宽度计：Dispatch 取输入张量的 `shape[-1] * element_size()`（packed FP4 按 packed 字节），Combine 取转换后张量的宽度。只计 hidden payload，不含 scale、weights 和控制信息。
- RDMA 是 V1 兼容的算法计数，不是 NIC 实测流量：普通 V2 对每个 token 的目标 node 去重，并包含本节点；LL 按 `tokens * node_count` 计数。
- 旧表的 `bandwidth_gbps` 是 rank 0 接收字节数除以平均延迟，与新定义不可比；只把 `bandwidth_metric` 改名不算完成迁移。MI355X 的 V2 默认 JSON 已于 2026-10-13 按新定义重测（geometry 不变），见 [bw.md](bw.md#epv2-json-refresh-20261013)。
- `_row_metrics(..., evidence_callback=...)` 可以导出所有 rank 的原始样本供离线复算，但 `--tuning-save` 不会自动归档，需要调用方另行保存。

<a id="bandwidth-meaning"></a>

### 各带宽的物理含义

三个带宽都是固定字节数除以阶段延迟，同一形状下只差一个常数，用它们比较快慢与直接比较延迟等价。

| 字段 | 物理含义 |
|---|---|
| `avg_xgmi_bandwidth_gbps` | 每个 rank 实际收到的 payload，按目标 rank 去重；最后一跳基本经 XGMI 写入 |
| `avg_rdma_bandwidth_gbps` | 每个 rank 发往各节点的 payload，计数规则见上，含本节点；跨机实际流量是它的 `(node_count - 1) / node_count`，两机时为一半 |
| `avg_ll_bandwidth_gbps` | 没有对应的实际流量。分子 `max_tokens * topk` 按每个 (token, expert) 各传一份计，这是 DeepEP LL kernel 的发送方式；`v2_ll` 跨机时把整块 token 发给每个远端节点的同号 rank，机内再按目标 rank 去重，所以 LL 恒为 XGMI 的 `max_tokens * topk / (recv[0] + 1)` 倍。只用于和 DeepEP 公布的 LL 数字对照 |

MI355X EP16、256 专家 topk 8、BF16、T1024 的入表值：

| 字段 | `v2` Dispatch | `v2_ll` Dispatch | `v2` Combine | `v2_ll` Combine |
|---|---|---|---|---|
| `bandwidth_gbps` | 67.14（RDMA） | 274.93（LL） | 64.04（RDMA） | 251.05（LL） |
| `avg_rdma_bandwidth_gbps` | 67.14 | 68.93 | 64.04 | 62.95 |
| `avg_xgmi_bandwidth_gbps` | 220.19 | 225.29 | 210.01 | 205.73 |
| `avg_ll_bandwidth_gbps` | 268.70 | 274.93 | 256.28 | 251.05 |
| `avg_latency_us` | 437.00 | 427.35 | 457.71 | 467.63 |

- 两个 family 的 `bandwidth_gbps` 相差约 4 倍，延迟只差约 2%，差距来自主指标分别取 RDMA 和 LL。跨 family 比较用延迟或同一个 `avg_*` 字段。
- LL / XGMI ≈ 1.22：8192 个 (token, expert) 中约 18% 是同一 token 落在同一 rank 上的重复专家，没有实际传输。
- 每个 GPU 对应一块 400 Gb/s（50 GB/s）NIC。`v2` Dispatch 的 RDMA 67.14 GB/s 高于线速，折半后约 33.6 GB/s，即整个 Dispatch 阶段平均使用约 67% 的 NIC 线速。

### 选优与只刷新指标

- 搜索仍走 `_timed_pass()`：每个 pass 先求 rank/round 平均延迟，候选与基准配对重复测试，再用 median、配对差值和最差轮次约束判断。默认 `--tuning-metric total` 比较 D+C 延迟和，`phase` 只用于单阶段诊断。这条路径本次没有修改，搜索分数不能当作入表带宽。
- 只刷新已选配置的指标：清除 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`，用 `--cmd tuning --tuning-save --tuning-phase dispatch`（或 `combine`），并把 `--tuning-candidate B,R,W` 设为该形状当前解析出的 geometry。输出 `no candidate measured (refresh of the shipped geometry)` 即为纯刷新；geometry 与基准不同时是单候选调优。一次运行只保存一个阶段。
- `--tuning-config-dir` 既是读取基准的目录也是写入目录；独立复测时要先放入待验证的两阶段表，不能用空目录。
- 同一个脚本在单节点上（`torchrun --nnodes=1`）也能跑 `--cmd test` 和 `--cmd tuning`，此时构建单机 op：候选只有 block × warp，geometry 通过配置字段固定，结果写入 `IntraNodeV2` 表。
- 复测要记录 dtype、H、token/capacity、专家数、Top-k、kernel family、QP、CCQE、weights/scale 和采样参数，并固定 `--kernel-type v2` 或 `v2_ll`；双机运行前确认没有竞争负载。

## 实验结论总览

`B/R/W` 表示总 block 数、RDMA block 数、每 block 的 warp 数；`S` 是每 chunk 的分片数，Dispatch 和 Combine 各自独立。原编号 1–9、C1–C14 保留，便于对照原始记录。

- **有效**：三组 A–B–A 确认完整 Dispatch+Combine 周期有收益。
- **无效**：所测配置下回退或持平；不代表整个方向无效。
- **未确认**：只有小幅初筛差异，或计时受并发影响。
- **静态排除 / 未重跑**：没有 GPU 性能结论。

kernel 改动的整体效果（2026-09-21，A 为 `fac2e18e`，两边 geometry 相同，B 不含之后的同步修复）：T ≤ 256 时，BF16 的 Dispatch 延迟下降 3.20%–7.99%，Combine 下降 2.08%–17.23%；FP8 → FP8 的 Combine 下降 2.49%–4.53%，但 Dispatch 回退 2.30%–7.34%，两阶段合计 −0.74% 到 +0.56%，不能概括为全面加速。详见 [bw.md](bw.md#epv2-recorded-geometry-20260921)。

## Dispatch

### JSON 复核（commit `1b3e7dba`）

数据见 [bw.md](bw.md#json-recheck)。

| 尝试 | 结论 | 处理 |
|---|---|---|
| BF16/K6/T64：96/64/8 → 16/8/8 | **有效**：三次加长配对复测及三组普通表路径 ABA 均改善 | 已入表 |
| BF16/K8/T64：64/32/8 → 32/21/8 | **未采用**：只有门槛内的小幅配对收益，Dispatch 本身回退，没有做普通 ABA 确认 | 保留原几何，不与 Combine 候选叠加推算 |
| 其余七档、三种 dtype/K 组合的单 OP quick 扫描 | **没有新的稳定赢家** | 保留原几何；结论只覆盖所扫范围 |
| BF16/K8 的非 2 次幂 block、RDMA 比例及 W6/W12 邻域补扫 | **初筛未胜出** | 保留原几何 |

### kernel 与参数实验

| 原编号 | 尝试 | 结论 | 处理 |
|---|---|---|---|
| 1 | T4096，B/R/W 改成 256/128/8，保留 S8 | **有效** | 三组独立确认，已加入 JSON 的 T1024/2048/4096 档 |
| 1 | T64，缩小为 64/32/8 或 32/16/8 | **未确认** | 只有小幅单组差异；64/32/8 的基线还与短暂 CPU 缓存检查重叠，未采用 |
| 1 | T4096，固定 96/64，只把 W 改成 4 或 16 | **无效：回退** | 保留 W8 |
| 2 | T4096、W8，S8 改成 S16 或 S32 | **无效：回退** | 保留 S8；更多分片会增加 bid 遍历和无 payload 的 warp 工作 |
| 2 | T4096，S/W 联合改成 4/16，几何 96/64/16 | **有效** | 只有实验 patch（不在仓库中），未替换全局 S8；优先用上面的无 patch 配置 |
| 2 | T4096、W8，只把 S8 改成 S4 | **未确认** | 清洁复测只有小幅差异 |
| 2 | T4096，S/W 联合改成 16/4，几何 96/64/4 | **无效：回退** | S×W 相同不代表延迟相同 |
| 4 | 把并行 slot 分配改回串行做反向对照 | **串行无收益** | 保留并行实现；T64/512 Dispatch 变慢，T4096 约持平 |

### 未重跑的历史方向

以下方向的历史实现或完整快照缺失，没有重跑，不能标为有效或无效。

| 原编号 | 方向 | 复核结论 |
|---|---|---|
| 3 | 缓存远端 node 完成量，跳过空 chunk 轮询 | 应关注 T 远小于容量的场景；不能用本 rank 的 T 推断远端工作量 |
| 5 | 读取一次 payload tile，复用到多个目标 | 会改变跨 peer 写流和寄存器使用；V1 的收益不能代替 V2 验证 |
| 6 | 条件省略 send 结束 AMO | 须证明每个远端 node 容量内的 slot 都实际发送；稀疏、尾 chunk 和空 rank 仍需完成通知 |
| 7 | 用 LDS 任务列表摊平 warp 间 copy 负载 | 同时改变轮询、任务分配和 block 同步，不能从旧结果单独否定其中一个机制 |
| 8 | 全局 barrier 改成 LDS 加全局两级计数 | 减少原子次数不等于缩短关键路径，须保留 payload 发布和 epoch 清零顺序 |
| 9 | 按目标 node 紧凑打包 send source，合并 put | 需要额外打包和 buffer；仅凭 put 尺寸不能预测完整周期收益 |

## Combine

### JSON 复核（commit `1b3e7dba`）

| 尝试 | 结论 | 处理 |
|---|---|---|
| K8/T64：96/64/8 → 16/10/8 | **有效，小幅收益**：BF16 和 FP8 输入各三组普通表路径 ABA 均改善 | 已入表，两种输入共用 BF16 Combine 行 |
| K8/T256：64/32/8 → 64/42/8 | **有效**：BF16 和 FP8 输入各三组普通表路径 ABA 均改善 | 已入表；FP8 均值受尖峰影响 |
| FP8/K8/T256：64/32/8 → 256/128/8 | **无效**：初筛 WIN 在加长复测中回退，初筛差异主要来自原配置 Dispatch 偶发偏慢 | 未采用 |
| 其余 Combine quick 扫描及 W6/W12 等邻域补扫 | **没有新的稳定赢家** | 保留原几何 |

### kernel 与参数实验

| 原编号 | 尝试 | 结论 | 处理 |
|---|---|---|---|
| C1 | 容量 64/128，本地和远端普通 gather 先读两个相邻 4B 向量步再累加（LF4×2） | **有效，已落地** | 之后容量上限扩到 256，见下节 |
| C2 | T4096，普通 gather 扩宽到 16B，必要时回退 8/4B | **无效：回退** | 不采用 |
| C3 | T4096，普通 gather 扩宽到 8B，必要时回退 4B | **未确认** | Combine 本身变慢，完整周期小幅变化；不采用 |
| C4 | T4096，LF4×2 推广到大容量；另试 LF4×4 | **无效：两者均回退** | LF 保留小容量限制 |
| C6 | T4096、W8，远端 chunk 分片 S8 改成 S4 | **无效：回退** | 不采用 |
| C7 | T4096、W8，S8 改成 S16，两阶段 96/64/8 | **有效** | 只有实验 patch（不在仓库中），只在实测配置下验证，不能全局开启或与 Dispatch 收益相加 |
| C8 | QP 增加到 4/8/16/32，同时影响两个 OP | **无效：QP4 持平，其余回退** | 保持 QP2 |
| C9 | T64，Combine 几何改成 64/32/8 | **未确认** | 只有小幅单组差异 |
| C9 | T4096，Combine 几何改成 96/48/8 或 128/64/8 | **无效：前者约持平，后者回退** | 不采用 |
| C12 | T4096，只在本地或只在远端 gather 启用 LF4×2 | **无效：两者均回退** | 不采用单侧 LF |
| C13 | T4096，用 S16/W4 或 S4/W16 替换 S8/W8 | **无效：两者均回退** | 不按 S×W 相等选配置 |
| C7/C13 | 固定 W4，S8 对比 S16 | **无效：约持平** | S16 的收益只覆盖 W8 |
| C7/C13 | 固定 W16，S8 对比 S16 | **未确认**：计时受 CI 编译影响 | 数值通过，性能样本不用于判断 |

### LF 推广（2026-09-20）

C1 之后做了两轮只改 Combine LF 的实验（两边 Dispatch 都是并行实现，只作控制项），数据见 [bw.md](bw.md#epv2-lf-generalize)：

- **broad**（84 组 ABA）：只保留 4 字节对齐检查，放开 topk、dtype、H、capacity、拓扑条件。核心 12 格（BF16 / FP8 E4M3 / packed FP4 输入 × K6/K8 × T64/128，Combine 为 BF16，各 3 组）的 Combine 延迟下降 11.36%–15.54%。H7168 BF16 的 T256 在 K6/K8 各 3 组下降 2.00%/2.22%，是把容量上限放到 256 的依据；T512/T4096 回退 0.91%–2.80%。短行 H128/H192 回退 7.02%/5.64%；BF16 H1152/H2176、FP8 H768/H4352 等尾部余量恰好是一个完整向量步的形状，部分回退 0.94%–5.43%。这些扩展点均为单组诊断。
- **tail**（25 组 ABA）：在 broad 上补一个单向量步的尾部 helper，BF16 H1152/H2176、FP8 H768 和 2×2 BF16 H2176 四个回退点转为改善（−1.17% 到 −10.36%，各 3 组）。该 helper 未进入当前源码。

当前源码的 LF 条件（gfx950、capacity ≤ 256、4 字节对齐）在 capacity ≤ 256 时与 broad 相同，上述短行和尾部回退点没有被排除。

### 静态排除与待验证项

| 原编号 | 方向 | 状态 |
|---|---|---|
| C5 | 压缩非空源指针，再选择累加模板 | **未重跑**：历史快照缺失；原路径已跳过空源 payload 读取，压缩不会减少有效 payload 字节 |
| C10 | 让同一向量步的 expert 读取并行 | **静态排除**：常用 K8 实现已先读取各源再累加；C1 改变的是跨两个向量步的调度 |
| C11 | 只改 `WARP_ACCUM_UNROLL` 让普通 gather 展开 | **静态排除**：普通 wrapper 没有用该参数展开 |
| C14 | 定位 gather、传输、peer 等待与 barrier 成本 | **已做阶段 profile，优化待验证**：尚未完成内部等待归因，没有新的代码收益 |

## 验证

- 当前 kernel（含同步修复）的验证来自 9 月 30 日的重扫：12/12 数值门槛，以及已扫完的 22 个形状各三轮数值检查；另外 10 月 12 日、13 日分别在 mi355-gpu-49/51 和 49/50 上做的双机数值检查（BF16/K8/T128，5 轮，16 个 rank）都通过了。完整 strict 验收尚未执行。
- 单机 EP8 新表：op 直接读表选档，6 个覆盖所有新 geometry 的单机用例（各 3 轮 × 8 rank）通过。
- V2LL EP16 新表：op 直接读表选档，5 个跨机用例（含 `auto` 模式下 T256 自动走 LL，各 3 轮 × 16 rank）通过。9 月 21 日两组 A/B 的全部运行通过了全 rank hidden/weights 预检，但早于同步修复。9 月 20 日冻结的带 tail helper 和 dtype/topk/拓扑条件的版本曾通过 39/39 项 strict 检查（hidden 逐位无不匹配，FP4 接收行 multiset 一致），该结论不覆盖当前源码。
- `1b3e7dba` 的 JSON 通过了 BF16/K6、BF16/K8、FP8/K8 各 21 轮不同 token 档的数值检查（每轮 16 个 rank，含 uniform/skewed/local 路由和 weights），以及每个组合在容量 64/256/4096 下各 128 轮的持续测试；持续测试只检查 hang/fault，不逐轮核对数值。
- CQ 模式一致性、provider 生命周期和 V1 benchmark 入口属于正确性修复，不计入 kernel 优化收益。
- 独立扫描是固定另一 OP 的有限候选集，不是全局最优证明。性能结论限定于实际 T = 容量、QP2、CCQE 关闭的实测条件；LL 和其它平台的结果不用于这里的判定。
