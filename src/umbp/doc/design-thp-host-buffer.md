# 设计：UMBP 主机缓冲区的可选 THP 支持（`UMBP_USE_THP`）

状态：v0.2，已实现（分支 `feat/umbp-thp-host-buffer`，未开 PR），CPU 单测通过，未做集群 / AINIC 注册验证　范围：`mori/src/umbp/local/host_mem_allocator.*` 及其调用方

v0.2（2026-11-09）相对草案的变化：补上本仓库以前的 THP 尝试（#679 / #687）与实测（§1.5）；prefault 改为按 64 MiB 分块并用 `MADV_COLLAPSE` 逐块核实，`REQUIRE_FULL` 在第一块不足时就停（§4.4）；覆盖率判定合并 smaps 与 `MADV_COLLAPSE` 两个来源（§4.5）；`kAnonymousThp` 可以直接请求（§4.1）；n06-21 上的小规模实测（§5.1）。

## 1. 背景与目标

### 1.1 问题

UMBP 的主机 DRAM 池要向 RDMA NIC 注册。部分 NIC（MI355X 上的 Pensando AINIC/ionic 等）对内存注册有硬限制：

- 一个 MR 只有一种页大小；混入任何一页 4 KiB 页，整个 MR 都按 4 KiB 页计。
- 单个 MR 的页表项数有上限，每块 NIC 的总项数也有上限。
- 因此大块缓冲区必须**几乎 100% 由 2 MiB 页支撑**，否则注册返回 ENOMEM（InferenceX #4045 中 Jasen 的实测：8 GiB 里仅 0.02% 为 4 KiB 页即失败）。

目前 UMBP 只有一种方式拿到 2 MiB 页：**预留的 HugeTLB 页**（`MAP_HUGETLB` / `MFD_HUGETLB`）。它确定性强，但要求运维提前设置 `vm.nr_hugepages`；预留不足时 `LogHugepageFallbackOnce` 会回退到 4 KiB 页，随后 NIC 注册失败。

### 1.2 目标

增加**可选**的透明大页（THP）路径：

- 通过环境变量 `UMBP_USE_THP=1` 开启，**默认关闭**，关闭时行为与现在逐字节一致。
- 不能预留 HugeTLB 的环境（共享集群、无 root）也能尽量获得 2 MiB 页。
- THP 是 best-effort，所以必须**能量化实际覆盖率**，并能在启动时快速失败，而不是等到 NIC 注册阶段才失败。

### 1.3 非目标

- 不替代 HugeTLB；HugeTLB 仍是需要确定性时的首选。
- 不在库内清 page cache、触发 compaction、修改 `/sys/kernel/mm/transparent_hugepage/*`（需要 root，且影响宿主）。
- 第一版不支持 memfd/shm 路径（见 §5.2）。
- 不改 Mooncake；不改 NIC 固件相关检查。

### 1.4 相关讨论与链接

本设计源于 InferenceX #4045 中关于"预留 HugeTLB 还是用 THP"的讨论。

**InferenceX #4045**（GLM-5.2 MI355X ATOM PD 分离 + Mooncake Store 基准）
- PR：https://github.com/SemiAnalysisAI/InferenceX/pull/4045
- 该 PR 在 head 节点的 `setup.sh` 里通过 `HEAD_NODE_HUGEPAGES_PER_NUMA` 预留 2 MiB HugeTLB 页，并设置 `MC_STORE_USE_HUGEPAGE=1`。
- review 线程（`setup.sh` 第 71 行）：
  - functionstackx 质疑：vLLM/SGLang 用 Mooncake Store 时都不预留大页，并指出 Mooncake #4393 / #5151 或许能解决：https://github.com/SemiAnalysisAI/InferenceX/pull/4045#discussion_r4490417205
  - Jasen2201 的回复（NIC 三条限制、实测数据、为何仅靠 madvise/THP 不够、可选的退路）：https://github.com/SemiAnalysisAI/InferenceX/pull/4045#discussion_r4505513616
  - functionstackx 的再回复（称已与 AWS 工程师解决过同类问题，细节转到 Slack）：https://github.com/SemiAnalysisAI/InferenceX/pull/4045#discussion_r4505734566
- Slack 中的观点（Oren）：AWS 已用 madvise 请求 THP 解决 NIC 页表不足的问题，集群已启用 madvise，除非 ROCm Mooncake 栈有根本问题，否则 madvise 应该可行；vLLM Engram offloading 也使用 madvise。（该观点来自截图，未附链接，我们未独立核实 Engram 部分。）

**Mooncake PR**
- #4393（已合并，作者 whn09）：https://github.com/kvcache-ai/Mooncake/pull/4393 ——"[Store] Back EFA host segments with THP interleaved across NUMA nodes"。即常说的"AWS 解决 THP 的 PR"，仅作用于 `protocol=efa`。
- #5151（未合并）：https://github.com/kvcache-ai/Mooncake/pull/5151 ——"[Store] Request THP for base-page RDMA global segments"。把同样思路扩展到 RDMA；其 MI355X 实测见 https://github.com/kvcache-ai/Mooncake/pull/5151#pullrequestreview-5821333852 。
- InferenceX #3523（#4393 的起因，8 rank × 190 GB 时 Store 初始化 12–23+ 分钟）：https://github.com/SemiAnalysisAI/InferenceX/pull/3523

**其他相关**
- sglang UMBP host allocator（目前只有 HugeTLB，无 THP）：https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/umbp/umbp_host_allocator.py
- sglang #27267（2026-11-09 合并，host pool 的 hugetlb 预检与 `SGLANG_HUGEPAGE_MODE`）：https://github.com/sgl-project/sglang/pull/27267
- sglang 中确实使用 `MADV_HUGEPAGE` 的位置：`python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py`（best-effort，不涉及 NIC 注册）。

**讨论进展（2026-11-10 核实）**
- functionstackx 在 11-08 把 InferenceX #4045 的标题改为 "[Waiting for Mooncake ROCm to use madvise for THP requests for the Pollara NIC just like how AWS engineers solved it on Mooncake EFA on Nitro EFA NIC]"，即要等 Mooncake 的 RDMA 路径也请求 THP，而不是在 `setup.sh` 里预留 HugeTLB。没有找到 SA 在 Mooncake 仓库提的 issue；对应的改动是 #5151。
- AWS 一侧的其他依据：#4564（已合入，p6-b200 上每节点 1536 GiB 用 THP，没拿全时自动退回 4 KiB 方案并把注册拆到多张 NIC）；#5018（已合入，文档：r8i 上 1 TB 用 THP 注册成功；能控制宿主机就用 HugeTLB，容器里用 THP，碎片化时先压缩内存）。
- Jasen2201 在 #5151 的 review（11-09，https://github.com/kvcache-ai/Mooncake/pull/5151#pullrequestreview-5821333852）给出 MI355X 实测（ionic 26.03.3.001，内核 6.8）：每节点两个 768 GiB 段，页缓存少的节点约 20 s 全部是大页、注册成功；每 NUMA 节点约 1 TiB 页缓存时 0.2%–34% 停在 4 KiB、注册报 EINVAL/ENOMEM，填页耗时 1 分钟到 1 小时以上。#5151 第二轮修改后在描述中写明：THP 路径要求先清页缓存，否则用 `MC_STORE_USE_HUGEPAGE`。
- **争论的核心不在于 THP 能不能用**（干净内存上双方都认可能用），而在于：
  1. **ionic 不能降级**：EFA 的 4 KiB 额度大（每 NIC 约 2200 万项），没拿全时还能拆分注册继续跑；ionic 一个 MR 混入一页 4 KiB 就整体按 4 KiB 计，单 MR 上限 2 GiB、每 NIC 约 100 万项，所以必须 100% 大页。
  2. **节点状态**：MI355X 节点上页缓存大（读 checkpoint 等），每个 owner 要一次拿到几百 GiB 的大页。
  3. **失败时机与节点准备的归属**：THP 失败发生在作业开始很久之后；HugeTLB 在 host_setup 中约 1 分钟预留完，不够当场失败。SA 不希望 recipe 依赖宿主机预留。functionstackx 称有误解，细节在 Slack，公开内容到此为止。

**要点小结**：#4393 证明了 EFA 上 "对齐 + `MADV_HUGEPAGE` + interleave" 可行，且其作者明确写到 hugetlb 预留仍是"最快且最确定"的方案；#5151 在 MI355X 上的实测表明，page cache 占用大时 THP 覆盖率可能很低。因此本设计把 THP 做成可选，并增加覆盖率校验。

### 1.5 本仓库以前的 THP 尝试

- **ROCm/mori#679**（2026-09，分支 `perf/umbp-tier-thp` @ `09a88076`）：在 `ApplyPostMappingPolicies` 里对所有非 hugetlb 映射**默认**调用 `madvise(MADV_HUGEPAGE)`（`UMBP_DRAM_TRANSPARENT_HUGEPAGES=0` 关闭），目的是缩短 `hipHostRegister` 的启动时间。合入后被 **#687 整个撤回**，当前 main 中没有 `MADV_HUGEPAGE`。
- 撤回依据是工作记录 §12.53（work-docs `projects/umbp/evidence/22-tier-page-backing.md`）的实测，1.5 TB DRAM tier：
  - 只要最终是 2 MiB 页，注册耗时就与来源无关：THP 98.2 s（n08-33），hugetlb 97.1 s（n06-21），4 KiB 为 242–620 s。
  - **THP 的问题在获取阶段**：`defrag=[madvise]` 下缺页同步压缩，两个集群三次运行都在填到 0.8–1.2 TB 后变慢到 0.2–0.4 GB/s（crusoe 22 分钟才 populate 完；n06-21 9 分钟只到 85%，被掐掉）。
  - 含建池时间，hugetlb 对 4 KiB 仍快 1.7–3.3 倍，且注册时间稳定（97–98 s 对 265–620 s）。先 `drop_caches` 再建池是净损失。
- 本设计与 #679 的区别：默认关闭；目的从缩短注册时间改为满足 AINIC 的 MR 页表项限制；增加覆盖率测量与 `REQUIRE_FULL`；增加 NUMA interleave（#679 与 §12.53 都没有，当时也还没有 #718 的分节点 prefault）。
- 未验证的推测：§12.53 中的拐点（n06-21 约 1.14 TB）接近单个 NUMA 节点的容量（约 1.58 TB），可能部分来自"先填满一个节点"，这正是 #4393 用 interleave 解决的问题。但 §5.1 的 11-09 实测表明，在页缓存很大的节点上，即使 interleave 或分节点 prefault，16 GiB 也只有 40–56% 覆盖率，所以 NUMA 放置不能单独解决这些节点上的问题。

## 2. 参考：Mooncake #4393

kvcache-ai/Mooncake #4393（2026-09-30 合并，仅 EFA 路径）新增 `allocate_buffer_thp_interleaved()`，要点：

1. `mmap(MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)` 多映射 2 MiB，再裁掉头尾，得到 **2 MiB 对齐**的区间，使每个 PMD 都可能被 THP 覆盖。
2. `madvise(MADV_HUGEPAGE)`，不依赖系统默认值或 `GLIBC_TUNABLES`；失败只警告。
3. 若任务没有显式内存策略，则对所有允许的 NUMA 节点 `mbind(MPOL_INTERLEAVE)`。原因：默认本地策略下一个节点先被填满，之后每次 THP 缺页都在该节点上做 direct compaction 且大多失败，8 rank × 190 GB 时 Store 初始化耗时 12–23+ 分钟。若有 `numactl --membind` 等显式策略则不改。
4. 要求大小为 2 MiB 的整数倍。

它**没有**做的：不校验 THP 覆盖率，不保证拿到大页，不覆盖 RDMA 路径（Mooncake #5151 尝试覆盖 RDMA，尚未合并；其 MI355X 实测 209 GiB 段仅 2.8 GiB 为 THP，因为每个 NUMA 节点约 1 TiB page cache 使空闲连续内存不足）。

本设计吸收 1–4，并补上覆盖率校验与失败策略。

## 3. 现状（代码事实）

`HostMemAllocator::Alloc` 按 `HostBufferOptions::backing` 分派：

| backing | 实现 | 说明 |
|---|---|---|
| `kAnonymous` | `AllocAnonymous`：`mmap` 匿名 4 KiB 页 | THP 要改的地方 |
| `kAnonymousHugetlb` | `MAP_HUGETLB` | 失败时回退到 `AllocAnonymous` |
| `kAnonymousShm` / `kAnonymousShmHugetlb` | memfd + SCM_RIGHTS | standalone 进程模式用 |

- 所有路径最后走 `ApplyPostMappingPolicies`：先按 `numa_node` 做 `mbind`（bind/preferred），再 `PrefaultPages`（`MADV_POPULATE_WRITE`，失败则手工触页）。
- `PrefaultPages` 的 stride：hugetlb 用 `actual_alignment`，其余用 4 KiB。多线程时按 stride 切块，并可按 NUMA 节点 `spread_prefault_across_nodes`（first-touch 摊平）。
- `HostBufferHandle` 记录 `actual_backing` 和 `actual_alignment`。`nic_limits.cpp` 的 `WarnIfNicRegionExceedsAinicLimit` 按"实际页大小"计算页表项数。
- 调用方：`page_backend.cpp`（DRAM 池，`use_hugepages` → `kAnonymousHugetlb` 否则 `kAnonymous`）、`distributed_client.cpp`（ranged scratch）、`ssd_backend.cpp`（staging）、pybind（`UMBPHostMemAllocator`，sglang 使用）。
- sglang 的 `umbp_host_allocator.py`：非 standalone 时 `SGLANG_HICACHE_HOST_HUGEPAGE`（默认 1）→ `AnonymousHugetlb`，设为 0 → `Anonymous`；standalone 时对应 `AnonymousShm*`。并且仅当 `actual_backing == Anonymous` 时才提示"被降级为 4 KiB"。
- 项目内 `UMBP_*` 环境变量统一记录在 `runtime-env-vars.md`；C++ 侧多为函数内 `static const` 缓存首次读取值。

## 4. 设计

### 4.1 对外接口

**环境变量**

| 变量 | 默认 | 含义 |
|---|---|---|
| `UMBP_USE_THP` | `0` | `1`：`kAnonymous` 路径改用 THP 映射。 |
| `UMBP_THP_REQUIRE_FULL` | `0` | `1`：覆盖率未达 100% 时分配失败（返回无效 handle），而不是继续运行。仅在 `UMBP_USE_THP=1` 且 prefault 开启时有意义。 |

解析沿用 `runtime-env-vars.md` 的语义：未设置/空 → 默认；非法值 → 默认并每个变量 WARN 一次；进程内缓存。

**`HostBufferOptions` 新增字段**（保证单测不依赖进程环境）：

```cpp
enum class ThpMode : int { kFromEnv = 0, kOff = 1, kOn = 2 };
ThpMode thp = ThpMode::kFromEnv;   // kFromEnv 读 UMBP_USE_THP
bool thp_require_full = false;     // 与 UMBP_THP_REQUIRE_FULL 取或
```

**`HostBufferHandle` 新增字段**：

```cpp
double thp_coverage = -1.0;  // 0..1；-1 表示未测量（未请求 THP 或未 prefault）
```

**`HostBufferBacking` 新增** `kAnonymousThp = 4`。`pybind_umbp.cpp` 增加同名枚举值 `AnonymousThp`，handle 暴露 `thp_coverage`。也可以直接请求 `backing = kAnonymousThp`：等价于 `kAnonymous` 加 `thp = kOn`，不看环境变量，所以 pybind 的 `alloc(size, AnonymousThp)` 无需新增参数就能单独开启 THP（回答了 §9 第 4 条）。宿主 THP 为 `[never]` 时结果仍是 `kAnonymous`。

**其他新增接口**：`HostBufferBackingName()`（日志用）；`ThpCoverageFromSmaps(std::istream&, begin, size)`（§4.5 的纯函数，供单测）；`SetThpEnabledModeForTesting()`（只供测试，伪造 sysfs `enabled` 的取值）。

`actual_alignment` 的约定：THP 路径下，覆盖率 ≥ 100% 时为 2 MiB，否则为 4 KiB。这样 `nic_limits` 的页表项预估自动正确，不需要改它。

### 4.2 分配流程（`AllocAnonymousThp`）

触发条件：`opts.backing == kAnonymous`（或 HugeTLB 回退到 `kAnonymous`）且 THP 开启。在 `AllocAnonymous` 入口分流（原实现改名 `AllocAnonymousBasePages`），HugeTLB 失败后的回退也自然走到 THP，而不是 4 KiB；此时 `LogHugepageFallbackOnce` 的提示改为"回退到 THP"。`Alloc()` 的 switch 只为直接请求 `kAnonymousThp` 加了一个 case。

1. **预检**（仅一次，结果缓存）：读 `/sys/kernel/mm/transparent_hugepage/enabled`。
   - `[never]` 或文件不存在（内核未编入 THP）：WARN 一次并退回普通 4 KiB 路径（`madvise` 不会有效果）。
   - `[always]` 或 `[madvise]`：继续。
   - 同时读 `defrag` 并记入日志（`madvise`/`defer+madvise` 会对 `MADV_HUGEPAGE` 区域做 direct compaction，影响启动时延）。
2. **大小**：`mapped_size = AlignUp(size, 2 MiB)`（沿用 `AlignUpChecked`）。
3. **对齐映射**：与 #4393 相同，`mmap(MAP_NORESERVE)` 多映射 2 MiB，`munmap` 裁掉头尾，返回 `[ptr, ptr + mapped_size)`。`Free` 对该区间 `munmap(mapped_size)`，无需改动。
4. `madvise(ptr, mapped_size, MADV_HUGEPAGE)`；失败 WARN 并继续（覆盖率校验会暴露结果）。
5. **NUMA 放置**（见 §4.3）。
6. **`numa_node` 绑定**：照常走 `ApplyPostMappingPolicies`（含 `require_numa_binding` 的失败处理），但不在其中 prefault。
7. **prefault + 逐块核实**：按 64 MiB 分块 populate，每块后 `MADV_COLLAPSE`（见 §4.4）。
8. **覆盖率判定**（见 §4.5），写入 `handle.thp_coverage`；`actual_backing = kAnonymousThp`，`actual_alignment` 只在完全覆盖时为 2 MiB。每次分配打一行 INFO（大小、prefault 耗时、覆盖率、是否 interleave）。
9. 未完全覆盖：WARN 一次，日志包含覆盖率、原因（缺页没拿到大页 / `MADV_COLLAPSE` 的 errno）以及 `MemFree`/`Cached`/`AnonHugePages`/`HugePages_*` 摘要，并提示补救手段（预留 HugeTLB；由运维在启动前 `compact_memory`）。不建议 `drop_caches`：§12.53 九之三 实测它对 hugetlb 建池是净损失，对 THP 未测。若 `thp_require_full`，则 ERROR、释放并返回无效 handle。

### 4.3 NUMA 策略

三种情形，按优先级：

1. `opts.numa_node >= 0`：保持现有行为（`mbind` bind/preferred），**不**加 interleave。
2. `numa_node < 0` 且 `spread_prefault_across_nodes`：保持现有 first-touch 摊平，**不**加 interleave。THP 下需保证每个 prefault 分块按 2 MiB 对齐（§4.4），否则一个大页会被两个线程/节点分摊。
3. `numa_node < 0` 且未 spread：若 `get_mempolicy` 显示任务策略为 `MPOL_DEFAULT` 且允许节点数 > 1，则 `mbind(MPOL_INTERLEAVE)` 覆盖允许的所有节点；若已有显式策略（如 `numactl --membind`）则不改。这就是 #4393 避免"单节点被填满后 compaction 风暴"的做法。

`mbind` 失败只 WARN，沿用 `ApplyPostMappingPolicies` 里现有的日志风格。

### 4.4 Prefault：分块、逐块核实、尽早失败

`PrefaultPages` 的切分逻辑抽成 `ForEachPrefaultSlice`（线程数、按节点轮转、`stride` 对齐的切分都不变），普通路径和 THP 路径共用。THP 路径（`PrefaultThp`）：

- stride = 2 MiB，所以每个线程的切片都从 2 MiB 边界开始，分节点 prefault 时一个大页不会被两个节点分摊。
- 每个切片再按 **64 MiB** 分块：先 `MADV_POPULATE_WRITE`（失败时回退到逐页触碰，同现有实现），再对该块 `MADV_COLLAPSE`（Linux 6.1+）：
  - 块已全是大页：内核把"已经是 PMD 映射"当作成功，代价很小。
  - 块里有 4 KiB 页：内核同步尝试合并，成功则补上，失败返回 `ENOMEM`/`EAGAIN`，这是逐块的**精确**结果。
  - `EINVAL`/`ENOSYS`（内核太旧，或进程 `PR_SET_THP_DISABLE`）：停止使用 `MADV_COLLAPSE`，只靠 smaps。
- 合并失败后：有 `require_full` 就让所有线程在下一块之前停下；没有则不再对后续块调用 `MADV_COLLAPSE`（避免每块都同步重试），只按缺页的结果继续。
- 分块大小决定 `REQUIRE_FULL` 多快失败。§5.1 的实测：n06-21 上 16 GiB、8 线程，块为 1 GiB 时 70 s 才失败，改为 64 MiB 后 **4.4 s**。

`prefault = false` 时不会产生任何物理页，覆盖率无法测量：`thp_coverage = -1`，`actual_alignment` 为 4 KiB，只打印一次提示说明"未 prefault，THP 覆盖率未知，NIC 注册时可能失败"。`thp_require_full` 在此情形下无效，并在同一条提示里说明。

### 4.5 覆盖率测量

使用 `/proc/self/smaps`，找出与 `[ptr, ptr + mapped_size)` 相交的 VMA，累加 `AnonHugePages`，除以 `mapped_size`：

- 无需特权（`pagemap`/`kpageflags` 需要 `CAP_SYS_ADMIN`）。
- 近似性：相邻且属性相同的 VMA 可能被内核合并，导致统计包含区间外的页。缓解方式：只对该区间自身的 VMA 取 `Size`/`AnonHugePages` 的比例，并在区间不是单独 VMA 时标注"近似"；若后续发现误差显著，可在映射两侧各保留一页 `PROT_NONE` guard 来阻止合并（以及相应的 `munmap` 对称处理）。
- 实现上，区间外的 VMA 按重叠比例计入并标记 `exact = false`；日志带"smaps approximate"。目前没有加 guard page。
- `smaps` 读不到或没有相交 VMA：smaps 一侧记为"未测量"。
- 判定阈值 100% 的含义：由于大小已向上对齐到 2 MiB 且 prefault 会触及全部页，覆盖率应恰为 100%；允许 1e-9 的浮点误差。

**判定**（两种来源，按可靠程度取）：

| smaps | `MADV_COLLAPSE` | `thp_coverage` | 完全覆盖 |
|---|---|---|---|
| 精确（缓冲区自成 VMA） | 任意 | smaps 值 | smaps = 100% 且未提前停止 |
| 近似或读不到 | 每块都成功 | 1.0 | 是 |
| 近似 | 不支持或失败 | smaps 近似值 | 否 |
| 读不到 | 不支持或失败 | -1 | 否 |

`MADV_COLLAPSE` 逐块成功是精确的（内核只在块内每个 2 MiB 都是 PMD 映射时返回 0），所以 smaps 不精确时以它为准。

### 4.6 与现有 HugeTLB 路径的关系

| 场景 | `UMBP_USE_THP=0`（默认） | `UMBP_USE_THP=1` |
|---|---|---|
| backing = HugeTLB，预留足够 | HugeTLB | HugeTLB（THP 不参与） |
| backing = HugeTLB，预留不足 | 回退 4 KiB（现状） | 回退 **THP**，再按覆盖率告警/失败 |
| backing = Anonymous | 4 KiB | THP |
| backing = Shm / ShmHugetlb | 现状 | 现状（见 §5.2） |

即 THP 只替换"本来会落到 4 KiB 页"的那条路径，不会削弱已经工作的 HugeTLB 配置。

### 4.7 调用方与 sglang

- `page_backend.cpp`、`distributed_client.cpp`、`ssd_backend.cpp` 都通过 `HostBufferOptions` 调用 allocator；因为 THP 开关由 `kFromEnv` 默认读取，**这些调用方无需修改**。
- 日志：`distributed_client.cpp` 的 ranged scratch 两行改用 `HostBufferBackingName()`，THP 时带上覆盖率（如 `thp, 99.500% hugepages`）。DRAM tier（`page_backend.cpp`）不需要改，allocator 自己的 INFO 行已经给出每块缓冲区的覆盖率和 prefault 耗时。
- sglang `umbp_host_allocator.py`：
  - 单进程模式下，要走 THP 需要 `SGLANG_HICACHE_HOST_HUGEPAGE=0` 并设置 `UMBP_USE_THP=1`；或保持默认的 HugeTLB，仅在其预留不足时回退到 THP。
  - "demoted" 判断目前是 `actual_backing == Anonymous`，需要同时识别 `AnonymousThp` 且 `thp_coverage < 1` 的情形，给出"THP 覆盖率不足"的警告。该 Python 侧改动属于 sglang 仓库，应单独提 PR，并等 mori 先发布新枚举。
  - pybind 的 `alloc(...)` 没有新增参数；sglang 若要不依赖环境变量，可以直接传 `UMBPHostBufferBacking.AnonymousThp`（§4.1）。

## 5. 限制与风险

### 5.1 THP 本质上是 best-effort

覆盖率取决于分配当时的连续空闲内存。碎片化或 page cache 占用大时覆盖率会不足（#5151 的 MI355X 实测）。本方案无法消除这点，只做到：测得出、报得清、可选择启动即失败。需要确定性时仍应预留 HugeTLB。

**本机实测（2026-11-09，smci355 n06-21 宿主机，内核 6.8.0-138，THP `enabled=[madvise] defrag=[madvise]`，MemFree 约 490 GB，Cached 约 1.7 TB，节点上有其他用户的作业；每格 n=1。探针源码与命令见 work-docs `projects/umbp/evidence/data/58-thp-host-buffer-20261109/`）：**

| 16 GiB，8 线程 | 耗时 | 覆盖率 | 结果 |
|---|---:|---:|---|
| 4 KiB（`thp = kOff`，分节点 prefault） | 0.8 s | — | 正常 |
| THP + interleave | 167.8 s | 40.4% | 告警，`MADV_COLLAPSE` 返回 `ENOMEM` |
| THP + 分节点 prefault | 137.1 s | 53.7% | 同上 |
| 只 `MADV_HUGEPAGE` + populate，不 collapse | 166.4 s | 55.3% | 耗时来自缺页时的同步压缩，不是 `MADV_COLLAPSE` |
| THP + `REQUIRE_FULL`，1 GiB 分块 | 69.9 s | — | 分配失败 |
| THP + `REQUIRE_FULL`，64 MiB 分块 | **4.4 s** | — | 分配失败 |

结论：
- 在页缓存很大、长期运行的共享节点上，THP 连 16 GiB 都拿不全，而且比 4 KiB 慢约 200 倍；NUMA 放置不改变这一点。这与 §1.5（1.5 TB）和 #5151 一致。
- 在这类节点上，`UMBP_USE_THP=1` 解决不了问题，只是让 NIC 注册失败提前到分配阶段并说明原因；要大页仍需 HugeTLB 预留。**主要价值在干净节点**：§1.5 中 n08-33（开跑 Cached 3.5 GB）用 THP 填 1.5 TB 用了 142 s，注册 98.2 s，与 HugeTLB 相同，即不预留也能拿到 2 MiB 页。
- 小规模（≤ 66 MiB）的单测分配在同一节点上都是 100%：碎片化在需要大量连续内存时才显现。
- 干净节点上本实现未单独测；按 §1.5 的 n08-33 数据可以拿全。

### 5.2 shm 路径不在第一版范围

`kAnonymousShm*` 基于 memfd，THP 要依赖 `/sys/kernel/mm/transparent_hugepage/shmem_enabled`，且 `madvise(MADV_HUGEPAGE)` 对 shmem 的作用不同于匿名内存。standalone 进程模式在 `UMBP_USE_THP=1` 时保持原行为并打印一次 INFO 说明。后续如需支持，需单独验证。

### 5.3 其他风险

- **启动时延**：`defrag=madvise` 时 direct compaction 会使 prefault 显著变慢（§5.1：16 GiB 约 140–170 s）。每次分配的 INFO 行记录 prefault 耗时；不带 `REQUIRE_FULL` 时不会提前停止，大缓冲区可能要等很久，部署时应与 `REQUIRE_FULL` 一起用，或先在目标节点上量一次。
- **容器内的 NUMA 策略**：Docker 默认 seccomp 拒绝 `get_mempolicy`/`mbind`，interleave 会被静默跳过（只在宿主机或放开 seccomp 的容器里生效）；`numa_node >= 0` 的绑定路径本来就受同样限制。
- **内存记账**：`MAP_NORESERVE` 不预占 commit；cgroup 内存限制下 THP 回退行为需在实际环境确认。
- **进程环境缓存**：与其他 `UMBP_*` 一致，首次使用后缓存；测试通过 `HostBufferOptions::thp` 绕开。
- **fork**：THP 区域在 fork 后的 COW 行为与普通匿名内存相同，不引入新要求。
- **mbind 与 spread 并存**：§4.3 已规定互斥，避免 VMA 策略覆盖 first-touch。

## 6. 实现清单

已全部完成（第 6 项除外），分支 `feat/umbp-thp-host-buffer`（基于 main `af56d9b5`，本地未推送）。实际改动约 C++ 480 行、测试 370 行，比草案估计多，主要是分块核实（`ForEachPrefaultSlice` / `PrefaultThp`）、日志与测试。

1. `include/umbp/local/host_mem_allocator.h`：`ThpMode`、`HostBufferBacking::kAnonymousThp`、`HostBufferOptions::{thp, thp_require_full}`、`HostBufferHandle::thp_coverage`。
2. `local/host_mem_allocator.cpp`：THP 预检、`AllocAnonymousThp`（对齐映射、`MADV_HUGEPAGE`、条件 interleave）、`PrefaultThp`（64 MiB 分块 + `MADV_COLLAPSE`）、`MeasureThpCoverage`（smaps）、覆盖率判定、`AllocAnonymous` 入口分流、HugeTLB 回退指向同一入口、shm 路径的一次性 INFO。
3. `pybind/pybind_umbp.cpp`：枚举 `AnonymousThp`；handle 暴露 `thp_coverage`。
4. `distributed_client.cpp`：日志区分 THP。
5. 文档：`runtime-env-vars.md` 增加两个变量。
6. （另一仓库，未做）sglang `umbp_host_allocator.py` 的告警判断。

## 7. 测试计划

单测加在 `tests/cpp/umbp/local/test_host_mem_allocator.cpp`（沿用其"plain asserts"风格）：

1. `thp = kOff`：行为与现状一致（backing 为 `kAnonymous`，`thp_coverage == -1`）。
2. `thp = kOn`：返回的指针 2 MiB 对齐，`mapped_size` 为 2 MiB 倍数，`Free` 正常，内容可读写。
3. 大小不是 2 MiB 倍数：向上取整。
4. 无显式策略且多 NUMA 节点：区域 VMA 策略为 `MPOL_INTERLEAVE`；设置 `MPOL_BIND` 任务策略后不覆盖（参照 #4393 的测试）。
5. `numa_node >= 0` 或 `spread_prefault_across_nodes`：不安装 interleave。
6. HugeTLB 预留为 0 的环境下 `kAnonymousHugetlb` + THP：回退到 THP 而非 4 KiB。
7. `THP enabled=never` 的环境（可通过注入预检结果测试）：退回 4 KiB 并 WARN。
8. `prefault=false`：`thp_coverage == -1`，`thp_require_full` 不使分配失败。
9. 覆盖率解析：对 smaps 文本样例做纯函数测试（含相邻 VMA 与解析失败），不依赖宿主 THP 状态。
10. 覆盖率不足与 `thp_require_full` 的失败路径：用 `prctl(PR_SET_THP_DISABLE)` 让本进程所有缺页都拿 4 KiB 页，确定性地得到覆盖率 0、对齐 4 KiB、`REQUIRE_FULL` 失败。小缓冲区的 100% 路径按宿主实际结果断言（满则对齐 2 MiB，否则必须失败）。
11. 直接请求 `kAnonymousThp`（`thp = kOff` 也生效）；`HostBufferBackingName`。

单测结果（2026-11-09）：n06-21 宿主机上 g++ 直接编译，全部通过，THP 相关用例无跳过；mori-dev 容器内 CMake 构建（`BUILD_UMBP=ON`），全部通过，interleave 用例因容器 seccomp 跳过。同一构建中 `umbp_core`、`libmori_umbp_pybinds.so` 编译通过，`test_standalone_shm_ipc` 11 项通过。

Python：`tests/python/umbp/test_umbp_host_mem_allocator.py` 增加两项：直接请求 `AnonymousThp`（按宿主 THP 状态断言）、`Anonymous` 的 `thp_coverage == -1`。用临时 shim 加载新构建的 UMBP 扩展（IO 类型借镜像内 mori 9e851ae5 的 `libmori_pybinds` 注册）运行：分配器相关用例全部通过。注意：设置 `UMBP_USE_THP=1` 运行时，原有 `test_anonymous_alloc_free_round_trip` 会失败，因为环境变量按设计把 `Anonymous` 变成了 THP。CI 不设该变量。

### 集群验证（MI355X + AINIC）

1. 在 THP `enabled=madvise` 的干净节点上，用 `UMBP_USE_THP=1` 分配 ≥200 GiB，记录 `thp_coverage` 与 NIC 注册是否成功。
2. 在 page cache 很大（每 NUMA 节点约 1 TiB）的节点上重复，确认覆盖率不足时 WARN/`REQUIRE_FULL` 失败按预期触发，且失败发生在分配阶段而非注册阶段。
3. 对比 HugeTLB 路径的启动时延与成功率，作为是否值得推广 THP 的依据。

## 8. 发布与兼容

- 默认关闭，不改变现有 recipe 和 CI 行为。
- 新增枚举值 `kAnonymousThp = 4` 是追加，不改变已有数值；使用 `actual_backing == kAnonymousHugetlb` 的现有判断不受影响。
- 开启后若 sglang 仍用旧版 `umbp_host_allocator.py`，只是不会提示"THP 覆盖率不足"，功能不受影响。

## 9. 未决问题

1. 是否需要 `UMBP_THP_REQUIRE_FULL` 之外的"最小覆盖率阈值"？NIC 要求 100%，目前认为没有必要。
2. smaps 的 VMA 合并误差是否需要 guard page 方案——已实现近似版；smaps 不精确时以逐块 `MADV_COLLAPSE` 为准（§4.5），guard page 暂不需要。
3. HugeTLB 预留不足时"回退到 THP"是否应默认启用（仅在 `UMBP_USE_THP=1` 时），还是需要单独开关——目前按 §4.6 的表格处理。
4. ~~是否把 `thp` 选项暴露给 pybind 的 `alloc(...)`~~：不加参数，直接请求 `AnonymousThp` 即可（§4.1）。
5. 是否值得继续：§5.1 表明在页缓存大的共享节点上 THP 拿不全大缓冲区。是否在干净节点上做 §7 的集群验证、是否推上游，需要先决定目标部署环境是否能保证"干净启动"。
6. 不带 `REQUIRE_FULL` 时大缓冲区的 prefault 可能要几十分钟；是否需要时间上限（例如超时后停止使用 `MADV_HUGEPAGE`，剩余部分按 4 KiB 填）？目前没有做。
