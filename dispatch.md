# InterNode V2：Dispatch / Combine 优化结论

本文按两个OP记录本次尝试、效果和是否采用；**带宽、延迟及前后对照统一放在 [bw.md](bw.md)**。

## 当前实现与最新复测（2026-09-22 整理）

当前普通 v2 的 Dispatch 在去重后并行分配目标 slot。普通 Combine 在 gfx950、声明 capacity≤256 且行/staging stride 均为4字节整数倍时，直接调用已有的 `WarpAccumLF<T,4,2>`；其余情况使用 `WarpAccum<T,4>`。当前实现没有独立的向量尾部 helper，也没有额外的类型、topk 或节点拓扑选择条件；其它架构不启用这条 LF 路径。具体代码见 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp)。

9月21日已对当前 kernel 完成 BF16→BF16、FP8→FP8 的[固定 record geometry 对照](bw.md#epv2-recorded-geometry-20260921)，并分别完整构建 stash HEAD 与恢复改动后的 Python/native wheel，完成[默认分支 A–B–A 对照](bw.md#epv2-default-branch-20260921)。后者包含54个 dtype/token case，均通过数值预检，测量中未观察到竞争负载。两项实验均强制普通 v2、使用 g17/g09、EP16、H7168、K8、QP2，关闭 CCQE；详细数据及边界以对应章节为准。FP8 Dispatch 存在部分回退，不能把所有改动概括为全面加速。

HIP op 默认读取已有 geometry 记录，普通 v2 与 v2_ll 分别查表；公开 API 的 family 默认仍是 `auto`。当前包内没有 FP8 Combine 记录，其默认 geometry 保持96/64/8；此前实验补调的 FP8 Combine 配置尚未进入默认表。

下文保留此前实验原始说明，其中“当前”“final”“最终”均指该段当时冻结的版本。9月20日的额外类型/拓扑限制、向量尾部 helper 和 gfx942 策略属于历史实现，不描述上面的当前源码。实验归档目录及其原始日志不包含在本仓库中，指向 `../mori_epv2_recheck_20260917/` 的链接用于本地工作区查阅。

## 历史实验记录

2026-09-21：LF选择和尾部函数原文合回 `ep_internode_kernel.hpp`，独立helper头文件已删除。下述验收记录对应9月20日冻结版本，本次文件整理未重跑GPU测试或性能扫描。

**2026-09-20 历史完整 kernel 对照已完成**：以此前扫描并复测确认的同一份最优 JSON 为两边配置，对比 HEAD `fac2e18e` 的串行 Dispatch slot / 普通 Combine 与当时冻结的并行 slot / 条件 LF4×2。BF16/K8、T16–4096 共九档，每档三组 A–B–A，最终81次均通过数值检查且无并发记录。Dispatch T16–2048、Combine T16–128 三组均改善；T4096 两阶段整体持平。完整带宽、延迟、配置和证据见 [bw.md](bw.md) 顶部。该表的B是当时冻结的小容量LF版本，不能按后续cap≤256策略重新解释。普通Combine的后续off→broad 84组与off→tail 25组性能实验也已汇总；当前final源码已选定cap≤256及局部向量尾部，**39项final strict已通过，两端编译/缓存报告已验证；完整ELF等价不作声明**。最终策略与证据入口见 [bw.md最终策略段](bw.md#epv2-final-policy)。以下保留此前各项实验和JSON调优记录，按各自基线和冻结版本解释。

上述完整对照及早期JSON复核的实测范围：pit2-p03-g17 + pit2-p03-g09，MI355X/gfx950、ionic、EP16、experts/rank16，普通V2、H7168、QP2，**`MORI_DISABLE_IONIC_CCQE=1`**。本轮复核commit `1b3e7dba`的JSON，覆盖BF16→BF16的K6/K8及FP8→BF16的K8；性能测试的实际token数等于声明容量。下方原编号的kernel实验来自上一轮BF16/K8、两阶段96/64/8基线，其数值与本轮配置对照分别记录。LL和其它平台的结果不用于这里的优化判定。

`B/R/W`分别表示总block数、RDMA block数、每block的warp数；`S`是每chunk的分片数，Dispatch和Combine各自独立。下文保留原编号1–9、C1–C14，便于查原始记录。

- **有效**：三组新的A–B–A确认完整Dispatch+Combine周期有收益。
- **无效**：本次候选在所测配置下回退或持平；不推广为整个方向永远无效。
- **未确认**：只有小幅初筛差异，或计时受并发影响；不作为收益。
- **静态排除 / 未重跑**：没有本次GPU性能结论。

## Dispatch

commit `1b3e7dba`已加入MI355X普通V2的Dispatch JSON，按实际token数自动选档，包含T1024/2048/4096的256/128/8。最新独立复测、配置选择及与固定96/64/8的对照见 [bw.md](bw.md#json-recheck)。

### 本轮JSON复核

| 尝试 | 结论 | 处理 |
|---|---|---|
| BF16/K6/T64：96/64/8 → 16/8/8 | **有效**：三次加长配对复测及三组普通表路径ABA均改善 | 追加`experts_per_rank=16`行；保留原wildcard |
| BF16/K8/T64：64/32/8 → 32/21/8 | **未采用**：只有门槛内的小幅pair收益，Dispatch本身回退；普通ABA未确认此项 | 保留原Dispatch行，不与Combine候选叠加推算 |
| 其余七档、三种dtype/K组合的单OP quick扫描 | **没有新的稳定赢家** | 保留commit几何；只代表所扫范围 |
| BF16/K8的非2次幂block、RDMA比例及W6/W12邻域补扫 | **初筛未胜出** | 不修改几何 |

### 上一轮kernel与参数实验

| 原编号 | 尝试 | 上一轮结论 | 当时处理 |
|---|---|---|---|
| 1 | T4096，将B/R/W改成256/128/8，保留S8 | **有效** | 三组独立确认；新commit已将该几何加入对应JSON档位 |
| 1 | T64，缩小为64/32/8或32/16/8 | **未确认收益** | 只有小幅单组差异；64/32/8的前置基线还与短暂CPU缓存检查重叠，未采用 |
| 1 | T4096，固定96/64，仅将W改成4或16 | **无效：回退** | 保留W8；单独增减warp数没有改善完整周期 |
| 2 | T4096、W8，将S8改成S16或S32 | **无效：回退** | 保留S8；增加分片还会增加bid遍历和无payload的warp工作 |
| 2 | T4096，将S/W联合改成4/16，几何96/64/16 | **有效** | 保留[实验patch](docs/patches/epv2_dispatch_s4.patch)；未替换全局S8，优先使用上面的无patch配置 |
| 2 | T4096、W8，只将S8改成S4 | **未确认收益** | 清洁复测只有小幅差异，不单独采用 |
| 2 | T4096，将S/W联合改成16/4，几何96/64/4 | **无效：回退** | 不采用；S×W相同不代表相同延迟 |
| 4 | 将已有并行slot分配改成串行，做反向对照 | **串行方案无收益，保留已有并行实现** | T64/512的Dispatch变慢，T4096完整周期约持平；本轮没有新增slot优化 |

几何由 [hip_tuning_configs.py](python/mori/ops/dispatch_combine_v2/hip_tuning_configs.py) 的 internode API 通过共享的 `mori.ops.tuning_config` 读取分OP的JSON；`MORI_EP_V2_TUNING_DIR`可指定表目录。每相匹配自身dtype，`num_tokens`为包含上界的ceiling，最大档向上延用；同一档位的`experts_per_rank`精确匹配行优先于wildcard。测试自动选档时必须清除 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`，任一几何override都会绕过两个OP的表选择。显式使用普通 `v2`；`auto`的小T可能运行LL。

分片与slot代码在 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp) 的 `DispatchInterNodeRecv`。S实验保持 `bid % S` 的token归属；串行slot对照保留了去重、有效性检查、maps与payload copy顺序。它是机制对照，不是历史版本的完整revert。

### 只做了机制复核的历史方向

下列历史实现或完整快照缺失，本机没有重跑，不能标为本次有效或无效。

| 原编号 | 方向 | 复核结论 |
|---|---|---|
| 3 | 缓存远端node完成量，跳过空chunk轮询 | 应关注T远小于容量的场景；不能用本rank的T推断远端工作量 |
| 5 | 读取一次payload tile，复用到多个目标 | 会改变跨peer写流和寄存器使用；V1的收益不能代替V2验证 |
| 6 | 条件省略send结束AMO | 必须证明每个远端node的容量内slot都实际发送；稀疏、尾chunk和空rank仍需完成通知 |
| 7 | 用LDS任务列表摊平warp间copy负载 | 同时改变轮询、任务分配和block同步，不能从旧结果单独否定其中一个机制 |
| 8 | 将全局barrier改成LDS加全局两级计数 | 减少原子次数不等于缩短关键路径，必须保留payload发布和epoch清零顺序 |
| 9 | 按目标node紧凑打包send source，合并put | 需要额外打包和buffer；仅凭put尺寸不能预测完整周期收益 |

## Combine

当前final源码在gfx950为 **K6/K8、实际Combine类型BF16/FP32/OCP FP8、恰好两节点且每节点2/4/8 GPU、声明capacity≤256** 选择LF4×2，并用局部helper补齐单向量尾部；保留4字节stride对齐和短行回退。gfx942仍使用原BF16/H7168/K8/EP16/cap64或128条件。**最终39项strict均通过；两端编译与既有cache检查完成，二进制差异范围见下方。**

此前加入的Combine JSON独立于Dispatch选几何；FP8→BF16和BF16→BF16共用BF16 Combine行，修改该行须验证两种输入。T4096的S16/W8仍是限定配置的历史实验patch，未叠加到JSON复测。

### 本轮JSON复核

| 尝试 | 结论 | 处理 |
|---|---|---|
| K8/T64：96/64/8 → 16/10/8 | **有效，小幅收益**：BF16和FP8输入各三组普通表路径ABA均改善 | 两种输入共用一条BF16/epr16 Combine规则，保留原wildcard |
| K8/T256：64/32/8 → 64/42/8 | **有效**：BF16和FP8输入各三组普通表路径ABA均改善 | 追加BF16/epr16 Combine规则；FP8均值受尖峰影响，完整样本见bw.md |
| FP8/K8/T256：64/32/8 → 256/128/8 | **无效：初筛WIN未复现** | 加长复测回退；初筛差异主要来自原配置Dispatch的异常偏慢，不采用 |
| 其余Combine quick扫描及W6/W12等邻域补扫 | **初筛没有新的稳定赢家** | 保留commit几何 |

### 上一轮kernel与参数实验

| 原编号 | 尝试 | 上一轮结论 | 当时处理 |
|---|---|---|---|
| C1 | 容量64/128，本地和远端普通gather都先读两个相邻4B向量步，再累加 | **有效，已落地** | 两容量各三组独立确认；通过数值、最终包和目标机器码验收 |
| C2 | T4096，将普通gather扩宽到16B，必要时回退8/4B | **无效：回退** | 不采用 |
| C3 | T4096，将普通gather扩宽到8B，必要时回退4B | **未确认收益** | Combine本身变慢，完整周期小幅变化、wall持平；不采用 |
| C4 | T4096，将LF4×2推广到大容量，另新增LF4×4加深实验 | **无效：两者均回退** | LF保留小容量限制 |
| C6 | T4096、W8，将远端chunk分片S8改成S4 | **无效：回退** | 不采用 |
| C7 | T4096、W8，将S8改成S16，两阶段仍96/64/8 | **有效** | 保留[实验patch](docs/patches/epv2_gfx950_combine_s16.patch)，未默认启用 |
| C8 | 增加QP至4/8/16/32，同时影响两个OP | **无效：QP4持平，其余回退** | 继续QP2；旧bnxt平台的QP16结论不适用于本机 |
| C9 | T64，将Combine几何改成64/32/8 | **未确认收益** | 只有小幅单组差异，未采用 |
| C9 | T4096，将Combine几何改成96/48/8或128/64/8 | **无效：前者约持平，后者回退** | 保持96/64/8 |
| C12 | T4096，只在本地或只在远端gather启用LF4×2 | **无效：两者均回退** | 不采用单侧LF |
| C13 | T4096，以S16/W4或S4/W16替换S8/W8 | **无效：两种联合方案均回退** | 不根据S×W相等选配置 |
| C7/C13 | 固定W4，对比S8与S16 | **无效：约持平** | S16的已确认收益只覆盖W8 |
| C7/C13 | 固定W16，对比S8与S16 | **未确认：计时受CI编译影响** | 数值通过，性能样本不用于判断 |

LF调用位于普通 `CombineIntraNodeTyped` / `CombineInterNodeTyped` → `CombineGatherNormal`，具体策略和尾部处理位于 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp)。gfx950取消H7168特例，声明capacity上限扩为256，支持上述三种实际Combine类型和三种两节点拓扑；NIC/QP不参与选择。其它架构或不满足topk、类型、拓扑、capacity、对齐条件时使用原普通gather。FNUZ Combine没有新启用；FNUZ/FP4输入搭配BF16/FP32 Combine按后者判断。声明capacity4096而实际T64仍回退；性能结论限定于当时冻结的Ionic实测配置。

设单步为 `64×4/sizeof(Combine dtype)` 个元素：总行长不足两个步（512字节）时回到原 `WarpAccum`；完整LF双步后若仍有一个完整向量步，就用原4B向量实现处理，再执行不足单步的标量尾部；余量不足单步时保留原LF与标量实现。无需另设hidden下限。helper保持原源指针累加顺序及最后一次类型转换，没有修改公共primitive、LL、权重和最终跨节点归约。

性能扩展已完成：[off→broad统计](../mori_epv2_recheck_20260917/lf_generalize_20260920/results.md)包含84组ABA/252次运行，[off→tail统计](../mori_epv2_recheck_20260917/lf_generalize_20260920/tail_results.md)包含25组/75次运行。核心12格使用已扫描并复测确认的selected baseline，每格3组且A/B几何相同；BF16/K6的T64/128 Combine延迟分别下降15.54%/12.40%。默认/回退几何及tail两阶段96/64/8属于diagnostic，没有为每个扩展shape重新扫最优配置。两轮LF专项的Dispatch均保持并行，仅作控制项。

局部向量尾部使先前BF16 H1152/H2176、OCP FP8 H768及2×2 BF16 H2176四个回退点在独立off→tail复测中转为改善；对应Combine延迟变化为−2.70%/−6.02%/−10.36%/−1.17%，均3组。T256的H7168/BF16 K6/K8各3组off→broad分别下降2.00%/2.22%，但tail T256交叉诊断中的短BF16及2×2 BF16接近持平，不能声称所有cap256形状均稳定受益。具体数值和限制见bw.md；broad/tail仍是实验版本，最终条件以final源码为准。

FP4性能输入为真实E2M1双nibble packed bytes，每token dispatch payload是逻辑H/2字节；公开codec解码后交给BF16或FP32 Combine，没有启用FP4 Combine。本测试不自动应用opaque scale metadata，不等同于完整MXFP4量化；pack/unpack不计入两阶段kernel时间，payload与scale/weights分项记录。

**最终正确性：39/39项通过。** 38项主队列加BF16 H7200原LF非零标量尾部补充均完成，每个有效rank各3轮；共1,704条STRICT记录，逐位检查2,469,003,264个hidden元素，hidden不匹配为0。weights采用atol=rtol=1e-6，实测最大绝对误差为0。6个FP4用例有288条PACKED记录、207,954个接收行、745,307,136个packed字节，完整接收行multiset检查无不匹配。该检查不要求接收顺序一致，数值范围为有限值identity-expert、quant none的普通v2。见[主审计](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_v2_audit.json>)、[H7200补充审计](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_v2_supplement_audit.json>)和[独立复核](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_independent_review.md>)。

**最终编译及缓存检查：两端各105份普通Combine报告均验证成功。** 每端含30份离线编译和75份既有缓存inspect，后者全部命中，未通过补编译填MISS。每端35项final/reference比较中，34项原始.text字节相同，35项descriptor、只读数据和kernel metadata/资源相同；启用路径分别对照broad或tail，回退路径对照off。**完整代码对象等价审计保留EQUIVALENCE_NOT_PROVEN：** 所有对照都有真实导出符号`__hip_cuid_*`及相关ELF查找表差异，个别标识长度还改变DT_STRSZ；H192回退另有4字节指令差异，不能把off的H192性能直接转作final结论。没有抹掉这些差异或将完整ELF判为相同。原始审计见[node0](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_offline_audit_node0_v2.json>)、[node1](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_offline_audit_node1_v3.json>)，逐字节差异分类见[node0诊断](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_elf_cuid_host_diagnostic_v1.json>)、[node1诊断](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_elf_cuid_host_node1_v3.json>)。本次offline只覆盖ordinary Combine，Dispatch/LL仅有源码保持审查和各自历史验证；gfx942/mlx5仅作离线编译与代码对象覆盖，不构成这些平台的硬件正确性或性能结论。

最终状态和全部审计SHA256集中在[final_verification_summary.json](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_verification_summary.json>)（SHA256 `563bddc4824d3504d6b49239f129d7a7310f013462b2da30f24895b4077a84c0`）。这份结果更新冻结记录及下方实验发布快照中的pending状态，原始证据未改写。g09首次辅助程序构建因本机缺少host headers退出，已保留失败日志并用g17导出的相同头文件在独立v3目录完成；没有修改GPU测试来源或既有cache。结束时两台机器均无本任务活动runner。当前更新为源码及报告，测试显式使用冻结MORI_SOURCE_ROOT，未重新构建wheel或生成包内源码副本。

此前移除NIC/QP限制时的历史验证（对应当时源码，不代替本次final验收）：gfx950/Ionic与mlx5的QP4、容量64普通Combine均通过[离线编译](../mori_epv2_recheck_20260917/lf_portability_20260920/offline/summary.json)。g17/g09上的Ionic数值检查覆盖QP2/T64 BF16、QP4/T64 BF16、QP4/T128 BF16 skewed、QP8/T64 FP8→BF16，每组3轮、16个rank的输出和权重均通过，见[数值结果](../mori_epv2_recheck_20260917/lf_portability_20260920/gpu_checks.json)。额外的QP8/T128 local用例因CI持续占用未启动；mlx5仅有编译验证，未作对应硬件运行或性能声明。

S16改的是普通远端Combine的分片，token归属 `bid % S` 和完成阈值 `S * W`保持一致。实验patch自身没有完整限制NIC/QP/W，只能在上述实测配置下使用；不能全局开启或把它与Dispatch的收益相加。两种实验patch均未进入上一轮已验收wheel，应在独立checkout分别应用；基线header SHA256为 `5b1a92a5ed89b683539ac07590320fca63d568acbf90405ecf0fafdb0ff448d8`。

### 静态排除与待验证项

| 原编号 | 方向 | 本次状态 |
|---|---|---|
| C5 | 压缩非空源指针，再选择累加模板 | **本机未重跑**：历史快照缺失；原路径已跳过空源payload读取，压缩不会减少有效payload字节 |
| C10 | 让同一向量步的expert读取并行 | **静态排除**：常用K8实现已先读取各源再累加；C1改变的是跨两个向量步的调度 |
| C11 | 只改 `WARP_ACCUM_UNROLL` 让普通gather展开 | **静态排除**：普通wrapper没有使用该参数展开，修改宏不能实现目标优化 |
| C14 | 定位gather、传输、peer等待与barrier成本 | **已做阶段profile，优化待验证**：主Combine仍包含这些成本，尚未完成内部等待归因，没有新的代码收益 |

## 验证与记录

上一轮小容量LF、推荐Dispatch几何及V1/V2/LL兼容性已通过对应数值验收。CQ模式一致性、provider生命周期和V1 benchmark入口属于正确性修复，不计入kernel优化收益。此前JSON复核仅更新表和对应配置测试；本次LF扩展另增加普通Combine的局部helper及上述选择条件，保留已有改动。LL、`CombineAll`和共享gather primitive没有新增优化改动；最终源码的运行、编译结果及二进制核对范围见上方。

新加的三条JSON规则只覆盖epr16，其他expert数继续命中原wildcard。token档位按实际输入做ceiling匹配，容量、QP和CQ模式不是匹配键；性能结论限定于实际T=容量、QP2、CCQE关闭的实测条件。独立扫描是固定另一OP的有限候选集，不是全局最优证明。

此前采用的JSON已通过BF16/K6、BF16/K8、FP8/K8各21轮不同token档位的数值检查（每轮16个rank，含uniform/skewed/local），以及每个dtype/K组合在容量64/256/4096各128轮持续测试。数值检查含weights；持续测试检查hang/fault，未逐轮核对数值。

性能数值、复现入口和原始证据索引统一见 [bw.md](bw.md)。
