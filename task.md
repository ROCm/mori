# PR #696 第二轮评审修改清单

来源：Jiahao（@jhchouuu）2026-11-06 对 `2bcfcde5` 的评审，包括[总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895)和 21 条行内评论，另有第一轮（2026-10-19）仍未解决的 2 条。

标签：

- **合并前**：总结里明确要求合并前完成
- **上机器**：要在 MI300X / MI308X / MI355X 上重测或验证
- **小改** / **大改**：纯代码或文档改动的大致工作量
- **商量**：可以和他商量放到后续 PR

## 1. 表格数据可信度

- [ ] **合并前 · 上机器** MI300X 的 T4096 行重测，保存时强制检查门槛
  - `gfx942_mi300x_InterNodeV2_ep16.csv` L81、L103、L136、L194
  - L81 记录的是 2677.69 µs，按它自己的带宽列算约 1790 µs：均值混进了卡顿轮次，README 里 mean/median ≤ 1.10 的门槛没拦住
  - L81 的规则 304/152/8 比 304/202/4（T2048 的规则）和 256/170/8 慢约 7%：他交替跑 3 次，分别是 1690、1572、1580 µs
  - 调优脚本保存行时要真正检查这个门槛
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575140)
- [ ] **合并前** MI355X 有 12 行 EP16 规则只测了时间、没验证正确性，先撤出表
  - 即 tuning README L223-231 描述的那批：FP4 dispatch T4–T16 和 FP8 combine T4–T64 里改动过的行，以及 896/16 InterNodeV2LL 的 T2048 bf16、T4096 fp8 两条 dispatch 规则
  - 用 git 历史找出这 12 行，回退到之前验证过的规则；以后在正常的集群上验证通过再加回来
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575162)
- [ ] **合并前 · 上机器** 大 token 数的调优收益要用交替 A/B 确认，并把这一步写进 tuning README
  - 他在 MI308X T1024 上看到：调优工具报告有 100–175 µs 的收益，交替跑时却和表里原来的规则打平（1265–1283 vs 1276–1283 µs）
  - 要回复：大 T 的行有没有这样确认过；同时把这一步写进 README（旧的 MI308X 表注释里写过）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Before merge 第 4 条
- [ ] **合并前** 说明 FP4 dispatch 行怎么复现
  - 测试脚本的 `_DTYPES`（`test_dispatch_combine_v2_internode.py` L132）没有 FP4，但 6 张 inter-node 表各有 11 行 FP4
  - 二选一：给脚本加 FP4 支持，或者在 README 里说明这些行是怎么测出来的
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Before merge 第 5 条
- [ ] **上机器** 复核 MI308X fp8 T4096 dispatch 的 QP 数
  - `gfx942_mi308x_InterNodeV2_ep16.csv` L92，规则是 80/20/16、`active_qps=2`
  - 他测到：用这条规则，dispatch 1731–1812 µs，D+C 4937–4981 µs；两个阶段都用 8 个 QP 时，dispatch 1215–1222 µs，D+C 4766–4781 µs，快 3.7%
  - pair 扫描选的是 dispatch 8 / combine 4，但没做交替确认
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575153)

## 2. 正确性问题

- [ ] **上机器** 896/16 LL dispatch 在 AINIC / gfx950 上偶尔结果出错，单独开 issue
  - 他的复现结果：
    - MI355X + AINIC（amdgpu 6.16.6），几何 256/128/16 和 256/170/16，T2048，20 次里 8 次出错，每次都是某一轮里 16 个 rank 中有 1 个结果错
    - 合进 main（#895）后，10 次里仍有 4 次出错
    - MI300X + ConnectX-7 上 10 次全部通过
  - 只在表里排除这组几何等于把问题藏起来，其他几何也可能出错，只是更少见
  - 起草 issue 内容，由你提交
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575162)

- [ ] **上机器** MI300X 上 auto 模式的 stress 会卡死（2026-11-06 发现，原版 `2bcfcde5` 一样卡）
  - 复现：MI300X + ConnectX-7 两台，`--cmd stress --kernel-type auto --max-tokens 1024 --rounds 300 -- --stress-datasets 16`，加不加 `--stress-check-interval 10` 都会卡
  - 每个 rank 的 token 数在 94–929 之间各自随机，同一轮里有的 rank 走 v2_ll、有的走 v2
  - 16 个 rank 都停在 combine 之后第一个 host 同步点上，说明是 GPU kernel 没有返回
  - 对照：本 PR 的改动版本卡 2/2 次，`2bcfcde5` 原版卡 1/1 次，都是 7 分钟超时被杀
  - **原因（已确认）**：v2 combine 对每个空 chunk 都会发一个 0 字节的 `EpInterNodePut`（`ep_internode_kernel.hpp` combine 末尾，长度是 `thisChunkTokenNum * tokCombXferBytes`）。mlx5 的 RDMA 写请求固定带一个数据段，`Mlx5PostReadWriteImpl` 把长度 0 原样写进 `byte_count`，而 mlx5 硬件把 0 解释为 2^31 字节，于是这个 QP 出错；combine 之后的屏障标记在这个 QP 上永远送不到，对端一直等下去
  - 只有存在空 chunk 时才触发：某个 rank 的 token 数少于容量，或者路由是 local/skewed。只用 v2 会卡，只用 v2_ll 不卡（它不发 0 字节写入）；dispatch 正常，卡在 combine；只在 mlx5 上出现，AINIC 和 Broadcom 能处理 0 字节写入
  - 验证：在 `EpInterNodePut` 里加一行 `if (bytes == 0) return;`（这个 put 不带信号，0 字节本来就不写任何东西），三组原本卡住的用例全部通过，逐轮检查 0 个 rank 结果不一致
  - [ ] 修复：EP 层跳过 0 字节 put；mlx5 层对长度为 0 的写入不带数据段（和 rdma-core 的做法一样），这样所有调用方都受保护

## 3. 文档和 PR 描述

- [ ] **合并前** 按现状重写 PR 描述（英文草稿在 `/apps/qizzhang/pr696_description.md`，需要你补 TODO 后贴到 PR）
  - 删掉：按 shape 存的 JSON、QP2、按容量选 load-first、已经删除的 `bw.md` / `dispatch.md`
  - 补上：
    - CSV 表
    - 合进 #698，包括 `num_qp_per_pe` 的默认值从 2 改成 8
    - inter-node 的 FP4 dispatch
    - 两个同步修复：CombineLL 里等节点计数，以及 combine 完成计数的 release/acquire
    - `device_primitives.hpp` 的改动影响所有调用方
  - 改掉"查不到时退回同架构其他型号的表"这个说法，代码只读当前型号自己的 CSV（第一轮[评论](https://github.com/ROCm/mori/pull/696#discussion_r4312595414)）
  - tuning README "Measurement conditions" 里删下来的历史信息放进描述
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Before merge 第 1 条
- [x] **小改** tuning README 的 "Measurement conditions" 缩成只讲测量方法
  - `tuning_configs/README.md` L157 起：只保留路由、warmup/rounds、接受门槛、整表 A/B 规则
  - 日期、集群、驱动和固件版本、候选数量、运行结果挪进 PR 描述
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575171)
- [x] **小改** `EPv2_PERF.md` 移出 Python 包目录
  - 二选一：挪进 PR 描述，或者翻成英文放进 `docs/`
  - 注明 MI308X 那节用的是从 CLR 源码编的 HIP runtime（文件 L26），用标准 ROCm 7.2.1 复现不了
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575176)
- [x] **小改** 修正 `dispatch_combine_v2/README.md` 里的调优示例命令（L224-230）
  - `--num-qp 2` 会把 `active_qps=4` 的行截成 2，而表是在分配 8 个 QP 时测的
  - 测试脚本默认 `--num-qp 1`（L217），`--scale-dim` 默认 32，而表用的是 0
  - `run_internode_test.sh` 默认带 `timeout ${MORI_INTERNODE_TIMEOUT:-120}`，照着示例跑扫描，120 秒就会被杀掉
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575209)
- [x] **小改** 统一 `docs/MORI_JIT_V2_DESIGN.md` 里关于专家数匹配的说法
  - L333-337 说专家数用通配（"那个通配是个结论，不是省事"），L363 说必须精确匹配；要么说清楚两处各指哪张表，要么统一成一种说法
  - 顺带删掉 L348 关于 `Config.tuned()` 的那句（见第 5 类）
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575252)
- [x] **小改** 在 PR 描述里写明 `device_primitives.hpp` 的影响范围（或者把改动限定在 v2，见第 6 类）
  - 这三处改动会影响 v1、intranode、AsyncLL 和 shmem 的 kernel（`WarpCopy` 约有 120 处调用）：`WarpCopy` 窄尾部的处理、`WarpAccumDynamic` 的分批、`accumNum == 9` 的分支
  - 两处注释要改：
    - L319 附近的 "512 B of a 3584 B row" 把通用函数和某个具体形状绑在一起
    - L494-498 说一次有 8 × 16 B 在途就会溢出寄存器，但 `WARP_ACCUM_CASE(8)` 和 `(10)` 在 topk-8 的 LL 路径上本来就这么做
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575204)

## 4. 打包和查表的缺陷

- [x] **合并前 · 小改** `MANIFEST.in` L13 加上 `*.csv`
  - 现在是 `recursive-include python *.py *.md *.bc *.json *.sh`：用 sdist 安装时没有表，查表会悄悄退回默认几何
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Before merge 第 2 条
- [x] **小改** 清掉 896/16 行的 `combine_load_first`，删掉 README 里对应的说法
  - topk 16 时，`WarpAccumLF` 和 `WarpAccum` 最后都走同一个 `WarpAccumDynamic`（`device_primitives.hpp` L821、L930）。开了这个标志只会多编一个一模一样的 combine kernel，当初决定保留它们的 A/B 测到的只是噪声
  - 涉及的行：MI355X L225、L227、L234；MI300X L226、L227、L230；MI308X L226、L227、L230、L231
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575158)
- [x] **小改** 查 inter-node CSV 时改用 `_device_key()`
  - `hip_tuning_configs.py` L1079 用的是 `detect_model()` 的原始结果；Python 表用的是 `_device_key()`（L46），它会把 mi350x 映射成 mi355x，并单独处理 gfx1250
  - 结果是 MI350X 能拿到 MI355X 的 Python intranode 表，却拿不到 inter-node CSV；gfx1250 会去找 `gfx1250_None_*.csv`
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575196)
- [x] **小改** 加载 inter-node 表时要求 `rdma_block_num ≥ 1`
  - `fit_internode_geometry`（L683）会保留 0。inter-node 的行如果是 0，就没有 block 去跑 RDMA，节点计数的信号永远不会到，op 会卡死
  - 已核对：现在 6 张 inter-node 表里都没有 0，所以这是缺校验，不是已经存在的 bug
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575186)
- [x] **小改** 只设置 `MORI_EP_DISP_GEOM` 或 `MORI_EP_COMB_GEOM` 其中一个时，另一个阶段要走表
  - 现在另一个阶段拿到的是没有截断的默认几何（`_geometry_from_env`，L706）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup

## 5. 死代码和小 bug

- [x] **小改** 删除 `internode_tuning_configs.py`
  - 仓库里已经没有地方 import 它；它的签名也变了（`experts_per_rank` 变成必填，返回 4 元组），旧的调用方本来就会出错
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575180)
- [x] **小改** 删除 `EpDispatchCombineConfig.tuned()`
  - 在 `dispatch_combine_op.py` L547；仓库、aiter、ATOM 里都没有调用方
  - 一并删掉 `dispatch_combine_v2/README.md` L45、L207 和设计文档 L348 里提到它的地方
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575247)
- [x] **小改** 清理 `hip_backend.py`
  - `self._asyncll_plans`（L1407）只写不读
  - `self._plans = []` 设了两次：L1383（`_build_asyncll_kernels`）和 L1690（`_build_kernels`）
  - L1345 的注释还在引用已经删除的 `_INTERNODE_LEG`
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [x] **小改** 在 `hip_backend.py` 约 L1451 写回 combine 为什么要复用 dispatch 的 `tokenIndices` 和权重
  - 要写回的两个事实：`EpCombineAll` 无条件解引用 `tokenIndices`，传空指针会导致 GPU fault；输入权重按源 token 索引，不是按收到的 token
  - 第一轮[评论](https://github.com/ROCm/mori/pull/696#discussion_r4312595439)
- [x] **小改** `ep_internode_kernel.hpp` L1044 的对齐检查永远成立，删掉
  - `tokCombXferBytes` 等于 `tokHiddenBytes` 加 0 或 topk × 4 B；不走 load-first 时用的 `WarpAccum<TokT, 4>` 也做同样的 4 字节读取，却不检查。`UseCombineLoadFirst()`（L1031）只是把字段返回
  - 改成 `if constexpr (kConfig.combineLoadFirst)`，去掉两个 size 参数
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575206)
- [x] **小改** `test_v2_tuning_tables.py` L51 改成检查源码目录
  - 现在检查的是已安装的包，而 package_data 只带 `*.csv`（`setup.py` L1108），所以在 CI 里永远不会失败
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575242)
- [x] **小改** 调优脚本：`--tuning-phase pair` 模式下保存的行会缺 QP 数（真 bug）
  - 这个模式下 `candidate_fields` 一直是 `()`（L1780）
  - 用 `--tuning-candidate "…/…"` 做确认、又不传 `--tuning-qps` 时，`report_active_qps` 只从表里已有的行取值
  - 如果那一行没有 `active_qps`，`--tuning-save` 写出的行就不含实测用的 QP 数，但指标是用那些 QP 数测的
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575226)
- [x] **小改** 调优脚本：L2079 的分支走不到，提示信息也不对
  - 候选列表为空时更早就返回了（约 L1799-1806）；现在 `best_median` 为 None 只会是所有候选都被拒，这时打印的 "refresh" 提示是错的
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575233)
- [x] **小改** 调优脚本：保存时的变更报告要显示 `active_qps` 和 `combine_load_first`
  - 现在只显示 block、warp、rdma，只改了这两项的规则会被报成 "(unchanged)"
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [x] **小改** 调优时不要为每个候选 op 都打印完整的 schedule
  - 打印的地方在 `hip_backend.py` L430（`logger.info` 加 `describe_schedule`）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [x] **小改** 调优脚本：删掉没有调用方的 `evidence_callback`
  - `_row_metrics`（L1356）里的 evidence 字典和回调（L1405-1416）都是死代码
  - L1359-1400 对脚本自己刚生成的数据做逐项检查，看起来是删掉的单元测试留下的
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575222)
- [x] **小改** 还原 `tests/python/ops/dispatch_combine_v2/test_op.py` L137 只改了格式的那处
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup

## 6. 结构重构

- [x] **大改 · 商量** `hip_tuning_configs.py` 里的分桶、合并、选择逻辑合成一份（`_ceilings()` + `_probe()` + `pick_bucket()`；`internode_schedule` 合并时保留 kernel 族分界，`resolve_schedule` 不再重算边界；1796 个配置的快照前后一致）
  - 现在写了四五遍：
    - `_merge`（L564）
    - `internode_buckets`（L1051），里面内联了一份 `pick_bucket`
    - `internode_schedule`（L729）
    - `resolve_schedule`（L451），会把 `internode_schedule` 合并掉的桶边界又加回来
    - `intranode_buckets`（L1149）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [x] **小改** CU 数只用一个来源（backend 把 HIP 的 `multi_processor_count` 传给 `resolve_schedule`；没有设备的调用方才读 sysfs；去掉写死的 80。FlyDSL 的 `tuning_configs.py` 本 PR 没改，未动）
  - `fit_internode_geometry`（L690）和 `tuning_configs.py` L263 读 sysfs，读不到就写死 80；backend 用的是 `multi_processor_count`（`hip_backend.py` L398）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [x] **大改 · 商量** 去掉 `zero_copy`、`quant_type` 两列和相关检查（9 个 CSV 共 1755 行，删前逐行确认只有 0/none 或空）
  - V2 没有 zero-copy 模式；inter-node 只接受 "none"；intranode 的 CSV 也只在 "none" 时才读。这两列填了别的值的行永远查不到（L1110）
  - 9 个 CSV 的表头都要改
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575193)
- [x] **大改 · 商量** intranode 只保留一个来源（CSV 或 Python 表）——部分：只传 `dtype=` 的调用方也先读 CSV，两条路径的补缺策略统一，删掉与默认值相同的 Python combine 条目。Python 的 dispatch 条目保留，因为它按专家数通配，并且有 CSV 没有的 FP4 和 topk 6 的 FP8
  - IntraNodeV2 的 CSV 只在传了 `dispatch_dtype` 且 `quant_type == "none"` 时才读（L628）。只传 `dtype=` 的调用方（比如 aiter 的 `_mori_tuning`）拿到的是旧的 Python 表：MI355X EP8 256/8 bf16 在旧表里只有 1 档，CSV 里有 8 档
  - 两个分支对只调了一半的形状处理也不一样：CSV 分支用 `_hip_default` 补上，Python 分支直接拒绝（L640-649）
  - Python 表里 mi355x 的条目已经被 CSV 盖住，那里的注释（"64x8 wins at every token count"）也和 CSV 矛盾
  - [评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575200)
- [ ] **大改 · 商量** 调优代码挪出 `tests/`，并去掉重复——去重已做（计时循环合成 `_timed_rounds`，三份 golden 合成 `_golden`，统一考虑 `--quant-type`）；挪目录会改 `--cmd tuning` 入口和文档，留给后续 PR
  - 约 1000 行只用于调优，CI 从来不跑，建议挪到 `tools/` 或 `examples/`
  - 计时循环有两份：`_timed_pass`（L1067）和 `_metric_timing_pass`（L1182）
  - golden 检查有三份（约 L453、L1482、L2409），其中 L1482 那份忽略了 `--quant-type`（[评论](https://github.com/ROCm/mori/pull/696#discussion_r4489575218)）
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895) Cleanup
- [ ] **大改 · 商量** 把 `device_primitives.hpp` 的改动限定在 v2（第 3 类"写进描述"的另一种做法）——不做：已在 PR 描述里写明影响范围

## 7. 性能（不阻塞合并）

- [ ] **上机器** LL 在 T64 变慢了
  - 固定几何和 QP 数、对比 merge-base：MI300X 从 142 变成 145 µs，MI355X 从 166 变成 173 µs。这是 decode 路径
  - 看看 CombineLL 里等节点计数和每个 warp 的 fence 能不能更省
  - [总结](https://github.com/ROCm/mori/pull/696#pullrequestreview-5796438895)

## 不用处理

- 第一轮关于 fence scope 的评论：他测了 producer 用 agent scope 做 release 的情况，MI300X 没有可测出的差别，MI355X 的 combine 也看不到 fence 的开销，这条他已经撤回
