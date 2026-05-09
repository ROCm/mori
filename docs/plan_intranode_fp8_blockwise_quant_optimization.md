# Intranode FP8 Blockwise Quantization 后续性能优化计划

## 背景与原则

当前 IntraNode `fp8_blockwise` 已经具备向量化 quant/dequant 实现，并支持 MVP 路径：

- `kernel_type=IntraNode`
- `dtype=bf16`
- `--zero-copy 0`，即 `useExternalInpBuffer=true` / `_nop2p`
- staging layout: `[fp8 token][float scales][weights]`

已有验证显示，向量化版本相对早期标量版本有明显改善，但与 `fp8_direct_cast` 仍有差距。这个差距不能只用 synthetic `randn` benchmark 解释，也不能把“绝大多数 token 不超过 FP8 e4m3 最大值 448”作为优化前提。

本计划的核心原则：

1. **主优化目标是 scale-active 路径。** 假设真实上层应用中经常存在超过 FP8 range 的 activation，blockwise scale/reduce/dequant multiply 是必须优化的热路径。
2. **no-scale fast path 只能作为 opportunistic optimization。** 可以实现，但不能作为性能达标依据。
3. **性能评估必须覆盖强制 scale-active 分布和真实 trace。** 不能只看默认 random normal benchmark。
4. **先建立可解释的 profiling，再做 kernel 改动。** 每个优化阶段都要能说明收益来自 quant、dequant、scale traffic、launch config 还是其他因素。

## 当前性能基线

初始参考数据，EP8, bf16, `max_tokens=4096`, `hidden_dim=7168`, `scale_dim=32`, `--zero-copy 0`：

| quant type | combine latency | 备注 |
| --- | ---: | --- |
| `fp8_direct_cast` | ~668 us | 无 scale direct cast |
| `fp8_blockwise` vectorized | ~1275 us | 当前向量化 blockwise |
| `fp8_blockwise` scalar MVP | ~4195 us | 已淘汰，仅作历史对比 |

这些数据只说明当前实现相对 direct cast 的开销规模。后续不应只用这组 random-normal 输入作为优化判断。

## Phase 0: Benchmark 与可观测性补强

### 0.1 增加输入分布控制

文件：

- `tests/python/ops/bench_dispatch_combine.py`
- `tests/python/ops/dispatch_combine_test_utils.py`

新增参数建议：

```bash
--input-dist normal|uniform|lognormal|two_bucket|trace
--input-scale <float>
--input-shift <float>
--force-scale-active 0|1
--scale-active-ratio-target <float>
--scale-dim <int>
--trace-path <path>
--report-scale-stats 0|1
```

需要覆盖的分布：

| 分布 | 目的 |
| --- | --- |
| `normal, scale=1` | 低动态范围 sanity，不作为主性能目标 |
| `normal, scale=512` | 大量 block 触发 scale |
| `uniform[-1024, 1024]` | 稳定 scale-active 压测 |
| `two_bucket` | 大部分普通值 + 少量极大值，模拟长尾 activation |
| `trace` | 上层应用保存的真实 combine input trace |

注意：

- 当前 benchmark 入口里 `scale_dim` 仍是硬编码 `32`，若要执行 Phase 1/7 的 `scaleDim=16/64` 矩阵，必须先把 `--scale-dim` 从 CLI 透传到 `bench_dispatch_combine()`、worker config 和 tuning save/load。
- scale-active 统计应基于实际进入 combine 的 input tensor 和有效 token 集合，而不是只统计原始 `all_rank_input`。否则 routing / dispatch 后每 rank 实际 combine 处理的数据分布会被掩盖。

### 0.2 统计 scale-active 情况

需要在测试或 debug kernel 中报告：

- token 级别 `any_scaled` 比例，对应当前 `scale[0] < 0` token marker
- block 级别 `abs(scale) != 1.0` 比例
- 每 rank 的 scale-active 分布
- max abs 分布的 p50/p90/p99/max

如果只在 host 侧统计，可先用 PyTorch reference 在 benchmark 输入上统计；后续再决定是否增加 device-side debug counters。

注意当前 marker 语义：

- quant 写完每个 block scale 后，如果 token 内任意 block 发生 scale，会把 `scale[0]` 取负作为 token marker。
- 因此 block 级统计不能直接用 `scale != 1.0`，否则当 block0 本身没 scale 但 token 内其他 block scaled 时，`scale[0] == -1.0` 会被误算为 block0 scale-active。
- no-scale fast path 后续如果复用 marker，也必须保持这个统计口径。

### 0.3 拆分性能指标

当前 combine latency 包含：

1. quant/copy input 到 combine staging 或 registered buffer
2. cross-device barrier
3. dequant + accumulate
4. weights accumulate

需要进一步拆分：

- 现有 profiler slots 已覆盖 `CombineCopyInput`、`CombineBarrier` 与 `CombineAccum`，先打开并记录。
- 现有 `CombineCopyInput` 对 blockwise 包含 quant、scale store、weights copy；`CombineAccum` 包含 pointer prep、dequant accumulate、weights accumulate。这个粒度不足以归因 blockwise 额外开销。
- 如果粒度不够，新增临时 profiler slots：
  - `CombineQuantizeInput`
  - `CombineCopyWeights`
  - `CombineDequantAccum`
  - `CombineAccumWeights`

目标是明确 blockwise 的额外开销主要来自 quant reduce、dequant scale multiply、scale memory traffic，还是 launch config 不匹配。

### 0.4 rocprof 指标

对 `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` 收集：

- kernel duration
- occupancy / waves per CU
- VGPR / SGPR / LDS usage
- VALU utilization
- L2 read/write bytes
- memory coalescing 指标
- flat/global load/store 指标

同配置对比：

- `EpCombineIntraNodeKernel_bf16_nop2p_fp8cast`
- `EpCombineIntraNodeKernel_bf16_nop2p`
- `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq`

## Phase 1: 建立性能验收矩阵

### 1.1 功能矩阵

最小覆盖：

| EP | max tokens | hidden dim | scale dim | distribution |
| --- | ---: | ---: | ---: | --- |
| 2 | 16 | 256 | 32 | normal |
| 4 | 4096 | 7168 | 32 | normal |
| 4 | 4096 | 7168 | 32 | scale-active |
| 8 | 4096 | 7168 | 32 | normal |
| 8 | 4096 | 7168 | 32 | scale-active |
| 8 | 4096 | 7168 | 16/64 | scale-active |

### 1.2 性能矩阵

每组都记录：

- dispatch latency
- combine latency
- E2E latency
- scale-active ratio
- blockwise / direct_cast latency ratio
- blockwise / bf16 normal latency ratio
- accuracy max diff / allclose 结果

初始目标不是固定绝对值，而是建立趋势：

- scale-active 输入下，blockwise 相比 direct cast 的额外开销是否随 `scaleDim`、`hiddenDim` 合理增长。
- no-scale 输入下，blockwise 是否能接近 direct cast。
- mixed 输入下，性能是否随 scale-active ratio 平滑变化。

## Phase 2: scaled path dequant 优化

这是最高优先级。即使所有 block 都 scale-active，dequant 路径也应该尽量接近 packed FP8 accumulate 的吞吐。

### 2.1 scale hoisting

当前 dequant 的 vectorized path 在 blockwise scale 场景下仍可能频繁：

- 计算 block id
- 读取 `srcScales[i][sb]`
- 执行 `fabsf`
- 对 vector chunk 做 scale multiply

优化方向：

1. 外层按 scale block 遍历。
2. 对每个 scale block，把所有 source 的 scale 读到 register array。
3. block 内多个 vector chunk 复用同一组 scale。
4. segment path 同样按当前 segment 覆盖的 scale block 分段处理。

关键文件：

- `include/mori/core/transport/p2p/device_primitives.hpp`

重点函数：

- `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`
- `WarpAccumFp8DequantSegmentBlockwiseVec`
- `WarpAccumFp8DequantFullImpl`
- `WarpAccumFp8DequantSegmentImpl`

注意当前热路径：

- `WarpAccumFp8DequantBlockwiseVec` 目前只有定义，没有被主路径调用；优化它本身不会改善现有 benchmark。
- full path 对常见 aligned shape 当前通过 `WarpAccumFp8DequantFullImpl` 调用 `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`。
- segment path 在 segment aligned to scale block 时也会走 `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`，否则才会按 block 分段走 `WarpAccumFp8DequantSegmentBlockwiseVec`。
- 因此 Phase 2.1 的实际落点应是修改 `WarpAccumFp8DequantFullImpl` / `WarpAccumFp8DequantSegmentImpl` 的分支选择与主 helper，而不是只优化未接入 helper。

### 2.2 移除 hot path `fabsf`

当前 scale 使用负号 marker 表示 token/block 是否发生 scale。dequant 内层为了安全使用 `fabsf(scale)`。

优化方案：

- 在进入 dequant 前或每个 scale block 开始时处理 marker。
- hot path 中保存正 scale 到 register。
- 内层 vector loop 只做 `fp8 * positive_scale`。

注意：

- 不能破坏 no-scale marker 语义。
- 当前负号 marker 只应表示 token 级 `any_scaled`，不表示 block0 一定 scaled。
- 若保留 `scale[0] < 0` 作为 token marker，应只对 `scale[0]` 做一次 abs 修正，并保证后续 block scale register 全部是正值。
- 对未 scaled block，正 scale 仍必须是 `1.0f`。

### 2.3 减少非 2 的幂 blockElems 除法

常见参数 `hiddenDim=7168, scaleDim=32` 时：

```text
blockElems = 224
```

224 不是 2 的幂，按元素计算 `idx / blockElems` 代价较高。

优化方向：

- 在按 scale block 遍历的实现里避免除法。
- 对常见 `(hiddenDim, scaleDim)` 或 `blockElems` 增加 specialized branch：
  - `blockElems=224`
  - `blockElems=128`
  - `blockElems=256`
  - `blockElems=448`
- 对不常见 shape 保留 generic fallback。

### 2.4 accumNum 特化检查

当前 switch 覆盖 `1/2/4/6/8/10`。上层常见 `num_experts_per_token=8`，应确认：

- `AccumNum=8` 是否走完全 vectorized scaled path。
- register pressure 是否过高。
- 是否需要为 `AccumNum=8` 写更紧的 specialized implementation，减少 local array 和 lambda 生成的指令。

## Phase 3: quant reduce 优化

### 3.1 subwarp mapping 评估

当前 vectorized quant 会根据 alignment 和 block size 选择：

- `SubwarpSize=16, InVecBytes=16`
- `SubwarpSize=32, InVecBytes=8/4`
- `SubwarpSize=warpSize`

对于 `blockElems=224`：

```text
SubwarpSize=16, InVecBytes=16 -> 每 lane 8 bf16 elems, stride=128 elems, maxIters=2
```

需要用 profiler 验证：

- 这种 mapping 是否造成 register pressure 过高。
- `MaxCacheIters=4` 是否合适。
- `SubwarpSize=32` 是否在某些 shape 下更快。

### 3.2 max reduce 与 quant store fusion

当前实现为了避免二次 load，部分路径会 cache packed bf16 数据再 quantize store。需要确认：

- cache path 是否真的覆盖主 shape。
- cache array 是否带来过高 VGPR。
- 对 `maxIters=2` 的主 shape，是否可以写专用 two-iteration path，减少循环和分支。

### 3.3 scale store 优化

scale store 是 `scaleDim * sizeof(float)` per token。对 `scaleDim=32` 是 128 bytes，流量不大，但 store ordering 和 cache behavior 可能影响 latency。

检查点：

- scale store 是否 coalesced。
- scale buffer alignment 是否足够。
- `scaleDim` 非 32 时是否保持 alignment。

## Phase 4: no-scale fast path

该阶段不是主性能目标，但语义安全且对低动态范围场景有收益。

### 4.1 quant no-scale detection

启用或完善 `WarpQuantizeBf16ToFp8NoScaleVec`：

- 如果 token 内所有 value 都不超过 FP8 max，则直接 vector cast 并设置 no-scale marker。
- 如果发现需要 scale，再走 blockwise scale quant。

注意：

- 这会增加一次检测逻辑，必须确认 scale-active 输入下不会反而拖慢。
- 可以按配置开关或 shape heuristic 控制。
- 不能只写 `scale[0] = 1.0f` 就返回，除非 dequant no-scale bypass 在同一提交中保证不会读取 `scale[1:]`。
- 更保守的做法是在 no-scale quant path 中把当前 token 的所有 `scale[0:scaleDim]` 写成 `1.0f`，这样即使后续 dequant 仍按 blockwise scaled path 读取，也不会读到 stale scale。
- 如果选择“不写全 scale，只依赖 marker bypass”，必须把 quant no-scale detection、dequant no-scale bypass 和覆盖 `scaleDim > 1` 的正确性测试作为同一原子改动落地。

### 4.2 dequant no-scale bypass

对每个 source token：

- 如果 marker 表示 no-scale，则该 source 可以走 unscaled accumulate。
- 如果所有 source 都 no-scale，则直接调用 existing no-scale vector dequant。
- 如果部分 source scaled，考虑拆成：
  - unscaled source group
  - scaled source group

不要让这个优化掩盖 scaled path 的真实性能。

正确性约束：

- 判断 no-scale 只能使用 token marker / 明确 invariant，不能从单个 block scale 推断整个 token no-scale。
- partial source mixed path 必须保证 scaled source 仍使用对应 block scale，unscaled source 不读取 stale scale。
- 测试必须覆盖 `scaleDim=16/32/64`、`scale[0]` marker 为负但 block0 实际 no-scale 的情况，以及一个 token 内只有非 block0 scale-active 的情况。

## Phase 5: launch config 与 tuning

当前 blockwise 沿用普通 nop2p 的默认 launch config，未必适合 scale reduce + dequant multiply。

### 5.1 独立 tuning key

确保 tuning config 维度包含：

- `quant_type=fp8_blockwise`
- `zero_copy`
- `hidden_dim`
- `scale_dim`
- `num_experts_per_token`
- `world_size`
- `scale_active_mode` 或 benchmark distribution 标签

当前状态与迁移要求：

- 现有 combine tuning schema 只按 `dtype`、`num_tokens`、`hidden_dim`、`zero_copy`、`quant_type` 匹配；`world_size` 体现在文件名 `ep{n}` 中。
- `scale_dim`、`num_experts_per_token`、`scale_active_mode` 当前不会参与 lookup，保存进 JSON 也不会自动生效。
- 如果新增这些 required fields，需要同步更新 rule validation、lookup、save tuning config、已有 JSON 迁移；否则旧配置会被跳过或新配置查找时被忽略。
- 建议先做 backward-compatible migration：新字段可选，lookup 优先精确匹配 `scale_dim/num_experts_per_token/scale_active_mode`，找不到时 fallback 到旧规则，并在日志里标记 legacy fallback。

### 5.2 sweep 范围

建议 sweep：

```text
combine_block_num: 40, 64, 80, 128, 160, 256
combine_warp_per_block: 4, 6, 8, 12, 16
```

重点记录：

- EP4 与 EP8 是否需要不同默认值。
- scale-active 与 no-scale 分布是否需要不同默认值。
- register pressure 是否导致高 `warp_per_block` 反而变慢。

## Phase 6: zero-copy / p2p 支持

`--zero-copy 1` 对应 `_p2p`，当前没有开放 `fp8_blockwise`，原因不是算法不能支持，而是 buffer 语义不同：

- zero-copy 下 Python 写入 registered combine input buffer，当前语义是 bf16。
- p2p kernel 直接 remote read 这个 buffer。
- blockwise FP8 不能把这个 bf16 buffer 原地解释为 fp8，也不能安全原地 bf16->fp8。

建议方案：

1. 为 `_p2p` 增加 FP8 shadow buffer。
2. Python 仍写 bf16 registered input buffer。
3. kernel barrier 前把 bf16 input quantize 到 shadow FP8 buffer。
4. scale 写入 `shmemInpScalesMemObj`。
5. p2p dequant remote read shadow FP8 buffer + scale buffer。
6. 注册 `EpCombineIntraNodeKernel_bf16_p2p_fp8bwq`。
7. 放开 Python/C++ guard。

需要修改：

- `src/ops/dispatch_combine/dispatch_combine.cpp`
- `src/ops/dispatch_combine/intranode.hpp`
- `src/ops/kernels/ep_intranode.hip`
- `src/ops/dispatch_combine/launch.cpp`
- `python/mori/ops/dispatch_combine.py`
- tests/bench

## Phase 7: 验收标准

### 7.1 正确性

必须通过：

```bash
/home/nima/build.sh
git diff --check
```

功能测试：

```bash
PYTHONPATH=/home/nima/mori:$PYTHONPATH python tests/python/ops/bench_dispatch_combine.py \
  --cmd bench --dtype bf16 --quant-type fp8_blockwise \
  --max-tokens 4096 --zero-copy 0 --world-size 8 \
  --scale-dim 32 --input-dist uniform --input-scale 1024 --report-scale-stats 1
```

还需覆盖：

- EP4
- small hidden dim
- `scaleDim=16/32/64`
- no-scale distribution
- mixed distribution
- real trace

### 7.2 性能

不要只报告一个 latency。每次提交优化结果时至少报告：

- direct cast baseline
- blockwise no-scale case
- blockwise force-scale-active case
- blockwise mixed case
- scale-active ratio
- `CombineQuantizeInput` 与 `CombineDequantAccum` 拆分 latency

初步目标：

- scale-active 主场景下，明确 quant 与 dequant 各自瓶颈。
- 在不牺牲 scale-active 性能的前提下，no-scale case 接近 direct cast。
- blockwise 的额外开销随 `scaleDim` 增长符合预期，不出现异常超线性退化。

## 推荐执行顺序

1. 增加 benchmark input distribution 与 scale-active stats。（已完成）
2. 打开或补充 profiler slots，拆分 quant/dequant latency。（已完成）
3. 对当前 vectorized scaled path 做 rocprof。（已完成）
4. 优化 dequant scale hoisting 与 hot-path `fabsf`。（已完成，收益有限）
5. 对 `hidden_dim=7168, scale_dim=56, topk=8, weights=None` 增加
   `blockElems=128 / VecBytes=8` 专用 dequant symbol。（已完成）
6. 优化 quant/stage input 的主 shape path。
7. 做独立 tuning sweep。
8. 支持 `--zero-copy 1` / p2p shadow buffer。

## 当前收敛状态

最终保留的性能改动是一个窄 specialization：

```text
EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_vec8
```

启用条件：

```text
kernel_type = IntraNode
dtype = bf16
quant_type = fp8_blockwise
zero-copy = 0
weights = None
hidden_dim = 7168
scale_dim = 56
block_elems = 128
num_experts_per_token = 8
world_size > 4
```

不满足这些条件时继续使用 generic
`EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq`。这避免了 runtime 双 hot path
导致 register allocator 取 liveness union 的问题，也避免影响 weighted
combine、EP4 和其他 shape。

EP8, bf16, `max_tokens=4096`, `hidden_dim=7168`, `scale_dim=56`,
`zero-copy=0`, `dispatch=128/16`, `combine=128/16` 的最终结果：

| Case | generic / pre-specialization | final specialization | delta |
| --- | ---: | ---: | ---: |
| `normal` no-scale | `~1040 us` | `~834 us` | `~ -20%` |
| `uniform[-1024,1024]` scale-active | `~1064 us` | `~862 us` | `~ -19%` |

最终 profiler slot 显示 dequant 已不是最大项：

| Case | `CombineStageInput` | `CombineDequantAccum` |
| --- | ---: | ---: |
| `normal` no-scale | `~495 us` | `~224 us` |
| `uniform[-1024,1024]` scale-active | `~506 us` | `~230 us` |

因此下一步优先级已经从 dequant 转为 quant/stage input：

- 分析 `WarpQuantizeBf16ToFp8BlockwiseVec` 的 bf16 read、max/reduce、fp8
  store、scale store 是否仍有 register pressure 或 memory coalescing 问题。
- 针对 `blockElems=128` 的 quant path 做更窄实现或 disasm 检查。
- 对最终 specialization 做 launch tuning，确认 `128/16` 是否仍最优。
- 后续如果要支持其他 hidden dim，应按 `blockElems=128` 归类，而不是按
  固定 `scale_dim=56` 归类。例如 `hidden_dim=4096, scale_dim=32` 和
  `hidden_dim=8192, scale_dim=64` 都属于同一类 128-element activation block。
