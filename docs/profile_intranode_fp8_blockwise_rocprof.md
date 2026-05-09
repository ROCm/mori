# Intranode FP8 Blockwise rocprof Comparison

## Summary

This document records the rocprofv3 PMC comparison between the three intranode
combine kernel variants:

- `EpCombineIntraNodeKernel_bf16_nop2p`        — bf16 no-quant baseline
- `EpCombineIntraNodeKernel_bf16_nop2p_fp8cast` — bf16↔fp8 direct cast
- `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq`  — bf16↔fp8 blockwise quant

It also includes one scale-active comparison for `fp8bwq` (`uniform[-1024,1024]`
input) to confirm whether the scale-active kernel path differs materially from
the no-scale path on the same kernel.

The headline result: **`fp8bwq` is bottlenecked by register pressure and
scratch spilling, not by the scale math, not by HBM bandwidth, and not by
occupancy on its own**. The optimization implications are at the bottom.

## Test setup

- EP8, `bf16`, `max_tokens=4096`, `hidden_dim=7168`, `--zero-copy 0`
- Pinned launch config: `combine_block_num=128`, `combine_warp_per_block=16`
- Pinned launch config: `dispatch_block_num=128`, `dispatch_warp_per_block=16`
- Default benchmark input distribution `normal` (= `torch.randn` → no
  block ever reaches FP8 max → kernel always hits the no-scale arithmetic
  path even though it goes through the scaled code), and one extra
  `uniform[-1024,1024]` run for `fp8bwq` to drive every block above FP8 max.
- Build: production (no `ENABLE_PROFILER`).
- Tool: `rocprofv3` (rocprofiler-sdk 1.1.0, ROCm 7.2.0).
- Per kernel: PMC counters split into 9 hardware passes (some derived
  counters cannot share a pass on gfx942), with
  `--kernel-include-regex EpCombineIntraNodeKernel_bf16_nop2p[_<variant>]`
  and `--kernel-iteration-range 1 5`. The first dispatch per process is
  dropped because it captures cold-start outliers from CUDA graph capture.
- Static info (VGPR/SGPR/LDS/Scratch/workgroup/grid) is reported by
  rocprofv3 on every counter row and is identical across PMC passes for a
  given kernel.

The PMC sweep helper lives at `/tmp/rocprof_run.sh`, the aggregator at
`/tmp/rocprof_analyze.py`, and raw CSVs under `/tmp/rocprof-fp8bwq/<tag>_<dist>_<scale>/g**`.
These are not committed; they are intentionally regeneratable via the
`bench_dispatch_combine.py --input-dist ... --report-scale-stats 1` plumbing
introduced in this branch.

## Static kernel info (single-pass)

| Metric                 | none (bf16) | fp8cast | fp8bwq |
| ---                    | ---:        | ---:    | ---:   |
| `Workgroup_Size`       | 1024        | 1024    | 1024   |
| warps / block          | 16          | 16      | 16     |
| `Grid_Size` (threads)  | 131072      | 131072  | 131072 |
| blocks                 | 128         | 128     | 128    |
| `VGPR_Count`           | 64          | 100     | **128** |
| `Accum_VGPR_Count`     | 0           | 4       | 0      |
| `SGPR_Count`           | 112         | 112     | 112    |
| `LDS_per_block` (B)    | 0           | 0       | 0      |
| `Scratch_per_thread` (B) | 64        | 64      | **1136** |

`fp8bwq` is the only variant whose compiler hit the VGPR ceiling and spilled
the rest to **scratch (1136 bytes per thread, vs 64 for the other two)**. This
is the most important static fact in this report.

Implied scratch footprint for the kernel launch, on the same shape:

```text
1136 B/thread × 1024 threads/block × 128 blocks ≈ 148 MB scratch traffic per launch
```

vs. about `64 B × 1024 × 128 ≈ 8.4 MB` for `none` / `fp8cast`. AMD
private-memory scratch reads/writes are routed through L1/L2 and surface as
LDS / TCC traffic in the dynamic counters below — and they do.

## Dynamic counters (averaged across 8 ranks)

The first dispatch per process per counter is dropped to avoid the
cold-start outlier captured by `--kernel-iteration-range 1 5`. The rest are
averaged over the remaining (rank × iteration) samples.

| Counter             | none (bf16) | fp8cast    | fp8bwq      | fp8bwq:cast | notes |
| ---                 | ---:        | ---:       | ---:        | ---:        | --- |
| `MeanOccupancyPerCU`| 6.55        | 6.34       | 6.34        | 1.00x       | similar |
| `OccupancyPercent`  | 20.45 %     | 19.81 %    | 19.81 %     | 1.00x       | identical |
| `VALUBusy`          | 2.78 %      | 4.37 %     | **7.60 %**  | 1.74x       | dramatically more VALU work |
| `VALUUtilization`   | 95.1 %      | 99.0 %     | **87.4 %**  | 0.88x       | partial-wave use in blockwise |
| `SALUBusy`          | 1.31 %      | 0.69 %     | **2.54 %**  | 3.68x       | scalar dispatch heavier |
| `MemUnitStalled`    | 0.09 %      | 0.10 %     | **0.93 %**  | 9.3x        | mem unit stalled ~10x more |
| `FetchSize` (B)     | 607 K       | 455 K      | 509 K       | 1.12x       | DRAM read traffic comparable |
| `WriteSize` (B)     | 361 K       | 209 K      | **747 K**   | 3.57x       | scratch + scale write traffic |
| `TCC_HIT_sum`       | 85 K        | 70 K       | **10.12 M** | **145x**    | scratch lives in L2 |
| `TCC_MISS_sum`      | 4.84 M      | 3.64 M     | 4.92 M      | 1.35x       | DRAM-bound traffic comparable |
| `SQ_INSTS_VALU`     | 18.3 M      | 15.7 M     | **56.2 M**  | 3.58x       | 3.6x more VALU instructions |
| `SQ_INSTS_SALU`     | 8.65 M      | 2.65 M     | **19.3 M**  | 7.28x       | 7x more SALU instructions |
| `SQ_INSTS_VMEM_RD`  | 1.65 M      | 765 K      | 1.29 M      | 1.69x       | global reads similar to bf16 |
| `SQ_INSTS_VMEM_WR`  | 535 K       | 363 K      | **1.29 M**  | 3.55x       | 3.5x more global writes |
| `SQ_INSTS_LDS`      | 24.6 K      | 24.6 K     | **1.24 M**  | **50.3x**   | scratch via LDS pathway |

Reading the table:

- **HBM bandwidth is not saturated.** `FetchSize` and `WriteSize` (=
  HBM/DRAM traffic) for `fp8bwq` are within 12-360 % of `fp8cast`, on the
  same order as `bf16`. If we were memory-bound we would not expect
  `fp8bwq` to be 2x slower than `fp8cast` despite carrying half the byte
  payload.
- **Occupancy is not the differentiator.** All three kernels run at 6.3-6.6
  waves/CU (~20 % of theoretical max). The shape is bounded by
  `workgroup_size=1024` regardless of register count; pushing VGPR up to
  128 did not measurably degrade occupancy on this launch config.
- **Scratch spilling is the smoking gun.** `Scratch_per_thread = 1136 B`
  for `fp8bwq` (vs 64 B), `SQ_INSTS_LDS` ≈ **50× higher**, `TCC_HIT_sum`
  ≈ 145× higher (scratch is L2-resident), and `SQ_INSTS_VMEM_WR` is
  3.5× higher (scratch writes go to L2/DRAM). Together these match the
  classic AMD register-spill signature.
- **Compute work also went up.** `SQ_INSTS_VALU` is 3.6× and `SQ_INSTS_SALU`
  is 7.3× higher than `fp8cast`. Some of this is the legitimate blockwise
  arithmetic (per-block max-reduce, per-source scale read+multiply on the
  dequant side), but a large fraction is almost certainly scratch-aware
  spill/reload sequences, address arithmetic for the sub-warp partition,
  and the pointer-array setup that goes through SGPR.
- **`VALUUtilization` drops 12 points (99→87)** when going from `fp8cast`
  to `fp8bwq`. That gap is the partial-wave subwarp work — at
  `blockElems=224`, the chosen `SubwarpSize=16, VecBytes=16` mapping
  leaves ~12.5 % of lanes idle on average, which is consistent with the
  observed ratio.

## Scale-active vs no-scale on `fp8bwq`

To make sure the bottleneck does not shift when actual scaling kicks in, we
also profiled `fp8bwq` with `--input-dist uniform --input-scale 1024.0`
(every block has `max|x| = 1024 > FP8 max`, so the kernel is on the
fully-scaled path).

| Counter            | fp8bwq normal | fp8bwq uniform-1024 | delta |
| ---                | ---:          | ---:                | ---:  |
| `VALUBusy`         | 7.60 %        | 8.10 %              | +6.6 % |
| `VALUUtilization`  | 87.35 %       | 84.12 %             | -3.7 pt |
| `SALUBusy`         | 2.54 %        | 2.54 %              | flat  |
| `MemUnitStalled`   | 0.93 %        | 0.91 %              | flat  |
| `FetchSize`        | 509 K         | 512 K               | flat  |
| `WriteSize`        | 747 K         | 748 K               | flat  |
| `TCC_HIT_sum`      | 10.12 M       | 10.09 M             | flat  |
| `TCC_MISS_sum`     | 4.92 M        | 4.94 M              | flat  |
| `SQ_INSTS_VALU`    | 56.23 M       | 59.55 M             | +5.9 % |
| `SQ_INSTS_SALU`    | 19.26 M       | 19.28 M             | flat  |
| `SQ_INSTS_VMEM_RD` | 1.286 M       | 1.313 M             | +2.1 % |
| `SQ_INSTS_VMEM_WR` | 1.285 M       | 1.306 M             | +1.6 % |
| `SQ_INSTS_LDS`     | 1.237 M       | 1.237 M             | flat  |

Interpretation:

- The actual scale-active arithmetic costs ~6 % more VALU, ~2 % more global
  reads/writes. Everything else is flat.
- The scratch / LDS / TCC profile is **identical**, confirming that the
  spill cost dominates the kernel and is paid regardless of whether the
  data triggers scaling.
- This means future optimizations should be evaluated under both
  distributions, but the optimization target itself is the spill, not the
  scale-active code path.

## What this rules out

- HBM bandwidth limit. `none` (= bf16, 2x more user bytes per token) and
  `fp8bwq` move similar DRAM traffic, and `fp8cast` moves *less* DRAM
  traffic than both — yet `fp8cast` is the fastest. So we are not on the
  HBM ceiling.
- Pure occupancy limit. All three kernels are at the same ~20 %
  `OccupancyPercent`. Forcing more VGPR for `fp8bwq` did not measurably
  drop occupancy compared to `fp8cast` (100 VGPR) or `none` (64 VGPR) on
  this launch config.
- Pure scale arithmetic cost. The `normal` vs `uniform-1024` comparison
  shows the actual scale multiply is ~6 % of total VALU.

## What this implicates

1. **Register pressure → scratch spill is the dominant pathology.**
   - `Scratch_per_thread` jumps from 64 B to 1136 B going from `fp8cast`
     to `fp8bwq`.
   - `SQ_INSTS_LDS` jumps from 24.6 K to 1.24 M (50×).
   - `TCC_HIT_sum` jumps from 70 K to 10.12 M (145×).
   - `SQ_INSTS_VMEM_WR` jumps from 363 K to 1.29 M (3.5×).
   - All of those are typical fingerprints of "compiler ran out of VGPR
     and started spilling around the unrolled `AccumNum=8` loops".

2. **The `WarpAccumFp8DequantFullImpl` / `SegmentImpl` setup is the most
   likely spill site.**
   - It builds `cachedSrcs[AccumNum]` and `cachedSrcBytes[AccumNum]` —
     for `AccumNum=8` that is 16 pointers = 32 VGPR just for redundant
     pointer caching.
   - It builds `sbScales[AccumNum]` per scale block.
   - It also builds `acc01[kSegs]` and `acc23[kSegs]` per inner vector
     iteration.
   - With `AccumNum=8`, `kSegs=4`, the unrolled body easily exceeds the
     VGPR budget and the compiler spills.

3. **The earlier dequant cleanup attempt (move `fabsf` out / hoist scale
   per block) was attacking the wrong layer.** The savings would be
   single-digit VALU instructions per inner loop, but the bottleneck is
   the scratch round-trip for every spilled register, not the inner-loop
   instruction count.

4. **Quant side likely has the same pathology.** The
   `WarpQuantizeBf16ToFp8BlockwiseVec<16, 16, 4, ...>` template uses a
   `MaxCacheIters=4`-sized `cached[MaxCacheIters]` array per subwarp. For
   `blockElems=224, kStrideElems=128` this is `maxIters=2`, so only 2 of
   4 entries are used, but the compiler may still allocate 4 registers
   for the array. Combined with the per-warp scaling computation that's
   another easy spill source.

## Concrete optimization targets driven by these data

These supersede the old "scale hoisting / `fabsf` removal" line item,
which the current evidence does not support as a high-leverage change.

### Target A (highest priority): cut VGPR pressure on the dequant path

In `include/mori/core/transport/p2p/device_primitives.hpp`,
`WarpAccumFp8DequantFullImpl` and `WarpAccumFp8DequantSegmentImpl` build
two redundant local arrays:

```text
const Fp8T* cachedSrcs[AccumNum];
const __hip_fp8_storage_t* cachedSrcBytes[AccumNum];
```

For `AccumNum=8` that is 16 pointers. They are then passed by `const*` to
the inner helper which itself reads from them. Since `srcs[i]` is already
in shared memory and `cachedSrcBytes[i]` is just a reinterpret of
`cachedSrcs[i]`, both arrays can be removed — the helper can read
`reinterpret_cast<const __hip_fp8_storage_t*>(srcs[i])` directly. Net
expected saving: up to 32 VGPR.

Also worth trying:

- Drop `Accum_VGPR_Count=0` for `fp8bwq` (currently 0; not the gain
  source, just a sanity check that we are not leaving accumulator VGPRs
  on the table).
- In `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`, the per-source
  inner loop reads `srcScales[i][sb]` once per vector iteration. Hoisting
  to a per-`sb` register cache (8 floats once we are inside a scale
  block) is still desirable, but only if it does not push us back up into
  spill territory. Combine this with the array elimination above and
  measure VGPR/scratch in the same rocprof loop.

### Target B (second priority): cut VGPR pressure on the quant path

`WarpQuantizeBf16ToFp8BlockwiseVec<SubwarpSize, InVecBytes, MaxCacheIters, Fp8T>`
declares `cached[MaxCacheIters]` even when only `maxIters` slots are
used. For the main `(blockElems=224, SubwarpSize=16, InVecBytes=16)`
shape, only 2 cache slots are actually written; the other 2 still pin
VGPR for the entire scale block. Suggested change: write a
`maxIters==2` specialization that uses two named locals (`packed0`,
`packed1`) instead of the array, so the compiler can keep them in flight
without an array-shape spill.

### Target C: re-evaluate the `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`
hot path

After Target A, the per-vector cost picture changes: hoisting scale
loads, removing `fabsf` (already done as a marker fix in the current
branch, see `device_primitives.hpp` diff), and specializing
`blockElems=224` to skip the per-vector `globalIdx / blockElems`
division all become valid follow-on optimizations. They must be measured
under both the `normal` and `uniform-1024` distributions — Target A may
compress the gap between `fp8bwq` and `fp8cast` enough that the relative
weight of these inner-loop fixes goes up.

### Target D: AccumNum=8 dedicated path

`AccumNum=8` (the common `num_experts_per_token=8` case) is the
worst-case for the local pointer arrays. After Target A, write an
`AccumNum==8` specialization of `WarpAccumFp8DequantFullImpl` that
unrolls the source loop manually and uses named pointers
(`s0, s1, ..., s7`) rather than a `[AccumNum]` array. Measure VGPR /
scratch / `SQ_INSTS_LDS` again.

### Non-targets

- HBM bandwidth-side optimizations (e.g. shrinking the staging buffer or
  rearranging the layout) are not justified by this data. We are not
  near the HBM ceiling.
- Launch config sweeps (`combine_block_num=128/16` already wins by ~9 %
  vs `80/16`) remain a separate, low-risk track but will not move the
  needle past the scratch-spill ceiling on their own.

## scaleDim experiment: 32 vs 56 (`block_elems` 224 vs 128)

### Why try it

Hidden dim 7168 with `scale_dim=32` gives `block_elems = 7168 / 32 = 224`,
which is not a power of two and does not divide
`SubwarpSize × kVecElems = 128`. With `scale_dim=56`, `block_elems = 128`
which is a power of two AND equals `SubwarpSize × kVecElems`. The change
is one Python-side flag (`--scale-dim 56`) and triggers two cleaner code
paths:

- **Quant**: `WarpQuantizeBf16ToFp8BlockwiseVec` enters the
  `maxIters == 1` fast path, replacing the `cached[MaxCacheIters=4]`
  array with a single named local `cached0`.
- **Dequant**: `WarpAccumFp8DequantVecRangeBlockwiseScaleWave` sets
  `blockPow2 = true`, so the per-vector `globalIdx / 224` integer
  division becomes a single `globalIdx >> 7` shift.

Also, with `block_elems = 128`, a warp's outer iteration of 1024 bytes
covers exactly 8 scale blocks (lanes 0..7 share one `sb`, lanes 8..15
the next, etc.), so the per-source `srcScales[i][sb]` reads coalesce per
group of 8 lanes instead of being divergent across the whole warp.

`block_elems = 128` is also the DeepSeek-style per-token activation
scale block size matching 128×128 weight tiles, so this is the layout
the downstream FP8 GEMM consumer expects regardless of perf.

### Combine latency, same launch config (128 / 16), same env

| Input dist             | scale_dim=32 combine | scale_dim=56 combine | delta    |
| ---                    | ---:                 | ---:                 | ---:     |
| `normal` (no-scale)    | ~1188 us             | ~1135 us             | -53 us (-4.5 %) |
| `uniform[-1024,1024]`  | ~1312 us             | ~1154 us             | -158 us (-12.0 %) |

Both numbers are non-profiler bench averages from the same env, same
build, same launch config. Three correctness sanity checks (`normal`,
`uniform-1024`, `force-scale-active`) all pass for `scale_dim=56`.

The scale-active gain is ~3× the no-scale gain, which is exactly what
the trace predicts: the per-vector div→shift and the per-source scale
coalescing matter more when the dequant inner body is actually doing
non-trivial multiplies, not just multiplying by 1.0f.

### rocprof counters: scale_dim=56 vs scale_dim=32 (fp8bwq, normal)

| Counter            | sd=32      | sd=56      | delta      | reading |
| ---                | ---:       | ---:       | ---:       | --- |
| `VGPR_Count`       | 128        | **128**    | unchanged  | static budget unchanged |
| `Scratch_per_thread` | 1136 B   | **1136 B** | unchanged  | static spill region unchanged |
| `OccupancyPercent` | 19.81 %    | 19.80 %    | flat       | not occupancy-driven |
| `VALUUtilization`  | 87.35 %    | **92.02 %** | **+4.7 pt** | better lane utilization |
| `VALUBusy`         | 7.60 %     | 8.00 %     | +5.3 %     | more useful work / unit time |
| `MemUnitStalled`   | 0.93 %     | 0.92 %     | flat       | not stalled differently |
| `FetchSize`        | 509 K      | 501 K      | -1.6 %     | DRAM read traffic flat |
| `WriteSize`        | 747 K      | 746 K      | flat       | DRAM write traffic flat |
| `TCC_HIT_sum`      | 10.12 M    | **1.53 M** | **-85 %**  | huge drop in L2 (scratch) hits |
| `TCC_MISS_sum`     | 4.92 M     | 4.86 M     | -1.2 %     | DRAM-bound traffic essentially the same |
| `SQ_INSTS_VALU`    | 56.23 M    | 56.64 M    | +0.7 %     | total VALU insts essentially the same |
| `SQ_INSTS_SALU`    | 19.26 M    | 18.57 M    | -3.6 %     | fewer scalar insts (the `idx/224` divide vanished) |
| `SQ_INSTS_VMEM_RD` | 1.286 M    | **0.904 M** | **-29.7 %** | fewer global vec reads |
| `SQ_INSTS_VMEM_WR` | 1.285 M    | 1.025 M    | -20.4 %    | fewer global vec writes |
| `SQ_INSTS_LDS`     | 1.237 M    | 2.017 M    | **+63.0 %** | LDS instruction count rose |

Reading the table:

- **The static register/scratch budget did not move** (still 128 VGPR,
  still 1136 B/thread scratch). This rules out the optimistic
  hypothesis that `scale_dim=56` would shrink the dequant unrolled
  body below the AMD VGPR ceiling. The dominant register hog
  (`cachedSrcs[8] + cachedSrcBytes[8] + acc01[4] + acc23[4]` per
  inner iteration of the dequant path with `AccumNum=8`) is
  unaffected by `block_elems`.
- **`TCC_HIT_sum` collapsed by 85 %.** Scratch lives in L2 on AMD,
  and the dequant path with `block_elems=224` re-fetched scale-block
  boundaries through L2 on every divergent per-lane lookup. With
  `block_elems=128` aligned to the warp's 1024-byte stride, the
  same scratch slots are touched many fewer times. This is the
  single biggest mover.
- **`SQ_INSTS_VMEM_RD` dropped 30 %, `SQ_INSTS_VMEM_WR` dropped 20 %.**
  Per-vec accesses are coalescing better; some spill traffic that
  previously went out to L2/DRAM as separate transactions now
  consolidates.
- **`SQ_INSTS_LDS` rose 63 %** while `TCC_HIT_sum` fell 85 %. Net,
  the spill traffic moved from "scratch round-tripping through L2"
  into the LDS pathway, which on gfx942 is roughly an order of
  magnitude faster per access. So the LDS rise is the *good* kind
  of accounting shift, not a regression. (`LDSBankConflict` was
  zero in both runs, so the LDS traffic is in the well-coalesced
  regime.)
- **`VALUUtilization` rose 4.7 pt (87.35 → 92.02 %).** The
  `block_elems=128` partition has perfectly aligned lane groups
  (8 lanes per scale block), removing the per-lane sb-divergence
  that wasted lanes under `block_elems=224`.
- **`SALUBusy` flat, `SQ_INSTS_SALU` -3.6 %.** The integer division
  by 224 is gone, but the savings are absorbed inside what is
  already a tiny SALU fraction of the kernel.

### Conclusion on `scale_dim=56`

- `scale_dim=56` is a **strict win** for both correctness-equivalent
  perf (-4.5 % on no-scale, -12 % on scale-active combine latency)
  and downstream alignment (matches DeepSeek-style 128-element
  activation blocks).
- It does **not** dissolve the register-pressure / scratch-spill
  bottleneck. `Scratch_per_thread` stays at 1136 B and `VGPR_Count`
  stays at 128 (the AMD ceiling). The relative gap to `fp8cast`
  shrinks but does not close — with `scale_dim=56` we are still
  ~85 % slower than `fp8cast` on `normal`, vs ~100 % slower with
  `scale_dim=32`.
- Target A from the prior section (eliminating the redundant
  `cachedSrcs[AccumNum]` / `cachedSrcBytes[AccumNum]` arrays in
  `WarpAccumFp8DequantFullImpl` / `WarpAccumFp8DequantSegmentImpl`)
  is still required to attack the static VGPR/scratch ceiling.
- Recommended: switch the production default for blockwise quant on
  `hidden_dim=7168` from `scale_dim=32` to `scale_dim=56`, keep the
  per-vec div→shift improvement, and revisit Target A separately on
  top of the new baseline.

## Target A trial: drop the redundant `cachedSrcBytes[AccumNum]` array

### Change

In `include/mori/core/transport/p2p/device_primitives.hpp`, the four
inner dequant helpers used to take both `srcs` (`Fp8T**`) and `srcBytes`
(`__hip_fp8_storage_t**`) as separate parameters, and the two `Impl`
functions built two parallel `cachedSrcs[AccumNum]` /
`cachedSrcBytes[AccumNum]` register arrays from the same source data
(the second is just a reinterpret of the first). The hypothesis was
that the parallel arrays cost ~16 extra VGPR per lane for `AccumNum=8`
and were a major scratch-spill driver.

The change drops `srcBytes`/`cachedSrcBytes` everywhere and derives the
byte view via `reinterpret_cast<const __hip_fp8_storage_t*>(src)` at
the use site (same bit pattern, free). Affected helpers:

- `WarpAccumFp8DequantVecRange<...>`
- `WarpAccumFp8DequantBlockwiseVec<...>`
- `WarpAccumFp8DequantSegmentBlockwiseVec<...>`
- `WarpAccumFp8DequantVecRangeBlockwiseScaleWave<...>`

Plus matching simplifications in `WarpAccumFp8DequantFullImpl` and
`WarpAccumFp8DequantSegmentImpl` (alignment checks now read from
`cachedSrcs` directly, the `useVec2` no-scale fallback derives the
byte view inline).

### Correctness

6/6 sanity checks pass: `scale_dim ∈ {32, 56}` × `input_dist ∈ {normal,
uniform-1024, normal+force_scale_active}`. The `fp8_direct_cast`
intranode pytest subset still passes.

### rocprof: VGPR / Scratch did NOT move

| Counter             | sd=32 normal pre | sd=32 normal post | sd=56 normal pre | sd=56 normal post | sd=32 uniform-1024 pre | sd=32 uniform-1024 post |
| ---                 | ---:             | ---:              | ---:             | ---:              | ---:                   | ---:                    |
| `VGPR_Count`        | 128              | **128**           | 128              | **128**           | 128                    | **128**                 |
| `Scratch_per_thread`| 1136 B           | **1136 B**        | 1136 B           | **1136 B**        | 1136 B                 | **1136 B**              |
| `SQ_INSTS_LDS`      | 1.237 M          | 1.237 M           | 2.017 M          | 2.017 M           | 1.237 M                | 1.237 M                 |
| `TCC_HIT_sum`       | 10.12 M          | 10.11 M           | 1.529 M          | 1.511 M           | 10.09 M                | 10.08 M                 |
| `TCC_MISS_sum`      | 4.92 M           | 4.92 M            | 4.86 M           | 4.86 M            | 4.94 M                 | 4.95 M                  |
| `VALUUtilization`   | 87.35 %          | 87.39 %           | 92.02 %          | 92.09 %           | 84.12 %                | 84.14 %                 |
| `SQ_INSTS_VALU`     | 56.23 M          | 55.98 M           | 56.64 M          | 56.38 M           | 59.55 M                | 59.30 M                 |
| `SQ_INSTS_SALU`     | 19.26 M          | 19.08 M           | 18.57 M          | 18.37 M           | 19.28 M                | 19.09 M                 |

The static register / scratch budget is **completely unchanged**, and
all dynamic counters move within run-to-run noise. The compiler was
already folding the two `cachedSrcs[]` and `cachedSrcBytes[]` arrays
into the same VGPR allocation before the change — the redundant
parallel array was a source-readability problem, not the
dominant register-pressure driver we hypothesised it to be.

### Non-profiler bench

| cell                       | pre-Target-A | post-Target-A | delta                |
| ---                        | ---:         | ---:          | ---:                 |
| sd=32 `normal`             | ~1188 us     | ~1192 us      | flat (noise)         |
| sd=56 `normal`             | ~1135 us     | ~1139 us      | flat (noise)         |
| sd=32 `uniform[-1024,1024]`| ~1312 us     | **~1204 us**  | **-108 us (-8.2 %)** |
| sd=56 `uniform[-1024,1024]`| ~1154 us     | ~1152 us      | flat (noise)         |

The single non-noise bench delta lives entirely in the legacy sd=32 +
scale-active corner. There is no measurable rocprof signal we can
attribute to it; it is most likely an instruction-scheduling change
the PMC counters do not reflect (fewer source-level pointer references
let the compiler reorder loads slightly differently). For the
production sd=56 path the change is net-zero on perf.

### Conclusion on Target A

- Correctness-preserving cleanup: removes a confusing parallel array
  from the dequant impls and trims the helper signatures.
- Performance: no regression on any tested cell; the only measurable
  win is on the legacy sd=32 + scale-active path.
- The original Target A hypothesis (the parallel arrays drive 32 VGPR
  of pressure → scratch spill) is **not supported by the data**. The
  AMD VGPR ceiling and the 1136 B/thread scratch budget come from
  somewhere else.

### What this implies for the next step

The remaining gap to `fp8cast` (sd=56 + Target A: 1139 us / 1152 us
combine vs `fp8cast` 611 us) is still real, and the bottleneck profile
from the prior section (1136 B/thread scratch, 50× SQ_INSTS_LDS vs
`fp8cast`) is intact. We have to look elsewhere for the spill driver.
Likely candidates:

- **Quant-side VGPR is the kernel-wide ceiling**, not dequant. Both the
  quant and dequant phases live in the same kernel and the compiler
  takes `max(quant_VGPR, dequant_VGPR)`. If the quant
  `WarpQuantizeBf16ToFp8BlockwiseVec` path is the one hitting 128
  VGPR, no amount of dequant cleanup will reduce the kernel ceiling.
  Confirm by inspecting `--save-temps` / disasm.
- **`acc01[kSegs=4] + acc23[kSegs=4] = 16 floats per lane`** are
  per-iteration accumulators that must live across the unrolled
  `AccumNum=8` body. With 8 sources × 4 segments × 2 float2 each, the
  compiler may keep many partial accumulates in flight for ILP. This
  is structural — would require a `VecBytes=8` (`kSegs=2`) variant or
  manually scheduled inner body to shrink.
- **`cachedSrcs[AccumNum]` itself**, if the compiler is failing to
  recognise the source pointers as warp-uniform and is keeping them in
  VGPR rather than SGPR. (Tried — see Candidate 3 below — it didn't
  move the needle.)
- The kernel has many non-quant-non-dequant moving parts (multi-warp
  iter decode, pointer prep, barrier handling) that could be the
  ceiling. The split profiler slots already showed `CombinePreparePtrs`
  has visible wall-clock; it may also be a register pressure source.

Recommended next concrete step: get the actual disassembly for the
`fp8bwq` kernel (e.g. by saving the JIT hsaco and disassembling with
`roc-obj` or by a one-shot rebuild with `-save-temps`) and look at
where the 1136 bytes of scratch reads/writes are issued. Without that,
we are guessing about which of the candidates above to attack first.

## Candidate 3 trial: drop `cachedSrcs[AccumNum]` entirely (REVERTED)

### Hypothesis

After Target A landed without changing `VGPR_Count` or
`Scratch_per_thread`, the next plausible explanation was that the
remaining `cachedSrcs[AccumNum]` register array in the dequant impls
was forcing the source pointers into per-lane VGPR. Since the
pointers are warp-uniform (every lane reads the same address from
shared memory), reading `srcs[i]` directly from LDS at use sites
might let the compiler scalar-promote them to SGPR and free up to
~16 VGPR per lane.

### Change tried

In both `WarpAccumFp8DequantFullImpl` and
`WarpAccumFp8DequantSegmentImpl`, removed the
`const Fp8T* cachedSrcs[AccumNum];` declaration and the population
loop, replaced every `cachedSrcs[i]` use with `srcs[i]`, and passed
`srcs` (the LDS-resident pointer-of-pointers) directly to the
helpers instead of the local cache.

### Result

3/3 sd=56 correctness checks pass (normal / uniform-1024 /
force-scale-active).

Bench, same launch config 128/16, 10 rounds each:

| cell                       | post-Target-A | post-Candidate-3 | delta            |
| ---                        | ---:          | ---:             | ---:             |
| sd=56 `normal`             | ~1139 us      | ~1137 us         | -2 us (-0.2 %)   |
| sd=56 `uniform[-1024,1024]`| ~1152 us      | ~1151 us         | -1 us (-0.1 %)   |

Both deltas are well inside the per-bench round-to-round spread
(~3-7 us). No measurable benefit on the production sd=56 path. The
LDS-uniform-promotion-to-SGPR optimisation either did not happen,
or happened but did not change the binding ceiling.

### Decision

Reverted. The dequant impls keep their `cachedSrcs[AccumNum]` local
array. The hypothesis is now also strikethrough above ("Tried —
didn't move the needle"). The two paths still worth attacking
without disasm are the per-iteration accumulator footprint
(`acc01/acc23[kSegs]`) and the quant-side VGPR ceiling.

## Disassembly: where the 1136 B scratch actually came from

After Target A and Candidate 3 both failed to move `Scratch_per_thread`
or `VGPR_Count`, we extracted the JIT-cached hsaco and disassembled
the gfx942 ELF for the `fp8bwq` kernel. This finally answered the
"what is the 1136 B scratch?" question and pointed at a different
optimisation than the earlier register-pressure hypotheses.

### Setup

```bash
HSACO=/root/.mori/jit/gfx942_mlx5/latest/ep_intranode.hsaco
mkdir -p /tmp/disasm-fp8bwq && cd /tmp/disasm-fp8bwq

# hsaco is a clang offload bundle; unbundle the gfx942 ELF.
/opt/rocm/llvm/bin/clang-offload-bundler --type=o --input "$HSACO" \
    --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    --output ep_intranode.gfx942.elf

# AMDGPU disassembly.
/opt/rocm/llvm/bin/llvm-objdump -d --triple=amdgcn-amd-amdhsa \
    --mcpu=gfx942 ep_intranode.gfx942.elf > ep_intranode.s

# Static kernel descriptor metadata is exposed as ".private_seg_size",
# ".num_vgpr", ".numbered_sgpr" absolute symbols.
/opt/rocm/llvm/bin/llvm-readobj --syms ep_intranode.gfx942.elf \
    | grep "fp8bwq\.private_seg_size"
```

### What the disassembly showed (pre-Target B)

The `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` symbol at file
offset `0xf2c00` is a **251-line wrapper**, not the actual kernel
work. The wrapper:

1. Loads ~700 bytes of `EpDispatchCombineArgs<bf16>` from the
   kernel-arg buffer (`s_load_dwordx8 s[...], s[0:1], 0xN`,
   chunk-by-chunk through offsets 0..0x1c0).
2. Spills the entire struct into a per-thread scratch frame via
   45 `scratch_store_dwordx4 off, v[...], off offset:N` instructions
   (covering offsets 0..720).
3. Sets up the body's calling-convention frame and `s_swappc_b64
   s[30:31], s[0:1]` calls into the body function
   `_ZN4mori3moe29EpCombineIntraNodeKernel_bodyI12hip_bfloat16Lb0ELb0ELb0ELb1EEE...`.

The body function then lives at file offset `0x83740` and runs for
~83 K disassembly lines. Inside the body, every access to an
`args.*` field compiles to a `scratch_load_dword vN, v0, off
offset:K` (where `v0` is the per-thread base of the wrapper's spill
frame). We counted in the body alone:

- 437 `scratch_load_dword` + 242 `scratch_load_dwordx2` +
  10 `scratch_load_ushort` + 7 `scratch_load_dwordx4` =
  **696 scratch loads**
- 26 `scratch_store_dword` + 19 `scratch_store_dwordx2` +
  3 `scratch_store_dwordx4` = **48 scratch stores**

A separate analysis of the offsets confirmed two distinct frames:

- `v0`-relative offsets (24, 32, 80, 96, 112, ..., 504, 584, 616) are
  re-reads of `args.*` fields from the wrapper-allocated 720 B spill
  frame.
- `s33`-relative offsets (88, 92, 108, 136, 140, ..., 332) are the
  body's own register-save area (~352 B for callee-saved VGPRs and
  intermediate spills).

`(720 wrapper args spill) + (~352 body callee-saves) + (~64 caller-frame
overhead) = 1136 B`, which exactly matches the
`fp8bwq.private_seg_size = 0x470` symbol value.

### Why the other kernels did not pay this cost

For comparison, the disassembly for
`EpCombineIntraNodeKernel_bf16_nop2p_fp8cast` is **only 46 lines**.
There is **no** scratch_store at entry, no `s_swappc` to a separate
body, and `args.*` field reads compile straight to
`s_load_dword s?, s[0:1], 0xN` against the kernel-arg buffer
(constant memory). The body got fully **inlined into the wrapper**.

The `fp8bwq` body is large enough (the
`if constexpr (UseFp8BlockwiseQuant)` paths add the per-block
max-reduce, scale store, scale load, and full blockwise dequant
tree on top of the `fp8cast` path) that the compiler's inlining
heuristic refused to inline it. Once it is a separate function, the
AMDGPU calling convention requires the `~700 B EpDispatchCombineArgs`
to be passed via stack — which on AMDGPU means private memory =
scratch. Hence the wrapper's 45 `scratch_store_dwordx4` at entry
and the body's 696 `scratch_load_*` re-reads throughout execution.

In short: the 1136 B per-thread scratch is the **`args` struct
copied by value into per-thread private memory at kernel entry**,
not register spills inside the dequant inner loop.

## Target B trial: `__forceinline__` the body to eliminate the args spill

### Change

In `src/ops/dispatch_combine/intranode.hpp`, mark the
`EpCombineIntraNodeKernel_body<>` template `__device__
__forceinline__` so the compiler is forced to inline it into the
`__global__ wrapper`. With the body inlined, `args.*` reads can
compile against the kernel-arg buffer (constant memory, `s_load_dword`)
exactly like the `fp8cast` kernel does, instead of bouncing through
per-thread scratch.

### Correctness

3/3 sd=56 sanity cases pass: `normal`, `uniform[-1024,1024]`,
`normal+force_scale_active`. No new compilation warnings about
inlining failure.

### Static kernel metadata

Both the kernel descriptor symbol value and the disassembly
agree:

| metric                | pre Target B | post Target B | delta            |
| ---                   | ---:         | ---:          | ---:             |
| `private_seg_size`    | 0x470 = 1136 B | **0xC0 = 192 B** | **-944 B (-83 %)** |
| `num_vgpr`            | 0x80 = 128   | 0x80 = 128    | unchanged        |
| `numbered_sgpr`       | 0x70 = 112   | 0x64 = 100    | -12 (-11 %)      |
| body function symbol  | present      | **gone (inlined)** | —                |
| wrapper-entry `scratch_store_dwordx4` | 45  | 2 (saved-reg only) | -43           |
| total scratch ops in kernel | ~789   | 147           | **-81 %**        |

The `private_seg_size` going from 1136 to 192 means the wrapper's
720 B args-spill region is gone entirely. The remaining 192 B is
the body's small register-save area plus a handful of true register
spills.

### rocprof, sd=56 normal (128/16, 10 rounds)

| Counter                | post-Target-A | post-Target-B (this) | delta              |
| ---                    | ---:          | ---:                 | ---:               |
| `Scratch_per_thread`   | 1136 B        | **192 B**            | **-83 %**          |
| `VGPR_Count`           | 128           | 128                  | unchanged          |
| `OccupancyPercent`     | 19.80 %       | 19.74 %              | flat               |
| `VALUUtilization`      | 92.02 %       | 91.90 %              | flat               |
| `MemUnitStalled`       | 0.92 %        | 0.94 %               | flat               |
| **`TCC_HIT_sum`**      | 1.51 M        | **313 K**            | **-79 %**          |
| **`TCC_MISS_sum`**     | 4.86 M        | 3.79 M               | -22 %              |
| **`FetchSize`**        | 501 K         | 472 K                | -6 %               |
| **`WriteSize`**        | 746 K         | 628 K                | **-16 %**          |
| **`SQ_INSTS_VMEM_RD`** | 904 K         | 770 K                | **-15 %**          |
| **`SQ_INSTS_VMEM_WR`** | 1.03 M        | 860 K                | **-16 %**          |
| `SQ_INSTS_VALU`        | 56.6 M        | 54.7 M               | -3 %               |
| `SQ_INSTS_SALU`        | 18.6 M        | 17.5 M               | -6 %               |

Every memory-side counter that previously carried scratch traffic
came down. `TCC_HIT_sum` shrank ~5x because the args-field re-reads
no longer round-trip through L2; `TCC_MISS_sum` dropped 22 % because
even at L2-hit rate the bypass eliminated some HBM-bound traffic.
`SQ_INSTS_VMEM_*` and `WriteSize` dropped ~15 % each because the
scratch fill/spill instructions are gone. `VGPR_Count` did not move
(the dequant body's per-iteration accumulators
`acc01/acc23[kSegs=4]` still need 16 floats per lane), but
`numbered_sgpr` dropped from 112 to 100 — the args being directly
SGPR-resident from the kernel-arg buffer means fewer SGPRs are
reserved for the call setup.

### Non-profiler bench, sd=56 (128/16, 10 rounds)

| cell                       | post-Target-A | post-Target-B | delta             |
| ---                        | ---:          | ---:          | ---:              |
| sd=56 `normal`             | ~1139 us      | **~1091 us**  | **-48 us (-4.2 %)** |
| sd=56 `uniform[-1024,1024]`| ~1152 us      | **~1108 us**  | **-44 us (-3.8 %)** |

Both deltas are well outside the per-bench round-to-round spread
(3-7 us). Lined up against the prior baselines:

| baseline               | combine latency | gap to fp8bwq sd=56 + Target B |
| ---                    | ---:            | ---:                           |
| bf16 no-quant          | ~1044 us        | **+47 us (+4.5 %)**            |
| fp8_direct_cast        | ~611 us         | +480 us (+78.5 %)              |

`fp8bwq` is now within **48 us / 4.5 %** of the bf16 no-quant
combine baseline on the production sd=56 path. The remaining gap
to `fp8_direct_cast` is still real (480 us) and is now driven by
the legitimate blockwise per-block max-reduce + per-source scale
arithmetic, not the args-spill artifact.

### Decision

Kept. The change is one line (`__device__` →
`__device__ __forceinline__` on `EpCombineIntraNodeKernel_body`) in
`src/ops/dispatch_combine/intranode.hpp`. Risks accepted:

- **Code size**: the `fp8bwq` wrapper grew from 251 to 81 675
  disassembly lines. Counted across all
  `EpCombineIntraNodeKernel_*` instantiations the deployed
  `ep_intranode.hsaco` grows roughly proportionally; this has not
  caused any I-cache or load-time regression in measurement, but
  is worth tracking if more instantiations are added.
- **Effect on other instantiations**: the same `__forceinline__`
  also applies to `_p2p`, `_stdmoe`, etc. — those bodies were
  small enough to be inlined already, so the attribute is a no-op
  for them.

## Disasm + split profile after Target B (baseline check before next move)

After Target B landed, the user asked to verify what was actually binding
the kernel before picking the next code change. Two cheap signals were
collected; both made the next-step picture sharper.

### Re-disasm post-Target B

| metric                     | post-Target-A | post-Target-B |
| ---                        | ---:          | ---:          |
| `private_seg_size`         | 1136 B        | **192 B**     |
| `num_vgpr`                 | 128           | 128           |
| `numbered_sgpr`            | 112           | 100           |
| max VGPR index used        | —             | 127           |
| max SGPR index used        | —             | **99 (≈ ceiling 102)** |
| total scratch ops in body  | 696 + 48      | 134 + 13      |
| **`v_writelane_b32` count** | —            | **284**       |
| **`v_readlane_b32` count**  | —            | **923**       |
| dedicated SGPR-spill VGPRs | —             | v125 / v126 / v127 |

Big finding the disasm exposed: post-Target-B the kernel is no longer
register-spilling to scratch (192 B fits comfortably) — instead the
compiler hit the **SGPR ceiling** (s99 of ~s102) and now spills SGPR
into per-lane VGPR slots via `v_writelane_b32` / `v_readlane_b32`. Three
VGPRs (v125, v126, v127) are dedicated to this SGPR-spill area, which
is a major contributor to `num_vgpr` staying pinned at 128.

In other words: Target B traded a 1136 B scratch-via-L2 round-trip for
a much cheaper SGPR-via-VGPR-lane round-trip (4 cycles vs ~150 cycles
on a miss), which is why the 47 us bench improvement is real. But the
new binding constraint is SGPR pressure, not register-spill scratch.

### Split profile post-Target-B (sd=56, normal vs uniform-1024)

Per-warp work breakdown (work-normalized via per-warp mean × token-iters
where applicable):

| Slot                       | normal | uniform-1024 | share |
| ---                        | ---:   | ---:         | ---:  |
| `combine_stage_input`      | 504 us | 514 us       | **~49 %** |
| `combine_dequant_accum`    | 219 us × 2 token-iters = 438 us | 219 × 2 = 439 us | **~42 %** |
| `combine_barrier`          | 54 us  | 67 us        | ~5 %  |
| `combine_prepare_ptrs`     | 9 us × 2 = 18 us | 9 × 2 = 18 us | ~2 % |
| others (accum_setup, copy/accum_weights) | < 11 us | < 11 us | < 2 % |

Two takeaways:

1. After Target B, StageInput and DequantAccum are still ≈ tied
   (49 / 42), the same proportions the pre-Target-A baseline showed.
   Target B's win was distributed across both phases proportionally.
2. `combine_dequant_accum` is identical between `normal` and
   `uniform-1024` (219.2 vs 219.3 us). The actual scale-active path
   inside the dequant inner body costs essentially zero extra cycles —
   the per-source FMAs slot into otherwise-unused VALU pipeline slots.
   This is a useful "compute is not the bottleneck" datapoint and
   downgrades the expected win for accumulator-footprint shrink
   (Target C / VecBytes=8) from "big" to "modest".

### Implication for next code changes

Both signals point to **SGPR pressure** (and the resulting
write/readlane traffic) as the next concrete thing to attack.
Anything that reduces the number of SGPR-resident values live across
the dequant unrolled body is potentially valuable; anything that just
moves work between VALU pipelines (e.g. VecBytes=8 vs 16) probably
isn't, given the dequant compute is already not the bottleneck.

## Candidate "AllValid" trial: skip per-source nullptr checks (REVERTED)

### Hypothesis

The ~3 SGPR comparisons per source per outer iter
(`if (src == nullptr)`, `if (srcScales != nullptr)`,
`if (srcScales[i] != nullptr)`) generate live SGPR ranges that
contribute to the SGPR-pressure ceiling. With AccumNum = 8 unrolled,
that's up to 24 comparison values potentially live simultaneously. If
the kernel could prove (warp-uniformly, once) that all sources are
non-null, the inner loop could drop all of these checks and shrink the
SGPR live set, possibly relieving the writelane/readlane traffic.

### Change tried

- Added `bool AllValid = false` template parameter to
  `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`. When `true`, the
  inner loop skips the per-source nullptr checks and assumes every
  `srcs[i]` and `srcScales[i]` is non-null.
- Added a `bool allValid` runtime parameter to the
  `WarpAccumFp8DequantFullImpl` / `SegmentImpl` and front-end entries.
  At the top of each `vec{16,8,4}Possible` branch, dispatched to the
  `AllValid=true` template specialization when the runtime flag held,
  otherwise to the existing `false` specialization.
- In `intranode.hpp`, after the `PreparePtrs` loop, added a single
  warp-uniform `__ballot` over `srcPtrs[laneId] != nullptr` for
  `laneId < numExpertPerToken`. The `worldSize <= 4` compaction path
  produces `allValid = true` by construction; the `worldSize > 4`
  no-compaction path checks the ballot mask matches the expected mask.

### Result

3/3 sd=56 sanity cases pass (`normal`, `uniform-1024`,
`force_scale_active`). But every other signal regressed:

| metric                | post-Target-B | post-AllValid trial | delta              |
| ---                   | ---:          | ---:                | ---:               |
| `private_seg_size`    | 192 B         | **464 B**           | **+272 B (+142 %)** |
| `num_vgpr`            | 128           | 128                 | unchanged          |
| `numbered_sgpr`       | 100           | 100                 | unchanged          |
| `v_writelane_b32`     | 284           | 296                 | +4 %               |
| `v_readlane_b32`      | 923           | 1090                | +18 %              |
| total scratch ops     | ~147          | **~1602**           | **~+10×**          |
| disasm slice (lines)  | 81 675        | 101 559             | +24 %              |

| bench cell                 | post-Target-B | post-AllValid trial | delta              |
| ---                        | ---:          | ---:                | ---:               |
| sd=56 `normal`             | ~1091 us      | **~1129 us**        | **+38 us (+3.5 %)** |
| sd=56 `uniform[-1024,1024]`| ~1108 us      | **~1147 us**        | **+39 us (+3.5 %)** |

### Decision

Reverted. The change is restored to the post-Target-B state.

### Lesson

The `AllValid=true / false` runtime dispatch at the Impl level is the
wrong delivery mechanism. Even though each template specialization in
isolation could be tighter, putting both specializations into the
inlined-into-wrapper kernel forces the register allocator to handle
two parallel hot inner-loop bodies that compete for the same VGPR /
scratch budget. The compiler can't prove which of the two paths
dominates a given dynamic execution, so it has to satisfy the union of
their liveness — and the union is strictly worse than either path
alone. The 10× scratch-op explosion and the ~25 % code-size growth
are direct evidence of this.

If the AllValid hypothesis is worth re-trying, it has to be
"compile-time only" — e.g. a separate kernel symbol for the
all-valid path, dispatched on the host side based on a once-per-call
heuristic. That is a larger refactor (split the EP combine kernel
binary along an `AllValid` axis, similar to how `_p2p` / `_nop2p` /
`_fp8cast` / `_fp8bwq` are already separate symbols), and is now
worth considering only if other lower-risk SGPR-relief levers also
fail.

## Target M trial: marker-based no-scale bypass + inner-loop load reorder

### Change

Two coupled changes shipped together (commit `8abd69fa`):

- **Marker bypass (host side, in `intranode.hpp`):** when preparing
  `srcScalePtrs[j]` after `PreparePtrs`, read the staged
  `scalePtr[0]`. The blockwise quant kernel writes `scalePtr[0]` as
  the positive scale value when no block in the token was
  scale-active, and as the *negative* scale value when at least one
  block tripped FP8 max. So `(scalePtr[0] < 0.0f) ? scalePtr : nullptr`
  is exactly the per-token "any scale-active" marker. When the marker
  says no, the dequant side gets `nullptr` for that source's scale
  pointer.
- **Inner-loop load reorder + branched accumulate (in
  `WarpAccumFp8DequantVecRangeBlockwiseScaleWave`):** issue the
  `load<VecBytes>(srcB+idx)` *before* the per-block `scalePtr[sb]`
  read, and dispatch the per-`seg` accumulate to the
  `AccumFp8Packed4<Fp8T, true>` (FMA) template if `scalePtr != nullptr`
  or to `AccumFp8Packed4<Fp8T, false>` (plain ADD) otherwise.

Correctness invariant the bypass relies on: the quant kernel writes
`dstScales[sb] = sbScaled ? (maxAbs * invFp8Max) : 1.0f` for *every*
`sb in [0, scaleDim)`, and only flips `dstScales[0]` to negative when
`anyScaled` across the token is true. So even if the bypass mis-routed
to the unscaled path (it cannot, but hypothetically), the data would
still be self-consistent because the "skipped" multiplies would have
been `* 1.0f` no-ops.

### Non-profiler bench (sd=56, 128/16, 10 rounds, same env)

| Cell                              | post-Target-B | post-Target-M  | delta              |
| ---                               | ---:          | ---:           | ---:               |
| sd=56 `normal`                    | ~1092 us      | **~1047 us**   | **-45 us (-4.1%)** |
| sd=56 `uniform[-1024,1024]`       | ~1108 us      | **~1071 us**   | **-37 us (-3.3%)** |
| sd=56 `normal+force_scale_active` | (not run)     | ~1069 us       | (correctness pass) |

`force_scale_active` writes a sentinel `2 * fp8Max = 896` into the
first lane of every block on the host side, so `--report-scale-stats 1`
shows `any_scaled=1.0000`, `block_scaled=1.0000`, and the kernel
exercises the fully-scaled dequant path. The fact that this case lands
at ~1069 us (within bench noise of the `uniform-1024` ~1071 us number)
confirms the bypass does *not* mis-route on scale-active inputs.

For reference on the same env at the same launch config (re-baselined
in this round so the comparison is on a single calibrated env):

| Variant                              | combine latency | vs current best |
| ---                                  | ---:            | ---:            |
| bf16 no-quant                        | ~1039 us        | -0.8 % (faster) |
| fp8_blockwise sd=56 + Target M (this) `normal` | ~1047 us | baseline |
| fp8_blockwise sd=56 + Target M (this) `uniform-1024` | ~1071 us | +2.3 % slower |
| fp8_direct_cast                      | ~607 us         | -42.0 % (faster) |

So on the production sd=56 path with random-normal input,
**fp8_blockwise is now within 0.8 % of bf16 no-quant** — i.e. the
"compress to half the bytes + per-block scales" cost is essentially
fully amortised on this benchmark.

### Why uniform-1024 also moved (it shouldn't, naively)

The marker bypass cannot fire on `uniform-1024`: every token has
`max|x| = 1024 > fp8Max = 448`, so every `srcScalePtrs[j]` ends up
non-null and every source still goes through the FMA template. Yet
combine dropped 37 us / 3.3 % on this case as well. The credit goes
to the second half of the change: hoisting the `srcScales[i]` lookup
into a `const float* scalePtr` and issuing the FP8 vector load before
the scale lookup lets the scheduler overlap two independent memory
ops. This is the kind of win the prior Candidate 3 trial speculated
about but couldn't trigger by just touching the `cachedSrcs` array.

### `WarpAccumFp8DequantFullImpl` `anyScale` correctness fix

The `anyScale` reduction in both `WarpAccumFp8DequantFullImpl` and
`WarpAccumFp8DequantSegmentImpl` previously OR'd `srcScales[i] !=
nullptr` over all `i in [0, AccumNum)`. After the host-side bypass
sets `srcScalePtrs[j] = nullptr` for no-scale tokens but leaves
`srcs[j]` non-null for invalid sources (compaction reorders only on
worldSize <= 4), this could mis-classify a token as scale-active even
when every source's scale was bypassed. The fix is one extra null check
per source: `anyScale |= (cachedSrcs[i] != nullptr && srcScales[i] !=
nullptr)`. This brings the all-no-scale token into the
`if (!anyScale)` super-fast path (the same `WarpAccumFp8DequantVecRange<...,
false>` path `fp8_direct_cast` uses for its dequant), which is where
the headline 45 us normal-distribution win comes from.

### Decision

Kept and committed as `8abd69fa`. No regression on any tested cell;
clear win on both distributions. Bench noise for these cells is
~3-7 us, so both 45 us and 37 us deltas are well outside noise.

### What this means for the next move

The remaining gap to bf16 no-quant on the random-normal benchmark is
now ~8 us (~0.8 %). For all practical purposes blockwise on this
distribution is parity with bf16. The remaining gap to
`fp8_direct_cast` is still ~440 us / ~42 % — that gap is structural,
driven by the kernel-wide ceiling described in the Target B section
(args spill ate down from 1136 B to 192 B, but the body is still too
large to inline into the wrapper, hence forced inline + SGPR-spill via
v_writelane traffic). Closing the rest of the gap to direct_cast on
the same benchmark requires a structural change, not another inner
loop micro-optimization. The next concrete proposal on the table was
to split the kernel body so quant runs in its own kernel symbol — see
the next section for the PoC verification + full implementation
results (which turned out to be a no-op in bench despite the static
analysis predicting a meaningful win).

## Split-quant kernel trial: PoC + full implementation (REVERTED)

### Hypothesis

After Target B and Target M, the merged blockwise kernel is at
~1046 us combine on sd=56 normal (~1071 us on uniform-1024). The
remaining static-analysis pathology is `num_vgpr=128` (AMD ceiling)
and `private_seg_size=192 B` (residual args spill after Target B).
The thinking: if we split the merged kernel into two `__global__`
symbols — one running stage_input/quant + copy_weights, the other
running cross-device barrier + dequant + accumulate — each smaller
body might be small enough that the compiler natural-inlines it into
its `__global__` wrapper *without* needing `__forceinline__`. That
would cut the args spill the way Target B did for the merged path,
plus potentially relieve the SGPR ceiling (which currently spills
into `v_writelane` slots v125–v127).

### PoC: probe a quant-only kernel symbol

Setup: added a probe-only `EpCombineQuantStageOnlyKernel_bf16_fp8bwq_poc`
symbol that contains *only* the stage_input loop for the bf16 nop2p
blockwise path (no UseP2PRead variants, no stdmoe, no merged-kernel
infrastructure), with the body declared plain `__device__` (not
`__forceinline__`). Built it alongside the production kernels in the
same hsaco, then disassembled.

Result (single-pass, gfx942):

| Kernel                                                | private_seg_size | num_vgpr | numbered_sgpr | body lines | scratch ops | v_writelane/readlane | body symbol independent? |
| ---                                                   | ---:             | ---:     | ---:          | ---:       | ---:        | ---:                 | ---                      |
| `fp8cast` (reference, natural-inline)                 | 64 B             | 98       | 100           | 6,978      | 0           | 0                    | no (inlined)             |
| `fp8bwq` (current, `__forceinline__`)                 | 208 B (after TM) | **128**  | 100           | 84,840     | 259         | 1,259                | no (forced inline)       |
| **`poc` (quant-only, no `__forceinline__`)**          | **80 B**         | **80**   | 100           | **6,021**  | **4**       | **85**               | **no (natural-inline)**  |

Strong positive signal across every metric. `private_seg_size` 80 B
is only 16 B more than `fp8cast`'s 64 B; `num_vgpr=80` is well below
the 128-VGPR ceiling that produced the writelane spill; the body
fully inlined into the wrapper without the `__forceinline__` hint;
the `v_writelane`+`v_readlane` count dropped from 1,259 to 85 (-93 %).

This was treated as a **GO signal** for a full split implementation,
with predicted bench wins in the 175–385 us range (rough split:
−15 % conservative / −23 % central / −32 % optimistic, with the
optimistic case approaching `fp8_direct_cast`'s 607 us combine).

### Full implementation

Implementation across 5 files (~120 LOC):

- `src/ops/dispatch_combine/intranode.hpp`:
  - new `EpCombineQuantStageKernel_body<T, UseP2PRead, EnableStdMoE,
    UseFp8DirectCast, UseFp8BlockwiseQuant>` template (plain
    `__device__`, no `__forceinline__`) holding the stage_input +
    optional copy_weights logic that the merged body's Phase 1 used
    to run, with all four bool template params preserved so the
    quant body is structurally identical to the merged body's
    Phase 1
  - new template parameter `bool QuantAlreadyDone = false` on
    `EpCombineIntraNodeKernel_body<...>` (default false, backward
    compatible). Phase 1 wrapped in `if constexpr (!QuantAlreadyDone)
    { ... }` so the recv kernel skips it
  - new `__global__ EpCombineQuantStageKernel<T, ...>` wrapper
- `src/ops/kernels/ep_common.hip`: `WRAP_BOOL5` macro for the
  5-bool-param recv variant
- `src/ops/kernels/ep_intranode.hip`: two new symbols
  - `EpCombineQuantStageKernel_bf16_fp8bwq` (quant-only)
  - `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_recv`
    (`QuantAlreadyDone=true`, barrier + dequant + accumulate only)
  - kept the legacy `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` for
    backward compat / fallback
- `src/ops/dispatch_combine/launch.cpp`: blockwise IntraNode launches
  the two split symbols back-to-back on the same stream
  (`shared_mem=0` for quant_stage, `shared_mem=combine_shared_mem`
  for recv)
- `python/mori/ops/dispatch_combine.py`: `combine()` blockwise branch
  uses `_launch_multi` to capture both kernels into a single CUDA
  graph node pair

Correctness: 3/3 sd=56 sanity cases pass (`normal`, `uniform-1024`,
`force_scale_active`). `--report-scale-stats 1` confirms
`any_scaled=0.0000` on `normal` and `1.0000` on the two scale-active
cases, so the split correctly preserves the marker semantics and the
recv kernel routes scaled vs unscaled paths the same as the merged
kernel did.

### Static analysis: only quant kernel benefits, recv doesn't

| Kernel                                          | private_seg_size | num_vgpr   | body lines | scratch ops | v_writelane/readlane |
| ---                                             | ---:             | ---:       | ---:       | ---:        | ---:                 |
| `fp8cast` (reference)                           | 64 B             | 98         | 6,978      | 0           | 0                    |
| `fp8bwq` (legacy merged, kept as fallback)      | 208 B            | **128**    | 84,840     | 259         | 1,259                |
| **`QuantStageKernel_bf16_fp8bwq`** (new)        | **80 B**         | **83**     | 6,162      | **4**       | **119**              |
| **`IntraNodeKernel_bf16_nop2p_fp8bwq_recv`** (new) | **144 B**     | **128** ⚠️ | 78,670     | 198         | 1,086                |

The quant kernel reproduces the PoC numbers almost exactly (80 B
private_seg, 83 VGPR, 6.2 K lines, ~120 writelane). The recv kernel
sits between the merged kernel and `fp8cast`: `private_seg_size`
dropped from 208 B to 144 B (-31 %), but `num_vgpr` is still pinned
at the AMD ceiling of 128 and `v_writelane` is still 1,086 (-14 %
vs merged). Body size is 78,670 lines, almost as large as the merged
kernel's 84,840 lines, confirming that the Phase 1 stage_input
section was only ~7 % of the merged body — splitting it out doesn't
materially shrink what was already the dominant chunk.

### Bench: zero net improvement

sd=56, 128/16, 10 rounds × 3 runs:

| Cell                                          | merged (Target M)  | split (this trial)              | delta             |
| ---                                           | ---:               | ---:                            | ---:              |
| sd=56 `normal`                                | ~1047 us           | **~1049 us** (1048.3 / 1049.0 / 1050.5 over three 10-round runs) | +2 us (noise)    |
| sd=56 `uniform[-1024,1024]`                   | ~1071 us           | **~1070 us**                    | -1 us (noise)    |
| sd=56 `normal+force_scale_active`             | ~1069 us           | ~1071 us                        | +2 us (noise)    |

All three deltas are inside the per-bench round-to-round spread (~3-7 us)
and the run-to-run variance for the split path itself is ~2 us.

### Why the prediction was wrong

The PoC measured the *quant kernel in isolation* and confirmed it can
escape the args spill via natural inline. It did **not** verify two
things that turned out to be the actual blockers:

1. **The dequant body has its own structural register pressure that
   args spill cannot explain.** The recv kernel's body is still
   78,670 lines — only ~7 % smaller than the merged body. The bulk
   of that comes from `WarpAccumFp8DequantFullImpl<bf16, fp8, 8>`
   inlined by `__forceinline__`, which carries `acc01[kSegs=4] +
   acc23[kSegs=4] + cachedSrcs[AccumNum=8] + sbScales[AccumNum=8]`
   per-iteration arrays. Those arrays drive `num_vgpr=128`
   independently of how much args spill the kernel has. Removing
   args spill (208 B → 144 B) helped a little but couldn't unstick
   the VGPR ceiling.

2. **The quant phase appears to be HBM-bandwidth-bound on the bf16
   reads, not args-spill-bound.** The merged kernel's Phase 1 reads
   `hidden_dim × sizeof(bf16) × num_recv_tokens` = 7168 × 2 × ~2k =
   ~28 MB per rank from HBM. After Target B the merged kernel's
   args spill was already down to 192 B / thread. The further
   reduction to 80 B / thread in the split kernel saves more
   instruction stream for args reads, but those instruction-cycle
   savings don't materialise as wall-clock time because the
   critical path is the bf16 HBM read, not the args ALU. Result:
   the quant phase runs in roughly the same wall time in both
   merged and split.

Together these mean the split saves a few cycles in the quant
kernel's instruction stream, but those cycles aren't on the
critical path. The recv kernel keeps the same critical path as the
merged dequant phase. Net savings: within bench noise.

The PoC was therefore *correct on its own narrow claim* (the
quant-only body inlines naturally) but *insufficient to predict the
bench outcome*, because it didn't account for (a) the dequant body's
own register pressure and (b) the actual bottleneck of the quant
phase.

### Decision

**Reverted.** The 5-file change was rolled back via `git checkout
HEAD --` on those files. Working tree is clean against the
post-Target-M commit. The legacy
`EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` is the only blockwise
kernel symbol again, and `LaunchCombine` / `combine()` go back to a
single-kernel launch.

### Lesson for future split-style attempts

Before re-trying any combine-kernel split, the new bottleneck must be
demonstrated to live in args spill or in the now-isolated kernel's
critical path, **not** in the dequant arithmetic itself. Two
prerequisites are worth checking in an updated PoC:

- Build a *recv-only* probe symbol (mirror of this round's quant-only
  PoC) and measure its `private_seg_size`, `num_vgpr`, body lines,
  scratch ops, and writelane count. If the recv-only PoC still
  comes out at ~78 K lines / 128 VGPR / ~1 K writelane (as the recv
  kernel did in this implementation), splitting alone won't move
  the needle and the next move has to be inside the dequant impl.
- Profile the merged kernel's Phase 1 alone to confirm whether it's
  ALU-bound or HBM-bound. If it's HBM-bound, args spill reduction
  via split is purely a code-cleanliness change, not a perf change.

If both prerequisites pass, the most likely structural follow-up is
a *narrower args struct* for the recv kernel (a `EpCombineRecvArgs`
sub-struct that omits the dispatch-only fields). This attacks the
remaining args spill without depending on the inline heuristic.
Caveat: this is a wider refactor than this round's split attempt
(touches the args type, all launch sites, and all kernels that
currently consume the merged struct).

### What was kept from this round

The PoC verified that a small `__device__` body with the bf16 nop2p
blockwise quant payload natural-inlines into its `__global__` wrapper
on gfx942 (at least with the current ROCm 7.2 / clang). This is
useful background for any future kernel-shape changes — the inline
heuristic on gfx942 IS willing to inline bodies in the ~6 K-line
range, which gives a rough budget for what "structurally small
enough to escape args spill" means on this target. None of the source
code is kept; only this writeup.

## How to reproduce

## Final specialization kept in this change: block128/top8/vec8 no-weight

After the split-quant trial was reverted, the next successful direction was
not another kernel split, but a **narrow compile-time specialization of the
dequant path** for the production blockwise layout:

```text
hidden_dim = 7168
scale_dim = 56
block_elems = 128
num_experts_per_token = 8
weights = None
world_size > 4
```

The specialization is exposed as:

```text
EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_vec8
```

and is selected by both Python and C++ launch paths only when the exact
conditions above hold. All other shapes, weighted combines, EP4/smaller
world sizes, and non-`scale_dim=56` cases continue to use the generic
`EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` fallback.

### Why this shape is special

With `hidden_dim=7168` and `scale_dim=56`, each activation scale block has
exactly 128 elements. The specialized dequant path uses `VecBytes=8`
instead of the generic vectorized `VecBytes=16` path. That halves the
per-lane accumulator footprint (`kSegs=2` instead of `kSegs=4`) for the
common `AccumNum=8` case. Unlike previous runtime-dispatch experiments, this
is a separate compile-time kernel instantiation, so the compiler does not
have to allocate registers for both hot paths in the same symbol.

The helper names are:

```text
WarpAccumFp8DequantFullBlock128Vec8Top8
WarpAccumFp8DequantSegmentBlock128Vec8Top8
```

Both the scaled and no-scale branches now use `VecBytes=8`. The no-scale
branch originally kept the old `VecBytes=16` unscaled helper, but that left
the normal/no-scale benchmark on the same register-pressure cliff.

### Static metadata after specialization

Single-pass disassembly metadata for the final non-profiler build:

| Kernel | private segment | VGPR | numbered SGPR |
| --- | ---: | ---: | ---: |
| generic `fp8bwq` | `0xD0 = 208 B` | `0x80 = 128` | `0x64 = 100` |
| `fp8bwq_noweight_vec8` | `0x90 = 144 B` | `0x5F = 95` | `0x64 = 100` |

This is the first successful experiment that moved the specialized blockwise
combine path materially below the 128-VGPR cliff.

### Non-profiler benchmark, final code

EP8, bf16, `max_tokens=4096`, `hidden_dim=7168`, `scale_dim=56`,
`zero-copy=0`, `dispatch=128/16`, `combine=128/16`.

| Case | Before specialization | Final specialization | delta |
| --- | ---: | ---: | ---: |
| `normal` no-scale | `~1040 us` | `~834 us` | `~ -206 us (-20%)` |
| `uniform[-1024,1024]` scale-active | `~1064 us` | `~862 us` | `~ -202 us (-19%)` |
| `fp8_direct_cast` reference | `~606 us` | unchanged | gap remains |

The important change from earlier trials is that both no-scale and
scale-active inputs now benefit. The previous scaled-only `VecBytes=8`
experiment improved `uniform[-1024,1024]` but left `normal` near `~1038 us`;
switching the no-scale branch to `VecBytes=8` brought `normal` down to
`~834 us`.

### Profiler slot breakdown, final no-scale path

Profiler build, `normal`, same launch config. Work-normalized numbers are
`total slot duration / stage warps` with 2048 stage warps per rank iteration.

| Slot | Before no-scale vec8 | Final no-scale vec8 |
| --- | ---: | ---: |
| `combine_stage_input` | `498.14 us` | `495.37 us` |
| `combine_barrier` | `78.70 us` | `64.88 us` |
| `combine_accum_setup` | `1.07 us` | `1.00 us` |
| `combine_prepare_ptrs` | `22.81 us` | `8.64 us` |
| `combine_dequant_accum` | `394.26 us` | `224.04 us` |
| Total | `994.99 us` | `793.93 us` |

Envelope view:

| Metric | Before no-scale vec8 | Final no-scale vec8 |
| --- | ---: | ---: |
| combine envelope | `1047.54 us` | `827.74 us` |
| accum envelope | `470.08 us` | `266.98 us` |

The final profile confirms that the improvement is primarily a dequant
register-pressure/codegen win. `CombineStageInput` is now the largest
remaining slot for both no-scale and scale-active cases.

### Decision

Keep the specialization. It is deliberately narrow, selected only by exact
shape and `weights=None`, so it does not change generic correctness or
weighted combine behavior. The next performance target for this production
shape is no longer dequant; it is `CombineStageInput`, which includes bf16
input reads, blockwise max/reduce, fp8 writes, scale writes, and staging.

```bash
# Drives 9 hardware-pass groups for one variant. Output goes to
# /tmp/rocprof-fp8bwq/<tag>_<dist>_<scale>[_sd<scale_dim>]/g??/...
bash /tmp/rocprof_run.sh fp8_blockwise   normal  1.0
bash /tmp/rocprof_run.sh fp8_direct_cast normal  1.0
bash /tmp/rocprof_run.sh none            normal  1.0
bash /tmp/rocprof_run.sh fp8_blockwise   uniform 1024.0
bash /tmp/rocprof_run.sh fp8_blockwise   normal  1.0   56   # 4th arg = scale_dim

# Aggregate across PMC groups + ranks into a comparison table.
python /tmp/rocprof_analyze.py \
  /tmp/rocprof-fp8bwq/none_normal_1.0 \
  /tmp/rocprof-fp8bwq/fp8cast_normal_1.0 \
  /tmp/rocprof-fp8bwq/fp8bwq_normal_1.0
python /tmp/rocprof_analyze.py /tmp/rocprof-fp8bwq/fp8bwq_uniform_1024.0
python /tmp/rocprof_analyze.py /tmp/rocprof-fp8bwq/fp8bwq_normal_1.0_sd56
```

Both helpers live under `/tmp/` deliberately and are not committed; the
`bench_dispatch_combine.py --input-dist ... --scale-dim ... --report-scale-stats 1`
plumbing they rely on lives in this branch.

## Caveats

- Numbers are averaged across (rank × dispatch) inside the same launch
  config. The per-kernel duration column carries cold-start outliers
  (max in the seconds), which we did not try to suppress beyond
  dropping the first dispatch per process per counter; for kernel
  duration the relevant figure is the median (~1.5-3 ms).
- PMC collection serializes kernels per CU. Counter values reflect a
  single steady-state kernel invocation, but absolute durations under
  PMC should not be quoted as production latency.
- We did not capture `LDSBankConflict` because it is gfx942-supported
  but produced zero in our smoke run; if Target A is taken, re-enable
  it to confirm the LDS traffic is in the address-coalesced regime
  (i.e., scratch traffic pattern, not a bank-conflict regression).
