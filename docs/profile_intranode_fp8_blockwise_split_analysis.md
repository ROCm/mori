# Intranode FP8 Blockwise Split Profile Analysis

## Summary

This document records the current profiling and benchmark status for the
intranode `fp8_blockwise` combine path after adding finer profiler slots and
testing the first dequant cleanup.

Current test scope:

- EP8, `bf16`, `max_tokens=4096`, `hidden_dim=7168`, `zero-copy=0`
- default benchmark input distribution from the original
  `bench_dispatch_combine.py`
- default launch config for the profile run:
  `combine_block_num=80`, `combine_warp_per_block=16`
- one additional non-profiler launch comparison with
  `combine_block_num=128`, `combine_warp_per_block=16`
- bf16 no-quant baseline runs for both `80/16` and `128/16`

The current conclusion is:

1. The split profiler slots are useful and should be kept.
2. The hot work is still split between input-side blockwise quant/staging and
   output-side fp8 dequant/accumulate.
3. The attempted dequant scale-hoisting / hot-path `fabsf` removal did not
   produce a clear latency win on the original random-normal benchmark.
4. Launch config `128/16` is still meaningfully faster than the original
   default `80/16`, but `bench_dispatch_combine.py` should remain unchanged for
   now.
5. At `128/16`, `fp8_blockwise` is still slower than the bf16 no-quant combine
   baseline, so current blockwise quantization has not yet paid back its extra
   scale/reduce/dequant cost.

## Code Under Test

### Profiler Split

File:

- `src/ops/dispatch_combine/intranode.hpp`

The previous coarse combine profiler slots were split as follows:

| Old slot | New slots |
| --- | --- |
| `CombineCopyInput` | `CombineStageInput`, `CombineCopyWeights` |
| `CombineAccum` | `CombineAccumSetup`, `CombinePreparePtrs`, `CombineDequantAccum`, `CombineAccumWeights` |

For the nop2p path used by `--zero-copy 0`, token staging and optional weight
copy are split only under `ENABLE_PROFILER`. The non-profiler build keeps the
original single-loop structure.

### Dequant Cleanup Attempt

File:

- `include/mori/core/transport/p2p/device_primitives.hpp`

The dequant changes under test were:

- vectorized full and segment blockwise dequant paths now route through the
  blockwise helper that iterates by scale block;
- scale loads are hoisted per scale block inside that helper;
- `fabsf(srcScales[i][sb])` is removed from the repeated vector inner path;
- only `sb == 0` handles the negative token marker by converting the loaded
  scale to a positive value.

The goal was to reduce three costs in the hot path for the common shape:

```text
hiddenDim = 7168
scaleDim  = 32
blockElems = 224
```

Because `224` is not a power of two, the previous vector-range helper had to
compute `globalIdx / blockElems` on the repeated vector path. The new structure
avoids that division and reuses scale registers within a scale block.

## Profile Command

Profiler build:

```bash
ENABLE_PROFILER=ON /home/nima/build.sh
```

Profile run with the original benchmark parameters:

```bash
rm -rf /tmp/mori-prof-fp8bwq-opt
mkdir -p /tmp/mori-prof-fp8bwq-opt
cd /tmp/mori-prof-fp8bwq-opt

ENABLE_PROFILER=ON \
MORI_EP_LAUNCH_CONFIG_MODE=MANUAL \
PYTHONPATH=/home/nima/mori:$PYTHONPATH \
python /home/nima/mori/tests/python/ops/bench_dispatch_combine.py \
  --cmd profile \
  --dtype bf16 \
  --quant-type fp8_blockwise \
  --max-tokens 4096 \
  --hidden-dim 7168 \
  --zero-copy 0 \
  --world-size 8
```

Trace output:

```text
/tmp/mori-prof-fp8bwq-opt/trace_intranode_rank*_0507_024642.json
```

Each rank produced:

- 3 profiled iterations;
- 1280 active combine warps;
- 127488 trace events.

After profiling, the environment was rebuilt without profiler:

```bash
/home/nima/build.sh
```

## How To Read The Numbers

The profiler timestamps are useful for attribution, not as the final production
latency source of truth. Use the non-profiler benchmark for production latency.

There are two different views below:

- **Envelope / wall-clock view**: earliest slot start to latest slot end for a
  phase. This is closest to what the benchmark observes.
- **Work-normalized view**: total slot duration divided by the active combine
  warp count. This is better for comparing how much work each slot contributes.

Individual slot wall-clock unions can overlap because different warps enter the
next profiler slot while slower warps are still in the previous slot. This is
especially visible for `CombineCopyWeights` and `CombineAccumWeights` when
`weights=None`.

## Profile Results

Average over 8 ranks x 3 iterations.

### Envelope View

| Metric | Average | Min | Max |
| --- | ---: | ---: | ---: |
| Combine envelope | `1366.78 us` | `1357.32 us` | `1380.00 us` |
| Accum envelope | `713.33 us` | `706.58 us` | `724.24 us` |

The combine envelope matches the non-profiler default benchmark result closely
enough to be useful for phase attribution.

### Slot Wall-Clock Union

| Slot | Average | Min | Max |
| --- | ---: | ---: | ---: |
| `combine_stage_input` | `641.40 us` | `633.94 us` | `646.73 us` |
| `combine_barrier` | `123.41 us` | `115.21 us` | `130.02 us` |
| `combine_accum_setup` | `12.19 us` | `11.67 us` | `12.76 us` |
| `combine_prepare_ptrs` | `469.59 us` | `438.95 us` | `496.79 us` |
| `combine_dequant_accum` | `701.36 us` | `693.46 us` | `711.01 us` |

`combine_copy_weights` and `combine_accum_weights` are intentionally omitted
from this wall-clock table. In the current benchmark `weights=None`, so their
wall-clock unions mostly capture warp skew and sequential slot gaps rather than
real weight work.

### Work-Normalized View

Total slot work divided by 1280 active combine warps:

| Slot | Average | Share |
| --- | ---: | ---: |
| `combine_stage_input` | `579.79 us` | `47.32%` |
| `combine_dequant_accum` | `526.43 us` | `42.96%` |
| `combine_barrier` | `55.55 us` | `4.53%` |
| `combine_prepare_ptrs` | `37.99 us` | `3.10%` |
| `combine_accum_weights` | `16.10 us` | `1.31%` |
| `combine_accum_setup` | `7.07 us` | `0.58%` |
| `combine_copy_weights` | `2.40 us` | `0.20%` |
| Total | `1225.34 us` | `100.00%` |

This view gives the cleanest current attribution:

1. `CombineStageInput` is the largest component.
2. `CombineDequantAccum` is very close behind.
3. Barrier and pointer setup are visible but secondary.
4. Weight slots are not a real workload in this benchmark.

## Non-Profiler Benchmark Results

All runs used original benchmark input generation. `bench_dispatch_combine.py`
was not modified.

| Build / code | Quant type | Combine launch | Best combine avg latency |
| --- | --- | --- | ---: |
| current code | `none` / bf16 | `80/16` | `~1127.7 us` |
| current code | `none` / bf16 | `128/16` | `~1044.5 us` |
| current code | `fp8_blockwise` | `80/16` | `~1366.3 us` |
| current code | `fp8_blockwise` | `128/16` | `~1239.6 us` |
| temporary HEAD baseline | `fp8_blockwise` | `128/16` | `~1242.0 us` |
| current code | `fp8_direct_cast` | `128/16` | `~611.5 us` |

BF16 no-quant baseline details:

| Combine launch | Best dispatch avg | Best combine avg | Best E2E avg |
| --- | ---: | ---: | ---: |
| `80/16` | `~1013.5 us` | `~1127.7 us` | `~2121.9 us` |
| `128/16` | `~1013.1 us` | `~1044.5 us` | `~2031.5 us` |

Interpretation:

- Current bf16 no-quant combine improves from about `1127.7 us` at `80/16` to
  about `1044.5 us` at `128/16`, a `~7.4%` launch-config gain.
- The current dequant cleanup is effectively noise-level versus the temporary
  HEAD baseline at `128/16`: about `1239.6 us` versus `1242.0 us`.
- It does not show a meaningful regression, but it also does not prove the
  expected high-impact win.
- `128/16` is still much better than the default `80/16` for this shape:
  about `9.3%` faster in the current non-profiler benchmark.
- At `128/16`, `fp8_blockwise` is about `18.7%` slower than the bf16 no-quant
  combine baseline: `1239.6 us` versus `1044.5 us`.
- `fp8_blockwise` remains roughly 2x slower than `fp8_direct_cast` at `128/16`.

The bf16 baseline commands were:

```bash
MORI_EP_LAUNCH_CONFIG_MODE=MANUAL \
PYTHONPATH=/home/nima/mori:$PYTHONPATH \
python /home/nima/mori/tests/python/ops/bench_dispatch_combine.py \
  --cmd bench \
  --dtype bf16 \
  --quant-type none \
  --max-tokens 4096 \
  --hidden-dim 7168 \
  --zero-copy 0 \
  --world-size 8

MORI_EP_LAUNCH_CONFIG_MODE=MANUAL \
PYTHONPATH=/home/nima/mori:$PYTHONPATH \
python /home/nima/mori/tests/python/ops/bench_dispatch_combine.py \
  --cmd bench \
  --dtype bf16 \
  --quant-type none \
  --max-tokens 4096 \
  --hidden-dim 7168 \
  --zero-copy 0 \
  --world-size 8 \
  --combine-block-num 128 \
  --combine-warp-per-block 16
```

## Why The Dequant Cleanup Did Not Clearly Move Latency

The optimization removed plausible instruction-level overhead:

- repeated non-power-of-two division by `blockElems=224`;
- repeated scale loads inside the vector loop;
- repeated `fabsf` on scale values.

The measured result suggests those savings were offset by other costs in the
new traversal:

- subwarp block traversal may reduce lane utilization compared with the
  previous full-wave vector-range helper;
- extra control flow for block-local traversal can hide the saved division;
- `AccumNum=8` keeps register pressure high because each source has its own
  scale and vector contribution;
- the original random-normal benchmark may not stress the same scale-active
  behavior as real activation traces or forced scale-active tests.

This means the idea is still plausible, but the broad helper reroute is not the
right final form unless later data shows a benefit on scale-active inputs.

## Current Bottleneck Read

For the current original benchmark:

- `CombineStageInput` is still the first target. It includes bf16 input reads,
  blockwise max/reduce, fp8 writes, scale writes, and staging into the combine
  buffer.
- `CombineDequantAccum` is the second target and is close enough to
  `CombineStageInput` that both sides matter.
- The bf16 no-quant `128/16` result gives a practical near-term target:
  `fp8_blockwise` needs to close roughly `195 us` of combine latency before it
  beats the current bf16 combine baseline on this benchmark.
- The barrier is not dominant, but its cost is large enough that load balance
  and launch config can move end-to-end latency.
- Weight handling cannot be judged from this benchmark because combine weights
  are `None`.

## Recommended Next Steps

### 1. Keep The Profiler Split

The new slots give useful attribution without changing the non-profiler hot
path. They should stay.

### 2. Do Not Change `bench_dispatch_combine.py` Yet

The current baseline should continue to use the original benchmark parameters.
Scale-active input controls are still useful later, but they should not be
mixed into this immediate baseline comparison.

### 3. Replace The Broad Dequant Reroute With A Narrower Experiment

The next dequant experiment should specialize the common case directly:

```text
blockElems == 224
VecBytes == 16
AccumNum == 8
hiddenDim == 7168
scaleDim == 32
```

The goal is to preserve the full-wave utilization pattern of the old
`WarpAccumFp8DequantVecRangeBlockwiseScaleWave` path while still avoiding the
hot `idx / 224` division. A likely approach is to track the scale block and
block-local vector range explicitly, instead of switching the whole full path
to a subwarp-per-block traversal.

Acceptance criterion:

- compare against the temporary HEAD baseline and current code at `128/16`;
- require a clear non-profiler improvement, not only profiler slot movement;
- keep correctness coverage for full and segment paths.

### 4. Move Quant-Side Analysis Up In Priority

The work-normalized profile says `CombineStageInput` is about `47%` of combine
work. The next useful profiling step is rocprof on:

- `EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq`
- `EpCombineIntraNodeKernel_bf16_nop2p_fp8cast`

The decision point is whether `CombineStageInput` is mainly:

- compute-bound from max reduce / scale computation;
- memory-bound from bf16 reads, fp8 writes, and scale writes;
- occupancy-limited by VGPR pressure.

For the current shape, also test whether a dedicated two-iteration quant path
for `blockElems=224` helps, since the main quant mapping has only a small fixed
iteration count.

### 5. Keep Launch Config Tuning As A Separate Track

`128/16` is currently much faster than `80/16`, but changing defaults or
benchmark parameters should be treated separately from the kernel optimization
itself. The useful sweep is still:

```text
combine_block_num: 40, 64, 80, 128, 160, 256
combine_warp_per_block: 4, 6, 8, 12, 16
```

The main thing to watch is VGPR pressure and occupancy, especially after any
`AccumNum=8` specialization.

## Caveats

- The profile run uses the original random-normal benchmark input, not a forced
  scale-active distribution.
- Profiler-build timings should not be used as production latency numbers.
- The benchmark currently passes `weights=None`, so weight slots are not
  representative of a weighted combine workload.
- The dequant cleanup may still help on different distributions, but the
  current original benchmark does not show a clear win.

## Validation

Commands run successfully during this round:

- the two bf16 baseline benchmark commands listed above;

```bash
git diff --check

rm -rf /tmp/mori-prof-gen
mkdir -p /tmp/mori-prof-gen/include /tmp/mori-prof-gen/src
python tools/profiler/generate_profiler_bindings.py \
  /home/nima/mori \
  /home/nima/mori/src \
  /tmp/mori-prof-gen/include/mori/profiler \
  /tmp/mori-prof-gen/src/profiler_bindings_generated.cpp

/home/nima/build.sh
ENABLE_PROFILER=ON /home/nima/build.sh
/home/nima/build.sh
```

## Update: final block128/top8/vec8 specialization

Later experiments changed the immediate conclusion of this split profile. The
broad dequant reroute still was not useful, but a narrow compile-time
specialization for the production shape did move latency substantially:

```text
hidden_dim = 7168
scale_dim = 56
block_elems = 128
num_experts_per_token = 8
weights = None
world_size > 4
```

The new symbol is:

```text
EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_vec8
```

It is selected only for the exact conditions above. The generic
`EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq` path remains the fallback for
weighted combine, EP4/smaller world sizes, different `scale_dim`, different
hidden dimensions, and other top-k values.

### Final non-profiler benchmark

EP8, bf16, `max_tokens=4096`, `hidden_dim=7168`, `scale_dim=56`,
`zero-copy=0`, `dispatch=128/16`, `combine=128/16`.

| Case | before specialization | final specialization | reading |
| --- | ---: | ---: | --- |
| `normal` no-scale | `~1040 us` | `~834 us` | no-scale dequant was still on the register-pressure cliff |
| `uniform[-1024,1024]` scale-active | `~1064 us` | `~862 us` | scaled dequant also benefits from `VecBytes=8` |

### Final profiler slot read

Profiler build, `normal`, final specialization:

| Slot | work-normalized |
| --- | ---: |
| `combine_stage_input` | `495.37 us` |
| `combine_barrier` | `64.88 us` |
| `combine_accum_setup` | `1.00 us` |
| `combine_prepare_ptrs` | `8.64 us` |
| `combine_dequant_accum` | `224.04 us` |
| Total | `793.93 us` |

Envelope view:

| Metric | value |
| --- | ---: |
| combine envelope | `827.74 us` |
| accum envelope | `266.98 us` |

For the same specialized path under `uniform[-1024,1024]`, the earlier
scale-active profiler showed `combine_stage_input ~= 506 us` and
`combine_dequant_accum ~= 230 us` work-normalized. That is now consistent
with the no-scale profile: dequant is no longer the largest slot.

### Current bottleneck read

For this production specialization, the next bottleneck is `CombineStageInput`,
not dequant. This stage includes:

- bf16 input reads;
- per-block max/reduce;
- fp8 writes;
- scale writes;
- staging layout writes.

The next useful optimization track is therefore quant/stage-side analysis and
tuning, not another dequant-only micro-optimization.
