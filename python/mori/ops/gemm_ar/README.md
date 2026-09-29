# `mori.ops.gemm_ar`

FP8 GEMM and GEMV operators, with an optional GEMM + all-reduce path for
tensor-parallel attention.

| Operator | Use | Operation |
|---|---|---|
| `GemmAllReduceOp` (`op.py`) | Prefill, row-parallel linear | GEMM with the all-reduce scatter issued from its epilogue, followed by reduce and gather |
| `Mxfp8GemmOp` (`gemm.py`) | Prefill, MXFP8 linear | GEMM without a communicator |
| `Mxfp8GemvOp` (`gemv.py`) | Decode, M <= 32 | Skinny MXFP8 GEMM without a collective |

The 2026-09-29 experiments support short final chunks, reduced padding,
shape-specific GEMV wave counts and workspace reuse. FP8 scatter offers a
latency/precision tradeoff. These remain explicit kernel or harness options;
the public operator routing and GEMV configuration table have not changed.

- [Using it](#using-it)
- [Measured results](#measured-results)
- [Earlier baseline measurements](#earlier-baseline-measurements)
- [End to end, in SGLang](#end-to-end-in-sglang)
- [How it works](#how-it-works)
- [Benchmarks and tests](#benchmarks-and-tests)
- [Measurement traps](#measurement-traps)

## Using it

### Fused GEMM + all-reduce

Build MORI with `BUILD_CCO_SDMA=ON`, and set both `BUILD_CCO_SDMA=ON` and
`MORI_ENABLE_SDMA=1` when running the SDMA benchmarks. The host library and
resolved device bitcode must both support SDMA. The standalone GEMM and GEMV
operators need no communicator or SDMA queues.

```python
import torch, torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

# The process group and this process's local GPU must already be initialized.
rank, world = dist.get_rank(), dist.get_world_size()

# The window comes out of the communicator's VMM reservation, so size that
# first -- the op cannot grow it later.
N, K, M_MAX = 7168, 2048, 16384
vmm = 2 * GemmAllReduceOp.window_bytes_for(world, m_max=M_MAX, n=N) + (64 << 20)

uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

# init takes the UniqueId wrapper, not the bytes that survive a broadcast.
with Communicator.init(
    world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
) as comm:
    with GemmAllReduceOp(comm, n=N, k=K, m_max=M_MAX) as op:
        b_shuffled = preshuffle_b(b_fp8)          # [N, K], once per weight

        m_pad = op.padded_m(x.shape[0])           # instance method: M only
        a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))
        # The result aliases the window, which the `with` frees on exit --
        # so clone inside it, after the work it was issued behind has run.
        out = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]]
        torch.cuda.current_stream().synchronize()
        out = out.clone()
```

**Pad before quantising.** A padded row produces a zero output row, which the
caller slices off; quantising first and padding after would also have to extend
the scales, and in the layout below that is not a row append.

**Scales.** `a_scale` is the A 1x128 block scale as fp32. The kernel reads
element `(row, kb)` at `kb * M + row`, so three spellings are accepted because
each already *is* that order: flat 1-D, `[K/128, M]`, and `[M, K/128]`
**column-major** (stride `(1, M)`), which is transposed for free.

`[M, K/128]` **row-major is rejected** with a `ValueError`. It is not that the
op cannot transpose it -- it is that the two `[M, K/128]` tensors are
indistinguishable by shape, and they need opposite treatment: the column-major
one must be read flat, the row-major one must be transposed. Guessing gets one
of them silently wrong (relL2 0.26 either way). Pass `.reshape(-1)` or `.t()`
yourself to say which you have. `aiter_per1x128_quant(transpose_scale=True)`
returns the column-major one, so `.reshape(-1)` is right for it.

`b_scale` is `[N/128, K/128]` fp32 row-major.

**Shapes.** `supports(m, n, k, world_size)` answers whether a shape is
expressible. `K >= 256` and a multiple of 128; `N` a multiple of 256, and of
1024 if the gather is fp8; `world_size` in [2, 8] and dividing 512.

**Collective contract.** Every rank must construct the op and call it the same
number of times in the same order -- the phases carry device-side barriers.
Work is issued on the current stream and completes asynchronously, so the usual
stream rules apply to the result.

**The result aliases the window** and is overwritten by the next call; clone it
to keep it. `pad_rows` returns a reused buffer with the same caveat. One
instance is not usable concurrently.

**Distinct M values.** Each M compiles its own kernel and takes its own counter
set, capped by `max_shapes`. That defaults to *every* legal M under `m_max`
(`padded_m(m_max) / (world_size * block_m)`). Warm each M up once before
capturing a graph.

**Cleanup.** `close()` (or the `with` above) releases the window and its memory
-- about 700 MiB per rank at TP8, M=16384, N=7168. The communicator holds a strong
reference to each, so dropping the op is not enough. The dev-comm is
deliberately *not* destroyed here: its SDMA queues are communicator-scoped, and
tearing them down ahead of the communicator aborts the process. They go with the
communicator. Synchronise first if anything may still be in flight.

**dtype.** A and B must be `torch.float8_e4m3fn`. `float8_e4m3fnuz` is the same
width but a different exponent bias, and the MMA atom implements OCP's, so it is
rejected rather than silently returning a result 4x too large.

### The GEMM without a collective

Neither of these takes a communicator, a window or a rank: the output is an
ordinary tensor. They exist because at DeepSeek-V4.1-Flash's shapes the GEMM is
worth more than the fusing, and because a `ColumnParallelLinear` has no
all-reduce to fuse with at all.

```python
from mori.ops.gemm_ar import (
    Mxfp8GemmOp, Mxfp8GemvOp, preshuffle_a_scale, preshuffle_b,
    supports_gemm, supports_gemv,
)

# Once per weight. w_exps is the checkpoint's [N/32, K/32] ue8m0 bytes.
b_shuffled = preshuffle_b(w_fp8)                            # [N, K]
b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)   # K-block major

# Prefill. M only has to be a multiple of 64 -- the grid is ceildiv(M, BLOCK_M)
# and the tail block masks its stores.
assert supports_gemm(N, K)
op = Mxfp8GemmOp(n=N, k=K)
m_pad = op.padded_m(x.shape[0])
a_fp8, a_exps = quantize_mxfp8(op.pad_rows(x, m_pad))   # pad before quantising
out = op(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale)[: x.shape[0]]

# Decode. No padding and none needed; M is a runtime argument, and both scales
# go in as the checkpoint stores them -- w_exps itself, not b_scale.
assert supports_gemv(N, K)
gemv = Mxfp8GemvOp(n=N, k=K)
out = gemv(x_fp8, b_shuffled, x_exps, w_exps)
```

**They share the weight and not the scales.** `preshuffle_b` output serves both,
so a server shuffles once. But `Mxfp8GemmOp` wants the A scale through
`preshuffle_a_scale` -- K-block major, four M tiles to a dword -- while
`Mxfp8GemvOp` takes **both scales exactly as the checkpoint stores them**,
row-major ue8m0 bytes, `[M, K/32]` and `[N/32, K/32]`. That is not an
inconsistency: the GEMM's sixteen lanes want sixteen *rows* of one K block and
have to be coalesced, where the GEMV's are sixteen *tokens*, M is at most 32, and
the A scale is small compared with the weight tensor. See
[the layout results](#the-mxfp8-gemm-on-its-own).

**Shapes.** `supports_gemm(n, k)` and `supports_gemv(n, k)` answer whether a
shape is expressible: K is a multiple of 128, with K >= 256 for the GEMM;
N is a multiple of 256 for the GEMM and of 32 for the GEMV. The GEMM accepts
M padded to a multiple of 64; the GEMV accepts M <= 32. These predicates check
shape support, not performance.

**GEMM dispatch depends on the grid.** `Mxfp8GemmOp` chooses the 128-column
internal tile when `ceildiv(M, 256) * (N / 256) < 140`, otherwise the
256-column tile. This is separate from a caller's decision to use MORI at all:
the earlier cold SGLang comparison supported a grid threshold around 64 for the
measured shapes. Recheck that threshold for another backend or workload.

**The GEMV's ceiling is 32 tokens**, because a token is an MFMA row and two tiles
of them is where the register file runs out. It picks a compile-time
configuration per M bucket from a tuned table in `gemv.py`; an untuned shape gets
a heuristic rather than an error. The 5/10-wave configurations measured below
are available through the kernel builder; the public dispatch table still uses
its existing configurations.

### FP8 on the wire

`GemmAllReduceOp(gather_dtype="fp8", gather_transport="lsa")` quantizes the
all-gather payload to e4m3 with one FP32 scale per row. `gather_transport="sdma"`
sends the payload and scales with copy engines and dequantizes in another
kernel. The LSA variant pulls and widens peer payloads in one kernel. The local
output slice retains its BF16 value; remote slices are quantized and restored.
The default gather dtype is BF16.

FP8 **scatter** is now implemented in the low-level fused SDMA path and exposed
by `bench_gemm_ar.py --scatter-dtype fp8`. It is not a constructor argument of
`GemmAllReduceOp`. The prototype requires aligned, non-compact row slices,
`BLOCK_N == scatter_scale_n`, and separate reduce/quantize phases. It cannot be
combined with relaxed rows, compact receive storage, direct LSA scatter,
`fuse_quantize`, or `fuse_reduce_push` in this version.

The scatter prototype reads each completed BF16 GEMM tile from the output
region, quantizes row-by-256-column groups into the input region, and sends
both payload and scales. It does not quantize directly from GEMM accumulators.
Both FP8 legs change the arithmetic; their error measurements and tested scope
are given below.

## Measured results

### 2026-09-29 experiment scope

Base commit: `78b7a5f31`. Single node, 8 gfx950 GPUs with 256 CUs per GPU,
PyTorch 2.10.0, HIP 7.2.26015 and FlyDSL 0.2.4. Other processes retained device
memory; GPU activity was zero before the batch measurements.

Prefill comparisons alternate variants in one process and communicator, using
the same weights and true input rows with separate windows. Each graph has
8 calls and each round has 21 replays. Each round records the maximum of the
per-rank medians; tables report medians over 3–5 alternating rounds. GEMV
comparisons share one
400 MiB weight ring and the same graph pointer order, with 80 calls per graph
and 5 alternating rounds. Launchers, tensor storage and graphs remain alive
through measurement.

The following timings cover GEMM + collective, or GEMV alone where indicated.
They exclude compilation, weight packing, activation quantization/padding and
Python initialization. They are **kernel/collective microbenchmarks, not model
end-to-end speedups**. Raw results and the experiment harness are maintained
outside the repository. Only the final shared-ring GEMV measurements are used;
records marked `invalid-harness` are excluded.

### Short final chunks

TP8, N=7168, K=2048, blockscale operands, BF16 communication. Each chunk has at
most two 128-row bands; the final chunk's completion count and SDMA byte count
follow its actual length.

| M | Baseline µs | Short-tail µs | Latency reduction |
|---:|---:|---:|---:|
| 11264 | 1046.23 | 819.09 | 21.7% |
| 13312 | 1242.10 | 949.82 | 23.5% |
| 16384 | 1133.65 | 1138.83 | -0.5% |

At M=11264/13312, each peer owns 11/13 bands. The divisor-based baseline uses
one chunk; the new path uses 6/7 and restores scatter overlap. M=16384 already
has eight chunks and gains nothing. Outputs are bitwise equal to the baseline,
including after changing inputs.

Enable with `compile_fused_gemm_scatter(chunk_bands=2)` and a matching
`counter_chunks=ceil(bands_per_peer / 2)`. The public operator still uses its
divisor-based chunk selection.

### Reduced padding

TP4, N=5120, K=2048, MXFP8 operands, BF16 communication. Tiles may cross
owner/chunk row boundaries; completion counters count tile/segment intersections.

| True M | Baseline padded M / µs | Align 256 padded M / µs | Align 64 padded M / µs | Best latency reduction |
|---:|---|---|---|---:|
| 4200 | 5120 / 518.56 | 4352 / 460.18 | 4224 / 449.26 | 13.4% |
| 8200 | 9216 / 862.05 | 8448 / 822.24 | 8256 / 807.00 | 6.4% |

True output rows are bitwise equal to the baseline, including changed-input
checks. `relaxed_rows=True` requires fused SDMA, `chunk_bands > 0`, at least
one full tile per peer, and BF16 scatter. The external harness supplies the
reduced padded shapes; public support predicates, padding and dynamic-M plan
caching still use the existing alignment contract.

### GEMV wave counts and compact reduction

Single GPU, N=8192, K=1280, MXFP8 input; shared weight ring. These are GEMV-only
timings against the existing MORI GEMV configuration.

| M | Default µs | Compact reduction µs | Specialized waves | Specialized µs | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.060 | 4.098 | 5 | 3.899 | 4.0% |
| 4 | 4.202 | 4.251 | 5 | 3.953 | 5.9% |
| 8 | 4.279 | 4.258 | 5 | 4.062 | 5.1% |
| 16 | 4.778 | 4.700 | 10 | 4.330 | 9.4% |
| 32 | 5.389 | 5.745 | 10 | 4.947 | 8.2% |

All variants are bitwise equal to the default GEMV. K=1280 contains ten
128-wide steps; five or ten waves avoid the unused steps in four- or
sixteen-wave partitions. Use these configurations only for tested shapes.
Reducing only valid tokens (`valid_tokens`) has no consistent benefit and is
not recommended as a default.

| Configuration | VGPRs | LDS bytes | Scratch bytes |
|---|---:|---:|---:|
| M4, default 4 waves | 54 | 4096 | 0 |
| M4, compact reduction | 52 | 1024 | 0 |
| M4, 5 waves | 48 | 5120 | 0 |
| M32, default 16 waves | 46 | 65536 | 0 |
| M32, compact reduction | 44 | 65536 | 0 |
| M32, 10 waves | 52 | 40960 | 0 |

### Workspace reuse

TP8, M=16384, N=7168, K=2048, blockscale operands, BF16 communication.

| Layout | Window MiB/rank | µs |
|---|---:|---:|
| Baseline | 700.006 | 1134.29 |
| Remove unused SDMA tmp | 672.006 | 1136.39 |
| Also remove self receive slot | 644.006 | 1134.72 |
| Also alias input/output | 420.006 | 1135.58 |

The final layout saves 280 MiB/rank (40%) with effectively unchanged latency
and bitwise equal output, including changed inputs. It relies on serial calls
and scatter drain completing before input/output reuse. The external harness
supplies this layout; it is not the public operator's allocation policy and
has not been validated for concurrent calls or arbitrary transport/option
combinations.

### FP8 scatter

TP4, M=16384, N=5120, K=2048, MXFP8 operands. Three-round paired performance
runs, eight scatter chunks:

| Gather | BF16 scatter µs | FP8 scatter µs | Latency reduction | FP32 reference relL2, before / after |
|---|---:|---:|---:|---|
| BF16 | 1487.88 | 1182.65 | 20.5% | 0.00237 / 0.02618 |
| FP8 / LSA | 1183.30 | 881.21 | 25.5% | 0.02309 / 0.03482 |

Four chunks were slower: approximately 1207.06/904.56 µs with BF16/FP8 gather.
The additional correctness reruns are separate from these paired timings.

At M=4096, stage-by-stage validation checked the actual FP8 wire payload and
scales. Scale relative error was about 6.6e-8; differences from nominal PyTorch
quantization were adjacent FP8 values at rounding midpoints. An independent
sum reconstructed from the actual payload and scales matched the collective
output exactly with BF16 gather.

Relative to the FP32 reference, FP8 scatter raises L2 error to about **2.62%**;
using FP8 for both legs raises it to about **3.48%**. No model-quality test has
been run for FP8 scatter. Earlier gather-only quality results do not validate
this option.

### Decode GEMV + all-reduce

TP8, N=5120, K=2048. Baseline: staged GEMV + LSA all-reduce. The tagged
prototype uses double buffering and an atomic 64-bit payload plus epoch.

| M | Staged µs | Tagged fusion µs | Slowdown |
|---:|---:|---:|---:|
| 1 | 7.247 | 8.406 | 16.0% |
| 4 | 8.229 | 10.056 | 22.2% |
| 8 | 13.819 | 17.059 | 23.4% |

The ordinary ready/done-flag prototype took about 10.834 µs at M=4. Both
prototypes pass an ordered BF16-partial reference and changed-input checks,
but neither should replace the staged path based on these measurements.

### Grouped reduce + quantize

TP4, M=16384, FP8 gather. Folding groupwise quantization into the flat-pack
reduce mapping did not improve latency:

| Variant | µs | Change |
|---|---:|---:|
| Separate reduce + quantize | 1182.9 | — |
| Group 128 | 1212.1 | 2.5% slower |
| Group 256 | 1213.1 | 2.6% slower |
| Group 512 | 1213.7 | 2.6% slower |

Correctness passed. This is a negative result for these mappings; fewer
launches did not offset the additional reduction/quantization work.

## Earlier baseline measurements

These predate the 2026-09-29 experiments and use BF16 scatter. They describe
separate comparisons and must not be combined with the percentages above.

### The mxfp8 GEMM, on its own

`preshuffle_a_scale` puts scales in K-block-major order and packs four M tiles
into one dword for the scaled MFMA's `opsel` byte selection. In the earlier
M=4096–16384 layout comparison, row-major scales increased cache accesses from
8,622,080 to 27,729,920 and GEMM latency by 50–67%. Packing reduced VMEM
instructions from 896,000 to 634,880 and latency by 5–7.5%.

Use `bench_gemm.py --scope linear` with the shared timer for whole-linear
comparisons, including quantization and enough graph calls to amortize replay
overhead.

### The mxfp8 GEMM against SGLang, across every shape

The corrected cold sweep compared both MORI tiles against
`mxfp8_native_blockscaled_linear`, including BF16 input quantization and
rotating every weight representation used by either route. Across 110 points
where both tiles were measured, the 128-column tile won at wide-tile grids
up to 128 and the 256-column tile won at grids of 129 or more. The implementation uses a
wide-tile grid threshold of 140.

Caller admission is a different choice. Across the 77 supported points above
the GEMV token range, grid thresholds 48 and 64 each misclassified 2.9% of
points, while 80 misclassified 7.3%. This supports a grid threshold around 64
for that comparison, not a universal M threshold. Untuned GEMV shapes also need
measurement: `wo_a` lost 12–16% at M >= 8 in that sweep. The two tuned TP4
shapes do not establish a gain for every N/K or TP degree.

### Prefill baselines

All latencies are in µs; FP8 here means **FP8 gather with BF16 scatter**.

| Quantization / TP / N / K | M | Split SDMA | Fused SDMA, BF16 gather | Fused SDMA, FP8 gather |
|---|---:|---:|---:|---:|
| blockscale / TP8 / 7168 / 2048 | 4096 | 398.1 | 351.0 | 329.5 |
| blockscale / TP8 / 7168 / 2048 | 8192 | 722.0 | 621.1 | 539.3 |
| blockscale / TP8 / 7168 / 2048 | 16384 | 1472.9 | 1148.8 | 979.3 |
| MXFP8 / TP4 / 5120 / 2048 | 4096 | 443.6 | 437.3 | 358.8 |
| MXFP8 / TP4 / 5120 / 2048 | 8192 | 827.3 | 801.8 | 643.1 |
| MXFP8 / TP4 / 5120 / 2048 | 16384 | 1593.5 | 1500.7 | 1195.1 |

At the MXFP8 TP4 M=16384 point, BF16 fusion reduced latency by 5.8%; FP8
gather plus fusion reduced it by 25.0% against split/BF16. The contributions
of GEMM, overlap and communication precision depend on shape and quantization.

### End to end, in SGLang

These are earlier server measurements with **BF16 scatter**, not end-to-end
validation of the new chunk, padding, wave-count, workspace or scatter options.
V4.1-Flash TP4 prefill throughput, `bench_one_batch_server`, in tokens/s:

| Batch size | Baseline | Fused wo_b, BF16 gather | Fused wo_b, FP8 gather | FP8 gather + standalone MORI GEMM |
|---:|---:|---:|---:|---:|
| 1 | 28148 | 28469 | 29107 | 29504 |
| 4 | 34386 | 35356 | 36442 | 37023 |
| 8 | 35153 | 35920 | 37217 | 37870 |
| 16 | 35423 | 36084 | 37416 | 38058 |

The recorded GSM8K scores were 0.917 / 0.912 / 0.918 / 0.918. These runs used
different admission thresholds for BF16 and FP8 gather, so batch size 1 was a
control only for the BF16-gather configuration. The combined configuration
reached about 7.7% higher throughput at batch sizes 4 and 8; the separate gains
must not be added because the paths serve overlapping layers.

One combined run reported an illegal memory access after 13 minutes; a repeat
with the same configuration completed successfully. The fault was not
reproduced or attributed. These results do not establish long-running
stability of that integration.

For V4-Pro TP8, an earlier profiled prefill measured:

| Path | GPU busy ms | Wall seconds |
|---|---:|---:|
| Unfused GEMM + NCCL | 1096.0 | 1.1969 |
| Fused, BF16 gather | 1070.3 | 1.1728 |
| Fused, FP8 / SDMA gather | 1052.1 | 1.1455 |
| Fused, FP8 / LSA gather | 1041.2 | 1.1385 |

Gather-only quality checks on 10,941 source-text tokens placed FP8 perplexity
at 3.2671 within the BF16 rerun range 3.2547–3.2680; a needle task scored
24/24 for both. These are limited task checks, not evidence that FP8 preserves
quality on all workloads, and they do not cover FP8 scatter.

## How it works

`kernels_fused.py` holds the GEMM and fused scatter, `kernels_sdma.py` the
SDMA phases, `kernels_lsa.py` the LSA collective, and `kernels_gemv.py` the
skinny GEMM. `layout.py` defines symmetric-window offsets and counts.
`_gemm_a8w8_8wave.py` and `_shuffle.py` contain the vendored aiter pieces.

For BF16 scatter, GEMM writes C into the registered input region. A tile retires
its stores and participates in a monotonic completion counter for each
destination/chunk it touches. The elected producer submits an SDMA put when
the counter reaches a multiple of that chunk's expected contribution count.
The default path counts whole row bands; short tails change the count and
transfer length, and relaxed rows count tile/segment intersections.

Destination-rotated tile order lets several peer links begin while GEMM is
still running. A per-destination submit lock serializes potentially competing
chunk producers. The epilogue issues puts asynchronously; a separate drain
waits for completion before local reduction and gather. No `quiet` is issued
from the GEMM epilogue. Counters persist across graph replays, and each public
M plan has its own counter set.

Phase launchers encode the complete configuration in an explicit specialization
identity and unique kernel names. This resolved the small-K factory/cache
anomaly observed during the experiments; two independent process reruns passed.
`_PinnedLaunch` retains the compiled launch state, and graphs must retain the
launchers that own their code modules as well as operand storage.

Earlier negative experiments included direct LSA scatter, persistent GEMM
tiles, rowwise fused reduce/quantize, and reduce-triggered gather puts. Their
extra synchronization, mapping or issue costs outweighed the saved work on
the tested shapes. The current decode-fusion and grouped-quantize results above
provide additional measured examples; none justifies changing the default path.

## Benchmarks and tests

Run from the repository root with MORI's native libraries and FlyDSL available:

```bash
export PYTHONPATH="$PWD/python"
export BUILD_CCO_SDMA=ON MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo

# Baseline fused GEMM + all-reduce.
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048

# Short-tail chunks at a shape where the baseline only has one chunk.
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 11264 -n 7168 -k 2048 \
  --chunk-bands 2

# Lossy FP8 scatter with BF16 gather.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant mxfp8 -m 16384 -n 5120 -k 2048 \
  --scatter-dtype fp8 --gather-dtype bf16

# Operator, layout and kernel regressions, including the public multi-rank op.
python -m pytest -q \
  tests/python/cco/test_gemm_ar_op.py tests/python/cco/test_gemm_ar.py \
  -k 'not fused_is_stable_across_repeats and not gemm_ar_modes_agree'
```

Run GPU jobs sequentially and leave 20 seconds between independent CCO process
groups so SDMA queues can be reclaimed. These commands exercise the paths;
the paired performance tables were produced by the external harness described
in the measurement scope.

Validation when preparing this branch: **83 passed, 4 deselected** for the
regression command above; two independent K=256 process runs; and TP4 FP8
scatter/gather checks. The original experiment regression selected 54 tests.
Correctness checks include an independent FP32 reference, repeated calls and
changed inputs; precision-preserving variants additionally check bitwise
agreement, and FP8 communication has a staged wire reference. Black, Ruff,
Python compilation and `git diff --check` passed.

More benchmark flags and sweep commands are documented in
[`docs/MORI-GEMM-AR-BENCHMARK.md`](../../../../docs/MORI-GEMM-AR-BENCHMARK.md).

### Measurement traps

- Amortize graph replay overhead with multiple calls per graph. Use
  `benchmark/cco/flydsl/gemm_ar/timing.py`; a one-call graph inflated short
  kernel measurements in the earlier harness.
- Rotate enough weight storage to exceed LLC, and capture enough calls to visit
  that storage. Cool every weight representation consumed by either route,
  including a dequantized BF16 baseline weight. Share the same ring and pointer
  order when comparing GEMV variants.
- Keep tensors, graphs and FlyDSL launchers alive until all replays finish.
  The first shared-ring harness lost launcher ownership; those records are
  marked `invalid-harness` and are excluded from the results.
- Pin the quantization mode and communication formats. The benchmark defaults
  to `ptpc`; the blockscale and MXFP8 tables explicitly select another mode.
  Verify that the requested collective actually runs and check its output.
- Keep compilation and warmup outside timing, and compare eager/graph overhead
  consistently. Use pinned launchers when reproducing the public operator.
- Test true M around padding boundaries. One padded shape can have different
  economics against a baseline whose kernel selection follows the true M.
- Preserve anomalous observations and rerun paired measurements. The early
  isolated chunk sweep contained outliers; the short-tail table uses subsequent
  same-communicator alternating runs.
