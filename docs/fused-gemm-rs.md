# GEMM + reduce-scatter

`GemmReduceScatterOp` is the [GEMM + all-reduce](fused-gemm-ar.md) pipeline
stopped before the gather, for sequence parallelism. Each rank multiplies its
activations by its shard of a row-parallel weight; rank `r` keeps rows `r*M/P`
onward of the sum: the dim-0 split `reduce_scatter_tensor` makes, and what a
sequence-parallel layer keeps after a RowParallelLinear. The GEMM's epilogue
pushes each destination's row band as it is produced, so the transfer overlaps
the rest of the GEMM, and the reduce then sums this rank's band. It lives in the
same module, `mori.ops.gemm_ar`, and subclasses `GemmAllReduceOp`.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. What it computes](#1-what-it-computes)
- [2. How it works](#2-how-it-works)
- [3. Requirements](#3-requirements)
- [4. Constructor](#4-constructor)
- [5. Calling the operator](#5-calling-the-operator)
- [6. Other methods](#6-other-methods)
- [7. Shapes and limits](#7-shapes-and-limits)
- [8. Choosing settings](#8-choosing-settings)
- [9. Performance](#9-performance)
- [10. Complete example](#10-complete-example)
- [11. Testing and benchmarking](#11-testing-and-benchmarking)
- [12. Troubleshooting](#12-troubleshooting)
- [Further reading](#further-reading)

## Quick Reference

```python
from mori.ops.gemm_ar import GemmReduceScatterOp, preshuffle_a_scale, preshuffle_b

# comm: a mori.cco.Communicator whose per_rank_vmm was sized from
# GemmReduceScatterOp.window_bytes_for(world, **settings); see section 10.
settings = dict(n=5120, k=2048, m_max=16384, quant="mxfp8")  # V4.1-Flash wo_b, TP4
with GemmReduceScatterOp(comm, **settings) as op:
    op.self_test()                                       # collective, once per op
    b_shuffled = preshuffle_b(b_fp8)                     # once per weight
    b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)  # [N/32, K/32] ue8m0
    m_pad = op.padded_m(x.shape[0])
    a_fp8, a_exps = quantize_mxfp8(op.pad_rows(x, m_pad))  # your quantiser
    band = op(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale).clone()  # [m_pad / P, N] bf16
    torch.cuda.synchronize()                             # before close() frees the window
```

| What | Import |
|---|---|
| The operator | `from mori.ops.gemm_ar import GemmReduceScatterOp` |
| Weight and mxfp8 A-scale layouts | `from mori.ops.gemm_ar import preshuffle_b, preshuffle_a_scale` |
| Communicator | `from mori.cco import Communicator, UniqueId` |
| The all-reduce it extends | `from mori.ops.gemm_ar import GemmAllReduceOp` ([its page](fused-gemm-ar.md)) |

## 1. What it computes

Rank `r` of `P` passes its activations `A_r` and its weight shard `B_r`, and
receives its row band of the sum:

```text
band [M/P, N] bf16 = rows r*M/P .. (r+1)*M/P - 1 of
                     sum over ranks of  dequant(A_r [M, K]) @ dequant(B_r [N, K])^T
```

As for the all-reduce, `K` is the per-rank K and `N` is not sharded; the GEMM
accumulates in fp32, the partials travel as bf16, and the reduce adds them in
fp32 and rounds once to bf16.

| | Shape | dtype |
|---|---|---|
| A (`a_fp8`) | `[M, K]`, the whole padded sequence: `M = op.padded_m(rows)`, the same on every rank | `float8_e4m3fn`; bf16 with `quant="bf16"` |
| B (`b_preshuffled`) | `[N, K]` | `float8_e4m3fn` through `preshuffle_b`; with `quant="bf16"` the row-major bf16 weight |
| Scales | per `quant`, as for the [all-reduce](fused-gemm-ar.md#5-calling-the-operator) | fp32, int32, uint8, or none |
| Result | `[M/P, N]` row-major: rows `r*M/P` onward | bf16 |

**The padding decides the bands.** The split is over the padded M, and the
padded rows, which come back as zeros, sit at the end, in the last band or
bands: rank `r` holds `max(0, min(M/P, rows - r*M/P))` real rows. A
sequence-parallel layer that shards the unpadded token count would disagree
about which rows each rank holds, so pad the sequence to `op.padded_m()` before
it is sharded.

The layer is a RowParallelLinear under sequence parallelism; the measured one is
DeepSeek-V4.1-Flash's `wo_b` at TP4 (N=5120, K=2048, mxfp8).

## 2. How it works

No kernel differs from the all-reduce's
([how that works](fused-gemm-ar.md#2-how-it-works)). The GEMM pushes each
destination's band in chunks from its epilogue, with the same default chunk
rule (the largest divisor of the bands per destination, up to 8). Then three
kernels run:

1. **Drain**: waits for this rank's pushes to land, then a cross-rank barrier.
2. **Reduce**: sums the `P` contributions to this rank's band.
3. **`rs_barrier`**: a barrier-only kernel standing in for the one the gather
   used to provide.

The last barrier is load-bearing. The reduce reads receive slots that the peers
fill, and the next call's epilogue refills them, so without it a rank one call
ahead overwrites a slower peer's slot mid-reduce. Before it existed this showed
as relL2 0.71 on one call of an alternating-M test; the test now holds one rank
between drain and reduce, and runs the same schedule without `rs_barrier` as a
negative control that has to corrupt.

The window is the all-reduce's: it still holds a whole `[m_max, N]` output
region, of which this op writes `1/P`, so it is about 700 MiB per rank at
`m_max=16384`, N=7168, `P=8` and 520 MiB at N=5120, `P=4`. Shrinking it would
change the layout every other region is placed by.

## 3. Requirements

The same as for [GEMM + all-reduce](fused-gemm-ar.md#3-requirements):

- MI355X (gfx950), and the FlyDSL extra (`pip install 'amd_mori[flydsl]'`).
- MORI built with `BUILD_CCO_SDMA=ON` (the default) and `MORI_ENABLE_SDMA=1` at
  run time. The constructor does not check that the communicator has SDMA
  queues, so `self_test()` is the only guard.
- One node, `P` = 2, 4 or 8; every rank constructs the op and makes every call.
- A `Communicator` whose `per_rank_vmm` is sized from
  `GemmReduceScatterOp.window_bytes_for`, called with the constructor's
  settings.

The rules shared by every fused operator are in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common).

## 4. Constructor

```python
GemmReduceScatterOp(comm, *, n, k, m_max, block_m=None, block_n=256,
                    sdma_queues=1, quant="blockscale", max_shapes=None)
```

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | *(required)* | The communicator; `P = comm.nranks`. The op does not take ownership. |
| `n` | `int` | *(required)* | Output width N, a multiple of 256. There is no gather, so no fp8-wire rule. |
| `k` | `int` | *(required)* | Per-rank K. fp8 quants: a multiple of 128, at least 256. `"bf16"`: a multiple of 64, at least 128. |
| `m_max` | `int` | *(required)* | Largest per-rank M before the split (the whole sequence), at least 1. Rounded up to a multiple of `P * block_m`; sizes the window, which cannot grow. |
| `block_m` | `int \| None` | `None` | Tile rows and padding granule. `None` picks 128 for `"blockscale"` and 256 otherwise; `"mxfp8"` and `"mxfp8_raw"` accept only 256, `"bf16"` 128 or 256, and `"blockscale"` any positive multiple of 128 (128 is the measured tile). |
| `block_n` | `int` | `256` | Tile columns; only 256 is accepted. |
| `sdma_queues` | `int` | `1` | SDMA queues per (source, destination) pair, at least 1. One is all the pipeline uses, and the measurements below use 1. |
| `quant` | `str` | `"blockscale"` | `"blockscale"`, `"mxfp8"`, `"mxfp8_raw"` or `"bf16"`, as for the [all-reduce](fused-gemm-ar.md#4-constructor). |
| `max_shapes` | `int \| None` | `None` | Most distinct M the instance serves; `None` reserves one counter set for every padded M up to `m_max`. |

**Not accepted.** The all-reduce's `gather_dtype`, `gather_transport`,
`scatter_dtype` and `schedule` are not parameters, so passing one raises
`TypeError`. The scatter is the only leg and stays bf16, so there is no fp8
wire, and M pads by the default rule, so the bands are even: there is no
`schedule="wo_b"`. The attributes keep the all-reduce's defaults (`"bf16"`,
`"lsa"`, `"bf16"`, `"default"`). Construction is collective, raises
`ValueError` on a rule from [section 7](#7-shapes-and-limits) before allocating
anything, and releases what it acquired if the communicator fails.

### window_bytes_for()

```python
GemmReduceScatterOp.window_bytes_for(world_size, *, m_max, n, block_m=None,
                                     max_shapes=None, k=None, quant="blockscale") -> int
```

Static. Returns `GemmAllReduceOp.window_bytes_for` for these arguments with the
default wires and schedule, which is what the constructor allocates. Pass the
constructor's `quant`, and `block_m` if it set one: when `block_m` is `None` the
tile, and so the padding, follows the quant. `k` is optional. Raises
`ValueError` for an unknown `quant`, a `block_m` the quant forbids, a
non-positive size, or a world size the window cannot take.

## 5. Calling the operator

```python
band = op(a_fp8, b_preshuffled, a_scale=None, b_scale=None)
```

The operands are exactly the all-reduce's, including the scale layouts for each
`quant` ([section 5 there](fused-gemm-ar.md#5-calling-the-operator)); for
`quant="bf16"` they are bf16 A and the checkpoint's row-major `[N, K]` bf16
weight, with no scales.

| Argument | Shape | dtype | Notes |
|---|---|---|---|
| `a_fp8` | `[M, K]` | `float8_e4m3fn` (bf16 with `quant="bf16"`) | The whole padded sequence: M a multiple of `P * block_m`, the same on every rank, at most `op.m_max`. |
| `b_preshuffled` | `[N, K]` | `float8_e4m3fn` (bf16 with `quant="bf16"`) | `preshuffle_b(w)`; for `"bf16"` not shuffled. |
| `a_scale` | per `quant` | fp32, int32 or uint8 | `None` for `"bf16"`. |
| `b_scale` | per `quant` | fp32, int32 or uint8 | `None` for `"bf16"`. |

There is no `logical_m`: M itself must be padded, and passing it raises
`TypeError`.

**Returns** an `[M/P, N]` bf16 row-major tensor: a view of this rank's band in
the op's window, rows `rank*M/P` onward of the summed GEMM, valid once the work
queued on the current stream has run. This rank's next call overwrites it and
`close()` frees it: order its readers before the next call (same stream, or an
event the next call's stream waits on), and `.clone()` it to keep it. The first
call at a new M compiles that M's kernels; one instance is not usable
concurrently.

## 6. Other methods

All inherited from `GemmAllReduceOp`; the
[all-reduce page](fused-gemm-ar.md#6-other-methods) has the details.

### padded_m()

`op.padded_m(m) -> int`: `m` rounded up to a multiple of `P * block_m`, which is
also the granule a sequence-parallel caller must pad to before sharding.
`ValueError` if `m < 1`.

### pad_rows()

`op.pad_rows(x, m_pad) -> Tensor`: zero-extends a 2-D `[M, K]` tensor to
`m_pad` rows in a buffer the op reuses, which the next `pad_rows` overwrites.
Pad before quantising, so the scales cover the padded rows.

### self_test()

`op.self_test(m=None)`, collective: writes `rank + 1` over the input region,
runs scatter, reduce and `rs_barrier` without a GEMM, and checks that every
element of this rank's band is `P * (P + 1) / 2` within 1%; otherwise it raises
`RuntimeError: gemm_ar self-test failed: the reduce-scatter returned ...`. `m`
defaults to `m_max`. Call it once on every rank after construction, before any
capture.

### close() and the context manager

`op.close()` deregisters the window and frees its memory; it is idempotent, and
a closed op raises `RuntimeError`. Synchronise first if work may be in flight.
`with GemmReduceScatterOp(...) as op:` calls `close()` on exit.

### Attributes

`window_bytes` (equal to `window_bytes_for`), `m_max` (padded), the
constructor's settings and the inherited wire settings are plain attributes,
for reading; `dev_comm.sdma_num_queue` is the number of SDMA queues per peer the
communicator actually connected.

## 7. Shapes and limits

The rules are the all-reduce's
([its table](fused-gemm-ar.md#7-shapes-and-limits)) without the wire options.
The ones that differ or matter most here:

| Constraint | Value | Error if violated |
|---|---|---|
| Call M | a multiple of `P * block_m` (`op.padded_m()`) | `ValueError`: `M=... is not a multiple of world_size*block_m; pad the rows to padded_m(...)=... before sharding them` |
| Call M | at most `op.m_max` | `ValueError`: `M=... exceeds the window's m_max=...` |
| All-reduce-only options | not accepted | `TypeError`: `... got an unexpected keyword argument 'gather_dtype'` (likewise `gather_transport`, `scatter_dtype`, `schedule`, and `logical_m` on the call) |
| `quant`, `block_m`, `block_n` | as for the all-reduce | `ValueError`, as for the all-reduce |
| K | fp8 quants: a multiple of 128, at least 256; `"bf16"`: a multiple of 64, at least 128 | `ValueError`: `unsupported shape: K=...` |
| N | a multiple of 256 | `ValueError`: `unsupported shape: N=... must be a multiple of block_n=256` |
| Ranks `P` | 2, 4 or 8 | `ValueError`: `world_size must be in [2, 8]`, `threads (512) must be divisible by world_size ...` |
| Operands and scales | as for the all-reduce | `ValueError`, as for the all-reduce |
| Lifetime | not closed | `RuntimeError`: `this GemmReduceScatterOp has been closed` |
| Window | fits the free VMM reservation | `RuntimeError`: `ccoMemAlloc failed: ...` |

## 8. Choosing settings

| Setting | What the default picks | Change it when |
|---|---|---|
| `quant` | `"blockscale"` | the checkpoint is mxfp8 (`"mxfp8"`, or `"mxfp8_raw"` for the quantiser's row-major bytes) or BF16 |
| `block_m` | 128 for blockscale, 256 otherwise | rarely: blockscale takes other granule-legal tiles, and bf16 also takes 128, which doubles the grid at small M; the mxfp8 forms need 256 |
| Chunks | the largest divisor of the bands per destination, up to 8: at V4.1-Flash TP4 that is 2 at M=2048, 4 at 4096 and 8 above | not a parameter |
| Wires | bf16 scatter, no gather | not available |
| `sdma_queues` | 1 | keep |
| `max_shapes` | every padded M up to `m_max` | keep |

**Whether to fuse at your M.** At V4.1-Flash TP4 the fused op was ahead of the
GEMM followed by `reduce_scatter_tensor` at every M measured, 2048 included. At
M=2048 there are only two chunks per destination and it trails the split SDMA
path by 2.5%, but that split has no public op.

The operator reads no environment variable of its own. `MORI_ENABLE_SDMA=1` is
required, `MORI_SOCKET_IFNAME` picks the bootstrap interface (the tests use
`lo`), and `MORI_GEMM_AR_PULL_BLOCKS` has no effect because there is no gather.

## 9. Performance

DeepSeek-V4.1-Flash `wo_b` (`[M, 5120]`, K=2048, mxfp8) on 4x MI355X (ROCm 7.2,
FlyDSL 0.4.0), from the
[README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md):
`bench_gemm_ar.py --collective rs --sdma-queues 1`, graph replay, maximum over
ranks, median of 3 interleaved rounds on an idle box (spread at most 4.7% at
M=2048 and 2.7% above). The baseline is the GEMM followed by
`reduce_scatter_tensor`, what a serving stack runs today. µs; the fused column
gives the chunks per destination:

| M | GEMM only | GEMM + `reduce_scatter_tensor` | split SDMA | fused | vs RCCL | vs split SDMA |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 39.6 | 161.7 | 152.8 | 156.6 (2) | -3.2% | +2.5% |
| 4096 | 66.4 | 295.4 | 269.9 | **262.9** (4) | -11.0% | -2.6% |
| 8192 | 104.4 | 540.5 | 484.2 | **460.2** (8) | -14.9% | -5.0% |
| 16384 | 174.9 | 1065.3 | 917.3 | **834.5** (8) | **-21.7%** | -9.0% |

- Two separate effects. The copy engines alone beat RCCL by 5-14% (split SDMA
  against GEMM + `reduce_scatter_tensor`); fusing then hides about 83 of the
  175 µs of GEMM at M=16384.
- At M=2048 fusing loses to split SDMA, which has no public op; against the
  split a caller actually has, the fused op is ahead at every M measured.
- The link is still the limit: the unfused transfer is about 740 µs at M=16384,
  about 170 GB/s per rank over TP4's three links.
- The RCCL path validates at relL2 3.15e-3 against 2.37e-3 for the others. That
  is not an error: RCCL adds the partials in bf16 hop by hop, where the SDMA
  reduce accumulates in fp32 and rounds once.

## 10. Complete example

Run with `MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=4 example.py`.
`quantize_mxfp8` stands for your quantiser; MORI ships none. It must return
`a_fp8` as `[m_pad, K]` `float8_e4m3fn` and `a_exps` as the `[m_pad, K/32]`
ue8m0 exponent bytes, which `preshuffle_a_scale` turns into the flat int32
layout `quant="mxfp8"` reads (with `quant="mxfp8_raw"`, pass them as uint8
instead).

```python
import os
import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmReduceScatterOp, preshuffle_a_scale, preshuffle_b

# DeepSeek-V4.1-Flash wo_b at TP4; the same settings size the window and build the op.
SETTINGS = dict(n=5120, k=2048, m_max=16384, quant="mxfp8")
N, K = SETTINGS["n"], SETTINGS["k"]

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="cpu:gloo")  # only carries the unique id
rank, world = dist.get_rank(), dist.get_world_size()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

vmm = 2 * GemmReduceScatterOp.window_bytes_for(world, **SETTINGS) + (256 << 20)
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with GemmReduceScatterOp(comm, **SETTINGS) as op:
        op.self_test()  # every rank, once, before any capture

        # This rank's weight shard and its [N/32, K/32] ue8m0 scales; stand-ins.
        w_fp8 = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        w_exps = torch.full((N // 32, K // 32), 120, device="cuda", dtype=torch.uint8)
        b = preshuffle_b(w_fp8)
        b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)  # K-block major

        def wo_b(x):  # x: the whole sequence, [rows, K] bf16, rows <= m_max
            a_fp8, a_exps = quantize_mxfp8(op.pad_rows(x, op.padded_m(x.shape[0])))
            return op(a_fp8, b, preshuffle_a_scale(a_exps), b_scale)  # [m_pad / P, N]

        # Warm every M you will run (the first call compiles it), largest first.
        for rows in (16384, 5000):
            band = wo_b(torch.randn(rows, K, device="cuda", dtype=torch.bfloat16))
        band_rows = op.padded_m(5000) // world
        real = max(0, min(band_rows, 5000 - rank * band_rows))  # padding sits last
        mine = band[:real].clone()

        # CUDA graph: a static input, warm-up on a side stream, then capture.
        x_static = torch.randn(16384, K, device="cuda", dtype=torch.bfloat16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                wo_b(x_static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            band_static = wo_b(x_static)  # a view of this rank's band in the window
        x_static.copy_(torch.randn_like(x_static))  # new data, same buffer
        graph.replay()  # on every rank, in the same order
        result = band_static.clone()

        torch.cuda.synchronize()  # nothing in flight when close() frees the window
dist.destroy_process_group()
```

## 11. Testing and benchmarking

```bash
# The public op's reduce-scatter cases (blockscale and bf16), on 2 ranks
BUILD_CCO_SDMA=ON MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  pytest tests/python/cco/test_gemm_ar_op.py -k reduce_scatter

# The section 9 table at M=16384; repeat per --mode and -m
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=4 benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --collective rs --sdma-queues 1 --mode fused-sdma --quant mxfp8 -m 16384 -n 5120 -k 2048
```

`-k reduce_scatter` selects `test_reduce_scatter_returns_my_band_of_the_sum`
(each rank's band, its shape, two M through one instance, fresh operands, and the
self-test's teeth) and `test_reduce_scatter_survives_a_slow_reader` (the
`rs_barrier` check and its negative control). The module's worker runs all its
2-rank cases in one spawn, so it needs two GPUs.

With `--collective rs` the table's modes are `gemm-only`, `split-rccl`,
`split-sdma` and `fused-sdma`; the benchmark refuses `split-lsa`,
`--gather-dtype fp8`, `--scatter-dtype fp8` and `--fuse-reduce-push`. It
drives the same kernels through the low-level builders, prints a `RESULT_JSON`
line with `max_rank_time_us`, `rel_l2` and `validated`, and defaults to
`--quant ptpc` and `--sdma-queues 8`, so pin both. The
[all-reduce page](fused-gemm-ar.md#11-testing-and-benchmarking) has the rest of
the suite and the sweeps.

## 12. Troubleshooting

| Symptom | What it means |
|---|---|
| `ValueError: M=... is not a multiple of world_size*block_m; pad the rows to padded_m(...)=... before sharding them` | Pad the whole sequence with `op.padded_m()` and `op.pad_rows()` before quantising, and before your sequence-parallel split. |
| `TypeError: ... got an unexpected keyword argument 'gather_dtype'` (or `gather_transport`, `scatter_dtype`, `schedule`, `logical_m`) | Those are all-reduce options. This op has no gather, a bf16 scatter and the default padding. |
| The bands do not match the next layer's shards | The bands split the padded M, with the padding in the last band or bands. Shard the padded sequence, or slice each band to its real rows. |
| `RuntimeError: gemm_ar self-test failed: the reduce-scatter returned ... where ... was expected` | The SDMA puts moved nothing: MORI built with `BUILD_CCO_SDMA=OFF`, or `MORI_ENABLE_SDMA=1` not set; `op.dev_comm.sdma_num_queue` is then 0. |
| `ccoMemAlloc failed`, `hsaKmtCreateQueueExt` failures, a refused scale layout or `float8_e4m3fnuz`, `max_shapes` exhausted, a hang | As for the [all-reduce](fused-gemm-ar.md#12-troubleshooting). |

## Further reading

- The [`gemm_ar` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md),
  next to the code, has the full design, every measurement table and the
  negative results; its "Fused GEMM + reduce-scatter" section covers this op.
- [GEMM + all-reduce](fused-gemm-ar.md): the pipeline this op stops early, with
  the operand, scale and padding details it shares.
- [MORI-FUSED overview](MORI-CCO-FUSED-OPS-GUIDE.md): what the six operators
  share, and the headline numbers.
