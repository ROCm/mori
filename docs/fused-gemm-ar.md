# GEMM + all-reduce

`GemmAllReduceOp` (in `mori.ops.gemm_ar`) multiplies each rank's activations by
its shard of a row-parallel weight and all-reduces the product, so every rank
gets the `[M, N]` sum in bf16. It serves a RowParallelLinear under tensor
parallelism, such as DeepSeek-V4's attention output projection `wo_b`. The GEMM
writes its C straight into the operator's window, and the copy engine pushes
each destination's slice as soon as its last tile is written, so the
reduce-scatter half of the all-reduce overlaps the rest of the GEMM. The reduce
and all-gather run after it. The same pipeline stopped before the gather is
[GEMM + reduce-scatter](fused-gemm-rs.md).

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
- [Standalone GEMM and GEMV](#standalone-gemm-and-gemv)
- [Further reading](#further-reading)

## Quick Reference

```python
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

# comm: a mori.cco.Communicator whose per_rank_vmm was sized from
# GemmAllReduceOp.window_bytes_for(world, **settings); see section 10.
settings = dict(n=7168, k=2048, m_max=16384, quant="blockscale")  # V4-Pro wo_b, TP8
with GemmAllReduceOp(comm, **settings) as op:
    op.self_test()                                       # collective, once per op
    b_shuffled = preshuffle_b(b_fp8)                     # [N, K] e4m3fn, once per weight
    m_pad = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))  # your quantiser
    out = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]].clone()  # [M, N] bf16
    torch.cuda.synchronize()                             # before close() frees the window
```

| What | Import |
|---|---|
| The operator | `from mori.ops.gemm_ar import GemmAllReduceOp` |
| Weight and mxfp8 A-scale layouts | `from mori.ops.gemm_ar import preshuffle_b, preshuffle_a_scale` |
| Shape predicate | `from mori.ops.gemm_ar import supports` |
| Communicator | `from mori.cco import Communicator, UniqueId` |
| Reduce-scatter variant | `from mori.ops.gemm_ar import GemmReduceScatterOp` ([its page](fused-gemm-rs.md)) |
| GEMM and GEMV with nothing fused | `from mori.ops.gemm_ar import Mxfp8GemmOp, Mxfp8GemvOp` |

## 1. What it computes

Rank `r` of `P` passes its activations `A_r` and its weight shard `B_r`, and
every rank receives

```text
out [M, N] bf16 = sum over r of  dequant(A_r [M, K]) @ dequant(B_r [N, K])^T
```

`K` is the per-rank K, the slice of the layer's input this rank holds; `N` is
not sharded. The GEMM applies the `quant` scales and accumulates in fp32, and
the reduce adds the `P` partials in fp32 and rounds once to bf16.

| | Shape | dtype |
|---|---|---|
| A (`a_fp8`) | `[M, K]`, `M = op.padded_m(rows)`, the same on every rank | `float8_e4m3fn`; bf16 with `quant="bf16"` |
| B (`b_preshuffled`) | `[N, K]` | `float8_e4m3fn` through `preshuffle_b`; with `quant="bf16"` the row-major bf16 weight |
| Scales | per `quant`, see [section 5](#5-calling-the-operator) | fp32, int32, uint8, or none |
| Result | `[M, N]` row-major; padded rows are zero | bf16 |

With bf16 wires every rank gets the same bytes; with `gather_dtype="fp8"` the
bands a rank receives come through e4m3 and its own band does not. The layer is
a RowParallelLinear under tensor parallelism: the measured ones are the `wo_b`
of DeepSeek-V4-Pro (TP8) and DeepSeek-V4.1-Flash (TP4) and Kimi-K3's MLP
`down_proj`; `quant="bf16"` serves layers a checkpoint keeps in BF16, such as
Kimi-K3's attention `o_proj`.

## 2. How it works

A call issues four or more kernels on the current stream:

1. **GEMM + scatter.** The GEMM writes C into the window. Its rows split into
   `P` bands, one per destination, and each band into chunks; when a chunk's
   last tile is written, the copy engine pushes the chunk into the
   destination's receive slot. Tiles are walked chunk by chunk with the
   destination rotated by rank, so every link starts early.
2. **Drain**: waits for this rank's pushes to land, then a cross-rank barrier.
3. **Reduce**: sums the `P` contributions to this rank's band.
4. **Gather**: sends the reduced band to every peer. An fp8 gather adds a
   quantise kernel, then a CU pull that widens on the way in (`"lsa"`) or a
   copy-engine push and a dequantise kernel (`"sdma"`).

Only the scatter overlaps the GEMM, so the gain depends on how much GEMM there
is to hide it behind. A destination's band is `M / (P * block_m)` tile rows,
which the default plan pushes in the largest number of chunks, up to 8, that
divides that count: more chunks start the first push earlier until the copy
engine's per-packet cost shows, and 8 is the measured knee. At `M = P * block_m`
there is one chunk, pushed after the GEMM, and nothing overlaps.

The window holds C for `m_max` rows, the `[m_max, N]` bf16 result, a `tmp` slot
for this rank's reduced slice (`m_max / P` rows), `P` receive slots of
`m_max / P` rows, an fp8 copy of the result when the gather is fp8,
and control words (barrier flags, a counter set per distinct M, submit locks),
all laid out for `m_max`. It is about 700 MiB per rank at `m_max=16384`,
N=7168, `P=8` with bf16 wires, and 812 MiB with an fp8 gather.

## 3. Requirements

| Requirement | Detail |
|---|---|
| GPU | MI355X (gfx950). The GEMM kernels are written for gfx950 only and every measurement is on MI355X. The constructor does not check the GPU; the only architecture check in the code is `schedule="wo_b"`, which keeps the default plan anywhere else. |
| FlyDSL | `pip install 'amd_mori[flydsl]'`. `import mori.ops.gemm_ar`, `preshuffle_b` and `preshuffle_a_scale` work without it; touching `GemmAllReduceOp` raises `ModuleNotFoundError`. |
| SDMA | MORI built with `BUILD_CCO_SDMA=ON` (the default) and `MORI_ENABLE_SDMA=1` at run time. The constructor does not check that the communicator has SDMA queues, so `self_test()` is the only guard. |
| Ranks | One node (one LSA domain), `P` = 2, 4 or 8. Every rank constructs the op and makes every call. |
| Memory | A `Communicator` whose `per_rank_vmm` is sized from `window_bytes_for`, called with the constructor's settings. Each op allocates its own window from that reservation, so add up the windows of every op alive at once. |

The rules this op shares with the other fused operators (padding, result
aliasing, the collective contract, cleanup, fp8 operand and scale layouts) are
in [What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common).

## 4. Constructor

```python
GemmAllReduceOp(comm, *, n, k, m_max, block_m=None, block_n=256, sdma_queues=1,
                gather_dtype="bf16", gather_transport="lsa", quant="blockscale",
                max_shapes=None, scatter_dtype="bf16", schedule="default")
```

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | *(required)* | The communicator; `P = comm.nranks`. The op does not take ownership. |
| `n` | `int` | *(required)* | Output width N: a multiple of 256, and of 1024 with `gather_dtype="fp8"`. |
| `k` | `int` | *(required)* | Per-rank K. fp8 quants: a multiple of 128, at least 256. `"bf16"`: a multiple of 64, at least 128. |
| `m_max` | `int` | *(required)* | Largest per-rank M the op will serve, at least 1. Rounded up to a multiple of `P * block_m` (`op.m_max` holds the rounded value). Sizes the window, which cannot grow. |
| `block_m` | `int \| None` | `None` | Tile rows, which are also the padding granule. `None` picks the quant's tile: 128 for `"blockscale"`, 256 otherwise. `"mxfp8"` and `"mxfp8_raw"` accept only 256, `"bf16"` 128 or 256, and `"blockscale"` any positive multiple of 128, though 128 is its measured tile (256x256 spills registers). |
| `block_n` | `int` | `256` | Tile columns. Only 256 is accepted: the epilogue's lane transpose pairs exactly two N-tiles. |
| `sdma_queues` | `int` | `1` | SDMA queues per (source, destination) pair, at least 1. One is all the pipeline uses; more only adds idle queues, and in a process that already holds SDMA queues their creation fails. The first dev-comm on a communicator fixes the count. |
| `gather_dtype` | `str` | `"bf16"` | Wire dtype of the all-gather leg: `"bf16"` (exact) or `"fp8"` (e4m3 with one fp32 scale per row, half the bytes; relL2 2.5e-2 against 2.4e-3 with a bf16 wire). |
| `gather_transport` | `str` | `"lsa"` | Who moves an fp8 gather: `"lsa"` (the CUs pull each peer's band over xGMI and widen it, one kernel) or `"sdma"` (the copy engines push, then a dequantise kernel). No effect with a bf16 gather. |
| `quant` | `str` | `"blockscale"` | Operand contract. `"blockscale"`: fp8, A 1x128 and B 128x128 fp32 scales (DeepSeek-V4-Pro). `"mxfp8"`: fp8, 32-wide ue8m0 scales, A's through `preshuffle_a_scale` (DeepSeek-V4.1-Flash). `"mxfp8_raw"`: the same numbers with the quantiser's row-major uint8 scales, staged through LDS. `"bf16"`: bf16 A and B, no scales, bf16 scatter wire. There is no `"ptpc"`. |
| `max_shapes` | `int \| None` | `None` | Most distinct M plans the instance serves, at least 1; each takes a counter set. `None` reserves one for every padded M up to `m_max`, plus the `wo_b` plans, so a legal call never runs out. |
| `scatter_dtype` | `str` | `"bf16"` | Wire dtype of the scatter leg: `"bf16"` or `"fp8"`, a lossy opt-in (the epilogue quantises with one fp32 scale per row and 256-column tile). Not with `quant="bf16"`. |
| `schedule` | `str` | `"default"` | `"default"` or `"wo_b"`, the measured MI355X padding and chunk profile for two `wo_b` targets ([section 8](#8-choosing-settings)). With `"wo_b"`, pass `logical_m=` on every call. |

Construction is collective. It raises `ValueError` for a setting outside these
rules ([section 7](#7-shapes-and-limits)) before allocating anything, and
`RuntimeError` from the communicator if the window does not fit; on failure it
releases whatever it had acquired.

### window_bytes_for()

```python
GemmAllReduceOp.window_bytes_for(world_size, *, m_max, n, block_m=None,
    gather_dtype="bf16", gather_transport="lsa", max_shapes=None, k=None,
    quant="blockscale", scatter_dtype="bf16", schedule="default") -> int
```

Static. The bytes the window will take, by the constructor's own planning rules:
`op.window_bytes` equals it for the same settings. `world_size` is
`comm.nranks`; the rest mean what they mean for the constructor, and `k` may be
omitted except with `schedule="wo_b"`. `block_n` and `sdma_queues` do not change
the window. It raises `ValueError` for an unknown `quant`, a `block_m` the quant
forbids, a non-positive size, or a world size or fp8-gather N the window cannot
take; N and K against the GEMM are the constructor's checks.

## 5. Calling the operator

```python
out = op(a_fp8, b_preshuffled, a_scale=None, b_scale=None, *, logical_m=None)
```

| Argument | Shape | dtype | Notes |
|---|---|---|---|
| `a_fp8` | `[M, K]` | `float8_e4m3fn` (bf16 with `quant="bf16"`) | `M = op.padded_m(rows)`, the same on every rank, at most `op.m_max`. On the GPU; a non-contiguous tensor is copied. |
| `b_preshuffled` | `[N, K]` | `float8_e4m3fn` (bf16 with `quant="bf16"`) | `preshuffle_b(w)`, once per weight (the byte order of aiter's `shuffle_weight(x, layout=(16, 16))`); for `"bf16"` the checkpoint's row-major weight, not shuffled. |
| `a_scale`, `b_scale` | per `quant`, below | per `quant`, below | On the GPU; `None` for `"bf16"`. |
| `logical_m` | | `int` | Keyword-only: the rows before padding. Selects the fitted plan with `schedule="wo_b"`; otherwise optional, and `op.padded_m(logical_m)` must equal M. |

| `quant` | `a_scale` | `b_scale` |
|---|---|---|
| `"blockscale"` | fp32, `M*K/128` values: `[M, K/128]` column-major (stride `(1, M)`), `[K/128, M]`, or flat with `(row, kb)` at `kb * M + row` | fp32 `[N/128, K/128]` row-major |
| `"mxfp8"` | int32, flat: `preshuffle_a_scale(a_exps)`, `a_exps` the `[M, K/32]` ue8m0 bytes | int32, flat: `w_exps.t().contiguous().to(torch.int32).reshape(-1)` from the `[N/32, K/32]` bytes |
| `"mxfp8_raw"` | uint8 `[M, K/32]` as the quantiser emits it, 2-D or flat | uint8 `[N/32, K/32]`, 2-D or flat |

A blockscale `a_scale` that is `[M, K/128]` row-major is refused as ambiguous:
pass `.reshape(-1)` for a K/128-major buffer (what
`aiter_per1x128_quant(transpose_scale=True)` returns) or `.t()` for a logical
row-major tensor.

**Returns** an `[M, N]` bf16 row-major tensor that is a view of the op's window,
holding the result once the work queued on the current stream has run. Every
M's result starts at the same address, so this rank's next call overwrites it
and `close()` frees it: order its readers before the next call (same stream, or
an event the next call's stream waits on), and `.clone()` it to keep it. Padded
rows are zero; slice `[:logical_m]`. The first call at a new M compiles that M's
kernels. One instance is not usable concurrently: the window, the kernel cache
and the padding buffer are shared.

## 6. Other methods

### padded_m()

`op.padded_m(m) -> int`: the physical rows for `m` logical rows, `m` rounded up
to a multiple of `P * block_m` or, with `schedule="wo_b"`, the fitted plan's
rows at a profiled M (8200 gives 9216 at V4-Pro TP8, 4200 gives 4224 at
V4.1-Flash TP4). It keeps no state. `ValueError` if `m < 1`. The module
function `padded_m(m, world_size, block_m=128)` applies the default rule.

### pad_rows()

`op.pad_rows(x, m_pad) -> Tensor`: zero-extends a 2-D `[M, K]` tensor to
`m_pad` rows in a buffer the op reuses and returns a view of it, which the next
`pad_rows` overwrites. Pad before quantising. The buffer is reallocated when a
call needs more rows or a different K, dtype or device, and a captured graph
keeps the old address, so run the largest M through it before capturing.
`ValueError` if `x` is not 2-D or `m_pad < M`.

### self_test()

`op.self_test(m=None)`, collective: writes `rank + 1` over the input region,
runs scatter, reduce and gather without a GEMM, and checks every output element
is `P * (P + 1) / 2` within 1%; otherwise it raises `RuntimeError` naming
`BUILD_CCO_SDMA` and `MORI_ENABLE_SDMA`. The logical `m` defaults to `m_max`,
which compiles what the first real call needs. Call it once on every rank after
construction and before any capture; it synchronises and overwrites the window.

### close() and the context manager

`op.close()` deregisters the window and frees its memory. It is idempotent, and
a closed op raises `RuntimeError`. Dropping the object is not enough, because
the communicator holds the handles; the dev-comm and its SDMA queues stay with
the communicator. Synchronise first if work may be in flight.
`with GemmAllReduceOp(...) as op:` calls `close()` on exit.

### Attributes

`window_bytes` (equal to `window_bytes_for`), `m_max` (padded) and the
constructor's settings are plain attributes, for reading. `dev_comm.sdma_num_queue`
is the number of SDMA queues per peer the communicator actually connected.

## 7. Shapes and limits

`supports(m, n, k, world_size, *, block_m=None, block_n=256, gather_dtype="bf16",
gather_transport="lsa", quant="blockscale", scatter_dtype="bf16",
schedule="default")` takes a logical M and the constructor's settings and
returns `False` where these rules fail, without constructing anything; it does
not apply the `block_n == 256` rule or the bf16-scatter rule. It says whether a
shape is expressible, not whether fusing it is faster.

| Constraint | Value | Error if violated |
|---|---|---|
| `quant` | `"blockscale"`, `"mxfp8"`, `"mxfp8_raw"`, `"bf16"` | `ValueError`: `quant=... is not one of [...]` |
| `block_m` | 256 for the mxfp8 quants; 128 or 256 for `"bf16"`; a positive multiple of 128 for `"blockscale"` | `ValueError`: `quant='mxfp8' requires block_m=256 ...`, `quant='bf16' requires block_m in (128, 256)`, or `unsupported tile: block_m=... must be a positive multiple of 128` |
| `block_n` | 256 | `ValueError`: `block_n=... must be 256: the permlane epilogue's lane transpose pairs exactly two N-tiles` |
| Wire and transport strings | dtypes `"bf16"`, `"fp8"`; transports `"lsa"`, `"sdma"` | `ValueError`: `scatter_dtype=... is not one of ...` (likewise the others) |
| Scatter wire with `quant="bf16"` | `"bf16"` | `ValueError`: `quant='bf16' supports only the bf16 scatter wire` |
| `schedule` | `"default"`, `"wo_b"` | `ValueError`: `unknown schedule=...; use 'default' or 'wo_b'` |
| `m_max`, `sdma_queues`, `max_shapes` | at least 1 | `ValueError`: `m_max and sdma_queues must both be >= 1`, `max_shapes must be >= 1` |
| K, fp8 quants | a multiple of 128, at least 256 | `ValueError`: `unsupported shape: K=... must be a multiple of 128 ...` or `... is below the minimum 256` |
| K, `"bf16"` | a multiple of 64, at least 128 | `ValueError`: `unsupported shape: K=... must be a multiple of 64 and >= 128` |
| N | a multiple of 256 | `ValueError`: `unsupported shape: N=... must be a multiple of block_n=256` |
| N, `gather_dtype="fp8"` | a multiple of 1024 | `ValueError`: `the fp8 wire gives one wave one row ... must be a multiple of 1024` |
| Ranks `P` | 2, 4 or 8 (in [2, 8] and dividing 512) | `ValueError`: `world_size must be in [2, 8]`, `threads (512) must be divisible by world_size ...` |
| Call M | `op.padded_m(logical)`, at most `op.m_max` | `ValueError`: `M=... does not match planned padded M=...; use padded_m()/pad_rows() and pass the original logical_m`, `M=... exceeds the window's m_max=...` |
| Distinct M plans | at most `max_shapes` | `ValueError`: `this op already serves ... distinct M values ...; raise max_shapes, or pad to fewer distinct M` |
| A and B | exact shape; dtype per `quant` (`float8_e4m3fnuz` refused); on the GPU | `ValueError`: `a_fp8 must be (M, K), got ...`, `a_fp8 must be one of ['torch.float8_e4m3fn'], got ...`, `... must be on a GPU` |
| Scales | dtype, element count and layout per `quant` | `ValueError`: `a_scale must be torch.float32, got ...`, `... must have ... elements`, `a_scale is (M, K/128) row-major, which is ambiguous ...`, `quant='bf16' takes no scales; pass None` |
| Lifetime | not closed | `RuntimeError`: `this GemmAllReduceOp has been closed` |
| Window | fits the free VMM reservation | `RuntimeError`: `ccoMemAlloc failed: ...` (logged: `slot exhausted ... Increase perRankVmmSize`) |

## 8. Choosing settings

| Setting | What the default picks | Change it when |
|---|---|---|
| `quant` | `"blockscale"` | the checkpoint is mxfp8 or BF16. `"mxfp8_raw"` takes the quantiser's bytes with no shuffle; GEMM-only it measured faster cold and at large M, a few percent slower hot at small M or short K, and the two are not compared at K of 4096 and more |
| `block_m` | the quant's tile: 128 for blockscale, 256 otherwise | rarely: blockscale takes other granule-legal tiles, and bf16 also takes 128, which doubles the grid at small M; the mxfp8 forms need 256 |
| Chunks | the largest divisor of the bands per destination, up to 8 | not a parameter; `schedule="wo_b"` changes it at measured M |
| `schedule` | `"default"`: pad to `P * block_m` | V4-Pro TP8 or V4.1-Flash TP4 `wo_b` on MI355X (below) |
| `gather_dtype` | `"bf16"`, exact | the accuracy budget allows relL2 about 2.5e-2 for a gather half the size |
| `gather_transport` | `"lsa"` | keep: the pull measured faster than the push on both targets |
| `scatter_dtype` | `"bf16"` | V4.1-Flash TP4, once accuracy is checked: faster there, slower on V4-Pro TP8 |
| `sdma_queues`, `max_shapes` | 1; every padded M up to `m_max` | keep |

**The fp8 gather** halves the bytes of the leg that runs at the xGMI ceiling. At
V4-Pro TP8, M=16384, the fused layer measured 957.3 µs with the LSA pull,
1018.9 µs with the SDMA push and 1150.7 µs on the bf16 wire; relL2 against an
fp32 reference went from 2.35e-3 to 2.49e-2. At the model output it was not
detectable: V4-Pro's perplexity on 10941 tokens was 3.2671, inside the 3.2547 to
3.2680 of three bf16 runs. No reasoning, code or maths benchmark has been run.

**`schedule="wo_b"`** applies only on gfx950, to two targets. At `P=8`, N=7168,
K=2048, `"blockscale"`, `block_m=128`, both wires bf16: logical M=8200 keeps
9216 rows and pushes two bands at a time, and M=11264 and 13312 push one band
at a time. At `P=4`, N=5120, K=2048, `"mxfp8"`, `block_m=256`: M=4200 and 8200
pad to a multiple of 64 (4224, 8256) and push one band at a time, with both
wires bf16, or with `scatter_dtype="fp8"` and a bf16 gather or an fp8 gather
over LSA; with both wires bf16, M=11264 also pushes two bands at a time. Every
other M, shape or setting keeps the default plan: it is a finite measured
profile, not an interpolation rule.

**Whether to fuse at your M.** At V4.1-Flash TP4 against the model's unfused
path ([section 9](#9-performance)), the fp8 wire is ahead from M=2048 in every
row, and the bf16 wire is behind at 1024, 2048, 4200 and 4700 and ahead from
7200. SGLang's integration fuses that layer from M=8192 on the bf16 wire and
from M=2048 on the fp8 wire, and V4-Pro's from M=4096. Below that, run the GEMM
(or [`Mxfp8GemmOp`](#standalone-gemm-and-gemv)) and your usual all-reduce.

| Environment variable | Default | Effect |
|---|---|---|
| `MORI_ENABLE_SDMA` | unset | Must be `1`. Without it the communicator creates no SDMA queues and every put moves nothing. |
| `MORI_GEMM_AR_PULL_BLOCKS` | 64 | Grid of the LSA pull, so only with an fp8 gather over `"lsa"`; read when an M is compiled. At V4-Pro TP8, M=16384, 48 to 80 blocks measured 959 to 963 µs, against 1261 at 16 and 1184 at 512. |
| `MORI_SOCKET_IFNAME` | first non-loopback NIC | Interface the communicator's bootstrap uses; the tests set `lo`. |

## 9. Performance

All from the [README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md),
on MI355X (gfx950), in µs, with the conditions given for each table.

`wo_b` at M=16384, maximum over ranks. V4-Pro is on 8x MI355X, median of 11.
V4.1-Flash is on 4x MI355X, from the README's fp8-wire comparison; its fused
bf16-wire cell comes from a different run and is left out here:

| | DeepSeek-V4-Pro (TP8, blockscale) | DeepSeek-V4.1-Flash (TP4, mxfp8) |
|---|---:|---:|
| split SDMA, bf16 wire | 1472.9 | 1592.5 |
| fused, bf16 wire | 1148.8 | — |
| fused, fp8 wire | **979.3** | **1192.1** (over LSA) |
| GEMM + NCCL, as the model ships | 1419.5 | — |

At smaller M on V4-Pro the order holds: 398.1 split, 351.0 fused and 329.5
fused with the fp8 gather at M=4096; 722.0, 621.1 and 539.3 at M=8192.

V4.1-Flash TP4 against what the model runs today,
`mxfp8_native_blockscaled_linear` and then an all-reduce (the route is
`dot_scaled` below M=8192 and `hipblaslt_bf16` above). One M per process, two
runs; `fill` is M over the padded M; negative is fused faster:

| M | padded M | fill | today | fused, bf16 wire | fused, fp8 wire |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1.000 | 164.6 | +16.4% | +10.3% |
| 2048 | 2048 | 1.000 | 283.3 | +1.8% | -8.8% |
| 4096 | 4096 | 1.000 | 482.8 | -0.9% | -15.4% |
| 4200 | 5120 | 0.820 | 506.1 | +17.0% | -2.3% |
| 4700 | 5120 | 0.918 | 548.3 | +8.4% | -9.3% |
| 5120 | 5120 | 1.000 | 601.5 | -5.8% | -20.7% |
| 7200 | 8192 | 0.879 | 1051.2 | -17.6% | -32.7% |
| 8192 | 8192 | 1.000 | 1042.5 | -19.4% | -34.1% |
| 8200 | 9216 | 0.890 | 1185.1 | -20.8% | -35.7% |
| 9200 | 9216 | 0.998 | 1157.5 | -19.2% | -34.2% |
| 13000 | 13312 | 0.977 | 1547.8 | -8.5% | -25.3% |
| 16384 | 16384 | 1.000 | 1851.3 | -16.2% | -32.9% |

The opt-in settings: prequantised operands, 8 calls per graph, 21 replays, 3
alternating rounds (5 for the short tails). The 2026-09-29 runs used Torch 2.11,
ROCm 7.2.4 and FlyDSL 0.2.4; the 2026-09-30 timings are from prototype modules
since integrated into the op:

| Setting | Target | Date | Result |
|---|---|---|---|
| `schedule="wo_b"` | V4-Pro TP8 | 2026-09-30 | M=11264 1074.35 → 819.42 (-23.7%); M=13312 1274.86 → 957.80 (-24.9%); M=8200 728.82 → 702.48 |
| `schedule="wo_b"` | V4.1-Flash TP4 | 2026-09-29 | reduced padding: M=4200 533.56 → 460.27 (-13.7%); M=8200 894.46 → 830.37 (-7.2%) |
| `scatter_dtype="fp8"` | V4.1-Flash TP4, M=16384 | 2026-09-29 | -21.1% with a bf16 gather, -26.6% with the fp8 gather over LSA; relL2 2.62% and 3.48% |
| `scatter_dtype="fp8"` | V4-Pro TP8, M=16384 | 2026-09-29 | 3.0% and 2.5% slower |

- The two models do not reach the same conclusion. On V4-Pro the overlap
  dominates. On V4.1-Flash the GEMM is 178 µs against 1414 µs of communication,
  so most of the gain comes from the fp8 wire (-15.6% on its own) and the mxfp8
  GEMM, not from fusing.
- The fused cost follows the padded M while the baseline follows the true M, so
  one padded M can win at full fill and lose at low fill: read the whole curve.
  On Kimi-K3's MLP `down_proj` (blockscale, bf16 wires, TP8 and TP4, against
  the op's own split path) fusing loses 2.0-4.0% at M=1024 and gains 9.7-21.6%
  at M=16384.
- In an SGLang server, V4.1-Flash prefill throughput at batch 4-16 rises
  **+5.6% to +6.0%** with the fused `wo_b` on the fp8 wire and **+7.4% to
  +7.7%** with the standalone mxfp8 GEMM added. V4-Pro's GPU-busy time over a
  profiled prefill falls **5.0%**. Layer gains exceed these because `wo_b` is
  about 12% of the profile.

## 10. Complete example

Run with `MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 example.py`.
`quantize_per_1x128` stands for your quantiser; MORI ships none. It must return
`a_fp8` as `[m_pad, K]` `float8_e4m3fn` and `a_scale` as fp32 1x128 block scales:
`[m_pad, K/128]` column-major, `[K/128, m_pad]`, or flat
([section 5](#5-calling-the-operator)).

```python
import os
import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

# DeepSeek-V4-Pro wo_b at TP8; add gather_dtype="fp8" for the shorter, lossy gather.
SETTINGS = dict(n=7168, k=2048, m_max=16384, quant="blockscale")
N, K = SETTINGS["n"], SETTINGS["k"]

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="cpu:gloo")  # only carries the unique id
rank, world = dist.get_rank(), dist.get_world_size()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

vmm = 2 * GemmAllReduceOp.window_bytes_for(world, **SETTINGS) + (256 << 20)
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with GemmAllReduceOp(comm, **SETTINGS) as op:  # the same settings as the sizing
        op.self_test()  # every rank, once, before any capture

        # This rank's weight shard; stand-ins for the checkpoint's tensors.
        w_fp8 = (torch.randn(N, K, device="cuda") / 8).to(torch.float8_e4m3fn)
        b_scale = torch.full((N // 128, K // 128), 1e-2, device="cuda")
        b = preshuffle_b(w_fp8)

        def wo_b(x):  # x: [M, K] bf16, M <= m_max
            m = x.shape[0]
            a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, op.padded_m(m)))
            return op(a_fp8, b, a_scale, b_scale)[:m]

        # Warm every M you will run (the first call compiles it), largest first.
        for m in (16384, 5000):
            y = wo_b(torch.randn(m, K, device="cuda", dtype=torch.bfloat16)).clone()

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
            y_static = wo_b(x_static)  # a view of the window
        x_static.copy_(torch.randn_like(x_static))  # new data, same buffer
        graph.replay()  # on every rank, in the same order
        result = y_static.clone()

        torch.cuda.synchronize()  # nothing in flight when close() frees the window
dist.destroy_process_group()
```

## 11. Testing and benchmarking

```bash
# Host-only, single-GPU, 2-rank (public op), 8-rank (kernels) and TP8/TP4
# (wo_b schedule) tests; cases needing more GPUs than the node has are skipped
BUILD_CCO_SDMA=ON MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_flydsl_ar.py \
         tests/python/cco/test_gemm_ar_op.py tests/python/cco/test_gemm_ar_schedule.py

# DeepSeek-V4-Pro, TP8
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=8 benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048

# DeepSeek-V4.1-Flash, TP4, with the wire that wins there
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=4 benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant mxfp8 -m 16384 -n 5120 -k 2048 \
  --gather-dtype fp8 --gather-transport lsa --no-fuse-quantize

# Matrices, one point per process; the exit status counts failed points
python benchmark/cco/flydsl/gemm_ar/sweep.py --list
python benchmark/cco/flydsl/gemm_ar/sweep.py fused      # writes fused.jsonl
python benchmark/cco/flydsl/gemm_ar/report.py fused.jsonl
```

`tests/python/cco/test_gemm_ar_bench.py` runs the benchmark as a test, and
`gemm_ar_schedule_worker.py` beside it runs the `wo_b` regression directly.
`--mode` takes `gemm-only`, `split-rccl`, `split-sdma`, `fused-sdma`,
`split-lsa` or `fused-lsa`; each run prints a `RESULT_JSON` line with
`max_rank_time_us`, `rel_l2` and `validated`. Pin `--quant` (the default `ptpc`
is not an op setting), and note `--sdma-queues 8` and `--gather-transport sdma`
against the op's 1 and `lsa`. Allow about 10 s between back-to-back runs, and
read the README's
[Measurement traps](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md#measurement-traps)
before trusting a small-M number of your own.

## 12. Troubleshooting

| Symptom | What it means |
|---|---|
| `RuntimeError: gemm_ar self-test failed: the all-reduce returned ... where ... was expected` | The SDMA puts moved nothing: MORI was built with `BUILD_CCO_SDMA=OFF`, or `MORI_ENABLE_SDMA=1` is not set (the communicator also logs `no SDMA queues will be created`), and `op.dev_comm.sdma_num_queue` is 0. Without `self_test()` this is silent: the model still answers fluently and the fused path measures faster than it is. |
| `ModuleNotFoundError: No module named 'flydsl'` on first use of the op | Install the extra: `pip install 'amd_mori[flydsl]'`. |
| `RuntimeError: ccoMemAlloc failed` while constructing | The window does not fit the communicator's VMM reservation. Size `per_rank_vmm` for every op alive at once, or close ops you no longer need. |
| `hsaKmtCreateQueueExt` failures (`anvil.cpp`) at start-up | A previous run's ranks have not released their SDMA queues: wait about 10 s between runs (the tests wait 20 s) and check that no leftover process holds them. Keep `sdma_queues=1` in a process that already uses SDMA. |
| `ValueError: M=... does not match planned padded M=...` | M is not padded: use `op.padded_m()` and `op.pad_rows()`, and with `schedule="wo_b"` pass `logical_m=`. |
| `ValueError: a_scale is (M, K/128) row-major, which is ambiguous` | Say which you have: `.reshape(-1)` for a K/128-major buffer, `.t()` for a logical row-major tensor. Guessing wrong pairs every scale with the wrong K block (relL2 0.26). |
| `ValueError: ... must be one of ['torch.float8_e4m3fn'], got torch.float8_e4m3fnuz` | FNUZ has a different exponent bias; read as OCP e4m3 the product would be 4x too large. Quantise to `float8_e4m3fn`. |
| `ValueError: this op already serves ... distinct M values` | An explicit `max_shapes` is too small: leave it at `None`, or pad to fewer distinct M. |
| A call hangs | The phases carry device-side barriers: every rank must call with the same M, in the same order. Two unexplained hangs were seen at 8 ranks while a sweep was being set up; use `timeout` when sweeping. |
| `hipErrorIllegalAddress` under sustained load | Seen once, in one SGLang server out of nine, with the fused `wo_b` and the standalone GEMM both enabled; it has not reproduced and the cause is not established. |

## Standalone GEMM and GEMV

`Mxfp8GemmOp` is the same mxfp8 GEMM with nothing fused; `Mxfp8GemvOp` is a
separate skinny kernel for decode. Neither takes a communicator, a window or a
rank or needs SDMA (both need FlyDSL), and both return an ordinary tensor. They
are for a ColumnParallelLinear, which has no collective to fuse with, and for a
RowParallelLinear below the fusion's useful M.

| | `Mxfp8GemmOp(*, n, k, block_n=256, quant="mxfp8")` | `Mxfp8GemvOp(*, n, k)` |
|---|---|---|
| Call | `op(a_fp8, b_preshuffled, a_scale, b_scale, out=None)`, `[M, N]` bf16 | `op(x_fp8, w_preshuffled, x_scale, w_scale, out=None)`, `[M, N]` bf16 |
| M | `"mxfp8"`: a multiple of 64 (`op.padded_m()`, `op.pad_rows()` before quantising); `"mxfp8_raw"`: any | 1 to 32; above, `ValueError` |
| Weight | `preshuffle_b(w)`, shared with the GEMV, so shuffle once | the same tensor |
| Scales | as the fused op's `"mxfp8"` or `"mxfp8_raw"` ([section 5](#5-calling-the-operator)) | uint8 `[M, K/32]` and `[N/32, K/32]`, 2-D, as the checkpoint stores them |
| Shapes | K a multiple of 128, at least 256; N a multiple of 256; `block_n` 256 only; `quant` `"mxfp8"` or `"mxfp8_raw"` | K a multiple of 128; N a multiple of 32 |
| Predicate | `supports_gemm(n, k, *, block_n=256)` | `supports_gemv(n, k)` |
| Kernels | two: the 256-wide N tile, and a 128-wide one while the wide grid `ceil(M/256) * N/256` is under 140 workgroups (`op.wants_narrow_n(m)`) | one per M bucket (1, 2, 4, 8, 16, 32); tuned for `(8192, 1280)` and `(5120, 2048)`, a heuristic elsewhere |

`out=`, if given, must be `[M, N]` bf16. On MI355X at V4.1-Flash `wo_b`
(N=5120, K=2048), M=16384, bf16 in and out with quantisation, the GEMM measured
196.0 µs against 281.3 µs for bf16 (fake_quant + hipBLASLt), -30.3%, and 274.9
µs for SGLang's mxfp8 route (`tl.dot_scaled`); that was hot, one call per graph,
and `bench_gemm.py --scope linear` puts the comparison at -27%. Cold, across
V4.1-Flash's fp8 linear shapes against SGLang's native mxfp8 linear, what decides
it is the wide-tile grid `ceil(M/256) * N/256`: the median is +246.7% at 1-32
workgroups, -0.7% at 33-64, -10.2% at 65-128, -26.6% at 129-256 and -26.4%
above. On its two tuned shapes the GEMV beats SGLang's `mxfp8_gemv` on the same
fp8 bytes at M=1, 8 and 32; six untuned shapes land within ±2%, `wo_a` is
12-16% slower at M=8 and above, and `wq_b` at TP8 is slower at M=32.

## Further reading

- The [`gemm_ar` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md),
  next to the code, has the full design, every measurement table and the
  negative results.
- [MORI-FUSED overview](MORI-CCO-FUSED-OPS-GUIDE.md): what the six operators
  share, and the headline numbers.
