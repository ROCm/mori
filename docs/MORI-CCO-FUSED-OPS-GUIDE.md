# MORI Fused GEMM + Collective Guide

MORI fuses six collective patterns with the GEMM next to them, in one node,
over the SDMA copy engines. Each one is a Python operator built on
[CCO](MORI-CCO-GUIDE.md) and the FlyDSL 8-wave GEMM. The operator overlaps the
transfer with the GEMM instead of running it before or after.

| Pattern | Operator | Module | Typical layer | Each rank passes | Each rank gets |
|---|---|---|---|---|---|
| GEMM + all-reduce | `GemmAllReduceOp` | `mori.ops.gemm_ar` | RowParallelLinear (TP) | `A [M, K]`, its `B [N, K]` shard | `[M, N]`, the sum over ranks |
| GEMM + reduce-scatter | `GemmReduceScatterOp` | `mori.ops.gemm_ar` | RowParallelLinear under sequence parallelism | `A [M, K]`, its `B [N, K]` shard | `[M/P, N]`, its row band of the sum |
| GEMM + all-to-all | `GemmA2AOp` | `mori.ops.gemm_a2a` | QKV projection under Ulysses SP | `A [M, K]`, the same `B [N, K]` | `[P*M, N/P]`, its head columns for every token |
| GEMM + all-gather | `GemmAgOp` | `mori.ops.gemm_ag` | a projection on a CP/SP shard whose output every rank needs | `A [M, K]`, the same `B [N, K]` | `[P*M, N]`, every rank's rows |
| all-gather + GEMM | `AgGemmOp` | `mori.ops.ag_gemm` | ColumnParallelLinear after sequence parallelism | `A [M, K]`, its `B [N, K]` shard | `[P*M, N]`: every token times its shard |
| all-to-all + GEMM | `A2aGemmOp` | `mori.ops.a2a_gemm` | o-projection under Ulysses SP | `[P*M, K/P]` (all tokens, its heads), the same `B [N, K]` | `[M, N]`, its tokens over all of K |

`P` is the number of ranks and `M` is rows per rank. Every operator also has a
`mode="split"` (GEMM and transfer run back to back over the same window) as a
baseline and a fallback. All measurements are on **MI355X (gfx950)**.

Each operator's README, next to its code, has the full design, every
measurement table and the negative results:
[`gemm_ar`](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md),
[`gemm_a2a`](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_a2a/README.md),
[`gemm_ag`](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ag/README.md),
[`ag_gemm`](https://github.com/ROCm/mori/blob/main/python/mori/ops/ag_gemm/README.md),
[`a2a_gemm`](https://github.com/ROCm/mori/blob/main/python/mori/ops/a2a_gemm/README.md).
This page covers what is common to all six, how to call each one, and the
headline numbers.

## Table of Contents

- [Performance at a glance](#performance-at-a-glance)
- [What every operator has in common](#what-every-operator-has-in-common)
- [GEMM + all-reduce and reduce-scatter](#gemm--all-reduce-and-reduce-scatter)
- [GEMM + all-to-all](#gemm--all-to-all)
- [GEMM + all-gather](#gemm--all-gather)
- [All-gather + GEMM](#all-gather--gemm)
- [All-to-all + GEMM](#all-to-all--gemm)
- [Running the tests and benchmarks](#running-the-tests-and-benchmarks)

## Performance at a glance

The best fused result at each operator's largest measured model shape, against
the unfused path a serving stack runs today: the GEMM, then the RCCL/NCCL
collective. Each row comes from its own operator's measurements, on its own
date and timing method (in the sections below). Read the rows separately and
do not average them. Times in µs; negative is faster.

| Operator | Shape (per rank) | Precision | P | Unfused (GEMM + RCCL) | Fused | Change |
|---|---|---|---:|---:|---:|---:|
| `GemmAllReduceOp` | DeepSeek-V4-Pro `wo_b`, M=16384, N=7168, K=2048 | fp8 blockscale, fp8 gather | 8 | 1419.5 | **979.3** | **-31%** |
| `GemmReduceScatterOp` | DeepSeek-V4.1-Flash `wo_b`, M=16384, N=5120, K=2048 | mxfp8 | 4 | 1065.3 | **834.5** | **-21.7%** |
| `GemmA2AOp` | Llama-3.1-70B QKV, M=16384, N=10240, K=8192 | fp8 ptpc | 8 | 1824 | **1138** | **-37.6%** |
| `GemmAgOp` | DeepSeek-V4-Pro `wkv_gate`, M=16384, N=2048, K=7168 | bf16 in, fp32 out | 8 | 3008.2 | **2527.7** | **-16.0%** |
| `AgGemmOp` | Kimi-K3 MLA, global M=32768, N=3840, K=7168 | fp8 ptpc | 8 | 1316.6 | **896.9** | **-31.9%** |
| `A2aGemmOp` | MiniMax-M3 o-proj, M=8192, N=K=6144 | fp8 ptpc | 4 | 479.0 | **300.2** | **-37.3%** |

The `GemmAllReduceOp` baseline is the model as it ships (GEMM, then an NCCL
all-reduce, measured in SGLang). The other rows' baselines are `split-rccl`,
measured in the same process as the fused path.

**The gain depends on M.** Fusion hides the transfer behind GEMM work that is
still to come, so it needs a GEMM long enough to hide behind. The operators
handle short GEMMs in different ways:

- `GemmAgOp` defaults to `mode="auto"`, which fuses only once a 256x256 tile's
  grid covers every CU and splits below that.
- `AgGemmOp` and `A2aGemmOp` pick fewer bands at small M.
- `GemmAllReduceOp` is fused by SGLang only above a per-wire M floor.

The per-operator sections give the small-M numbers.

## What every operator has in common

**Build and environment.** Every operator moves data with the SDMA copy
engines. That needs:

- a MORI built with `BUILD_CCO_SDMA=ON` (the default);
- `MORI_ENABLE_SDMA=1` at run time;
- every rank in one node (one LSA domain).

Setting the variable is not enough on a host library built without the flag:
that library has no SDMA queues, every put silently does nothing, and the
result is wrong rather than an error. Each operator's `self_test()` checks
this. It is a collective, so call it on every rank once after construction.
The constructors also refuse a communicator without the SDMA queues they need.
The standalone `Mxfp8GemmOp` / `Mxfp8GemvOp` move no data and need neither.

**A window sized up front.** Each operator owns a symmetric window inside the
communicator's VMM reservation, and the window cannot grow. Size the
reservation from the operator's static `window_bytes_for(...)`, called with
the same settings as the constructor, then construct with `m_max` set to the
largest per-rank M the deployment will see:

```python
import torch, torch.distributed as dist
from mori.cco import Communicator, UniqueId

rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

vmm = 2 * Op.window_bytes_for(world, ...) + (256 << 20)   # Op: any operator below
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with Op(comm, ..., m_max=M_MAX) as op:
        op.self_test()
        ...
```

| Operator | `window_bytes_for` |
|---|---|
| `GemmAllReduceOp` | `(world, m_max=, n=, k=, quant=, gather_dtype=, gather_transport=, scatter_dtype=, schedule=, ...)` |
| `GemmReduceScatterOp` | `(world, m_max=, n=, k=, quant=, block_m=, max_shapes=)` |
| `GemmA2AOp` | `(world, n=, m_max=, quant=, max_shapes=, chunks=)` |
| `GemmAgOp` | `(world, n=, m_max=, out_dtype=, max_chunks=)` |
| `AgGemmOp`, `A2aGemmOp` | `(world, k=, m_max=)` |

**Padding.** M has to be a multiple of a tile granule, and the same on every
rank. `op.padded_m(m)` gives the padded count. `op.pad_rows(x, m_pad)`
zero-extends into a reused buffer, where the operator has it. Pad *before*
quantising, so the scales are built over the padded rows. Padded rows come
back as zero rows; slice them off.

**The result aliases the operator's buffer** and the next call overwrites it.
Whatever reads it must be ordered before this rank's next call: on the same
stream, or behind an event the next call's stream waits on. Clone it to keep
it longer. `AgGemmOp` also takes `out=`. For all-gather and all-to-all, the
operators carry a receive-reuse credit or a closing barrier, so a rank that is
a call ahead cannot overwrite rows a slower rank is still reading. That
guarantee rests on this ordering rule.

**Collective contract.** Every rank constructs the operator and calls it the
same number of times, with the same M, in the same order. Work is issued on
the current stream. One instance serves any M up to `m_max`, in any order,
and compiles each M once. Warm every M before capturing a CUDA graph. Calls
replay from a graph like fresh launches.

**Cleanup.** `close()`, or leaving the `with` block, releases the window.
Synchronise first if a call may still be in flight. The communicator holds a
reference, so dropping the Python object does not free it.

**fp8 operands.** The fp8 operators take `torch.float8_e4m3fn` and reject
`float8_e4m3fnuz`. The weight goes in once through
`mori.ops.gemm_ar.preshuffle_b`. Scales follow the `quant` contract:
- `ptpc`: `a_scale [M]` and `b_scale [N]`, fp32.
- `blockscale`: A 1x128 and B 128x128, fp32. `a_scale` is `[M, K/128]`
  column-major, flat, or `[K/128, M]`.
- `mxfp8`: 32-wide ue8m0. `a_scale` goes through `preshuffle_a_scale`;
  `b_scale` is the `[N/32, K/32]` bytes K-block-major as int32.

## GEMM + all-reduce and reduce-scatter

`GemmAllReduceOp` writes the GEMM's C straight into the window. The copy
engine pushes each destination's slice as soon as its last tile is written,
so the reduce-scatter half of the all-reduce overlaps the rest of the GEMM.
The reduce and all-gather run after it. `GemmReduceScatterOp` is the same
pipeline stopped before the gather: rank `r` keeps rows `r*M/P` onward of the
sum, the dim-0 split `reduce_scatter_tensor` makes.

```python
from mori.ops.gemm_ar import GemmAllReduceOp, GemmReduceScatterOp, preshuffle_b

with GemmAllReduceOp(comm, n=7168, k=2048, m_max=16384, quant="blockscale") as op:
    b_shuffled = preshuffle_b(b_fp8)                       # once per weight
    m_pad = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))
    out = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]]   # [M, N] bf16

with GemmReduceScatterOp(comm, n=5120, k=2048, m_max=16384, quant="mxfp8") as op:
    band = op(a_fp8, b_shuffled, a_scale, b_scale)         # [m_pad / P, N]
```

- **Quantisation**: `quant="blockscale"` (the default), `"mxfp8"`,
  `"mxfp8_raw"` (row-major uint8 scales staged through LDS), `"ptpc"`.
- **fp8 on the wire** (all-reduce only): `gather_dtype="fp8"` sends the
  all-gather leg as e4m3 with one fp32 scale per row, halving its bytes.
  `gather_transport="lsa"` lets the CUs pull and widen in one step. This is
  lossy: relL2 2.5e-2 against 2.4e-3 with a bf16 wire.
- **Shapes**: K a multiple of 128 and at least 256; N a multiple of 256 (and
  of 1024 with an fp8 gather); 2 to 8 ranks.
- **`schedule="wo_b"`** selects the measured padding and chunk profile for the
  two `wo_b` targets below. Pass `logical_m=` to the call when using it.
- **Standalone GEMM**: `Mxfp8GemmOp` and `Mxfp8GemvOp` (decode, M ≤ 32) are
  the same mxfp8 GEMM with nothing fused, for a ColumnParallelLinear that has
  no collective to fuse with.

**GEMM + all-reduce**, `wo_b` at M=16384, maximum over ranks. µs. V4-Pro is
on 8x MI355X, median of 11. V4.1-Flash is on 4x MI355X, from the README's
fp8-wire comparison; its fused bf16-wire cell comes from a different run and is
left out here:

| | DeepSeek-V4-Pro (TP8, blockscale) | DeepSeek-V4.1-Flash (TP4, mxfp8) |
|---|---:|---:|
| split SDMA, bf16 wire | 1472.9 | 1592.5 |
| fused, bf16 wire | 1148.8 | — |
| fused, fp8 wire | **979.3** | **1192.1** (over LSA) |
| GEMM + NCCL, as the model ships | 1419.5 | — |

The two models do not reach the same conclusion. On V4-Pro the overlap
dominates. On V4.1-Flash the GEMM is 178 µs against 1414 µs of communication,
so most of the gain comes from the fp8 wire (-15.6% on its own) and the mxfp8
GEMM, not from fusing. In an SGLang server, V4.1-Flash prefill throughput at
batch 4-16 rises **+5.6% to +6.0%** with the fused `wo_b` on the fp8 wire and
**+7.4% to +7.7%** with the standalone mxfp8 GEMM added. V4-Pro's GPU-busy time
over a profiled prefill falls **5.0%**.

**GEMM + reduce-scatter**, DeepSeek-V4.1-Flash `wo_b` (N=5120, K=2048, mxfp8)
on 4x MI355X. Graph replay, maximum over ranks, median of 3 interleaved
rounds. µs:

| M | GEMM only | GEMM + `reduce_scatter_tensor` | split SDMA | fused | vs RCCL |
|---:|---:|---:|---:|---:|---:|
| 2048 | 39.6 | 161.7 | 152.8 | 156.6 | -3.2% |
| 4096 | 66.4 | 295.4 | 269.9 | **262.9** | -11.0% |
| 8192 | 104.4 | 540.5 | 484.2 | **460.2** | -14.9% |
| 16384 | 174.9 | 1065.3 | 917.3 | **834.5** | **-21.7%** |

## GEMM + all-to-all

`GemmA2AOp`: every rank computes the whole `C = A @ B.T` and sends column
block `j` to rank `j`. Rank `d` gets `[P*M, N/P]`, source rank `r` at rows
`[r*M, (r+1)*M)`. This is a QKV projection under Ulysses sequence
parallelism: N is the QKV width, `N/P` one rank's head shard, and `M = S/P`.
The epilogue stages C per destination, and the copy engine pushes each chunk
as soon as its tiles are done.

```python
from mori.ops.gemm_a2a import GemmA2AOp
from mori.ops.gemm_ar import preshuffle_b

with GemmA2AOp(comm, n=10240, k=8192, m_max=16384, quant="ptpc") as op:
    op.self_test()
    b_shuffled = preshuffle_b(w_qkv)                       # [N, K], same on every rank
    m = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m))
    qkv = op(a_fp8, b_shuffled, a_scale, b_scale)          # [P*m, N/P] bf16
```

- **Quantisation**: `ptpc` or `mxfp8`.
- **Shapes**: N a multiple of `P * 256`, so each destination's columns are
  whole GEMM tiles. At P=8 that is N a multiple of 2048, which Llama-3.1-8B /
  70B / 405B, Qwen2.5-72B and Qwen3-32B satisfy. A model with fewer KV heads
  than ranks (Qwen3-235B at P=8) does not fit.
- `chunks=` pins the number of pushes per destination; the tile and the chunk
  count are otherwise picked from M.

8x MI355X, fp8 ptpc in, bf16 out. Every case ran in one process, with three
interleaved rounds and a median launch gap, maximum over ranks. µs:

| Shape | M (S) | GEMM only | split SDMA | split RCCL | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-70B, N=10240 K=8192 | 2048 (16k) | 151 | 244 | 344 | **203** | -16.6% | -40.9% |
| | 4096 (32k) | 257 | 441 | 513 | **328** | -25.5% | -36.0% |
| | 8192 (64k) | 451 | 813 | 930 | **578** | -28.8% | -37.8% |
| | 16384 (128k) | 918 | 1619 | 1824 | **1138** | **-29.7%** | -37.6% |
| Llama-3.1-405B, N=18432 K=16384 | 2048 (16k) | 502 | 681 | 738 | **568** | -16.6% | -23.1% |
| | 16384 (128k) | 3383 | 4694 | 4998 | **3678** | **-21.6%** | -26.4% |

The mxfp8 path gains 11-24% over split SDMA at the same shapes. On MiniMax-M3's
QKV at P=4, the operator's defaults are 1.6-10.4% ahead of aiter's fused
`sp_qkv` kernels.

## GEMM + all-gather

`GemmAgOp`: every rank computes `C = A @ B.T` on its own rows and every rank
receives `[P*M, N]`, source rank `r` at rows `[r*M, (r+1)*M)`. It was built
for DeepSeek-V4's compressor `wkv_gate`, a projection on a context-parallel
token shard whose output every rank consumes. The GEMM writes C into this
rank's receive slot, and its epilogue pushes each finished chunk of rows to
every peer.

```python
from mori.ops.gemm_ag import GemmAgOp

N, K, M_MAX = 2048, 7168, 16384
vmm = 2 * GemmAgOp.window_bytes_for(world, n=N, m_max=M_MAX, out_dtype="fp32") + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, in_dtype="bf16", out_dtype="fp32") as op:
        op.self_test()
        m = op.padded_m(x.shape[0])
        c = op(op.pad_rows(x, m), w)                       # [P*m, N] fp32
```

- **Precisions**: `in_dtype="bf16"` takes bf16 `a`/`b` (plain row-major
  weight), no scales, and returns bf16 or fp32. `in_dtype="fp8"` takes
  `quant` = `ptpc` / `blockscale` / `mxfp8` exactly as `GemmAllReduceOp` does,
  and returns bf16.
- **`mode="auto"`** (the default) fuses an M once `ceil(M/256) * ceil(N/256)`
  is at least the device's CU count, and splits below that, per M on one
  instance. `"fused"` and `"split"` pin a path.
- **Shapes**: N a multiple of 256. K a multiple of 128 (fp8, at least 256) or
  of 64 (bf16, at least 128). M a multiple of 128 (256 under mxfp8).

Through the operator, Pro `wkv_gate` (bf16 in, fp32 out). Graph-timed, maximum
over ranks. µs:

| P | M | `auto` | fused | split |
|---:|---:|---:|---:|---:|
| 8 | 2048 | **351.3** (split) | 362.8 | 351.3 |
| 8 | 4096 | **666.2** (split) | 685.5 | 666.1 |
| 8 | 8192 | **1239.2** (fused) | 1240.0 | 1288.7 |
| 4 | 8192 | **1231.3** (fused) | 1231.3 | 1282.2 |

Against Torch + RCCL on the same shapes (8x MI355X, three interleaved rounds),
the best MORI path is 10-16% faster on V4-Pro and 6-11% faster on V4.1-Flash.
For V4-Pro: M=2048 373.4 against 415.1 µs; M=16384 2527.7 against 3008.2 µs.
The steady part is the transport: SDMA's all-gather alone is about 13% faster
than RCCL's.

## All-gather + GEMM

`AgGemmOp`: sequence parallelism leaves every rank with its own tokens
`A_r [M, K]`, and a ColumnParallelLinear needs every rank's tokens times this
rank's weight shard: `C_r = concat_s(A_s) @ B_r.T`, `[P*M, N]`. The copy
engines push this rank's rows to every peer in row bands. The GEMM starts on
its own rows, and each remote tile starts as soon as the band it needs has
landed.

```python
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

N, K, M_MAX = 6288, 7168, 4096
vmm = 2 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
        op.self_test()
        b_shuffled = preshuffle_b(b_fp8)                   # this rank's [N, K] shard
        m = op.padded_m(x.shape[0])
        rows, scales = op.local_rows(m)                    # quantise straight into the window
        quantize_per_token_into(op.pad_rows(x, m), rows, scales)
        c = op(rows, b_shuffled, scales, b_scale)          # [P*m, N] bf16
```

- **Quantisation**: fp8 ptpc. The per-token A scale travels with its rows.
- **Shapes**: N any multiple of 16. The last column tile is masked, so a
  shard like 6288 goes in without padding the weight. M a multiple of 256.
- **Zero copy**: `local_rows(m)` is this rank's slot in the window. Quantising
  into it skips the copy into the window, and `out=` skips the copy out.
- **Bands**: by default the most bands per peer, up to 4, that still hold 20
  output tiles each. `chunks=` pins the count.

Kimi-K3 column-parallel projections at TP8 (K=7168), fp8 in, bf16 out, 8x
MI355X. Median of 2 interleaved rounds, maximum over ranks. µs:

| Projection | global M | split RCCL | split SDMA | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|
| KDA, N=6400 | 4096 | 257.5 | 258.7 | **237.9** | -8.0% | -7.6% |
| | 32768 | 1751.5 | 1693.1 | **1322.3** | -21.9% | -24.5% |
| MLA, N=3840 | 4096 | 188.2 | 178.8 | **176.8** | -1.1% | -6.1% |
| | 32768 | 1316.6 | 1212.5 | **896.9** | -26.0% | **-31.9%** |

## All-to-all + GEMM

`A2aGemmOp`: after Ulysses attention every rank holds every token for its own
head shard, `[P*M, K/P]`. The o-projection needs this rank's M tokens over all
of K. Block 0 of the GEMM posts every band to every peer. The GEMM starts on
this rank's own K segment, and each remote segment is read as soon as its band
of rows has landed. This is the reverse of `GemmA2AOp`.

```python
from mori.ops.a2a_gemm import A2aGemmOp
from mori.ops.gemm_ar import preshuffle_b

with A2aGemmOp(comm, n=N, k=K, m_max=max_tokens_per_rank) as op:
    op.self_test()
    b = preshuffle_b(w_o)                                  # [N, K], same on every rank
    m = op.padded_m(tokens_per_rank)
    a = op.send_buffer(m)                                  # [P*m, K/P] fp8, in the window
    quantize_attention_output_into(a)
    c = op(a, b, a_scale, b_scale)                         # [m, N] bf16
```

- **Quantisation**: fp8 ptpc. The per-token scale is taken over all heads
  before quantisation, from the row maxima gathered across ranks, so it is
  local and never travels.
- **Shapes**: N a multiple of 256; K a multiple of `P * 128`; M a multiple of
  256.
- **Bands and waits** are picked per M from the measurements below (`chunks=`
  and `wait=` override them). `send_buffer(m)` lets attention write its output
  straight into the window.

8x MI355X (4 for M3), fp8 ptpc in, bf16 out. Three interleaved rounds, median
launch gap, maximum over ranks. µs:

| Shape | M per rank | GEMM only | split RCCL | split SDMA | best fused | vs split SDMA | vs RCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M3 o-proj, P=4, N=K=6144 | 2048 | 63.0 | 144.0 | 132.9 | **121.7** | -8.5% | -15.5% |
| | 4096 | 125.1 | 264.9 | 244.0 | **184.1** | -24.5% | -30.5% |
| | 8192 | 208.6 | 479.0 | 438.7 | **300.2** | -31.6% | **-37.3%** |
| Llama-3.1-70B o-proj, P=8, N=K=8192 | 2048 | 94.3 | 156.2 | 145.8 | **134.4** | -7.8% | -14.0% |
| | 4096 | 182.6 | 294.6 | 269.6 | **258.6** | -4.1% | -12.2% |
| | 8192 | 356.4 | 552.5 | 506.4 | **435.2** | -14.0% | -21.2% |

Through the operator with its defaults, the 70B shape lands within 1.3% of the
best fused column (136.6 / 262.0 / 430.9 µs). At P=8 the fused path still
trails the GEMM alone by 40-80 µs. The first wave of tiles needs half of the
transferred bytes, and they arrive at the link's rate. Closing that gap needs
the bands posted before the GEMM starts, from attention's epilogue.

## Running the tests and benchmarks

Every operator has a numerics test of its kernels and an 8-rank test of the
public operator. CI runs both in the CCO job. They need the environment
described [above](#what-every-operator-has-in-common):

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_gemm_ar_op.py
pytest tests/python/cco/test_gemm_a2a.py tests/python/cco/test_gemm_a2a_op.py
pytest tests/python/cco/test_gemm_ag.py tests/python/cco/test_gemm_ag_op.py
pytest tests/python/cco/test_ag_gemm.py tests/python/cco/test_ag_gemm_op.py
pytest tests/python/cco/test_a2a_gemm.py tests/python/cco/test_a2a_gemm_op.py
```

Each benchmark under `benchmark/cco/flydsl/<op>/` runs every mode (`gemm-only`,
`split-rccl`, `split-sdma`, `fused-sdma`) against a reference and prints one
`RESULT_JSON` line per configuration. For example:

```bash
torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048
```

Each operator's README has the exact commands behind its tables. Before
trusting a number of your own:

- Keep the node otherwise idle.
- Interleave the configurations you compare across rounds.
- Time CUDA-graph replays of many calls, not one.
- Pin `--quant`.
- Allow about 10 s between back-to-back 8-rank runs, so the SDMA queues are
  released.

The `gemm_ar` README's
[Measurement traps](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md#measurement-traps)
lists the ones that have produced wrong numbers here.
