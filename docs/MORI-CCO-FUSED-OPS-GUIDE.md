# MORI-FUSED: Fused GEMM + Collective Operators

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

`P` is the number of ranks and `M` is rows per rank. Every operator except
`GemmAllReduceOp` and `GemmReduceScatterOp` also has a `mode="split"` (GEMM and
transfer run back to back over the same window) as a baseline and a fallback.
All measurements are on **MI355X (gfx950)**.

This page covers what the six have in common and the headline numbers. Each
operator has its own page in this section, with how to call it and its
measurements; its README, next to its code, has the full design, every
measurement table and the negative results.

## Table of Contents

- [Performance at a glance](#performance-at-a-glance)
- [What every operator has in common](#what-every-operator-has-in-common)
- [GEMM + all-reduce](fused-gemm-ar.md)
- [GEMM + reduce-scatter](fused-gemm-rs.md)
- [GEMM + all-to-all](fused-gemm-a2a.md)
- [GEMM + all-gather](fused-gemm-ag.md)
- [All-gather + GEMM](fused-ag-gemm.md)
- [All-to-all + GEMM](fused-a2a-gemm.md)
- [Running the tests and benchmarks](fused-tests-and-benchmarks.md)

## Performance at a glance

The best fused result at each operator's largest measured model shape, against
the unfused path a serving stack runs today: the GEMM, then the RCCL/NCCL
collective. Each row comes from its own operator's measurements, on its own
date and timing method (on the operator pages). Read the rows separately and
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

The operator pages give the small-M numbers.

## What every operator has in common

**Build and environment.** The GEMMs are FlyDSL kernels written for gfx950
(MI355X), so every operator needs FlyDSL, which is an optional extra
(`pip install 'amd_mori[flydsl]'`). Every operator moves data with the SDMA
copy engines (one exception: `GemmAllReduceOp`'s fp8 gather is, by default, a
pull by the CUs over LSA, `gather_transport="lsa"`). That needs:

- a MORI built with `BUILD_CCO_SDMA=ON` (the default);
- `MORI_ENABLE_SDMA=1` at run time;
- every rank in one node (one LSA domain).

Setting the variable is not enough on a host library built without the flag:
that library has no SDMA queues, every put silently does nothing, and the
result is wrong rather than an error. Each operator's `self_test()` checks
this. It is a collective, so call it on every rank once after construction.
The `GemmA2AOp`, `GemmAgOp`, `AgGemmOp` and `A2aGemmOp` constructors also
refuse a communicator without the SDMA queues they need; `GemmAllReduceOp`
and `GemmReduceScatterOp` do not check, so for them `self_test()` is the only
guard. The standalone `Mxfp8GemmOp` / `Mxfp8GemvOp` move no data and need no
SDMA.

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

vmm = 2 * Op.window_bytes_for(world, ...) + (256 << 20)   # Op: any of the six operators
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
`mori.ops.gemm_ar.preshuffle_b`. Quantising the activations is the caller's
job: the examples on the operator pages call `quantize_*` functions that stand for your own
quantiser and are not part of MORI. Scales follow the `quant` contract:
- `ptpc`: `a_scale [M]` and `b_scale [N]`, fp32.
- `blockscale`: A 1x128 and B 128x128, fp32. `a_scale` is `[M, K/128]`
  column-major, flat, or `[K/128, M]`.
- `mxfp8`: 32-wide ue8m0. `a_scale` goes through `preshuffle_a_scale`;
  `b_scale` is the `[N/32, K/32]` bytes K-block-major as int32.
