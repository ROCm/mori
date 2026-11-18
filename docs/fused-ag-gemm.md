# All-gather + GEMM

`AgGemmOp` (`mori.ops.ag_gemm`) computes `C_r = concat_s(A_s) @ B_r.T`: every
rank's tokens times this rank's weight shard, `[P*M, N]` in bf16. It serves a
ColumnParallelLinear after sequence parallelism, where each rank holds only its
own tokens. The copy engines push this rank's rows to every peer in row bands.
The GEMM starts on its own rows, and each remote tile starts as soon as the
band it needs has landed.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. What it computes](#1-what-it-computes)
- [2. How it works](#2-how-it-works)
- [3. Requirements](#3-requirements)
- [4. Constructor](#4-constructor)
- [5. Calling the operator](#5-calling-the-operator)
- [6. Other methods](#6-other-methods)
  - [padded_m()](#padded_m) · [pad_rows()](#pad_rows) · [local_rows()](#local_rows) · [chunks_for()](#chunks_for)
  - [self_test()](#self_test) · [close() and with](#close-and-with) · [Attributes](#attributes) · [supports() and shape_constraint()](#supports-and-shape_constraint)
- [7. Shapes and limits](#7-shapes-and-limits)
- [8. Choosing settings](#8-choosing-settings)
- [9. Performance](#9-performance)
- [10. Complete example](#10-complete-example)
- [11. Testing and benchmarking](#11-testing-and-benchmarking)
- [12. Troubleshooting](#12-troubleshooting)
- [Further reading](#further-reading)

## Quick Reference

```python
from mori.cco import Communicator
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

# world, rank and uid (a mori.cco.UniqueId) come from your launcher: see section 10.
N, K, M_MAX = 6288, 7168, 4096          # per rank: weight-shard rows, hidden size, most tokens
vmm = 2 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
        op.self_test()                                   # collective, once
        b_shuffled = preshuffle_b(b_fp8)                 # this rank's [N, K] shard, once
        m = op.padded_m(x.shape[0])                      # the same m on every rank
        a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m))  # your quantiser
        c = op(a_fp8, b_shuffled, a_scale, b_scale)      # [world*m, N] bf16
```

| What | Import |
|---|---|
| Operator | `from mori.ops.ag_gemm import AgGemmOp` |
| Weight layout | `from mori.ops.gemm_ar import preshuffle_b` |
| Communicator | `from mori.cco import Communicator, UniqueId` |
| Shape checks | `from mori.ops.ag_gemm import supports`, `from mori.ops.ag_gemm.op import shape_constraint` |
| Limits (8 ranks, 16 bands) | `from mori.ops.ag_gemm import MAX_WORLD, MAX_CHUNKS` |
| Kernel-level pieces the benchmark drives | `from mori.ops.ag_gemm import AgGemmConfig, compile_ag_gemm, build_phases` |

## 1. What it computes

Sequence parallelism leaves rank `s` with its own tokens `A_s [M, K]`. A
ColumnParallelLinear after it needs every rank's tokens times this rank's weight
shard `B_r [N, K]`. Unfused, that is an all-gather of the tokens, then the GEMM.

```text
C_r = concat_s(A_s) @ B_r.T                                      # [P*M, N]
C_r[s*M + i, j] = a_scale_s[i] * b_scale_r[j] * sum_k A_s[i, k] * B_r[j, k]
```

| Per rank | Shape | Dtype | Notes |
|---|---|---|---|
| Tokens `A_r` | `[M, K]` | `torch.float8_e4m3fn` | M padded to a multiple of 256, the same on every rank |
| Token scale `a_scale` | `[M]` | `torch.float32` | per token (ptpc); it travels with its rows |
| Weight shard `B_r` | `[N, K]` | `torch.float8_e4m3fn` | this rank's output columns, through `preshuffle_b` |
| Weight scale `b_scale` | `[N]` | `torch.float32` | per output channel |
| Result `C_r` | `[P*M, N]` | `torch.bfloat16` | contiguous; rows `[s*M, (s+1)*M)` are rank `s`'s tokens |

Every rank ends with all `P*M` tokens and its own N output columns, the layout a
column-parallel projection hands to the next tensor-parallel layer. The measured
case is Kimi-K3's column-parallel projections at TP8: K=7168, and N per rank
6288 (KDA) and 3648 (MLA).

## 2. How it works

The window holds the gathered operand, so the GEMM reads one `[P*M, K]` tensor
whatever rank a row came from:

```text
[ control | rows: [P][M][K] fp8 | scales: [P][M] fp32 ]
```

Slot `r` holds rank `r`'s rows. The call copies this rank's rows into its own
slot, unless the caller quantised straight into it with `local_rows()`. In the
default `mode="fused"`, a call is one GEMM kernel and one barrier kernel:

1. **Block 0 pushes.** One lane per peer pushes this rank's rows and scales to
   that peer in `chunks` row bands, and sets an arrival flag in the peer's
   window as each band lands. All peers are served at once; on MI355X each
   peer is its own xGMI link.
2. **Local tiles go first.** Tiles over this rank's own rows wait for nothing.
3. **A remote tile waits for one band.** The rest run in (band, source, row
   tile, column tile) order, with the source rotated by rank, so the first wave
   of remote tiles needs only each peer's band 0.
4. **A barrier closes every call.** It advances the epoch the flags are
   compared against, so nothing is ever reset. No rank starts its next call
   until every rank's GEMM has finished reading its slots.

`mode="split"` runs push, barrier, the same GEMM, barrier, and returns the same
bytes. More bands start the remote tiles earlier, but each band adds a wait
(section 8). The closing barrier is why every call is collective, why the next
call may use another M, and why a CUDA graph replays like a fresh launch.

## 3. Requirements

The operator follows the contract in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common)
(build and environment, window sizing, padding, result aliasing, the collective
contract, cleanup, fp8 operands). For this operator:

| Requirement | Detail |
|---|---|
| GPU | gfx950 (MI355X). The GEMM is `gemm_ar`'s 8-wave fp8 FlyDSL kernel, written for gfx950 only. |
| FlyDSL | `pip install 'amd_mori[flydsl]'`. `import mori.ops.ag_gemm` works without it; `AgGemmOp` and `supports` import FlyDSL on first access. |
| Build | MORI built with `BUILD_CCO_SDMA=ON` (the default). |
| Environment | `MORI_ENABLE_SDMA=1`, set before `Communicator.init`. |
| SDMA queues | `sdma_queues` per peer (default 1). The first device communicator created on a communicator fixes its queue count; there are at most 8 per GPU pair. |
| Ranks | P from 2 to 8, every rank in one node (one LSA domain). |
| VMM | `per_rank_vmm` holds the window of every operator on the communicator. |

Size the reservation from `window_bytes_for()` with the constructor's `k` and
`m_max`; the examples reserve twice the window plus 256 MiB. With several
operators on one communicator, add up their windows: the operator test, which
keeps two instances alive at once, reserves
`4 * window_bytes_for(...) + (256 << 20)`. The output buffer (section 5) is
ordinary PyTorch memory, outside the reservation.

## 4. Constructor

```python
op = AgGemmOp(comm, *, n, k, m_max, quant="ptpc", mode="fused", chunks=None, sdma_queues=1)
```

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | required | P is `comm.nranks`. |
| `n` | `int` | required | Rows of this rank's weight shard, which is its output columns. A positive multiple of 16. The last 256-column tile may be partial: past N it reads zeros and drops its stores, so a shard like 6288 goes in without padding the weight. |
| `k` | `int` | required | Hidden size. A multiple of 128, at least 256. |
| `m_max` | `int` | required | Largest per-rank M the instance will serve, at least 1. Rounded up to a multiple of 256 (`op.m_max`). It sizes the window, which cannot grow. |
| `quant` | `str` | `"ptpc"` | Only `"ptpc"`: fp8 with a per-token A scale and a per-channel B scale. blockscale and mxfp8 store the A scale K-block major, where a row band is strided, so they are refused. |
| `mode` | `str` | `"fused"` | `"fused"`: the push runs inside the GEMM. `"split"`: push, barrier, GEMM, barrier; the baseline and a fallback. |
| `chunks` | `int` or `None` | `None` | Row bands per peer in fused mode. `None` uses the default rule (section 8). An integer of at least 1 pins it: at each M, the largest divisor of the row tiles (M/256) up to `chunks`, which must come out at most 16 (`MAX_CHUNKS`). Split mode always pushes one band. |
| `sdma_queues` | `int` | `1` | SDMA queues to request per peer, at least 1. Peer `p` is served on queue `p % sdma_queues`. |

All arguments after `comm` are keyword-only. The constructor is collective:
every rank calls it with the same arguments. It allocates and zeroes the window,
checks the SDMA queues and the LSA domain, and ends in `comm.barrier()`. If it
raises after allocating, it releases the window first.

### window_bytes_for()

```python
nbytes = AgGemmOp.window_bytes_for(world_size, *, k, m_max)   # static
```

Returns the bytes the constructor will allocate, so the communicator can be
sized before the operator exists. `world_size` (`int`) is P, the communicator's
`nranks`; `k` and `m_max` (`int`) are the constructor's, and `m_max` is padded
to 256 the same way. The window is a 4 KiB control area, then `P * m_max * K`
bytes of fp8 rows and `P * m_max * 4` bytes of scales, each rounded up to
4 KiB. `n`, `mode`, `chunks` and `sdma_queues` do not change it.

## 5. Calling the operator

```python
c = op(a_fp8, b_preshuffled, a_scale, b_scale, out=None)
```

| Argument | Shape | Dtype | Notes |
|---|---|---|---|
| `a_fp8` | `[M, K]` | `torch.float8_e4m3fn` | This rank's tokens. M is `a_fp8.shape[0]`: a multiple of 256, at most `op.m_max`, the same on every rank. Copied into this rank's window slot unless it is that slot (`local_rows(M)[0]`), so any strides work. |
| `b_preshuffled` | `[N, K]` | `torch.float8_e4m3fn` | This rank's weight shard through `preshuffle_b`. A non-contiguous one is copied on every call; `preshuffle_b` returns a contiguous tensor. |
| `a_scale` | `[M]` or `[M, 1]` | `torch.float32` | Per-token scale, M elements. Copied into the window unless it is `local_rows(M)[1]`. |
| `b_scale` | `[N]` or `[1, N]` | `torch.float32` | Per-channel scale of this rank's shard, N elements. |
| `out` | `[P*M, N]` | `torch.bfloat16` | Optional, contiguous. The result is written there and `out` is returned. |

Every tensor must be on the GPU. The scales multiply back:
`A ≈ a_fp8.float() * a_scale[:, None]`, and likewise each row of B with
`b_scale`.

**Returns** `C_r`, `[P*M, N]` bf16, rows `[s*M, (s+1)*M)` from rank `s`.
Without `out=` it is a view of the operator's own output buffer, allocated once
at `P * m_max * N` bf16. That buffer is not the window, so the result survives
`close()`, but this rank's next call overwrites it: consume it on the same
stream first (or behind an event the next call waits on), or clone it. `out=`
saves that clone.

The work goes on the current stream with no host synchronisation. Every rank
makes the same calls, with the same M, in the same order. The first call at
each M compiles its kernels. Rows that `pad_rows()` zero-filled come back as
zero rows, provided your quantiser maps a zero row to zeros with a finite
scale. With `valid` tokens on every rank, keep `c.view(P, M, N)[:, :valid]`.

## 6. Other methods

### padded_m()

`op.padded_m(m) -> int` returns `m` (at least 1) rounded up to a multiple of
256: the M to pass for `m` tokens per rank. It does not check `m_max`. Every
rank must pass the same M, so if token counts differ, pad every rank to the
padded M of the largest.

### pad_rows()

`op.pad_rows(x, m_pad) -> Tensor` zero-extends a 2-D `x [m, K]` of any dtype to
`m_pad >= m` rows, in a buffer reused across calls, and returns a `[m_pad, K]`
view of it. The buffer grows when `m_pad` does, and the next `pad_rows()`
overwrites the view. Pad before quantising, so the scales cover the padded
rows. It copies even when `x` already has `m_pad` rows; skip it then.

### local_rows()

`op.local_rows(m) -> (rows, scales)` returns this rank's slot in the window at
`m`: `rows [m, K]` float8_e4m3fn and `scales [m]` fp32. Quantise straight into
them and pass them to the call, which then skips its copy into the window
(section 10 does this). Write into them on the stream you call the operator on,
after the previous call. They are valid until a call at another M, and not
after `close()`. `self_test()` overwrites the window, so run it first.
`local_rows()` at a new M sets that M up as a call would, and raises the same
errors.

### chunks_for()

`op.chunks_for(m) -> int` returns the row bands per peer that fused mode uses
at a padded M, from the default rule or the pinned `chunks` (section 8). Split
mode always uses one. It is arithmetic only: use it to log what the operator
does, or to pass the same count to the benchmark's `--chunks`.

### self_test()

`op.self_test()` checks that the transport moves bytes and raises
`RuntimeError` if it does not. A MORI built with `BUILD_CCO_SDMA=OFF`, or run
without `MORI_ENABLE_SDMA=1`, still launches every kernel, but its puts move
nothing and the GEMM reads whatever the slots held. The test fills this rank's
slot with `rank + 1`, runs the split push at `op.m_max`, and checks every
peer's slot. It is collective: call it on every rank, once, after construction.
All ranks finish checking before any of them raises.

### close() and with

`op.close()` releases the window, the compiled plans, and the output and
padding buffers. It is idempotent, and leaving a `with AgGemmOp(...) as op:`
block calls it. The device communicator is left to the communicator.
Synchronise first if a call may still be in flight, and do not replay a CUDA
graph that captured the operator once it is closed. Afterwards the call,
`local_rows()` and `self_test()` raise `RuntimeError`. The communicator holds a
reference to the window, so without `close()` it is freed only when the
communicator is destroyed.

### Attributes

The constructor arguments stay readable as `op.n`, `op.k`, `op.quant`,
`op.mode`, `op.chunks` (`None` under the default rule) and `op.sdma_queues`.
`op.m_max` is the padded value, the largest M a call accepts. `op.rank` and
`op.world_size` come from the communicator, and `op.window_bytes` is what
`window_bytes_for()` returned.

### supports() and shape_constraint()

`supports(n, k, world_size) -> bool` (from `mori.ops.ag_gemm`) is true when
`2 <= world_size <= 8` and `shape_constraint(n, k)` (from `mori.ops.ag_gemm.op`)
returns `None`; otherwise `shape_constraint` returns the reason as a string.
They say whether a shape is expressible, not whether fused is faster.

## 7. Shapes and limits

| Constraint | Value | Error if violated |
|---|---|---|
| Constructor arguments | `quant="ptpc"`; `mode` `"fused"` or `"split"`; `m_max`, `sdma_queues` and a pinned `chunks` at least 1 | `ValueError` at construction, naming the argument |
| K | a multiple of 128, at least 256 | `ValueError: unsupported shape: k=... must be a multiple of 128`, or `... below the minimum 256` |
| N | a positive multiple of 16 | `ValueError: unsupported shape: n=... must be a positive multiple of 16` |
| Ranks P | 2 to 8 (`MAX_WORLD`) | `ValueError: world_size must be in [2, 8]` at first use of any M |
| Bands after rounding | at most 16 (`MAX_CHUNKS`) | `ValueError: chunks must be in [1, 16]` at first use of that M |
| One node | every rank in one LSA domain | `RuntimeError: AgGemmOp needs every rank in one LSA domain (one node)` |
| SDMA queues | at least `sdma_queues` per peer on the communicator | `RuntimeError: AgGemmOp needs 1 SDMA queue(s) per peer but the communicator has 0: ...` |
| Window | fits in `per_rank_vmm` | `RuntimeError: ccoMemAlloc failed: -1`, with "slot exhausted" in the log |
| M per call | a multiple of 256, at most `op.m_max` | `ValueError: M=... must be a multiple of 256; use padded_m() and pad_rows()`, or `M=... exceeds the window's m_max=...` |
| M per call | the same on every rank | not checked: wrong results or a hang |
| `a_fp8`, `b_preshuffled` | `[M, K]` and `[N, K]`, `torch.float8_e4m3fn`, on the GPU | `ValueError: ... must be (..., ...)`, `must be one of ['torch.float8_e4m3fn']`, or `must be on a GPU` |
| `a_scale`, `b_scale` | `torch.float32`, M and N elements, on the GPU | `ValueError: ... must be torch.float32`, or `must have ... elements` |
| `out` | contiguous `[P*M, N]` bf16 on the GPU | `ValueError: out must be a (..., ...) bfloat16 tensor`, or `out must be a contiguous GPU tensor` |
| After `close()` | | `RuntimeError: this AgGemmOp has been closed` |

The first use of an M is the first call, `local_rows()` or `self_test()` at
that M; `self_test()` uses `op.m_max`.

## 8. Choosing settings

**Bands.** With `chunks=None`, fused mode takes the most bands per peer, up to
4 (`DEFAULT_CHUNKS`), whose band still holds 20 output tiles (`MIN_BAND_TILES`):
the largest divisor `d` of the row tiles `M/256` with `d <= 4` and
`(M/256/d) * ceil(N/256) >= 20`, else 1. A thinner band costs more in waits
than its earlier start buys. The rule sees N only as its count of 256-column
tiles, so 6288 behaves as 6400, and it does not depend on P.

| M per rank | Row tiles | N=6400 (25 column tiles) | N=3840 (15 column tiles) |
|---:|---:|---:|---:|
| 256 | 1 | 1 | 1 |
| 512 | 2 | 2 | 1 |
| 1024 | 4 | 4 | 2 |
| 2048 | 8 | 4 | 4 |
| 4096 | 16 | 4 | 4 |

The rule was fitted on Kimi-K3's TP8 projections at P=8, where it picks the
best measured count in every row of the tables in section 9. A pinned `chunks`
rounds down to a divisor of the row tiles: `chunks=4` at M=768 (3 row tiles)
gives 3, and `chunks=8` at M=1536 gives 6.

**When to override.**

- Pin `chunks` for a shape far from the measured ones: sweep the benchmark's
  `--chunks` at your M, N, K and P, and pin the winner. Above 4 is unmeasured.
- Use `mode="split"` as the baseline to measure fused against, or as a
  fallback; it returns the same bytes.
- Leave `sdma_queues=1`; the README has no measurement of other values.
- Set `m_max` to the largest per-rank M the deployment will see. It costs
  window (about `P * m_max * K` bytes) and output buffer (`P * m_max * N` bf16).

**Environment variables.** The operator reads none itself. MORI reads
`MORI_ENABLE_SDMA` once, when it sets up its context: without `1` the
communicator has no SDMA queues and the constructor refuses it.
`MORI_SOCKET_IFNAME` picks the interface for the communicator's rendezvous;
the tests and the benchmark use `lo` on one node.

## 9. Performance

Kimi-K3's column-parallel projections at TP8 (K=7168), fp8 ptpc in, bf16 out,
256x256 tile, on 8x MI355X (P=8) with ROCm 7.14, PyTorch 2.12 and FlyDSL
0.3.4.1. CUDA-graph replay, median of 51 after 20 warm-ups, maximum over ranks;
median of 2 interleaved rounds on an idle node (spread at most 1% except one
4.4% cell). All 92 runs validated at the fp8 floor (relL2 1.66e-3) against
every row of C for two operand sets. Global M is P times the M per rank. The
tables predate unpadded N, so they run the padded 6400 and 3840. `split-rccl`
is `all_gather_into_tensor` (rows, then scales) and the same GEMM; `split-sdma`
is push, barrier, GEMM, barrier; `c1`, `c2`, `c4` are fused with 1, 2 and 4
bands per peer. µs:

**N=6400 (KDA)**

| global M | gemm-only | split-rccl | split-sdma | fused c1 | c2 | c4 | best vs split-sdma | vs RCCL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 169.8 | 257.5 | 258.7 | 260.6 | **237.9** | - | -8.0% | 1.08x |
| 8192 | 321.6 | 478.7 | 462.6 | 427.9 | 385.0 | **377.9** | -18.3% | 1.27x |
| 16384 | 614.3 | 894.6 | 864.4 | 847.6 | 738.8 | **702.9** | -18.7% | 1.27x |
| 32768 | 1195.8 | 1751.5 | 1693.1 | 1628.1 | 1397.4 | **1322.3** | -21.9% | 1.32x |

**N=3840 (MLA)**

| global M | gemm-only | split-rccl | split-sdma | fused c1 | c2 | c4 | best vs split-sdma | vs RCCL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 104.0 | 188.2 | 178.8 | **176.8** | 190.4 | - | -1.1% | 1.06x |
| 8192 | 183.7 | 332.2 | 324.4 | 324.5 | **276.6** | 291.3 | -14.8% | 1.20x |
| 16384 | 344.1 | 646.6 | 618.0 | 629.2 | 527.7 | **486.0** | -21.4% | 1.33x |
| 32768 | 730.8 | 1316.6 | 1212.5 | 1182.1 | 985.5 | **896.9** | -26.0% | 1.47x |

c4 needs four row tiles per rank, so it starts at global M=8192. The same fused
c4 at Kimi-K3's real shards and at the padded widths, 2 interleaved rounds, all
validated, µs:

| global M | KDA 6288 | padded 6400 | MLA 3648 | padded 3840 |
|---:|---:|---:|---:|---:|
| 8192 | 374.8 | 373.3 | 271.6 | 283.3 |
| 32768 | 1300.3 | 1295.5 | 881.7 | 884.9 |

How to read them:

- **The bands are the mechanism.** With one band every remote tile waits for a
  peer's whole slab, so c1 stays close to split-sdma and is sometimes behind
  it. More bands start the first remote wave sooner; the best count rises
  with M.
- **The gain grows with M.** The best fused call takes 7.6% (KDA) and 6.1%
  (MLA) less time than split RCCL at global M=4096, and 24.5% and 31.9% less
  at 32768. Part of the transfer stays exposed: 10.6-22.7% over the bare GEMM
  at 32768, and up to 70% at 4096.
- **Unpadded N is not faster**, since the partial last tile still computes all
  256 columns. It buys the interface: no padded copy of the weight, and a
  contiguous `[P*M, 6288]` result instead of a slice of a 6400-wide one.

## 10. Complete example

```python
# MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo torchrun --standalone --nproc_per_node=8 example.py
import os

import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

N, K = 6288, 7168        # this rank's shard rows (KDA at TP8) and the hidden size
M_MAX = 4096             # the most tokens one rank will pass
TOKENS = 1000            # tokens per rank in the step captured below


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)

    vmm = 2 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
    with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
        with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
            op.self_test()                               # every rank, once
            w_fp8, w_scale = load_shard(rank)            # [N, K] float8_e4m3fn, [N] fp32
            b = preshuffle_b(w_fp8)                      # once per weight

            def column_parallel(x):                      # x: this rank's [tokens, K] bf16
                m = op.padded_m(x.shape[0])              # the same m on every rank
                rows, scales = op.local_rows(m)          # this rank's slot in the window
                quantize_per_token_into(op.pad_rows(x, m), rows, scales)  # your quantiser
                return op(rows, b, scales, w_scale)      # [world*m, N] bf16, no copy in

            # Warm up outside the graph: compiles this M and allocates the
            # operator's output and padding buffers. Every rank makes the same calls.
            x = next_activations(0)                      # static [TOKENS, K] bf16 input
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                column_parallel(x)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                c = column_parallel(x)                   # aliases the operator's buffer
            m = op.padded_m(TOKENS)
            for step in range(1, 4):
                x.copy_(next_activations(step))          # refill the captured input
                graph.replay()                           # every rank, in the same order
                consume(c.view(world, m, N)[:, :TOKENS]) # same stream, before the next replay

            torch.cuda.synchronize()                     # nothing in flight at close()
            del graph                                    # it addresses the window
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

The placeholders are yours; MORI ships no quantiser. `load_shard` returns this
rank's fp8 shard and its `[N]` fp32 per-channel scale. `quantize_per_token_into`
writes `rows` (`[m, K]` float8_e4m3fn) and `scales` (`[m]` fp32, one per token)
so that `src[i] ≈ rows[i].float() * scales[i]`, turning a zero row into zeros
with a finite scale. `next_activations` and `consume` stand for the layers
around this one. `thread_local` capture follows the benchmark: it lets RCCL's
watchdog thread query events during capture.

## 11. Testing and benchmarking

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest -q tests/python/cco/test_ag_gemm.py tests/python/cco/test_ag_gemm_op.py
```

`test_ag_gemm.py` checks the window layout, then runs the benchmark in every
mode against an fp32 reference, back to back. `test_ag_gemm_op.py` exercises
the public operator in one torchrun launch, from `self_test()` and padding to
`local_rows()` with `out=` and calls with one rank held. The GPU tests need at
least 2 GPUs and run on up to 8.

The benchmark drives the kernels through `compile_ag_gemm()` and
`build_phases()`, not through `AgGemmOp`. Its `--chunks` defaults to 1 and is
not rounded, so it must divide `m/256`. The README's command, for the global
M=32768 row of the KDA table (`-m` is per rank):

```bash
MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/ag_gemm/bench_ag_gemm.py --mode fused-sdma --chunks 4 \
  -m 4096 -N 6400 -k 7168 --back-to-back 8
```

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `fused-sdma` | `gemm-only`, `split-rccl`, `split-sdma` or `fused-sdma` |
| `-m`, `-N` (`--out-dim`), `-k` | 512, 6400, 7168 | rows per rank, N, K |
| `--chunks` | 1 | bands per peer, for `fused-sdma` |
| `--warmup`, `--iters` | 20, 51 | timing loop; the median is reported |
| `--eager` | off | time eager launches instead of CUDA-graph replays |
| `--back-to-back R` | 0 | R more calls with no host sync, a different rank held before each, each validated |
| `--skip-validation`, `--tolerance` | off, 3e-3 | skip the fp32 reference, or set its relL2 threshold |
| `--block-m`, `--waves-per-eu`, `--skew-cycles` | 256, 2, 50000000 | tile rows (only 256 is valid), the GEMM's waves per EU, the held rank's sleep |

Rank 0 prints one `RESULT_JSON` line (time as the maximum over ranks,
`rel_l2`, `validated`, `remote_bytes_per_rank`) and exits 1 if validation
fails. Before trusting a number of your own, read
[Running the tests and benchmarks](fused-tests-and-benchmarks.md).

## 12. Troubleshooting

| Error or symptom | What it means |
|---|---|
| `ModuleNotFoundError: No module named 'flydsl'` on `AgGemmOp` | Install the extra: `pip install 'amd_mori[flydsl]'`. |
| `RuntimeError: AgGemmOp needs 1 SDMA queue(s) per peer but the communicator has 0` | MORI was built without `BUILD_CCO_SDMA=ON`, `MORI_ENABLE_SDMA=1` was not set before `Communicator.init`, or the first device communicator on this communicator asked for fewer queues than `sdma_queues`. |
| `RuntimeError: ag_gemm self-test failed: the slot from rank S holds X, expected S+1` | The puts move nothing; check the same two settings. Without the test, calls would return wrong rows without an error. |
| `RuntimeError: AgGemmOp needs every rank in one LSA domain (one node)` | The ranks span nodes. The operator is single-node only. |
| `RuntimeError: ccoMemAlloc failed: -1`, "slot exhausted" in the log | `per_rank_vmm` has no room for the window. Size it from `window_bytes_for()`, adding up every operator on the communicator. |
| `ValueError: ... must be one of ['torch.float8_e4m3fn'], got torch.float8_e4m3fnuz` | Only OCP e4m3fn is accepted; the launch would reinterpret other fp8 bytes. |
| The GPU hangs in a call (the next synchronize never returns) | A rank did not make the same call: a different number of calls, a skipped warm-up or replay, a different M, or a dead rank. Every call ends in a cross-rank barrier that spins on the GPU with no timeout. |
| NaN or non-zero values in padding rows | The padded rows were not zero after quantising. Pad with `pad_rows()`, and make the quantiser turn a zero row into zeros with a finite scale. |
| Results change after a later call | The result is the operator's buffer and the next call overwrites it. Clone it, pass `out=`, or order the reader before the next call. |
| The first call at a new M is slow | Each M compiles once. Warm every M before timing or capturing. |
| A launch right after another 8-rank run fails in `hsaKmtCreateQueueExt` (`anvil.cpp`) | The SDMA queues of a finished process group take seconds to come back. Wait about 10 s between back-to-back runs. |

## Further reading

- The [`ag_gemm` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/ag_gemm/README.md),
  next to the code: the full design, every measurement, and the comparison with
  vllm-project/vllm#54151 (Kimi-K3 AG-GEMM on GB300).
- [MORI-FUSED overview](MORI-CCO-FUSED-OPS-GUIDE.md): the contract every
  operator shares and the headline numbers of all six.
