# GEMM + all-to-all

`GemmA2AOp` (`mori.ops.gemm_a2a`) is an fp8 GEMM with a column-sharded
all-to-all fused into it. Every rank computes `C = A @ B.T` for its own rows
and receives, from every rank, the block of columns it owns. This is the QKV
projection under Ulysses sequence parallelism. The GEMM epilogue stages C per
destination, and the SDMA copy engines push each chunk as soon as its tiles are
done, while the GEMM computes the rest.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. What it computes](#1-what-it-computes)
- [2. How it works](#2-how-it-works)
- [3. Requirements](#3-requirements)
- [4. Constructor](#4-constructor): [window_bytes_for()](#window_bytes_for)
- [5. Calling the operator](#5-calling-the-operator)
- [6. Other methods](#6-other-methods): [padded_m()](#padded_m),
  [pad_rows()](#pad_rows), [self_test()](#self_test),
  [block_m_for() and chunks_for()](#block_m_for-and-chunks_for),
  [close()](#close-and-the-context-manager), [attributes](#attributes)
- [7. Shapes and limits](#7-shapes-and-limits)
- [8. Choosing settings](#8-choosing-settings)
- [9. Performance](#9-performance)
- [10. Complete example](#10-complete-example)
- [11. Testing and benchmarking](#11-testing-and-benchmarking)
- [12. Troubleshooting](#12-troubleshooting)
- [Further reading](#further-reading)

## Quick Reference

The snippet assumes the setup in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common):
FlyDSL and SDMA, a communicator whose VMM reservation is sized with
`window_bytes_for`, and one `self_test()` per operator.

```python
from mori.ops.gemm_a2a import GemmA2AOp
from mori.ops.gemm_ar import preshuffle_b

# comm: a mori.cco.Communicator with per_rank_vmm sized from
# GemmA2AOp.window_bytes_for(world, n=10240, m_max=16384), see section 10
with GemmA2AOp(comm, n=10240, k=8192, m_max=16384, quant="ptpc") as op:
    op.self_test()                                         # collective, once
    b_shuffled = preshuffle_b(w_qkv)                       # [N, K], same on every rank
    m = op.padded_m(x.shape[0])
    a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m)) # your quantiser
    qkv = op(a_fp8, b_shuffled, a_scale, b_scale)          # [P*m, N/P] bf16
```

| What | Import |
|---|---|
| Operator | `from mori.ops.gemm_a2a import GemmA2AOp` |
| Weight preshuffle (also re-exported by `mori.ops.gemm_a2a`) | `from mori.ops.gemm_ar import preshuffle_b` |
| mxfp8 A-scale layout | `from mori.ops.gemm_ar import preshuffle_a_scale` |
| Communicator | `from mori.cco import Communicator, UniqueId` |

## 1. What it computes

`P` is the number of ranks, `M` the rows each rank passes and `s = N/P` the
columns each rank receives. Rank `r` passes its own `A_r [M, K]` and the same
`B [N, K]` as every other rank, and computes `C_r = A_r @ B.T` (`[M, N]`, never
as one tensor). Column block `d` of `C_r` goes to rank `d`. Rank `d` gets
`[P*M, s]` bf16, source rank `r` at rows `[r*M, (r+1)*M)`, its own block
included. With ptpc, rank `d` gets, up to rounding:

```python
c = [(A[r].float() @ B.float().T) * a_scale[r][:, None] * b_scale[None, :] for r in range(P)]
out_d = torch.cat([c_r[:, d * s:(d + 1) * s] for c_r in c]).to(torch.bfloat16)  # [P*M, s]
```

Under Ulysses, `K` is the hidden size, `N` the QKV width, `s` one rank's head
shard and `M = S/P`, so a rank gets every token for its own heads. Output
column `i` comes from weight row `i`, so order the weight's rows per
destination: rank `j`'s Q, K and V heads in rows `[j*s, (j+1)*s)`. For
Llama-3.1-70B at P=8, `s = 1280` is 8 q heads, 1 k head and 1 v head of 128:
one whole GQA group per rank. `A` and `B` are `torch.float8_e4m3fn`; the output
is bf16.

## 2. How it works

The GEMM is `gemm_ar`'s 8-wave fp8 kernel; only where C goes is different.

1. **Staging.** Tiles are 256 columns wide and `s` is a whole number of tiles,
   so each tile belongs to one destination. The epilogue stores it into this
   rank's staging slab for that destination, `[P][M][s]` in the window, so a
   destination's bytes are one contiguous range: a copy engine cannot gather.
   This rank's own columns go straight into its receive region. Blocks visit
   the destinations round-robin from a rank-dependent start, so every link has
   bytes to move early.
2. **Chunks.** Each destination's slab is cut along M into `chunks` runs of row
   tiles. A counter per (destination, chunk) finds the block that finishes a
   chunk's last outstanding tile; that block posts one SDMA put of the chunk,
   and the copy engine moves it while the GEMM computes later tiles.
3. **Close.** A one-block kernel waits for this rank's puts to drain and meets
   every peer at a barrier. The receive region then holds all `P` blocks.

`mode="split"` runs the same GEMM into the same slabs, then one kernel that
posts one put per destination, drains them and meets the peers. Fusion cannot
hide the wait for the first chunk, the last chunk's transfer, or differences in
GEMM speed between ranks; the more GEMM follows the first chunk, the more of
the transfer it hides, so the gain grows with M. There is no LSA (peer-store)
transport: it lost to the copy engines on this operator and was removed.

**Receive reuse.** Each call posts a credit to every peer when it starts, and a
put to rank `d` waits for `d`'s credit for the same call. A rank that is a call
ahead therefore cannot overwrite a result a slower rank is still reading,
provided the reader is ordered before the slower rank's next call (section 5).

**The window** holds a control region of a few KiB (barrier flags, one set of
completion counters per distinct M, submit locks, credits), the staging slabs
`[P, m_max, s]` and the receive region `[P*m_max, s]`, both bf16: about
`4 * m_max * N` bytes, 640 MiB at `N=10240, m_max=16384`.

## 3. Requirements

The shared setup is in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common).
For this operator:

| Requirement | Detail |
|---|---|
| GPU | gfx950 (MI355X). The GEMM is `gemm_ar`'s FlyDSL 8-wave fp8 kernel, which is gfx950 only. |
| FlyDSL | `pip install 'amd_mori[flydsl]'`. `import mori.ops.gemm_a2a` works without it; importing `GemmA2AOp` raises `ModuleNotFoundError`. |
| SDMA | MORI built with `BUILD_CCO_SDMA=ON` (the default), and `MORI_ENABLE_SDMA=1` set before the communicator is created. The constructor raises if the communicator has fewer SDMA queues than `sdma_queues`; `self_test()` checks that bytes actually move. |
| Ranks | P from 2 to 8, all in one node (one LSA domain), one process per GPU. Set the device before constructing: the operator reads its CU count to pick the tile. |
| Communicator | A `mori.cco.Communicator` whose `per_rank_vmm` holds the window. The overview uses `2 * window_bytes_for(...) + (256 << 20)`; add the windows of any other operator on the same communicator. |

## 4. Constructor

```python
GemmA2AOp(comm, *, n, k, m_max, quant="ptpc", mode="fused",
          sdma_queues=1, max_shapes=8, chunks=None, war_credit=True)
```

Collective: every rank passes the same arguments. It allocates and zeroes the
window, creates a dev-comm and ends in a barrier; kernels compile on the first
call at each M. The weight is a call argument, so layers that share `(n, k)`
can share one instance.

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | *(required)* | Every rank in one node. Its VMM reservation must hold `window_bytes_for(...)`. |
| `n` | `int` | *(required)* | N, the full output width (the QKV width). A multiple of `P * 256`. |
| `k` | `int` | *(required)* | K, the hidden size. A multiple of 128, at least 256. |
| `m_max` | `int` | *(required)* | The largest per-rank M any call passes, at least 1. Rounded up to the M granule (128, or 256 for mxfp8). Sizes the window, which cannot grow. |
| `quant` | `str` | `"ptpc"` | `"ptpc"`: fp32 per-token A scale and per-channel B scale. `"mxfp8"`: ue8m0 scales, 1x32 blocks for A and 32x32 for B; raises the M granule to 256. |
| `mode` | `str` | `"fused"` | `"fused"`: puts posted from the GEMM epilogue, chunk by chunk. `"split"`: the GEMM, then one put per destination. Same window, same result. |
| `sdma_queues` | `int` | `1` | SDMA queues per peer, at least 1. The communicator's first dev-comm fixes the count, and the constructor raises if it has fewer. Leave it at 1 (section 8). |
| `max_shapes` | `int` | `8` | How many distinct padded M values the instance serves, at least 1. Each takes a counter set in the window; `self_test()` takes one for `m_max`. Pass the same value to `window_bytes_for`. |
| `chunks` | `int` or `None` | `None` | Puts per destination per call, at least 1. `None` picks per M (section 8). A number is rounded down per M to a divisor of the row tiles. No effect with `mode="split"`. Above 8 it enlarges the control region: pass the same value to `window_bytes_for`. |
| `war_credit` | `bool` | `True` | The receive-reuse credit. `False` exists only as the race test's negative control: a peer can then overwrite a result this rank has not read. |

### window_bytes_for()

```python
GemmA2AOp.window_bytes_for(world_size, *, n, m_max, quant="ptpc", max_shapes=8, chunks=None) -> int
```

Static. Returns the exact size the constructor allocates (`op.window_bytes`), so
the communicator can be sized before the operator exists. `world_size` is P
(the communicator's `nranks`); pass `n`, `m_max`, `quant`, `max_shapes` and
`chunks` as you pass them to the constructor, with the same defaults. `quant`
sets the M granule `m_max` is padded to, `max_shapes` the number of counter
sets and `chunks` their capacity, `max(8, chunks)`. `k`, `mode` and the tile do
not enter it. A bad `world_size` or `n` raises the constructor's `ValueError`.

## 5. Calling the operator

```python
out = op(a_fp8, b_preshuffled, a_scale, b_scale)   # [P*M, N/P] bf16
```

A call is collective (every rank calls, with the same M, in the same order) and
is issued on the current stream. There are no optional arguments: no `out=`
and no stream argument.

| Argument | Shape | dtype | Notes |
|---|---|---|---|
| `a_fp8` | `[M, K]` | `torch.float8_e4m3fn` | This rank's rows. M is a multiple of `op.m_granule`, at most `op.m_max`, the same on every rank: use `padded_m()` and `pad_rows()`. Copied if not contiguous. |
| `b_preshuffled` | `[N, K]` | `torch.float8_e4m3fn` | The whole weight through `preshuffle_b`, identical on every rank (not checked). |
| `a_scale` | ptpc: `[M]` or `[M, 1]`; mxfp8: flat, `M*K/128` elements | ptpc: `float32`; mxfp8: `int32` | ptpc: one dequantisation scale per row. mxfp8: `preshuffle_a_scale(e)` of the `[M, K/32]` ue8m0 bytes. Pass it contiguous. |
| `b_scale` | ptpc: `[N]` or `[1, N]`; mxfp8: `(N/32)*(K/32)` elements | ptpc: `float32`; mxfp8: `int32` | ptpc: one scale per output channel. mxfp8: the `[N/32, K/32]` ue8m0 bytes, K-block major and widened: `e.t().contiguous().to(torch.int32)`. |

All four must be on the GPU. FNUZ fp8 (`float8_e4m3fnuz`) is rejected.

**The return value** is a `[P*M, N/P]` bf16 view of the operator's window, rows
`[r*M, (r+1)*M)` from rank `r`. Every call at the same M returns the same
tensor, and this rank's next call, at any M, overwrites it. Whatever reads it
must be ordered before that call on the device: the same stream, or an event
the next call's stream waits on. Clone it to keep it longer. If you padded `m`
rows to `M`, each source's block ends in `M - m` padding rows, zero when your
quantiser maps a zero row to zeros with a finite scale. Drop them per source:
`out.view(P, M, -1)[:, :m]`.

The first call at each M compiles that M's kernels (FlyDSL JIT): warm every M
before timing or capturing a CUDA graph. A replayed call behaves like a fresh
one, because the call counter behind the credit and the barrier lives on the
device.

## 6. Other methods

### padded_m()

`op.padded_m(m: int) -> int` rounds the logical row count up to `op.m_granule`
(128, or 256 for mxfp8). Call it on each call's row count, then `pad_rows()`.
`ValueError` if `m < 1`. It does not check `m_max`.

### pad_rows()

`op.pad_rows(x: torch.Tensor, m_pad: int) -> torch.Tensor` zero-extends a 2-D
`[m, K]` tensor of any dtype to `[m_pad, K]`, in a buffer the operator reuses:
the next `pad_rows` overwrites the returned view. Pad before quantising, so the
scales cover the padded rows. `ValueError` if `x` is not 2-D or `m_pad < m`.

### self_test()

`op.self_test() -> None` is collective: call it on every rank once after
construction, before real calls. It writes a known value into every staging
slab, runs the split push at `m_max`, synchronises and checks every remote
block. A MORI built without `BUILD_CCO_SDMA`, or run without
`MORI_ENABLE_SDMA=1`, moves nothing; this then raises `RuntimeError` instead of
letting calls return stale data. It overwrites the staging and receive regions
and registers `m_max` as one of the `max_shapes` M values.

### block_m_for() and chunks_for()

`op.block_m_for(m: int) -> int` is the tile's M (128 or 256) and
`op.chunks_for(m: int) -> int` the puts per destination that a call at padded
`m` uses. Host-side only; use them to log or check what a shape runs at.

### close() and the context manager

`op.close() -> None` releases the window and its memory; `with GemmA2AOp(...)
as op:` calls it on exit. It is idempotent. Synchronise first if a call may
still be in flight. Afterwards calls and `self_test()` raise `RuntimeError`,
and tensors the operator returned point at released memory. The dev-comm it
created stays with the communicator until the communicator is destroyed. A
constructor that fails releases what it had allocated.

### Attributes

`op.m_max` is `m_max` rounded up to the granule, `op.m_granule` is 128 or 256,
`op.shard_n` is `N/P`, `op.window_bytes` equals `window_bytes_for(...)`, and
`op.rank` and `op.world_size` come from the communicator. The other
constructor arguments are kept under their own names.

## 7. Shapes and limits

| Constraint | Value | Error if violated |
|---|---|---|
| Ranks | P from 2 to 8 | `ValueError`: `world_size must be in [2, 8]` |
| One node | every rank in one LSA domain | `RuntimeError`: needs every rank in one LSA domain (one node) |
| N | a multiple of `P * 256` (2048 at P=8) | `ValueError`: `n=... must be a multiple of world_size*block_n=...` |
| K | a multiple of 128, at least 256 | `ValueError`: `k=... must be a multiple of 128`, or `k=... is below the minimum 256` |
| `quant`, `mode` | `ptpc` or `mxfp8`; `fused` or `split` | `ValueError`: `... is not one of ...` |
| `m_max`, `sdma_queues`, `max_shapes`, `chunks` | at least 1 (`chunks` may be `None`) | `ValueError`: `... must be >= 1` |
| SDMA queues on the communicator | at least `sdma_queues` | `RuntimeError`: `GemmA2AOp needs 1 SDMA queue(s) per peer but the communicator has 0: build mori with BUILD_CCO_SDMA=ON, set MORI_ENABLE_SDMA=1, ...` |
| VMM reservation | holds the window | `RuntimeError`: `ccoMemAlloc failed`, with `slot exhausted` in the log |
| M per call | a multiple of 128 (256 for mxfp8), at most `op.m_max` | `ValueError`: `M=... must be a multiple of 128; use padded_m() and pad_rows()`, or `M=... exceeds the window's m_max=...` |
| Distinct M per instance | at most `max_shapes` | `ValueError`: `this op already serves 8 distinct M values ...; Raise max_shapes or pad to fewer distinct M.` |
| `a_fp8`, `b_preshuffled` | `[M, K]` and `[N, K]`, `float8_e4m3fn`, on the GPU | `ValueError`: `a_fp8 must be (M, K), got ...`, `... must be one of ['torch.float8_e4m3fn'], got ...`, or `... must be on a GPU` |
| Scales | dtype and element count as in section 5, on the GPU | `ValueError`: `a_scale must be torch.float32, got ...`, `a_scale must have M elements, got ...`; mxfp8: `a_scale must be the flat buffer preshuffle_a_scale returns` |
| Instance open | not closed | `RuntimeError`: `this GemmA2AOp has been closed` |

At P=8 the QKV widths of Llama-3.1-8B, 70B and 405B, Qwen2.5-72B and Qwen3-32B
fit. Qwen3-235B-A22B (N=9216) does not: it has fewer KV heads than ranks, so
Ulysses at P=8 would have to replicate KV.

## 8. Choosing settings

| Setting | What the default does | When to change it |
|---|---|---|
| Tile (not a parameter; `block_m_for()` shows it) | mxfp8: 256x256, always (the packed A scale needs it). ptpc: 256x256 when M is a multiple of 256 and that grid is at least two waves, `(M/256) * (N/256) >= 2 * CUs` (MI355X has 256 CUs); else 128x256. | Only the benchmark's `--block-m` overrides it. The rule was fitted at P=8, where the pick had the faster fused path in all eight model cells. At P=4 (MiniMax-M3) 128x256 was fastest at every M, 1-7% ahead of the 256x256 the rule picks at M=4096 and 8192. |
| `chunks=None` | The largest count up to 8 that divides the row tiles and keeps every put at least 1 MiB, `(M/c) * (N/P) * 2` bytes; else 1. | With ptpc at P=8 the default is the measured best except 70B M=8192, where 4 and 8 are within 1%; with mxfp8, 70B did best with 4 at M=4096 and 8192, where the default is 8. At P=4, M=2048, `chunks=4` measured 231.8 µs against 242.4 µs for the default 8 (MiniMax-M3 dense). A request is capped at one row tile per chunk. |
| `mode="fused"` | Puts from the epilogue. | Keep it: it beat `split` in every measured cell. `split` is the baseline and a fallback. |
| `quant="ptpc"` | fp32 per-token and per-channel scales. | Use the format your checkpoint is in. At the same tile, mxfp8's GEMM is 13-22% slower than ptpc's and its best fused path 10-21% slower: its scale loads stall the tile prefetch. `blockscale` is not accepted. |
| `sdma_queues=1` | One queue per (source, destination) pair, so every peer already has its own. | Leave it. Asking for P makes P per peer and uses one; in a process that already holds SDMA engines, queue creation then fails (`hsaKmtCreateQueueExt`, `anvil.cpp`). |
| `m_max`, `max_shapes=8` | The window is about `4 * m_max * N` bytes and cannot grow; each distinct padded M is one counter set and one compile. | Set `m_max` to the largest S/P you serve. If M varies, pad to a few buckets, or raise `max_shapes`. |

The operator reads no environment variables itself. MORI needs
`MORI_ENABLE_SDMA=1`, read once when its context is created (with the
communicator), and `BUILD_CCO_SDMA=ON` at build time (the default; `OFF` builds
a library whose puts move nothing). `MORI_SOCKET_IFNAME` picks the interface for
the communicator's rendezvous; the tests set `lo`
([CCO guide](MORI-CCO-GUIDE.md#9-environment-variables)).

## 9. Performance

**Conditions.** 8x MI355X (gfx950), one otherwise idle node, 2026-10-28; fp8
e4m3 in, bf16 out; FlyDSL 0.4.0, torch 2.12.0+rocm7.14.0. Every case ran in one
process on one communicator (`sweep_inproc.py`), through the benchmark, which
builds the kernels the operator builds. A case replays its CUDA graph back to
back, 50 launches to a batch with an event between each, for at least 3 s (8 s
in the last round) and 100 launches. The number is the median launch-to-launch
gap, maximum over ranks. Three interleaved rounds, spread 0.0-3.3%; every
sample validated against a host reference on every source rank.

`split SDMA` is `mode="split"`. `split RCCL` is the same GEMM, then
`all_to_all_single`. `best fused` is `mode="fused"` at the better of 4 and 8
chunks (in parentheses). Times in µs; negative is faster.

| Shape (fp8 ptpc, P=8) | M (S) | Tile | GEMM only | split SDMA | split RCCL | best fused | vs split SDMA | vs RCCL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-70B / Qwen2.5-72B, N=10240 K=8192 | 2048 (16k) | 128x256 | 151 | 244 | 344 | **203** (c4) | -16.6% | -40.9% |
| | 4096 (32k) | 256x256 | 257 | 441 | 513 | **328** (c8) | -25.5% | -36.0% |
| | 8192 (64k) | 256x256 | 451 | 813 | 930 | **578** (c4) | -28.8% | -37.8% |
| | 16384 (128k) | 256x256 | 918 | 1619 | 1824 | **1138** (c8) | **-29.7%** | -37.6% |
| Llama-3.1-405B, N=18432 K=16384 | 2048 (16k) | 256x256 | 502 | 681 | 738 | **568** (c8) | -16.6% | -23.1% |
| | 4096 (32k) | 256x256 | 899 | 1204 | 1303 | **989** (c8) | -17.9% | -24.1% |
| | 8192 (64k) | 256x256 | 1699 | 2280 | 2457 | **1858** (c8) | -18.5% | -24.4% |
| | 16384 (128k) | 256x256 | 3383 | 4694 | 4998 | **3678** (c8) | **-21.6%** | -26.4% |

mxfp8 (256x256, same node and run, two rounds, spread 0.0-2.3%) gains 11-24%
over split SDMA at the same shapes; the README has its table.

**Against aiter #6689's fused `sp_qkv`, MiniMax-M3 QKV, P=4.** K=6144; dense
sends Q, K, V per destination (N=9216), indexer adds IQ and IK (N=10240; mori
replicates IK's weight rows per destination, 3.9% more GEMM work). aiter
474128f, r13 kernels, through `SpQkvLarge.run`; mori fp8 ptpc. GPUs 0-3 of one
8x MI355X node, ROCm 7.14 CI image, FlyDSL 0.4.0. Both sides: graph replay
back to back with an event between launches, median gap, maximum over ranks,
three interleaved rounds (spread at most 0.4%), all validated at relL2
1.66e-3. `128 c8`: 128x256 tile, 8 chunks.

| Case | M | aiter r13 | `GemmA2AOp` default | vs aiter | best arm | vs aiter |
|---|---:|---:|---|---:|---|---:|
| dense | 2048 | 247.0 | 242.4 `128 c8` | -1.8% | **231.8** `128 c4` | -6.2% |
| dense | 4096 | 439.3 | 431.0 `256 c8` | -1.9% | **404.4** `128 c8` | -7.9% |
| dense | 8192 | 778.1 | 751.7 `256 c8` | -3.4% | **747.2** `128 c16` | -4.0% |
| indexer | 2048 | 289.6 | 259.5 `128 c8` | -10.4% | **248.6** `128 c4` | -14.2% |
| indexer | 4096 | 475.2 | 467.4 `256 c8` | -1.6% | **438.8** `128 c8` | -7.7% |
| indexer | 8192 | 870.6 | 825.4 `256 c8` | -5.2% | **814.9** `128 c16` | -6.4% |

How to read them:

- Fusion wins in every measured cell, and its margin over split SDMA grows
  with M (70B: 16.6% to 29.7%). Nothing below M=2048 was measured.
- At large M the transfer is nearly hidden, which bounds the gain: at 405B
  M=16384 the exposed communication drops from 1372 to 176 µs (-87%), but the
  GEMM is 95% of the fused total, so end to end the gain is 21.6%.
- `split RCCL` runs the same GEMM over the same bytes in the same layout, so
  its column compares transports only. RCCL was last in every cell.
- At P=4 the defaults are 1.6-10.4% ahead of aiter but leave up to 6.6% against
  the best arm (M=4096): the tile rule was fitted at P=8.

## 10. Complete example

`quantize_per_token` stands for your own quantiser; MORI ships none. For ptpc
it returns `a_fp8 [m, K]` `float8_e4m3fn` and `a_scale [m]` fp32, with
`x[i] ~= a_fp8[i].float() * a_scale[i]`. For mxfp8 it returns the `[m, K/32]`
ue8m0 bytes, which go through `preshuffle_a_scale`.

```python
# qkv_a2a.py -- MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 qkv_a2a.py
import os

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.gemm_a2a import GemmA2AOp
from mori.ops.gemm_ar import preshuffle_b

N, K, M_MAX = 10240, 8192, 16384  # Llama-3.1-70B QKV; M_MAX is the largest S/P served
TOKENS = 4000                     # this rank's S/P here; padded to 4096

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="cpu:gloo")
rank, world = dist.get_rank(), dist.get_world_size()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)
vmm = 2 * GemmA2AOp.window_bytes_for(world, n=N, m_max=M_MAX, quant="ptpc") + (256 << 20)

with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with GemmA2AOp(comm, n=N, k=K, m_max=M_MAX, quant="ptpc") as op:
        op.self_test()  # collective; raises if SDMA moves nothing

        g = torch.Generator(device="cuda").manual_seed(0)  # the same weight on every rank
        w = (torch.randn(N, K, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
        w_scale = torch.full((N,), 0.01, device="cuda")    # [N] fp32, per output channel
        b = preshuffle_b(w)

        m = op.padded_m(TOKENS)
        x = torch.randn(TOKENS, K, device="cuda", dtype=torch.bfloat16)
        a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m))  # your quantiser
        out = op(a_fp8, b, a_scale, w_scale)       # the first call at this M compiles it
        qkv = out.view(world, m, -1)[:, :TOKENS]   # [P, TOKENS, N/P], padding dropped

        # CUDA graph: warm on a side stream, capture, then refill the static
        # inputs and replay. A replay is a call, so every rank replays.
        static_a, static_s = a_fp8.clone(), a_scale.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(static_a, b, static_s, w_scale)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(static_a, b, static_s, w_scale)

        for _ in range(3):
            x = torch.randn(TOKENS, K, device="cuda", dtype=torch.bfloat16)
            a_new, s_new = quantize_per_token(op.pad_rows(x, m))  # your quantiser
            static_a.copy_(a_new)
            static_s.copy_(s_new)
            graph.replay()
            qkv = out.view(world, m, -1)[:, :TOKENS].clone()  # same stream, before the next call
        torch.cuda.synchronize()  # nothing in flight before close()
dist.destroy_process_group()
```

## 11. Testing and benchmarking

From the repository root:

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest tests/python/cco/test_gemm_a2a.py      # window layout and shape rules, on the CPU
pytest tests/python/cco/test_gemm_a2a_op.py   # GemmA2AOp on 8 ranks
```

`test_gemm_a2a_op.py` needs 8 GPUs and FlyDSL, and skips without them. It runs
`self_test()`, both modes and both quantisations at two M values and through a
CUDA graph, the receive-reuse race (with `war_credit=False` as the negative
control) and the constructor's refusals, each through
`tests/python/cco/gemm_a2a_op_worker.py` under `torch.distributed.run`. CI runs
both files from `/tmp` with absolute paths, so the installed `mori` is tested.

`benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py` calls the kernel builders,
not `GemmA2AOp`. Its `--chunks` defaults to 1, so pass the count you want;
`--block-m 0` (the default) applies the operator's tile rule. `--mode` is
`gemm-only`, `gemm-staging`, `split-sdma`, `split-rccl`, `fused-sdma` or
`gemm-to-window`; other flags include `--quant`, `--chunk-plan`, `--timing
{sync,b2b,stream}`, `--timeline PATH` and `--no-war-credit`. The transfer modes
check what arrived from every source against a host reference; each run prints
one `RESULT_JSON` line. `sweep_inproc.py` runs a whole table in one process and
skips rows its `--out` already holds, so a stopped sweep resumes.

```bash
# one cell
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py \
  --mode fused-sdma --chunks 8 -m 16384 -n 18432 -k 16384 \
  --warmup 30 --iters 21 --phase-split
# the ptpc tables (mxfp8: --quant mxfp8 --rounds 2)
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/sweep_inproc.py --suite models --quant ptpc \
  --alt-tile --rounds 3 --out models.jsonl
# the MiniMax-M3 comparison, on four GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/gemm_a2a/sweep_inproc.py --suite minimax --rounds 3 \
  --out minimax.jsonl
```

Keep the node idle and interleave rounds: a neighbour on the box was worth 9%
on a pure GEMM and changed which chunk count won.

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError` for `flydsl` when importing `GemmA2AOp` | FlyDSL is not installed: `pip install 'amd_mori[flydsl]'`. |
| `RuntimeError: GemmA2AOp needs 1 SDMA queue(s) per peer but the communicator has 0` | MORI built without `BUILD_CCO_SDMA=ON`; or `MORI_ENABLE_SDMA=1` not set before the communicator was created; or the communicator's first dev-comm asked for fewer queues than `sdma_queues`. |
| `RuntimeError: gemm_a2a self-test failed: recv block from rank ... holds ..., expected ...` | The puts move nothing; same causes. Without this check the operator would return this rank's block plus stale data. |
| `RuntimeError: GemmA2AOp needs every rank in one LSA domain (one node)` | The ranks span nodes. The operator is single-node. |
| `RuntimeError: ccoMemAlloc failed` at construction, `slot exhausted` in the log | `per_rank_vmm` is too small. Size it from `window_bytes_for` with the constructor's settings, plus other operators' windows. |
| `ValueError: ... got torch.float8_e4m3fnuz` | The kernel reads OCP `float8_e4m3fn`; the same FNUZ byte means a different value, so FNUZ is rejected. |
| A launch fails in `hsaKmtCreateQueueExt` (`anvil.cpp`) | The previous run's SDMA queues are not reclaimed yet. Wait for its processes to exit and retry. Keep `sdma_queues=1`. |
| A call hangs, or rows land in the wrong place | The ranks did not make the same calls, with the same M, in the same order. Every call waits on every peer. |
| The result changes after you read it | It was read after this rank's next call, or on a stream not ordered before it. Read it on the same stream, or clone it. |
| Wrong values, no error | B differs between ranks; a scale is in the wrong layout (mxfp8 `a_scale` not through `preshuffle_a_scale`, `b_scale` not K-block major); or the scales were computed before padding. |
| Any other `ValueError` | An argument check; [7. Shapes and limits](#7-shapes-and-limits) lists them. |

## Further reading

- The [`gemm_a2a` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_a2a/README.md),
  next to the code: the full design, every measurement (the mxfp8 table, the
  tile table, the removed LSA transports), the negative results and what was
  not measured.
- [MORI-FUSED overview](MORI-CCO-FUSED-OPS-GUIDE.md): what the six operators
  share, and [Performance at a glance](MORI-CCO-FUSED-OPS-GUIDE.md#performance-at-a-glance).
- [All-to-all + GEMM](fused-a2a-gemm.md): the reverse operator, for the
  o-projection after Ulysses attention.
