# All-to-all + GEMM

`A2aGemmOp` (`mori.ops.a2a_gemm`) fuses the all-to-all at the end of Ulysses
attention into the fp8 GEMM of the o-projection. Every rank passes its head
shard for every token, `[P*M, K/P]`, and the same weight `B [N, K]`, and gets
back `C [M, N]` in bf16: its own M tokens over all of K. Block 0 of the GEMM
posts every band of rows to every peer over the SDMA copy engines. The GEMM
starts on this rank's own K segment and reads each remote segment as soon as
its band has landed. This is the reverse of [`GemmA2AOp`](fused-gemm-a2a.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. What it computes](#1-what-it-computes)
- [2. How it works](#2-how-it-works)
- [3. Requirements](#3-requirements)
- [4. Constructor](#4-constructor)
  - [window_bytes_for()](#window_bytes_for)
- [5. Calling the operator](#5-calling-the-operator)
- [6. Other methods](#6-other-methods)
  - [padded_m()](#padded_m)
  - [send_buffer()](#send_buffer)
  - [self_test()](#self_test)
  - [chunks_for() and wait_for()](#chunks_for-and-wait_for)
  - [close() and the context manager](#close-and-the-context-manager)
  - [Attributes](#attributes)
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
from mori.cco import Communicator, UniqueId
from mori.ops.a2a_gemm import A2aGemmOp
from mori.ops.gemm_ar import preshuffle_b

# rank, world and uid (a mori.cco.UniqueId): see the complete example
vmm = 2 * A2aGemmOp.window_bytes_for(world, k=K, m_max=max_tokens_per_rank) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with A2aGemmOp(comm, n=N, k=K, m_max=max_tokens_per_rank) as op:
        op.self_test()                             # collective; raises if no bytes move
        b = preshuffle_b(w_o)                      # [N, K] fp8, the same on every rank
        m = op.padded_m(tokens_per_rank)
        a = op.send_buffer(m)                      # [P*m, K/P] fp8, in the window
        quantize_attention_output_into(a)          # your quantiser
        c = op(a, b, a_scale, b_scale)             # [m, N] bf16; a_scale [m], b_scale [N]
```

| What | Import |
|---|---|
| Operator | `from mori.ops.a2a_gemm import A2aGemmOp` |
| Weight layout | `from mori.ops.gemm_ar import preshuffle_b` |
| Communicator | `from mori.cco import Communicator, UniqueId` |
| Default bands and wait per M | `from mori.ops.a2a_gemm import default_chunks, default_wait` |
| Limits (`MAX_WORLD` 8, `MAX_CHUNKS` 16, `BLOCK_K` 128) | `from mori.ops.a2a_gemm import MAX_WORLD, MAX_CHUNKS, BLOCK_K` |
| Window layout and kernel builders, which the benchmark drives | `from mori.ops.a2a_gemm import A2aGemmConfig, compile_a2a_gemm, build_phases` |

## 1. What it computes

Ulysses sequence parallelism over P ranks ends attention with every rank
holding all `P*M` tokens for its own `K/P` columns of heads. The o-projection
needs this rank's M tokens over all of K, times the replicated weight
`B [N, K]`. Rank `d` computes

```text
C_d = concat_s(A_s[d*M:(d+1)*M]) @ B.T                  # [M, N]
C_d[i, j] = a_scale[i] * b_scale[j] * sum_k A[i, k] * B[j, k]
```

`A_s` is source rank `s`'s input. The concatenation is along K: source `s`
owns columns `[s*K/P, (s+1)*K/P)`, and `A` is the gathered `[M, K]` operand.
Unfused, this is `all_to_all_single` followed by the GEMM.

| Per rank | Shape | Dtype | Content |
|---|---|---|---|
| `a_fp8` (input) | `[P*M, K/P]` | `float8_e4m3fn` | Every token, this rank's head columns. Rows `[d*M, (d+1)*M)` are rank `d`'s tokens. |
| `b_preshuffled` | `[N, K]` | `float8_e4m3fn` | The o-projection weight through `preshuffle_b`, identical on every rank |
| `a_scale` | `[M]` | `float32` | One scale per token of this rank, over all of K |
| `b_scale` | `[N]` | `float32` | One scale per output channel |
| `C` (output) | `[M, N]` | `bfloat16` | This rank's tokens, rows `[rank*M, (rank+1)*M)` of the sequence |

fp8 ptpc is the only precision. The README measures the MiniMax-M3
o-projection at Ulysses SP4 and the Llama-3.1-70B o-projection at P=8.

## 2. How it works

The operator's window keeps what this rank sends and what it receives side by
side, in the same per-source layout:

```text
[ control | send: [P][M][K/P] fp8 | recv: [P][M][K/P] fp8 ]
```

`send[d]`, the rows bound for rank `d`, is one contiguous slab, and
`send_buffer(M)` is the whole send region as `[P*M, K/P]`. `recv[s]` receives
source `s`'s rows. The GEMM reads A one K segment at a time: `send[rank]` for
this rank's own columns, which never travel, and `recv[s]` for every other
source. One call with `mode="fused"`:

1. **Post.** In block 0 of the GEMM, one lane per peer pushes `send[peer]` in
   `chunks` row bands, back to back, on the copy engine. Each band bumps a
   counter at the receiver once its bytes have landed.
2. **Compute.** Tiles go band-major: the first tiles need only band 0 from
   each source, the next ones band 1, and so on. A tile starts on this rank's
   own K segment, then takes the other sources in rank-rotated order.
3. **Wait.** `wait="segment"`: each wave waits for a segment's band just
   before its first load of it, so the own segment overlaps the transfer.
   `wait="tile"`: one thread waits for the tile's band from every source
   before the tile starts. Block 0 takes the last tile, which needs the last
   band anyway.
4. **Close.** A barrier-only kernel ends every call, so no rank refills a
   peer's `recv` while that peer still reads it. It also advances the epoch
   and re-takes the counter snapshot the band waits count from, both kept in
   the window: the next call, at any M and band count, and a CUDA-graph replay
   start from there.

`mode="split"` runs a push kernel, a barrier, the GEMM with no waits and the
closing barrier, over the same window. For the settings this means:

- More bands let the first tiles start after less of the transfer, but each
  band has a fixed cost. On an idle GPU the README's 70B exchange moves
  ~47 GB/s per link in one band and ~33 GB/s in eight.
- With `wait="segment"`, what a tile can do before remote bytes arrive is its
  own segment: a quarter of K at P=4 and an eighth at P=8, too little at P=8
  to cover the wait for the first bands.

## 3. Requirements

The shared contract (build and environment, window sizing, padding, result
aliasing, the collective contract, cleanup, fp8 operands) is in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common).
For this operator:

| Requirement | Detail |
|---|---|
| GPU | gfx950 (MI355X). The GEMM is `gemm_ar`'s 8-wave fp8 FlyDSL kernel, written for gfx950 only, and every measurement is on MI355X. The operator does not check the architecture. |
| FlyDSL | `pip install 'amd_mori[flydsl]'` ([Install MORI](installation.md#optional-flydsl-integration)). `import mori.ops.a2a_gemm` works without it, for `A2aGemmConfig` and the constants. `A2aGemmOp`, `default_chunks`, `default_wait` and the kernel builders import FlyDSL. |
| SDMA | MORI built with `BUILD_CCO_SDMA=ON` (the default), and `MORI_ENABLE_SDMA=1` set before the communicator is created. The operator asks for one SDMA queue per peer, and its constructor raises if the communicator has none. `self_test()` checks that bytes actually move. |
| Ranks | P from 2 to 8, all in one node (one LSA domain). |
| Communicator | A `mori.cco.Communicator` whose `per_rank_vmm` (4 GiB by default) holds the window: `2 * m_max * K` bytes plus 4 KiB. Size it from `window_bytes_for()`. |

## 4. Constructor

```python
A2aGemmOp(comm, *, n, k, m_max, mode="fused", chunks=None, wait=None, acquire="inv")
```

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | required | The communicator every rank shares: 2 to 8 ranks in one node, with SDMA queues. Its `per_rank_vmm` must hold the window. |
| `n` | `int` | required | N, the output columns and the rows of `B [N, K]`. A multiple of 256. |
| `k` | `int` | required | K, the full reduction dimension (every head). A multiple of `P * 128`, at least 256. Each rank owns `K/P` columns (`op.k_shard`). |
| `m_max` | `int` | required | The largest M (tokens per rank) the operator will see, at least 1. Rounded up to a multiple of 256 (`op.m_max`). It sizes the window, which cannot grow. |
| `mode` | `str` | `"fused"` | `"fused"`: the GEMM posts the bands and each tile waits only for what it reads. `"split"`: push, barrier, GEMM, barrier over the same window; the baseline and a fallback. |
| `chunks` | `int` or `None` | `None` | Bands per peer per call, 1 to 16. `None` picks a count per M with `default_chunks`. A number is an upper bound: at each M the operator uses the largest divisor of `M/256` that does not exceed it. In `"split"` it sets the number of puts per peer. |
| `wait` | `str` or `None` | `None` | How a tile waits in `"fused"`: `"segment"` (each wave, before its first load of a remote segment) or `"tile"` (once per tile, for every source, before the tile starts). `None` picks per M with `default_wait`. No effect in `"split"`. |
| `acquire` | `str` | `"inv"` | What a tile does after a band has landed, in `"fused"`. `"inv"`: an acquire fence that invalidates the caches. `"none"`: nothing, which is correct only while cco windows are uncached, the default (not with `CCO_UNCACHED_WINDOW=0`). The README's tables use `"none"`. |

The constructor is collective: call it on every rank with the same arguments.
It checks its arguments before allocating, then allocates and zeroes the
window, creates a device communicator with one SDMA queue per peer, checks the
queues and the LSA domain, and snapshots the copy engines' signal counters
between two host barriers. One instance serves any M up to `m_max`, and every
layer with this N and K: the weight is a call argument.

### window_bytes_for()

`A2aGemmOp.window_bytes_for(world_size, *, k, m_max) -> int` is a static method.

| Name | Type | Description |
|---|---|---|
| `world_size` | `int` | P, the number of ranks |
| `k`, `m_max` | `int` | As passed to the constructor; `m_max` is rounded up to 256 the same way |

It returns the bytes the constructor allocates: 4 KiB of control plus
`2 * M * K`, with M the padded `m_max`, a send and a receive copy of the fp8
operand. The window does not depend on `n`, `mode`, `chunks`, `wait` or
`acquire`. The method does not validate its arguments. Size the communicator
with `per_rank_vmm = 2 * window_bytes_for(...) + (256 << 20)`, as the overview
and the tests do.

## 5. Calling the operator

```python
c = op(a_fp8, b_preshuffled, a_scale, b_scale)    # torch.Tensor [M, N] bf16
```

| Argument | Shape | Dtype | Description |
|---|---|---|---|
| `a_fp8` | `[P*M, K/P]`, 2-D | `torch.float8_e4m3fn`, on the GPU | Every token, this rank's `K/P` head columns. Rows `[d*M, (d+1)*M)` go to rank `d`. `M = rows / P` must be a multiple of 256 and at most `m_max`. Pass `op.send_buffer(M)` to skip the copy into the window; any other tensor is made contiguous and copied in. |
| `b_preshuffled` | `[N, K]` | `torch.float8_e4m3fn`, on the GPU | The replicated weight through `mori.ops.gemm_ar.preshuffle_b`, identical on every rank. Shuffle it once, at load time. |
| `a_scale` | `[M]` (any shape with M elements) | `torch.float32`, on the GPU | One scale per token of this rank, over all of K. See below. |
| `b_scale` | `[N]` (N elements) | `torch.float32`, on the GPU | One scale per output channel |

**The A scale covers all of K.** Each rank quantises only its own head
columns, but a token has one scale over every head, so every rank must
quantise token `t` with the same `scale[t]`. Compute it before quantising:
take each row's absolute maximum over this rank's heads (`[P*M]`), max-reduce
it across ranks, and derive the scale from that. Quantise
`a[t] = attn_out[t] / scale[t]` and pass this rank's own tokens,
`a_scale = scale[rank*M:(rank+1)*M]`. The scale never travels. A scale
computed per head shard gives wrong results without an error.
[Section 10](#10-complete-example) shows the computation.

**Returns** `C [M, N]` bf16 on the current GPU: this rank's tokens,
`[rank*M, (rank+1)*M)` of the sequence. It is the operator's buffer for that M
(one per M), and the next call with the same M writes it. Read it before this
rank's next call, on the same stream or behind an event, or `clone()` it.

**Ordering.** The call is a collective: every rank makes the same calls, with
the same M, in the same order. The work goes on the current stream: a copy
into the window unless `a_fp8` is `send_buffer(M)`, the GEMM (which posts the
bands) and the closing barrier. `"split"` runs a push and a barrier first. The
send region may be refilled on the same stream as soon as a call is issued:
the closing barrier holds every rank until every GEMM has finished reading.
The first call at each M compiles its kernels. There are no call-time
options: `mode`, `chunks`, `wait` and `acquire` are fixed at construction.

**Padding.** Pad per destination. Rank `d`'s block, rows `[d*M, (d+1)*M)`,
holds its real tokens first and zeros after them. Pad before computing the
scales. Zero rows come back as zero rows of C; slice `c[:tokens]`.

**CUDA graphs.** A captured call replays like a fresh call. Warm each M
eagerly first, then refill the captured `a_fp8` (or `send_buffer(M)`) and
`a_scale` in place before each replay, the same number of times on every rank.

## 6. Other methods

### padded_m()

`op.padded_m(m: int) -> int` rounds a per-rank token count up to a multiple of
256: the M to pass. It raises `ValueError` for `m < 1`. It does not check
`m_max`; the call does.

### send_buffer()

`op.send_buffer(m: int) -> torch.Tensor` returns the window's send region for
`M = m`, `[P*m, K/P]` `float8_e4m3fn`. Rows `[d*m, (d+1)*m)` go to rank `d`.
Quantise the attention output straight into it and pass the same tensor as
`a_fp8`: the call then skips its copy into the window. Views for different `m`
share the same memory. Write into it on the stream of the previous call, or
from a stream that waits on that one. `m` must be a multiple of 256 and at most
`m_max` (`ValueError`).

### self_test()

`op.self_test() -> None` is a collective check that the copy engines move
bytes. It fills every send slab with `rank + 1`, runs the split exchange at
`m_max`, and raises `RuntimeError` unless every slab received from a peer holds
that peer's value: without SDMA, every put returns and moves nothing. Call it
once on every rank after construction, outside CUDA-graph capture (it
synchronises and runs host barriers), and before writing into `send_buffer`,
which it overwrites.

### chunks_for() and wait_for()

`op.chunks_for(m: int) -> int` and `op.wait_for(m: int) -> str` return the
band count and wait mode a call at M uses. These are the constructor's `chunks`
(rounded down to a divisor of `M/256`) and `wait` if given, otherwise
`default_chunks(M, P)` and `default_wait(M, P)`. Log them to record what a
deployment runs with.

### close() and the context manager

`op.close() -> None` releases the window and its memory. It is idempotent, and
leaving a `with A2aGemmOp(...) as op:` block calls it. It does not
synchronise: call `torch.cuda.synchronize()` first if a call may still be in
flight. The device communicator the operator created stays with the
communicator until that is destroyed. After `close()`, calling the operator,
`send_buffer()` or `self_test()` raises `RuntimeError`. A constructor that
fails releases the window before it raises.

### Attributes

| Attribute | Meaning |
|---|---|
| `op.k_shard` | `K/P`, the head columns each rank owns |
| `op.m_max` | `m_max` rounded up to a multiple of 256 |
| `op.window_bytes` | The window's size, `window_bytes_for(P, k=k, m_max=m_max)` |
| `op.rank`, `op.world_size` | This rank and P, from the communicator |
| `op.n`, `op.k`, `op.mode`, `op.chunks`, `op.wait`, `op.acquire` | The constructor's arguments; `chunks` and `wait` stay `None` when defaulted |

## 7. Shapes and limits

| Constraint | Value | Error if violated |
|---|---|---|
| Ranks P | 2 to 8 | `ValueError`: world_size must be in [2, 8] |
| One node | `lsa_size == P` | `RuntimeError`: A2aGemmOp needs every rank in one LSA domain (one node) |
| SDMA queues | at least 1 per peer | `RuntimeError`: A2aGemmOp needs an SDMA queue per peer but the communicator has none: build mori with BUILD_CCO_SDMA=ON and set MORI_ENABLE_SDMA=1 |
| `n` | multiple of 256 | `ValueError`: n must be a multiple of 256 |
| `k` | multiple of `P * 128`, at least 256 | `ValueError`: K must be a multiple of world_size*128, or must be at least 256 |
| `m_max` | at least 1 | `ValueError`: m_max must be >= 1 |
| Window span | `2 * m_max * K` below 2 GiB | `ValueError`: send+recv span ... bytes; the A buffer descriptor takes 32-bit offsets |
| Per-rank VMM | holds the window | `RuntimeError`: ccoMemAlloc failed, with "slot exhausted ... Increase perRankVmmSize" in the log |
| `mode`, `wait`, `acquire` | the values in [section 4](#4-constructor) | `ValueError`: ... is not one of ... |
| `chunks` | `None` or 1 to 16 | `ValueError`: chunks must be in [1, 16] |
| `a_fp8` shape | 2-D `[P*M, K/P]` | `ValueError`: a_fp8 must be [world*M, K/P] |
| M per call | multiple of 256, at most `m_max` | `ValueError`: M must be a multiple of 256; use padded_m(), or M exceeds the window's m_max |
| `b_preshuffled` shape | `(N, K)` | `ValueError`: b_preshuffled must be (N, K) |
| fp8 operands | `torch.float8_e4m3fn`, not `fnuz` | `ValueError`: must be one of ['torch.float8_e4m3fn'] ...; the launch reinterprets it as raw bytes |
| Scales | `float32`, M and N elements | `ValueError`: must be float32, or must have M (N) elements |
| Device | every tensor on the GPU | `ValueError`: must be on a GPU |
| `padded_m(m)` | `m >= 1` | `ValueError`: m must be >= 1 |
| After `close()` | — | `RuntimeError`: this A2aGemmOp has been closed |

Not implemented: blockscale and mxfp8, N that is not a multiple of 256, a tile
other than 256x256, and a per-shard A scale.

## 8. Choosing settings

`default_chunks(m, world_size)` and `default_wait(m, world_size)` pick the
settings per M. They are fitted on the README's two shapes:

| P | Bands | Wait |
|---|---|---|
| 2 to 7 | `M // 2048`, at least 1: one band per 2048 rows | `"segment"` |
| 8 | 1 up to M = 2048, then `M // 1024` | `"segment"` below M = 4096, `"tile"` from 4096 |

The band count is capped at 16 and rounded down to a divisor of `M/256`. At
the six measured points the defaults pick the best cell of the sweep in
[section 9](#9-performance), except for the 70B shape at M = 4096. There they
pick `tile` c4 (259.0 µs) against the best, `tile` c2 (258.6 µs).

When to override:

- **Other shapes or rank counts.** Only P = 4 and P = 8 were measured; P = 2,
  3, 5, 6 and 7 use the P = 4 rule. Sweep `--chunks` and `--wait` with the
  benchmark ([section 11](#11-testing-and-benchmarking)) and pin the winners
  with `chunks=` and `wait=`.
- **`acquire="none"`** drops a cache invalidate after each wait. Every
  measurement here used it. It is correct while cco windows are uncached, the
  default; keep `"inv"` if `CCO_UNCACHED_WINDOW=0` is set.
- **`mode="split"`** is the baseline and a fallback. It pushes in
  `chunks_for(M)` bands; the README's `split-sdma` column pushes one.

The operator reads no environment variables itself. These affect it:

| Variable | Effect |
|---|---|
| `MORI_ENABLE_SDMA` | Must be `1` when the communicator is created; without it there are no SDMA queues and the constructor raises |
| `CCO_UNCACHED_WINDOW` | Unset (default): cco windows are uncached. `0`: cached memory, where `acquire="none"` is not safe |
| `MORI_SOCKET_IFNAME` | Network interface for the communicator's unique-id rendezvous; the tests and benchmarks use `lo` |

## 9. Performance

fp8 ptpc in, bf16 out, 256x256 tile. MI355X, one node, 2026-10-28, FlyDSL
0.4.0, torch 2.12.0+rocm7.14.0. Every case ran in one process on one
communicator and replayed its CUDA graph back to back, with an event between
launches, for at least 3 s. Each reports the median launch gap, maximum over
ranks; three rounds were interleaved (spread 0.0-2.0%). Every case was
validated against an fp32 reference at the fp8 floor (relL2 1.66e-3), with
`acquire="none"`. `gemm-only` is the GEMM over an already exchanged operand;
`split-rccl` is `all_to_all_single`, then the same GEMM; `split-sdma` is push
(one band), barrier, the same GEMM, barrier. Times in µs:

| Shape | M per rank | gemm-only | split-rccl | split-sdma | best fused | vs split-sdma | vs RCCL |
|---|---:|---:|---:|---:|---|---:|---:|
| MiniMax-M3 o-proj, P=4, N=K=6144 | 2048 | 63.0 | 144.0 | 132.9 | **121.7** `segment c1` | -8.5% | -15.5% |
| | 4096 | 125.1 | 264.9 | 244.0 | **184.1** `segment c2` | -24.5% | -30.5% |
| | 8192 | 208.6 | 479.0 | 438.7 | **300.2** `segment c4` | -31.6% | **-37.3%** |
| Llama-3.1-70B o-proj, P=8, N=K=8192 | 2048 | 94.3 | 156.2 | 145.8 | **134.4** `segment c1` | -7.8% | -14.0% |
| | 4096 | 182.6 | 294.6 | 269.6 | **258.6** `tile c2` | -4.1% | -12.2% |
| | 8192 | 356.4 | 552.5 | 506.4 | **435.2** `tile c8` | -14.0% | -21.2% |

Every fused configuration of the same run (`cN` is N bands):

| Shape | M per rank | wait | c1 | c2 | c4 | c8 |
|---|---:|---|---:|---:|---:|---:|
| M3 P=4 | 2048 | segment | **121.7** | 128.2 | 136.8 | 157.5 |
| | 2048 | tile | 128.8 | 136.6 | 144.6 | 165.5 |
| | 4096 | segment | 234.0 | **184.1** | 190.8 | 209.4 |
| | 4096 | tile | 240.7 | 188.1 | 198.8 | 217.0 |
| | 8192 | segment | 424.0 | 356.1 | **300.2** | 317.6 |
| | 8192 | tile | 434.7 | 363.7 | 307.6 | 326.2 |
| 70B P=8 | 2048 | segment | **134.4** | 139.0 | 149.0 | 176.0 |
| | 2048 | tile | 141.8 | 141.8 | 151.6 | 179.0 |
| | 4096 | segment | 262.4 | 259.7 | 261.4 | 263.0 |
| | 4096 | tile | 264.0 | **258.6** | 259.0 | 259.7 |
| | 8192 | segment | 515.2 | 499.8 | 466.0 | 448.3 |
| | 8192 | tile | 506.9 | 482.9 | 452.8 | **435.2** |

Through `A2aGemmOp` with its defaults (`acquire="none"`, graph replay, input
written into `send_buffer`), the 70B shape at P=8 measured 136.6 / 262.0 /
430.9 µs fused and 149.7 / 290.1 / 557.4 µs split, at M = 2048 / 4096 / 8192
per rank.

- **Fusing wins in all six cells**, by 4-32% against `split-sdma` and 12-37%
  against RCCL, most where the transfer is the larger half: at P=4 the
  exchange (`split-sdma` minus `gemm-only`) is 70-230 µs against a 63-209 µs
  GEMM.
- **The bands are the mechanism.** With one band a remote segment waits for a
  peer's whole slab; more bands let a wave start after a fraction of the
  transfer, and smaller bands lose bandwidth per put. At P=4 the best count
  rises with M: 1, 2, 4.
- **`segment` is best or within 0.5% in five of six cells**, and wins whenever
  the tiles fit in one wave (M = 2048). `tile` is ahead only at P=8 with
  M >= 4096, by 0.4-3%.
- **At P=8 the fused path still trails the GEMM alone by 40-80 µs.** The first
  wave of tiles needs half of the transferred bytes, and they arrive at the
  link's rate. Closing that gap needs the bands posted before the GEMM starts,
  from attention's epilogue, which is not done yet.

## 10. Complete example

```python
# o_proj.py: MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 o_proj.py
import os

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.a2a_gemm import A2aGemmOp
from mori.ops.gemm_ar import preshuffle_b

N = K = 8192                       # Llama-3.1-70B o-projection
M_MAX = 8192                       # most tokens per rank this deployment will see

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
rank, world = dist.get_rank(), dist.get_world_size()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

# Stand-in for the checkpoint's fp8 o_proj weight and per-channel scale.
g = torch.Generator(device="cuda").manual_seed(0)       # the same on every rank
w_o = (torch.randn(N, K, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
b_scale = torch.rand(N, generator=g, device="cuda") * 0.01 + 0.01      # [N] fp32
b = preshuffle_b(w_o)

tokens = 4000                                          # real tokens per rank
vmm = 2 * A2aGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with A2aGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
        op.self_test()
        m = op.padded_m(tokens)                        # 4096
        a = op.send_buffer(m)                          # [P*m, K/P] fp8, in the window
        a_scale = torch.empty(m, device="cuda")        # [m] fp32

        def prepare():
            # Stand-in for Ulysses attention: every token, this rank's heads.
            # Rank d's tokens are rows [d*m, d*m + tokens); the padding is zero.
            attn = torch.randn(world, m, op.k_shard, device="cuda", dtype=torch.bfloat16)
            attn[:, tokens:] = 0
            attn = attn.view(world * m, op.k_shard)
            amax = attn.abs().amax(dim=1).float()      # [P*m], over this rank's heads
            dist.all_reduce(amax, op=dist.ReduceOp.MAX)  # over every rank's heads
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            scale = (amax / fp8_max).clamp_min(1e-12)  # the same on every rank
            quantize_rows_into(attn, scale, out=a)     # your quantiser
            a_scale.copy_(scale[rank * m:(rank + 1) * m])

        prepare()
        side = torch.cuda.Stream()                     # eager call on a side stream: it
        side.wait_stream(torch.cuda.current_stream())  # compiles this M and warms it
        with torch.cuda.stream(side):
            c = op(a, b, a_scale, b_scale)             # [m, N] bf16
            y = c[:tokens].clone()
        torch.cuda.current_stream().wait_stream(side)

        # Capture, then refill and replay. thread_local: RCCL's watchdog thread
        # may query events while this one captures.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            c = op(a, b, a_scale, b_scale)
        for _ in range(3):
            prepare()
            graph.replay()
            y = c[:tokens].clone()                     # before the next call writes c
        torch.cuda.synchronize()                       # nothing in flight at close()
dist.destroy_process_group()
```

`quantize_rows_into` stands for your quantiser; MORI ships none. It must write
`a[t] = attn[t] / scale[t]` as `torch.float8_e4m3fn` for all `P*m` rows, with
the scale every rank shares (unfused:
`a.copy_((attn.float() / scale[:, None]).to(torch.float8_e4m3fn))`). `a_scale`
is this rank's `m` entries of that scale, fp32.

## 11. Testing and benchmarking

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest -q tests/python/cco/test_a2a_gemm.py tests/python/cco/test_a2a_gemm_op.py
```

- `test_a2a_gemm.py` checks the window layout on the CPU. On 2 to 8 GPUs it
  runs the benchmark over every mode, 1 and 2 bands, both waits and both
  `acquire` values with 8 back-to-back calls, then signal counters that wrap
  past 2^32 and band counts that change between calls.
- `test_a2a_gemm_op.py` needs 8 GPUs: `self_test()`, `"fused"` and `"split"`
  over several M on one instance (with `send_buffer` and graph replays),
  back-to-back calls with one rank 20 ms late, and the constructor's refusals,
  with SDMA on and off.

Allow about 10 s between back-to-back 8-rank runs so the SDMA queues are
released; `test_a2a_gemm_op.py` retries once when it hits this.

`bench_a2a_gemm.py` times `gemm-only`, `split-rccl`, `split-sdma` and
`fused-sdma`, driving the kernel builders directly rather than `A2aGemmOp`.
The MiniMax-M3 rows above came from this command; for the 70B rows, use 8 GPUs
with `--nproc_per_node=8 -n 8192 -k 8192`:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \
  torchrun --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/a2a_gemm/bench_a2a_gemm.py -m 2048 4096 8192 \
  -n 6144 -k 6144 --chunks 1 2 4 8 --wait segment tile --acquire none \
  --rounds 3 --out m3.jsonl
```

| Flag | Default | Meaning |
|---|---|---|
| `--modes` | all four | Any of `gemm-only`, `split-rccl`, `split-sdma`, `fused-sdma` |
| `-m`, `-n`, `-k` | `2048`, `6144`, `6144` | Tokens per rank after the exchange (several allowed), N and K |
| `--chunks`, `--wait` | `1 2 4`, `segment` | Band counts and waits for `fused-sdma`; a count is skipped unless M is a multiple of `256 * count`. `--wait none` (with `--no-post`) is a diagnostic and is not validated. |
| `--acquire` | `inv` | `inv` or `none` |
| `--rounds` | `1` | Rounds over every case, alternating forward and reverse order |
| `--warmup`, `--iters`, `--min-seconds` | `20`, `100`, `3.0` | Warm-up launches; the least number of launch gaps and seconds timed |
| `--back-to-back` | `0` | Adds that many calls with no host sync, one rank held before each, and validates every one |
| `--tolerance`, `--out` | `3e-3`, none | The largest relL2 that passes; a JSONL file rank 0 appends one record per case to |

Each case also prints a `RESULT_JSON` line, and the script exits 1 if any case
fails validation. `timeline_a2a_gemm.py`, in the same directory, records when
each band lands and when each tile starts and ends; its docstring has the
command.

## 12. Troubleshooting

| Error or symptom | Cause | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'flydsl'` on importing `A2aGemmOp` | FlyDSL is not installed | `pip install 'amd_mori[flydsl]'` |
| `RuntimeError: A2aGemmOp needs an SDMA queue per peer ...` | MORI built with `BUILD_CCO_SDMA=OFF`, `MORI_ENABLE_SDMA` not set when the communicator was created, or no peer reports SDMA. cco logs "no SDMA queues will be created". | Rebuild with `BUILD_CCO_SDMA=ON`; export `MORI_ENABLE_SDMA=1` before `Communicator.init` |
| `RuntimeError: a2a_gemm self-test failed: recv slab from rank s holds v, expected s+1` | The copy engines move no bytes | As for the SDMA error above |
| `RuntimeError: A2aGemmOp needs every rank in one LSA domain (one node)` | The ranks span nodes | Run every rank on one node |
| `RuntimeError: ccoMemAlloc failed`, "slot exhausted" in the log | `per_rank_vmm` is too small for the window | Size it from `window_bytes_for` with the same `k` and `m_max` |
| `ValueError: M=... exceeds the window's m_max=...` | M above the window | Construct with a larger `m_max`; the window cannot grow |
| `ValueError: a_fp8 must be [world*M, ...]` | `a_fp8` is `[M, K]`, or has the wrong width | Pass every token, with this rank's `K/P` columns |
| `ValueError: a_scale must have M elements, got P*M` | The whole `[P*M]` scale was passed | Pass `scale[rank*M:(rank+1)*M]` |
| A call never returns | A rank skipped a call, used another M, or was built with another `mode`, or (fused) a sender splits its slab into fewer `chunks` than its receiver, so a band or a barrier never arrives | Same arguments and the same calls on every rank |
| Wrong results, no error | A per-shard A scale; the same scale slice on every rank; (fused) a sender with more `chunks` than its receiver, whose band waits then pass before the rows have landed; a weight not through `preshuffle_b` or not identical on every rank; `acquire="none"` with `CCO_UNCACHED_WINDOW=0`; `c` read after the next call at that M; `send_buffer` written from a stream not ordered after the previous call | See [5. Calling the operator](#5-calling-the-operator) |
| The first call at a new M is slow | It compiles that M's kernels | Warm every M before timing or capturing |

## Further reading

- The [`a2a_gemm` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/a2a_gemm/README.md),
  next to the code: the full design, every measurement, where the time goes at
  P=8, and the negative results.
- [MORI-FUSED overview](MORI-CCO-FUSED-OPS-GUIDE.md): what the six operators
  have in common, and the headline numbers.
- [GEMM + all-to-all](fused-gemm-a2a.md), the QKV projection on the other side
  of Ulysses attention, and
  [Running the tests and benchmarks](fused-tests-and-benchmarks.md).
