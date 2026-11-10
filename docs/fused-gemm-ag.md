# GEMM + all-gather

`GemmAgOp` (`mori.ops.gemm_ag`) computes `C = A @ B.T` on each rank's own rows
and gathers the results, so every rank gets every rank's rows, `[P*M, N]`. It
serves a projection on a context- or sequence-parallel token shard whose output
every rank consumes, and was built for DeepSeek-V4's compressor `wkv_gate`. The
GEMM writes C into this rank's receive slot. On the fused path, which the default
picks at large M, its epilogue pushes each finished chunk of rows to every peer
over the SDMA copy engines while the GEMM computes the rest.

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
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ag import GemmAgOp

rank, world = dist.get_rank(), dist.get_world_size()   # after torch.cuda.set_device()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

N, K, M_MAX = 2048, 7168, 16384          # DeepSeek-V4-Pro wkv_gate; M_MAX is rows per rank
vmm = 2 * GemmAgOp.window_bytes_for(world, n=N, m_max=M_MAX, out_dtype="fp32") + (256 << 20)
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, in_dtype="bf16", out_dtype="fp32") as op:
        op.self_test()                    # collective, once
        m = op.padded_m(x.shape[0])       # x: this rank's [rows, K] bf16; the same m on every rank
        c = op(op.pad_rows(x, m), w)      # w: [N, K] bf16, the same on every rank
        # c: [world*m, N] fp32; source s owns rows [s*m, (s+1)*m); padding rows are zero
```

| What | Import |
|---|---|
| Operator | `from mori.ops.gemm_ag import GemmAgOp` |
| Shape check | `from mori.ops.gemm_ag import supports` |
| Why a shape is rejected | `from mori.ops.gemm_ag.op import shape_constraint` |
| fp8 weight layout | `from mori.ops.gemm_ag import preshuffle_b` (the same function as `mori.ops.gemm_ar.preshuffle_b`) |
| mxfp8 A-scale layout | `from mori.ops.gemm_ar import preshuffle_a_scale` |
| Communicator | `from mori.cco import Communicator, UniqueId` |
| Window layout and kernel builders | `from mori.ops.gemm_ag import ag_config, compile_bf16_gemm_ag, ...` ([section 6](#kernel-level-exports)) |

---

## 1. What it computes

Every rank `r` passes its own `A_r [M, K]` and the same `B [N, K]`, and gets
back `C = concat(A_0 @ B.T, ..., A_{P-1} @ B.T)`, `[P*M, N]`, with rank `r`'s
rows at `[r*M, (r+1)*M)`, its own included. `P` is the number of ranks and `M`
the padded rows per rank, the same on every rank. `B` must be the same on every
rank; the operator does not check it.

| `in_dtype` | `a` | `b` | Scales | Result (and wire) |
|---|---|---|---|---|
| `"bf16"` | `[M, K]` bf16 | `[N, K]` bf16, plain row-major | none | `[P*M, N]` bf16 or fp32 (`out_dtype`) |
| `"fp8"` (default) | `[M, K]` `float8_e4m3fn` | `[N, K]` `float8_e4m3fn` through `preshuffle_b` | by `quant`: `ptpc`, `blockscale` or `mxfp8` | `[P*M, N]` bf16 |

The layer is a projection on a context-parallel (CP) or sequence-parallel token
shard whose output every rank needs: in DeepSeek-V4's `wkv_gate` each CP rank
scores its own tokens, then every rank needs every token's score. The README's
model shapes are DeepSeek-V4-Pro's ratio-4 `compressor.wkv_gate` (K=7168,
N=2048, bf16 in, fp32 out) and DeepSeek-V4.1-Flash's ratio-2 KV and gate
projections (K=5120, N=512 each, or 1024 combined); M is the runtime token count
per rank. SGLang's Flash low-ratio CP path currently gathers the hidden states
before the projection, so for Flash this operator is a reordered candidate.

## 2. How it works

Each call launches two kernels on the current stream. The GEMM writes C
straight into this rank's slot of the receive buffer; every peer receives that
same contiguous slab, so nothing is staged.

- **`split`**: a one-block kernel then pushes the slab to every peer, one SDMA
  put each, waits for the copies and runs a cross-rank barrier.
- **`fused`**: the rows are cut into chunks of whole row tiles, one chunk per
  workgroup round of the GEMM grid, up to `max_chunks`. When a chunk's last
  tile is stored, the epilogue posts the chunk to every peer. Two chunks for one
  peer use different SDMA queues, so a fused call needs a queue per chunk per
  peer. The second kernel only waits for the queues and runs the barrier.
- **`auto`** (the default) picks one per M ([section 8](#8-choosing-settings)).

Fusion hides a chunk's transfer behind the GEMM rounds still to come. At these
shapes the transfer is several times the GEMM (Pro M=2048: 69 µs of GEMM
against about 300 µs of all-gather), so fusion pays only at large M.

**Receive-reuse credit.** The result is the receive buffer, and the next call
refills it. Each call first tells every peer it has started, and each push
waits until its destination has started too, so a rank a call ahead never
overwrites rows a slower rank is still reading. That holds provided whatever
reads the result is ordered before this rank's next call.

**The window** holds a few KiB of flags, chunk counters and credits, then the
receive buffer, `P * m_max * N` elements of `out_dtype`; it is allocated once
and cannot grow. A rank sends `(P-1) * M * N * elem_bytes` per call, 112 MiB at
Pro M=2048 with fp32 output and P=8. The wire carries the output dtype, so fp32
doubles the bytes of bf16.

## 3. Requirements

The contract every fused operator shares (build and environment, window sizing,
padding, result aliasing, the collective contract, cleanup, fp8 layouts) is in
[What every operator has in common](MORI-CCO-FUSED-OPS-GUIDE.md#what-every-operator-has-in-common).
For this operator:

- **GPU**: MI355X (gfx950). The GEMMs are FlyDSL kernels written for gfx950 and
  every measurement is on MI355X; the operator does not check the architecture.
  It reads the current device's CU count, so call `torch.cuda.set_device()`
  before constructing it.
- **FlyDSL**: `pip install 'amd_mori[flydsl]'`. `import mori.ops.gemm_ag`,
  `preshuffle_b` and the layout helpers work without it; `GemmAgOp`,
  `supports` and the kernel builders import it on first use.
- **SDMA**: a MORI built with `BUILD_CCO_SDMA=ON` (the default), and
  `MORI_ENABLE_SDMA=1` in the environment when the communicator is created,
  which is when MORI reads it.
- **SDMA queues**: `mode="auto"` and `"fused"` ask for `max_chunks` per peer
  (default 4), `"split"` for 1. The first device communicator on a communicator
  fixes its queue count, later ones reuse it, and MORI creates at most 8 per
  pair of GPUs. The constructor raises `RuntimeError` if there are too few.
- **Ranks**: 2 to 8, all in one node (one LSA domain).
- **VMM**: the window comes out of the communicator's `per_rank_vmm`; size it
  with `window_bytes_for` ([section 4](#window_bytes_for)) plus every other
  window on the communicator. The examples reserve
  `2 * window_bytes_for(...) + (256 << 20)`, as the overview does.

## 4. Constructor

```python
GemmAgOp(comm, *, n, k, m_max, in_dtype="fp8", quant="ptpc", out_dtype="bf16",
         mode="auto", max_chunks=4, chunks=None, tile=None, fence="release",
         peer_uncached=True, war_credit=True)
```

Collective: every rank constructs with the same arguments. It validates them,
allocates and zeroes the window, creates a device communicator with its SDMA
queues, and ends with a barrier.

| Name | Type | Default | Description |
|---|---|---|---|
| `comm` | `mori.cco.Communicator` | required | Its ranks are the all-gather's ranks. The only positional argument. |
| `n` | `int` | required | Output columns N, the weight's rows. A positive multiple of 256. |
| `k` | `int` | required | Inner dimension K. fp8: a multiple of 128, at least 256. bf16: a multiple of 64, at least 128. |
| `m_max` | `int` | required | The most rows per rank any call will pass, at least 1. Rounded up to the M granule (`op.m_max` holds the result); sizes the window. |
| `in_dtype` | `str` | `"fp8"` | `"fp8"` (`float8_e4m3fn` operands, gemm_ar's 8-wave fp8 GEMM) or `"bf16"`. |
| `quant` | `str` | `"ptpc"` | fp8 scaling: `"ptpc"` (per token, per channel), `"blockscale"` (A 1x128, B 128x128) or `"mxfp8"` (32-wide ue8m0). With `in_dtype="bf16"` leave it at `"ptpc"`. |
| `out_dtype` | `str` | `"bf16"` | Result and wire dtype, `"bf16"` or `"fp32"`; `"fp32"` needs `in_dtype="bf16"`. |
| `mode` | `str` | `"auto"` | `"fused"`, `"split"`, or `"auto"`: fused where `ceil(M/256) * ceil(N/256)` is at least the device's CU count. |
| `max_chunks` | `int` | `4` | The most chunks a fused call pushes in, and the SDMA queues per peer that `"auto"` and `"fused"` ask for. At least 1. Pass the same value to `window_bytes_for`. |
| `chunks` | `int` or `None` | `None` | Pins the fused chunk count instead of one per workgroup round; rounded down to a divisor of the row tiles, capped by `max_chunks`. At least 1. |
| `tile` | `tuple[int, int]` or `None` | `None` | Pins `(BLOCK_M, BLOCK_N)` for every M; M then pads to `BLOCK_M`. bf16: each 128 or 256, `BLOCK_N=128` only with fp32 output. fp8: `BLOCK_N=256`, `BLOCK_M` 128 or 256 (256 under mxfp8). |
| `fence` | `str` | `"release"` | The fused epilogue's fence before it pushes a chunk: `"release"` (one release-only system fence per workgroup), `"leader"` (a system fence from one wave per workgroup), `"all"` (from every lane) or `"none"` (no fence; gemm_ar measured the copy engine then reading some tiles stale on 8 ranks). |
| `peer_uncached` | `bool` | `True` | Fused calls store C uncached (`sc0`, `sc1`). With `fence="release"` this is the README's `sc_release`, used by every fused number there. |
| `war_credit` | `bool` | `True` | The receive-reuse credit. `False` exists only for the race test; back-to-back calls can then return another call's rows. |

`chunks`, `fence` and `peer_uncached` affect fused calls only.

### window_bytes_for()

```python
GemmAgOp.window_bytes_for(world_size, *, n, m_max, out_dtype="bf16", max_chunks=4) -> int
```

Static: the most bytes the constructor will allocate for its window, for sizing
the communicator's VMM before the operator exists. `world_size` (2 to 8) is the
number of ranks; pass `n`, `m_max`, `out_dtype` (`"fp32"` doubles the receive
buffer) and `max_chunks` (it sizes the chunk counters) as the constructor gets
them. It rounds `m_max` up to 256, the largest M granule, so the value holds for
every `in_dtype`, `quant`, `mode` and `tile`. Raises `ValueError` for a world
size outside 2-8, an `m_max` below 1, or an `n` that is not a positive multiple
of 256.

## 5. Calling the operator

```python
out = op(a, b, a_scale=None, b_scale=None)    # -> torch.Tensor [P*M, N]
```

One GEMM + all-gather, issued on the current stream; it returns without
synchronising. `M = a.shape[0]` must be a positive multiple of the M granule
(use `padded_m`), at most `op.m_max`, and the same on every rank. The first
call at each M compiles that M's kernels.

| Argument | `in_dtype="bf16"` | `in_dtype="fp8"` |
|---|---|---|
| `a` | `[M, K]` `torch.bfloat16` | `[M, K]` `torch.float8_e4m3fn` (`float8_e4m3fnuz` is rejected) |
| `b` | `[N, K]` `torch.bfloat16`, plain row-major | `[N, K]` `torch.float8_e4m3fn` through `preshuffle_b` |
| `a_scale`, `b_scale` | `None` | required, by `quant` (below) |

| `quant` | `a_scale` | `b_scale` |
|---|---|---|
| `"ptpc"` | `[M]` fp32, one per row | `[N]` fp32 |
| `"blockscale"` | fp32 per 1x128 block: `[K/128, M]`, `[M, K/128]` column-major, or flat in `[K/128, M]` order | `[N/128, K/128]` fp32 |
| `"mxfp8"` | the flat `int32` buffer `preshuffle_a_scale(a_exps)` returns, `a_exps` being the `[M, K/32]` ue8m0 bytes | the `[N/32, K/32]` ue8m0 bytes `w_exps` K-block-major, one per `int32`: `w_exps.t().contiguous().to(torch.int32).reshape(-1)` |

All operands must be GPU tensors; `a` and `b` are made contiguous (a copy if
they are not). A row-major `[M, K/128]` blockscale `a_scale` is rejected as
ambiguous: pass `a_scale.reshape(-1)` if its storage is already `[K/128, M]`
(as `aiter_per1x128_quant(transpose_scale=True)` returns it), or `a_scale.t()`
for a logical `[M, K/128]`.

**Return value**: a contiguous `[P*M, N]` tensor in `out_dtype`, rank `s`'s
rows at `[s*M, (s+1)*M)`. Zero-padded input rows come back as zero rows in
every source's block; slice them off per source. It is a view of the
operator's window, valid until this rank's next call, `self_test()` or
`close()`. Order whatever reads it before that call (same stream, or an event
the next call's stream waits on), or clone it.

## 6. Other methods

### padded_m()

`op.padded_m(m) -> int` rounds a row count up to `op.m_granule`: 128, 256
under `quant="mxfp8"`, or `BLOCK_M` when `tile` is pinned. Pad from a count
that is the same on every rank. `ValueError` if `m < 1`.

### pad_rows()

`op.pad_rows(x, m_pad) -> torch.Tensor` zero-extends a 2-D `[m, K]` tensor to
`[m_pad, K]` in a buffer the operator reuses while `m_pad` fits and K, dtype
and device stay the same, so the next `pad_rows` overwrites it and a CUDA graph
that reads it sees each refill. Pad before quantising, so fp8 scales cover the
padded rows. `ValueError` if `x` is not 2-D or `m_pad` is too small.

### fuses_at()

`op.fuses_at(m) -> bool`: whether a call at this M runs the fused path. Under
`mode="auto"` that is `ceil(M/256) * ceil(N/256) >= op.cus`; under `"fused"` or
`"split"` it is always `True` or `False`. No GPU work.

### tile_for()

`op.tile_for(m) -> tuple[int, int]`: the `(BLOCK_M, BLOCK_N)` a call at this
padded M uses: the pinned `tile`, else for bf16 the per-shape rule in
[section 8](#8-choosing-settings), else 128x256 (256x256 under mxfp8).

### chunks_for()

`op.chunks_for(m) -> int`: the chunks a call at this padded M pushes in. 1 on
the split path; otherwise the pinned `chunks` or one per workgroup round of the
tile, capped at `max_chunks` and rounded down to a divisor of the row tiles.

### self_test()

`op.self_test()` fills this rank's slot with `rank + 1`, runs the split push at
`m_max` and checks every peer's slot, raising `RuntimeError` at the first wrong
one; a put that silently moves nothing fails here. A collective: call it on
every rank once after construction, before capturing a graph. It synchronises
the device and overwrites the receive buffer.

### close() and the context manager

`op.close()`, or leaving a `with GemmAgOp(...) as op:` block, releases the
window and drops the compiled plans and the `pad_rows` buffer. Idempotent.
Synchronise first if a call may still be in flight. Afterwards a call or
`self_test()` raises `RuntimeError: this GemmAgOp has been closed`.

### Attributes

The constructor also sets, for reading: `m_max` (rounded up to the granule),
`m_granule`, `window_bytes` (this window's size), `sdma_queues` (`max_chunks`,
or 1 under `mode="split"`), `cus` (the CU count the `auto`, tile and chunk
rules use, read at construction), `rank` and `world_size`.

### supports()

`supports(n, k, world_size, **kwargs) -> bool` is `True` when
`2 <= world_size <= 8` and `shape_constraint(n, k, **kwargs)` finds nothing;
`kwargs` are `in_dtype`, `quant` and `out_dtype`, with the constructor's
defaults. It says whether a shape is expressible, not whether it is faster, and
looks at neither M, the GPU nor SDMA.

### shape_constraint()

`shape_constraint(n, k, *, in_dtype="fp8", quant="ptpc", out_dtype="bf16") -> str | None`
returns why the constructor would reject this N, K and precision (it raises
`ValueError("unsupported configuration: " + reason)`), or `None`. Import it
from `mori.ops.gemm_ag.op`, which also exports the allowed values as
`IN_DTYPES`, `QUANTS`, `OUT_DTYPES`, `MODES` and `FENCES`.

### preshuffle_b()

`preshuffle_b(w) -> torch.Tensor` reorders an fp8 `[N, K]` weight into the
order the fp8 GEMM reads; shape and dtype are unchanged. Call it once per
weight. It needs a 1-byte dtype, N a multiple of 16 and K a multiple of 64
(`ValueError` otherwise). bf16 weights are not shuffled.

### Kernel-level exports

`mori.ops.gemm_ag` also exports what `GemmAgOp` is built from. `AgConfig`,
`ag_config`, `counter_chunks` and `MAX_WORLD` (8) are the window layout and
launch arithmetic, importable without FlyDSL. `compile_bf16_gemm_ag`,
`compile_fused_gemm_ag`, `compile_gemm_local`, `build_sdma_phases` and
`build_sdma_chunk_post` build the kernels. The benchmarks call them directly;
the operator's users do not need them. `build_sdma_chunk_post` serves a
two-stream pipeline experiment and carries no receive-reuse credit.

## 7. Shapes and limits

| Constraint | Value | Error if violated |
|---|---|---|
| `in_dtype`, `quant`, `out_dtype` | as in [section 4](#4-constructor) | `ValueError: unsupported configuration: in_dtype=... is not one of ...` (likewise `quant`, `out_dtype`); `... quant does not apply to in_dtype='bf16'; leave it at its default`; `... the fp8 GEMM's epilogue only emits bf16` |
| N | a positive multiple of 256 | `ValueError: unsupported configuration: N=... must be a positive multiple of 256` |
| K, fp8 | a multiple of 128, at least 256 | `ValueError: unsupported configuration: K=... must be a multiple of 128 and at least 256` |
| K, bf16 | a multiple of 64, at least 128 | `ValueError: unsupported configuration: K=... must be a multiple of 64 and at least 128` |
| `mode`, `fence` | the listed values | `ValueError: mode=... is not one of ...`, `fence=... is not one of ...` |
| `m_max`, `max_chunks`, `chunks` | at least 1 (`chunks` may be `None`) | `ValueError: m_max and max_chunks must be >= 1; ...`, `chunks must be >= 1, got ...` |
| `tile`, fp8 | `BLOCK_N=256`, `BLOCK_M` 128 or 256 (256 under mxfp8) | `ValueError: tile=...: the fp8 GEMM takes BLOCK_N=256 and BLOCK_M 128 or 256 (256 under mxfp8)` |
| `tile`, bf16 | each 128 or 256, `BLOCK_N=128` only with fp32 output | `ValueError` from the window layout: `block_m=... must be a positive multiple of 128 ...`, `block_n must be 128 or 256 ...`, `block_n=128 needs elem_bytes=4 (an fp32 C) ...`. A `BLOCK_M` of 384 or more passes construction and fails the first call with `BLOCK_M must be 128 or 256 ...` |
| Ranks | 2 to 8 | `ValueError: world_size must be in [2, 8], got ...` (also from `window_bytes_for`) |
| One node | every rank in one LSA domain | `RuntimeError: GemmAgOp needs every rank in one LSA domain (one node); ...` |
| SDMA queues | at least `sdma_queues` per peer | `RuntimeError: GemmAgOp needs ... SDMA queue(s) per peer but the communicator has ...` |
| Open (each call) | not closed | `RuntimeError: this GemmAgOp has been closed` |
| M (each call) | a positive multiple of `op.m_granule`, at most `op.m_max` | `ValueError: M=... must be a positive multiple of ...; use padded_m() and pad_rows()`; `M=... exceeds the window's m_max=...` |
| `a`, `b` (each call) | `[M, K]` and `[N, K]`, the dtype of `in_dtype`, on a GPU | `ValueError: a must be (M, K), got ...`; `... must be one of [...] for in_dtype=..., got ...; the launch reinterprets it as raw bytes`; `... must be on a GPU, got ...` |
| Scales (each call) | none for bf16; both for fp8 | `ValueError: in_dtype='bf16' takes no scales`; `quant=... needs a_scale and b_scale` |
| Scale dtype and size (each call) | as in [section 5](#5-calling-the-operator) | `ValueError: a_scale must be torch.float32, got ...`; `a_scale must have ... elements, got ...`; `b_scale must have ... elements, got ...`; the layout messages for blockscale and mxfp8 |

## 8. Choosing settings

The defaults were tuned on MI355X for the bf16-in, fp32-out `wkv_gate` shapes;
leave them unless a measurement of your own says otherwise.

- **`mode="auto"`** fuses once even the largest tile's grid covers the CUs,
  `ceil(M/256) * ceil(N/256) >= CUs`: a GEMM long enough for the transfer to
  hide behind. The rule fits every pair in the README's results table at 256
  CUs; a GPU with more CUs needs a larger M. Pin `"split"` when M stays small
  or SDMA queues are scarce (it asks for one per peer).
- **bf16 tile**, per M and per path, among 256x256, 128x256, 256x128 and
  128x128. Split: the largest tile whose grid still covers the CUs, else the
  largest grid. Fused: the largest tile other than 256x256 whose grid runs at
  least two workgroup rounds, else the largest grid. fp8 uses 128x256 (256x256
  under mxfp8).
- **Chunks**: one per workgroup round of the fused tile, capped at
  `max_chunks`. The cap defaults to 4 because 4 measured within 2% of 8 where
  the rule would pick 8, and a queue per chunk per peer is what runs out of
  SDMA engine slots first.
- **`fence`, `peer_uncached`**: the defaults are the README's `sc_release`. On
  2026-10-29 the benchmark's default fence (`leader`) measured within 0.2% of it
  at M >= 4096, and 1.3% to 2.5% slower at M=2048. Do not use `"none"`.
- **`war_credit`**: leave it on.
- **No `max_shapes`**: one instance serves any number of distinct M up to
  `m_max`, in any order. Each new M compiles once; warm every M before
  capturing a CUDA graph.

What the defaults pick on MI355X (256 CUs), bf16 in, fp32 out, as `fuses_at`,
`tile_for` and `chunks_for` report it:

| Shape | M | Path | Tile | Chunks |
|---|---:|---|---|---:|
| Pro `wkv_gate`, N=2048 | 2048 | split | 128x128 | 1 |
| | 4096 | split | 128x256 | 1 |
| | 8192 | fused | 128x256 | 2 |
| | 16384 | fused | 128x256 | 4 |
| Flash combined, N=1024 | 2048 | split | 128x128 | 1 |
| | 16384 | fused | 128x256 | 2 |

**Environment variables.** The operator reads none itself. MORI reads
`MORI_ENABLE_SDMA` when the communicator is created; `MORI_SOCKET_IFNAME` names
the network interface of the communicator's bootstrap (the tests use `lo`).

## 9. Performance

From the
[`gemm_ag` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ag/README.md),
on MI355X, in µs. Each table is its own session; compare within a table only.

**Through the operator**: Pro `wkv_gate` (N=2048, K=7168), bf16 in, fp32 out,
graph-timed, maximum over ranks. `sc_release` is the default publication;
`leader` is the leader fence.

| P | M | `auto` | `fused` (sc_release) | `fused` (leader) | `split` |
|---:|---:|---:|---:|---:|---:|
| 8 | 2048 | **351.3** (split) | 362.8 | 361.0 | 351.3 |
| 8 | 4096 | **666.2** (split) | 685.5 | 687.6 | 666.1 |
| 8 | 8192 | **1239.2** (fused) | 1240.0 | 1240.9 | 1288.7 |
| 4 | 2048 | **349.1** (split) | 357.0 | 358.5 | 349.4 |
| 4 | 4096 | **662.0** (split) | 682.8 | 683.3 | 661.7 |
| 4 | 8192 | **1231.3** (fused) | 1231.3 | 1232.7 | 1282.2 |

**Against Torch + RCCL**: 2026-10-29, eight MI355X (crsuse2-m2m-v3-054), ROCm
7.2, PyTorch 2.11.0+rocm7.2, FlyDSL 0.4.0; the benchmark's modes, bf16 in, fp32
out, MORI's GEMM at a fixed 128x128 tile; 50 warm-ups and 101 graph samples,
maximum over ranks, median of three interleaved rounds. Fused rows use
`sc_release` at the best chunk count (`cN`).

| Shape | M | Torch + RCCL | MORI GEMM + SDMA | best fused SDMA | best vs Torch + RCCL |
|---|---:|---:|---:|---:|---:|
| Pro `wkv_gate`, N=2048 K=7168 | 2048 | 415.1 | **373.4** | 376.6 (c1) | **-10.0%** |
| | 4096 | 823.2 | **705.6** | 716.8 (c2) | **-14.3%** |
| | 8192 | 1510.5 | 1381.7 | **1314.8** (c4) | **-13.0%** |
| | 16384 | 3008.2 | 2844.2 | **2527.7** (c8) | **-16.0%** |
| Flash combined, N=1024 K=5120 | 2048 | 227.6 | 213.6 | **213.5** (c1) | **-6.2%** |
| | 16384 | 1465.9 | 1372.7 | **1309.2** (c4) | **-10.7%** |

How to read them:

- Fusion pays from M=8192 on Pro and at M=16384 on Flash N=1024, not below;
  `mode="auto"` encodes that and takes the faster path in every row of the
  first table. At its own tile (128x256, a chunk per round) the fused path
  measured 3.9%, 7.4% and 2.2% under split SDMA at Pro M=8192, Pro M=16384 and
  Flash M=16384, on the box of the 2026-10-29 table.
- Against Torch + RCCL the steady part of the gain is the transport: SDMA's
  all-gather alone is 12.5-13.9% faster than RCCL's at every shape. The fixed
  128x128 tile falls behind Torch's GEMM at large M; the per-M tile closes that
  from M=8192.
- The README's only fp8 measurement, an earlier protocol at M=N=2048, K=7168,
  has one-chunk fusion behind split SDMA under ptpc, blockscale and mxfp8.

## 10. Complete example

bf16 in, fp32 out, the Pro `wkv_gate` shape, replayed from a CUDA graph:

```python
# MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo torchrun --standalone --nproc_per_node=8 example.py
import os

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ag import GemmAgOp

N, K, M_MAX = 2048, 7168, 16384   # DeepSeek-V4-Pro wkv_gate; the most rows per rank ever passed
TOKENS = 3000                     # rows per rank this step, the same on every rank

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group(backend="cpu:gloo")
rank, world = dist.get_rank(), dist.get_world_size()
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

vmm = 2 * GemmAgOp.window_bytes_for(world, n=N, m_max=M_MAX, out_dtype="fp32") + (256 << 20)
with Communicator.init(world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm) as comm:
    with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, in_dtype="bf16", out_dtype="fp32") as op:
        op.self_test()                                     # collective, once

        g = torch.Generator(device="cuda").manual_seed(0)  # the same weight on every rank
        w = (torch.randn(N, K, generator=g, device="cuda") / 8).to(torch.bfloat16)
        torch.manual_seed(1 + rank)                        # a different shard per rank
        x = (torch.randn(TOKENS, K, device="cuda") / 8).to(torch.bfloat16)

        m = op.padded_m(TOKENS)                            # 3072
        a = op.pad_rows(x, m)                              # the buffer the graph reads
        print(rank, op.fuses_at(m), op.tile_for(m), op.chunks_for(m))  # MI355X: False (128, 128) 1

        op(a, w)                                           # warm-up: compiles this M
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(a, w)                                 # [world*m, N] fp32, in the window

        for _ in range(3):                                 # the same count on every rank
            x = (torch.randn(TOKENS, K, device="cuda") / 8).to(torch.bfloat16)
            op.pad_rows(x, m)                              # refills `a` in place
            graph.replay()
            result = out.view(world, m, N)[:, :TOKENS].reshape(world * TOKENS, N)  # a copy
        torch.cuda.synchronize()                           # nothing in flight at close()
dist.destroy_process_group()
```

fp8 in, bf16 out (the constructor's default precision) with ptpc scales: as
above, but size the VMM for bf16 output (`window_bytes_for(world, n=N,
m_max=M_MAX)`) and replace the operator block with

```python
from mori.ops.gemm_ag import preshuffle_b

with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, quant="ptpc") as op:   # in_dtype="fp8" by default
    op.self_test()
    b = preshuffle_b(w_fp8)                               # [N, K] float8_e4m3fn, once per weight
    m = op.padded_m(TOKENS)
    a, a_scale = quantize_per_token(op.pad_rows(x, m))    # your quantiser
    out = op(a, b, a_scale, w_scale)                      # [world*m, N] bf16
```

MORI ships no quantiser. `quantize_per_token` must return `a` as `[m, K]`
`torch.float8_e4m3fn` and `a_scale` as `[m]` fp32, one scale per row including
the padded rows; `w_scale` is the weight's `[N]` fp32 scale. For `blockscale`
and `mxfp8`, produce the layouts in [section 5](#5-calling-the-operator).

## 11. Testing and benchmarking

Run against an installed MORI: the checkout's `python/` has no compiled
`mori.cco.cco`, so keep it off `PYTHONPATH`.

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
pytest tests/python/cco/test_gemm_ag.py tests/python/cco/test_gemm_ag_op.py
```

`test_gemm_ag.py` covers the layout arithmetic, the tile and chunk rules,
models of the LDS ordering and SDMA completion, and single-GPU GEMM numerics
(skipped without gfx950). `test_gemm_ag_op.py` checks the configuration rules,
then runs one 8-rank `torchrun` of `gemm_ag_op_worker.py`: every precision in
both modes, every bf16 tile, chunk counts, back-to-back calls with a held
reader with the credit on and off, alternating M, padding, operand checks,
`self_test` and `close`. The multi-rank parts need 8 GPUs and FlyDSL.

The benchmark `benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py` calls the
kernels, not `GemmAgOp`, and defaults to the leader fence with cached stores.
With `--block-m`, `--block-n` and `--chunks` left at 0 it picks the tile and
chunks by the operator's rules at 256 CUs, with up to 8 chunks. Fused SDMA at
the Pro shape, as the operator runs it:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode fused-sdma --peer-uncached --fence release \
  --in-dtype bf16 --out-dtype fp32 -m 8192 --out-dim 2048 -k 7168 \
  --rounds 5 --warmup 50 --iters 101 --tolerance 1e-5
```

`--mode` also takes `gemm-only`, `gemm-to-window`, `split-rccl` and
`split-sdma` (`--fence release` is refused outside `fused-sdma`). `--in-dtype`
defaults to `fp8`, with `--quant ptpc|blockscale|mxfp8` and a default
`--tolerance` of 3e-3. `--torch-gemm` swaps in `torch.mm` for bf16-in,
fp32-out `gemm-only`, `split-rccl` and `split-sdma`; `--phase-split` also times
the GEMM alone. Each run prints `RESULT_JSON` and `TIMING_JSON`; the time is
the median over rounds of the per-round maximum over ranks. Run one
configuration at a time on an idle node, and let one process group's SDMA
queues tear down before the next. The
[benchmark README](https://github.com/ROCm/mori/blob/main/benchmark/cco/flydsl/gemm_ag/README.md)
also covers the two-stream pipeline experiment and the GEMM-only comparison.

## 12. Troubleshooting

| Error or symptom | What it means |
|---|---|
| `RuntimeError: GemmAgOp needs 4 SDMA queue(s) per peer but the communicator has 0` | No SDMA: MORI was built without `BUILD_CCO_SDMA=ON`, or `MORI_ENABLE_SDMA=1` was not set when the communicator was created. MORI also logs `no SDMA queues will be created (...)` with the reason. |
| The same error with a non-zero count | An earlier device communicator on this communicator fixed a smaller queue count. Construct the operator that needs the most queues first, lower `max_chunks`, or use `mode="split"`. More than 8 per peer is never available. |
| `RuntimeError: GemmAgOp needs every rank in one LSA domain (one node)` | The ranks span nodes; the operator is single-node. |
| `RuntimeError: gemm_ag self-test failed: the slot from rank s holds ..., expected ...` | The puts complete but move nothing. Check `BUILD_CCO_SDMA` and `MORI_ENABLE_SDMA`. |
| `RuntimeError: ccoMemAlloc failed: ...` in the constructor | Usually the window does not fit in what is left of `per_rank_vmm`. Size it with `window_bytes_for` for every window on the communicator. |
| `ValueError: M=... must be a positive multiple of ...` | Pad with `padded_m()` and `pad_rows()`. Under mxfp8 or a pinned 256-row tile the granule is 256. |
| `ValueError: M=... exceeds the window's m_max=...` | Construct with a larger `m_max`, and size the VMM for it. |
| `ValueError: a must be one of ['torch.float8_e4m3fn'] ...` | Wrong operand dtype, for example `float8_e4m3fnuz`, or bf16 on an fp8 instance. |
| `ValueError: a_scale is (M, K/128) row-major, which is ambiguous ...` | Pass `a_scale.reshape(-1)` for a buffer already in `[K/128, M]` order, or `a_scale.t()` for a logical `[M, K/128]`. |
| Wrong numbers, no error | An fp8 `b` not passed through `preshuffle_b`, a `b` that differs between ranks, or scales in another layout. None of these shows in the shapes. |
| The result changes after the call | It was read after this rank's next call (or `self_test()`), or on another stream without an event. Order the read before the next call, or clone it. |
| A hang | The ranks disagree on the collective contract: a different number of calls or graph replays, a different M, or only some ranks constructing or calling `self_test()`. |
| `hsaKmtCreateQueueExt` or `anvil.cpp` errors at start-up | The SDMA queues of the previous multi-rank run are not released yet. Wait and retry; the tests wait for the previous run's processes to leave the GPU, then retry once. |

## Further reading

- The [`gemm_ag` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ag/README.md),
  next to the code: the full design, every measurement, and what was tried and
  dropped (LSA transports, a short final chunk, smaller tiles, a lossy wire, a
  two-stream pipeline).
- [MORI-FUSED: Fused GEMM + Collective Operators](MORI-CCO-FUSED-OPS-GUIDE.md):
  what the six operators share, and performance at a glance.
- [Running the tests and benchmarks](fused-tests-and-benchmarks.md) for every
  fused operator.
