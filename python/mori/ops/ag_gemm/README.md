# `mori.ops.ag_gemm` -- all-gather fused into the fp8 GEMM that consumes it

Sequence parallelism leaves every rank with `A_r [m_local, K]`, its own tokens.
A ColumnParallelLinear after it needs every rank's tokens times this rank's
weight shard `B_r [N, K]`:

    C_r = concat_s(A_s) @ B_r.T          # [world * m_local, N]

Split, that is `all_gather` then GEMM. Fused here, the GEMM starts on its own
rows while the copy engines move the rest, and each remote tile starts as soon
as the band of rows it needs has landed. It is the AMD counterpart of
vllm-project/vllm#54151 (Kimi-K3 AG-GEMM), which overlaps the same thing with a
ring of `cuMemcpyDtoDAsync` and one GEMM launch per shard.

## Using it

`AgGemmOp` is the public entry point (`op.py`). Needs `MORI_ENABLE_SDMA=1` and
a mori built with `BUILD_CCO_SDMA=ON` (the default); the constructor refuses a
communicator without SDMA queues, and `self_test()` checks that the puts move
bytes.

```python
from mori.cco import Communicator
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

N, K, M_MAX = 6400, 7168, 4096        # per rank: weight shard rows, hidden, tokens
vmm = 2 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
        op.self_test()                                  # collective
        b_shuffled = preshuffle_b(b_fp8)                 # [N, K], once per weight

        m = op.padded_m(x.shape[0])                      # same m on every rank
        a_fp8, a_scale = quantize_per_token(op.pad_rows(x, m))
        c = op(a_fp8, b_shuffled, a_scale, b_scale)      # [world*m, N] bf16
        # rows [s*m, s*m + valid_s) are source s's tokens; padding rows are zero
```

- **Operands.** `a_fp8` is this rank's `[M, K]` fp8 e4m3fn tokens and
  `a_scale` their `[M]` fp32 per-token scale; `b_shuffled` this rank's `[N, K]`
  weight shard through `preshuffle_b` and `b_scale` its `[N]` fp32 per-channel
  scale. Anything the launch would reinterpret as bytes -- a wrong dtype, shape
  or M -- raises `ValueError`.
- **M** must be the same on every rank and a multiple of 256: pad with
  `padded_m()` / `pad_rows()` before quantising. One instance serves any such M
  up to `m_max`, in any order, and compiles each M once.
- **Zero copy.** `op.local_rows(m)` returns this rank's slot in the window,
  `([m, K] fp8, [m] fp32)`. Quantise into it and pass those tensors, and the
  call skips copying them in.
- **The result** is the op's own buffer, overwritten by the next call; pass
  `out=` (a contiguous `[world*M, N]` bf16 tensor) or clone to keep it.
- **Collective contract.** Every rank constructs the op and calls it the same
  number of times, with the same M, in the same order: each call ends in a
  cross-rank barrier.
- `mode="split"` runs push, barrier, the same GEMM: the baseline, and a
  fallback. `chunks=` caps the row bands per peer (default 4, rounded down to
  a divisor of the row tiles).

## How it works

The window holds the gathered operand, `recv [world][m_local][K]` fp8 plus its
per-row fp32 scale. The caller writes its rows into `recv[rank]`; from then on
the GEMM reads one `[world * m_local, K]` tensor and only three things differ
from a plain GEMM (`kernels.py`):

1. **Block 0 is the poster.** Lanes `0..world-1` each own a peer and push this
   rank's rows to it in `chunks` row bands: put, `quiet`, then a system-scope
   store of that band's arrival flag into the peer's window. All peers are
   served at once -- one per lane, each on its own (source, destination) queue.
   That is deliberately not #54151's ring: on NVSwitch one peer can take a
   GPU's whole NVLink bandwidth, but on MI355X each peer is its own xGMI link,
   and a ring would use one of seven at a time.
2. **Tile order.** Tiles over this rank's own rows come first and never wait.
   The rest go (chunk, source, row tile, column tile), the source rotated by
   rank, so the first wave of remote tiles needs only each peer's band 0.
3. **The wait.** A remote tile's thread 0 spins on its band's flag, then
   fences, before the prologue issues a single A load.

A barrier-only kernel closes every call and advances the epoch the flags are
compared against. It is load-bearing twice over: nothing is ever reset, so the
epoch is what tells this call's flags from the last one's; and without the
barrier a rank a call ahead refills a slot a slower rank is still reading.
Removing it fails the back-to-back test deterministically (relL2 0.8-1.3).

The poster lives *in* the GEMM rather than in a kernel on a second stream on
purpose: remote tiles spin, and a grid whose resident blocks all spin waiting
for a push kernel that has not been dispatched yet never finishes. Block 0 is
dispatched first, so as part of the GEMM the push cannot be starved. Its own
tile pays the push time, and it is a local tile that nothing waits on.

**ptpc only for now.** Its A scale is one fp32 per row, so it travels with its
row band. blockscale and mxfp8 store the A scale K-block major, where a row band
is strided, and need a push of their own.

## Measured

Kimi-K3's column-parallel projections at TP8: K=7168, N per rank 6288 (KDA)
and 3648 (MLA), rounded up to 6400 and 3840 because the store needs whole
256-column tiles. fp8 ptpc in, bf16 out, 256x256 tile. 8x MI355X, ROCm 7.14,
PyTorch 2.12, FlyDSL 0.3.4.1; graph replay, median of 51 after 20 warmups, max
over ranks; median of 2 interleaved rounds, every run on an idle box (spread
<= 1% except one 4.4% cell). All 92 runs validated at the fp8 floor
(relL2 1.66e-3), each against every row of C for two different operand sets.

`split-rccl` is `all_gather_into_tensor` (rows, then scales) and the same GEMM;
`split-sdma` is push, barrier, the same GEMM, barrier. Times in us.

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

c4 needs `m_local` to be four 256-row tiles, so it starts at global M=8192.

**Reading it.**

- **The chunks are the mechanism.** One chunk barely beats the split path,
  because every remote tile waits for a peer's whole slab. Splitting it lets the
  first remote wave start after a quarter of the transfer; the best count rises
  with M, and c8 was not measured.
- **The win grows with M and with the transfer's share of the layer.** MLA, whose
  GEMM is smaller against the same bytes, gains more at large M (26%) and least
  at small M, where there is little GEMM left to hide anything behind once the
  local tiles are done.
- **What is left exposed** is 10.6-22.7% over the bare GEMM at global M=32768
  and up to 70% at 4096. The unfused transfer at M=32768 is ~500us, about
  410 GB/s per rank across seven links.
- **Against #54151** (GB300, BF16, ring): its AG-GEMM over AG+GEMM is
  1.26-1.36x for KDA and 1.07-1.54x for MLA at global M >= 3072, and 0.88x /
  0.73x at 2048. The ratios here are of the same order -- 1.08-1.32x KDA,
  1.06-1.47x MLA -- on different hardware, precision and GEMM, so only the
  ratios compare, not the times.

## Running it

```bash
MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/ag_gemm/bench_ag_gemm.py --mode fused-sdma --chunks 4 \
  -m 4096 -N 6400 -k 7168 --back-to-back 8

MORI_ENABLE_SDMA=1 pytest -q tests/python/cco/test_ag_gemm.py tests/python/cco/test_ag_gemm_op.py
```

`--back-to-back R` adds R calls with no host sync between them, each with its
own operands and output, one rank held on the GPU before each, and validates
every one. That is the test that fails when the closing barrier is removed.

## Not done yet

blockscale / mxfp8 (the A-scale push), N that is not a multiple of 256, c8 and
beyond, and a model integration.
