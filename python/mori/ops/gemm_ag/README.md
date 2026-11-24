# GEMM + All-Gather

`mori.ops.gemm_ag` computes `C = A @ B.T` from local `A [M,K]` and replicated
`B [N,K]`, then gathers C along rows. Every rank receives `[world_size*M,N]`,
with source rank `r` at rows `[r*M,(r+1)*M)`. M is the row count **per rank**.
FP8 inputs produce BF16 output; BF16 inputs support BF16 or FP32 output.
Communication uses the output dtype.

[`GemmAgOp`](#using-it) is the public entry point. See
[benchmark commands](../../../../benchmark/cco/flydsl/gemm_ag/README.md) for the
benchmarks, and [Experiments](#experiments) for what was tried and why. The
retained measurements show:

- For the DeepSeek-V4-Pro ratio-4 main compressor, at M=N=2048, K=7168,
  BF16-to-FP32 GEMM with an explicit `128x128` tile measures **69.4 µs**
  against native Torch's **77.8 µs** on the current stack (72.0 vs 71.1 in
  September -- Torch moved, not MORI). `pick_tile` now chooses the tile per
  shape and per path (see [GEMM tile](#gemm-tile-retuned-per-shape)).
- At that shape, non-fused GEMM + SDMA takes **355.9 µs**; fused SDMA takes
  **360.6 µs** with one chunk (373.4 / 376.6 µs after merging `main`, on the
  crsuse2 stack, against 415.1 for Torch + RCCL -- see
  [below](#re-measured-after-merging-main-2026-10-29)).
- At larger M the same shape gains from four-chunk single-stream fusion:
  **5.3% at M=8192** and **11.9% at M=16384**, with no benefit at M=4096
  (re-measured 2026-10-19; September's v2-015 pairs gave 4.9–5.2% and 11.1%,
  and the 2026-10-29 re-measurement after merging `main` 4.8% and 11.1%).
- Against Torch + RCCL the best MORI path is 10-16% faster on Pro and 6-11%
  on Flash (2026-10-29). The steady part is the transport -- SDMA's AG is
  ~13% under RCCL's -- while the `128x128` GEMM falls behind Torch's at large
  M at `128x128`. Retuning the tile per shape closes that gap from M=8192
  (23% ahead at M=4096) and speeds the split path, but not the fused one:
  a tile big enough to launch only one round of workgroups leaves fusion
  nothing to overlap. See
  [GEMM tile, retuned per shape](#gemm-tile-retuned-per-shape).
  Fusion performance depends on shape and chunk size.

## Using it

`GemmAgOp` (`op.py`). Needs `MORI_ENABLE_SDMA=1` and a mori built with
`BUILD_CCO_SDMA=ON` (the default); the constructor refuses a communicator
without the SDMA queues it needs, and `self_test()` checks that the puts move
bytes.

```python
from mori.ops.gemm_ag import GemmAgOp, preshuffle_b

N, K, M_MAX = 2048, 7168, 16384          # wkv_gate on DeepSeek-V4-Pro, rows per rank
vmm = 2 * GemmAgOp.window_bytes_for(world, n=N, m_max=M_MAX, out_dtype="fp32") + (256 << 20)
with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
    with GemmAgOp(comm, n=N, k=K, m_max=M_MAX, in_dtype="bf16", out_dtype="fp32") as op:
        op.self_test()                                    # collective
        m = op.padded_m(x.shape[0])                       # same m on every rank
        c = op(op.pad_rows(x, m), w)                      # [world*m, N] fp32
        # rows [s*m, s*m + valid_s) are source s's; padding rows are zero
```

- **Precision.** `in_dtype="bf16"` takes bf16 `a`/`b` (`b` plain row-major), no
  scales, and returns bf16 or fp32. `in_dtype="fp8"` takes float8_e4m3fn `a`
  and `preshuffle_b(b)`, scales by `quant` exactly as
  `gemm_ar.GemmAllReduceOp` does (ptpc, blockscale, mxfp8), and returns bf16.
- **Shapes.** `N` a multiple of 256; `K` a multiple of 128 (fp8, at least 256)
  or 64 (bf16, at least 128); M the same on every rank and a multiple of 128
  (256 under mxfp8), through `padded_m()` / `pad_rows()`. One instance serves
  any number of such M up to `m_max`, in any order, compiling each once; the
  bf16 tile and the fused chunk count are picked per M from the measured tables
  (`pick_tile`, `pick_chunks`) against the device's CU count.
- **Modes.** `mode="auto"` (default) fuses an M once even a 256x256 tile's grid
  covers the CUs -- `ceil(M/256) * ceil(N/256) >= CUs`, M=8192 for Pro and
  16384 for Flash on MI355X -- and splits below it, where the table below has
  split ahead or level; one instance switches per M. `"fused"` and `"split"`
  pin a path. The fused path keeps two chunks of one destination apart by SDMA
  queue, so `"auto"` and `"fused"` ask for `max_chunks` queues per peer
  (default 4) and cap the chunk count there. Fused calls publish with
  `fence="release"` and `peer_uncached=True` (`sc_release` below) unless told
  otherwise.

  Through the op, Pro `wkv_gate` (bf16 in / fp32 out), graph-timed, maximum
  over ranks, MI355X (us):

  | P | M | `auto` | `fused` (sc_release) | `fused` (leader) | `split` |
  |---:|---:|---:|---:|---:|---:|
  | 8 | 2048 | **351.3** (split) | 362.8 | 361.0 | 351.3 |
  | 8 | 4096 | **666.2** (split) | 685.5 | 687.6 | 666.1 |
  | 8 | 8192 | **1239.2** (fused) | 1240.0 | 1240.9 | 1288.7 |
  | 4 | 2048 | **349.1** (split) | 357.0 | 358.5 | 349.4 |
  | 4 | 4096 | **662.0** (split) | 682.8 | 683.3 | 661.7 |
  | 4 | 8192 | **1231.3** (fused) | 1231.3 | 1232.7 | 1282.2 |
- **The result** is this op's receive buffer and the next call refills it.
  Whatever reads it must be ordered before this rank's next call -- the same
  stream, or an event the next call waits on; clone it to keep it longer. That
  is the contract the recv-reuse credit (below) rests on.
- **Collective contract.** Every rank constructs the op and calls it the same
  number of times, with the same M, in the same order.

## Model shapes

| Model / projection | K | N | Source |
|---|---:|---:|---|
| DeepSeek-V4-Pro, ratio-4 main `compressor.wkv_gate` | 7168 | 2048 | `hidden_size=7168`; output `2 * coff * head_dim`, with `coff=2`, `head_dim=512` |
| DeepSeek-V4.1-Flash, ratio-2 KV or gate projection | 5120 | 512 each | Two separate projections in the current SGLang ROCm path |
| DeepSeek-V4.1-Flash, combined KV/gate candidate | 5120 | 1024 | Concatenate the two 512-row weight matrices; the consumer still needs separate KV/gate outputs |

M is selected from the runtime token count, not the model configuration.
These are synthetic-input operator benchmarks at model-derived dimensions,
not end-to-end SGLang serving measurements. SGLang's Flash low-ratio CP path
currently gathers hidden states before projection. Its GEMMs use the gathered
global row count; a local GEMM followed by AG is a reordered candidate.

## Execution paths

The window-based paths write C directly into the rank's own receive slot.
All peers receive the same contiguous slab, so no staging layout is needed.
Bytes sent per rank are `(world_size-1) * M * N * elem_bytes`.

| Benchmark mode | Execution |
|---|---|
| `gemm-only` | GEMM into a plain tensor |
| `gemm-to-window` | GEMM into the local receive slot |
| `split-rccl` | GEMM, then RCCL `all_gather_into_tensor` |
| `split-sdma` | GEMM, then SDMA gather and completion barrier |
| `fused-sdma` | GEMM epilogue submits completed M chunks; a drain finishes AG |

The SDMA modes submit kernels on one stream. SDMA can still overlap a fused
producer's later computation. The separate M-chunk pipeline uses two streams;
its `serial` control retains those streams but disables compute/transfer overlap.

**The transport is SDMA only.** Three LSA modes -- `split-lsa-push` (a copy
kernel storing C into the peers), `split-lsa-pull` (one loading the peers' C)
and `fused-lsa` (the epilogue storing every tile into every peer) -- were
measured and removed, as `gemm_a2a` did with its own. Their rows stay in the
dated tables below, marked *removed*. In short: at the Pro shape the best LSA
path was 4.6% behind split SDMA (372.2 against 355.9 µs) and fused LSA 16-19%
behind; LSA was ahead only on the smallest payload measured, Flash N=512 at
M=2048, by 3% (126.4 against 130.5 µs), and within 2% behind on the other
M=2048 shapes. A fused LSA epilogue writes `world` copies of every tile from
inside the GEMM and slows the GEMM with it, and cached peer stores reached
58% of line rate where the copy engines reach 94% (`gemm_a2a`). Keeping two
transports doubled the test matrix for a few percent at one small shape. The
removed kernels are in this branch's history.

## Interfaces and constraints

| Source | Entry points |
|---|---|
| [_gemm_a16w16_8wave.py](_gemm_a16w16_8wave.py) | `compile_bf16_gemm_ag`, including local split-K partials |
| [kernels_fused.py](kernels_fused.py) | FP8 `compile_fused_gemm_ag` and `compile_gemm_local` |
| [kernels_sdma.py](kernels_sdma.py) | `build_sdma_phases` and `build_sdma_chunk_post` |
| [layout.py](layout.py) | `ag_config`, window offsets and counter layout |
| [op.py](op.py) | `GemmAgOp`, the public wrapper over all of the above |
| [_credit.py](_credit.py) | the recv-reuse credit the fused GEMMs and `gather` carry |

- Configuration supports 2–8 ranks. M and N must be positive multiples of
  their tile dimensions; N need not be divisible by `world_size * block_n`.
- BF16 tiles use BM/BN in `{128,256}`; BN=128 requires FP32 output. K must
  contain at least two complete 64-element steps.
- FP8 uses BN=256 and K divisible by 128. MXFP8 requires BM=256 and N divisible
  by 32; its scale operands follow the MXFP8 format.
- `split_k=2/4/8` writes local FP32 `[split_k,M,N]` partials, retaining A/B row
  stride K. Each partition needs at least two complete 64-element steps.
  Reduce partials before consumption or communication; fused and peer-published
  partial output are rejected.
- Fused chunks divide the M tiles. Lane-parallel posting requires at least
  one queue per chunk per peer; the benchmark derives that queue count.
  `build_sdma_chunk_post` instead uses queue 0 and requires serialized submissions.

## Re-measured after merging `main` (2026-10-29)

On **crsuse2-m2m-v3-054, eight MI355X GPUs**, ROCm 7.2, PyTorch
`2.11.0+rocm7.2`, FlyDSL 0.4.0, this branch merged with `main` at `b853716`.
BF16 in / FP32 out, MORI at `128x128`, 50 warmups and 101 graph samples,
maximum over ranks; every configuration in a fresh process group on an idle
box. The three columns of each row ran interleaved in one session, three
rounds, median of rounds (spread <= 2.0%; 3.0% on Flash M=2048 split).
Fused rows use `--peer-uncached --fence release` at the chunk count a separate
sweep found best. All samples validated, worst relL2 `9.93e-7`.

| Shape | M | Torch + RCCL | MORI GEMM + SDMA | best fused SDMA | best vs Torch + RCCL |
|---|---:|---:|---:|---:|---:|
| Pro `wkv_gate`, N=2048 K=7168 | 2048 | 415.1 | **373.4** | 376.6 (c1) | **-10.0%** |
| | 4096 | 823.2 | **705.6** | 716.8 (c2) | **-14.3%** |
| | 8192 | 1510.5 | 1381.7 | **1314.8** (c4) | **-13.0%** |
| | 16384 | 3008.2 | 2844.2 | **2527.7** (c8) | **-16.0%** |
| Flash combined, N=1024 K=5120 | 2048 | 227.6 | 213.6 | **213.5** (c1) | **-6.2%** |
| | 16384 | 1465.9 | 1372.7 | **1309.2** (c4) | **-10.7%** |

Fusion pays from M=8192 (4.8% / 11.1% over split SDMA on Pro, 4.6% on Flash
at M=16384) and not below it; the default fence measured within 0.2% of
`sc_release` at M >= 4096, 1.3% slower on Flash M=2048 and 2.5% slower at Pro
M=2048 c1. Absolute times sit
3-5% above mi355-gpu-51's 2026-10-19 tables and match September's v2-015
numbers (1382.3 / 1314.3 at M=8192, 2841.4 / 2526.1 at M=16384), so read the
gpu-51 tables as that host's.

### Where the gain over Torch + RCCL comes from

`--phase-split` times the GEMM alone in the same process after the end-to-end
measurement; the AG is the difference. Same box, three interleaved rounds of
all four combinations, µs:

| Shape | M | GEMM Torch | GEMM MORI | AG RCCL | AG SDMA | Torch+RCCL | Torch+SDMA | MORI+RCCL | MORI+SDMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pro | 2048 | 71.4 | 73.2 | 344.9 | 300.3 | 416.4 | **370.0** | 426.6 | 373.6 |
| | 4096 | 144.3 | **118.8** | 679.3 | 585.1 | 823.6 | 737.2 | 815.6 | **704.2** |
| | 8192 | **180.8** | 224.0 | 1329.2 | 1159.5 | 1510.0 | **1343.7** | 1605.3 | 1383.5 |
| | 16384 | **361.5** | 526.6 | 2648.0 | 2317.7 | 3009.5 | **2673.3** | 3195.0 | 2844.3 |
| Flash | 2048 | **44.2** | 54.0 | 185.2 | 161.3 | 228.2 | **203.0** | 238.6 | 215.3 |
| | 16384 | **139.4** | 212.4 | 1329.5 | 1160.3 | 1468.6 | **1296.6** | 1564.0 | 1372.5 |

GEMM columns are from the Torch+RCCL and MORI+SDMA runs, AG columns likewise.

* **The transport is the steady part of the gain.** SDMA's AG is 12.5-13.9%
  under RCCL's at every shape, with either GEMM in front of it.
* **The `128x128` GEMM is not.** It is level with Torch at Pro M=2048 and 18%
  faster at M=4096, but 24% slower at M=8192, 46% at M=16384 and 22-52% on
  Flash. The tile was chosen at M=2048, where `256x256` launches 64 workgroups
  on 256 CUs; at large M that argument no longer holds, and this sweep did not
  retune it.
* **So Torch's GEMM + SDMA is the best non-fused path almost everywhere**, and
  beats MORI's own split path by 1-6%. Fusion still wins on Pro at large M
  despite its slower GEMM -- 1314.8 against 1343.7 at M=8192 and 2527.7
  against 2673.3 at M=16384 (cross-session, same box and day) -- but not on
  Flash, where Torch + SDMA (1296.6) edges the fused c4 (1309.2).

### GEMM tile, retuned per shape

The `128x128` tile above was picked at M=2048. Sweeping all four BF16 tiles
with the same phase split (same box, three interleaved rounds, µs; GEMM is
the same-run GEMM-alone time):

| Shape | M | GEMM Torch | 128x128 | 128x256 | 256x128 | 256x256 |
|---|---:|---:|---:|---:|---:|---:|
| Pro | 2048 | 71.0 | **73.3** | 91.0 | 93.0 | 132.0 |
| | 4096 | 146.0 | 120.9 | **112.5** | 116.4 | 143.4 |
| | 8192 | 181.2 | 223.0 | 211.4 | 212.2 | **185.0** |
| | 16384 | 365.1 | 527.1 | 455.9 | 465.5 | **364.6** |
| Flash | 2048 | 43.9 | **54.0** | 65.7 | 68.6 | 99.2 |
| | 16384 | 140.0 | 212.2 | 156.4 | 209.8 | **138.0** |

With the right tile MORI's GEMM matches Torch's from M=8192 and beats it by
23% at Pro M=4096; only Flash M=2048 stays behind (54.0 against 43.9) with
every tile. `pick_tile`'s rule -- the largest tile whose grid still covers
the 256 CUs, else the largest grid -- names the measured best at all six
shapes, and is now the default: `TILE_CANDIDATES` holds all four tiles.

Then the fused path at that tile, in one session with the baselines (µs):

| Shape | M | Torch + RCCL | Torch GEMM + SDMA | MORI split, best tile | fused, 128x128 | fused, best tile | best vs Torch + RCCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pro | 4096 | 823.8 | 736.1 | **697.4** (128x256) | 718.1 (c2) | 699.7 (128x256 c1) | **-15.3%** |
| | 8192 | 1508.4 | 1342.4 | 1340.2 (256x256) | **1314.5** (c4) | 1352.9 (256x256 c2) | **-12.9%** |
| | 16384 | 3010.9 | 2671.0 | 2675.6 (256x256) | 2526.7 (c8) | **2525.2** (256x256 c2) | **-16.1%** |
| Flash | 16384 | 1466.8 | 1296.7 | **1294.9** (256x256) | 1308.0 (c4) | 1308.1 (256x256 c2) | **-11.7%** |

**A faster tile speeds the split path, not the fused one.** Fusion overlaps
a chunk's transfer with the GEMM rounds still to come, and a larger tile
removes those rounds: `256x256` launches 256 workgroups at Pro M=8192 and
Flash M=16384 -- one per CU, a single round -- so every chunk completes at
once and fused runs about 1% *behind* split at the same tile. At Pro M=16384
it launches 512, two rounds, and exactly two chunks is the only count that
still overlaps (c2 2525.2; c4 2545.0, c8 2584.2).

So the fused path wants its own tile. Every tile at every chunk count, one
session, `sc_release`, µs (best per tile):

| Shape | M | split, best tile | fused 128x128 | fused 128x256 | fused 256x128 | fused 256x256 |
|---|---:|---:|---:|---:|---:|---:|
| Pro | 4096 | **696.2** | 717.2 (c2) | 712.4 (c2) | 700.0 (c1) | -- |
| | 8192 | 1341.6 | 1314.2 (c4) | **1289.6** (c2) | 1291.0 (c2) | 1354.9 (c2) |
| | 16384 | 2676.2 | 2525.8 (c8) | **2479.2** (c4) | 2479.0 (c4) | 2527.1 (c2) |
| Flash | 16384 | 1294.2 | 1310.0 (c4) | **1265.7** (c2) | 1298.1 (c2) | 1307.4 (c2) |

`128x256` with **one chunk per workgroup round** is the best fused
configuration at every shape where fusion pays: two rounds and c2 at Pro
M=8192 and Flash M=16384, four and c4 at Pro M=16384. That is 3.9%, 7.4% and
2.2% under split SDMA -- the first time fusion beats split on Flash -- and
1.9% / 1.9% / 3.4% under `128x128`, the previous fused tile. `256x256` is
the best at none: with a chunk per round the first chunk is ready after about
GEMM / rounds, and its faster GEMM (364.6 against 455.9 µs at M=16384) does
not make up for half the rounds. `pick_tile(..., fused=True)` and
`pick_chunks` encode this, and the benchmark uses them when `--block-m`,
`--block-n` and `--chunks` are left at 0.

The best path per shape is therefore MORI split at the unfused tile below
M=8192 and fused at M >= 8192: against Torch + RCCL in the session above,
-15.5% at Pro M=4096 (696.2), -14.5% at M=8192 (1289.6), -17.7% at M=16384
(2479.2) and -13.7% on Flash M=16384 (1265.7) -- the fused numbers across two
sessions on the same box and day.

`main`'s short final chunk for `gemm_ar` (#748's `chunk_bands`) was tried
here and does not help: see [Experiments](#short-final-chunk-748).

## BF16 benchmarks: DeepSeek-V4-Pro

Model shape: **DeepSeek-V4-Pro ratio-4 main compressor**, M=N=2048, K=7168,
BF16 inputs and FP32 output. Measured on 2026-10-19 on **mi355-gpu-51, eight
MI355X GPUs**, ROCm 7.14, PyTorch `2.12.0+rocm7.14.0`, FlyDSL 0.3.4.1, MORI
`main` at `8ae8156` (which includes the SDMA changes #603 and #711).
The first tables were taken on 2026-09-28 with PyTorch 2.10 / ROCm 7.2 /
FlyDSL 0.2.4; the host and the whole stack changed between the two, so a
difference between them is not attributable to MORI alone. The
[v2-015 controls](#larger-m-on-v2-015) belong to that earlier
study and compare only with each other.

### Pure GEMM

Native Torch uses `torch.mm(a, b.T, out_dtype=torch.float32, out=c)`. MORI
uses `fuse=False, peer_uncached=False`. Outputs are preallocated; no AG or
peer-publication cost is included.

Five rounds rotate implementation order. Each captures one GEMM in a graph,
uses 100 warmups and 101 event samples, and takes the maximum of per-rank
medians; the reported time is the median across rounds.

| Implementation | Median, µs | Round range, µs | Latency vs Torch |
|---|---:|---:|---:|
| Native Torch BF16 → FP32 | 77.8 | 77.4–79.8 | baseline |
| MORI `256x256` (default) | 127.4 | 127.2–127.4 | +63.8% |
| MORI `128x256` | 83.9 | 83.4–84.2 | +7.8% |
| MORI `128x128` | 69.4 | 69.1–69.7 | −10.8% |

All implementations passed on all ranks, with worst relative L2 error
`9.95e-7`. The `128x128` tile is now 10.8% below Torch, but most of that is
Torch moving: it measured 71.1 µs on the September stack and MORI `128x128`
72.0, so read the sign as specific to this stack rather than as a GEMM win.
Select the tile explicitly with `--block-m 128 --block-n 128`.
BF16 output followed by widening is not an equivalent FP32-output reference.

### GEMM + AG — DeepSeek-V4-Pro

Each rank produces 16 MiB and sends 112 MiB. Times include the complete AG
and final cross-rank synchronization. Native Torch + RCCL takes **407.4 µs**.

| Mode | `256x256`, µs | `128x128`, µs |
|---|---:|---:|
| GEMM + RCCL | 472.0 | 414.8 |
| GEMM + LSA push *(removed)* | 438.7 | 372.2 |
| GEMM + LSA pull *(removed)* | 443.8 | 399.2 |
| Fused LSA, cached stores *(removed)* | 466.9 | 413.7 |
| Fused LSA, uncached stores *(removed)* | 467.2 | 422.6 |
| **GEMM + SDMA** | **421.8** | **355.9** |
| Fused SDMA, 1 chunk | 422.5 | 360.6 |
| Fused SDMA, 2 chunks | 433.5 | 372.1 |
| Fused SDMA, 4 chunks | 455.8 | 394.0 |

These fused runs use the default leader fence and lane-parallel SDMA posting;
all multi-queue measurements use the corrected drain. Five rounds use 50
warmups and 101 graph samples each, taking per-round maxima across ranks
before the final median. Every configuration ran in a fresh process group
with no foreign GPU processes before or after it.

All 19 configurations passed initial output and two changed-input replays,
with finite output and `relL2 < 1e-5`; worst error was `9.93e-7`. The receive
buffer was cloned immediately after replay, before reference computation.
Allocation, compilation, capture and validation were outside timing.

At this shape, the benefit comes from tile selection and SDMA transport:
355.9 µs is 12.6% below native Torch + RCCL. Fusing still does not pay here:
every chunk count is slower than the split SDMA path, as in September. The
cached fused-LSA store, 70 µs behind the uncached one then, now measures
level with it. The publication optimization and
larger-M fusion results are in [Experiments](#experiments).

### Earlier FP8 reference — DeepSeek-V4-Pro dimensions

These FP8-input/BF16-output measurements used eight MI355X GPUs,
M=N=2048, K=7168, and 21 graph samples after 10 warmups, taking the maximum
across ranks. PTPC uses `128x256`; MXFP8 uses `256x256`. This older protocol
is separate from the BF16/FP32 measurements above. Only the dimensions come
from the Pro compressor; this FP8-input path is not its native BF16 projection.
Times are µs.

| Mode | PTPC | Blockscale | MXFP8 |
|---|---:|---:|---:|
| GEMM only | 57.3 | 96.3 | 86.3 |
| GEMM + RCCL | 231.8 | 266.9 | 266.0 |
| GEMM + LSA push *(removed)* | 212.6 | 248.3 | 244.2 |
| GEMM + LSA pull *(removed)* | 217.0 | 251.0 | 248.8 |
| Fused LSA *(removed)* | 265.8 | 328.1 | 292.4 |
| GEMM + SDMA | 212.5 | 245.4 | 242.2 |
| Fused SDMA, 1 chunk | 223.6 | 259.1 | 250.7 |

All modes validated at about `1.66e-3` relative L2 error against the
precision-specific reference. These results do not establish a fusion win
for that shape or a precision-equivalent replacement for BF16-to-FP32 GEMM.

## BF16 benchmarks: DeepSeek-V4.1-Flash

Measured on 2026-10-19 on **mi355-gpu-51, eight MI355X GPUs**, ROCm 7.14,
PyTorch `2.12.0+rocm7.14.0`, FlyDSL 0.3.4.1, MORI `main` at `8ae8156` (which
includes the SDMA changes #603 and #711).
The September tables came from v2-015 with ROCm 7.2.4, PyTorch 2.11 and FlyDSL
0.2.4. All use 128x128 tiles. K=5120 comes from
Flash's hidden size. **N=512 is one KV or gate projection** in the ratio-2
path; **N=1024 is a combined KV/gate candidate**. Both use BF16 input and
FP32 output/wire. The N=512 timings must not be read as the total for both
projections.

Each case uses five rounds, 50 warmups and 101 graph samples per round, with
the same rank/round aggregation and changed-input validation described above.
All 34 re-measured configurations passed, with worst relative L2 error
`8.28e-07`.
These are local-projection-then-AG tests; token-order restoration and KV/gate
unpacking are excluded from these tables. A separate ratio-2 subgraph test
includes those steps.

### GEMM + AG at local M=2048

| Mode | N=512, µs | N=1024, µs |
|---|---:|---:|
| Native Torch + RCCL | 153.5 | 273.4 |
| MORI GEMM + RCCL | 173.1 | 282.2 |
| MORI GEMM + LSA push *(removed)* | **126.4** | 206.5 |
| MORI GEMM + LSA pull *(removed)* | 138.0 | 220.1 |
| Fused LSA, uncached *(removed)* | 128.4 | 221.2 |
| MORI GEMM + SDMA | 130.5 | 203.2 |
| Fused SDMA, 1 chunk | 129.7 | **202.8** |
| Fused SDMA, 4 chunks | 157.7 | 236.0 |

Fused SDMA uses `--peer-uncached --fence release`; fused LSA uses uncached
stores. RCCL uses thread-local graph capture to allow its watchdog's event
queries on another thread.

For N=512, the leading single-stream paths are close (about 126–131 µs), with
LSA push now ahead of SDMA by 3%, and without a material fusion win. For
N=1024, split SDMA (203.2 µs) and one-chunk fusion (202.8) are within 0.2%;
four-chunk fusion at this small M is slower. RCCL fell further behind than in
September on both N. Do not compare these numbers directly with
the Pro table's different host/runtime, K, N and communication volume.

### Larger local M, SDMA

Times in µs, with the same K and tile:

| Local M | N=512 split | N=512 fused c4 | N=1024 split | N=1024 fused c4 |
|---|---:|---:|---:|---:|
| 4096 | 201.5 | 235.9 | 346.8 | 385.5 |
| 8192 | 346.2 | 382.3 | 669.0 | 705.6 |
| 16384 | 671.8 | 707.7 | 1314.7 | **1257.7** |

Four chunks do not help the individual N=512 projection. The N=1024 combined
candidate first shows a useful four-chunk gain at M=16384, 4.3% here; smaller
M does not show that benefit. Both agree with September's sweep.

The September study also confirmed the SDMA comparison with independent
adjacent pairs on v2-015 (N=512 uses one fused chunk; N=1024 uses four); these
were not repeated:

| N | Local M | Repeat | Split SDMA, µs | Fused SDMA, µs | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 512 | 2048 | 1 | 138.0 | 140.4 | -1.68% |
| 1024 | 16384 | 1 | 1370.1 | 1308.5 | 4.50% |
| 1024 | 16384 | 2 | 1369.6 | 1308.0 | 4.49% |

Thus the small N=512 result does not establish a fusion win, while the
combined N=1024 projection at M=16384 has a repeatable benefit: **about 4.5%**
in those pairs and 4.3% in the current single pass.

### Two-stream pipeline at local M=2048

| Four-chunk, two-stream backend | N=512, µs | N=1024, µs |
|---|---:|---:|
| MORI | 221.6 | 242.7 |
| MORI, four-way split-K | 144.4 | 202.3 |
| Native Torch | 130.3 | 192.6 |

These are total GEMM+AG pipeline times, not isolated GEMM times. The N=512
pipeline does not improve the better whole-matrix paths. The N=1024 Torch
pipeline (192.6 µs) is faster than MORI split SDMA (203.2) here, but that
comparison changes the GEMM backend as well as the scheduling; it does not
isolate overlap alone.

### Complete ratio-2 projection subgraph

The separate [Flash projection-order comparison](#flash-ratio-2-projection-order)
times both 512-dimensional outputs, includes packing/unpacking and restores
uniform interleave token order. It compares input-AG-first with local
projection-then-output-AG at total S=16384 and local M=2048. It uses CCO/SDMA
for every collective and is not an end-to-end SGLang measurement.

### Smaller-tile follow-up

An isolated [smaller-tile study](#smaller-tile-study) from the
September stack, not repeated here, tested
64×128, 128×64, 64×64, 32×64 and 32×32 layouts with fewer waves. At local
M=2048, the selected Flash non-fused paths improved by 7.7% (N=512) and 2.4%
(N=1024) in adjacent repeats. Fusion did not beat the tuned non-fused path.
A 96 KiB LDS reservation demonstrated a gain relative to its own slower
baseline, but still lost overall. These prototypes are archived separately;
they do not extend the production API's supported tile sizes.

## Correctness requirements

The implementation depends on four distinct ordering contracts:

1. **LDS reuse and the K tail.** Drain S2R reads with `lgkmcnt(0)` before the
   barrier that permits B0 overwrite, and drain final G2S prefetches before
   the last tile's reads. Preserve the staggered-wave closing barrier.
   The asymmetric tile's partial wait is bounded by `min(2A+B, A+2B)`, where
   A/B count G2S load instructions per subtile.
2. **Initial state.** Initialize the entire window, including arrival flags
   and counters, before first use; counters remain monotonic across replays.
3. **SDMA completion.** Fused producers may submit different chunks on
   different queues. Drain every relevant queue before publishing arrival.
   Joining the submission stream only establishes that PUTs were submitted.
4. **Recv reuse.** The gathered result *is* the receive buffer, and the
   end-of-call barrier says every rank has written it, not that every rank has
   read it. Each call posts a credit to every peer at its start and waits for
   a destination's before pushing to it -- in the fused GEMMs' epilogues and in
   `gather` (`_credit.py`). Without it a rank a call ahead overwrites a result
   a slower rank is still reading; the benchmark replays identical operands, so
   a stale read there was the right answer and nothing caught it.

Coverage: [`test_gemm_ag.py`](../../../../tests/python/cco/test_gemm_ag.py)
holds everything that needs no process group -- the layout arithmetic, the
LDS dependency model and the SDMA completion model (both execute the real
kernel bodies against a model), the benchmark's validation contract, and the
single-GPU GEMM numerics with exact small-integer references, changed inputs
and poisoned buffers, split-K checked per partial plane.
[`test_gemm_ag_op.py`](../../../../tests/python/cco/test_gemm_ag_op.py) runs
`GemmAgOp` on 8 ranks from one spawn: every precision and mode, every bf16
tile, chunk counts, back-to-back calls with a held rank with the credit on and
off, alternating M, `self_test` and its teeth.

Benchmark commands and options are maintained in the
[benchmark README](../../../../benchmark/cco/flydsl/gemm_ag/README.md).
Historical run archives are separate from the maintained source tree.

## Experiments

What was tried on the way to the defaults above, kept because the reasons a
thing did not pay are as useful as the numbers that did. The datasets differ in
host and stack; compare within one.

The original Pro measurements below were collected on 2026-09-28 on eight MI355X GPUs,
using the **DeepSeek-V4-Pro ratio-4 main compressor** dimensions N=2048, K=7168,
and 128x128 tiles. M is **per rank**. Except for the explicitly
lossy-wire cases, inputs are BF16 and accumulation, output and wire are FP32.
The separate [Flash subgraph](#flash-ratio-2-projection-order) uses its own
model dimensions and conditions. The sections above hold the main transport
comparison and the correctness requirements; the
[benchmark commands](../../../../benchmark/cco/flydsl/gemm_ag/README.md) use the
maintained kernel interfaces directly.

### Measurement conditions

| Dataset | Local M | Software |
|---|---|---|
| Original host | 2048 | PyTorch `2.10.0+rocm7.2.0.gitb6ee5fde`, FlyDSL 0.2.4 |
| v2-015 | 2048/4096/8192/16384 | ROCm 7.2.4, PyTorch `2.11.0+rocm7.2`, FlyDSL 0.2.4 |

Compare candidates with controls from the same host. Each case uses a fresh
eight-rank process group with no foreign GPU processes before or after it.

Everything in this file predates the SDMA changes #603 and #711. Only the
[larger-M study](#larger-m-on-v2-015) was re-measured with them, on
2026-10-19 on the host and stack of the tables above, as were those tables;
compare figures within one dataset only.
Five rounds use 50 warmups and 101 graph samples each. The result is the
median across rounds of the maximum per-rank sample median. Allocation,
compilation, graph capture and validation are excluded; complete AG and
cross-rank synchronization are included.

Same-precision runs check initial output and changed-input graph replays,
clone the received output before constructing references, and require finite
values and `relL2 < 1e-5`. The original 62-run suite included four local-C
no-PUT diagnostics and four lossy-wire runs; each passed its declared checks.
No-PUT diagnostics do not represent completed all-gathers. All 46 cases and
repeats in the v2-015 sweep passed, with worst relative L2 error `9.94e-7`.

### Larger M on v2-015

**Four-chunk single-stream fusion has a repeatable advantage at M=8192 and
M=16384. M=4096 still shows no benefit.** Independent adjacent pairs measured:

| Local M | Repeat | Fused variant | Split, us | Fused, us | Latency reduction |
|---|---:|---|---:|---:|---:|
| 4096 | 1 | sc_release c2 | 703.8 | 726.0 | -3.17% |
| 8192 | 1 | sc_release c4 | 1382.3 | 1314.3 | **4.92%** |
| 8192 | 2 | sc_release c4 | 1389.3 | 1317.5 | **5.17%** |
| 16384 | 1 | sc_release c4 | 2841.4 | 2526.1 | **11.10%** |
| 16384 | 2 | sc_release c4 | 2841.4 | 2523.7 | **11.18%** |

`sc_release` denotes `--peer-uncached --fence release`: `sc0|sc1` C stores and
a release-only system fence. At M=2048 on this host, split SDMA was 371.9 µs,
original fused c1 was 385.6 µs, and sc_release c1 was 375.2 µs. One chunk was
slower than split SDMA at every M tested.

The original c4 kernel also beat split SDMA in the larger-M first pass:
1378.8 → 1315.1 µs at M=8192 and 2840.8 → 2530.0 µs at M=16384. At M=16384,
sc_release c4/c8 were close (2526.0/2523.4 µs); c4 was selected for paired
confirmation. An M=4096 original-c4 run had an unstable 4137.3 µs median;
its independent repeat was 737.2 µs. That outlier did not reproduce and is
not used as a stable baseline; its cause remains unknown.

Re-measured on 2026-10-19 on the stack of the tables above, single pass:
split / sc_release c4 are 680.0 / 712.6 µs at M=4096 (c2: 690.8), 1338.0 /
1266.5 at M=8192 (**5.3%**) and 2760.6 / 2432.1 at M=16384 (**11.9%**) --
the same pattern.

Each rank sends 224/448/896 MiB at M=4096/8192/16384. The grid grows from
256 CTAs at M=2048 to 512/1024/2048 CTAs. The gains are consistent with more
computation remaining after early chunks become ready. Increasing M also
increases the total problem size; this sweep neither retunes tiles nor
measures larger-M LSA. No GPU timeline was captured to quantify overlap.

The separate **two-stream**, four-chunk pipelines measured:

| Local M | MORI GEMM, no split-K, us | MORI 4-way split-K + reduction, us | Native Torch GEMM, us |
|---|---:|---:|---:|
| 4096 | 669.0 | 663.8 | 652.8 |
| 8192 | 1245.6 | 1274.7 | 1245.9 |
| 16384 | 2462.8 | 2530.5 | 2470.1 |

The pipelines were faster than the single-stream candidates in this sweep.
At M=8192/16384, each chunk is large enough that extra split-K/reduction is
slower than plain MORI GEMM; plain MORI and native Torch are close. These
backend comparisons are five-round first passes, not independent paired
repeats. No gain over a native Torch whole-matrix control is claimed here.

### M=2048 on the original host

#### C stores and publication

| Variant | Initial measurement, us | Confirmation, us |
|---|---:|---:|
| Original fused SDMA | 376.7 | 377.3 |
| Release-only system fence | 373.4 | 374.9 |
| `sc0\|sc1` C stores | 366.4–368.6 | 366.4 |
| `sc0\|sc1` + release-only fence | 364.7 | 366.2 |

The cache policy supplies most of the improvement, bringing fused SDMA near
the 363–364 µs split control without a demonstrated win. Release-only fencing
retains system publication and counter `acq_rel` ordering. It removes the
producer fence's acquire half; it does not remove the release.

#### Chunk pipeline

A compute stream produces each chunk and records a ready event. A submission
stream waits on that event and posts SDMA while subsequent chunks compute.
Posts are serialized on queue 0 per peer, and the final drain completes AG.
Split-K writes `[splits,rows,N]` FP32 partials and reduces them into the output
window, preserving A/B row stride K. One partial workspace is reused.

The `serial` control computes all chunks before submitting transfers; it
still uses two streams. Initial results were:

| Compute | M chunks | K splits | Serial, us | Overlapped, us |
|---|---:|---:|---:|---:|
| MORI whole matrix | 1 | 1 | 364.1 | — |
| Native Torch whole matrix | 1 | 1 | 360.7 | — |
| MORI small-M GEMM | 4 | 1 | 532.9 | 376.3 |
| MORI split-K + FP32 reduction | 4 | 4 | 421.7 | 356.6 |
| Native Torch small-M GEMM | 4 | 1 | 429.0 | 349.9 |

Overlap saves 65.1 µs against split-K's own serial control, but most of that
pays for chunking and reduction relative to the whole-matrix path. Independent
pairs measured **362.8 → 357.6 µs** and **363.8 → 358.6 µs**, both a **1.43%**
latency reduction. Torch's four-chunk pipeline repeated at **350.0 µs**, about
3% below its 360.7 µs whole-matrix control. Eight M chunks with eight K splits
measured 380.5 µs and lost the benefit.

A shorter final drain alone does not prove useful overlap: slow submission
can give DMA a head start while extending the GEMM phase. End-to-end paired
comparisons determine the net gain.

### Other experiments

| Change | Result at M=2048 on the original host |
|---|---|
| One counter per chunk with wave broadcast | No material end-to-end gain: 377.3 → 377.0 µs in confirmation. Fewer active atomic lanes did not remove an instruction site. |
| Late metadata/descriptors or tensor-based C stores | Approximately 375–376 µs; no useful gain over the split control. No baseline register spills to eliminate. |
| LDS reorder for fused LSA peer stores | Matched uncached runs improved to 391.0–394.3 µs, still slower than split LSA push (~379 µs) and split SDMA (~363–364 µs). Cached results varied and do not establish a gain. |

These variants are archived experiments, not retained tuning interfaces.
The tensor-store experiment passed with C pointing to the registered receive
slot; the fused API still anchors its output to that window.

### Lossy wire

These experiments retain BF16 input GEMM and materialize FP32 output, but
narrow the communicated values. Conversion and widening are included in time.
They do **not** preserve native FP32-output accuracy.

| Wire / schedule | Bytes sent per rank | Total, us | relL2 vs FP32 reference |
|---|---:|---:|---:|
| FP32, whole matrix | 112 MiB | 364.7 | 9.93e-7 |
| BF16, whole matrix | 56 MiB | 261.3 | 1.66e-3 |
| FP8, whole matrix | 28 MiB | 247.2 | 2.65e-2 |

BF16/FP8 wire reduced latency by 28.4%/32.2% in this fixture. Transport also
passed an exact comparison against each source's encoded/decoded data, so
the reported error is narrowing error. FP8 used unscaled E4M3FN, with no
dynamic scaling, saturation coverage or model-quality evaluation. The
four-chunk split-K versions were slower: 270.2 µs for BF16 and 292.9 µs for
FP8. These are separate accuracy/performance tradeoffs, not same-precision
fusion wins.

### Flash ratio-2 projection order

This separate test uses **DeepSeek-V4.1-Flash** dimensions K=5120 and
head_dim=512 on v2-015 (ROCm 7.2.4, PyTorch 2.11.0+rocm7.2, FlyDSL 0.2.4).
There are eight ranks, local M=2048 and total S=16384, with synthetic BF16
inputs and replicated weights. Each rank must finish with two separate,
contiguous FP32 `[S,512]` KV and gate buffers in global token order.

The input-AG-first order matches the structure of SGLang's low-ratio CP
branch, but this is an isolated subgraph using **CCO/SDMA for every AG** and
native `torch.mm` for the Torch projections. It does not profile SGLang's
collective implementation or full runtime. Compression pooling, normalization,
RoPE, cache writes and scheduling are excluded.

All schedules include explicit uniform-interleave row restoration. The two
local Torch GEMMs also include packing their outputs before AG. Combined
projections include splitting/reordering the gathered `[S,1024]` output into
the same contiguous KV/gate buffers. Weights and buffers are prepared outside
timing. The protocol is five rounds, 50 warmups and 101 samples, with immediate
output snapshots and changed-input validation against the same reference.

| Schedule | GEMM rows × calls per rank | Sent per rank | Total, µs |
|---|---|---:|---:|
| Gather BF16 inputs, then two global Torch projections | 16384 × 2 | 140 MiB | 657.3 |
| Two local Torch projections, pack, then output AG | 2048 × 2 | 56 MiB | 255.0 |
| One local combined Torch projection, then output AG | 2048 × 1 | 56 MiB | 238.2 |
| One local combined MORI projection, then output AG | 2048 × 1 | 56 MiB | 257.3 |
| Combined MORI GEMM + fused SDMA, four chunks | 2048 × 1 | 56 MiB | 280.5 |

Moving the two projections before AG reduced this subgraph's latency by
61.2%. It reduces per-rank communication from 140 to 56 MiB and avoids
repeating the global projection on every rank. Combining the local Torch
projections reduces the total further to 238.2 µs. These benefits change the
operation order, GEMM workload and packing; they are not pure kernel-fusion
gains and must not be reported as model-level speedups.

The adjacent same-backend MORI comparison was **257.3 → 280.5 µs**:
four-chunk fusion was **9.0% slower** at local M=2048. Its separate first
pass measured 281.6 µs and agreed with the repeat. All five schedules passed
initial and changed-input checks, with relative L2 error below 1e-5.

The [Flash shape tables](#bf16-benchmarks-deepseek-v41-flash) separately
report N=512 per projection and N=1024 combined candidates over several M.
Their larger-M fusion benefit must not be transferred to this S=16384,
local-M=2048 case. Raw commands, scripts, source hashes and logs are archived at
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/v41-flash/`.

### Smaller tile study

Tested on 2026-09-28 on v2-015, eight MI355X GPUs, ROCm 7.2.4, PyTorch
2.11.0+rocm7.2 and FlyDSL 0.2.4. Local M=2048 throughout. Flash uses
N=512/1024, K=5120; the Pro spot check uses N=2048, K=7168. All inputs are
BF16 and output/wire are FP32. These are projection-then-AG tests, excluding
consumer layout conversion and full model execution.

The prototypes retain the repaired quad-subtile pipeline and SDMA protocol,
while changing wave geometry, cooperative LDS loading and output mapping.
The base sweep tests 128×128 with 8/4 waves, 64×128 and 128×64 with 4 waves,
64×64 with 4/2 waves, 32×64 with 2 waves, and 32×32 with 1 wave. The shipped
128×128 / 8-wave kernel is the control. No split-K is used. Fused paths use
uncached C stores and a release-only fence.

All 78 benchmark cases/repeats passed initial and changed-input checks,
using five rounds, 50 warmups and 101 samples, with the usual maximum across
ranks and median across rounds. Pure GEMM is timed separately. Validation
also covers 150 CPU dependency cases and 40 variant/K configurations on GPU
(K=128/192/5120/7168), each with three exact small-integer input generations.

#### Normal small tiles

Representative screening results below include selected chunk tuning. The
initial screen uses four fused chunks for every geometry, preserving the
same 28 PUTs per rank; selected candidates also test one and two chunks.
Split SDMA sends the whole slab with seven PUTs. No tile comparison changes
the output bytes for its shape.

| Flash N | Tile | Waves | GEMM, µs | GEMM + SDMA, µs | Fused, µs | Fused chunks |
|---:|---|---:|---:|---:|---:|---:|
| 512 | 128×128 | 8 | 50.6 | 139.5 | 163.4 | 4 |
| 512 | 64×128 | 4 | 48.9 | 136.7 | 166.2 | 4 |
| 512 | 64×64 | 4 | 43.8 | 131.0 | 134.1 | 1 |
| 512 | 32×64 | 2 | 45.8 | 139.1 | 169.3 | 4 |
| 1024 | 128×128 | 8 | 53.1 | 214.2 | 248.9 | 4 |
| 1024 | 128×64 | 4 | 47.9 | 214.0 | 209.8 | 1 |
| 1024 | 64×64 | 2 | 49.8 | 208.6 | 221.7 | 1 |
| 1024 | 32×32 | 1 | 64.2 | 223.2 | 288.0 | 4 |

Independent adjacent comparisons of the original tile's non-fused path with
the selected smaller tile's non-fused path confirmed:

- N=512: **143.4 → 132.3 µs (7.7% lower latency)**, with 64×64 / 4 waves.
- N=1024: **214.1 → 208.9 µs (2.4% lower latency)**, with 64×64 / 2 waves.

The small-tile fused paths did not beat the best non-fused result. At N=1024,
128×64 / 4-wave fusion with one chunk (209.8 µs) is close to the 64×64 / 2-wave
non-fused result, without an established advantage. Reducing only wave count
at 128×128 did not help. At N=1024, 32×32 / 1 wave slowed pure GEMM to 64.2 µs
and four-chunk fusion to 288.0 µs.

The Pro spot check also did not improve: the original tile measured
GEMM/split/fused-c4 at 71.5/377.5/408.6 µs; 64×64 / 2-wave measured
89.3/393.3/457.3 µs. This is a targeted check, not a complete Pro tile search.

#### Forcing additional execution rounds

The device reports **160 KiB LDS per CU**. Smaller tiles also reduce per-CTA
LDS: 64 KiB at 128×128, 48 KiB at the asymmetric tiles, 32 KiB at 64×64,
24 KiB at 32×64 and 16 KiB at 32×32. More CTAs therefore do not automatically
mean more serial rounds; multiple CTAs may reside on a CU.

Two diagnostic variants reserve **96 KiB per CTA**, limiting residency to
at most one CTA per CU. A/B LDS offsets and GEMM arithmetic are unchanged;
a single volatile write touches the last padding dword so the allocation
cannot disappear. These reservations deliberately sacrifice concurrency and
are not proposed as a production optimization. Emitted ISA confirms the LDS
sizes and no register spills for all ten prototype configurations.

| Shape | Tile / waves, 96 KiB LDS | GEMM, µs | Split SDMA, µs | Fused c4, µs |
|---|---|---:|---:|---:|
| Flash N=512 | 32×64 / 2 | 76.5 | 165.5 | 160.6 |
| Flash N=1024 | 32×64 / 2 | 137.4 | 297.2 | 241.4 |
| Flash N=1024 | 64×64 / 4 | 72.4 | 231.7 | 237.5 |
| Pro N=2048 | 32×64 / 2 | 364.4 | 668.7 | 509.4 |

For Flash N=1024, the 32×64 reservation increases pure GEMM from 56.4 to
137.4 µs. Fusion nevertheless improves against that same slow configuration:
an independent pair measured **297.0 → 243.8 µs (17.9% lower latency)**.
It remains **16.7% slower** than the tuned non-fused 208.9 µs path. The Pro
reservation likewise benefits from fusion relative to its own 668.7 µs split
control, but its 509.4 µs fused result loses to the original 377.5 µs split path.

This supports the local tradeoff: more computation can give communication
more opportunity to overlap. It does not make the communication itself
smaller, and delaying output readiness or reducing compute throughput can
outweigh the overlap. These are net end-to-end comparisons; no timing trace
was captured to assign an exact amount of hidden communication.

The practical gain in this study is the faster Flash **non-fused** small-tile
path. The prototypes and full records are archived outside the repository;
the production kernel's supported tiles and defaults are unchanged. Archive:
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/small-tile/`.

### Short final chunk (#748)

`gemm_ar` gained an opt-in short tail in #748 (`chunk_bands`): chunks of a
fixed number of row tiles, the last taking the rest, so a chunk count need not
divide the tiles. It was ported to both fused epilogues here and measured on
2026-10-29 alongside the [re-measured tables](#re-measured-after-merging-main-2026-10-29)
(crsuse2 v3-054, same protocol, default fence, µs; `bN` is N row tiles per
chunk):

| Shape | M | split | best even | short tails |
|---|---:|---:|---:|---|
| Pro, N=2048 | 2048 | **372.9** | 385.7 (c1) | b12 398.4, b6 407.4 |
| | 4096 | **705.8** | 718.9 (c2) | b24 746.7, b11 729.6 |
| | 8192 | 1381.5 | **1315.9** (c4) | b48 1423.2, b22 1342.4 |
| | 16384 | 2844.2 | **2528.0** (c8) | b96 2840.2, b43 2626.6 |
| Flash, N=1024 | 2048 | **212.8** | 218.0 (c1) | b12 229.9 |
| | 16384 | 1375.2 | **1311.1** (c4) | b96 1409.8 |

No short tail beat the best even split, and the head-heavy ones (b48, b96:
one large chunk then a quarter) are as slow as not fusing. That is what the
AG's balance predicts. A short tail shortens the transfer exposed after the
GEMM ends, which is what `gemm_ar` was paying at its single-chunk shapes; here
the transfer is several times the GEMM (Pro M=2048: 69 µs of GEMM against
about 300 of AG), so the end time is set by when the link starts and how long
it stays busy, not by the last chunk. A large first chunk delays the start.
The port was not kept; it validated on both GEMMs before it was dropped.

### Reproduction and scope

The LSA modes (`split-lsa-push`, `split-lsa-pull`, `fused-lsa`) that several
studies above compare against have since been removed from the operator and
the benchmark; reproducing those rows needs this branch's history before the
removal. Use `bench_gemm_ag.py` for single-stream transports and
`bench_gemm_ag_pipeline.py` for two-stream MORI, split-K and Torch backends;
see the [command reference](../../../../benchmark/cco/flydsl/gemm_ag/README.md).
The retained interfaces include local `split_k=2/4/8` partials,
`build_sdma_chunk_post` and `fence="release"`. Defaults were not changed.
`build_sdma_chunk_post` serves the two-stream pipeline experiment only and
carries no recv-reuse credit; `GemmAgOp` does not use it.

Historical generators, run plans and raw datasets are archived separately.
The v2-015 image was `lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260919`;
its logs are under
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/`.
