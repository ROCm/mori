# GEMM + All-Gather

`mori.ops.gemm_ag` computes `C = A @ B.T` from local `A [M,K]` and replicated
`B [N,K]`, then gathers C along rows. Every rank receives `[world_size*M,N]`,
with source rank `r` at rows `[r*M,(r+1)*M)`. M is the row count **per rank**.
FP8 inputs produce BF16 output; BF16 inputs support BF16 or FP32 output.
Communication uses the output dtype.

See [benchmark commands](../../../../benchmark/cco/flydsl/gemm_ag/README.md)
for setup and execution, and [EXPERIMENTS.md](EXPERIMENTS.md) for optimization
results. The retained measurements show:

- At M=N=2048, K=7168, BF16-to-FP32 GEMM with an explicit `128x128` tile is
  close to native Torch: **72.0 vs 71.1 µs**. The default remains `256x256`.
- At that shape, non-fused GEMM + SDMA takes **364.0 µs**; original fused
  SDMA takes **377.0 µs** with one chunk.
- In the separate v2-015 sweep, four-chunk single-stream fusion reduces
  latency by **about 5% at M=8192** and **11% at M=16384**, with no benefit
  at M=4096. Fusion performance depends on shape and chunk size.

## Execution paths

The window-based paths write C directly into the rank's own receive slot.
All peers receive the same contiguous slab, so no staging layout is needed.
Bytes sent per rank are `(world_size-1) * M * N * elem_bytes`.

| Benchmark mode | Execution |
|---|---|
| `gemm-only` | GEMM into a plain tensor |
| `gemm-to-window` | GEMM into the local receive slot |
| `split-rccl` | GEMM, then RCCL `all_gather_into_tensor` |
| `split-lsa-push` | GEMM, then a kernel stores C into peers |
| `split-lsa-pull` | GEMM publishes C, then a kernel reads peer slabs |
| `fused-lsa` | GEMM epilogue stores each tile into every peer |
| `split-sdma` | GEMM, then SDMA gather and completion barrier |
| `fused-sdma` | GEMM epilogue submits completed M chunks; a drain finishes AG |

The SDMA modes submit kernels on one stream. SDMA can still overlap a fused
producer's later computation. The separate M-chunk pipeline uses two streams;
its `serial` control retains those streams but disables compute/transfer overlap.

## Interfaces and constraints

| Source | Entry points |
|---|---|
| [_gemm_a16w16_8wave.py](_gemm_a16w16_8wave.py) | `compile_bf16_gemm_ag`, including local split-K partials |
| [kernels_fused.py](kernels_fused.py) | FP8 `compile_fused_gemm_ag` and `compile_gemm_local` |
| [kernels_lsa.py](kernels_lsa.py) | LSA push/pull and barriers |
| [kernels_sdma.py](kernels_sdma.py) | `build_sdma_phases` and `build_sdma_chunk_post` |
| [layout.py](layout.py) | `ag_config`, window offsets and counter layout |

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

## BF16 benchmarks after the correctness fixes

Measured on 2026-09-28 on eight MI355X GPUs, M=N=2048, K=7168, with BF16
inputs and FP32 output, using PyTorch `2.10.0+rocm7.2.0.gitb6ee5fde` and
FlyDSL 0.2.4.
Use the separate [v2-015 controls](EXPERIMENTS.md#larger-m-on-v2-015) when
comparing that host's measurements.

### Pure GEMM

Native Torch uses `torch.mm(a, b.T, out_dtype=torch.float32, out=c)`. MORI
uses `fuse=False, peer_uncached=False`. Outputs are preallocated; no AG or
peer-publication cost is included.

Five rounds rotate implementation order. Each captures one GEMM in a graph,
uses 100 warmups and 101 event samples, and takes the maximum of per-rank
medians; the reported time is the median across rounds.

| Implementation | Median, µs | Round range, µs | Latency vs Torch |
|---|---:|---:|---:|
| Native Torch BF16 → FP32 | 71.1 | 70.6–72.1 | baseline |
| MORI `256x256` (default) | 131.9 | 131.8–132.0 | +85.5% |
| MORI `128x256` | 88.6 | 88.2–88.7 | +24.6% |
| MORI `128x128` | 72.0 | 71.8–73.7 | +1.3% |

All implementations passed on all ranks, with worst relative L2 error
`9.95e-7`. The smaller tile is close to Torch at this shape; it does not
establish a speedup. Select it explicitly with `--block-m 128 --block-n 128`.
BF16 output followed by widening is not an equivalent FP32-output reference.

### GEMM + AG

Each rank produces 16 MiB and sends 112 MiB. Times include the complete AG
and final cross-rank synchronization. Native Torch + RCCL takes **397.5 µs**.

| Mode | `256x256`, µs | `128x128`, µs |
|---|---:|---:|
| GEMM + RCCL | 462.7 | 403.7 |
| GEMM + LSA push | 444.6 | 379.0 |
| GEMM + LSA pull | 448.3 | 406.3 |
| Fused LSA, cached stores | 542.4 | 509.2 |
| Fused LSA, uncached stores | 470.4 | 420.8 |
| **GEMM + SDMA** | **427.8** | **364.0** |
| Fused SDMA, 1 chunk | 429.0 | 377.0 |
| Fused SDMA, 2 chunks | 440.6 | 388.6 |
| Fused SDMA, 4 chunks | 462.2 | 409.9 |

These fused runs use the default leader fence and lane-parallel SDMA posting;
all multi-queue measurements use the corrected drain. Five rounds use 50
warmups and 101 graph samples each, taking per-round maxima across ranks
before the final median. Every configuration ran in a fresh process group
with no foreign GPU processes before or after it.

All 19 configurations passed initial output and two changed-input replays,
with finite output and `relL2 < 1e-5`; worst error was `9.94e-7`. The receive
buffer was cloned immediately after replay, before reference computation.
Allocation, compilation, capture and validation were outside timing.

At this shape, the benefit comes from tile selection and SDMA transport:
364.0 µs is 8.4% below native Torch + RCCL. The publication optimization and
larger-M fusion results are in [EXPERIMENTS.md](EXPERIMENTS.md).

### Earlier FP8 reference

These FP8-input/BF16-output measurements used eight MI355X GPUs,
M=N=2048, K=7168, and 21 graph samples after 10 warmups, taking the maximum
across ranks. PTPC uses `128x256`; MXFP8 uses `256x256`. This older protocol
is separate from the BF16/FP32 measurements above. Times are µs.

| Mode | PTPC | Blockscale | MXFP8 |
|---|---:|---:|---:|
| GEMM only | 57.3 | 96.3 | 86.3 |
| GEMM + RCCL | 231.8 | 266.9 | 266.0 |
| GEMM + LSA push | 212.6 | 248.3 | 244.2 |
| GEMM + LSA pull | 217.0 | 251.0 | 248.8 |
| Fused LSA | 265.8 | 328.1 | 292.4 |
| GEMM + SDMA | 212.5 | 245.4 | 242.2 |
| Fused SDMA, 1 chunk | 223.6 | 259.1 | 250.7 |

All modes validated at about `1.66e-3` relative L2 error against the
precision-specific reference. These results do not establish a fusion win
for that shape or a precision-equivalent replacement for BF16-to-FP32 GEMM.

## Correctness requirements

The repaired implementation depends on three distinct ordering contracts:

1. **LDS reuse and the K tail.** Drain S2R reads with `lgkmcnt(0)` before the
   barrier that permits B0 overwrite, and drain final G2S prefetches before
   the last tile's reads. Preserve the staggered-wave closing barrier.
   The asymmetric tile's partial wait is bounded by `min(2A+B, A+2B)`, where
   A/B count G2S load instructions per subtile.
2. **Peer publication and initial state.** BF16 pull uses a descriptor from
   the caller's C pointer, `sc0|sc1` stores and a system fence from every
   producer lane. Initialize the entire window, including arrival flags and
   counters, before first use; counters remain monotonic across replays.
3. **SDMA completion.** Fused producers may submit different chunks on
   different queues. Drain every relevant queue before publishing arrival.
   Joining the submission stream only establishes that PUTs were submitted.

Coverage includes the CPU [LDS dependency model](../../../../tests/python/cco/test_gemm_ag_pipeline.py),
the [SDMA completion model](../../../../tests/python/cco/test_gemm_ag_sdma_drain.py),
and [GPU numerical/replay tests](../../../../tests/python/cco/test_gemm_ag.py).
Numerical tests use exact small-integer CPU references, changed inputs and
poisoned buffers; pull tests also exercise delayed producers and dirty initial
flags. The direct split-K interfaces are checked per partial plane.

Benchmark commands and options are maintained in the
[benchmark README](../../../../benchmark/cco/flydsl/gemm_ag/README.md).
Historical run archives are separate from the maintained source tree.
