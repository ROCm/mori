# GEMM + All-Gather

`mori.ops.gemm_ag` computes `C = A @ B.T` from local `A [M,K]` and replicated
`B [N,K]`, then gathers C along rows. Every rank receives `[world_size*M,N]`,
with source rank `r` at rows `[r*M,(r+1)*M)`. M is the row count **per rank**.
FP8 inputs produce BF16 output; BF16 inputs support BF16 or FP32 output.
Communication uses the output dtype.

See [benchmark commands](../../../../benchmark/cco/flydsl/gemm_ag/README.md)
for setup and execution, and [EXPERIMENTS.md](EXPERIMENTS.md) for optimization
results. The retained measurements show:

- For the DeepSeek-V4-Pro ratio-4 main compressor, at M=N=2048, K=7168,
  BF16-to-FP32 GEMM with an explicit `128x128` tile measures **69.4 µs**
  against native Torch's **77.8 µs** on the current stack (72.0 vs 71.1 in
  September -- Torch moved, not MORI). The default remains `256x256`.
- At that shape, non-fused GEMM + SDMA takes **355.9 µs**; fused SDMA takes
  **360.6 µs** with one chunk.
- At larger M the same shape gains from four-chunk single-stream fusion:
  **5.3% at M=8192** and **11.9% at M=16384**, with no benefit at M=4096
  (re-measured 2026-10-19; September's v2-015 pairs gave 4.9–5.2% and 11.1%).
  Fusion performance depends on shape and chunk size.

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

## BF16 benchmarks: DeepSeek-V4-Pro

Model shape: **DeepSeek-V4-Pro ratio-4 main compressor**, M=N=2048, K=7168,
BF16 inputs and FP32 output. Measured on 2026-10-19 on **mi355-gpu-51, eight
MI355X GPUs**, ROCm 7.14, PyTorch `2.12.0+rocm7.14.0`, FlyDSL 0.3.4.1, MORI
`main` at `8ae8156` (which includes the SDMA changes #603 and #711).
The first tables were taken on 2026-09-28 with PyTorch 2.10 / ROCm 7.2 /
FlyDSL 0.2.4; the host and the whole stack changed between the two, so a
difference between them is not attributable to MORI alone. The
[v2-015 controls](EXPERIMENTS.md#larger-m-on-v2-015) belong to that earlier
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
| GEMM + LSA push | 438.7 | 372.2 |
| GEMM + LSA pull | 443.8 | 399.2 |
| Fused LSA, cached stores | 466.9 | 413.7 |
| Fused LSA, uncached stores | 467.2 | 422.6 |
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
larger-M fusion results are in [EXPERIMENTS.md](EXPERIMENTS.md).

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
| GEMM + LSA push | 212.6 | 248.3 | 244.2 |
| GEMM + LSA pull | 217.0 | 251.0 | 248.8 |
| Fused LSA | 265.8 | 328.1 | 292.4 |
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
| MORI GEMM + LSA push | **126.4** | 206.5 |
| MORI GEMM + LSA pull | 138.0 | 220.1 |
| Fused LSA, uncached | 128.4 | 221.2 |
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

The separate [Flash projection-order comparison](EXPERIMENTS.md#flash-ratio-2-projection-order)
times both 512-dimensional outputs, includes packing/unpacking and restores
uniform interleave token order. It compares input-AG-first with local
projection-then-output-AG at total S=16384 and local M=2048. It uses CCO/SDMA
for every collective and is not an end-to-end SGLang measurement.

### Smaller-tile follow-up

An isolated [smaller-tile study](EXPERIMENTS.md#smaller-tile-study) from the
September stack, not repeated here, tested
64×128, 128×64, 64×64, 32×64 and 32×32 layouts with fewer waves. At local
M=2048, the selected Flash non-fused paths improved by 7.7% (N=512) and 2.4%
(N=1024) in adjacent repeats. Fusion did not beat the tuned non-fused path.
A 96 KiB LDS reservation demonstrated a gain relative to its own slower
baseline, but still lost overall. These prototypes are archived separately;
they do not extend the production API's supported tile sizes.

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
