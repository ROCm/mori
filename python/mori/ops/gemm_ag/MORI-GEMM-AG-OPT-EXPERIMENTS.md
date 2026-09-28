# GEMM + All-Gather optimization experiments

Experiments on 2026-09-28, starting from `f1397867` plus the all-queue SDMA
drain correction in the worktree. Hardware is 8x MI355X (gfx950), with
M=N=2048 and K=7168. The main experiments use BF16 inputs, FP32 accumulation,
FP32 output and FP32 wire. All GEMM tiles below are 128x128.

**Larger-M update on v2-015:** single-stream four-chunk fused SDMA reduces
latency by about **5% at M=8192** and **11% at M=16384** in independent paired
repeats. M=4096 still shows no fusion benefit. See
[the larger-M sweep](#6-increasing-local-m-on-v2-015) for its separate host,
environment, full matrix and two-stream controls.

**At the original M=2048 workload, the useful result is a modest pipeline improvement.**
A four-chunk, four-way split-K MORI prototype measured 356.6–358.6 us,
against 362.8–364.1 us for the whole-matrix MORI + SDMA control. Two final
independent paired repeats both reduced latency by **1.43%**. A Torch
small-M GEMM + SDMA pipeline measured about **350.0 us**. These are
multi-kernel graph prototypes, not replacements for the default fused kernel.

The original monolithic fused-SDMA kernel can be brought much closer to the
split baseline by changing C's cache policy and publication fence, but it did
not reliably beat that baseline. Single-counter election and metadata
lifetime changes did not provide a meaningful end-to-end benefit.

## Protocol and artifacts

Each case runs in a separate eight-rank process group, after checking that
there are no other GPU processes. Timing uses five rounds, 50 warmups per
round and the median of 101 graph event samples on each rank. The result is
the median of the five per-round maxima across ranks. No allocation,
compilation, graph capture, or reference construction is included in timing.

End-to-end cases check their initial result and changed-input graph replays. The
received tensor is cloned immediately after replay, before reference GEMMs
can hide a late transfer. Finite values and relative L2 error are checked on
all ranks. Same-precision cases use `relL2 < 1e-5`. `--no-put` diagnostics
validate only the local C and are not reported as completed all-gathers.

The consolidated dataset contains 62 case/repeat runs: 54 same-precision end-to-end runs,
four local-C `--no-put` diagnostics, and four lossy-wire runs. All passed
their declared checks. The lossy cases additionally check that the widened
received data exactly matches the encoded/decoded source tensors from all
ranks; their numerical approximation error is reported separately.

The maintained [benchmark entry points](../../../../benchmark/cco/flydsl/gemm_ag/README.md)
call the MORI kernels directly. Selected functionality now lives in
`compile_bf16_gemm_ag(split_k=...)`, `build_sdma_chunk_post`, and the fused
producer's `fence="release"` option. The historical `sc_release` variant maps
to `--peer-uncached --fence release` in `bench_gemm_ag.py`.

The source-rewriting generators, negative-variant runners, saved run plans,
and raw JSON datasets are archived separately. They are not required to run
the retained implementations. The following tables preserve the historical
measurements; no experimental configuration was made the default.

## 1. One counter per chunk, then wave broadcast

The existing lane-parallel path maintains a counter for each of seven peers.
The variant lets lane 0 increment a single counter, broadcasts its result
within the wave, and retains seven-lane parallel PUT submission. System
publication and the counter's acquire/release ordering are retained.

| Case | Baseline, us | Single counter, us |
|---|---:|---:|
| One chunk, initial pair | 376.7 | 377.4 |
| One chunk, confirmation | 377.3 | 377.0 |
| Four chunks | 409.8 | 409.1 |
| One chunk, no-PUT diagnostic | 96.6 | 94.8 |

**No meaningful end-to-end improvement.** Logical atomic updates per rank
fall from 1792 to 256, but the ISA still contains one vector atomic-add
instruction site in either kernel. The active-lane count changes; this is not
seven serial instructions becoming one. The variant adds two
`v_readfirstlane_b32` instructions and retains the same register usage.
The diagnostic improvement does not translate into a clear end-to-end win.

## 2. Metadata, C stores, and publication

Variants separately move window/descriptor construction after the mainloop,
write through the benchmark's C tensor, change C stores to `sc0|sc1`, and
replace the producer's sequentially consistent system fence with a
release-only system fence. The last variant keeps the existing leader choice
and counter `acq_rel`; it removes the producer's unnecessary acquire half,
not the system release.

| Variant | Initial measurement, us | Confirmation, us |
|---|---:|---:|
| Original fused SDMA | 376.7 | 377.3 |
| Late metadata/descriptors | 376.2 | — |
| C tensor store | 375.4 | — |
| Late metadata + C tensor store | 376.2 | 376.2 |
| Release-only system fence | 373.4 | 374.9 |
| `sc0|sc1` C stores | 366.4–368.6 | 366.4 |
| `sc0|sc1` + release-only fence | 364.7 | 366.2 |
| Single counter + `sc0|sc1` + release | 365.2 | — |

**C's cache policy supplies most of the improvement.** The best variants
remove most of the original 13 us disadvantage, but do not establish a win
over the 363–364 us split-SDMA control. Adding single-counter election does
not improve the best result.

The tensor-store variant validates when C points at the rank's own window
slot. The old corruption observation in the source predates the LDS repair;
it is not reproduced here. This does not change the fused API's destination
contract, which is anchored to the registered window.

Offline gfx950 ISA at this tile:

| Variant | VGPR | SGPR | Register spills | `buffer_wbl2` sites | `buffer_inv` sites |
|---|---:|---:|---:|---:|---:|
| Original fused SDMA | 86 | 40 | 0 | 2 | 2 |
| Single counter | 86 | 40 | 0 | 2 | 2 |
| Late metadata + tensor store | 88 | 35 | 0 | 2 | 2 |
| `sc0|sc1` + release-only | 86 | 40 | 0 | 2 | 1 |

All use 64 KiB LDS. There is no spill to eliminate in the baseline. The
release-only variant removes one invalidate; the counter's ordered atomic
still has its own cache-ordering operations. The no-PUT diagnostics measured
96.6 us originally, 94.8 us with one counter, 93.5 us with a release-only
fence, and 88.5 us with `sc0|sc1` stores. These diagnostics include the empty
collective drain/barrier and must not be read as isolated GEMM durations.

## 3. Produce and transmit M chunks in a real pipeline

The prototype has a compute stream and a submission stream. After each M
chunk is computed, an event allows a small kernel to submit that chunk to
SDMA while the next chunk computes. PUT submission is serialized on the
submission stream, so one queue per peer suffices. The final drain waits for
all transfers and performs the cross-rank barrier.

The split-K prototype partitions K using a second grid dimension, retains
the original A/B row stride, writes separate FP32 partial outputs, and uses
an FP32 reduction into the chunk's output. It does not pack or cast the BF16
inputs. A single partial workspace is reused after each reduction.

A serial control computes all chunks before submitting any transfers, while
retaining the same chunk GEMMs/reductions and PUT sizes.

| Compute | M chunks | K splits | Serial, us | Overlapped, us |
|---|---:|---:|---:|---:|
| MORI whole matrix | 1 | 1 | 364.1 | — |
| Native Torch whole matrix | 1 | 1 | 360.7 | — |
| MORI small-M GEMM | 4 | 1 | 532.9 | 376.3 |
| MORI split-K + FP32 reduction | 4 | 4 | 421.7 | 356.6 |
| MORI split-K + FP32 reduction | 2 | 2 | — | 357.5 |
| MORI split-K + FP32 reduction | 8 | 8 | — | 380.5 |
| Native Torch small-M GEMM | 4 | 1 | 429.0 | 349.9 |
| Native Torch small-M GEMM | 2 | 1 | — | 351.8 |

**Overlap is real, but its net benefit is small after paying for small-M
compute and reduction.** The four-chunk split-K version saves 65.1 us
against its own serial control. Against the faster whole-matrix control,
final paired repeats were **362.8 → 357.6 us** and **363.8 → 358.6 us**:
1.43% less latency in both pairs. The Torch four-chunk result repeated at
350.0 us, about 3.0% below its whole-matrix Torch + SDMA control.

Naively splitting M without improving the small-M GEMM recovers much of its
large serial penalty through overlap, but remains slower than the original
whole-matrix path. Eight chunks and eight K partitions add enough overhead
to lose the improvement. The largest observed split-K relative error was
9.40e-07; the changed accumulation order stays within the FP32 check.

## 4. Reorder the LSA epilogue through LDS

This experiment is restricted to the 128x128 FP32 tile. After all operand
LDS reads finish, it reuses the existing 64 KiB A/B LDS allocation for the
64 KiB C tile. One barrier protects that reuse; another separates C staging
writes from the linear readout. Each wave then writes longer contiguous
row segments to every peer. A second variant XOR-swizzles the LDS column
index to distribute staging accesses across banks.

| Matched uncached-store comparison | Baseline, us | Reordered, us | Reduction |
|---|---:|---:|---:|
| Linear LDS, confirmation | 419.9 | 395.9 | 5.7% |
| Swizzled LDS, first matched pair | 419.9 | 391.0 | 6.9% |
| Swizzled LDS, second matched pair | 419.3 | 394.3 | 6.0% |

**The layout change improves fused LSA, but does not beat split LSA push
(~379 us) or split SDMA (~363–364 us).** Register usage changes from
86 VGPR / 50 SGPR to 80 VGPR / 56 SGPR, with no spills and no extra LDS.
Both emit 64 vector buffer-store instruction sites; the address layout and
LDS traffic change. The `sc0|sc1` policy was confirmed in the emitted stores.

Cached LSA showed substantial variation across runs, including a baseline
that did not reproduce. It is not used to claim a cached-path speedup here.
The conclusion above uses the repeated uncached controls and candidates.

## 5. Separate lossy-wire feasibility experiment

These cases retain BF16 input GEMM and return an FP32 tensor, but communicate
BF16 or unscaled E4M3FN FP8 values. Quantization and widening are separate GPU
kernels and are included in the total. There is no claim that these outputs
are numerically equivalent to native FP32 output.

| Wire / schedule | Bytes sent per rank | Total, us | relL2 vs FP32 reference |
|---|---:|---:|---:|
| FP32, whole matrix | 112 MiB | 364.7 | 9.93e-7 |
| BF16, whole matrix | 56 MiB | 261.3 | 1.66e-3 |
| BF16, four-chunk split-K pipeline | 56 MiB | 270.2 | 1.66e-3 |
| FP8, whole matrix | 28 MiB | 247.2 | 2.65e-2 |
| FP8, four-chunk split-K pipeline | 28 MiB | 292.9 | 2.65e-2 |

BF16 and FP8 whole-matrix wire reduce latency by about 28.4% and 32.2%
respectively. FP8 saves only another 14.1 us over BF16 in this prototype,
while increasing relL2 about sixteenfold. The source maximum was 7.212,
below E4M3FN's maximum of 448, so this fixture did not exercise saturation.
No dynamic scale or model-quality evaluation was performed.

The transport itself passes an exact comparison against each source's
encoded/decoded tensor. The errors above are numerical narrowing errors,
not accepted missing transfers. The FP32 consumer materialization cost is
included; fusing conversion into the producer or consumer was not tested.
Narrower wire also makes the extra computation in the split-K pipeline less
profitable: the four-chunk versions are slower than the whole-matrix ones.

## 6. Increasing local M on v2-015

This follow-up ran on 2026-09-28 in container
`mori-gemm-ag-m-sweep-20260928` on `v2-015`, with eight MI355X GPUs,
ROCm 7.2.4, PyTorch `2.11.0+rocm7.2` and FlyDSL 0.2.4. The image is
`lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260919`. The original host used a
different ROCm/PyTorch environment, so all gains below use controls measured
on v2-015. Its new M=2048 controls were 371.9 us for split SDMA, 385.6 us for
original fused SDMA c1, and 375.2 us for `sc0|sc1` + release-only fused SDMA c1.

M is **per rank**. N=2048, K=7168, BF16 input, FP32 accumulation/output/wire,
and 128x128 tiles remain fixed. At M=4096/8192/16384, each rank produces
32/64/128 MiB and sends 224/448/896 MiB to its seven peers. This increases the
global problem size while retaining eight ranks. This sweep compares SDMA
paths; it does not retune GEMM tiles or measure LSA at the larger sizes.

The five-round, 50-warmup, 101-sample graph protocol and immediate-snapshot
changed-input validation are retained. Every case runs in a fresh process
group, with no foreign GPU processes before or after it. Attempts on the
previous, occupied host were excluded entirely. `sc_release` below means the
existing generated `sc0|sc1` C-store + release-only publication variant.

Single-stream first-pass results, in us:

| Local M | Split SDMA | Original fused c1 | Original fused c4 | sc_release c1 | sc_release c2 | sc_release c4 | sc_release c8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 709.6 | 742.1 | 4137.3* | 717.9 | 715.7 | 735.4 | 779.0 |
| 8192 | 1378.8 | 1451.9 | 1315.1 | 1426.0 | 1325.7 | 1314.7 | 1356.8 |
| 16384 | 2840.8 | 2967.1 | 2530.0 | 2955.3 | 2670.4 | 2526.0 | 2523.4 |

*The initial M=4096 original-c4 run was numerically correct but had per-round
maxima of 4108.8, 4137.3, 4290.1, 4324.9 and 740.8 us. It is retained as an
unstable measurement, not treated as the stable latency of that configuration.
An independent repeat measured **737.2 us**, with all five round maxima between
736.4 and 737.5 us. Additional M=2048 c4 controls measured 418.0 us for the
original kernel and 407.4 us for sc_release. The initial multi-millisecond
outlier did not reproduce; its cause was not established.

Independent adjacent split/fused pairs confirm the useful configurations:

| Local M | Repeat | Fused variant | Split, us | Fused, us | Latency reduction |
|---|---:|---|---:|---:|---:|
| 4096 | 1 | sc_release c2 | 703.8 | 726.0 | -3.17% |
| 8192 | 1 | sc_release c4 | 1382.3 | 1314.3 | **4.92%** |
| 8192 | 2 | sc_release c4 | 1389.3 | 1317.5 | **5.17%** |
| 16384 | 1 | sc_release c4 | 2841.4 | 2526.1 | **11.10%** |
| 16384 | 2 | sc_release c4 | 2841.4 | 2523.7 | **11.18%** |

**Single-stream fusion now wins at M=8192 and M=16384.** M=4096 still does
not show a benefit. The original c4 kernel also wins in the larger-size
first pass, so the crossover does not depend solely on the publication
optimization. At M=16384, c4 and c8 are close; c4 was selected for paired
confirmation because the extra queues/requests did not produce a material
first-pass improvement.

The grid grows from 256 CTAs at M=2048 to 512, 1024 and 2048 CTAs. With more
work after the first chunk becomes ready, early PUT submission can amortize
its bookkeeping and request overhead. This is consistent with the crossover
observed here. A single chunk still waits for all output producers, and is
slower at every size measured. These are end-to-end gains; no GPU timeline
was captured to assign an exact overlap duration or percentage.

The separate **two-stream** four-chunk prototypes measured:

| Local M | MORI GEMM, no split-K, us | MORI 4-way split-K + reduction, us | Native Torch GEMM, us |
|---|---:|---:|---:|
| 4096 | 669.0 | 663.8 | 652.8 |
| 8192 | 1245.6 | 1274.7 | 1245.9 |
| 16384 | 2462.8 | 2530.5 | 2470.1 |

These pipelines remain faster than the single-stream candidates in this
sweep, but split-K stops helping once each M chunk is large enough. At
M=8192/16384, plain MORI chunk GEMMs are faster than the extra split-K/reduction
path, and are close to native Torch. These backend comparisons are the first
pass of five rounds, not independently paired backend repeats. No gain against
a native Torch whole-matrix control is claimed for this new environment.

The [benchmark README](../../../../benchmark/cco/flydsl/gemm_ag/README.md) shows the
single-stream, two-stream and native Torch commands. Set `-m` to each local
row count and `--chunks` to the candidate under test. Raw historical results
remain in the archived run directory rather than the committed source tree.
Remote logs are under
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/`.
All **46** cases/repeats passed their initial and changed-input checks; the
largest relative L2 error was **9.94e-7** against the 1e-5 threshold. The
original experiment harness also passed its 30 CPU checks in both environments.
The maintained tests now cover the direct kernel interfaces
and SDMA completion contract.

## Outcome

At the original M=N=2048 workload, retain the single-counter and metadata
variants as negative experiments. The cache/publication variant reduces
fused-SDMA overhead but does not establish a win over the split baseline.
The same-precision direction worth further work is efficient small-M compute
plus a correctly synchronized transfer pipeline; the current MORI prototype
has a small confirmed gain, and the Torch prototype provides a stronger
reference target. LDS repacking helps LSA specifically, without changing the
preferred transport at this shape. Narrow wire is a separate accuracy/performance
decision and needs model-level evaluation before adoption.

The larger-M sweep changes that conclusion for single-stream SDMA: four-chunk
fusion has repeatable gains of about 5% at M=8192 and 11% at M=16384. Chunk
size and compute backend must be selected for the workload; the small-M
split-K prototype is not the best pipeline at the larger sizes. No operator
default was changed by these experiments.
