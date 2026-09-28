# GEMM + All-Gather optimization results

These measurements were collected on 2026-09-28 on eight MI355X GPUs, using
N=2048, K=7168 and 128x128 tiles. M is **per rank**. Except for the explicitly
lossy-wire cases, inputs are BF16 and accumulation, output and wire are FP32.
The [operator README](README.md) contains the main transport comparison and
correctness requirements; [benchmark commands](../../../../benchmark/cco/flydsl/gemm_ag/README.md)
use the maintained kernel interfaces directly.

## Measurement conditions

| Dataset | Local M | Software |
|---|---|---|
| Original host | 2048 | PyTorch `2.10.0+rocm7.2.0.gitb6ee5fde`, FlyDSL 0.2.4 |
| v2-015 | 2048/4096/8192/16384 | ROCm 7.2.4, PyTorch `2.11.0+rocm7.2`, FlyDSL 0.2.4 |

Compare candidates with controls from the same host. Each case uses a fresh
eight-rank process group with no foreign GPU processes before or after it.
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

## Larger M on v2-015

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

## M=2048 on the original host

### C stores and publication

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

### Chunk pipeline

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

## Other experiments

| Change | Result at M=2048 on the original host |
|---|---|
| One counter per chunk with wave broadcast | No material end-to-end gain: 377.3 → 377.0 µs in confirmation. Fewer active atomic lanes did not remove an instruction site. |
| Late metadata/descriptors or tensor-based C stores | Approximately 375–376 µs; no useful gain over the split control. No baseline register spills to eliminate. |
| LDS reorder for fused LSA peer stores | Matched uncached runs improved to 391.0–394.3 µs, still slower than split LSA push (~379 µs) and split SDMA (~363–364 µs). Cached results varied and do not establish a gain. |

These variants are archived experiments, not retained tuning interfaces.
The tensor-store experiment passed with C pointing to the registered receive
slot; the fused API still anchors its output to that window.

## Lossy wire

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

## Reproduction and scope

Use `bench_gemm_ag.py` for single-stream transports and
`bench_gemm_ag_pipeline.py` for two-stream MORI, split-K and Torch backends;
see the [command reference](../../../../benchmark/cco/flydsl/gemm_ag/README.md).
The retained interfaces include local `split_k=2/4/8` partials,
`build_sdma_chunk_post` and `fence="release"`. Defaults were not changed.

Historical generators, run plans and raw datasets are archived separately.
The v2-015 image was `lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260919`;
its logs are under
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/`.
