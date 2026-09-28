# GEMM + All-Gather optimization results

The original Pro measurements below were collected on 2026-09-28 on eight MI355X GPUs,
using the **DeepSeek-V4-Pro ratio-4 main compressor** dimensions N=2048, K=7168,
and 128x128 tiles. M is **per rank**. Except for the explicitly
lossy-wire cases, inputs are BF16 and accumulation, output and wire are FP32.
The separate [Flash subgraph](#flash-ratio-2-projection-order) uses its own
model dimensions and conditions. The [operator README](README.md) contains the main transport comparison and
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

## Flash ratio-2 projection order

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

The [Flash shape tables](README.md#bf16-benchmarks-deepseek-v41-flash) separately
report N=512 per projection and N=1024 combined candidates over several M.
Their larger-M fusion benefit must not be transferred to this S=16384,
local-M=2048 case. Raw commands, scripts, source hashes and logs are archived at
`v2-015:/mnt/m2m_nobackup/feiyzhai/mori-gemm-ag-m-sweep-20260928/v41-flash/`.

## Smaller tile study

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

### Normal small tiles

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

### Forcing additional execution rounds

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
