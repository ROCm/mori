# MORI GEMM + All-Reduce Benchmark

Measurements for `mori.ops.gemm_ar`, the fused fp8 GEMM + all-reduce. The design
and the API live next to the code in
[`python/mori/ops/gemm_ar/README.md`](../python/mori/ops/gemm_ar/README.md);
this file is how to reproduce the numbers and what they were.

Every number below was taken on **8x MI355X (gfx950)**, one node, at the shape
`[M, 7168]` with `K=2048` — DeepSeek-V4-Pro's `wo_b` under TP8 with
`--chunked-prefill-size 16384`. Kernel timings are the median over 11
graph-replayed iterations, maximum over ranks, on an otherwise idle box.
Run-to-run spread at this shape is about **2%**, so differences below that are
not differences.

## Table of Contents

- [Running the benchmark](#running-the-benchmark)
- [Headline](#headline)
- [Where the time goes](#where-the-time-goes)
- [The fp8 wire](#the-fp8-wire)
  - [Who moves the gather](#who-moves-the-gather)
  - [The pull grid](#the-pull-grid)
  - [What fp8 costs, numerically](#what-fp8-costs-numerically)
- [Model-level evaluation](#model-level-evaluation)
- [Negative results](#negative-results)
- [Reproducing the end-to-end numbers](#reproducing-the-end-to-end-numbers)

## Running the benchmark

Needs a mori built with `BUILD_CCO_SDMA=ON`. Setting `MORI_ENABLE_SDMA` in the
environment only rebuilds the *device* bitcode — a host library built without
the flag has no SDMA queues, every put silently does nothing, and the all-reduce
quietly produces zeros.

```bash
cd /path/to/mori
BUILD_CCO_SDMA=ON pip install .

MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048
```

`--mode` selects what is measured, all reaching the same end state:

| mode | what runs |
|---|---|
| `gemm-only` | the GEMM alone, to size the ceiling |
| `split-sdma` | `gemm` then a 4-kernel SDMA all-reduce |
| `fused-sdma` | GEMM with the scatter fused into its epilogue |
| `split-lsa` | `gemm` then the 2-kernel LSA all-reduce |
| `fused-lsa` | GEMM storing straight into peers |

`--quant blockscale` is the model's own quantisation (A 1x128, B 128x128, fp32
scales) and is the column to read. `--gather-dtype fp8` and
`--gather-transport {sdma,lsa}` select the wire; see [the fp8 wire](#the-fp8-wire).

Each run prints a `RESULT_JSON` line with `max_rank_time_us`, `rel_l2` and
`validated`, so a sweep can be parsed rather than eyeballed.

> **Between runs, give the SDMA queues time to drain.** Back-to-back 8-rank runs
> hit `hsaKmtCreateQueueExt` failures (`anvil.cpp:237`) if a previous run's ranks
> have not exited. ~10s is enough; a leftover server holding queues is not.

## Headline

`--quant blockscale`, median of 11:

| M | `split-sdma` | `fused-sdma` | `fused-sdma` + fp8 gather |
|---|---:|---:|---:|
| 4096 | 398.1 us | 351.0 us | **329.5 us** |
| 8192 | 722.0 | 621.1 | **539.3** |
| 16384 | 1472.9 | 1148.8 | **979.3** |

Fusing is worth **-22%** at M=16384; the fp8 gather a further **-15%**.

For scale, the same layer as the model runs it today (a separate GEMM then an
NCCL all-reduce) measures **1419.5 us** at M=16384, and the GEMM alone is
**369.4 us**.

## Where the time goes

Per layer, captured in SGLang over one 20000-token prefill (not the standalone
benchmark — this is the pipeline as the model drives it):

| phase | bf16 | fp8 / sdma | fp8 / lsa |
|---|---:|---:|---:|
| gemm | 441.9 us | 438.2 us | 441.3 us |
| drain | 180.2 | 170.9 | 204.8 |
| reduce | 41.7 | 42.5 | 43.1 |
| quantize | — | 11.8 | 11.9 |
| gather | 437.8 | 256.4 | 7.5 (barrier only) |
| dequantize | — | 61.0 | — |
| pull | — | — | 230.5 |
| **wo_b layer** | **1101.6** | **980.8** | **939.1** |
| | | -11.0% | **-14.7%** |

Two things to read out of the bf16 column:

* **`gather` is the bottleneck**, not `drain`. It moves 196 MiB at 470 GB/s,
  which is 7 xGMI links flat out, so halving its bytes halves its time.
* **`drain`'s apparent 1140 GB/s is not a bandwidth.** Seven links cannot do
  that. It is the tell that the scatter's pushes already went out from the GEMM
  epilogue and the drain is only waiting for the tail — which is why the scatter
  leg has far less to give than its byte count suggests, and why it is still
  bf16.

## The fp8 wire

`--gather-dtype fp8` sends the all-gather leg as e4m3 with one fp32 scale per
row. The reduce still accumulates in fp32 and `output` is still bf16; only the
wire changes. The scatter leg is unchanged — it carries partial sums that are
then added across every rank, so its error would compound rather than being a
single rounding.

### Who moves the gather

| `--gather-transport` | how |
|---|---|
| `sdma` | copy engines push, a second kernel widens |
| `lsa` (default) | CUs pull over xGMI and widen on the way in |

A copy engine has no ALU, so for SDMA the widening *cannot* be the same step: it
is a second kernel that reads the landed fp8 back out of local HBM, 98 MiB a
layer. A CU pull has those bytes in registers already.

| gather | M=16384 |
|---|---:|
| bf16 / sdma | 1150.7 us |
| fp8 / sdma | 1018.9 |
| **fp8 / lsa** | **957.3** |

The pull also removes fp8's small-M penalty. With the SDMA gather the two
conversion kernels were a fixed cost against a transfer that shrinks with M, so
fp8 measured **+2.3% (slower)** at M=4096. With the pull there is no fixed
conversion cost left and fp8 wins wherever fusing does.

### The pull grid

The single most important tuning parameter, and it is not obvious from the
source. These are *xGMI* reads, so the grid throttles outstanding remote
requests rather than covering HBM latency — it wants roughly a **tenth** of what
the local conversion kernels want.

| blocks | 16 | 24 | 32 | 48 | 64 | 80 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| us | 1261 | 1091 | 1006 | 962 | **959** | 963 | 1018 | 1138 | 1184 |

Flat from 48 to 80, steep either side. The first implementation launched 512 —
the grid the local quantize kernel uses — and **lost to SDMA by 17%**, which
looked like "LSA is the wrong transport" rather than "the grid is wrong".

Note this is also not `LSA_BLOCK_CAP`'s 24: that cap is for a kernel moving bf16
with no arithmetic, while this one moves half the bytes and dequantises them, so
it needs more waves in flight to keep the links fed.

Sweep it with `MORI_GEMM_AR_PULL_BLOCKS`:

```bash
for b in 16 24 32 48 64 80 128 256; do
  MORI_GEMM_AR_PULL_BLOCKS=$b MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
    torchrun --standalone --nproc_per_node=8 \
    benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py --mode fused-sdma \
    --quant blockscale --gather-dtype fp8 --gather-transport lsa \
    -m 16384 -n 7168 -k 2048
  sleep 10
done
```

### What fp8 costs, numerically

relL2 against an fp32 host reference goes from **2.35e-3** (bf16 wire, which is
bitwise exact through the collective) to **2.49e-2**.

That is a floor, not a tuning problem. e4m3 carries 3 mantissa bits, and scale
granularity barely moves it — measured in torch on a `[2048, 7168]` standard
normal payload:

| scale granularity | relL2 | scale bytes |
|---|---:|---:|
| per row (7168) | 2.646e-2 | 0.06% |
| per 512 | 2.631e-2 | 0.78% |
| per 256 | 2.609e-2 | 1.56% |
| per 128 | 2.572e-2 | 3.12% |
| per 32 | 2.399e-2 | 12.5% |

**200x the scale bytes buys 9%.** Per-row is therefore the right choice, and
~2.5e-2 is what fp8 costs.

## Model-level evaluation

The kernel-level cost above is large — 6.7x the bf16 wire. Whether it *matters*
is a different question, and it needs the model, so this section was measured in
SGLang on DeepSeek-V4-Pro at TP8. The fp8 path was verified live throughout:
`SGLANG_DEBUG_FUSED_WO_B_AR=1` logs relL2 per layer call during the very
requests being scored.

**At the layer**, relL2 against the unfused path, 488 layer calls:

| wire | min | median | max |
|---|---:|---:|---:|
| bf16 | 3.706e-3 | — | 4.046e-3 |
| fp8 / sdma | 1.435e-2 | **2.505e-2** | 2.654e-2 |
| fp8 / lsa | 2.153e-2 | **2.496e-2** | 2.683e-2 |

**At the model output**, it is not detectable. Scoring 10941 tokens of real
source text in a single prefill (mean logprob; lower is a worse model):

| run | mean logprob | ppl |
|---|---:|---:|
| bf16 | -1.184169 | 3.2680 |
| bf16, rerun | -1.182018 | 3.2609 |
| bf16, again | -1.180106 | 3.2547 |
| **fp8** | **-1.183895** | **3.2671** |
| fp8, with debug | -1.183358 | 3.2653 |

fp8 lands **inside the bf16 run-to-run band**. Paired per token against the same
bf16 run:

| pair | mean d | sd d | max abs d |
|---|---:|---:|---:|
| CONTROL bf16 rerun | +0.002151 | 0.229 | 3.42 |
| CONTROL bf16 again | +0.004063 | 0.230 | 4.05 |
| **TEST fp8** | **+0.000274** | **0.219** | **2.93** |

On every statistic, fp8 is closer to bf16 than bf16 is to itself.

That band is wide because **this model is already strongly non-deterministic**:
two bf16 runs disagree on ~48% of tokens by more than 0.01 logprob, and greedy
decode diverges within 10-20 tokens. The likely cause is the MoE stage-2
epilogue, which accumulates with `atomic_fadd`. This is also why greedy token
agreement is useless as a metric here — the bf16-vs-bf16 control is as divergent
as bf16-vs-fp8:

| pair | ~12k tok | ~24k tok |
|---|---:|---:|
| CONTROL bf16 vs bf16 | 35.9% | 15.6% |
| TEST bf16 vs fp8 | 28.1% | 23.4% |

Needle-in-a-haystack retrieval at 15140 tokens is **24/24 on both wires** —
saturated, so it bounds gross damage without resolving anything finer.

**What this does and does not say.** It says fp8 causes no gross degradation and
no measurable shift in next-token distribution on one scoring task. It does not
say quality is unaffected on long-chain reasoning, code or maths — that needs a
task benchmark, which has not been run. Note also that short prompts and decode
never reach this path at all (it engages only at M >= 4096), so only
long-prefill workloads are affected.

## Negative results

Kept because each reads as obviously right and the reason it is not cannot be
seen from the source.

**Folding the narrowing into the reduce** (`fuse_quantize=True`) saves a 28 MiB
re-read and a kernel launch, and costs 20 us:

| | us |
|---|---:|
| split reduce + quantize | 957.4 |
| fused, row stashed in registers | 977.1 |
| fused, row re-read | 982.0 |

Not register pressure — the re-reading variant keeps no stash and is no better.
It is the thread map: a per-row amax cannot be taken by a block holding only part
of a row, so fusing forces one-wave-per-row, where `sdma_reduce` walks packs with
a flat grid stride and streams a block through all 8 source slices at once.

**A CK-shaped 4-wave GEMM**, chasing a 22% gap against CK's block-scale kernel
at the same shape, reached CK's instruction mix and not its speed. Ten hypotheses
were falsified by measurement; hardware counters show identical `SQ_INSTS_MFMA`
(7,340,032) and `SQ_VALU_MFMA_BUSY_CYCLES` (234,881,024), VALU within 1%,
`MemUnitStalled` at approximately zero — but `SQ_WAIT_ANY` 156.0M against 120.4M.
A K-sweep puts the whole difference per-iteration: our fixed cost is *lower*
(51.6 us against 66.0), while each K-block costs 20.9 us against 14.5. The gap is
wait, not work. See commits `cc696762`, `54fef960`, `9f990637`, `1dead3fc`.

## Reproducing the end-to-end numbers

The SGLang numbers need the integration branch and a mori built with
`BUILD_CCO_SDMA=ON` on `PYTHONPATH`.

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
export PYTHONPATH=/path/to/mori-with-sdma
export SGLANG_OPT_FUSED_WO_B_AR=1
export SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER=1     # optional, the fp8 wire
export SGLANG_DEBUG_FUSED_WO_B_AR=1              # optional, logs per-layer relL2

sglang serve --model-path <DeepSeek-V4-Pro> --tp 8 \
  --attention-backend dsv4 --page-size 256 --chunked-prefill-size 16384 \
  --mem-fraction-static 0.88 --kv-cache-dtype fp8_e4m3 \
  --enforce-shared-experts-fusion
```

`--mem-fraction-static` has to leave room for the symmetric window, which is VMM
memory **outside** torch's allocator: 700 MiB on the bf16 wire and 812 MiB on
fp8, which needs the extra staging region.

The capture protocol matters more than it looks. Profile a *different* prompt
than the one used to warm up, `flush_cache` between them, and compare the same
request on both sides — an earlier A/B without those controls reported a **+8.3%
regression** that did not exist.

```python
warm = "Pack my box with five dozen liquor jugs. " * 2800
main = "The quick brown fox jumps over the lazy dog. " * 2800
gen(warm); post("/flush_cache"); post("/start_profile")
gen(main); post("/stop_profile")
```

GPU busy time over that capture:

| | GPU busy | vs unfused |
|---|---:|---:|
| unfused (GEMM + NCCL) | 1101.9 ms | — |
| fused, bf16 wire | 1077.2 | -2.2% |
| fused, fp8 / sdma | 1050.4 | -4.7% |
| fused, fp8 / lsa | **1048.2** | **-4.9%** |

The three fused rows are one capture session; the unfused baseline is a separate
one, where the bf16 fused configuration measured 1071.7 ms rather than 1077.2.
Read the `-2.2%` as approximate and the differences between the fused rows as
the reliable part.

The layer-level win is much larger than the end-to-end one because `wo_b` is
about 12% of the profile.
