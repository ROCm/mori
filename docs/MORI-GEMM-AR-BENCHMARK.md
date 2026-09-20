# MORI GEMM + All-Reduce Benchmark

How to run the benchmarks for `mori.ops.gemm_ar` -- the fused fp8 GEMM +
all-reduce, and the mxfp8 GEMM it is built on -- and a summary of what they say.

The design, the API and **every full measurement table** live next to the code in
[`python/mori/ops/gemm_ar/README.md`](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md).
This page is the entry point: how to reproduce, what the knobs mean, and the
headline numbers.

Everything here is **MI355X (gfx950)**, one node, otherwise idle, with occupancy
recorded before and after each measurement. Kernel timings are the median over
repeated iterations, maximum over ranks. Run-to-run spread is about **2%**, so
differences below that are not differences.

## Table of Contents

- [Summary](#summary)
- [Running the benchmark](#running-the-benchmark)
  - [Modes and quantisation](#modes-and-quantisation)
  - [The fp8 wire's knobs](#the-fp8-wires-knobs)
  - [The GEMM on its own](#the-gemm-on-its-own)
- [Measurement discipline](#measurement-discipline)
- [Reproducing the end-to-end numbers](#reproducing-the-end-to-end-numbers)

## Summary

Two models are covered and **they do not reach the same conclusion**, so they
are never averaged:

| | DeepSeek-V4-Pro | DeepSeek-V4.1-Flash |
|---|---|---|
| `wo_b` per rank | `[M, 7168]` K=2048, TP8 | `[M, 5120]` K=2048, TP4 |
| quantisation | 1x128 / 128x128 fp8 block scale | 32-wide ue8m0 (mxfp8) |
| `--quant` | `blockscale` | `mxfp8` |
| best configuration | `fused-sdma` + fp8 gather | `fused-sdma` + fp8 gather over LSA |
| at M=16384 | **979.3 us against 1472.9, -33%** | **1192.1 us against 1592.5, -25%** |
| what dominates it | the overlap | the fp8 wire, then the GEMM |
| vs the model as it ships | 1419.5 us -> 979.3, **-31%** | see below |

**On V4.1-Flash most of the win is not the fusing.** The GEMM is 178us against
1414us of communication, so hiding it entirely would be worth 11-15% and fusing
collects about half of that. The fp8 wire is worth -15.6% on its own and the
mxfp8 GEMM another -30.3% against the route SGLang would otherwise take. Reading
V4-Pro's conclusion across would set every threshold wrong.

The GEMM is also usable on its own, with no collective attached
(`Mxfp8GemmOp`, and `Mxfp8GemvOp` for decode's token counts). Against SGLang's
native mxfp8 linear across all twelve of V4.1-Flash's fp8 linear shapes, what
decides the outcome is the grid, `ceildiv(M,256) * (N/256)`:

| grid (workgroups) | median vs SGLang |
|---|---:|
| 1-32 | +246.7% |
| 33-64 | -0.7% |
| 65-128 | **-10.2%** |
| 129-256 | **-26.6%** |
| >256 | **-26.4%** |

End to end in a server, V4.1-Flash prefill throughput is **+5.1% to +5.7%** at
batch sizes 4-16 on the fp8 wire; V4-Pro is **-5.0% GPU busy** over a profiled
prefill. Per-layer wins are larger than end-to-end ones because `wo_b` is about
12% of the profile.

Full tables, the numerical cost of the fp8 wire, the model-level quality
evaluation, and the threshold derivations are in
[the operator README](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md#measured-results).

## Running the benchmark

Needs a mori built with `BUILD_CCO_SDMA=ON`. Setting `MORI_ENABLE_SDMA` in the
environment only rebuilds the *device* bitcode -- a host library built without
the flag has no SDMA queues, every put silently does nothing, and the all-reduce
quietly produces zeros.

```bash
cd /path/to/mori
BUILD_CCO_SDMA=ON pip install .

# DeepSeek-V4-Pro, TP8
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048

# DeepSeek-V4.1-Flash, TP4, with the wire that wins there
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  torchrun --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant mxfp8 -m 16384 -n 5120 -k 2048 \
  --gather-dtype fp8 --gather-transport lsa --no-fuse-quantize
```

Each run prints a `RESULT_JSON` line with `max_rank_time_us`, `rel_l2` and
`validated`, so a sweep can be parsed rather than eyeballed.

Matrices are presets of `sweep.py`, which runs one point per process, enforces
the `BUILD_CCO_SDMA` guard, records occupancy either side, and **exits non-zero
with the count of failed points**:

```bash
python benchmark/cco/flydsl/gemm_ar/sweep.py --list
python benchmark/cco/flydsl/gemm_ar/sweep.py fused          # the mode matrix
python benchmark/cco/flydsl/gemm_ar/sweep.py fused-wire     # the fp8 leg's knobs
python benchmark/cco/flydsl/gemm_ar/report.py fused.jsonl
```

`report.py` keys on every configuration field a row carries, so a matrix that
varies `gather_transport` or `fuse_quantize` renders as separate tables rather
than silently overwriting itself.

Numerics:

```bash
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_flydsl_ar.py \
         tests/python/cco/test_gemm_ar_op.py
```

### Modes and quantisation

`--mode` selects what is measured, all reaching the same end state:

| mode | what runs |
|---|---|
| `gemm-only` | the GEMM alone, to size the ceiling |
| `split-sdma` | `gemm` then a 4-kernel SDMA all-reduce |
| `fused-sdma` | GEMM with the scatter fused into its epilogue |
| `split-lsa` | `gemm` then the 2-kernel LSA all-reduce |
| `fused-lsa` | GEMM storing straight into peers |

`--quant` picks the operand contract, and it is per model -- `blockscale` for
V4-Pro (A 1x128, B 128x128, fp32 scales), `mxfp8` for V4.1-Flash (32-wide ue8m0
on both, int32). Two more exist and are not for use: `mxfp8_unpacked` and
`mxfp8_row` are the scale layouts `mxfp8` was chosen over, kept so the choice
stays measurable.

> **Pin `--quant` when comparing anything against anything.** The benchmark
> defaults to `ptpc`, which is ~3% faster than `blockscale` and a different
> kernel again from `mxfp8`. Reading one against the other looks exactly like a
> machine that drifts overnight.

mxfp8 compiles at `BLOCK_M=256` where blockscale needs 128, so M pads to a
multiple of `world_size * 256`. That is not a tuning choice -- the packed scale
puts a lane's four M tiles in one dword, which is four tiles only at
`BLOCK_M//64 == 4`.

### The fp8 wire's knobs

`--gather-dtype fp8` sends the all-gather leg as e4m3 with one fp32 scale per
row. The reduce still accumulates in fp32 and the output is still bf16; only the
wire changes. The scatter leg is unchanged -- it carries partial sums that are
then added across every rank, so its error would compound rather than being a
single rounding.

| flag | recommendation |
|---|---|
| `--gather-transport {sdma,lsa}` | **`lsa`**: CUs pull and widen on the way in, where SDMA has to land the fp8 and read it back out of local HBM |
| `--no-fuse-quantize` | **on**: folding the narrowing into the reduce costs 1-2% |
| `MORI_GEMM_AR_PULL_BLOCKS` | **48-80**; these are xGMI reads, so the grid throttles outstanding remote requests rather than covering HBM latency |

The pull grid is the single most important tuning parameter and is not obvious
from the source -- the first implementation launched 512, the grid the local
quantize kernel uses, and lost to SDMA by 17%. Sweep it with:

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

### The GEMM on its own

Single process, no collective, so no `torchrun`:

`bench_gemm.py` covers both scopes, and they are different questions rather
than two views of one. `--scope kernel` takes operands that are already
quantised and already padded, which is what a tile or a scale layout is chosen
on; `--scope linear` starts from bf16 and puts quantisation, padding and the
op's own tile dispatch inside the measurement, which is what a layer costs.

```bash
B=benchmark/cco/flydsl/gemm_ar

# the multiply alone, either quantisation, any shape
python $B/bench_gemm.py -n 5120 -k 2048 -m 4096 --scope kernel --impl auto

# the layer, against SGLang. --impl gemm256,gemm128 pins mori's N tile so the
# dispatch can be measured rather than trusted
python $B/bench_gemm.py --shape wq_b -m 1024 --scope linear \
  --impl auto,gemm256,gemm128,sglang

# the skinny GEMM (M <= 32); --baseline none drops the SGLang import entirely
python $B/bench_gemv.py --shape wq_b -m 1 --sweep
```

The SGLang baseline is optional throughout -- mori's own numbers need nothing
installed but mori -- but `--scope linear` does need it, because the
quantisation inside the measurement is SGLang's quantiser.

Comparing against SGLang's own route needs its harness, which lives in the
SGLang tree because the baseline does
(`test/registered/perf/models/bench_mori_mxfp8_gemm.py`); it takes `-n`/`-k` for
any shape, or a name for any of V4.1-Flash's layers.

## Measurement discipline

These are the ones that have produced a confident wrong number here. The reasons
are in
[Measurement traps](https://github.com/ROCm/mori/blob/main/python/mori/ops/gemm_ar/README.md#measurement-traps).

- **Amortise the capture.** A single-call CUDA-graph replay has a **13.40 us
  floor** on this box, which is most of any small-M measurement. Use
  `benchmark/cco/flydsl/gemm_ar/timing.py`.
- **Report cold as well as hot.** A repeated call reads the weight out of the
  256 MB LLC at 1.7x the bandwidth a forward pass gets. Cold is what predicts a
  server.
- **One M per process.** A multi-M sweep once reported 697 us and 1083 us for
  two M that pad to the same size; isolated, both were 697.
- **Assert the fast path actually ran.** A path that silently declined looks
  exactly like one that ran and was not worth it. Both integrations have a shape
  log for this (`SGLANG_OPT_FUSED_WO_B_AR_SHAPE_LOG=1`,
  `SGLANG_OPT_MORI_MXFP8_GEMM_SHAPE_LOG=1`).
- **Check `BUILD_CCO_SDMA=ON` before believing any end-to-end number.** With it
  off the fused path measures *faster* than it is, because it is not moving
  data.
- **Let the SDMA queues drain between runs.** Back-to-back 8-rank runs hit
  `hsaKmtCreateQueueExt` failures (`anvil.cpp:237`) if a previous run's ranks
  have not exited. ~10s is enough; a leftover server holding queues is not.

## Reproducing the end-to-end numbers

The SGLang numbers need the integration branch and a mori built with
`BUILD_CCO_SDMA=ON` on `PYTHONPATH`.

```bash
export MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo
export PYTHONPATH=/path/to/mori-with-sdma
export SGLANG_OPT_FUSED_WO_B_AR=1
export SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER=1     # optional, the fp8 wire
export SGLANG_OPT_MORI_MXFP8_GEMM=1              # optional, the GEMM alone
export SGLANG_DEBUG_FUSED_WO_B_AR=1              # optional, logs per-layer relL2

sglang serve --model-path <DeepSeek-V4-Pro> --tp 8 \
  --attention-backend dsv4 --page-size 256 --chunked-prefill-size 16384 \
  --mem-fraction-static 0.88 --kv-cache-dtype fp8_e4m3 \
  --enforce-shared-experts-fusion
```

V4.1-Flash is the verified MI350X low-latency cell plus mori's variables, at
TP4. `AITER_BF16_FP8_MOE_BOUND=0` is load-bearing and is *not* part of the
published cell: the checkpoint has `swiglu_limit=10.0`, so its MoE takes AITER's
clamped-SwiGLU INTERLEAVE path, where no `ck_moe_stage1` kernel exists for
(bf16 activation x fp4 weight) -- every M below the default bound of 256, which
is every decode step, raises "Unsupported kernel config for moe heuristic
dispatch".

```bash
export SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1
export AITER_FLYDSL_FORCE_REDUCE=1 ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export AITER_BF16_FP8_MOE_BOUND=0

sglang serve --model-path <DeepSeek-V4.1-Flash> --tp 4 --ep-size 4 \
  --disable-radix-cache --mem-fraction-static 0.78 \
  --chunked-prefill-size 16384 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --cuda-graph-max-bs 64 --cuda-graph-backend-prefill breakable \
  --cuda-graph-max-bs-prefill 4096
```

`--chunked-prefill-size` has to be at least the wire's floor or no prefill chunk
ever reaches it and the fused variants quietly measure the base path.

`--mem-fraction-static` has to leave room for the symmetric window, which is VMM
memory **outside** torch's allocator: 700 MiB on V4-Pro's bf16 wire and 812 MiB
on fp8, which needs the extra staging region; 163 MiB at V4.1-Flash's TP4 shape
with `m_max=5120`.

The capture protocol matters more than it looks. Profile a *different* prompt
than the one used to warm up, `flush_cache` between them, and compare the same
request on both sides -- an earlier A/B without those controls reported a **+8.3%
regression** that did not exist.

```python
warm = "Pack my box with five dozen liquor jugs. " * 2800
main = "The quick brown fox jumps over the lazy dog. " * 2800
gen(warm); post("/flush_cache"); post("/start_profile")
gen(main); post("/stop_profile")
```
