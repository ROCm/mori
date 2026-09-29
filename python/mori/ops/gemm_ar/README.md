# `mori.ops.gemm_ar`

GEMM + all-reduce optimization for `wo_b`, and standalone MXFP8 GEMM optimization
for the two Flash TP4 attention projections below. The target list is fixed.
General API support and earlier broad surveys do not expand this work's scope.

**2026-09-29 operator retest on `v2-015`:** measured cells below use this run's
results; `待测` still means unmeasured. GEMV/decode, model-server/end-to-end
evaluation and model quality are deferred. Other layers and other TP
configurations are outside the
current optimization and test plan.

## Optimization targets

All N/K dimensions are per rank. These are four workloads over three distinct
N/K shapes; C2 and G1 share the multiply but measure different operator scopes.
G1/G2 run on one GPU per invocation; TP4 identifies the model's weight shard.

| ID | Model | TP | Layer | Operator | Operands | N | K |
|---|---|---|---|---|---|---|---|
| C1 | V4 Pro | 8 | wo_b | GEMM + AR | blockscale | 7168 | 2048 |
| C2 | V4.1 Flash | 4 | wo_b | GEMM + AR | mxfp8 | 5120 | 2048 |
| G1 | V4.1 Flash | 4 | wo_b | Standalone GEMM | mxfp8 | 5120 | 2048 |
| G2 | V4.1 Flash | 4 | wq_b | Standalone GEMM | mxfp8 | 8192 | 1280 |

- [API and constraints](#api-and-constraints)
- [Validation protocol](#validation-protocol)
- [Current decisions](#current-decisions)
- [Target measurement matrices](#measured-results)
- [Target historical records](#historical-records)
- [Regression checklist](#regression-checklist)

## API and constraints

| Entry point | Operation | Current contract |
|---|---|---|
| `GemmAllReduceOp` | `wo_b` GEMM + all-reduce | Targets C1/C2; serial use of one instance |
| `Mxfp8GemmOp` | Standalone MXFP8 GEMM | Flash TP4 targets G1/G2; pad M to a multiple of 64 |

A/B operands use `torch.float8_e4m3fn`; FNUZ is a different encoding and is
rejected. `supports` and `supports_gemm` describe shape support,
not a guarantee of a performance win. Wider API support does not add shapes or
parallelism configurations to this optimization plan.

For the public collective operator, M pads to `TP * block_m`: the default
`block_m` is 128 for blockscale and 256 for MXFP8. N is a multiple of 256, and
FP8 gather additionally requires N to be a multiple of 1024. K is at least 256
and a multiple of 128. `gather_dtype` defaults to BF16. FP8 gather supports SDMA
push/dequantize or LSA pull/dequantize; the constructor defaults its FP8 gather
transport to LSA. `split-lsa` has no FP8 gather leg.

FP8 gather retains the local output slice in BF16 and reconstructs remote slices
from FP8 payloads and scales. Its wire reference must account for that per-rank
behavior; earlier gather-only model results do not validate FP8 scatter.

### Collective operator example

Build MORI's host libraries with `BUILD_CCO_SDMA=ON`. When running SDMA tests,
set both `BUILD_CCO_SDMA=ON` and `MORI_ENABLE_SDMA=1` so the host library and
resolved device bitcode agree. Set the local GPU and initialize the process
group before this example; the quantizer and input tensors are caller supplied.

```python
import torch
import torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

rank, world = dist.get_rank(), dist.get_world_size()
assert world == 8  # C1: V4 Pro TP8.
N, K, M_MAX = 7168, 2048, 16384
vmm = 2 * GemmAllReduceOp.window_bytes_for(world, m_max=M_MAX, n=N) + (64 << 20)
uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

with Communicator.init(
    world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
) as comm:
    with GemmAllReduceOp(comm, n=N, k=K, m_max=M_MAX) as op:
        b_shuffled = preshuffle_b(b_fp8)
        m_pad = op.padded_m(x.shape[0])
        a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))
        result = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]]
        torch.cuda.current_stream().synchronize()
        result = result.clone()
```

Pad before quantization. Blockscale A scales are FP32 in K-block-major physical
order: flat, `[K/128, M]`, or column-major `[M, K/128]` with stride `(1, M)`.
Ambiguous row-major `[M, K/128]` is rejected; the caller must select the intended
layout explicitly. B scales are row-major `[N/128, K/128]` FP32.

Every rank constructs and calls the collective in the same order. The result
aliases the window and the next call overwrites it; `pad_rows` also reuses a
buffer. One instance is not a concurrent-call interface. Warm each supported M
before graph capture, retain launchers and operand storage for graph lifetime,
and synchronize before closing resources. `close()` frees the window/memory;
SDMA device-communicator queues remain communicator-owned.

### Standalone MXFP8 GEMM example

```python
import torch
from mori.ops.gemm_ar import Mxfp8GemmOp
from mori.ops.gemm_ar import preshuffle_a_scale, preshuffle_b

# Flash TP4 wo_b; use N=8192, K=1280 for wq_b.
N, K = 5120, 2048
b_shuffled = preshuffle_b(w_fp8)
b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)

gemm = Mxfp8GemmOp(n=N, k=K)
m_pad = gemm.padded_m(x.shape[0])
a_fp8, a_exps = quantize_mxfp8(gemm.pad_rows(x, m_pad))
result = gemm(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale)[: x.shape[0]]
```

MXFP8 GEMM uses packed A scales from `preshuffle_a_scale` and K-block-major
widened B scales. The example assumes caller-supplied quantizers and tensors.
The blockscale GEMM-only row for C1 is a control for its collective, not another
standalone MXFP8 optimization target.

### Dispatch and experimental options

`Mxfp8GemmOp` selects its internal 128-column tile when
`ceil(M_pad / 256) * (N / 256) < 140`, otherwise its 256-column tile. This tile
choice is distinct from a caller's decision to admit MORI at all. Re-evaluate
these choices only on G1/G2; do not derive a general all-layer dispatch policy
from this target-specific study.

| Option | Entry point / required conditions | Public default |
|---|---|---|
| Short final chunk | `compile_fused_gemm_scatter(chunk_bands=...)`; matching counter count; fused SDMA | Divisor-based chunks remain |
| Reduced padding | `relaxed_rows=True`, positive `chunk_bands`, at least one full tile per peer, BF16 scatter | Existing public padding remains |
| Workspace reuse | Experimental layout; serial calls and scatter drain protect reuse | Existing allocation remains |
| FP8 scatter | Low-level builder/benchmark, not the public constructor; aligned non-compact fused SDMA; matching scale tile; separate reduction | BF16 scatter |
| Grouped reduce/quantize | Experimental 128/256/512 groups with FP8 LSA gather | Separate phases |

FP8 scatter currently rejects relaxed rows, compact receive storage, direct LSA
scatter, fused rowwise reduce/quantize and reduce-triggered gather puts. Validate
unsupported combinations on the host before launching. These are compatibility
contracts, not new measurements or an instruction to enable the experiments.
The scatter prototype reads completed BF16 GEMM tiles and quantizes one row and
256 columns at a time for transfer; it does not quantize directly from accumulators.

## Validation protocol

### Scopes and baselines

| Scope | Timed work | Required reference / comparison |
|---|---|---|
| GEMM kernel | Prequantized, prepacked operands; state physical M explicitly | Independent dequantized FP32 product |
| Linear operator | Activation quantization, padding and multiply on the resolved GPU route | Independently validated conversion/packing plus product reference |
| GEMM + collective | Stated compute and communication phases | Per-rank partial reference plus independent ordered communication reference |
| FP8 communication | Exact scatter/gather choice and scale grouping | Mathematical error and actual payload+scale wire reference reported separately |
| Workspace/lifecycle | Initialization, allocations, steady-state calls and release | Output correctness across reuse and actual memory accounting |
| Phase/ISA study | Explicit profiler interval and kernel set | Same mathematical workload; counters kept separate from whole-call time |

A timing-only result (`validated: null`) is not a correctness pass. A baseline
implementation is a comparator, not its own golden reference. BF16 baselines and
lossy FP8 variants use different numerical acceptance criteria; document these
before filling any performance cell.

For FP8, report relL2 and absolute-error statistics together. Zero/near-zero
references need an absolute-error criterion. Do not require a positive relL2
floor for zero or exactly representable inputs. Check payloads, scales and the
executed route to establish that the requested wire format was used.

### Measurement record

Every case is keyed by C1/C2/G1/G2 plus M and the changed option. N, K,
quantization and model TP are fixed by that target; they are not sweep axes.
Each unique configuration has one case ID. Repeated appearances of the same
configuration are samples of that case, not additional coverage. `gemm-only`
is measured once per compute configuration; varying unused gather flags is not
a separate communication test. The shared FP8 rows of the old `fused-fp8` and
`fused-wire` sweeps belong to one canonical matrix below.

| Record field | v2-015 measurement record |
|---|---|
| MORI / comparator revisions and any compatibility patch | MORI `fd35a5ec79d8`; SGLang operator source `e29ebcd051`; optional `out=` adapter affects only the deferred GEMV path |
| GPU architecture, count, clocks and background activity | `crsuse2-m2m-v2-015`, 8 × MI355X / gfx950; default clocks, no locking; idle preflight (~0.28 GiB/card); GPU process ownership sampled every 2 seconds; no foreign process in the published run; default clocks |
| Torch / HIP / FlyDSL / compiler / FFI versions | Torch 2.11.0+rocm7.2 / ROCm 7.2.4 / FlyDSL 0.2.4; exact HIP, compiler, Triton and FFI versions in `environment.json` |
| Logical M, physical M, N, K, TP, operand and wire dtypes | Four-target registry and row keys below; L1/K1 M is already aligned; P states logical and physical M |
| GEMM tile, chunks, padding, workspace and transport | Per-case JSON and exact argv; SDMA uses one queue; public dispatch/allocation defaults unchanged |
| Scope, graph calls, warmup, replays and independent rounds | 4 warmup calls before each graph; L1/K1: hot 32 calls, cold 39 calls, 21 replays, 3 alternating rounds; A: 8 calls × 21 replays × 3 rounds; feature pairs: 8 calls × 21 replays × 3 rounds (short tails: 5) |
| Weight ring sizes and every rotated weight representation | L1/K1: 39 copies, 390 MiB of FP8 weights; SG also rotates its 780 MiB BF16 copy (1170 MiB total). A/features reuse one weight (hot) |
| Reference definition, tolerances and numerical results | Independent dequantized FP32 products; L1 also checks activation bytes/scales and changed inputs. MORI GEMM relL2 ≤0.0024, SG ≤0.003; BF16 collective <0.003; A random FP8 uses 0.005–0.04; FP8 features <0.045; wire checks separate |
| Per-rank times, paired round medians, spread / confidence interval | L1/K1: median of 3 round medians. A/features: median across round max-rank medians. All rank/round values retained; feature variants alternate; separate A modes are not paired |
| Window bytes, Torch allocation peak and external allocation accounting | W: logical/backing capacity for paired timing; separate one-layout-per-process HIP physical-allocation deltas. Full peaks and operator initialization remain 待测 |
| Exact command, raw log/result paths and final outcome | External archive `/workspace/reports/gemm-ar-v2-015-20260929/`; `results/manifest.json`, `jobs.jsonl`, per-case logs/JSON, scripts and source archives |

Use one frozen record for published results. Keep live progress, setup failures
and retry logs in the external archive. Record failed attempts and unsupported
configurations explicitly; do not silently filter them out.

### Measurement traps

- Amortize short-kernel graph overhead. A kernel profile, a one-call graph and
  a multi-call graph are different scopes; do not combine their latency columns.
- Rotate enough storage to exceed LLC, including any BF16 weight read by the
  comparator. Capture enough calls to visit the ring, and report hot and cold.
- Retain graph storage and the FlyDSL launchers owning compiled modules.
- Keep compilation, warmup and untimed preparation outside the selected scope.
- Use 3–5 paired alternating A/B rounds for decisions near the observed noise level.
  Report round-to-round spread; do not assume one universal 2% noise bound.
- Keep GPU measurements sequential. Preserve the established CCO queue-reclaim
  interval between independent multi-rank jobs (20 seconds in this run).
- Check the requested route actually ran. An overlapped drain's time is only
  exposed wait time and cannot be used as a whole-transfer bandwidth estimate.

### Statistics and decision rules

Latency cells use microseconds; memory uses MiB/rank. `Δ% = 100 *
(T_variant / T_baseline - 1)`, so negative means faster. For collectives, first
compute per-rank medians, then the maximum rank value for each paired round;
report the median and dispersion across independent rounds.

Tile crossover and caller admission are different decisions. For admission,
record both the **wrong-choice rate** (wrongly selected points / eligible
points) and **mean relative latency penalty** against the faster measured route:
`mean(T_selected / min(T_MORI, T_baseline) - 1)`. Use only G1/G2 for these summaries
and declare the noise/tie rule. Historical percentages using another denominator or formula
remain historical until recalculated under this definition.

## Current decisions

The current defaults remain unchanged. Historical directions and this machine's measurements are kept separate.
Use paired feature results and observed round spread before changing defaults.

| Topic | Target | Historical direction | Required confirmation | New result |
|---|---|---|---|---|
| Short final chunks | C1/C2 | Promising at awkward band counts | Tail/count/transfer coverage | C1 bands=2: M=11264 -22.2%, M=13312 -23.8%; C2/tail-boundary ladder pending |
| Reduced padding | C1/C2 | Promising at ragged M | Owner/chunk intersections | C1: -0.6% / -2.9%; C2: -13.7% / -7.2% (M=4200 / 8200, align64; small changes require spread checks) |
| Workspace reuse | C1/C2 | Reduced window requirement | Serial lifecycle and actual peak allocation | C1: logical bytes −40.0%, latency -0.2%; C2: logical bytes −46.2%, latency -0.1%; physical allocation is checked separately in W |
| FP8 scatter/gather | C1/C2 | Lossy performance option | Wire reference and numerical error | C1: +3.0% / +2.5%; C2: -21.1% / -26.6% (M=16384, BF16 / FP8 gather; lossy) |
| Grouped reduce/quantize | C1/C2 | Negative control | Same-shape time and correctness | C1: +4.2% to +4.3%; C2: +2.4% to +2.5%; separate phases remain preferable |
| GEMM tile / scale layout | G1/G2 | Internal tile and packing choices | Target-specific boundaries and references | G1/G2 main ladder and D measured; padding/tile boundaries and scale-layout study pending |

## Measured results

The completed target run contains 28 standalone jobs (98 implementation
configurations), 102 collective configurations, 28 paired feature jobs and
8 allocation-only layout measurements. All final jobs completed successfully.
The supervised runs contain 2,913 GPU-ownership samples, with no foreign GPU
process observed and a maximum within-run sampling gap of 2.24 seconds.
Seven unsuccessful attempts in this supervised queue remain in the archive:
five multi-window setup attempts, the predicted-gather reference mismatch and
the immediate-reclamation assertion. They are not counted as successful results.
The source machine was unavailable for this run: all eight visible GPUs held
257–259 GiB each despite 0% utilization during the samples. `v2-015` was idle
before launch. Its Torch/ROCm environment differs from the historical run, so
historical values below are preserved and are not pooled with this measurement.

The earlier attempt completed the standalone ladder but encountered intermittent
external GPU work and a Docker stop (exit 137, no OOM). Its timings are excluded
from the current tables. Raw logs, the provisional README snapshot and stop
records remain in the external archive's `results/` and
`interruption-evidence.log`. The replacement run uses `results-monitored/`,
starts after six idle checks and samples GPU process ownership every two
seconds; the supervisor stops this task if a foreign GPU process is observed.
This is sampled evidence, not a reservation of the host.

L1 and K1 use cold medians; hot timing and all round values are in the archive.
L1 compares the same FP8-quantized mathematical product: SGLang's BF16 GEMM
route also quantizes and dequantizes activations before multiplying. MORI's
activation bytes and packed scales are checked against an independent PyTorch
conversion. SG and MORI use separate numerical limits, not each other as the
reference. The first SG reference incorrectly used unquantized activations;
that failed attempt and the preliminary smoke logs are retained, excluded from
published results, and the corrected reference is used for the full ladder.

The feature harness initially failed CCO `hipMemSetAccess` when allocating or
registering multiple windows of different capacities (C1 M=1025). Retrying,
disabling GDR, adding 2 MiB padding and preallocating alone did not resolve it;
those attempts remain in the archive. Published feature pairs allocate equal
backing capacities (the largest variant rounded to 2 MiB), then register all
windows before preparing per-variant temporaries. Each variant retains its own
logical M, layout and transfer lengths. Tail guards beyond each logical layout
are checked on every rank after timing. GDR retains its default setting.

W reports logical layout requirements and equal backing requests for its
paired timings. Its physical column comes from separate fresh processes with
one layout per process, measuring the HIP free-memory decrease across window
allocation/registration; it is not a full operator/process allocation peak.
Feature baselines are measured within each pair and are not borrowed from A.

Collective numerical-error columns report rank 0; every rank must pass the
correctness gate. Timing uses all ranks as described above.

A reports whole prequantized GEMM + collective calls (GEMM-only is its control),
using multi-call graphs; it excludes activation quantization/padding and weight
packing. It cannot be substituted for L1 or for historical single-call graphs.
A's FP8 PASS is a random-input mathematical check; actual payload/scale
checks are reported separately in F. For two FP8 legs, F checks the local
reduction independently, validates each quantizer (allowing adjacent codes only
at rounding midpoints), then reconstructs the output from both actual wires.
The earlier predicted second quantization failed at FP8 midpoint choices in
C2 M=4096; the revised check retained the 1e-3 communication-error gate and
matched the actual reduction and gathered output exactly. That failed attempt
is retained in the archive. F reports errors against the FP32 product
sum; p99 uses a deterministic sample of about one million output elements.
Unmeasured diagnostics, phase profiles, full lifecycle/contract matrices and
special input families retain `待测`; successful random inputs do not fill them.

These matrices cover the four targets only. `待测` is not zero, a pass,
or a historical value. N/K/TP and operand quantization come from the target
registry. Keep M and a relevant implementation knob as the sweep axes.

The main standalone ladder is M=64/256/1024/2048/4096/8192/16384; nearby padding
and tile-switch values are separate boundary cases. There is no decode/GEMV
matrix and no small-M GEMM performance sweep intended to stand in for GEMV.

<details>
<summary>L1 — Target linear-operator matrix (14 shape/M points)</summary>

Flash TP4 only. Quantization, padding and multiply are inside the GPU graph scope.
The Python route decision and graph construction happen outside timed replay.
Cells are cold median µs; the per-case record also stores hot timing, spread,
resolved tile, physical M and the independent-reference result.

| Target | Layer | N | K | M | Expected M_pad | SG linear | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | wo_b | 5120 | 2048 | 64 | 64 | 12.56 | 23.84 | 23.88 | 28.46 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 256 | 256 | 19.62 | 27.16 | 27.16 | 29.73 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 1024 | 1024 | 33.38 | 30.21 | 30.15 | 32.29 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 2048 | 2048 | 52.94 | 35.88 | 50.88 | 35.53 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 4096 | 4096 | 82.22 | 61.35 | 74.34 | 61.39 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 8192 | 8192 | 140.69 | 97.50 | 124.06 | 97.81 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 16384 | 16384 | 248.95 | 179.68 | 243.06 | 179.04 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 64 | 64 | 10.20 | 18.60 | 18.62 | 21.57 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 256 | 256 | 12.07 | 21.16 | 21.20 | 22.62 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 1024 | 1024 | 24.67 | 24.46 | 24.55 | 25.71 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 2048 | 2048 | 42.42 | 30.25 | 39.73 | 30.42 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 4096 | 4096 | 74.62 | 51.47 | 71.17 | 51.09 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 8192 | 8192 | 135.02 | 91.98 | 132.16 | 91.32 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 16384 | 16384 | 313.51 | 185.21 | 260.36 | 187.02 | PASS; relL2 ≤0.00166 |

</details>

<details>
<summary>K1 — Target standalone MXFP8 GEMM kernels</summary>

G1/G2 only, with prequantized and prepacked operands. Kernel-only timing is
separate from the linear comparison and uses an independent FP32 reference.
C1's blockscale multiply is measured only as the GEMM-only control in A.

| Target | Layer | N | K | M | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|
| G1 | wo_b | 5120 | 2048 | 64 | 21.73 | 21.75 | 26.00 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 256 | 24.18 | 24.19 | 26.54 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 1024 | 25.06 | 25.16 | 27.50 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 2048 | 29.27 | 44.62 | 29.67 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 4096 | 53.69 | 66.30 | 53.50 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 8192 | 83.83 | 110.25 | 83.99 | PASS; relL2 ≤0.00166 |
| G1 | wo_b | 5120 | 2048 | 16384 | 151.46 | 214.57 | 151.78 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 64 | 16.41 | 16.38 | 19.48 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 256 | 18.91 | 18.96 | 20.45 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 1024 | 20.56 | 20.64 | 21.75 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 2048 | 25.26 | 35.51 | 25.10 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 4096 | 44.18 | 64.56 | 44.77 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 8192 | 82.29 | 123.09 | 82.79 | PASS; relL2 ≤0.00166 |
| G2 | wq_b | 8192 | 1280 | 16384 | 168.51 | 240.74 | 166.35 | PASS; relL2 ≤0.00166 |

</details>

<details>
<summary>B — Target boundaries and contract checks</summary>

| ID | Target | Boundary / input | Expected check | Result |
|---|---|---|---|---|
| B1 | G1/G2 | M=63,64,65,127,128,129,255,256,257 | Padding before quantization; true output rows | 待测 |
| B2 | G2 | M=1023,1024,1025,1088 | Both sides of the 140-grid tile switch | 待测 |
| B3 | G1 | M=1535,1536,1537,1600 | Both sides of the 140-grid tile switch | 待测 |
| B4 | G1/G2 | Nearest reachable M around admission grid 48/64/80/128 | Admission and tile choice scored separately | 待测 |
| B5 | C1/C2 | M=1023,1024,1025,4200,8200 | Target padding/owner boundaries | PASS in P; FP32, repeat, changed inputs and guards |
| B6 | All four targets | Wrong operand/scale dtype, shape or stride | Host rejection; no timing of invalid inputs | 待测 |
| B7 | C1/C2 | Invalid communication/option combinations | Host rejection without silently changing route | 待测 |

</details>

<details>
<summary>D — Target tile crossover and admission summaries</summary>

Derived only from G1/G2. Earlier all-layer aggregate scores are not reused
as evidence for these target-specific thresholds.

Derived from the 14 main-ladder points only; boundary points remain pending.
Wide grid = `ceil(M_pad / 256) * (N / 256)`. Tile Δ uses K1 cold
`100 * (N128 / N256 - 1)`. A win requires non-overlapping three-round ranges;
points counted in neither win column are ties. These ranges are observed spread,
not confidence intervals. Empty bins have no target point on this ladder.
Admission selects MORI auto when wide grid ≥ threshold, otherwise SG; the
penalty uses median latency for all 14 points, including ties. This summary
measures the candidate rule and does not change public dispatch.

| Wide-grid bin | Target points | N128 faster | N256 faster | Mean Δ% | Spread |
|---|---|---|---|---|---|
| 1–16 | 0 | — | — | — | — |
| 17–32 | 4 | 4 | 0 | -12.1% | max round range 1.1% |
| 33–64 | 0 | — | — | — | — |
| 65–128 | 2 | 2 | 0 | -6.8% | max round range 1.0% |
| 129–192 | 1 | 0 | 1 | +50.4% | max round range 0.6% |
| >192 | 7 | 0 | 7 | +39.4% | max round range 1.7% |

| Admission grid threshold | G1/G2 points | Wrong-choice rate | Mean latency penalty | Tie/noise rule |
|---|---|---|---|---|
| 48 | 14 | 0/14 (0.0%) | 0.00% | Overlapping round ranges = tie |
| 64 | 14 | 0/14 (0.0%) | 0.00% | Overlapping round ranges = tie |
| 80 | 14 | 0/14 (0.0%) | 0.00% | Overlapping round ranges = tie |
| 128 | 14 | 1/14 (7.1%) | 0.75% | Overlapping round ranges = tie |

</details>

<details>
<summary>A — wo_b GEMM + all-reduce mode/wire matrix</summary>

Only C1/C2 (`wo_b`). Every row uses the target's operand quantization and
BF16 scatter. GEMM-only appears once per target/M; unused gather options do not
create additional compute cases. Shared FP8 configurations have one case ID.

##### C1: V4 Pro, TP8, wo_b, blockscale, N=7168, K=2048

| Mode | Gather dtype | Actual gather transport | Fused quantize | M=4096 µs | M=8192 µs | M=16384 µs | Reference |
|---|---|---|---|---|---|---|---|
| gemm-only | none | none | off | 105.46 | 179.89 | 340.54 | PASS; rank0 ≤0.00166 |
| split-sdma | bf16 | sdma | off | 390.23 | 722.92 | 1487.51 | PASS; rank0 ≤0.00235 |
| split-lsa | bf16 | lsa | off | 386.60 | 726.90 | 1507.56 | PASS; rank0 ≤0.00235 |
| fused-sdma | bf16 | sdma | off | 339.20 | 611.74 | 1165.85 | PASS; rank0 ≤0.00235 |
| fused-lsa | bf16 | sdma | off | 516.18 | 878.98 | 1673.49 | PASS; rank0 ≤0.00235 |
| split-sdma | fp8 | sdma | off | 374.26 | 668.61 | 1314.18 | PASS math; rank0 ≤0.0249 |
| split-sdma | fp8 | sdma | on | 403.68 | 692.99 | 1337.43 | PASS math; rank0 ≤0.0249 |
| split-sdma | fp8 | lsa | off | 337.77 | 629.34 | 1266.60 | PASS math; rank0 ≤0.0249 |
| split-sdma | fp8 | lsa | on | 366.54 | 655.49 | 1289.19 | PASS math; rank0 ≤0.0249 |
| fused-sdma | fp8 | sdma | off | 321.57 | 545.95 | 1013.10 | PASS math; rank0 ≤0.0249 |
| fused-sdma | fp8 | sdma | on | 350.56 | 571.37 | 1029.51 | PASS math; rank0 ≤0.0249 |
| fused-sdma | fp8 | lsa | off | 286.04 | 506.52 | 961.92 | PASS math; rank0 ≤0.0249 |
| fused-sdma | fp8 | lsa | on | 314.70 | 530.27 | 982.75 | PASS math; rank0 ≤0.0249 |
| fused-lsa | fp8 | sdma | off | 499.38 | 869.70 | 1495.53 | PASS math; rank0 ≤0.0249 |
| fused-lsa | fp8 | sdma | on | 515.12 | 861.11 | 1515.65 | PASS math; rank0 ≤0.0249 |
| fused-lsa | fp8 | lsa | off | 463.71 | 794.85 | 1440.13 | PASS math; rank0 ≤0.0249 |
| fused-lsa | fp8 | lsa | on | 478.76 | 804.10 | 1465.93 | PASS math; rank0 ≤0.0249 |

##### C2: V4.1 Flash, TP4, wo_b, mxfp8, N=5120, K=2048

| Mode | Gather dtype | Actual gather transport | Fused quantize | M=4096 µs | M=8192 µs | M=16384 µs | Reference |
|---|---|---|---|---|---|---|---|
| gemm-only | none | none | off | 57.02 | 91.94 | 162.03 | PASS; rank0 ≤0.00166 |
| split-sdma | bf16 | sdma | off | 446.93 | 843.70 | 1639.26 | PASS; rank0 ≤0.00237 |
| split-lsa | bf16 | lsa | off | 439.63 | 841.65 | 1686.82 | PASS; rank0 ≤0.00237 |
| fused-sdma | bf16 | sdma | off | 440.17 | 816.57 | 1548.88 | PASS; rank0 ≤0.00237 |
| fused-lsa | bf16 | sdma | off | 541.50 | 1003.45 | 1911.96 | PASS; rank0 ≤0.00237 |
| split-sdma | fp8 | sdma | off | 379.84 | 696.13 | 1375.45 | PASS math; rank0 ≤0.0231 |
| split-sdma | fp8 | sdma | on | 386.82 | 700.91 | 1392.02 | PASS math; rank0 ≤0.0231 |
| split-sdma | fp8 | lsa | off | 367.23 | 683.14 | 1352.05 | PASS math; rank0 ≤0.0231 |
| split-sdma | fp8 | lsa | on | 373.90 | 689.19 | 1371.79 | PASS math; rank0 ≤0.0231 |
| fused-sdma | fp8 | sdma | off | 374.23 | 668.88 | 1257.35 | PASS math; rank0 ≤0.0231 |
| fused-sdma | fp8 | sdma | on | 380.06 | 675.18 | 1273.56 | PASS math; rank0 ≤0.0231 |
| fused-sdma | fp8 | lsa | off | 361.01 | 656.31 | 1234.04 | PASS math; rank0 ≤0.0231 |
| fused-sdma | fp8 | lsa | on | 367.41 | 661.72 | 1252.04 | PASS math; rank0 ≤0.0231 |
| fused-lsa | fp8 | sdma | off | 476.41 | 850.83 | 1575.56 | PASS math; rank0 ≤0.0231 |
| fused-lsa | fp8 | sdma | on | 477.55 | 854.00 | 1613.87 | PASS math; rank0 ≤0.0231 |
| fused-lsa | fp8 | lsa | off | 462.32 | 835.83 | 1571.08 | PASS math; rank0 ≤0.0231 |
| fused-lsa | fp8 | lsa | on | 462.84 | 840.93 | 1593.47 | PASS math; rank0 ≤0.0231 |

</details>

<details>
<summary>C — Target short-tail chunks</summary>

Only C1/TP8 and C2/TP4. M = bands_per_peer * TP * BLOCK_M; the baseline
uses the divisor rule and each candidate uses its matching counter count.
Check no missing/duplicate rows, transfer lengths and repeated calls.

##### C1: blockscale, TP8, BLOCK_M=128

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage/reference |
|---|---|---|---|---|---|---|
| 1 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 3072 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 5120 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 7168 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 10240 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 11264 | 1074.44 | 待测 | 835.64 | 待测 | PASS; bitwise + FP32 + changed inputs (bands=2) |
| 12 | 12288 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 13312 | 1273.25 | 待测 | 969.71 | 待测 | PASS; bitwise + FP32 + changed inputs (bands=2) |
| 14 | 14336 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### C2: mxfp8, TP4, BLOCK_M=256

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage/reference |
|---|---|---|---|---|---|---|
| 1 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 3072 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 5120 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 7168 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 10240 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 11264 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 12288 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 13312 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | 14336 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>P — Target wo_b padding/intersections</summary>

Use identical true rows in each target's independently padded buffers.
If a candidate has fewer than BLOCK_M rows per peer, check rejection/fallback
instead of launching relaxed rows.

##### C1: blockscale, TP8, BLOCK_M=128

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Reference |
|---|---|---|---|---|---|---|---|
| 1023 | 1024 | 1024 | 1024 | 125.43 | 125.67 | 125.71 | PASS; repeat + changed inputs |
| 1024 | 1024 | 1024 | 1024 | 126.95 | 126.62 | 126.82 | PASS; repeat + changed inputs |
| 1025 | 2048 | 1280 | 1088 | 198.33 | 168.80 | 141.36 | PASS; repeat + changed inputs |
| 4200 | 5120 | 4352 | 4224 | 408.35 | 410.42 | 405.97 | PASS; repeat + changed inputs |
| 8200 | 9216 | 8448 | 8256 | 728.86 | 710.08 | 707.91 | PASS; repeat + changed inputs |

##### C2: mxfp8, TP4, BLOCK_M=256

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Reference |
|---|---|---|---|---|---|---|---|
| 1023 | 1024 | 1024 | 1024 | 145.42 | 145.74 | 145.81 | PASS; repeat + changed inputs |
| 1024 | 1024 | 1024 | 1024 | 144.95 | 145.85 | 145.71 | PASS; repeat + changed inputs |
| 1025 | 2048 | 1280 | 1088 | 241.82 | 173.19 | 155.79 | PASS; repeat + changed inputs |
| 4200 | 5120 | 4352 | 4224 | 533.56 | 472.00 | 460.27 | PASS; repeat + changed inputs |
| 8200 | 9216 | 8448 | 8256 | 894.46 | 847.41 | 830.37 | PASS; repeat + changed inputs |

</details>

<details>
<summary>W — Target workspace and serial lifecycle</summary>

The physical column is a separate allocation-only measurement, using one fresh
process per target/layout and the layout's own capacity. All ranks reported the
same window deltas shown. The steady-state timings use the equal backing buffers
above. Full Torch peaks, allocation counts and operator initialization are pending.

Immediate HIP snapshots after window/memory close still showed the window's
allocation delta (282–702 MiB/rank). After communicator destruction,
the residual versus the pre-communicator snapshot was 36 MiB for C1
and 20 MiB for C2, independent of layout. These snapshots do not establish
immediate physical reclamation or a full operator lifecycle pass. The initial
probe's immediate-reclamation assertion failed and its log is retained.

C1/C2, capacity M=16384. Separate window requirements, actual allocator
peaks and initialization. Serial mixed-M sequences remain pending on these same
N/K/TP targets; do not expand the model/parallelism matrix.

| Target | Layout | Logical / backing MiB/rank | Torch peak MiB | External physical MiB | Allocations | Init µs | Steady µs | Reference |
|---|---|---|---|---|---|---|---|---|
| C1 | baseline | 700.006 / 702 | 待测 | 702 | 待测 | 待测 | 1165.69 | PASS; repeat + changed inputs |
| C1 | no unused tmp | 672.006 / 702 | 待测 | 674 | 待测 | 待测 | 1165.54 | PASS; repeat + changed inputs |
| C1 | compact receive | 644.006 / 702 | 待测 | 646 | 待测 | 待测 | 1166.18 | PASS; repeat + changed inputs |
| C1 | input/output alias | 420.006 / 702 | 待测 | 422 | 待测 | 待测 | 1163.93 | PASS; repeat + changed inputs |
| C2 | baseline | 520.006 / 522 | 待测 | 522 | 待测 | 待测 | 1549.09 | PASS; repeat + changed inputs |
| C2 | no unused tmp | 480.006 / 522 | 待测 | 482 | 待测 | 待测 | 1548.73 | PASS; repeat + changed inputs |
| C2 | compact receive | 440.006 / 522 | 待测 | 442 | 待测 | 待测 | 1547.04 | PASS; repeat + changed inputs |
| C2 | input/output alias | 280.006 / 522 | 待测 | 282 | 待测 | 待测 | 1548.02 | PASS; repeat + changed inputs |

| Lifecycle | Targets | Result |
|---|---|---|
| 4096 → 16384 → 8192 → 4096; distinct counter sets | C1/C2 | 待测 |
| Changed operands / rank values | C1/C2/G1/G2 | 待测 |
| Alternate warmed graphs; retain storage and launchers | C1/C2/G1/G2 | 待测 |
| Active M below capacity; region bounds and synchronized close | C1/C2 | 待测 |

</details>

<details>
<summary>F — Target FP8 communication</summary>

C1/C2 only. Use aligned supported configurations, separate reduction and
matching scale tiles. Record mathematical and actual-wire errors separately;
no model-quality claim follows from these operator tests.

| Target | M | Gather | BF16 scatter µs | FP8 scatter µs | FP32 relL2 | Max abs/p99 error | Wire reference |
|---|---|---|---|---|---|---|---|
| C1 | 4096 | bf16 | 338.57 | 342.65 | 0.02621 | 7.7e-05 / 3.33e-05 (sampled p99) | PASS; relL2 0 |
| C1 | 4096 | fp8/lsa | 286.48 | 285.41 | 0.03608 | 0.000131 / 5.02e-05 (sampled p99) | PASS; relL2 0 |
| C1 | 8192 | bf16 | 612.52 | 631.49 | 0.02620 | 8.18e-05 / 3.29e-05 (sampled p99) | PASS; relL2 0 |
| C1 | 8192 | fp8/lsa | 505.08 | 508.80 | 0.03607 | 0.000134 / 4.94e-05 (sampled p99) | PASS; relL2 0 |
| C1 | 16384 | bf16 | 1167.11 | 1201.84 | 0.02621 | 8.93e-05 / 3.32e-05 (sampled p99) | PASS; relL2 0 |
| C1 | 16384 | fp8/lsa | 966.39 | 990.95 | 0.03607 | 0.000158 / 4.98e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 4096 | bf16 | 440.07 | 367.70 | 0.02617 | 0.000111 / 4.49e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 4096 | fp8/lsa | 360.64 | 285.36 | 0.03482 | 0.00017 / 6.45e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 8192 | bf16 | 816.21 | 663.12 | 0.02618 | 0.000116 / 4.48e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 8192 | fp8/lsa | 655.43 | 500.65 | 0.03482 | 0.000194 / 6.44e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 16384 | bf16 | 1548.42 | 1221.29 | 0.02618 | 0.00012 / 4.48e-05 (sampled p99) | PASS; relL2 0 |
| C2 | 16384 | fp8/lsa | 1234.39 | 906.40 | 0.03482 | 0.000189 / 6.44e-05 (sampled p99) | PASS; relL2 0 |

| Input/protocol case | Required check | Result |
|---|---|---|
| All zero / exact FP8 values | Absolute error and finite scales; no positive-relL2 floor | 待测 |
| Per-rank constants / impulse / cancellation | Correct owners and contributors, including near-zero references | 待测 |
| Supported magnitudes, outliers and rounding midpoints | Payload/scales, error distribution and saturation behavior | 待测 |
| Both FP8 legs | Separate multiply, scatter reduction and gather error | PASS on random inputs; math and both actual wires checked |
| Unsupported layout/transport/fusion combinations | Host rejection on C1/C2 | 待测 |

</details>

<details>
<summary>Q — Target grouped reduce/quantize</summary>

C1/C2 at M=16384, FP8 LSA gather. Retain same-target negative controls;
fewer launches alone do not justify promotion.

| Target | Quantization | Reduce µs | Quantize µs | Gather/pull µs | Whole call µs | relL2 | Reference |
|---|---|---|---|---|---|---|---|
| C1 | separate per-row | 待测 | 待测 | 待测 | 964.51 | 0.02491 | PASS math; repeat + changed inputs |
| C1 | fused group128 | 待测 | 待测 | 待测 | 1005.04 | 0.02422 | PASS math; repeat + changed inputs |
| C1 | fused group256 | 待测 | 待测 | 待测 | 1005.00 | 0.02458 | PASS math; repeat + changed inputs |
| C1 | fused group512 | 待测 | 待测 | 待测 | 1005.75 | 0.02477 | PASS math; repeat + changed inputs |
| C2 | separate per-row | 待测 | 待测 | 待测 | 1233.42 | 0.02309 | PASS math; repeat + changed inputs |
| C2 | fused group128 | 待测 | 待测 | 待测 | 1262.56 | 0.02243 | PASS math; repeat + changed inputs |
| C2 | fused group256 | 待测 | 待测 | 待测 | 1264.07 | 0.02276 | PASS math; repeat + changed inputs |
| C2 | fused group512 | 待测 | 待测 | 待测 | 1264.41 | 0.02294 | PASS math; repeat + changed inputs |

</details>

<details>
<summary>T — Diagnostics at target shapes only</summary>

Diagnostics keep the target N/K/TP and operand quantization fixed. Vary M
or the implementation knob. Synthetic N/K scans and ptpc controls are outside
this plan. Kernel profiles and whole-call times remain separate metrics.

##### Pull grid: C1/C2, M=16384, FP8 LSA gather

| Target | Blocks | Whole call µs | Pull kernel µs | Reference |
|---|---|---|---|---|
| C1 | 16 | 待测 | 待测 | 待测 |
| C1 | 24 | 待测 | 待测 | 待测 |
| C1 | 32 | 待测 | 待测 | 待测 |
| C1 | 48 | 待测 | 待测 | 待测 |
| C1 | 64 | 待测 | 待测 | 待测 |
| C1 | 80 | 待测 | 待测 | 待测 |
| C1 | 128 | 待测 | 待测 | 待测 |
| C1 | 256 | 待测 | 待测 | 待测 |
| C1 | 512 | 待测 | 待测 | 待测 |
| C2 | 16 | 待测 | 待测 | 待测 |
| C2 | 24 | 待测 | 待测 | 待测 |
| C2 | 32 | 待测 | 待测 | 待测 |
| C2 | 48 | 待测 | 待测 | 待测 |
| C2 | 64 | 待测 | 待测 | 待测 |
| C2 | 80 | 待测 | 待测 | 待测 |
| C2 | 128 | 待测 | 待测 | 待测 |
| C2 | 256 | 待测 | 待测 | 待测 |
| C2 | 512 | 待测 | 待测 | 待测 |

##### Store mapping: target GEMM-only controls at M=4096

Both target quantizations require swapped operands. Do not benchmark the unsupported no-swap variant.

| Target | Store mapping | Kernel µs | Whole call µs | Store counters | VGPR/scratch | Reference |
|---|---|---|---|---|---|---|
| C1 | swap only | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | swap + permlane | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | swap + permlane + transpose | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | swap only | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | swap + permlane | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | swap + permlane + transpose | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | swap only | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | swap + permlane | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | swap + permlane + transpose | 待测 | 待测 | 待测 | 待测 | 待测 |

##### Grid remainder: G1/G2, fixed N/K, vary M

| Target | M | Tiles/remainder | N128 µs | N256 µs | Reference |
|---|---|---|---|---|---|
| G1 | 1024 | 待测 | 待测 | 待测 | 待测 |
| G1 | 1280 | 待测 | 待测 | 待测 | 待测 |
| G1 | 1536 | 待测 | 待测 | 待测 | 待测 |
| G1 | 1792 | 待测 | 待测 | 待测 | 待测 |
| G1 | 2048 | 待测 | 待测 | 待测 | 待测 |
| G1 | 2304 | 待测 | 待测 | 待测 | 待测 |
| G1 | 4096 | 待测 | 待测 | 待测 | 待测 |
| G2 | 1024 | 待测 | 待测 | 待测 | 待测 |
| G2 | 1280 | 待测 | 待测 | 待测 | 待测 |
| G2 | 1536 | 待测 | 待测 | 待测 | 待测 |
| G2 | 1792 | 待测 | 待测 | 待测 | 待测 |
| G2 | 2048 | 待测 | 待测 | 待测 | 待测 |
| G2 | 2304 | 待测 | 待测 | 待测 | 待测 |
| G2 | 4096 | 待测 | 待测 | 待测 | 待测 |

##### Aligned chunks: C1/C2, M=16384, target BLOCK_M

| Target | Chunks | GEMM µs | Drain µs | Reduce µs | Gather µs | Whole call µs | Reference |
|---|---|---|---|---|---|---|---|
| C1 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### Reduce-triggered gather: C1/C2, M=16384, BF16 wire

| Target | Publish | Bands | Whole call µs | Reference |
|---|---|---|---|---|
| C1 | writethrough | 1 | 待测 | 待测 |
| C1 | writethrough | 4 | 待测 | 待测 |
| C1 | writethrough | 8 | 待测 | 待测 |
| C1 | writethrough | 16 | 待测 | 待测 |
| C1 | writethrough | 32 | 待测 | 待测 |
| C1 | fence | 1 | 待测 | 待测 |
| C1 | fence | 4 | 待测 | 待测 |
| C1 | fence | 8 | 待测 | 待测 |
| C1 | fence | 16 | 待测 | 待测 |
| C1 | fence | 32 | 待测 | 待测 |
| C2 | writethrough | 1 | 待测 | 待测 |
| C2 | writethrough | 4 | 待测 | 待测 |
| C2 | writethrough | 8 | 待测 | 待测 |
| C2 | writethrough | 16 | 待测 | 待测 |
| C2 | writethrough | 32 | 待测 | 待测 |
| C2 | fence | 1 | 待测 | 待测 |
| C2 | fence | 4 | 待测 | 待测 |
| C2 | fence | 8 | 待测 | 待测 |
| C2 | fence | 16 | 待测 | 待测 |
| C2 | fence | 32 | 待测 | 待测 |

##### MXFP8 scale layout: G1/G2, identical operands

| Target | M | Layout | Hot µs | Cold µs | VMEM/cache counters | Reference/equality |
|---|---|---|---|---|---|---|
| G1 | 4096 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 4096 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 4096 | row-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 8192 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 8192 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 8192 | row-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 16384 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 16384 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G1 | 16384 | row-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 4096 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 4096 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 4096 | row-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 8192 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 8192 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 8192 | row-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 16384 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 16384 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| G2 | 16384 | row-major | 待测 | 待测 | 待测 | 待测 |

##### Window geometry: C1/C2, M=16384, target quantization

| Target | Mode | Window µs | CachedWindow µs | Paired spread | Reference |
|---|---|---|---|---|---|
| C1 | split-lsa | 待测 | 待测 | 待测 | 待测 |
| C1 | fused-sdma | 待测 | 待测 | 待测 | 待测 |
| C2 | split-lsa | 待测 | 待测 | 待测 | 待测 |
| C2 | fused-sdma | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>S — Target specialization and graph ownership</summary>

| Case | Targets and required evidence | Result |
|---|---|---|
| Factory specialization | C1/C2 configurations A → B → A; G1/G2 both tile variants | 待测 |
| Fresh and cached processes | Same target shapes and changed inputs; record cache state | 待测 |
| Graph ownership | Keep modules, launchers, tensors and graphs alive through completion | 待测 |
| Counter stability | C1/C2 repeated serial calls, timeout and exit status | 待测 |
| Supported composition | Combine options only within C1/C2 declared contracts | 待测 |
| Unsupported composition | Host rejection without silently selecting another route | 待测 |

</details>

## Historical records

Only target-workload measurements are displayed here. They keep their original
scope and numerical values and do not populate the new `待测` cells. Earlier
all-layer surveys, other TP configurations, decode studies and detailed unrelated
investigations remain in the external archive and in Git history (the complete
previous document is in commit `6a0cd24a`). They are not an optimization backlog.

Historical aggregate thresholds from the broad survey are not re-labeled as
G1/G2 results. Recompute target-specific summaries from the new matrices when
measurements are explicitly requested.

<details>
<summary>Historical Flash TP4 standalone GEMM — target rows</summary>

Flash TP4 `wo_b` and `wq_b` rows extracted verbatim from the historical
linear comparison with SGLang. Negative percentages mean MORI was faster. The
other shapes and the broad-survey aggregate statistics are outside this scope.

##### 256-column tile

| layer | N x K | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | +113% | +88% | +3% | **-28%** | **-30%** | **-32%** | **-43%** |
| `wo_b` (TP4) | 5120 x 2048 | +126% | +52% | **-5%** | **-34%** | **-26%** | **-31%** | **-27%** |

##### 128-column tile

| layer | N x K | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | +85% | +77% | -1% | **-2%** | **-3%** | -1% | **-17%** |
| `wo_b` (TP4) | 5120 x 2048 | +91% | +41% | **-10%** | **-6%** | **-10%** | **-12%** | +0% |

</details>

<details>
<summary>Historical V4 Pro TP8 wo_b GEMM + AR</summary>

V4 Pro TP8 `wo_b`, N=7168, K=2048, blockscale. The first table is the
recorded mode comparison. The phase table came from a model-driven prefill and
is retained only as historical operator breakdown, not a new end-to-end result.

| M | `split-sdma` | `fused-sdma` | `fused-sdma` + fp8 gather |
|---|---:|---:|---:|
| 4096 | 398.1 us | 351.0 us | **329.5 us** |
| 8192 | 722.0 | 621.1 | **539.3** |
| 16384 | 1472.9 | 1148.8 | **979.3** |

##### Historical wo_b phase breakdown

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

##### Historical FP8 scale-granularity study for the target slice

| scale granularity | relL2 | scale bytes |
|---|---:|---:|
| per row (7168) | 2.646e-2 | 0.06% |
| per 512 | 2.631e-2 | 0.78% |
| per 256 | 2.609e-2 | 1.56% |
| per 128 | 2.572e-2 | 3.12% |
| per 32 | 2.399e-2 | 12.5% |

</details>

<details>
<summary>Historical V4.1 Flash TP4 wo_b GEMM + AR</summary>

V4.1 Flash TP4 `wo_b`, N=5120, K=2048, MXFP8; BF16 scatter. These are
separate historical runs, with their original graph/linear comparison scopes.
FP8 denotes the gather leg in these tables.

##### Mode comparison

| M | wire | `gemm-only` | `split-sdma` | `fused-sdma` | gain | ceiling |
|---|---|---:|---:|---:|---:|---:|
| 4096 | bf16 | 65.6 | 443.6 | 437.3 | +1.4% | 14.8% |
| 8192 | bf16 | 105.0 | 827.3 | 801.8 | +3.1% | 12.7% |
| 16384 | bf16 | 180.2 | 1593.5 | 1500.7 | +5.8% | 11.3% |
| 4096 | fp8 | 65.8 | 364.8 | 358.8 | +1.7% | 18.0% |
| 8192 | fp8 | 102.9 | 669.7 | 643.1 | +4.0% | 15.4% |
| 16384 | fp8 | 176.3 | 1309.2 | 1195.1 | +8.7% | 13.5% |

##### Gather transport and fused quantization

| | gather=sdma, fq off | fq on | **gather=lsa, fq off** | fq on |
|---|---:|---:|---:|---:|
| `fused-sdma` | 1222.5 | 1244.1 | **1192.1** | 1209.9 |
| `fused-lsa` | 1412.1 | 1433.0 | 1375.6 | 1398.7 |
| `split-sdma` | 1342.1 | 1358.9 | 1310.3 | 1326.8 |

##### True-M / padding comparison against the native linear + AR path

| M | m_pad | fill | today | bf16 wire | fp8 wire |
|---|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1.000 | 164.6 us | +16.4% | +10.3% |
| 2048 | 2048 | 1.000 | 283.3 | +1.8% | -8.8% |
| 4096 | 4096 | 1.000 | 482.8 | -0.9% | -15.4% |
| 4200 | 5120 | 0.820 | 506.1 | +17.0% | -2.3% |
| 4700 | 5120 | 0.918 | 548.3 | +8.4% | -9.3% |
| 5120 | 5120 | 1.000 | 601.5 | -5.8% | -20.7% |
| 7200 | 8192 | 0.879 | 1051.2 | -17.6% | -32.7% |
| 8192 | 8192 | 1.000 | 1042.5 | -19.4% | -34.1% |
| 8200 | 9216 | 0.890 | 1185.1 | -20.8% | -35.7% |
| 9200 | 9216 | 0.998 | 1157.5 | -19.2% | -34.2% |
| 13000 | 13312 | 0.977 | 1547.8 | -8.5% | -25.3% |
| 16384 | 16384 | 1.000 | 1851.3 | -16.2% | -32.9% |

</details>

<details>
<summary>Historical target chunk, padding, workspace and FP8 experiments</summary>

Frozen 2026-09-29 C1/C2 experiments on gfx950, Torch 2.10 / HIP 7.2 /
FlyDSL 0.2.4. Prefill timings used paired multi-call graphs and exclude input
quantization/padding, weight packing and compilation. These are historical
operator microbenchmarks, not current template results.

##### Short final chunks

TP8, N=7168, K=2048, blockscale operands, BF16 communication. Each chunk has at
most two 128-row bands; the final chunk's completion count and SDMA byte count
follow its actual length.

| M | Baseline µs | Short-tail µs | Latency reduction |
|---:|---:|---:|---:|
| 11264 | 1046.23 | 819.09 | 21.7% |
| 13312 | 1242.10 | 949.82 | 23.5% |
| 16384 | 1133.65 | 1138.83 | -0.5% |

At M=11264/13312, each peer owns 11/13 bands. The divisor-based baseline uses
one chunk; the new path uses 6/7 and restores scatter overlap. M=16384 already
has eight chunks and gains nothing. Outputs are bitwise equal to the baseline,
including after changing inputs.

Enable with `compile_fused_gemm_scatter(chunk_bands=2)` and a matching
`counter_chunks=ceil(bands_per_peer / 2)`. The public operator still uses its
divisor-based chunk selection.

##### Reduced padding

TP4, N=5120, K=2048, MXFP8 operands, BF16 communication. Tiles may cross
owner/chunk row boundaries; completion counters count tile/segment intersections.

| True M | Baseline padded M / µs | Align 256 padded M / µs | Align 64 padded M / µs | Best latency reduction |
|---:|---|---|---|---:|
| 4200 | 5120 / 518.56 | 4352 / 460.18 | 4224 / 449.26 | 13.4% |
| 8200 | 9216 / 862.05 | 8448 / 822.24 | 8256 / 807.00 | 6.4% |

True output rows are bitwise equal to the baseline, including changed-input
checks. `relaxed_rows=True` requires fused SDMA, `chunk_bands > 0`, at least
one full tile per peer, and BF16 scatter. The external harness supplies the
reduced padded shapes; public support predicates, padding and dynamic-M plan
caching still use the existing alignment contract.

##### Workspace reuse

TP8, M=16384, N=7168, K=2048, blockscale operands, BF16 communication.

| Layout | Window MiB/rank | µs |
|---|---:|---:|
| Baseline | 700.006 | 1134.29 |
| Remove unused SDMA tmp | 672.006 | 1136.39 |
| Also remove self receive slot | 644.006 | 1134.72 |
| Also alias input/output | 420.006 | 1135.58 |

The final layout saves 280 MiB/rank (40%) with effectively unchanged latency
and bitwise equal output, including changed inputs. It relies on serial calls
and scatter drain completing before input/output reuse. The external harness
supplies this layout; it is not the public operator's allocation policy and
has not been validated for concurrent calls or arbitrary transport/option
combinations.

##### FP8 scatter

TP4, M=16384, N=5120, K=2048, MXFP8 operands. Three-round paired performance
runs, eight scatter chunks:

| Gather | BF16 scatter µs | FP8 scatter µs | Latency reduction | FP32 reference relL2, before / after |
|---|---:|---:|---:|---|
| BF16 | 1487.88 | 1182.65 | 20.5% | 0.00237 / 0.02618 |
| FP8 / LSA | 1183.30 | 881.21 | 25.5% | 0.02309 / 0.03482 |

Four chunks were slower: approximately 1207.06/904.56 µs with BF16/FP8 gather.
The additional correctness reruns are separate from these paired timings.

At M=4096, stage-by-stage validation checked the actual FP8 wire payload and
scales. Scale relative error was about 6.6e-8; differences from nominal PyTorch
quantization were adjacent FP8 values at rounding midpoints. An independent
sum reconstructed from the actual payload and scales matched the collective
output exactly with BF16 gather.

Relative to the FP32 reference, FP8 scatter raises L2 error to about **2.62%**;
using FP8 for both legs raises it to about **3.48%**. No model-quality test has
been run for FP8 scatter. Earlier gather-only quality results do not validate
this option.

##### Grouped reduce + quantize

TP4, M=16384, FP8 gather. Folding groupwise quantization into the flat-pack
reduce mapping did not improve latency:

| Variant | µs | Change |
|---|---:|---:|
| Separate reduce + quantize | 1182.9 | — |
| Group 128 | 1212.1 | 2.5% slower |
| Group 256 | 1213.1 | 2.6% slower |
| Group 512 | 1213.7 | 2.6% slower |

Correctness passed. This is a negative result for these mappings; fewer
launches did not offset the additional reduction/quantization work.

</details>

<details>
<summary>Earlier target-only operator rerun snapshot</summary>

Target rows from the earlier operator rerun snapshot. Linear timings did
not provide an independent reference and must not be treated as correctness
passes. The comparator was SGLang `e29ebcd051` with a host-only optional-output
adapter. The remaining snapshot and failed attempts stay in the external archive.

##### SGLang baseline, cold µs

| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 10.75 | 15.42 | 26.16 | 42.42 | 70.54 | 132.34 | 286.30 |
| wo_b | 12.64 | 19.80 | 35.09 | 50.65 | 76.19 | 135.67 | 242.37 |

##### MORI N256, cold µs and delta

| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 31.98 (+197.5%) | 35.45 (+130.0%) | 25.58 (-2.2%) | 29.35 (-30.8%) | 49.84 (-29.4%) | 90.50 (-31.6%) | 179.54 (-37.3%) |
| wo_b | 28.69 (+127.0%) | 29.47 (+48.9%) | 32.65 (-7.0%) | 34.95 (-31.0%) | 60.54 (-20.5%) | 95.85 (-29.4%) | 175.10 (-27.8%) |

##### MORI N128, cold µs and delta

| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 26.48 (+146.3%) | 30.66 (+98.9%) | 24.24 (-7.3%) | 39.45 (-7.0%) | 70.09 (-0.6%) | 130.37 (-1.5%) | 256.88 (-10.3%) |
| wo_b | 23.85 (+88.7%) | 27.10 (+36.9%) | 31.01 (-11.6%) | 50.80 (+0.3%) | 74.08 (-2.8%) | 121.65 (-10.3%) | 240.10 (-0.9%) |

##### C2 operator snapshot

| Case | TP | M | N | K | Quant | Mode | Actual gather | Fused quantize/push | Chunks/bands | µs | relL2 | Passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 4 | 4096 | 5120 | 2048 | mxfp8 | gemm-only | none | False/False | None/8 | 65.76 | 0.00166 | True |
| 01 | 4 | 4096 | 5120 | 2048 | mxfp8 | split-sdma | bf16/sdma | False/False | None/8 | 442.56 | 0.00237 | True |
| 02 | 4 | 4096 | 5120 | 2048 | mxfp8 | split-lsa | bf16/lsa | False/False | None/8 | 427.76 | 0.00237 | True |
| 03 | 4 | 4096 | 5120 | 2048 | mxfp8 | fused-sdma | bf16/sdma | False/False | 4/8 | 436.64 | 0.00237 | True |
| 04 | 4 | 4096 | 5120 | 2048 | mxfp8 | fused-lsa | bf16/sdma | False/False | 4/8 | 476.68 | 0.00237 | True |
| 05 | 4 | 8192 | 5120 | 2048 | mxfp8 | gemm-only | none | False/False | None/8 | 107.36 | 0.00166 | True |
| 06 | 4 | 8192 | 5120 | 2048 | mxfp8 | split-sdma | bf16/sdma | False/False | None/8 | 826.05 | 0.00237 | True |
| 07 | 4 | 8192 | 5120 | 2048 | mxfp8 | split-lsa | bf16/lsa | False/False | None/8 | 804.81 | 0.00237 | True |
| 08 | 4 | 8192 | 5120 | 2048 | mxfp8 | fused-sdma | bf16/sdma | False/False | 8/8 | 802.60 | 0.00237 | True |
| 09 | 4 | 8192 | 5120 | 2048 | mxfp8 | fused-lsa | bf16/sdma | False/False | 8/8 | 886.77 | 0.00237 | True |
| 10 | 4 | 16384 | 5120 | 2048 | mxfp8 | gemm-only | none | False/False | None/8 | 185.44 | 0.00166 | True |
| 11 | 4 | 16384 | 5120 | 2048 | mxfp8 | split-sdma | bf16/sdma | False/False | None/8 | 1590.17 | 0.00237 | True |
| 12 | 4 | 16384 | 5120 | 2048 | mxfp8 | split-lsa | bf16/lsa | False/False | None/8 | 1605.45 | 0.00237 | True |
| 13 | 4 | 16384 | 5120 | 2048 | mxfp8 | fused-sdma | bf16/sdma | False/False | 8/8 | 1504.89 | 0.00237 | True |
| 14 | 4 | 16384 | 5120 | 2048 | mxfp8 | fused-lsa | bf16/sdma | False/False | 8/8 | 1702.89 | 0.00237 | True |

##### C2 operator snapshot

| Case | TP | M | N | K | Quant | Mode | Actual gather | Fused quantize/push | Chunks/bands | µs | relL2 | Passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 4 | 4096 | 5120 | 2048 | mxfp8 | gemm-only | none | False/False | None/8 | 65.96 | 0.00166 | True |
| 01 | 4 | 4096 | 5120 | 2048 | mxfp8 | split-sdma | fp8/lsa | False/False | None/8 | 364.16 | 0.0231 | True |
| 02 | 4 | 4096 | 5120 | 2048 | mxfp8 | fused-sdma | fp8/lsa | False/False | 4/8 | 358.72 | 0.0231 | True |
| 03 | 4 | 4096 | 5120 | 2048 | mxfp8 | fused-lsa | fp8/lsa | False/False | 4/8 | 394.00 | 0.0231 | True |
| 04 | 4 | 8192 | 5120 | 2048 | mxfp8 | gemm-only | none | False/False | None/8 | 106.08 | 0.00166 | True |
| 05 | 4 | 8192 | 5120 | 2048 | mxfp8 | split-sdma | fp8/lsa | False/False | None/8 | 670.61 | 0.0231 | True |
| 06 | 4 | 8192 | 5120 | 2048 | mxfp8 | fused-sdma | fp8/lsa | False/False | 8/8 | 645.20 | 0.0231 | True |
| 07 | 4 | 8192 | 5120 | 2048 | mxfp8 | fused-lsa | fp8/lsa | False/False | 8/8 | 727.41 | 0.0231 | True |

</details>

<a id="end-to-end-in-sglang"></a>

Model end-to-end and quality records are archived outside this target operator
plan. GEMV/decode is also deferred. No test or implementation is removed from the
library merely because it is outside this document's optimization scope.

## Regression checklist

Only C1/C2/G1/G2 are optimization targets. Reuse existing relevant correctness
checks without adding model layers or TP variants to the performance matrix.
The tables are an inventory for later work, not a request to run sweeps now.

| Tier | Trigger | Target coverage | State |
|---|---|---|---|
| Core correctness | Relevant code/API changes | References, packing, B boundaries, S caching, W serial lifecycle and F wire checks | 待测 |
| Representative performance | Kernel/dispatch/transport changes | G1/G2 tile boundaries; C1/C2 ragged M, awkward chunks and large prefill | 待测 |
| Target performance matrix | Before changing defaults or a target-specific audit | L1/K1 and A; only the four registry entries | 待测 |
| Targeted diagnostics | Related mapping/publish/layout changes | C/P/W/F/Q/T at fixed target N/K/TP | 待测 |
| Deferred work | Separate explicit request | Decode and model evaluation | Deferred |

- [ ] Record the target ID, revisions, scope and effective configuration.
- [ ] Validate the corresponding independent mathematical/wire reference.
- [ ] Exercise relevant M boundaries, changing inputs, cache and serial reuse.
- [ ] Record unsupported option combinations and failed attempts separately.
- [ ] Keep absolute latency, paired dispersion and memory/initialization costs.
- [ ] Preserve target historical results; fill `待测` only from matching measurements.
- [ ] Do not promote a target-specific result to a general all-layer heuristic.

Future measurement entry points (reference only):

| Target | Entry point and fixed selection |
|---|---|
| G1 | [bench_gemm.py](../../../../benchmark/cco/flydsl/gemm_ar/bench_gemm.py) with `--shape wo_b`, choosing `--scope kernel` or `--scope linear` explicitly |
| G2 | Same benchmark with `--shape wq_b` and explicit scope |
| C1 | [bench_gemm_ar.py](../../../../benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py) with TP8, `--quant blockscale -n 7168 -k 2048` |
| C2 | Same collective benchmark with TP4, `--quant mxfp8 -n 5120 -k 2048` |
| Contract/lifecycle checks | Relevant GEMM/AR cases in [test_gemm_ar_op.py](../../../../tests/python/cco/test_gemm_ar_op.py) and [test_gemm_ar.py](../../../../tests/python/cco/test_gemm_ar.py); select those cases rather than every test in the files |

The [benchmark guide](../../../../docs/MORI-GEMM-AR-BENCHMARK.md) contains broader
historical tooling. Use the target selections above rather than an all-shape
preset for this work. Experimental scripts, logs and the full earlier reports
remain outside the repository.
