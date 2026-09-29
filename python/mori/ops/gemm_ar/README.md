# `mori.ops.gemm_ar`

GEMM + all-reduce optimization for `wo_b`, and standalone MXFP8 GEMM optimization
for the two Flash TP4 attention projections below. The target list is fixed.
General API support and earlier broad surveys do not expand this work's scope.

**Document preparation only:** result cells remain `待测`. No benchmarks are
launched by this edit. GEMV/decode, model-server/end-to-end evaluation and model
quality are deferred. Other layers and other TP configurations are outside the
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
- [Pending target matrices](#measured-results)
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
| Linear operator | Activation quantization, padding, dispatch and multiply | Independently validated conversion/packing plus product reference |
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

| Record field | Value for the next measurement |
|---|---|
| MORI / comparator revisions and any compatibility patch | 待测 |
| GPU architecture, count, clocks and background activity | 待测 |
| Torch / HIP / FlyDSL / compiler / FFI versions | 待测 |
| Logical M, physical M, N, K, TP, operand and wire dtypes | 待测 |
| GEMM tile, chunks, padding, workspace and transport | 待测 |
| Scope, graph calls, warmup, replays and independent rounds | 待测 |
| Weight ring sizes and every rotated weight representation | 待测 |
| Reference definition, tolerances and numerical results | 待测 |
| Per-rank times, paired round medians, spread / confidence interval | 待测 |
| Window bytes, Torch allocation peak and external allocation accounting | 待测 |
| Exact command, raw log/result paths and final outcome | 待测 |

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
  interval between independent multi-rank jobs when those jobs are run later.
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

The current defaults remain unchanged. Historical directions identify what to
check next on the target workloads; they are not newly measured results.

| Topic | Target | Historical direction | Required confirmation | New result |
|---|---|---|---|---|
| Short final chunks | C1/C2 | Promising at awkward band counts | Tail/count/transfer coverage | 待测 |
| Reduced padding | C1/C2 | Promising at ragged M | Owner/chunk intersections | 待测 |
| Workspace reuse | C1/C2 | Reduced window requirement | Serial lifecycle and actual peak allocation | 待测 |
| FP8 scatter/gather | C1/C2 | Lossy performance option | Wire reference and numerical error | 待测 |
| Grouped reduce/quantize | C1/C2 | Negative control | Same-shape time and correctness | 待测 |
| GEMM tile / scale layout | G1/G2 | Internal tile and packing choices | Target-specific boundaries and references | 待测 |

## Measured results

These are empty templates for the four targets only. `待测` is not zero, a pass,
or a historical value. N/K/TP and operand quantization come from the target
registry. Keep M and a relevant implementation knob as the sweep axes.

The main standalone ladder is M=64/256/1024/2048/4096/8192/16384; nearby padding
and tile-switch values are separate boundary cases. There is no decode/GEMV
matrix and no small-M GEMM performance sweep intended to stand in for GEMV.

<details>
<summary>L1 — Target linear-operator matrix (14 shape/M points)</summary>

Flash TP4 only. Quantization, padding and dispatch are inside linear scope.
Cells are cold median µs; the per-case record also stores hot timing, spread,
resolved tile, physical M and the independent-reference result.

| Target | Layer | N | K | M | Expected M_pad | SG linear | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | wo_b | 5120 | 2048 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>K1 — Target standalone MXFP8 GEMM kernels</summary>

G1/G2 only, with prequantized and prepacked operands. Kernel-only timing is
separate from the linear comparison and uses an independent FP32 reference.
C1's blockscale multiply is measured only as the GEMM-only control in A.

| Target | Layer | N | K | M | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|
| G1 | wo_b | 5120 | 2048 | 64 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 256 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 1024 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 4096 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 8192 | 待测 | 待测 | 待测 | 待测 |
| G1 | wo_b | 5120 | 2048 | 16384 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 64 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 256 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 1024 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 2048 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 4096 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 8192 | 待测 | 待测 | 待测 | 待测 |
| G2 | wq_b | 8192 | 1280 | 16384 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>B — Target boundaries and contract checks</summary>

| ID | Target | Boundary / input | Expected check | Result |
|---|---|---|---|---|
| B1 | G1/G2 | M=63,64,65,127,128,129,255,256,257 | Padding before quantization; true output rows | 待测 |
| B2 | G2 | M=1023,1024,1025,1088 | Both sides of the 140-grid tile switch | 待测 |
| B3 | G1 | M=1535,1536,1537,1600 | Both sides of the 140-grid tile switch | 待测 |
| B4 | G1/G2 | Nearest reachable M around admission grid 48/64/80/128 | Admission and tile choice scored separately | 待测 |
| B5 | C1/C2 | M=1023,1024,1025,4200,8200 | Target padding/owner boundaries | 待测 |
| B6 | All four targets | Wrong operand/scale dtype, shape or stride | Host rejection; no timing of invalid inputs | 待测 |
| B7 | C1/C2 | Invalid communication/option combinations | Host rejection without silently changing route | 待测 |

</details>

<details>
<summary>D — Target tile crossover and admission summaries</summary>

Derived only from G1/G2. Earlier all-layer aggregate scores are not reused
as evidence for these target-specific thresholds.

| Wide-grid bin | Target points | N128 faster | N256 faster | Mean Δ% | Spread |
|---|---|---|---|---|---|
| 1–16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 17–32 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 33–64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 65–128 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 129–192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| >192 | 待测 | 待测 | 待测 | 待测 | 待测 |

| Admission grid threshold | G1/G2 points | Wrong-choice rate | Mean latency penalty | Tie/noise rule |
|---|---|---|---|---|
| 48 | 待测 | 待测 | 待测 | 待测 |
| 64 | 待测 | 待测 | 待测 | 待测 |
| 80 | 待测 | 待测 | 待测 | 待测 |
| 128 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>A — wo_b GEMM + all-reduce mode/wire matrix</summary>

Only C1/C2 (`wo_b`). Every row uses the target's operand quantization and
BF16 scatter. GEMM-only appears once per target/M; unused gather options do not
create additional compute cases. Shared FP8 configurations have one case ID.

##### C1: V4 Pro, TP8, wo_b, blockscale, N=7168, K=2048

| Mode | Gather dtype | Actual gather transport | Fused quantize | M=4096 µs | M=8192 µs | M=16384 µs | Reference |
|---|---|---|---|---|---|---|---|
| gemm-only | none | none | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-lsa | bf16 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |

##### C2: V4.1 Flash, TP4, wo_b, mxfp8, N=5120, K=2048

| Mode | Gather dtype | Actual gather transport | Fused quantize | M=4096 µs | M=8192 µs | M=16384 µs | Reference |
|---|---|---|---|---|---|---|---|
| gemm-only | none | none | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-lsa | bf16 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | bf16 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| split-sdma | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | sdma | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | sdma | on | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | lsa | off | 待测 | 待测 | 待测 | 待测 |
| fused-lsa | fp8 | lsa | on | 待测 | 待测 | 待测 | 待测 |

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
| 11 | 11264 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 12288 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 13312 | 待测 | 待测 | 待测 | 待测 | 待测 |
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
| 1023 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1024 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1025 | 2048 | 1280 | 1088 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 5120 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 9216 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C2: mxfp8, TP4, BLOCK_M=256

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Reference |
|---|---|---|---|---|---|---|---|
| 1023 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1024 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1025 | 2048 | 1280 | 1088 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 5120 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 9216 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>W — Target workspace and serial lifecycle</summary>

C1/C2, capacity M=16384. Separate window requirements, actual allocator
peaks and initialization. Serial mixed-M sequences are checked on these same
N/K/TP targets; do not expand the model/parallelism matrix.

| Target | Layout | Window MiB/rank | Torch peak MiB | External physical MiB | Allocations | Init µs | Steady µs | Reference |
|---|---|---|---|---|---|---|---|---|
| C1 | baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | no unused tmp | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | compact receive | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | input/output alias | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | no unused tmp | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | compact receive | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | input/output alias | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

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
| C1 | 4096 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 4096 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 8192 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 8192 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 16384 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | 16384 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 4096 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 4096 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 8192 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 8192 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 16384 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | 16384 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 |

| Input/protocol case | Required check | Result |
|---|---|---|
| All zero / exact FP8 values | Absolute error and finite scales; no positive-relL2 floor | 待测 |
| Per-rank constants / impulse / cancellation | Correct owners and contributors, including near-zero references | 待测 |
| Supported magnitudes, outliers and rounding midpoints | Payload/scales, error distribution and saturation behavior | 待测 |
| Both FP8 legs | Separate multiply, scatter reduction and gather error | 待测 |
| Unsupported layout/transport/fusion combinations | Host rejection on C1/C2 | 待测 |

</details>

<details>
<summary>Q — Target grouped reduce/quantize</summary>

C1/C2 at M=16384, FP8 LSA gather. Retain same-target negative controls;
fewer launches alone do not justify promotion.

| Target | Quantization | Reduce µs | Quantize µs | Gather/pull µs | Whole call µs | relL2 | Reference |
|---|---|---|---|---|---|---|---|
| C1 | separate per-row | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | fused group128 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | fused group256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C1 | fused group512 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | separate per-row | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | fused group128 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | fused group256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| C2 | fused group512 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

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
