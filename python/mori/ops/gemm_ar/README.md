# `mori.ops.gemm_ar`

FP8 GEMM, GEMV and GEMM + all-reduce operators for tensor-parallel workloads.
This document separates the current API, the validation protocol, empty result
matrices, and preserved historical measurements.

**Documentation preparation only.** Every result cell in the new matrices is
`待测` (not measured in this revision). Historical values remain in the collapsed
records below and are not copied into the new result cells. This reorganization
launches no benchmarks. Model-server/end-to-end and model-quality tests are
outside the current operator-only scope.

- [API and constraints](#api-and-constraints)
- [Validation protocol](#validation-protocol)
- [Current decisions](#current-decisions)
- [Measured results: pending matrices](#measured-results)
- [Historical records](#historical-records)
- [Regression checklist](#regression-checklist)

## API and constraints

| Entry point | Operation | Current contract |
|---|---|---|
| `GemmAllReduceOp` | GEMM with fused scatter, followed by reduction and gather | Single node, TP2/4/8; blockscale or MXFP8; serial use of one instance |
| `Mxfp8GemmOp` | Standalone MXFP8 GEMM | N multiple of 256; K >= 256 and multiple of 128; pad M to a multiple of 64 |
| `Mxfp8GemvOp` | Standalone MXFP8 GEMV | 1 <= M <= 32; N multiple of 32; K multiple of 128 |

A/B operands use `torch.float8_e4m3fn`; FNUZ is a different encoding and is
rejected. `supports`, `supports_gemm` and `supports_gemv` describe shape support,
not a guarantee of a performance win. Unsupported shapes remain explicit
capability cases, with an expected rejection rather than an invented latency.

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

### Standalone operator examples

```python
from mori.ops.gemm_ar import Mxfp8GemmOp, Mxfp8GemvOp
from mori.ops.gemm_ar import preshuffle_a_scale, preshuffle_b

b_shuffled = preshuffle_b(w_fp8)
b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)

gemm = Mxfp8GemmOp(n=N, k=K)
m_pad = gemm.padded_m(x.shape[0])
a_fp8, a_exps = quantize_mxfp8(gemm.pad_rows(x, m_pad))
result = gemm(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale)[: x.shape[0]]

gemv = Mxfp8GemvOp(n=N, k=K)
result = gemv(x_fp8, b_shuffled, x_exps, w_exps)
```

The operators share the shuffled weight, but their scale layouts differ.
GEMM uses packed A scales from `preshuffle_a_scale` and K-block-major widened B
scales. GEMV uses row-major UE8M0 byte scales `[M, K/32]` and `[N/32, K/32]`.
The examples assume the caller has supplied the corresponding quantizers and
tensors; they do not include a model or a server.

### Dispatch and experimental options

`Mxfp8GemmOp` selects its internal 128-column tile when
`ceil(M_pad / 256) * (N / 256) < 140`, otherwise its 256-column tile. This tile
choice is distinct from a caller's decision to admit MORI at all. GEMV selects
from M buckets `(1, 2, 4, 8, 16, 32)`; record the resolved configuration in each
measurement instead of assuming a benchmark's fixed default matches the op.

| Option | Entry point / required conditions | Public default |
|---|---|---|
| Short final chunk | `compile_fused_gemm_scatter(chunk_bands=...)`; matching counter count; fused SDMA | Divisor-based chunks remain |
| Reduced padding | `relaxed_rows=True`, positive `chunk_bands`, at least one full tile per peer, BF16 scatter | Existing public padding remains |
| 5/10-wave GEMV | Explicit kernel configuration; validate N/K and token bucket | Existing dispatch table remains |
| Compact GEMV reduction | `valid_tokens` within the token tile, K-split | Disabled |
| Workspace reuse | Experimental layout; serial calls and scatter drain protect reuse | Existing allocation remains |
| FP8 scatter | Low-level builder/benchmark, not the public constructor; aligned non-compact fused SDMA; matching scale tile; separate reduction | BF16 scatter |
| Decode fusion | Experimental flags/tagged GEMV + AR builders | Separate GEMV and AR |
| Grouped reduce/quantize | Experimental 128/256/512 groups with FP8 LSA gather | Separate phases |

FP8 scatter currently rejects relaxed rows, compact receive storage, direct LSA
scatter, fused rowwise reduce/quantize and reduce-triggered gather puts. Validate
unsupported combinations on the host before launching. These are compatibility
contracts, not new measurements or an instruction to enable the experiments.
The scatter prototype reads completed BF16 GEMM tiles and quantizes row-by-256-
column groups for transfer; it does not quantize directly from accumulators.

## Validation protocol

### Scopes and baselines

| Scope | Timed work | Required reference / comparison |
|---|---|---|
| GEMM/GEMV kernel | Prequantized, prepacked operands; state physical M explicitly | Independent dequantized FP32 product |
| Linear operator | Activation quantization, padding, dispatch and multiply | Independently validated conversion/packing plus product reference |
| GEMM/GEMV + collective | Stated compute and communication phases | Per-rank partial reference plus independent ordered communication reference |
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
| Tile, waves, K split, chunks, padding, workspace and transport | 待测 |
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
`mean(T_selected / min(T_MORI, T_baseline) - 1)`. Declare eligible shapes and the
noise/tie rule. Historical percentages using another denominator or formula
remain historical until recalculated under this definition.

## Current decisions

The API/default column above describes current code. The evidence below is a
historical direction to recheck, not a newly measured result. No default is
changed by this document.

| Topic | Historical direction | Required confirmation | New result |
|---|---|---|---|
| Short final chunks | Promising at awkward band counts | All tail lengths, correct transfer/count coverage | 待测 |
| Reduced padding | Promising at ragged M | Owner/chunk intersections and real-M boundaries | 待测 |
| 5/10-wave GEMV | Shape-specific improvement | Resolved configs, bucket edges, resource use | 待测 |
| Workspace reuse | Reduced window requirement | Mixed-M lifecycle and actual allocation peak | 待测 |
| FP8 scatter | Lossy latency/precision tradeoff | Staged wire correctness and error distribution | 待测 |
| Compact reduction / decode fusion / grouped quantize | Negative results worth retaining | Same-shape controls and correctness | 待测 |
| Tile and admission thresholds | Depend on shape/grid | Boundary measurements and explicit scoring rule | 待测 |

## Measured results

The following are **empty templates** for future operator tests. `待测` is not a
zero, a pass or an unsupported result. Known API expectations are listed
separately. Each collapsed matrix retains the parameter axes needed to populate
it later. Do not copy historical numbers into these cells without a matching
measurement record.

<details>
<summary>L1 — Full linear-operator matrix (120 shape/M points)</summary>

MXFP8; no collective. All twelve historical shapes and all ten M values
are retained. M=1/8/32 GEMM runs are negative controls and belong to the full
study, not every routine regression. M_pad is the expected public 64-row padding.
N=1152 has a low-level 128-column candidate; public/256-column rejection is a
capability check whose observed outcome is still pending.

Cells are cold median µs. The measurement record also retains hot latency,
round spread, reference results and the chosen tile.

| Shape | N | K | M | Expected M_pad | SG linear | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|---|
| wq_b / TP4 | 8192 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 1 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 8 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 64 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 256 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>K1 — Standalone GEMM kernel matrix</summary>

Prequantized and prepacked MXFP8 inputs. The timed physical M is M_pad;
M identifies the logical rows being checked. No SGLang linear timing is used as
a kernel-only comparator. Record the independent FP32 reference separately.

| Shape | N | K | M | Expected M_pad | MORI auto | MORI N128 | MORI N256 | Reference |
|---|---|---|---|---|---|---|---|---|
| wq_b / TP4 | 8192 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 1 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 8 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 32 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 64 | 64 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 4096 | 4096 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 8192 | 8192 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 16384 | 16384 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>V1 — GEMV buckets and FP8/BF16 input comparison</summary>

Use the public resolved GEMV configuration for each shape/bucket; record
waves, steps, rows, tokens and K split. FP8-input and BF16-input pipelines are
separate comparisons. BF16 measurements include the stated quantization work.

| Shape | N | K | M | Config | SG FP8 µs | MORI FP8 µs | SG BF16 µs | MORI BF16 µs | Quantize µs | Reference |
|---|---|---|---|---|---|---|---|---|---|---|
| wq_b / TP4 | 8192 | 1280 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP4 | 8192 | 1280 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP4 | 5120 | 2048 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_a / TP4 | 1280 | 5120 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wkv / TP4 | 512 | 5120 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wqkv_a / TP4 | 1792 | 5120 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP4 | 2048 | 4096 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| shared gate_up / TP4 | 1152 | 5120 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP8 | 4096 | 1280 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP8 | 5120 | 1024 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_a / TP8 | 1024 | 4096 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wq_b / TP1 | 32768 | 1280 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| wo_b / TP1 | 5120 | 8192 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>B — Boundary and rejection cases</summary>

Expand the applicable TP/quantization axes into distinct case IDs. These
are correctness and dispatch boundaries; expected behavior is not a test result.

| ID | Boundary / inputs | Expected contract | Correctness | Dispatch / latency |
|---|---|---|---|---|
| B1 | GEMV M=1,2,3,4,5,7,8,9,15,16,17,31,32; two tuned shapes + a heuristic shape | Correct bucket and masked rows | 待测 | 待测 |
| B2 | GEMV M=33 | Reject above token ceiling | 待测 | 待测 |
| B3 | GEMM M=63,64,65,127,128,129,255,256,257 | Pad before quantization; check true output rows | 待测 | 待测 |
| B4 | wq_b 8192x1280, M=1023,1024,1025,1088 | Check both sides of 140-grid tile switch | 待测 | 待测 |
| B5 | wo_b 5120x2048, M=1535,1536,1537,1600 | Check both sides of 140-grid tile switch | 待测 | 待测 |
| B6 | Grid thresholds 48/64/80/128: nearest reachable M on either side | Separate admission from tile selection | 待测 | 待测 |
| B7 | GEMM K=128,256,384; legal/illegal N; N=1152 candidates | Reject unsupported contracts; measure supported candidates | 待测 | 待测 |
| B8 | TP2/4/8; invalid TP; FNUZ/wrong scale shape/stride | Explicit host validation | 待测 | 待测 |

</details>

<details>
<summary>D — Tile crossover and admission summaries (derived from measured rows)</summary>

| Wide-grid bin | Points | N128 faster | N256 faster | Mean Δ% | Round spread |
|---|---|---|---|---|---|
| 1–16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 17–32 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 33–64 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 65–128 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 129–192 | 待测 | 待测 | 待测 | 待测 | 待测 |
| >192 | 待测 | 待测 | 待测 | 待测 | 待测 |

| Admission grid threshold | Eligible points | Wrong-choice rate | Mean latency penalty | Tie/noise rule |
|---|---|---|---|---|
| 48 | 待测 | 待测 | 待测 | 待测 |
| 64 | 待测 | 待测 | 待测 | 待测 |
| 80 | 待测 | 待测 | 待测 | 待测 |
| 128 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>A — Deduplicated GEMM + all-reduce mode/wire matrix</summary>

Canonical collective configurations:

| ID | Operand quantization | N | K | Default BLOCK_M |
|---|---|---|---|---|
| C1 | blockscale | 7168 | 2048 | 128 |
| C2 | mxfp8 | 5120 | 2048 | 256 |

Each configuration below is measured at M=4096/8192/16384. C1 uses TP8
and C2 uses TP4 for the historical comparison. Add TP2/4/8 correctness coverage
in the boundary/new-option matrices. All rows here use BF16 scatter. GEMM-only
appears once; the FP8 rows shared by earlier sweep groups also appear once.

##### C1: blockscale, N=7168, K=2048, TP=8

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

##### C2: mxfp8, N=5120, K=2048, TP=4

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
<summary>C — Short chunks: exact division and every tail length</summary>

For each C1/C2 and TP2/4/8, M = bands_per_peer * TP * BLOCK_M. The
baseline uses the existing divisor rule. Each candidate uses its matching
counter count. Check byte coverage, no duplicate/missing rows and repeated calls.

##### C1, TP2, BLOCK_M=128

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
|---|---|---|---|---|---|---|
| 1 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 512 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 768 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 1792 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 2560 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 2816 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 3072 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 3328 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | 3584 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### C1, TP4, BLOCK_M=128

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
|---|---|---|---|---|---|---|
| 1 | 512 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 1536 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 2560 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 3584 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 5120 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 5632 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 6144 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 6656 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | 7168 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### C1, TP8, BLOCK_M=128

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
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

##### C2, TP2, BLOCK_M=256

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
|---|---|---|---|---|---|---|
| 1 | 512 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 1536 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 2560 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 3584 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 5120 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 5632 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 6144 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 6656 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | 7168 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 8192 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### C2, TP4, BLOCK_M=256

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
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

##### C2, TP8, BLOCK_M=256

| Bands/peer | M | Baseline µs | chunk_bands=1 µs | chunk_bands=2 µs | chunk_bands=4 µs | Coverage / reference |
|---|---|---|---|---|---|---|
| 1 | 2048 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 4096 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 6144 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | 10240 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 14336 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 16384 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | 20480 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | 22528 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | 24576 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | 26624 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | 28672 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 32768 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>P — Padding and owner/chunk intersections</summary>

Compare identical true rows with independently padded buffers. For a
candidate with fewer than BLOCK_M rows per peer, verify rejection/fallback
rather than launching relaxed rows. Include repeated input changes.

##### C1, TP2, BLOCK_M=128

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 255 | 256 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| 256 | 256 | 256 | 256 | 待测 | 待测 | 待测 | 待测 |
| 257 | 512 | 512 | 320 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 4352 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 8448 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C1, TP4, BLOCK_M=128

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 511 | 512 | 512 | 512 | 待测 | 待测 | 待测 | 待测 |
| 512 | 512 | 512 | 512 | 待测 | 待测 | 待测 | 待测 |
| 513 | 1024 | 768 | 576 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 4608 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 8704 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C1, TP8, BLOCK_M=128

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 1023 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1024 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1025 | 2048 | 1280 | 1088 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 5120 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 9216 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C2, TP2, BLOCK_M=256

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 511 | 512 | 512 | 512 | 待测 | 待测 | 待测 | 待测 |
| 512 | 512 | 512 | 512 | 待测 | 待测 | 待测 | 待测 |
| 513 | 1024 | 768 | 576 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 4608 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 8704 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C2, TP4, BLOCK_M=256

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 1023 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1024 | 1024 | 1024 | 1024 | 待测 | 待测 | 待测 | 待测 |
| 1025 | 2048 | 1280 | 1088 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 5120 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 9216 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

##### C2, TP8, BLOCK_M=256

| True M | Baseline M_pad | Align256 M_pad | Align64 M_pad | Baseline µs | Align256 µs | Align64 µs | Intersection / reference |
|---|---|---|---|---|---|---|---|
| 2047 | 2048 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| 2048 | 2048 | 2048 | 2048 | 待测 | 待测 | 待测 | 待测 |
| 2049 | 4096 | 2304 | 2112 | 待测 | 待测 | 待测 | 待测 |
| 4200 | 6144 | 4352 | 4224 | 待测 | 待测 | 待测 | 待测 |
| 8200 | 10240 | 8448 | 8256 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>V2 — Compact reduction and 5/10-wave specialization</summary>

N=8192, K=1280, MXFP8; identical raw operands and one shared weight ring.
Only waves/valid-token reduction change. Record the remaining configuration.

| M | Default µs | Compact µs | 5-wave µs | 10-wave µs | Reference / equality |
|---|---|---|---|---|---|
| 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 17 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 32 | 待测 | 待测 | 待测 | 待测 | 待测 |

| Variant | VGPRs | LDS bytes | Scratch bytes | Profiler / compiler record |
|---|---|---|---|---|
| M4 default | 待测 | 待测 | 待测 | 待测 |
| M4 compact | 待测 | 待测 | 待测 | 待测 |
| M4 5-wave | 待测 | 待测 | 待测 | 待测 |
| M32 default | 待测 | 待测 | 待测 | 待测 |
| M32 compact | 待测 | 待测 | 待测 | 待测 |
| M32 10-wave | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>W — Workspace size, allocations and serial lifecycle</summary>

Use C1/TP8/M=16384 for the historical memory comparison. Record window
requirements separately from actual allocator peaks; VMM reservation is not the
same as physically backed memory. Extend correctness to supported C1/C2, TP2/4/8.

| Layout | Window MiB/rank | Torch peak MiB | External physical MiB | Allocation count | Init µs | Steady µs | Reference |
|---|---|---|---|---|---|---|---|
| baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| no unused tmp | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| compact receive slots | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| input/output alias | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

| Lifecycle case | Layout/shape coverage | Result |
|---|---|---|
| Large → small → large | 4096 → 16384 → 8192 → 4096; baseline and reuse layouts | 待测 |
| Changed operands | Different rank values and alternating inputs | 待测 |
| Multiple captured graphs | Alternate warmed shapes while retaining storage/launchers | 待测 |
| Capacity and offsets | Active M below m_max; distinct counter sets and region bounds | 待测 |
| Synchronized close | Correct release/rollback and documented ownership | 待测 |

</details>

<details>
<summary>F — FP8 scatter/gather performance and staged numerics</summary>

Start with C2/TP4; extend valid aligned configurations to TP2/4/8. Use
separate reduction and the matching scale tile. Reject unsupported combinations
before execution. Record mathematical and actual-wire errors independently.

| TP | M | Gather | BF16 scatter µs | FP8 scatter µs | FP32 relL2 | Max abs / p99 error | Wire reference | Saturation / finite |
|---|---|---|---|---|---|---|---|---|
| 2 | 4096 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 4096 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 8192 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 8192 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 16384 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 16384 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 4096 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 4096 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 8192 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 8192 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 16384 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 16384 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 4096 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 4096 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 8192 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 8192 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 16384 | bf16 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 16384 | fp8/lsa | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

| Input / protocol case | Required check | Result |
|---|---|---|
| All zero | Absolute error; finite scales; no positive-relL2 requirement | 待测 |
| Per-rank constants / one-rank impulse | Correct owners, contributors and destination ranges | 待测 |
| Positive/negative cancellation | Wire reference plus absolute error near zero | 待测 |
| Exactly representable / rounding midpoints | Payload and scale agreement with stated rounding | 待测 |
| Supported small/large magnitudes and outliers | Error distribution and documented saturation behavior | 待测 |
| Both FP8 legs | Separate GEMM, scatter reduction and gather error contributions | 待测 |
| Illegal option combinations | Reject relaxed/compact/direct-LSA or incompatible fusion | 待测 |

</details>

<details>
<summary>G — Decode GEMV + AR fusion controls</summary>

N=5120, K=2048; compare staged GEMV + LSA AR against both prototypes.
Check supported resident-grid constraints before timing, and validate epoch/input
reuse across calls. M=16/32 extend the earlier M=1/4/8 study.

| TP | M | Staged µs | Flags µs | Tagged µs | Reference | Replay / changed input |
|---|---|---|---|---|---|---|
| 2 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 1 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 4 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 8 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 16 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>Q — Grouped reduce + quantize</summary>

C2/TP4/M=16384, FP8 LSA gather. Keep the negative controls; fewer launches
are not an acceptance criterion.

| Quantization | Reduce µs | Quantize µs | Gather/pull µs | Whole call µs | relL2 | Reference |
|---|---|---|---|---|---|---|
| separate per-row | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| fused group128 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| fused group256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| fused group512 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>T — Targeted tuning and profiling matrices</summary>

Run these when the corresponding implementation changes. Kernel-only
and whole-call columns are separate; never fill one from the other.

##### Pull grid: C1/TP8/M=16384, FP8 LSA gather

| Blocks | Whole call µs | Pull kernel µs | Reference |
|---|---|---|---|
| 16 | 待测 | 待测 | 待测 |
| 24 | 待测 | 待测 | 待测 |
| 32 | 待测 | 待测 | 待测 |
| 48 | 待测 | 待测 | 待测 |
| 64 | 待测 | 待测 | 待测 |
| 80 | 待测 | 待测 | 待测 |
| 128 | 待测 | 待测 | 待测 |
| 256 | 待测 | 待测 | 待测 |
| 512 | 待测 | 待测 | 待测 |

##### C store: TP8, M=4096, N=7168, K=1024, ptpc

| Variant | Kernel µs | Whole call µs | Store instructions / cycles | VGPR / scratch | Reference |
|---|---|---|---|---|---|
| original store | 待测 | 待测 | 待测 | 待测 | 待测 |
| swap AB | 待测 | 待测 | 待测 | 待测 | 待测 |
| swap + permlane | 待测 | 待测 | 待测 | 待测 | 待测 |
| swap + permlane + transpose | 待测 | 待测 | 待测 | 待测 | 待测 |
| previous + hoist scales | 待测 | 待测 | 待测 | 待测 | 待测 |

##### Grid remainder: TP8, M=4096, K=1024, ptpc

| N | Tiles / remainder | Kernel µs | Whole call µs | Reference |
|---|---|---|---|---|
| 4096 | 待测 | 待测 | 待测 | 待测 |
| 4352 | 待测 | 待测 | 待测 | 待测 |
| 4608 | 待测 | 待测 | 待测 | 待测 |
| 5120 | 待测 | 待测 | 待测 | 待测 |
| 6144 | 待测 | 待测 | 待测 | 待测 |
| 7168 | 待测 | 待测 | 待测 | 待测 |

##### Aligned chunks: TP8, M=16384, N=7168, K=2048, ptpc

| Chunks | BLOCK_M | GEMM µs | Drain µs | Reduce µs | Gather µs | Whole call µs | Reference |
|---|---|---|---|---|---|---|---|
| 1 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 128 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

##### Reduce-triggered gather: TP8, M=16384, N=7168, K=2048, BF16 wire, ptpc

| Publish | Bands | Whole call µs | Reference |
|---|---|---|---|
| writethrough | 1 | 待测 | 待测 |
| writethrough | 4 | 待测 | 待测 |
| writethrough | 8 | 待测 | 待测 |
| writethrough | 16 | 待测 | 待测 |
| writethrough | 32 | 待测 | 待测 |
| fence | 1 | 待测 | 待测 |
| fence | 4 | 待测 | 待测 |
| fence | 8 | 待测 | 待测 |
| fence | 16 | 待测 | 待测 |
| fence | 32 | 待测 | 待测 |

##### MXFP8 scale layout: N=5120, K=2048, identical operands

| M | Layout | Kernel hot µs | Kernel cold µs | VMEM/cache counters | Reference / equality |
|---|---|---|---|---|---|
| 4096 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| 4096 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| 4096 | row-major | 待测 | 待测 | 待测 | 待测 |
| 8192 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| 8192 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| 8192 | row-major | 待测 | 待测 | 待测 | 待测 |
| 16384 | packed K-major | 待测 | 待测 | 待测 | 待测 |
| 16384 | unpacked K-major | 待测 | 待测 | 待测 | 待测 |
| 16384 | row-major | 待测 | 待测 | 待测 | 待测 |

##### Window geometry: TP8, M=16384, N=7168, K=2048, ptpc

| Mode | Window median µs | CachedWindow median µs | Paired spread | Reference |
|---|---|---|---|---|
| split-lsa | 待测 | 待测 | 待测 | 待测 |
| fused-sdma | 待测 | 待测 | 待测 | 待测 |

##### FP8 scale granularity: fixed standard-normal payload, 2048x7168

| Group width | relL2 | Max abs / p99 error | Scale bytes / payload | Seed / dtype |
|---|---|---|---|---|
| 7168 | 待测 | 待测 | 待测 | 待测 |
| 512 | 待测 | 待测 | 待测 | 待测 |
| 256 | 待测 | 待测 | 待测 | 待测 |
| 128 | 待测 | 待测 | 待测 | 待测 |
| 32 | 待测 | 待测 | 待测 | 待测 |

</details>

<details>
<summary>S — Specialization, graph ownership and option-composition regression</summary>

| Case | Required ordering / evidence | Result |
|---|---|---|
| Minimum K | K=256 after a fresh process and after loading cache | 待测 |
| Factory specialization | Configuration A → B → A; distinct dtypes, phases and shapes | 待测 |
| Changed inputs under replay | Each graph must observe the new inputs | 待测 |
| Launcher ownership | Keep modules, launchers, tensors and graphs alive through completion | 待测 |
| Counter/epoch stability | Repeated calls plus process timeout/exit status | 待测 |
| Supported composition | Combine newly enabled options only within their declared contracts | 待测 |
| Unsupported composition | Host-side rejection; no silent fallback into an unimplemented branch | 待测 |

</details>

## Historical records

These are frozen records, including negative results and earlier measurement
limitations. They retain their own scope, baseline and statistical definitions.
They do not populate the new `待测` cells, establish a new default, or imply that
a model-level test was run during this reorganization. Raw experiments and logs
remain outside the MORI repository.

Some historical descriptions refer to intermediate implementations. The current
API and validation contract above take precedence. Live progress and setup/retry
transcripts are kept in the external run archive rather than the main document.

<details>
<summary>Historical scope, usage and FP8-wire measurements</summary>

### `mori.ops.gemm_ar` -- an fp8 GEMM, with or without the all-reduce fused in

Three operators over one mxfp8/blockscale GEMM kernel, for a tensor-parallel
attention block:

| op | for | what it does |
|---|---|---|
| `GemmAllReduceOp` (`op.py`) | a `RowParallelLinear` at prefill M | the GEMM with the all-reduce's scatter fused into its epilogue |
| `Mxfp8GemmOp` (`gemm.py`) | any mxfp8 linear at prefill M | the same GEMM, nothing fused, no communicator |
| `Mxfp8GemvOp` (`gemv.py`) | any mxfp8 linear at decode M (<= 32) | a skinny GEMM built for a handful of tokens |

Built for DeepSeek-V4-Pro's `wo_b` and measured there: **1146.2us against the
model's 1419.5us, -19.4%** at `[16384, 7168] K=2048` on 8x MI355X. On
DeepSeek-V4.1-Flash the same fusing is worth -25% -- but there **most of the win
is the multiply, not the overlap**, which is why the GEMM is also shipped on its
own. See [Measured results](#measured-results).

`kernels_fused.py`, `kernels_sdma.py`, `kernels_lsa.py` and `kernels_gemv.py`
hold the kernels, `layout.py` every window offset and count, and
`_gemm_a8w8_8wave.py` / `_shuffle.py` the two pieces vendored from aiter so mori
does not depend on it.

How to *run* the benchmarks, and a one-table summary of what they say, are in
[`docs/MORI-GEMM-AR-BENCHMARK.md`](../../../../docs/MORI-GEMM-AR-BENCHMARK.md).
Every full table lives here, next to the code it is about.

#### What it does

One GEMM kernel, compiled two ways. With the epilogue's tail switched on it
pushes each destination's slice as it is produced, which is `GemmAllReduceOp`;
with it off it is a plain GEMM that returns an ordinary tensor, which is
`Mxfp8GemmOp`. `Mxfp8GemvOp` is a *different* kernel for the token counts where
a 256-row tile has nothing to fill it.

The rest of this section, and all of [How it works](#how-it-works), is about the
fused one.

The target is DeepSeek-V4-Pro's `wo_b`: a RowParallelLinear whose per-rank GEMM
is `[M,1024] x [7168,1024]` fp8 -> `[M,7168]` bf16, immediately all-reduced.
Split, that is `gemm(); all_reduce()`. Fused, the GEMM's C lands directly in a
registered cco window and each destination's slice is pushed by the copy engine
as soon as its last tile is written, so the reduce-scatter transfer overlaps the
rest of the GEMM instead of following it.

Only the *scatter* half of the all-reduce is absorbed. The reduce and all-gather
phases still run as their own kernels, reused verbatim from `ar.kernels_sdma`
(`build_sdma_phases`), with `scatter` swapped for its drain-only twin.

**Nothing here is a production kernel.** aiter's tuned CSV picks a ck/asm/cktile
backend for (N=7168, K=1024), not this one, so the fused-vs-split comparison is
internally valid but is not a claim about DSV4 as shipped.

#### Using it

##### Fused GEMM + all-reduce

Needs a mori built with `BUILD_CCO_SDMA=ON` and `MORI_ENABLE_SDMA=1` in the
environment. Setting the variable alone is not enough: a host library built
without the flag has no queues, so every put silently does nothing and the
all-reduce quietly produces zeros. (The two standalone ops below need neither --
they move no data between ranks.)

```python
import torch, torch.distributed as dist
from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

# The window comes out of the communicator's VMM reservation, so size that
# first -- the op cannot grow it later.
N, K, M_MAX = 7168, 2048, 16384
vmm = 2 * GemmAllReduceOp.window_bytes_for(world, m_max=M_MAX, n=N) + (64 << 20)

uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
dist.broadcast_object_list(uid, src=0)

# init takes the UniqueId wrapper, not the bytes that survive a broadcast.
with Communicator.init(
    world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=vmm
) as comm:
    with GemmAllReduceOp(comm, n=N, k=K, m_max=M_MAX) as op:
        b_shuffled = preshuffle_b(b_fp8)          # [N, K], once per weight

        m_pad = op.padded_m(x.shape[0])           # instance method: M only
        a_fp8, a_scale = quantize_per_1x128(op.pad_rows(x, m_pad))
        # The result aliases the window, which the `with` frees on exit --
        # so clone inside it, after the work it was issued behind has run.
        out = op(a_fp8, b_shuffled, a_scale, b_scale)[: x.shape[0]]
        torch.cuda.current_stream().synchronize()
        out = out.clone()
```

**Pad before quantising.** A padded row produces a zero output row, which the
caller slices off; quantising first and padding after would also have to extend
the scales, and in the layout below that is not a row append.

**Scales.** `a_scale` is the A 1x128 block scale as fp32. The kernel reads
element `(row, kb)` at `kb * M + row`, so three spellings are accepted because
each already *is* that order: flat 1-D, `[K/128, M]`, and `[M, K/128]`
**column-major** (stride `(1, M)`), which is transposed for free.

`[M, K/128]` **row-major is rejected** with a `ValueError`. It is not that the
op cannot transpose it -- it is that the two `[M, K/128]` tensors are
indistinguishable by shape, and they need opposite treatment: the column-major
one must be read flat, the row-major one must be transposed. Guessing gets one
of them silently wrong (relL2 0.26 either way). Pass `.reshape(-1)` or `.t()`
yourself to say which you have. `aiter_per1x128_quant(transpose_scale=True)`
returns the column-major one, so `.reshape(-1)` is right for it.

`b_scale` is `[N/128, K/128]` fp32 row-major.

**Shapes.** `supports(m, n, k, world_size)` answers whether a shape is
expressible. `K >= 256` and a multiple of 128; `N` a multiple of 256, and of
1024 if the gather is fp8; `world_size` in [2, 8] and dividing 512.

**Collective contract.** Every rank must construct the op and call it the same
number of times in the same order -- the phases carry device-side barriers.
Work is issued on the current stream and completes asynchronously, so the usual
stream rules apply to the result.

**The result aliases the window** and is overwritten by the next call; clone it
to keep it. `pad_rows` returns a reused buffer with the same caveat. One
instance is not usable concurrently.

**Distinct M values.** Each M compiles its own kernel and takes its own counter
set, capped by `max_shapes`. That defaults to *every* legal M under `m_max`
(`m_max / (world_size * block_m)`), so the cap is not something a caller
normally meets; a ninth distinct shape used to raise. Warm each M up once before
capturing a graph.

**Cleanup.** `close()` (or the `with` above) releases the window and its memory
-- at the model shape about 700 MiB per rank. The communicator holds a strong
reference to each, so dropping the op is not enough. The dev-comm is
deliberately *not* destroyed here: its SDMA queues are communicator-scoped, and
tearing them down ahead of the communicator aborts the process. They go with the
communicator. Synchronise first if anything may still be in flight.

**dtype.** A and B must be `torch.float8_e4m3fn`. `float8_e4m3fnuz` is the same
width but a different exponent bias, and the MMA atom implements OCP's, so it is
rejected rather than silently returning a result 4x too large.

##### The GEMM without a collective

Neither of these takes a communicator, a window or a rank: the output is an
ordinary tensor. They exist because at DeepSeek-V4.1-Flash's shapes the GEMM is
worth more than the fusing, and because a `ColumnParallelLinear` has no
all-reduce to fuse with at all.

```python
from mori.ops.gemm_ar import (
    Mxfp8GemmOp, Mxfp8GemvOp, preshuffle_a_scale, preshuffle_b,
    supports_gemm, supports_gemv,
)

# Once per weight. w_exps is the checkpoint's [N/32, K/32] ue8m0 bytes.
b_shuffled = preshuffle_b(w_fp8)                            # [N, K]
b_scale = w_exps.t().contiguous().to(torch.int32).reshape(-1)   # K-block major

# Prefill. M only has to be a multiple of 64 -- the grid is ceildiv(M, BLOCK_M)
# and the tail block masks its stores.
assert supports_gemm(N, K)
op = Mxfp8GemmOp(n=N, k=K)
m_pad = op.padded_m(x.shape[0])
a_fp8, a_exps = quantize_mxfp8(op.pad_rows(x, m_pad))   # pad before quantising
out = op(a_fp8, b_shuffled, preshuffle_a_scale(a_exps), b_scale)[: x.shape[0]]

# Decode. No padding and none needed; M is a runtime argument, and both scales
# go in as the checkpoint stores them -- w_exps itself, not b_scale.
assert supports_gemv(N, K)
gemv = Mxfp8GemvOp(n=N, k=K)
out = gemv(x_fp8, b_shuffled, x_exps, w_exps)
```

**They share the weight and not the scales.** `preshuffle_b` output serves both,
so a server shuffles once. But `Mxfp8GemmOp` wants the A scale through
`preshuffle_a_scale` -- K-block major, four M tiles to a dword -- while
`Mxfp8GemvOp` takes **both scales exactly as the checkpoint stores them**,
row-major ue8m0 bytes, `[M, K/32]` and `[N/32, K/32]`. That is not an
inconsistency: the GEMM's sixteen lanes want sixteen *rows* of one K block and
have to be coalesced, where the GEMV's are sixteen *tokens*, M is at most 32, and
the whole A scale is under a kilobyte. See
[the layout results](#the-mxfp8-gemm-on-its-own).

**Shapes.** `supports_gemm(n, k)` and `supports_gemv(n, k)` answer whether a
shape is expressible -- K a multiple of 128, N a multiple of 256 for the GEMM
and of 32 for the GEMV. M is deliberately not an argument to either: any M is
*servable*, and whether it is *profitable* is the caller's call.

**Which M is profitable is decided by the grid, not by M.** The GEMM's tile is
256x256, so a caller should gate on `ceildiv(M, 256) * (N / 256)` and hand over
at about 80 workgroups; a floor on M alone is only ever right for the N it was
fitted to. The measurement behind that, and what it costs to get it wrong, is
[here](#the-mxfp8-gemm-against-sglang-across-every-shape).

**The GEMV's ceiling is 32 tokens**, because a token is an MFMA row and two tiles
of them is where the register file runs out. It picks a compile-time
configuration per M bucket from a tuned table in `gemv.py`; an untuned shape gets
a heuristic rather than an error. Between 32 and the GEMM's floor neither wins,
and the caller should use whatever it had.

##### fp8 on the wire

`gather_dtype="fp8"` sends the all-gather leg as e4m3 with one fp32 scale per
row, halving its bytes. That leg is ~40% of a fused layer and already runs at
the xGMI ceiling (470 GB/s over 7 links), so halving the bytes halves the time.

Measured, fused-sdma at `[16384, 7168]` K=2048 on 8x MI355X:

| gather wire | us | relL2 |
|---|---:|---:|
| bf16 | 1151.5 | 2.35e-3 |
| fp8 | **1033.9** (-10.2%) | **2.49e-2** |

###### Who moves the fp8 gather

`gather_transport="lsa"` (the default for fp8) pulls each peer's slice over
xGMI into registers and widens it on the way to memory. `"sdma"` pushes with
the copy engines and widens in a second kernel -- a copy engine has no ALU, so
for SDMA those cannot be one step, and the second kernel has to read the landed
fp8 back out of local HBM (98 MiB a layer).

| gather | us |
|---|---:|
| bf16 / sdma | 1150.7 |
| fp8 / sdma | 1018.9 |
| **fp8 / lsa** | **957.3** |

The pull's grid is the whole story and is not obvious: these are xGMI reads, so
the grid throttles outstanding remote requests rather than covering HBM latency,
and it wants roughly a tenth of what the local conversion kernels want.

| blocks | 16 | 24 | 32 | 48 | 64 | 80 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| us | 1261 | 1091 | 1006 | 962 | **959** | 963 | 1018 | 1138 | 1184 |

The first version launched 512 -- the quantize grid -- and lost to SDMA by 17%.

##### Benchmarks and tests

```bash
# numerics
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  pytest tests/python/cco/test_gemm_ar.py tests/python/cco/test_flydsl_ar.py \
         tests/python/cco/test_gemm_ar_op.py

# the full mode comparison at the model's shape
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo python -m torch.distributed.run \
  --standalone --nproc_per_node=8 benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py \
  --mode fused-sdma --quant blockscale -m 16384 -n 7168 -k 2048
```

Requires a mori built with `BUILD_CCO_SDMA=ON`. Setting `MORI_ENABLE_SDMA` in
the environment only rebuilds the device bitcode -- a host library built without
the flag has no queues, so every put silently does nothing and the all-reduce
quietly produces zeros.

Every flag, every sweep driver, and how to drive the whole thing from a server
are in
[`docs/MORI-GEMM-AR-BENCHMARK.md`](../../../../docs/MORI-GEMM-AR-BENCHMARK.md).
Read [Measurement traps](#measurement-traps) before trusting a small-M number
from a harness of your own.

</details>

<details>
<summary>Historical operator matrices, crossover studies and measurement caveats</summary>

#### Measured results

MI355X (gfx950), one node, otherwise idle, occupancy recorded before and after
each run. Kernel timings are the median over repeated iterations, maximum over
ranks. Run-to-run spread is about **2%**, so differences below that are not
differences.

Two models are covered and **they do not reach the same conclusion**, so they
are kept apart rather than averaged:

| | DeepSeek-V4-Pro | DeepSeek-V4.1-Flash |
|---|---|---|
| `wo_b` per rank | `[M, 7168]` K=2048, TP8 | `[M, 5120]` K=2048, TP4 |
| quantisation | 1x128 / 128x128 fp8 block scale | **32-wide ue8m0 (mxfp8)** |
| `--quant` | `blockscale` | `mxfp8` |
| fusing, best wire | **-25%** at M=16384 | **-25%** at M=16384 |
| what dominates it | the overlap | the fp8 wire and the GEMM |

##### The mxfp8 GEMM, on its own

CDNA4's `v_mfma_scale_f32_16x16x128_f8f6f4` takes e8m0 scales as instruction
operands, so a 32-wide ue8m0 GEMM needs no dequantisation arithmetic at all --
unlike `_BlockScaleK`'s promote/rescale chain, which exists precisely because a
128-wide block cannot be expressed that way. Two layout results came out of
making it fast, both about *addresses* rather than bytes.

**The scales must be K-block major.** Coalescing happens per quarter-wave, which
is exactly the sixteen lanes of one MFMA block group, and those sixteen want
sixteen consecutive rows of one K block. K-block major puts them at consecutive
addresses -- one access per instruction. The quantiser's own `[M, K/32]` row
major puts them `K/32` bytes apart:

| | K-block major | row major |
|---|---:|---:|
| `SQ_INSTS_VMEM` | 896,000 | 911,360 |
| `TCP_TOTAL_CACHE_ACCESSES` | 8,622,080 | **27,729,920** |
| GEMM, M=4096..16384 | — | **+50% to +67%** |

The same instruction count and 3.2x the cache accesses. Predicted +19,660,800
accesses (4 per instruction becoming 64), measured +19,107,840. Stripping the
byte-select arithmetic off the row-major path moved 117.4 -> 110.9us, so 41 of
the 47us is addresses. The data is 16 KB and fully L1-resident; the currency is
requests, not bytes.

**Four M tiles pack into one scale dword.** A lane's four A tiles differ only by
sixteen rows and share the K block, and `opsel_b` on the scaled MFMA is an
atom-time attribute naming which byte of the 32-bit scale operand to read. So
one load serves all four and the byte select is free:

| | unpacked | packed |
|---|---:|---:|
| `SQ_INSTS_VMEM` | 896,000 | **634,880** (-29.1%) |
| `SQ_INSTS_VALU` | 2,252,800 | 2,268,160 (+0.7%) |
| GEMM | — | **-5% to -7.5%** |

VALU flat is the check that matters: it confirms `opsel` really is free rather
than degrading to a shift and mask. CK does the same thing in
`preShuffleScaleBuffer_gfx950`; this packs all four M tiles rather than its
`MNXdlPack=2`, which fits opsel's two bits exactly and needs no cross-K-step
state, so the mainloop is untouched.

`preshuffle_a_scale` is that layout, and a quantiser can write it directly --
the permutation stays inside the 64 bytes one program already owns, so it is
address arithmetic rather than traffic, and measures slightly *faster* than the
stock kernel.

**Against the alternatives**, bf16 in and bf16 out, the whole pipeline including
quantisation, M=16384:

| route | us |
|---|---:|
| bf16 (`fake_quant` + hipBLASLt bf16 GEMM) | 281.3 |
| SGLang's own mxfp8 (`tl.dot_scaled`) | 274.9 |
| **mori mxfp8** | **196.0** |

**-30.3%**, before any fusion. That is most of what the integration is worth,
and it is not the overlap.

> Those three came off a harness that predates `timing.py`: hot only, one call
> per graph. At M=16384 each route is 200-280us, so a 13.4us floor is ~5% on all
> three columns and the ratio is close to right -- `bench_gemm.py --scope linear`, on the
> fixed timer, puts the same comparison at -27%. The split between the two
> baseline routes is not reproducible from what is checked in; the mori-vs-best
> -baseline column is, and is the one to quote.

##### The mxfp8 GEMM against SGLang, across every shape

The table above is two shapes and one kernel. This is every fp8 linear the
checkpoint has -- read off `layers.N.attn.*` and `layers.N.ffn.shared_experts.*`
and split by the parallelism each is declared with in SGLang's
`models/deepseek_v4.py`, at three TP degrees -- against **each of mori's tiles
separately**, rather than against whichever one its own dispatch picks. 120
points, `sweep.py gemm-full`, cold.

Baseline is `mxfp8_native_blockscaled_linear`, identical operands. Every cell is
the whole pipeline from bf16 with quantisation included, because that is what
the baseline does. Negative is mori faster.

> An earlier revision of these tables cooled only the fp8 weight. SGLang's route
> is `hipblaslt_bf16` on most of the large-M points and reads the *dequantised
> bf16* weight, which stayed pinned -- so the baseline was hot where mori was
> cold, and mori's margin was **understated** by up to 28 percentage points.
> `bench_gemm.py` now rotates every weight the chosen route may read, and the
> `gemv` and `dot_scaled` rows, which never read that tensor, are unchanged
> within a point either way.

**mori's GEMM on the 256x256 tile:**

| layer | N x K | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | +113% | +88% | +3% | **-28%** | **-30%** | **-32%** | **-43%** |
| `wo_b` (TP4) | 5120 x 2048 | +126% | +52% | **-5%** | **-34%** | **-26%** | **-31%** | **-27%** |
| `wq_a` (TP4) | 1280 x 5120 | +316% | +221% | +87% | +38% | +5% | **-19%** | **-15%** |
| `wkv` (TP4) | 512 x 5120 | +438% | +309% | +140% | +100% | +36% | +2% | **-21%** |
| `wqkv_a` (TP4) | 1792 x 5120 | +302% | +175% | +62% | +12% | **-16%** | **-28%** | **-24%** |
| `wo_a` (TP4) | 2048 x 4096 | +263% | +155% | +42% | +13% | **-41%** | **-25%** | **-22%** |
| `shared gate_up` (TP4) | 1152 x 5120 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `wq_b` (TP8) | 4096 x 1280 | +131% | +85% | +9% | **-14%** | **-22%** | **-25%** | **-27%** |
| `wo_b` (TP8) | 5120 x 1024 | +85% | +55% | **-8%** | **-28%** | **-23%** | **-26%** | **-26%** |
| `wo_a` (TP8) | 1024 x 4096 | +361% | +253% | +105% | +30% | +4% | **-39%** | **-31%** |
| `wq_b` (TP1) | 32768 x 1280 | +14% | **-22%** | **-34%** | **-34%** | **-31%** | **-23%** | **-17%** |
| `wo_b` (TP1) | 5120 x 8192 | +234% | +140% | +19% | **-19%** | **-25%** | **-31%** | **-21%** |

**mori's GEMM on the 256x128 tile:**

| layer | N x K | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | +85% | +77% | -1% | **-2%** | **-3%** | -1% | **-17%** |
| `wo_b` (TP4) | 5120 x 2048 | +91% | +41% | **-10%** | **-6%** | **-10%** | **-12%** | +0% |
| `wq_a` (TP4) | 1280 x 5120 | +243% | +178% | +60% | +20% | **-8%** | +11% | -1% |
| `wkv` (TP4) | 512 x 5120 | +349% | +256% | +111% | +75% | +20% | **-7%** | **-25%** |
| `wqkv_a` (TP4) | 1792 x 5120 | +238% | +141% | +41% | -2% | **-26%** | **-4%** | +1% |
| `wo_a` (TP4) | 2048 x 4096 | +201% | +123% | +25% | -0% | **-47%** | **-4%** | -1% |
| `shared gate_up` (TP4) | 1152 x 5120 | +243% | +185% | +71% | +30% | **-3%** | +12% | +1% |
| `wq_b` (TP8) | 4096 x 1280 | +93% | +70% | +3% | **-19%** | +4% | +2% | -1% |
| `wo_b` (TP8) | 5120 x 1024 | +58% | +49% | **-10%** | +2% | **-5%** | **-6%** | +2% |
| `wo_a` (TP8) | 1024 x 4096 | +282% | +207% | +87% | +20% | -1% | **-40%** | **-9%** |
| `wq_b` (TP1) | 32768 x 1280 | +0% | **-25%** | **-5%** | +2% | +8% | +33% | +47% |
| `wo_b` (TP1) | 5120 x 8192 | +175% | +102% | -0% | +15% | **-11%** | **-17%** | +19% |

###### Which tile, and where they cross

Comparing each tile against the *baseline* is the wrong way to read those two
tables -- it mixes in how fast the baseline happens to be on each shape.
Comparing them against **each other** is unambiguous, and the separation is
total: 110 points where both were measured, and not one on the wrong side of a
single threshold.

| wide-tile grid | points | 256 faster | 128 faster | mean 256/128 - 1 |
|---|---|---|---|---|
| 1-16 | 33 | 0 | 33 | +16.9% |
| 17-32 | 25 | 0 | 25 | +14.3% |
| 33-64 | 6 | 0 | 6 | +10.6% |
| 65-128 | 15 | 0 | 15 | +8.9% |
| 129-192 | 4 | 4 | 0 | -28.9% |
| >192 | 27 | 27 | 0 | -26.4% |

The narrow tile is worth a flat 9-17% below the crossover and costs 26-29% above
it, and the crossover is a step rather than a slope. `_WIDE_TILE_MIN_GRID = 140`
sits inside the gap; sweeping the threshold over this data gives **zero regret**
for anything in 129-192 and gets worse either side. This is the third data set
the constant has been re-derived on, including two with measurement bugs in
them, and it has not moved.

###### Which M is worth serving at all

Taking the better of the two tiles at each point:

| wide-tile grid | points | best mori vs baseline |
|---|---|---|
| 1-32 | 65 | +246.7% |
| 33-64 | 7 | -0.7% |
| 65-128 | 16 | -10.2% |
| 129-256 | 10 | -26.6% |
| >256 | 22 | -26.4% |

Scoring gate thresholds over the 77 points the gate actually decides -- M above
the GEMV's 32 tokens, and a shape `supports_gemm` accepts -- by the percentage
each gets wrong, losses served plus wins declined:

| gate | served and slower | wins forfeited | total |
|---|---|---|---|
| `grid >= 48` | 2.9% | 0.0% | **2.9%** |
| `grid >= 64` | 2.9% | 0.0% | **2.9%** |
| `grid >= 80` | 0.3% | 7.0% | 7.3% |
| `grid >= 128` | 0.3% | 60.7% | 60.9% |

**64, where an earlier revision of this said 80.** The move is the cold-baseline
fix: with SGLang measured as cold as mori, it is slower on the `hipblaslt_bf16`
rows, and mori starts winning at a smaller grid than it appeared to. 48 and 64
score identically because no measured point falls between them, so 64 is the
boundary of the evidence rather than a fitted optimum -- it is where the bin
below it ends.

A floor on M alone is a different matter and is wrong by an order of magnitude.
`M >= 1280`, read off `wq_b` and `wo_b`, served `wkv` at M=2048 for +100% and
`wq_a` for +38%.

###### `shared gate_up`, the one shape only the narrow tile can express

N=1152 is 4.5 tiles of 256 but nine of 128, so the wide tile cannot be built for
it at all and `supports_gemm` refuses the shape. Measured on the narrow tile
anyway, the answer is that it is not worth reaching: **+243% at M=64, +71% at
M=1024, +30% at M=2048, and never better than -3%** -- one point, at M=4096,
inside run-to-run noise. The GEMM's grid there is `ceildiv(M,256) * 4.5`, which
never leaves the starved region for any M a server produces. Relaxing
`supports_gemm` to accept it would buy nothing.

`bench_gemm.py` records this as `supported: false` rather than as a failure, so
a sweep over known-good shapes still exits zero and a real break is not buried
in ten expected declines.

The shared expert's *down* projection, `5120 x 576` at TP4, is absent from every
table above because **SGLang refuses it too** -- `native_route_supports` rejects
a K that is not a multiple of 128 -- so there is nothing to compare against and
no gate change on either side can reach it.

The routed experts are absent throughout for a different reason: they are fp4
(`expert_dtype` in the checkpoint's `quantization_config`), so they never take
an mxfp8 path at all.

###### The GEMV, and what its margin actually depends on

Against SGLang's own `mxfp8_gemv` on the **same fp8 bytes** -- no quantisation on
either side, which is the only form mori's is served in:

| layer | N x K | M=1 | M=8 | M=32 |
|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | **-4.7%** | **-2.7%** | **-4.9%** |
| `wo_b` (TP4) | 5120 x 2048 | **-7.6%** | **-6.0%** | **-7.8%** |
| `wo_b` (TP1) | 5120 x 8192 | **-5.3%** | **-5.9%** | **-6.5%** |
| `wo_b` (TP8) | 5120 x 1024 | **-4.3%** | +0.2% | +0.7% |
| `wq_a` (TP4) | 1280 x 5120 | -0.6% | -1.8% | +1.1% |
| `wkv` (TP4) | 512 x 5120 | -0.2% | -0.1% | +1.0% |
| `wqkv_a` (TP4) | 1792 x 5120 | +1.0% | -0.5% | +2.1% |
| `shared gate_up` (TP4) | 1152 x 5120 | -0.6% | -0.8% | +1.0% |
| `wq_b` (TP1) | 32768 x 1280 | +1.9% | +4.8% | +3.4% |
| `wq_b` (TP8) | 4096 x 1280 | -1.7% | +2.7% | **+13.8%** |
| `wo_a` (TP4) | 2048 x 4096 | +5.4% | **+11.7%** | **+16.1%** |
| `wo_a` (TP8) | 1024 x 4096 | +5.8% | **+13.8%** | **+16.1%** |

**The win is tuning, and it does not generalise.** `wq_b` and `wo_b` are the two
shapes `gemv.py`'s table was swept for, and they win at every M and every TP.
Everything else falls back to `_HEURISTIC`: six shapes land inside +-2%, which is
run-to-run noise, and **`wo_a` loses outright at both TP degrees, by 12-16% at
M=8 and above.** `wq_b` at TP8 joins it at M=32.

`wo_a` is the interesting one because it is not a tuning miss in the ordinary
sense -- it is the only shape here with N <= 2048 *and* K >= 4096, so the
heuristic's 4-wave 16x16 tile has both few N tiles to spread over and a long K to
walk. A shape that matters should be swept (`sweep.py gemv-tune`, 72 configs, about
two minutes a bucket) rather than assumed to inherit `wq_b`'s margin.

None of this reaches the deployed path: `wo_a` is applied through the model's own
batched absorb GEMM, not through the linear's quant method, so the hook never
sees it.

###### Why a bf16 activation is not simply worse

mori's GEMV takes fp8, so a bf16 caller pays a separate `mxfp8_e4m3_quantize`
launch. That pass is **1.9-2.2us on every shape**, near enough all of it launch,
since it moves at most 128 KB. It would follow that bf16 is always a loss --
SGLang's GEMV quantises inside the kernel and pays nothing.

It does not pay nothing. Subtracting SGLang's own fp8-in GEMV from its bf16 one
prices its in-kernel quantise, and that cost is not fixed at all (M=8):

| layer | N x K | SGLang's in-kernel quantise | mori's separate pass | bf16 pipeline |
|---|---|---|---|---|
| `wq_b` (TP4) | 8192 x 1280 | 0.31us | 1.97us | +32% |
| `wo_b` (TP4) | 5120 x 2048 | 0.36us | 1.99us | +25% |
| `wo_b` (TP8) | 5120 x 1024 | 0.44us | 1.90us | +35% |
| `wq_b` (TP8) | 4096 x 1280 | 0.83us | 1.94us | +27% |
| `wo_a` (TP8) | 1024 x 4096 | 1.99us | 2.00us | +10% |
| `wo_a` (TP4) | 2048 x 4096 | 2.01us | 2.09us | +10% |
| `wqkv_a` (TP4) | 1792 x 5120 | 2.51us | 2.07us | **-6%** |
| `wkv` (TP4) | 512 x 5120 | 2.64us | 2.09us | **-7%** |
| `wq_a` (TP4) | 1280 x 5120 | 2.65us | 2.02us | **-9%** |
| `shared gate_up` (TP4) | 1152 x 5120 | 2.66us | 2.06us | **-8%** |
| `wq_b` (TP1) | 32768 x 1280 | 4.70us | 2.10us | **-15%** |
| `wo_b` (TP1) | 5120 x 8192 | 5.31us | 2.17us | **-21%** |

The last column tracks the first two exactly: mori wins the bf16 pipeline at
every point where SGLang's in-kernel quantise costs more than one launch, and
loses at every point where it costs less. Nothing else is needed to explain it.

The reason SGLang's cost varies 17x is that **its fusion quantises the same
activation once per workgroup.** Each workgroup owns a different N tile and the
same tokens, so the work is redundant across the grid and grows with both N and
K; at `wq_b` TP1 (N=32768) it is 4.7us of the 15us kernel. mori's pass is run
once whatever the grid, which is why a separate launch can beat a free fusion.

**This is measured, not acted on.** The hook still declines every bf16 call
below 32 tokens, and that stays right for the layers it is enabled on -- `wq_b`
and `wo_b` are the four worst rows in the table. Turning it into a gate would
mean predicting SGLang's redundant quantise cost from the shape, and the
boundary is thin: `wo_a` TP4 sits at 2.01 against 2.09us, a 0.08us margin that
is inside run-to-run noise.

##### DeepSeek-V4-Pro (blockscale, TP8)

`[M, 7168]` K=2048 on 8 ranks with `--chunked-prefill-size 16384`.
`--quant blockscale`, median of 11:

| M | `split-sdma` | `fused-sdma` | `fused-sdma` + fp8 gather |
|---|---:|---:|---:|
| 4096 | 398.1 us | 351.0 us | **329.5 us** |
| 8192 | 722.0 | 621.1 | **539.3** |
| 16384 | 1472.9 | 1148.8 | **979.3** |

Fusing is worth **-22%** at M=16384; the fp8 gather a further **-15%**. For
scale, the same layer as the model runs it today (a separate GEMM then an NCCL
all-reduce) measures **1419.5 us** at M=16384, and the GEMM alone is **369.4
us**.

The model-captured phase breakdown is preserved in the model-evaluation fold below.

**What fp8 costs, numerically.** relL2 against an fp32 host reference goes from
**2.35e-3** (bf16 wire, bitwise exact through the collective) to **2.49e-2**.
That is a floor, not a tuning problem -- e4m3 carries 3 mantissa bits, and scale
granularity barely moves it, measured in torch on a `[2048, 7168]` standard
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

Model-quality records from this study are preserved in the model-evaluation fold below.

##### DeepSeek-V4.1-Flash (mxfp8, TP4)

`[M, 5120]` K=2048 on 4 ranks. **The conclusion is not V4-Pro's.** There, fusing
is the whole story. Here the overlap is the *smallest* of three effects, and
reading the V4-Pro numbers across would set every threshold wrong.

**What fusing is worth**, `sweep.py fused` and `sweep.py fused-fp8`, max over
ranks:

| M | wire | `gemm-only` | `split-sdma` | `fused-sdma` | gain | ceiling |
|---|---|---:|---:|---:|---:|---:|
| 4096 | bf16 | 65.6 | 443.6 | 437.3 | +1.4% | 14.8% |
| 8192 | bf16 | 105.0 | 827.3 | 801.8 | +3.1% | 12.7% |
| 16384 | bf16 | 180.2 | 1593.5 | 1500.7 | +5.8% | 11.3% |
| 4096 | fp8 | 65.8 | 364.8 | 358.8 | +1.7% | 18.0% |
| 8192 | fp8 | 102.9 | 669.7 | 643.1 | +4.0% | 15.4% |
| 16384 | fp8 | 176.3 | 1309.2 | 1195.1 | +8.7% | 13.5% |

The fp8 rows at 4096 and 8192 are new; the rest reproduce an earlier run of the
same matrix to within 2.5% on every cell, with the gains and ceilings identical
to a tenth of a point.

**The fp8 wire and the fusion are close to independent, and the wire is the
bigger of the two.** At M=16384 it is worth -17.8% on the split path on its own,
before anything is fused. It then *raises* what fusing is worth -- +5.8% to
+8.7% -- because shortening the collective makes the GEMM a larger share of the
layer, which is the same effect the ceiling column reports (11.3% -> 13.5%).
Together: **1195.1us against 1593.5 for split over the bf16 wire, -25.0%.**

The **ceiling** column is `gemm/split`: what fusing would be worth if it hid the
GEMM entirely. It is 11-18%, because the GEMM is ~180us against 1130-1410us of
communication -- between 1:6 and 1:8. Fusing collects about half of the ceiling.
Making the GEMM faster moved this number the wrong way, which is why V4-Pro's
+21% does not transfer.

**`fused-lsa` is negative everywhere**, on both TP splits, every M and both
wires: re-measured at TP4 it is 5.3-8.0% worse than `split-sdma` across the six
cells above. `split-lsa` is fine (427.0 against sdma's 443.6 at M=4096), so it
is the fusing that hurts, not the transport: direct-LSA moves the bytes with the
GEMM's own waves, and when the GEMM is 11% of the layer, spending its issue
slots on the transfer is a bad trade. Use `fused-sdma`. `chunks` wants more of
them as M grows (M=16384: c=1 1634.4 -> c=8 1503.6, -8%); lsa is flat and
marginally prefers c=1, and the existing default picks the winner at every point
measured.

**`split-lsa` has no fp8 leg, and asking for one used to be silent.**
`build_lsa_ar` takes no `gather_dtype` at all -- the LSA 2-stage all-reduce moves
bf16 and that is the whole of it. `--mode split-lsa --gather-dtype fp8` therefore
ran a bf16 collective and reported it under the fp8 label: same time as the bf16
wire to within 0.2%, and a relL2 at the bf16 quantisation floor rather than
fp8's. The validation gate is two-sided precisely for this -- its lower bound of
5e-3 exists to assert the fp8 wire was *taken*, not just requested -- so it
caught it, but only after paying for the run. `bench_gemm_ar.py` now refuses the
combination where it is asked for.

**Picking the fp8 gather's transport.** The fp8 leg is worth more than the
fusing here, so its own knobs matter. Three are real; two that look like knobs
are not -- `--gather-bands` and `--fuse-reduce-push` are read only by
`sdma_reduce_push`, which rejects `fp8_gather` outright, and
`scatter_dtype='fp8'` raises `NotImplementedError` (its regions are sized but
nothing writes the payload; that is the other leg and the larger remaining
lever). TP4, M=16384, all 24 runs validated and all at relL2 2.31e-02 -- these
knobs move time, not numbers:

| | gather=sdma, fq off | fq on | **gather=lsa, fq off** | fq on |
|---|---:|---:|---:|---:|
| `fused-sdma` | 1222.5 | 1244.1 | **1192.1** | 1209.9 |
| `fused-lsa` | 1412.1 | 1433.0 | 1375.6 | 1398.7 |
| `split-sdma` | 1342.1 | 1358.9 | 1310.3 | 1326.8 |

Every cell says the same two things, at M=4096 as well. **Pull the gather**:
2.4-2.6% faster than pushing at M=16384, 3.6% at M=4096, in all three modes.
**Do not fuse the quantise**: 1.1-2.3% everywhere, +17.8us on the winning
combination. Note this is the opposite of the scatter leg, where `fused-lsa`
loses -- not a contradiction: the scatter runs *inside* the GEMM and competes
with the MFMA for issue slots, the gather runs after it.

Best: `fused-sdma` + `gather_dtype=fp8` + `gather_transport=lsa` +
`--no-fuse-quantize`. **1192.1us against 1592.5 for split/bf16, -25.1%**, of
which the fp8 wire alone is -15.6% (it needs no fusing) and fusing on top of it
a further -8.8%.

**Against what the model runs today.** The numbers above compare mori against
mori. The threshold question needs the path being replaced:
`mxfp8_native_blockscaled_linear` followed by an all-reduce, where
`native_route_plan` picks `dot_scaled` below M=8192 and `hipblaslt_bf16` above
it. One M per process, two runs:

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

Two things this settles that a coarser sweep would not. **The baseline is not
monotonic** -- it retunes per M bucket and swings ~20% between them (M=8200
costs 943us where M=8800 costs 1133), so a threshold has to be read off the
whole curve, not interpolated between two points. And **one `m_pad` is not one
answer**: the fused cost is fixed by the padded size while the baseline follows
the true M, so m_pad 5120 wins at fill 1.000 and loses at 0.918 and 0.820. No
fill threshold separates those from the *winning* low-fill points above (0.879
at m_pad 8192 wins -17.6%), which is why the floors in SGLang's
`mori_gemm_ar.py` are 8192 (bf16) and 2048 (fp8) rather than a single number
plus a fill guard.

> These numbers are only meaningful because `_PinnedLaunch` exists -- see
> [Measurement traps](#measurement-traps).

##### Measurement traps

Each of these produced a confident wrong number before being caught, and none is
about the kernel.

**A single-call CUDA-graph capture has a 13.40us floor on this box.** The same
do-nothing kernel amortised over 200 calls in one graph is 1.51us, so anything
under about 15us measured that way is mostly harness. It is invisible at
M=16384, which is why it survived several rounds. Two committed thresholds had
to be re-derived because of it, and one of them was not a win at all. Use
`benchmark/cco/flydsl/gemm_ar/timing.py`.

**A hot loop reads the weight out of LLC.** MI355X has 256 MB of it and a
`wq_b` weight is 10.5 MB: 4308 GB/s hot against 2561 GB/s with copies rotated
past the cache, a 1.7x overstatement. A forward pass reads each layer's weight
once, so **cold is the number that predicts a server**; report both.

**And rotating the copies is not enough on its own -- the graph has to be long
enough to reach them.** `cold_hot_us` sized its ring at 384 MB and then captured
`reps=32` calls, but the ring advances at *capture* time, so the graph bakes in
32 pointers and every replay revisits those same ones. The working set was
`min(reps, n)` copies, not `n`, which for seven of these twelve shapes is under
the LLC: `wkv`'s 2.5 MB weight gave 80 MB. Those columns were labelled cold and
were not.

The shapes that landed *on* 256 MB read worst of all, which is the tell. A
working set at exactly cache capacity thrashes, where one comfortably over it
just streams -- so `wo_a` (8 MB, 256 MB at 32 reps) measured +10% against
SGLang cold and -2% hot, and the two disagreeing in sign is what exposed this.
Fixed by taking `cold_reps = max(reps, n)`. The correction is not cosmetic: on
the GEMV it turned four apparent 9-14% wins into ties and made two losses
larger. Shapes whose weight already exceeded 8 MB -- `wq_b`, `wo_b`, both TP1
shapes -- moved by less than a point, which is how the diagnosis was confirmed.

**An eager layer benchmark is biased against the fused path.** It launches four
FlyDSL kernels per call where the baseline launches two Triton ones, and before
`_PinnedLaunch` that was 181.6us of per-dispatch Python on the critical path
(`JitFunction.__call__` re-derived its cache key every launch -- an `inspect`
bind, a 35-global snapshot, a drift check -- 85.9us for the GEMM and 31.9us per
phase against 5.9us of actual `hipModuleLaunchKernel`). The server does not pay
it, since its prefill replays a CUDA graph, so the harness read the bf16 wire at
+11.0% where the server read -2.3%, and a threshold was set from the former. The
tell was arithmetic: 109us of "launch overhead" is impossible when a launch is
2-5us, and checking that rather than accepting it is what found it.

**Sweeping several M in one process contaminated one point.** M=4200 and M=4700
pad to the same 5120 and must therefore cost the same; in a multi-M process they
read 697us and 1083us. Isolated, both are 697. The mechanism was never
identified, which is the point -- one M per process is cheap insurance.

**A silent non-fusing path looks exactly like a slow one.** `layer.weight` is
rebound to its shuffled form `[N/16, K/128, 2048]` after
`prepare_mxfp8_native_weight`, so `shape[0]` is N/16; reading it as N turned
5120 into 320, `supports` rejected it for not being a multiple of `BLOCK_N`, and
the path fell back **without a word**. The server stayed correct and the profile
stayed plausible. Only a harness that asserts fusing *happened* catches this,
which is why both the correctness check and the end-to-end test now do.

**A threshold fitted on a coarse M grid is fitted to the gap.** `NARROW_N_BELOW_M`
was a bare `M < 2048`, measured on a grid that jumped 1024 -> 2048 and so never
looked between them. It was wrong on both shapes; at `wq_b` M=1280 correcting it
was +19.9% -> -12.8%.

**With `BUILD_CCO_SDMA=OFF` every put silently does nothing.** The all-reduce
returns mostly the local slice, the model still answers fluently, every mori
kernel still appears in the profile, and the fused path measures **faster** than
it is because it is not moving data -- -6.8% instead of -2.3%, with fp8/lsa
appearing *worst* of the three rather than best, since the pull is the one leg
that does not go through SDMA. Perplexity catches it and nothing cheaper does:
862511 against 3.26 on the same text. A short prompt cannot catch it either,
because fusing needs M >= 4096.

</details>

<details>
<summary>Historical implementation investigations and negative results</summary>

#### How it works

The fused path, from the epilogue outwards. Everything here is about
`GemmAllReduceOp` -- the two standalone ops compile the same GEMM with this
tail switched off, so only the mainloop and the C store apply to them.

##### Why the SDMA transport needs no epilogue change

`C` is an ordinary tensor argument, so pointing it at a cco window is a host-side
change; `StoreC` is untouched. That is the whole reason SDMA is the cheap
transport to fuse. The LSA alternative -- the epilogue storing straight into a
peer -- is implemented as `fused-lsa`, and it is the slower of the two for the
reason that was predicted: `StoreC._store_bf16` writes one bf16 at a time
through `BufferCopy16b`, and gcnasm measured that lane-scatter at 0.26x when
the destination is a peer. It spends ~560us of its GEMM pushing C over xGMI
where the copy engines move the same bytes in ~499.

##### Completion protocol

One monotonic counter per (destination, chunk) in the window (`cfg.counter_off`).
Every block, after its four `store_c.store` calls:

    s_waitcnt vmcnt(0) ; s_barrier       -- this block's C tile has retired
    __threadfence_system()               -- ...and is visible to the copy engine
    thread 0: prev = atomic_add(counter[dest][chunk], 1)
    if (prev + 1) % tiles_per_chunk == 0 -- I am the last tile of that chunk
        sdma.put(dest, ...)              -- fire and forget, no quiet

Counters are never reset, so the modulo test works on every launch and the kernel
stays CUDA-graph-safe, exactly like the barrier flags in `ar.kernels_lsa`.

`quiet` is deliberately *not* called here: gcnasm measured fusing it into the
GEMM as a 1.8-10.7% regression, so the drain kernel does it afterwards. No
per-destination spin lock either -- gcnasm needed one because several CTAs could
submit for one destination, whereas the modulo test elects exactly one.

Tiles are walked (chunk, destination, n) with the destination rotated by rank,
rather than in aiter's linear order. Linear order finishes destination 0's whole
slice, then 1's, and so on, so the last destination's link only starts at the end
of the GEMM and nothing overlaps. The rotation is gcnasm's
`opus_direct_stripe_tile` idea. It costs nothing: 49.72us against 49.80us for
the GEMM alone.

##### The chunks race: it was the GEMM, and it is gone

`--chunks` > 1 produced wrong output about 3 runs in 10 and was pinned to 1
for that reason. The cause was not the chunk protocol at all: it was aiter's
8-wave GEMM under-counting one `s_waitcnt` in its main loop, which corrupted
output non-deterministically on large grids whatever the epilogue did (see the
comment at that `wait_barrier` below). Since that fix, at [16384, 7168]
K=2048 on 8 ranks:

* chunks 1/2/4/8, 10 runs each -- 40/40 correct, no hang;
* chunks=8 alone, 25 more runs -- 25/25 correct, no hang.

Two hangs were seen at 8 ranks while the sweep was still being set up and never
reproduced in the 100+ runs after. The submit lock is a plain test-and-set spin
(`_acquire_peer_lock`), so a hang is not impossible; use `timeout` when
sweeping and treat one as a finding rather than a flake.

Two things in this file survived that misdiagnosis and are worth keeping
straight:

* The half-wave barrier pairing (`if wave_m == 0: s_barrier()` before
  `store_c`) is still needed and still right -- gcnasm does it
  unconditionally at kernel_template.hpp:693. It took `--chunks 2` from
  1-in-3 failing to 3-in-10, which at the time read as "better but not fixed";
  the residual 3-in-10 was the GEMM.
* Two *other* diagnoses were wrong and are recorded so they are not retried.
  **A shared SDMA queue**: gcnasm's per-destination submit lock was ported and
  ISA-verified; it fixed nothing, and is kept only because cco's
  one-issuing-warp-per-queue rule still applies. **The counter atomic's
  ordering**: `acq_rel` appeared to beat `monotonic`, which is how the
  acquire half got justified; against a 1-in-3 intermittent failure that
  comparison was noise. `acq_rel` stays because release/acquire is right for
  a producer handing tiles to a consumer, not because it was measured -- and at
  chunks=1 it demonstrably orders nothing, since 20 runs of `monotonic` pass
  too.

Repeated runs are still the only way to judge any of this, and the gate has to
be tight: the corruption landed at 4-9e-3 against an fp8 floor of 2.35e-3, so a
5e-3 threshold reported a corrupt run as validated (it did, at 3.97e-3). The
bench gates at 3e-3, and `test_fused_is_stable_across_repeats` requires three
runs to be *identical* rather than each small.

##### Mode comparison, once the chunks are unblocked

> These were taken in a different round from the
> [V4-Pro tables above](#deepseek-v4-pro-blockscale-tp8) -- before the fp8 wire
> existed -- and the absolute numbers do not line up with them (split-sdma reads
> 1261.7 here against 1472.9 there). What transfers is the *ordering* of the
> modes and the per-chunk breakdown, which is what this section is for.

8 ranks, [16384, 7168] K=2048 -- the real prefill shape -- graph replay, all
three C-store stages on (now the benchmark default), median of 31, max over
ranks:

    mode                        time
    fused-sdma  chunks=8      1114.1us   <- best
    split-sdma                1261.7
    split-lsa                 1262.4
    fused-lsa                 1571.4
    gemm-only                  228.9

Per-kernel, the overlap is visible directly:

    chunks   GEMM   drain  reduce  gather   total
         1  257.5   493.8    44.0   496.4  1291.7
         2  266.5   379.4    44.2   496.4  1186.6
         8  265.0   303.7    43.8   496.6  1109.1

The drain falls 38% while the GEMM grows 7.5us for the extra counter atomics
and the lock. Going finer is worse: at chunks=8 each PUT is 3.5 MiB, and 16
(via `--block-m 128`) halves that to 1.75 MiB, under the knee in the SDMA
bandwidth curve, for 1130.8us.

fused-lsa still loses, and for a reason that is not going away: it spends 730us
of its GEMM pushing C over xGMI where the copy engines move the same bytes in
499, and ATT shows 99% of that store time is *stall*, so coalescing the stores
(the three C-store stages) buys it nothing -- 954.5 -> 952.1us.

The remaining floor is the tail: reduce plus all-gather is 540us of the 1114,
and neither is touched by fusing the GEMM.

##### The C store: three stages off gcnasm

`StoreC` emits 128 `buffer_store_short` per block -- one bf16 at a time --
because of the MFMA accumulator layout, not a missed vectorization. Lane `l`
holds `D[4*(l/16)+i][l%16]`: four consecutive *rows*, stride `c_cols`, so its
four values are 14336 bytes apart in a row-major C, and the eight bf16 that would
make a 16-byte store live in eight different lanes.

    stage                              GEMM kernel   store instrs   store cycles
    (aiter as-is)                        38.30us       128 short         14,608
    --swap-ab                            slower         32 dwordx2       32,720
    --swap-ab --permlane                 37.04us        16 dwordx4       14,604
    --swap-ab --permlane --lane-transpose 35.74us       16 dwordx4        3,852

Median of 42 dispatches, 8 ranks, [4096,7168] K=1024. Bit-identical to aiter at
[512,512,256], [1024,768,512] and wo_b (`test_swap_ab_is_bitwise_identical`),
and registers are unchanged throughout (VGPR 128 / SGPR 112 / 0 scratch).

**1. `--swap-ab`** -- `mfma_adaptor_swap_ab` (opus.hpp:2064, literally
`base::operator()(b, a, c)` with `dim_c()` redefined). Computing
`B^T A^T = (A B)^T` in the accumulator's own layout moves a lane to
`D[l%16][4*(l/16)+k]`: four consecutive *columns*, 8 contiguous bytes. Alone it
is *slower* -- a row still only gets 32 bytes, so the same 16 transactions are
squeezed onto a quarter as many instructions and per-instruction address fan-out
quadruples (1022 cycles each against 114).

**2. `--permlane`** -- two `v_permlane16_swap_b32` per M-tile. Semantics
measured rather than assumed: `vdst' = [X.r0, Y.r0, X.r2, Y.r2]`,
`vsrc' = [X.r1, Y.r1, X.r3, Y.r3]`, so::

    (A, B) = permlane16_swap(tile0.d0, tile1.d0)
    (C, D) = permlane16_swap(tile0.d1, tile1.d1)
    lane group g stores (A, C, B, D)

lands g = 0,1,2,3 on columns 0-7, 16-23, 8-15, 24-31 -- together columns 0..31
contiguously, 16 bytes per lane and 64 per row. No `ds_bpermute`, no LDS; the
column permutation is absorbed into the address.

This does **not** speed the store up (14,608 -> 14,604). What it pays for is
everything a 2-byte store drags along: 128 stores need 128 addresses, 128 bounds
predicates and 128 scalar multiplies, 16 need 16, and the swapped layout makes
B's scale a vec4 so the scaling packs into `v_pk_mul_f32`::

    v_mul_f32_e32   257 ->   1     v_lshlrev_b32_e32  159 -> 45
    v_pk_mul_f32      0 -> 128     v_add_u32_e32      133 -> 21
    v_cvt_pk_bf16_f32 128 -> 64    v_cndmask_b32_e64   96 ->  8

**3. `--lane-transpose`** -- gcnasm's second stage
(kernel_template.hpp:491-510), one `ds_bpermute` per dword. After the permlane
stage a row's four 8-column chunks sit in lanes 16 apart, so a 16-lane group
touches 16 rows at 16 bytes each. Transposing the lane index -- lane
`l' = 4r'+q'` pulls from the lane holding `(row r', chunk q')`, with
`g = q`'s two bits swapped, 0,1,2,3 -> 0,2,1,3 -- puts adjacent lanes on one
row, so lanes 0-3 write 64 contiguous bytes and a group covers 4 rows.

Same instructions, same 64 addresses, only which lane holds which. The store's
own latency falls **14,604 -> 3,852**, which is the coalescer being sensitive to
lane adjacency and not just to the address set -- exactly what gcnasm's
"pair-coalesced" comment is about. The 64 `ds_bpermute` cost 9,604 cycles.

###### `--hoist-scales`: a real redundancy that does not pay to remove

The epilogue calls `store_c.store` four times, and those calls share base rows
pairwise and base columns pairwise, so every scale is fetched twice. The
source-attributed thread trace counts it exactly: 16 `buffer_load_dwordx4` at
`gemm_a8w8_8wave.py:191` where only 8 addresses are distinct, and 8
`buffer_load_dword` at :199 where only 4 are -- 12 of 24 loads redundant, plus
12 redundant address computations, 4,276 cycles or 0.8% of the kernel. The
compiler cannot merge them because all four calls write the same `reg_f32_*`
register buffer, which makes them a chain of overwrites rather than pure loads.

`store_all` loads each scale once. It removes exactly the predicted loads --
A-scale 16 -> 8, B-scale 8 -> 4 -- and is slightly **slower**:

    variant                VGPR  scratch  instrs  dwordx4  dword    gemm
    --permlane --lane-transpose  254    0B     1581      72     16   33.32us
      + --hoist-scales           256    0B     1595      68      8   33.96us

Keeping both scale sets live across all four stores costs more in register
moves and re-materialised addresses than the twelve loads were worth, and it
spends the last 2 VGPRs of headroom (254 -> 256). Bit-exact either way.

Kept as a switch rather than deleted: the redundant fraction grows with
`N_TILES_B`, so a larger `BLOCK_N` would change the arithmetic, and if
register pressure ever loosens this flips sign. It is also cheaper to re-measure
a flag than to re-derive why it was rejected.

Two limits. **It is worth nothing on `fused-lsa`**, where the store goes to a
peer over xGMI rather than to local memory::

                        gemm    barrier  reduce  gather     sum
    fused-lsa          177.4us     5.4    11.1   135.3    329.1
    + all three        175.6us     7.8    11.1   135.0    329.4

against 39.0 -> 35.7us for the same three stages on the split path's local
store. Stage 3 buys the local memory pipeline's sensitivity to lane adjacency;
a peer store crosses the fabric instead, and that phase is limited by ~44 GB/s
of link bandwidth, which no amount of transaction shaping changes. (This was
first "measured" with only stages 1 and 2 wired into the direct path -- which
are exactly the two that do *not* speed the store up -- and the conclusion was
right by accident. The direct-LSA branch for stage 3 exists now, and the
argument-validation above is there so a missing branch fails loudly instead of
silently running a weaker variant.)

And 64 bytes is one wave's ceiling here, since a wave owns
`N_TILES_B * 16 = 32` columns; gcnasm reaches a full 128-byte line only by
having four `wave_id_n` waves tile adjacent 16-column runs.

All three stages are candidates to push back into aiter's `StoreC`: bit-exact,
no register cost, and the 6.7% is on the GEMM itself, independent of any
all-reduce.

##### What gcnasm does differently

`/workspace/gcnasm/opus_gemm_dist/opus_gemm_a2a_lsa` fuses a GEMM with an
all-to-all and gets 16-27% out of it. Reading it explains most of why this does
not, and one of its lessons was worth ~90us here.

1. **Its collective is one phase; this one is three.** An a2a scatters the GEMM
   output once. An all-reduce is scatter + reduce + all-gather, and fusion only
   touches the scatter -- reduce + gather is 150us of the 327us baseline, 46%,
   untouchable by construction.
2. **Its ratio is inverted.** M=2048 N=18432 K=8192 gives ~518us of GEMM against
   ~200us of comm, 2.6:1. wo_b is 39us against 137us per phase, 0.29:1. Overlap
   can hide at most the smaller of the two, so theirs hides most of the comm and
   this hides at most one GEMM.
3. **Its best mode has no producer/consumer handoff at all.** "Direct LSA" has
   the GEMM epilogue store *straight into the destination rank's buffer*. No
   staging, no copy engine, nothing to publish mid-kernel -- the only sync is the
   barrier at the end. The publication problem this file spends all its time on
   simply does not exist there.
4. **Its fused-SDMA path uses no cache fence.** `opus_chunk_sdma_submit` is
   `s_waitcnt vmcnt(0)`, `s_barrier`, an `__ATOMIC_ACQ_REL` counter, and a
   per-destination submit spin lock -- no `__threadfence_system`, no
   `buffer_wbl2`. That is the lesson that transferred: the acq_rel counter *is*
   the release, and the explicit fence added here was 90us of pure waste
   (fused 440us -> 350us on removing it). Its ISA shows the atomic already emits
   its own `buffer_wbl2`/`buffer_inv` pair, on thread 0 only.

#### Negative results

Kept because each reads as obviously right and the reason it is not cannot be
seen from the source.

##### A CK-shaped 4-wave GEMM

An earlier revision carried `kernels_preshuffle4w.py`, a port of CK's 4-wave
B-out-of-LDS shape, chasing a 22% gap between this GEMM and CK's at the same
shape. It reached CK's instruction mix and not its speed, and it was deleted
rather than carried (it needed ~1000 lines of further aiter vendoring to serve
a kernel nothing calls). What the investigation ruled out, since the same
ground should not be walked twice:

Ten hypotheses were falsified by measurement -- promote arithmetic, register
spill, store width, VALU scheduling groups, MFMA batch size, scale loads,
address hoisting, load-to-use distance, occupancy, and dependency structure.
Hardware counters (`rocprofv3 --pmc`) show *identical* `SQ_INSTS_MFMA`
(7,340,032) and `SQ_VALU_MFMA_BUSY_CYCLES` (234,881,024), VALU within 1%, and
`MemUnitStalled` at approximately zero -- but `SQ_WAIT_ANY` at 156.0M against
CK's 120.4M. A K-sweep puts the whole difference per-iteration: our fixed cost
is *lower* (51.6us against 66.0us), while each K-block costs 20.9us against
14.5us. The gap is wait, not work, and it is not in any of the places listed
above. See commits `cc696762`, `54fef960`, `9f990637`, `1dead3fc` for the
traces.

##### Persistent tiles

gcnasm builds this kernel family both ways (`PERSISTENT=1|0`) and its README
has a tail-balance sweep: the win is entirely a function of the remainder after
whole 256-CTA batches -- +9.97% at remainder 8, +8.41% at 32, and only +0.77% at
192. wo_b is 16 x 28 = 448 tiles on 256 CUs, i.e. remainder **192**, the benign
end of that curve.

The tail itself is real here, and large. Sweeping N at M=4096, K=1024
(gemm-only, best of 3, 8 ranks):

    N       tiles  remainder   time
    4096      256      0      32.64us
    4352      272     16      45.88us     <- 6% more work, 40% more time
    4608      288     32      44.68
    5120      320     64      45.24
    6144      384    128      47.40
    7168      448    192      49.80

So it was implemented (one workgroup per CU, striding over tiles) and made
bit-exact. It is a ~2x regression, and the reason is a hard resource wall rather
than anything to do with scheduling. From the kernel metadata in the final ISA:

    variant                 VGPR  AGPR  SGPR  V-spill  scratch  instrs   gemm
    as committed             256     0    50        0       0B    2052  38.54us
      + 3-stage C store      254     0    51        0       0B    1573  35.74us
    body inside an scf.for   256     0    68       38     156B    2260  74.72us
    persistent, grid 256     256     0    67       38     156B    2264  72.22us

**The baseline already uses all 256 VGPRs with zero spill.** Wrapping the
pipeline in a runtime loop asks for 38 more -- the K pipeline's accumulators and
operand fragments die at the end of a tile in the flat version, but a loop makes
the compiler assume they may be live across the back-edge -- and there is nowhere
to put them. `--amdgpu-num-vgpr 256/512` changes nothing, because the cap was
never the constraint. Persistent and non-persistent spill identically, which
confirms the cost is `scf.for` itself and not the scheduling idea.

Note `fly-promote-regmem-to-vectorssa` is **not** the problem, contrary to what
an earlier version of this note claimed. It handles `scf::ForOp`, it promoted
all 451 register allocas here (zero left afterwards, zero `llvm.alloca` in the
final IR), and the emitted loop carries no `iter_args` at all. The 156 bytes
are ordinary register-allocator spill: `.vgpr_spill_count: 38`, 38
`scratch_store_dword` / 38 `scratch_load_dword`.

The pass does have one real inefficiency, just not one we hit: an alloca declared
*outside* a loop is carried as an `iter_arg` unconditionally, even when it is
fully overwritten every iteration, because `collectTouchedRegAllocaInRegion`
records on any load *or* store with no liveness test. Allocas declared *inside*
the loop are correctly materialised as `ub.poison` and not carried, and ours
are all inside.

Two bugs the attempt surfaced, worth knowing if anyone loops this pipeline:

* **The half-wave barrier does not survive a loop.** The prologue's
  `if wave_m == 1: rocdl.s_barrier()` deliberately runs the two half-waves one
  barrier out of phase. Fine once; in a loop the offset accumulates by one per
  tile, so from the second tile on the halves rendezvous at mismatched program
  points and the LDS double-buffering races -- silently, as partial corruption
  inside otherwise-correct tiles. A compensating `if wave_m == 0: s_barrier()`
  at the end of each tile fixes it exactly.
* **The LDS handles must be rebound per tile.** The pipeline swaps those Python
  bindings as it advances, so hoisting them makes eight shared-address-space
  pointers loop-carried, which fails to legalize.

And a FlyDSL gotcha: `range(...)` must appear literally in the `for`
statement or the AST rewriter does not see it and Python evaluates it eagerly
("dynamic 'ArithValue' has no Python integer representation"). Assigning it to a
variable first does not work, so a kernel cannot cheaply offer both a looped and
a flat form from one body.

Reviving this needs 38 VGPRs from somewhere. The 3-stage C store is the only
change measured to *reduce* pressure (256 -> 254, and 2052 -> 1573 instructions,
since 128 two-byte stores need 128 addresses and 128 predicates live at once),
and it is nowhere near enough. Halving BLOCK_M would free roughly 16 by halving
the accumulator count, but moves the tile count to 896 -- remainder 128, where
gcnasm measured +0.41%. There is no version of this that pays.

##### Direct LSA (`--mode fused-lsa`)

gcnasm's best mode has the GEMM epilogue store *straight into the destination
rank's window*: no staging buffer, no copy engine, nothing to publish mid-kernel.
Ported here, the structural half is emphatic -- the scatter collapses from
136.8us to 10.1us, i.e. the transfer is entirely absorbed, which fused-sdma never
managed. And the un-coalesced store is nowhere near as bad as feared: 7.34MB per
link in 166.6us is ~44 GB/s, 82% of the SDMA scatter's 53.7, against the 0.26x
gcnasm measured for a lane-scatter pushed to a peer.

It was wrong for a long time, and the fix is one line in the right place. The LSA
2-stage all-reduce publishes from the kernel that produced the data --
`ar/kernels_lsa.py` fences right after its `tmp` stores, in every block.
Direct LSA's producer is the GEMM, and the only fence was in the *separate*
barrier kernel: one block, therefore one XCD's L2 out of eight. The other seven
kept the peer-homed lines dirty, and the LSA flag, being a system-scope atomic,
overtook them. Moving the fence into the GEMM fixes it: **10/10 runs bit-correct**
where it had been 2-3 in 6.

`--direct-fence leader` (thread 0 only, the default) rather than every lane is
worth 80us, 434 -> 355. It is legal only because the half-wave barrier pair is
now closed -- `wait_barrier(0)` really does mean every wave's stores have
retired into this CU's L2, so one wave writing it back covers all eight. Measured
as incorrect before that fix, which is what made a per-wave release look
mandatory.

Two things that did *not* work, and are worth not re-trying:

* **Uncached (sc0|sc1) peer stores**, so there would be nothing to publish. Wrong
  consistently, ~1.7e-2, at every store width. The first time this was tried the
  store was one bf16 and the explanation looked like partial-line writes losing
  updates over the fabric; with `--permlane --lane-transpose` making it 16
  bytes per lane and 64 contiguous per row it is *still* wrong, so that
  explanation was not it and the real one is unknown.
* An uncached recv load in the reduce. No effect, which is what rules out a stale
  read on the consumer side.

It does not beat the split path. Per-kernel (rocprofv3, 8 ranks, [4096,7168]
K=1024, with all three C-store stages):

    fused-lsa   gemm 189.4  barrier 10.1  reduce 10.9  gather 135.6   sum 346.0
    split-lsa   gemm  35.7  + one LSA all-reduce kernel                sum ~310

    end to end, median of 4:   split-lsa 318.8us     fused-lsa 359.0us

The 41us gap is structural, not tuning. LSA's 2-stage does read + reduce + write
in a single pass over the wire, so its reduce is free; Direct LSA writes to the
peer and then pays a separate 10.9us local reduce pass, its store rate is ~14%
under LSA's read rate, and it adds the publishing fence LSA gets from being
local. Absorbing the scatter completely still does not cover that.

##### Folding the narrowing into the reduce

`fuse_quantize=True` makes the reduce write its bf16 and narrow to fp8 from the
same accumulators, saving a 28 MiB re-read and a launch. It is **off**, because
it costs 20 us rather than saving 12:

| | us |
|---|---:|
| split reduce + quantize | 957.4 |
| fused, row stashed in registers | 977.1 |
| fused, row re-read | 982.0 |

Not register pressure -- the re-reading variant keeps no stash and is no better.
It is the thread map: a per-row amax cannot be taken by a block holding only
part of a row, so fusing forces one-wave-per-row, where `sdma_reduce` walks
packs with a flat grid stride and streams a block through all 8 source slices at
once. Confining a wave to 14 KiB at a time costs the reduce more than the
re-read saves.

It is not free and the cost is not a tuning problem. e4m3 carries 3 mantissa
bits, so one rounding costs ~2.1e-2 on a normal payload whatever the scale
granularity -- per-row measures 2.65e-2 and per-32 measures 2.40e-2, 9% better
for 200x the scale bytes. Going from 2.35e-3 to 2.49e-2 is the price of the
10%, and whether that is payable is a model-level question, not a kernel one.

It also only pays at large M. The two conversion kernels are a fixed cost
against a transfer that shrinks with M, so on the standalone all-reduce it is
+2.4% at M=4096 and -11.2% at M=16384. Per phase at M=16384: quantize 12.4us,
dequantize 88.2us, against ~218us saved on the push.

The scatter leg stays bf16. It carries partial sums that are then added across
every rank, so its fp8 error compounds rather than being a single rounding, and
it is already mostly hidden behind the GEMM -- its 1140 GB/s is not a bandwidth,
it is the tell that the pushes went out from the epilogue and the drain is only
waiting for the tail. `scatter_dtype="fp8"` sizes its regions but raises
`NotImplementedError`.

##### Firing the gather's puts from inside the reduce

**Firing the gather's puts from inside the reduce** (`fuse_reduce_push=True`)
looked like the safest of the three: the push sends this rank's *own* slice, so
unlike the pull it has no cross-rank dependency, and SDMA is a copy engine so it
costs no CU time. It reaches parity and not a win — against 1148.7us unfused:

| bands | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| `publish="writethrough"` | 1162.1 | **1157.2** | 1160.3 | 1190.1 | 1323.5 |
| `publish="fence"` | 1227.3 | 1380.1 | 1630.3 | 2137.2 | 3026.7 |

The gap between those rows is the useful result, and it is a lesson about how to
pay for a release rather than whether to.

Handing a range to a copy engine *does* need one: the engine reads over the
fabric, not through a CU's cache, so `s_waitcnt vmcnt(0)` alone is not enough —
it only retires the stores as far as this XCD's L2. But the release can be paid
two ways. Releasing to system scope **after** the stores is L2-writeback work
charged once per block per band (256 x bands of it, ~61us per band, against a
reduce that is only 42us in total — unrepayable). Storing with `sc0+sc1` so the
bytes never stop in L1 or L2 makes the `waitcnt` itself the release, and that is
**free here**: applying the same store policy to the plain unfused reduce moves
it 1153.4 -> 1148.7us, i.e. nothing. This output is written once and nothing
local reads it again before the gather, so holding it in L2 bought nothing.

What remains after that fix is small on both sides and nearly cancels. At
bands=1 the publish carries the mechanism's cost with none of its benefit —
1162.1 vs 1148.7, about 13us for the per-band 256-block `wait_barrier`, the
counter atomic and the elected block's locked puts. Four bands buy back about
5us of overlap before the sync cost takes over again.

Dropping the release entirely is not an option even though it briefly looks like
one: with cached stores and no fence the kernel reaches 1157.9us, but at 32 bands
it produced relL2 1.8e-2 against the 2.35e-3 floor, differing per rank. The same
32 bands are exact under either correct publish mode, which rules out an indexing
bug.

The contrast with the GEMM's fused scatter is the transferable part: there the
publish is amortised against a 437us transfer hidden behind a compute-bound GEMM;
here against 42us of bandwidth-saturated reduce. The mechanism pays when what is
hidden is much larger than the cost of publishing it.

Re-measured after the window-geometry work below, with three alternating
repeats rather than a sweep, it is a clearer loss than the table suggests:
1128.4 us on against 1112.0 off, +16.4 us, against spreads of 2.2 and 1.5 us.

##### Hoisting the window geometry out of `lsa_ptr`

**Hoisting the window geometry out of `lsa_ptr`.** `cco_lsa_ptr` is
`winBase + peer*stride + offset` and loads both fields on every call, through a
*generic* pointer -- which has to be a `flat_load`, since the compiler cannot
rule out LDS, so it counts against `lgkmcnt` as well as `vmcnt`. FlyDSL emits it
as an opaque extern call, and a kernel storing through addresses derived from
that base gives LLVM no way to prove the loads are not clobbered.

Reading the geometry once and doing the arithmetic in the DSL removes all of
that. It was tried four ways -- hoisting out the band loop, a `lsa_geometry()`
API, `global_load` accessors in C++ (`cco_lsa_win_base` / `cco_lsa_stride`, which
take an `address_space(1)` pointer so each is a single `global_load`), and
finally `cco.CachedWindow`, which reads both in its constructor so a kernel
changes by one line. All four measured nothing on `kernels_sdma` (21 call sites)
and `kernels_fused`, in every wire configuration.

It pays in exactly one place (`ptpc`, three alternating repeats):

| | Window | CachedWindow |
|---|---|---|
| `split-lsa` | 1264.69 1264.23 1264.85 | 1250.25 1256.85 1254.53 |
| `fused-sdma` | 1110.45 1109.77 1112.85 | 1110.53 1113.69 1112.61 |

-10.7 us on `split-lsa`, against a 0.6 us spread; nothing on `fused-sdma`.
`ar_1stage`/`ar_2stage` build nine peer addresses in *every block* of a short
kernel; everywhere else the addresses are built once per launch against a body
that runs for a millisecond. **Count address constructions per launch, not
`grep -c lsa_ptr`.**

The same holds against PR #662's branch in `blockscale`, two alternating
repeats: `split-lsa` 1463.6 -> 1455.6 us, while `split-sdma` (1461.2 -> 1459.9),
`fused-sdma` bf16 (1144.9 -> 1145.8) and fp8/lsa (949.2 -> 949.8) do not move.

Two things worth carrying. A `CachedWindow` cannot cross an `scf.if` -- FlyDSL
captures every variable an if body reads as state and requires single MLIR
values, which `Window` satisfies only by having exactly one field -- and the way
out is to compute the addresses before the branch, which is what the offsets
usually allow. And pin `--quant` when comparing against anything: the benchmark
defaults to `ptpc`, ~3% faster than the `blockscale` every blockscale number here is
quoted in, and reading one against the other looks exactly like a machine that
drifts overnight.

</details>

<details>
<summary>Historical model evaluation and server profiles — deferred</summary>

##### Model-captured phase breakdown

**Where the time goes.** Per layer, captured in SGLang over one 20000-token
prefill -- the pipeline as the model drives it, not the standalone benchmark:

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

Two things to read out of the bf16 column. **`gather` is the bottleneck**, not
`drain`: it moves 196 MiB at 470 GB/s, which is 7 xGMI links flat out, so
halving its bytes halves its time. And **`drain`'s apparent 1140 GB/s is not a
bandwidth** -- seven links cannot do that. It is the tell that the scatter's
pushes already went out from the GEMM epilogue and the drain is only waiting for
the tail, which is why the scatter leg has far less to give than its byte count
suggests, and why it is still bf16.

##### Model-quality observations

**Whether that matters is a model-level question**, so it was measured in SGLang
on V4-Pro at TP8, with `SGLANG_DEBUG_FUSED_WO_B_AR=1` logging relL2 per layer
call during the very requests being scored. At the layer, against the unfused
path, 488 calls:

| wire | min | median | max |
|---|---:|---:|---:|
| bf16 | 3.706e-3 | — | 4.046e-3 |
| fp8 / sdma | 1.435e-2 | **2.505e-2** | 2.654e-2 |
| fp8 / lsa | 2.153e-2 | **2.496e-2** | 2.683e-2 |

At the model output it is not detectable. Scoring 10941 tokens of real source
text in a single prefill (mean logprob; lower is a worse model):

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

On every statistic, fp8 is closer to bf16 than bf16 is to itself. That band is
wide because **this model is already strongly non-deterministic**: two bf16 runs
disagree on ~48% of tokens by more than 0.01 logprob, and greedy decode diverges
within 10-20 tokens -- likely the MoE stage-2 epilogue, which accumulates with
`atomic_fadd`. So greedy token agreement is useless here; the bf16-vs-bf16
control is as divergent as bf16-vs-fp8 (35.9% / 15.6% against 28.1% / 23.4% at
~12k / ~24k tokens). Needle-in-a-haystack at 15140 tokens is 24/24 on both
wires -- saturated, so it bounds gross damage without resolving anything finer.

**What this does and does not say.** It says fp8 causes no gross degradation and
no measurable shift in next-token distribution on one scoring task. It does not
say quality is unaffected on long-chain reasoning, code or maths -- that needs a
task benchmark, which has not been run. Short prompts and decode never reach
this path at all (it engages only at M >= 4096), so only long-prefill workloads
are affected.

##### End to end, in SGLang

Prefill throughput, tok/s, `bench_one_batch_server`, TP4 on V4.1-Flash, one
server per variant with GSM8K in front of it.

**The fused wo_b**, three servers:

| bs | base | bf16 wire | | fp8 wire | |
|---|---:|---:|---:|---:|---:|
| 1 | 28148 | 28469 | +1.1% | 29107 | **+3.4%** |
| 4 | 34386 | 35356 | +2.8% | 36442 | **+6.0%** |
| 8 | 35153 | 35920 | +2.2% | 37217 | **+5.9%** |
| 16 | 35423 | 36084 | +1.9% | 37416 | **+5.6%** |

GSM8K 0.917 / 0.912 / 0.918, all three inside each other's noise, so the fp8
leg's relL2 2.3e-2 is below what 1319 questions resolve. That is not what the
gate is for: it catches a collective that moved nothing, which scores near zero
while still answering fluently.

**The two wires do not fuse at the same batch size, and an earlier revision of
this section had that wrong.** It called bs=1 a built-in control on the grounds
that M=4096 is below both floors -- but the floors are 8192 for bf16 and *2048*
for fp8. The shape log settles it: the bf16 wire's smallest served M is 7734,
the fp8 wire's is 1792. So bs=1 is a control for the bf16 column only, and the
fp8 column's +3.4% there is a real win rather than drift.

**The GEMM on its own**, a separate pair of servers with
`SGLANG_OPT_MORI_MXFP8_GEMM=1` and nothing fused:

| bs | base | mori GEMM | |
|---|---:|---:|---:|
| 1 | 28598 | 29313 | +2.5% |
| 4 | 34269 | 35047 | +2.3% |
| 8 | 35086 | 35865 | +2.2% |
| 16 | 35412 | 36232 | +2.3% |

GSM8K 0.920 / 0.928. Its shape log shows `wq_b`, `wo_b` and `wqkv_a` served and
`shared gate_up` declined, which is what the tables above predict.

**Both at once**, which is a fourth configuration rather than the sum of the
other two. Against the same session's base:

| bs | base | fp8 wire only | both | vs base | vs fp8 only |
|---|---:|---:|---:|---:|---:|
| 1 | 28148 | 29107 | 29504 | +4.8% | +1.4% |
| 4 | 34386 | 36442 | 37023 | **+7.7%** | +1.6% |
| 8 | 35153 | 37217 | 37870 | **+7.7%** | +1.8% |
| 16 | 35423 | 37416 | 38058 | **+7.4%** | +1.7% |

GSM8K 0.918.

**They stack, but they do not add.** The fp8 wire alone is +6.0% and the
standalone GEMM alone is +2.3%; together they are +7.7%, not +8.3%. At `wo_b`
the model calls `fused_wo_b` first and the linear only ever sees what it
declines, so the two divide that layer by M rather than both working on it.
What the GEMM adds on top is `wq_b` and `wqkv_a` -- column-parallel, no
collective to fuse with, so fusing could never have reached them.

> **One server out of nine hit an illegal memory access, and it has not
> reproduced.** It was a `both`-enabled run: it served GSM8K for 13 minutes at
> ~250 concurrent requests and then four ranks failed together with
> `hipErrorIllegalAddress`. Re-running the same variant with the same
> configuration completed cleanly -- 0.918, no faults -- and the numbers above
> are from that second run.
>
> **It is not established that the combination caused it**, and the evidence is
> thin in both directions: the combination is the only configuration that has
> ever shown it, and it has shown it once. What was ruled out, so it is not
> re-walked: the two paths do not share a FlyDSL `JitFunction` or `CallState`
> (two identical compiles return distinct objects); 30 rounds of interleaving
> the two paths on four ranks is clean; so is 40 rounds of driving one layer
> across the fusing threshold; `_PinnedLaunch` survives being pinned inside a
> CUDA graph and reused eagerly, replaying bit-identical to a fresh call; and
> memory is not it -- the two servers' pools are byte-identical and 31.7 GB was
> free at the fault. The standalone GEMM also served the same M distribution in
> the run that did not fault as in the one that did.
>
> `hipErrorIllegalAddress` is reported asynchronously, so the traceback points
> at the scheduler's next synchronisation rather than the faulting kernel.
> Pinning it down needs `AMD_SERIALIZE_KERNEL=3`, which is only worth spending
> on a repro that reproduces -- and serialising changes the timing a race
> depends on. Left open, to be re-tested under longer stress.

The decode column is not reported as evidence. `wo_b` never fuses at decode's M,
so the three variants should be identical, and they scatter -8.6% to +14.1% --
that is DSPARK's acceptance rate, not this change.

On V4-Pro, the same comparison as GPU busy time over one profiled prefill:

| | GPU busy | wall | vs unfused |
|---|---:|---:|---:|
| unfused (GEMM + NCCL) | 1096.0 ms | 1.1969 s | — |
| fused, bf16 wire | 1070.3 | 1.1728 | -2.3% |
| fused, fp8 / sdma | 1052.1 | 1.1455 | -4.0% |
| **fused, fp8 / lsa** | **1041.2** | **1.1385** | **-5.0%** |

An earlier capture of the same four read 1101.9 / 1077.2 / 1050.4 / 1048.2, so
this reproduces to about half a percent. The layer-level win is larger than the
end-to-end one because `wo_b` is about 12% of the profile.

</details>

<details>
<summary>2026-09-29 experiments — frozen measurements</summary>

##### 2026-09-29 experiment scope

Base commit: `78b7a5f31`. Single node, 8 gfx950 GPUs with 256 CUs per GPU,
PyTorch 2.10.0, HIP 7.2.26015 and FlyDSL 0.2.4. Other processes retained device
memory; GPU activity was zero before the batch measurements.

Prefill comparisons alternate variants in one process and communicator, using
the same weights and true input rows with separate windows. Each graph has
8 calls and each round has 21 replays. Each round records the maximum of the
per-rank medians; tables report medians over 3–5 alternating rounds. GEMV
comparisons share one
400 MiB weight ring and the same graph pointer order, with 80 calls per graph
and 5 alternating rounds. Launchers, tensor storage and graphs remain alive
through measurement.

The following timings cover GEMM + collective, or GEMV alone where indicated.
They exclude compilation, weight packing, activation quantization/padding and
Python initialization. They are **kernel/collective microbenchmarks, not model
end-to-end speedups**. Raw results and the experiment harness are maintained
outside the repository. Only the final shared-ring GEMV measurements are used;
records marked `invalid-harness` are excluded.

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

##### GEMV wave counts and compact reduction

Single GPU, N=8192, K=1280, MXFP8 input; shared weight ring. These are GEMV-only
timings against the existing MORI GEMV configuration.

| M | Default µs | Compact reduction µs | Specialized waves | Specialized µs | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.060 | 4.098 | 5 | 3.899 | 4.0% |
| 4 | 4.202 | 4.251 | 5 | 3.953 | 5.9% |
| 8 | 4.279 | 4.258 | 5 | 4.062 | 5.1% |
| 16 | 4.778 | 4.700 | 10 | 4.330 | 9.4% |
| 32 | 5.389 | 5.745 | 10 | 4.947 | 8.2% |

All variants are bitwise equal to the default GEMV. K=1280 contains ten
128-wide steps; five or ten waves avoid the unused steps in four- or
sixteen-wave partitions. Use these configurations only for tested shapes.
Reducing only valid tokens (`valid_tokens`) has no consistent benefit and is
not recommended as a default.

| Configuration | VGPRs | LDS bytes | Scratch bytes |
|---|---:|---:|---:|
| M4, default 4 waves | 54 | 4096 | 0 |
| M4, compact reduction | 52 | 1024 | 0 |
| M4, 5 waves | 48 | 5120 | 0 |
| M32, default 16 waves | 46 | 65536 | 0 |
| M32, compact reduction | 44 | 65536 | 0 |
| M32, 10 waves | 52 | 40960 | 0 |

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

##### Decode GEMV + all-reduce

TP8, N=5120, K=2048. Baseline: staged GEMV + LSA all-reduce. The tagged
prototype uses double buffering and an atomic 64-bit payload plus epoch.

| M | Staged µs | Tagged fusion µs | Slowdown |
|---:|---:|---:|---:|
| 1 | 7.247 | 8.406 | 16.0% |
| 4 | 8.229 | 10.056 | 22.2% |
| 8 | 13.819 | 17.059 | 23.4% |

The ordinary ready/done-flag prototype took about 10.834 µs at M=4. Both
prototypes pass an ordered BF16-partial reference and changed-input checks,
but neither should replace the staged path based on these measurements.

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
<summary>Earlier operator rerun snapshot — frozen, incomplete coverage</summary>

This is the result-table snapshot present before this document was reorganized.
Its live progress and setup logs were archived separately. The baseline was
SGLang `e29ebcd051` with an optional-output host adapter; timing-only linear rows
did not carry an independent correctness pass. It is retained as history rather
than treated as a completed version of the new matrices.

##### GEMM linear: SGLang baseline, cold µs


| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 10.75 | 15.42 | 26.16 | 42.42 | 70.54 | 132.34 | 286.30 |
| wo_b | 12.64 | 19.80 | 35.09 | 50.65 | 76.19 | 135.67 | 242.37 |
| wq_a_tp4 | 15.52 | 21.08 | 37.79 | 51.57 | 73.77 | 113.43 | 218.18 |
| wkv_tp4 | 11.73 | 16.30 | 28.58 | 35.41 | 54.00 | 84.31 | 152.88 |
| wqkv_a_tp4 | 15.85 | 24.24 | 42.94 | 64.65 | 92.92 | 133.06 | 258.41 |
| wo_a_tp4 | 13.70 | 20.63 | 40.10 | 52.48 | 99.59 | 111.57 | 220.42 |
| shared_gate_up_tp4 | 15.40 | 20.49 | 35.56 | 48.60 | 71.09 | 112.86 | 214.39 |
| wq_b_tp8 | 9.37 | 12.23 | 21.97 | 31.97 | 41.50 | 71.92 | 137.86 |
| wo_b_tp8 | 9.89 | 12.24 | 22.76 | 32.32 | 50.73 | 82.17 | 140.52 |
| wo_a_tp8 | 10.74 | 14.78 | 26.11 | 42.54 | 57.63 | 117.72 | 146.12 |
| wq_b_tp1 | 20.56 | 33.07 | 76.70 | 137.31 | 260.31 | 491.25 | 952.66 |
| wo_b_tp1 | 28.89 | 39.49 | 86.70 | 135.73 | 267.04 | 508.78 | 902.52 |


##### GEMM linear: MORI 256-column tile, cold µs (vs SGLang)


| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 31.98 (+197.5%) | 35.45 (+130.0%) | 25.58 (-2.2%) | 29.35 (-30.8%) | 49.84 (-29.4%) | 90.50 (-31.6%) | 179.54 (-37.3%) |
| wo_b | 28.69 (+127.0%) | 29.47 (+48.9%) | 32.65 (-7.0%) | 34.95 (-31.0%) | 60.54 (-20.5%) | 95.85 (-29.4%) | 175.10 (-27.8%) |
| wq_a_tp4 | 64.60 (+316.2%) | 67.75 (+221.4%) | 69.53 (+84.0%) | 72.54 (+40.7%) | 78.12 (+5.9%) | 91.17 (-19.6%) | 183.33 (-16.0%) |
| wkv_tp4 | 63.04 (+437.3%) | 66.23 (+306.2%) | 68.14 (+138.4%) | 70.68 (+99.6%) | 74.88 (+38.7%) | 86.60 (+2.7%) | 120.13 (-21.4%) |
| wqkv_a_tp4 | 63.30 (+299.4%) | 67.07 (+176.7%) | 69.09 (+60.9%) | 72.00 (+11.4%) | 77.99 (-16.1%) | 94.18 (-29.2%) | 194.52 (-24.7%) |
| wo_a_tp4 | 49.98 (+264.8%) | 52.31 (+153.5%) | 55.27 (+37.8%) | 58.49 (+11.5%) | 64.08 (-35.7%) | 82.68 (-25.9%) | 171.10 (-22.4%) |
| shared_gate_up_tp4 | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported |
| wq_b_tp8 | 21.71 (+131.7%) | 22.45 (+83.6%) | 24.11 (+9.8%) | 27.40 (-14.3%) | 31.90 (-23.1%) | 54.17 (-24.7%) | 99.63 (-27.7%) |
| wo_b_tp8 | 18.19 (+83.9%) | 18.94 (+54.7%) | 20.59 (-9.5%) | 23.51 (-27.2%) | 38.64 (-23.8%) | 60.39 (-26.5%) | 102.33 (-27.2%) |
| wo_a_tp8 | 49.15 (+357.9%) | 51.71 (+249.8%) | 53.48 (+104.8%) | 56.17 (+32.0%) | 59.77 (+3.7%) | 70.50 (-40.1%) | 100.79 (-31.0%) |
| wq_b_tp1 | 23.21 (+12.9%) | 25.82 (-21.9%) | 49.62 (-35.3%) | 89.15 (-35.1%) | 178.39 (-31.5%) | 374.74 (-23.7%) | 784.08 (-17.7%) |
| wo_b_tp1 | 96.09 (+232.6%) | 100.57 (+154.7%) | 105.37 (+21.5%) | 110.61 (-18.5%) | 199.99 (-25.1%) | 348.91 (-31.4%) | 710.16 (-21.3%) |


##### GEMM linear: MORI 128-column tile, cold µs (vs SGLang)


| Shape | M=64 | M=256 | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
|---|---|---|---|---|---|---|---|
| wq_b | 26.48 (+146.3%) | 30.66 (+98.9%) | 24.24 (-7.3%) | 39.45 (-7.0%) | 70.09 (-0.6%) | 130.37 (-1.5%) | 256.88 (-10.3%) |
| wo_b | 23.85 (+88.7%) | 27.10 (+36.9%) | 31.01 (-11.6%) | 50.80 (+0.3%) | 74.08 (-2.8%) | 121.65 (-10.3%) | 240.10 (-0.9%) |
| wq_a_tp4 | 53.26 (+243.1%) | 58.33 (+176.7%) | 59.61 (+57.7%) | 62.73 (+21.6%) | 68.72 (-6.8%) | 124.73 (+10.0%) | 214.37 (-1.7%) |
| wkv_tp4 | 52.48 (+347.4%) | 57.86 (+254.9%) | 59.60 (+108.5%) | 61.79 (+74.5%) | 66.13 (+22.5%) | 78.47 (-6.9%) | 112.48 (-26.4%) |
| wqkv_a_tp4 | 53.27 (+236.1%) | 58.60 (+141.8%) | 60.01 (+39.8%) | 63.07 (-2.4%) | 69.12 (-25.6%) | 126.30 (-5.1%) | 258.88 (+0.2%) |
| wo_a_tp4 | 40.99 (+199.2%) | 45.59 (+121.0%) | 48.90 (+22.0%) | 51.96 (-1.0%) | 58.21 (-41.6%) | 105.51 (-5.4%) | 216.77 (-1.7%) |
| shared_gate_up_tp4 | 53.02 (+244.2%) | 58.28 (+184.5%) | 60.13 (+69.1%) | 63.10 (+29.8%) | 68.97 (-3.0%) | 125.12 (+10.9%) | 217.60 (+1.5%) |
| wq_b_tp8 | 18.12 (+93.3%) | 20.70 (+69.3%) | 22.44 (+2.2%) | 25.75 (-19.5%) | 42.54 (+2.5%) | 73.31 (+1.9%) | 136.43 (-1.0%) |
| wo_b_tp8 | 15.37 (+55.4%) | 18.13 (+48.1%) | 20.02 (-12.0%) | 33.28 (+3.0%) | 47.68 (-6.0%) | 76.48 (-6.9%) | 142.00 (+1.1%) |
| wo_a_tp8 | 40.73 (+279.4%) | 45.56 (+208.2%) | 48.71 (+86.6%) | 51.60 (+21.3%) | 56.75 (-1.5%) | 69.51 (-41.0%) | 130.77 (-10.5%) |
| wq_b_tp1 | 20.29 (-1.3%) | 24.87 (-24.8%) | 72.79 (-5.1%) | 138.62 (+1.0%) | 277.34 (+6.5%) | 656.94 (+33.7%) | 1390.13 (+45.9%) |
| wo_b_tp1 | 79.15 (+174.0%) | 84.33 (+113.6%) | 88.34 (+1.9%) | 158.13 (+16.5%) | 236.18 (-11.6%) | 422.88 (-16.9%) | 1064.37 (+17.9%) |


##### Tile crossover and caller admission on the rerun matrix

These are descriptive summaries of this pass; dispatch defaults are unchanged.


| Wide-tile grid | Points | 256 faster | 128 faster | Mean 256/128 − 1 |
|---|---|---|---|---|
| 1–16 | 33 | 0 | 33 | +17.0% |
| 17–32 | 25 | 0 | 25 | +15.4% |
| 33–64 | 6 | 0 | 6 | +10.9% |
| 65–128 | 15 | 0 | 15 | +9.3% |
| 129–192 | 4 | 4 | 0 | -29.4% |
| >192 | 27 | 27 | 0 | -26.4% |


The gate table averages relative latency penalties over supported points above M=32, using the better measured MORI tile at each point.


| Grid threshold | Points | Served loss | Declined win | Total mean penalty |
|---|---|---|---|---|
| 48 | 77 | 0.05% | 0.00% | 0.05% |
| 64 | 77 | 0.05% | 0.03% | 0.08% |
| 80 | 77 | 0.02% | 0.15% | 0.18% |
| 128 | 77 | 0.00% | 0.88% | 0.88% |


##### GEMV: same FP8 input, cold µs

Each cell is MORI / SGLang, with the relative difference in parentheses.


| Shape | M=1 | M=8 | M=32 |
|---|---|---|---|
| wq_b | 4.00 / 4.22 (-5.3%) | 4.08 / 4.33 (-5.8%) | 5.32 / 5.60 (-5.0%) |
| wo_b | 3.94 / 4.52 (-12.8%) | 4.37 / 4.81 (-9.2%) | 6.31 / 6.91 (-8.6%) |
| wq_a_tp4 | 5.13 / 5.21 (-1.5%) | 5.58 / 5.61 (-0.4%) | 8.54 / 10.27 (-16.9%) |
| wkv_tp4 | 5.06 / 5.07 (-0.2%) | 5.50 / 5.53 (-0.6%) | 8.41 / 10.15 (-17.2%) |
| wqkv_a_tp4 | 4.81 / 4.78 (+0.6%) | 5.26 / 5.47 (-3.8%) | 8.22 / 10.10 (-18.6%) |
| wo_a_tp4 | 4.93 / 4.44 (+11.1%) | 5.57 / 5.11 (+8.9%) | 9.81 / 8.54 (+14.8%) |
| shared_gate_up_tp4 | 5.23 / 5.24 (-0.2%) | 5.56 / 5.69 (-2.3%) | 8.51 / 10.25 (-17.0%) |
| wq_b_tp8 | 3.62 / 3.58 (+1.2%) | 3.84 / 3.71 (+3.4%) | 4.62 / 4.56 (+1.5%) |
| wo_b_tp8 | 3.42 / 3.52 (-2.8%) | 3.78 / 3.78 (-0.1%) | 5.73 / 5.81 (-1.4%) |
| wo_a_tp8 | 4.94 / 4.60 (+7.4%) | 5.68 / 5.04 (+12.7%) | 9.90 / 8.49 (+16.6%) |
| wq_b_tp1 | 9.18 / 9.06 (+1.3%) | 10.26 / 9.76 (+5.2%) | 19.22 / 18.34 (+4.8%) |
| wo_b_tp1 | 11.10 / 11.76 (-5.7%) | 13.07 / 13.86 (-5.7%) | 27.37 / 29.33 (-6.7%) |


##### fused: operator timings


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


##### fused-fp8: operator timings


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

## Regression checklist

The matrices above are a test inventory, not a request to launch every sweep on
every change. Existing public-operator tests already cover scale contracts,
dynamic M, changing operands and cleanup; extend those checks to new options.

| Tier | When to run later | Coverage | State |
|---|---|---|---|
| Core correctness | Relevant code/API changes | Independent references, packing, B boundaries, S specialization, W serial lifecycle, F wire correctness | 待测 |
| Representative performance | Kernel/dispatch/transport changes | Tuned GEMV buckets, ragged M, awkward chunks, a large prefill control and a documented negative control | 待测 |
| Full performance | Before changing dispatch/defaults or a planned performance audit | L1, K1, V1 and the deduplicated A matrix | 待测 |
| Targeted diagnostics | Related mapping/publish/layout changes | C/P/V2/W/F/G/Q/T matrices; profiler counters when needed | 待测 |
| Historical prototype study | Only when its implementation is restored or reconsidered | CK-shaped/persistent and other archived prototypes | Historical-only |
| Model evaluation | Separate future scope | Server throughput and model quality | Deferred |

Before promoting an option or updating a result:

- [ ] Record the exact case ID, revisions, scope and effective configuration.
- [ ] Validate against the appropriate independent mathematical/wire reference.
- [ ] Exercise the relevant boundary, input-change, cache and lifecycle cases.
- [ ] Keep unsupported configurations and failed attempts distinct from passes.
- [ ] Report absolute latency and paired dispersion before interpreting a small Δ.
- [ ] Account separately for window requirements, actual memory and initialization.
- [ ] Preserve the previous result and its provenance; fill new cells only from
      matching measurements.

Future entry points (reference only; no tests are launched by this edit):

| Purpose | Entry point |
|---|---|
| Public API / lifecycle checks | [test_gemm_ar_op.py](../../../../tests/python/cco/test_gemm_ar_op.py) |
| Layout and kernel references | [test_gemm_ar.py](../../../../tests/python/cco/test_gemm_ar.py) |
| GEMV configuration grid | [test_mxfp8_gemv_grid.py](../../../../tests/python/cco/test_mxfp8_gemv_grid.py) |
| Kernel/linear GEMM measurements | [bench_gemm.py](../../../../benchmark/cco/flydsl/gemm_ar/bench_gemm.py) |
| GEMV measurements | [bench_gemv.py](../../../../benchmark/cco/flydsl/gemm_ar/bench_gemv.py) — pass the resolved configuration explicitly |
| Collective measurements | [bench_gemm_ar.py](../../../../benchmark/cco/flydsl/gemm_ar/bench_gemm_ar.py) |

For environment setup and additional flags, see
[the benchmark guide](../../../../docs/MORI-GEMM-AR-BENCHMARK.md). Historical
commands remain in their folds as provenance; use the current contracts and
measurement protocol when preparing a future run. Experimental scripts and raw
results remain outside the repository.
