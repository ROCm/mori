# RFC: SDMA AllGather and Zero-Copy Output for PyTorch FSDP2

- RFC Status: Draft
- Target Area: `torch.distributed.fsdp` (FSDP2 all-gather unshard path)
- Authors: AMD / ROCm MORI contributors
- Discussion: TBD (PyTorch RFC thread)
- Last Updated: 2026-06-08

## Summary

FSDP2 unshard all-gather currently pays for both communication and local layout copy-out.
Under overlap, RCCL collectives consume CUs and contend with model compute.

This RFC proposes two separable improvements:

1. **SDMA-backed AllGather backend (AMD-specific implementation)** for overlapped communication without consuming compute units (CUs) as RCCL collectives do.
2. **Backend-agnostic zero-copy output capability** for scatter-capable all-gather implementations, allowing direct write into parameter-contiguous layout and skipping `split_with_sizes_copy`.

Stage 1 is a communication backend substitution within existing extension points. Stage 2 is a generic FSDP2 capability that MORI SDMA can use first, but other scatter-capable backends (for example SymmMem or NVSHMEM-style implementations) may adopt later.

Native NCCL/RCCL behavior remains unchanged by default.

### Core Value Proposition: 0-CU Communication for Overlap

The primary value of Stage 1 is not only "faster all-gather", but **removing CU consumption by communication during overlap**:

- RCCL all-gather executes GPU kernels and consumes CUs.
- SDMA all-gather submits copy/atomic work to copy engines, targeting **zero CU occupancy for communication work**.
- Under FSDP2 prefetch overlap, this shifts the bottleneck from CU contention to transport/memory limits and preserves more CU capacity for GEMM/attention.

This is the same first-principles argument used in DeepSpeed ZeRO-3 SDMA discussions, and it is especially relevant when all-gather payloads are large and heavily overlapped.

### Two-Stage Plan at a Glance

| Stage | Scope | Upstream impact | Backend specificity |
| --- | --- | --- | --- |
| Stage 1 | SDMA-backed all-gather backend with existing rank-major output | Low (uses current custom all-gather extension points) | AMD-focused implementation |
| Stage 2 | Zero-copy parameter-contiguous output capability in FSDP2 API | Medium (new backend capability contract + lifetime semantics) | Backend-agnostic capability |

## Motivation

### Problem statement

FSDP2 unsharding all-gathers parameter shards before forward/backward. For large models, two costs are significant:

- **Communication cost** (all-gather itself)
- **Layout transformation cost** (`split_with_sizes_copy` from rank-major buffer to per-parameter full tensors)

On AMD systems, RCCL collective kernels consume CUs and can contend with GEMM/attention under overlap. SDMA communication engines avoid this CU contention for suitable intra-node paths.

### Why "0-CU" Matters More Than Raw Collective Latency

In highly overlapped training (prefetch all-gather + compute), end-to-end step time depends on **resource interference**, not standalone collective latency alone:

- A lower-latency collective that still consumes CUs can reduce GEMM throughput during overlap.
- A slightly slower collective with near-zero CU usage can still improve step time by preserving compute throughput.

Therefore this RFC treats **communication isolation from CUs** as the main optimization target of Stage 1.

Additionally, if a backend can directly produce **parameter-contiguous** output layout, FSDP2 can construct unsharded tensors as views and remove copy-out kernels.

```text
rank-major:
  [rank0 param0 param1 ...][rank1 param0 param1 ...]

parameter-contiguous:
  [param0 rank0 rank1 ...][param1 rank0 rank1 ...]
```

## User-Facing Behavior

- By default, nothing changes.
- Users may opt in to a custom all-gather backend (for example MORI SDMA).
- If backend and parameter group satisfy eligibility gates, zero-copy output path is used; otherwise FSDP2 falls back to current copy-out behavior.

## Prototype Runtime Controls (MORI)

```bash
MORI_ENABLE_SDMA=1
MORI_FSDP_ENABLE_SDMA=1
MORI_FSDP_ZERO_COPY_OUTPUT=1
```

`MORI_FSDP_ENABLE_SDMA=1` enables MORI all-gather backend for FSDP2.
`MORI_FSDP_ZERO_COPY_OUTPUT=1` enables zero-copy output when eligibility checks pass.

## Proposed Solution

Two-stage rollout is intentional:

- Stage 1 is independently useful and can land with minimal FSDP2 core risk.
- Stage 2 introduces API and storage-lifetime surface, so it is capability-gated and lands conservatively.

### Stage 1: SDMA-backed AllGather backend

Use existing FSDP2 extension points:

- `AllGather` / `Comm` abstractions in FSDP2 API
- `set_custom_all_gather(comm: AllGather)`
- Existing precedent from `SymmMemAllGather`

Contract remains unchanged:

- Rank-major all-gather output
- Existing `foreach_all_gather_copy_out` path retained
- Existing fallback retained

### Stage 2: Backend-agnostic zero-copy output capability

Introduce capability signaling in `AllGather` contract so backend can declare:

- it supports parameter-contiguous output
- it accepts split metadata
- it can return metadata needed for view construction

Strawman API sketch:

```python
class AllGather(Comm):
    supports_param_contiguous_output: bool = False

    def prepare_param_contiguous_output(
        self,
        input_split_sizes: list[int],
        *,
        element_size: int,
        device: torch.device,
    ) -> bool:
        ...

    def param_contiguous_metadata(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        ...
```

API names are illustrative; the key requirement is the capability contract.

#### Layout formula

For split `i`, backend places local rank shard at:

```text
output_offset = split_offset * world_size + rank * split_size
```

This yields:

```text
[param0 rank0 rank1 ...][param1 rank0 rank1 ...]
```

FSDP2 constructs parameter views using `narrow()` with correct `storage_offset()`.

#### Eligibility gate (initial conservative policy)

Zero-copy output is enabled only when all conditions hold:

- Multiple parameters in group
- Dim-0 sharding for all parameters
- No DTensor parameters
- No custom `fsdp_post_all_gather`
- Exactly one all-gather input per parameter
- No compiled autograd / Traceable FSDP2
- Mixed dtype, fp8, or `fsdp_pre_all_gather` cases excluded until explicitly validated
- Post-forward mesh-reshard modes excluded until storage lifetime behavior is validated

Any failure falls back to rank-major + copy-out.

#### Buffer ownership and lifetime contract

Default path:

```text
allgather output -> split_with_sizes_copy -> per-parameter full buffers
```

Zero-copy path:

```text
allgather output -> parameter views
```

Therefore, all-gather output becomes parameter-backing storage and must remain alive until reshard. Upstream contract:

```text
Concurrently unsharded FSDP parameter groups must not reuse the same all-gather output buffer.
```

With prefetch, no-copy communication needs enough distinct buffers (for example `prefetch_depth + 1`; potentially one per live group under `reshard_after_forward=False`).

## Performance Data (Prototype)

Configuration: Qwen 7B, BF16, 8 GPUs, seq 1024, micro-batch 1, single MI300 node.

| Mode | Avg step time | Avg tokens/s | Avg TFLOPs/GPU | Gain vs native |
| --- | ---: | ---: | ---: | ---: |
| Native baseline (RCCL) | `0.436266 s` | `18777.53` | `107.25` | baseline |
| MORI SDMA base path | `0.400631 s` | `20447.76` | `116.79` | `+8.9%` |
| MORI SDMA + zero-copy output | `0.375366 s` | `21824.02` | `124.65` | `+16.2%` |

Fixed-seed final loss parity:

```text
last_loss = 11.965459823608398
```

Profiler (rank 0, overlap case, summed durations):

| Trace metric | Native baseline (RCCL) | MORI SDMA base path | MORI SDMA + zero-copy output |
| --- | ---: | ---: | ---: |
| All-gather communication kernel | `437.1 ms` `nccl:_all_gather_base` | `276.2 ms` `OneShotAllGatherSdmaKernel_u32` | `303.9 ms` `OneShotAllGatherSdmaParamContiguousKernel_u32` |
| `split_with_sizes_copy` kernel | `46.4 ms` | `50.2 ms` | `0.0 ms` |
| `FSDP::all_gather_copy_out` annotation | `104.7 ms` | `97.7 ms` | `28.6 ms` |

Interpretation: zero-copy kernel may run longer than base SDMA due to layout work, but removes separate copy-out and improves end-to-end step time.

### Stage 1 vs Stage 2 Contribution Framing

- **Stage 1 (SDMA, 0-CU communication):** primarily improves comm/compute overlap efficiency by reducing CU interference from collectives.
- **Stage 2 (zero-copy output):** primarily removes local layout-copy overhead (`split_with_sizes_copy`) and shrinks FSDP copy-out region.

Keeping this decomposition explicit helps performance diagnosis and rollout decisions.

## Scope and Limitations

In scope:

- FSDP2 parameter all-gather during unshard.
- Single-node AMD GPU systems where SDMA can move data over local GPU fabric.
- Scatter-capable all-gather backends that can write non-rank-major output layouts.
- Conservative zero-copy eligibility gate with reliable fallback.

Future extensions:

- Multi-node communication (e.g., SDMA intra-node plus inter-node transport).
- Additional FSDP collectives such as reduce-scatter.
- Additional scatter-capable backends beyond MORI.
- `torch.compile` / Traceable FSDP2 support after explicit validation.

Limitations:

- Stage 1 implementation is AMD-specific.
- Stage 2 is backend-agnostic but only useful for scatter-capable all-gather backends.
- Performance data currently reflects one single-node MI300 workload; multi-node behavior may differ.

## Alternatives Considered

### Why not only Stage 1

Stage 1 is useful and should land independently, but it does not remove FSDP copy-out overhead.

### Why not a MORI-only hook

Upstream shape should be backend-agnostic. Capability-based design allows additional scatter-capable backends to participate.

### Why not keep per-parameter full-buffer materialization always

Current behavior is safe and simple, but incurs extra copy-out cost and misses zero-copy benefit.

## References

- MORI FSDP SDMA optimization notes: https://github.com/ROCm/mori/blob/fsdp-ccl/docs/MORI_FSDP_SDMA_OPTIMIZATION.md
- DeepSpeed SDMA ZeRO-3 motivation: https://github.com/deepspeedai/DeepSpeed/issues/7884
- Pre-RFC external review notes: https://gist.github.com/jeffdaily/703795f1eef12dc2eccd5775f06fdc81
