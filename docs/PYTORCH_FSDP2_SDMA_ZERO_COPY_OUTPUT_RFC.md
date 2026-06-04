# Pre-RFC: SDMA AllGather and Zero-Copy Output for PyTorch FSDP2

Status: draft for discussion with PyTorch FSDP2 maintainers

Authors: AMD / ROCm MORI contributors

Related:

- MORI FSDP SDMA optimization notes: `docs/MORI_FSDP_SDMA_OPTIMIZATION.md`
- Prior DeepSpeed ZeRO-3 SDMA motivation: https://github.com/deepspeedai/DeepSpeed/issues/7884
- Review and pre-RFC analysis: https://gist.github.com/jeffdaily/703795f1eef12dc2eccd5775f06fdc81

## Summary

This pre-RFC proposes two separable FSDP2 improvements:

1. Add or support an SDMA-backed `AllGather` implementation for AMD GPUs. This replaces
   CU-backed RCCL allgather kernels with MORI SDMA allgather for FSDP2 parameter
   unsharding.
2. Add a backend-agnostic zero-copy output capability for scatter-capable allgather
   backends. Such a backend can write directly into a parameter-contiguous output layout,
   allowing FSDP2 to construct unsharded parameters as views and skip the current
   `split_with_sizes_copy` copy-out.

Stage 1 is a communication backend change and fits the existing FSDP2 custom allgather
shape. Stage 2 is a core FSDP2 capability: it generalizes the copy-out removal so that
MORI SDMA is one implementation, while other scatter-capable backends such as SymmMem or
NVSHMEM could opt in later.

Native NCCL/RCCL allgather is unaffected. Standard allgather writes contiguous rank-major
blocks and cannot directly produce the parameter-contiguous layout.

## Motivation

FSDP2 allgathers sharded parameters before forward and backward computation. For large
models, this path has two costs:

- Communication cost: the allgather itself.
- Local layout cost: after rank-major allgather, FSDP2 copies/splits the gathered buffer
  into per-parameter full tensors using `split_with_sizes_copy`.

On AMD GPUs, RCCL collectives run as GPU kernels and consume compute units (CUs). When
FSDP2 overlaps allgather with model compute, RCCL allgather can still compete with GEMM
or attention for CU resources. The same concern motivated the DeepSpeed ZeRO-3 SDMA RFC:
route large overlapped allgathers through SDMA copy engines so communication does not
consume CUs.

MORI adds a second opportunity. A scatter-capable allgather can write the final
parameter-contiguous layout directly:

```text
rank-major:
  [rank0 param0 param1 ...][rank1 param0 param1 ...]

parameter-contiguous:
  [param0 rank0 rank1 ...][param1 rank0 rank1 ...]
```

If the allgather output is already parameter-contiguous, FSDP2 can build each unsharded
parameter as a view of that buffer and skip `split_with_sizes_copy`.

## Measured Results

The current MORI prototype was evaluated with Qwen 7B, BF16, 8 GPUs, sequence length
1024, and micro-batch size 1 on a single MI300 node.

| Mode | Avg step time | Avg tokens/s | Avg TFLOPs/GPU | Gain vs native |
| --- | ---: | ---: | ---: | ---: |
| Native baseline (RCCL) | `0.436266 s` | `18777.53` | `107.25` | baseline |
| MORI SDMA base path | `0.400631 s` | `20447.76` | `116.79` | about `8.9%` |
| MORI SDMA + zero-copy output | `0.375366 s` | `21824.02` | `124.65` | about `16.2%` |

The fixed-seed final loss matches the native baseline:

```text
last_loss = 11.965459823608398
```

## Profiler Evidence

A short rank-0 PyTorch profiler trace separates the two effects. The traces were
collected after warmup using:

```bash
--steps 10 --warmup 10 --profile-start-step 2 --profile-steps 3 --profile-dir <trace-dir>
```

The numbers below are total event durations summed over the captured rank-0 profiler
window, not single-operation latency:

| Trace metric | Native baseline (RCCL) | MORI SDMA base path | MORI SDMA + zero-copy output |
| --- | ---: | ---: | ---: |
| Allgather communication kernel | `437.1 ms` `nccl:_all_gather_base` | `276.2 ms` `OneShotAllGatherSdmaKernel_u32` | `303.9 ms` `OneShotAllGatherSdmaParamContiguousKernel_u32` |
| `split_with_sizes_copy` CUDA kernel | `46.4 ms` | `50.2 ms` | `0.0 ms` |
| `FSDP::all_gather_copy_out` annotation | `104.7 ms` | `97.7 ms` | `28.6 ms` |

The first row shows that MORI SDMA replaces RCCL/NCCL allgather and reduces allgather
kernel time by about `36.8%` in this trace window. The second row shows that zero-copy
output removes the separate `split_with_sizes_copy` CUDA kernel. The third row shows
that the broader FSDP copy-out region shrinks from `97.7 ms` to `28.6 ms`; what remains
is mostly view construction and bookkeeping.

The zero-copy SDMA kernel is slightly longer than the base SDMA kernel because it writes
directly into the final multi-parameter layout. That extra layout work replaces the
separate copy-out kernel, which is why the zero-copy output path still improves
end-to-end step time.

## Current FSDP2 Extension Points

FSDP2 already has a communication hook shape that can support stage 1:

- `AllGather` / `Comm` abstraction in `torch/distributed/fsdp/_fully_shard/_fsdp_api.py`
- `set_custom_all_gather(comm: AllGather)` in the fully-shard API
- existing precedent from `SymmMemAllGather`

An SDMA-backed `AllGather` can plug into this path and still produce the normal
rank-major output. That gives the communication-side win while leaving
`foreach_all_gather_copy_out` unchanged.

The zero-copy output path does not fit the current interface. Today, FSDP2 assumes the
allgather output is rank-major and later calls `split_with_sizes_copy`. The comm has no
standard way to say "I produced the final parameter-contiguous layout; please construct
views and skip copy-out."

## Proposed Solution

### Stage 1: SDMA-backed AllGather backend

Introduce or allow an AMD SDMA-backed `AllGather` implementation. This can initially be
out-of-tree via `set_custom_all_gather`, or upstreamed as a backend similar in spirit to
`SymmMemAllGather`.

This stage should preserve existing FSDP2 behavior:

- input/output tensor contract remains rank-major;
- copy-out path remains unchanged;
- fallback to default allgather remains available;
- no FSDP core layout changes are required.

### Stage 2: backend-agnostic zero-copy output capability

Extend the `AllGather` contract so a backend can advertise that it can produce
parameter-contiguous output. FSDP2 would pass split metadata to the backend, and the
backend would write each rank's shard directly into the final layout.

Strawman API:

```python
class AllGather(Comm):
    # Default False: backend produces rank-major output and FSDP2 uses copy-out.
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

The exact API is open for discussion. The important contract is the capability, not the
specific method names.

When the capability and eligibility gate pass, FSDP2 constructs unsharded parameters as
views into the allgather output buffer instead of calling `split_with_sizes_copy`.

### Prototype runtime controls

The current MORI prototype uses environment variables to keep the optimized path fully
opt-in:

```bash
MORI_ENABLE_SDMA=1
MORI_FSDP_ENABLE_SDMA=1
MORI_FSDP_ZERO_COPY_OUTPUT=1
```

`MORI_FSDP_ENABLE_SDMA=1` selects the MORI FSDP2 allgather backend.
`MORI_FSDP_ZERO_COPY_OUTPUT=1` additionally enables the parameter-contiguous zero-copy
output path when the eligibility gate passes. Without the zero-copy variable, MORI still
uses SDMA allgather but falls back to FSDP2's existing rank-major output plus copy-out
layout.

## Layout and Offset Formula

For each parameter split, the scatter-capable allgather writes each local rank shard to:

```text
output_offset = split_offset * world_size + rank * split_size
```

This produces:

```text
[param0 rank0 rank1 ...][param1 rank0 rank1 ...]
```

After allgather completes, FSDP2 can use cheap `narrow()` views into the shared output
buffer for each unsharded parameter. The views must preserve `storage_offset()`, since
non-first parameters start in the middle of the shared output buffer.

## Eligibility Gate

The zero-copy output path should be conservative initially. It should only be used when
the layout formula exactly matches FSDP2's parameter view construction.

Initial gate:

- multiple parameters in the group;
- dim-0 sharding for every parameter;
- no DTensor parameters;
- no custom `fsdp_post_all_gather`;
- exactly one allgather input per parameter;
- no compiled autograd / Traceable FSDP2 until trace safety is proven;
- mixed dtype, fp8, or `fsdp_pre_all_gather` cases gated out until byte-offset math is
  explicitly tested;
- post-forward mesh reshard modes gated out until lifetime and storage transitions are
  tested.

Groups that fail the gate still use the backend's normal allgather path and the existing
rank-major copy-out.

## Buffer Ownership and Lifetime

The zero-copy output path changes buffer ownership, not the fact that FSDP2 already
needs an allgather output tensor.

Default path:

```text
allgather output -> split_with_sizes_copy -> per-parameter full buffers
```

After copy-out, the allgather output is temporary and may be reused.

Zero-copy path:

```text
allgather output -> parameter views
```

The allgather output becomes parameter-backing storage and must remain live until the
parameter group is resharded.

The current MORI integration creates one `MoriSdmaAllGather` object per
`FSDPParamGroup`, and each object owns its own registered output buffer. In the current
layer-by-layer FSDP2 setup, each parameter group/layer has an independent registered
output buffer, so there is no cross-layer overwrite risk.

The upstream contract should be stated independently of MORI:

```text
concurrently unsharded FSDP parameter groups must not reuse the same allgather output buffer
```

Sharing one communication object across groups is only one way to violate this contract.
An allocator that returns the same output buffer to multiple live groups would have the
same problem. With forward prefetch, a no-copy comm needs at least `prefetch_depth + 1`
distinct output buffers. With `reshard_after_forward=False`, it may need one per FSDP
parameter group in the module.

## Memory Tradeoff

The zero-copy path does not allocate an extra shadow output buffer on top of the FSDP2
allgather output. It registers and reuses the allgather output tensor that FSDP2 already
needs.

However, because that output becomes parameter-backing storage, its lifetime extends
from "until copy-out completes" to "until reshard". The peak number of live registered
output buffers is therefore tied to how many parameter groups can be concurrently
unsharded. Upstream should decide whether FSDP owns this lifetime ceiling or whether
each backend owns its registered buffer pool.

## Correctness and Test Plan

Before enabling this upstream, the current gates and lifetime contract should be turned
into explicit tests:

- bit-exact unsharded parameter data versus the existing copy-out path;
- full train-step loss parity;
- uneven dim-0 sharding with tail padding;
- buffer multiplicity and aliasing under forward prefetch;
- buffer multiplicity and aliasing under `reshard_after_forward=False`;
- gate fall-through for DTensor, non-dim-0 sharding, custom post-allgather, multi-input,
  mixed dtype or fp8 pre-allgather, post-forward mesh reshard, and `torch.compile` /
  Traceable FSDP2.

The fallback path should be tested as carefully as the hit path. Unsupported groups must
reliably fall back to the existing rank-major copy-out path.

## Scope and Limitations

In scope:

- FSDP2 parameter allgather during unshard.
- Single-node AMD GPU systems where SDMA can move data over the local GPU fabric.
- Scatter-capable allgather backends that can write non-rank-major output layouts.
- A conservative zero-copy output gate with reliable fallback.

Future extensions:

- Multi-node communication, possibly combining SDMA intra-node transport with an
  inter-node transport.
- Reduce-scatter or other FSDP collectives.
- Additional scatter-capable backends beyond MORI SDMA.
- `torch.compile` / Traceable FSDP2 support once view construction and lifetime are
  validated.

Limitations:

- Stage 1 is AMD-specific because it depends on AMD SDMA engines and MORI's SDMA
  transport.
- Stage 2 is not AMD-specific, but it only helps backends that can scatter-write into
  the parameter-contiguous layout.
- The measured gain is from one Qwen 7B BF16 single-node MI300 configuration. The gain is
  expected to be strongest when FSDP allgather overlaps useful compute on the same node.
  Multi-node results may differ as network communication becomes a larger fraction of
  step time.

## Alternatives Considered

Tune RCCL channel or CU usage:

RCCL tuning may reduce CU contention, but it trades communication bandwidth for compute
availability. SDMA targets the root cause by moving communication work to copy engines.

Keep the communication backend change only:

This is useful and should remain stage 1, but it leaves the FSDP2 copy-out cost in place.
The profiler evidence shows that removing `split_with_sizes_copy` is a separate source
of speedup.

Add a MORI-only FSDP hook:

This would work for the prototype but is not the right upstream shape. The copy-out-free
layout should be a backend capability so other scatter-capable backends can use it.

Always allocate per-parameter full buffers:

This is the current safe default. It decouples allgather output lifetime from parameter
lifetime, but it requires the extra copy-out and loses the zero-copy output benefit.

## Compatibility and Fallback

The proposed path is opt-in. Existing FSDP2 users continue to use the current allgather
and copy-out path by default.

Fallback is required at two levels:

- If MORI SDMA is unavailable, use the default FSDP2 allgather backend.
- If a parameter group fails the zero-copy output gate, use the backend's rank-major
  allgather output and the existing `split_with_sizes_copy` copy-out.

The zero-copy capability should not change public FSDP2 semantics. It changes the
internal storage backing of unsharded parameters for eligible groups, so storage aliasing,
version counters, and compile behavior must be explicitly tested before broad enablement.

## Non-Goals

This proposal does not require changing native NCCL/RCCL allgather. Native allgather
writes rank-major blocks and cannot directly produce the parameter-contiguous layout.

This proposal also does not require making MORI the only user of the zero-copy output
capability. The core FSDP2 API should be backend-agnostic so other scatter-capable
backends can use the same hook.

## References

- DeepSpeed SDMA ZeRO-3 RFC: https://github.com/deepspeedai/DeepSpeed/issues/7884
- MORI FSDP SDMA optimization notes:
  https://github.com/ROCm/mori/blob/fsdp-ccl/docs/MORI_FSDP_SDMA_OPTIMIZATION.md
