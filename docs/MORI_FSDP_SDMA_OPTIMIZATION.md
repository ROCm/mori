# MORI FSDP SDMA Allgather Optimization

This note summarizes the MORI SDMA allgather optimization for PyTorch FSDP2 and the
Qwen 7B benchmark results used to evaluate it. The focus is the end-to-end impact of
replacing the native RCCL allgather plus FSDP copy-out path with MORI SDMA and
zero-copy output.

## Background

FSDP2 frequently allgathers sharded parameters before forward and backward computation.
For large models, this path includes both inter-GPU communication and local tensor
layout work. The local layout work becomes noticeable for multi-parameter FSDP groups:
after the rank-major allgather completes, FSDP still has to split and copy the gathered
buffer into per-parameter full tensors.

The MORI integration reduces this overhead in stages:

1. Use MORI SDMA for the allgather communication path.
2. Register direct output buffers so SDMA can write into user-visible GPU memory.
3. Skip input copies when a single contiguous input can be used directly.
4. Add a zero-copy output path backed by a param-contiguous multi-parameter layout.

## Why SDMA Allgather Helps

On AMD GPUs, RCCL allgather uses GPU kernels that run on compute units (CUs), so an
overlapped allgather can still compete with GEMM or attention for compute resources. MORI
SDMA allgather routes the bulk GPU-to-GPU transfer through SDMA copy engines instead,
giving the communication path a zero-CU implementation that is especially useful when
allgather overlaps with model compute. This follows the same resource-overlap motivation
as prior DeepSpeed ZeRO-3 SDMA allgather work; see
[SDMA-Accelerated ZeRO-3 on AMD GPUs](https://github.com/deepspeedai/DeepSpeed/blob/master/examples/sdma_allgather/README.md)
for related context.

## Runtime Modes

The benchmarks compare three execution modes. The native mode establishes the RCCL
baseline, the MORI SDMA base path isolates the communication replacement, and the
zero-copy output mode measures the output no-copy, copy-out-free path.

Native baseline (RCCL):

```bash
torchrun --nproc_per_node=8 examples/fsdp/bench_qwen7b_allgather.py \
  --mode native \
  --seq-len 1024 \
  --steps 2000 \
  --warmup 5
```

MORI SDMA base path:

```bash
MORI_ENABLE_SDMA=1 \
MORI_FSDP_ENABLE_SDMA=1 \
torchrun --nproc_per_node=8 examples/fsdp/bench_qwen7b_allgather.py \
  --mode mori \
  --seq-len 1024 \
  --steps 2000 \
  --warmup 5
```

MORI SDMA with zero-copy output using the param-contiguous layout:

```bash
MORI_ENABLE_SDMA=1 \
MORI_FSDP_ENABLE_SDMA=1 \
MORI_FSDP_ZERO_COPY_OUTPUT=1 \
torchrun --nproc_per_node=8 examples/fsdp/bench_qwen7b_allgather.py \
  --mode mori \
  --seq-len 1024 \
  --steps 2000 \
  --warmup 5
```

`MORI_FSDP_ENABLE_SDMA=1` selects the FSDP2 MORI allgather path, while
`MORI_ENABLE_SDMA=1` enables MORI's SDMA allocations and transport path. The benchmark
script sets `MORI_FSDP_ENABLE_SDMA=1` automatically for `--mode mori`, but manual
integration should set both variables explicitly. `MORI_FSDP_ZERO_COPY_OUTPUT=1` is an
additional opt-in: without it, MORI still uses SDMA allgather, but multi-parameter FSDP
groups fall back to the safer rank-major output plus copy-out layout path.

## Benchmark Results

The headline comparison is between the native RCCL baseline and MORI SDMA with
zero-copy output enabled. The run was measured on MI300 using Qwen 7B,
BF16, 8 GPUs, sequence length 1024, and micro-batch size 1.

First, the fixed-seed native (RCCL) and MORI loss curves overlap exactly:

![Qwen 7B FSDP loss comparison](fsdp_loss_compare_qwen7b_mori_sdma.svg)

This figure is the main correctness signal for the benchmark. With the same seed and
batch sequence, the optimized MORI path follows the native RCCL baseline without visible
loss drift over the full benchmark window.

The step-time comparison then shows the end-to-end speedup:

![Qwen 7B FSDP step time comparison](fsdp_step_time_compare_qwen7b.svg)

This figure shows that the MORI SDMA base path lowers per-step latency, and the
zero-copy output path adds a second improvement by removing the FSDP copy-out.
Average step time drops from `0.436266 s` to `0.400631 s` with SDMA alone, and to
`0.375366 s` with zero-copy output enabled.

The TFLOPs/GPU comparison shows the same improvement from the throughput side:

![Qwen 7B FSDP TFLOPs per GPU comparison](fsdp_tflops_per_gpu_compare_qwen7b.svg)

This figure shows that lower communication and layout overhead translate into higher GPU
efficiency, increasing average throughput from `107.25` to `116.79` TFLOPs/GPU with SDMA
alone, and to `124.65` TFLOPs/GPU with zero-copy output enabled.

| Mode | Avg step time | Avg tokens/s | Avg TFLOPs | Avg TFLOPs/GPU | Final loss | Gain vs native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native baseline (RCCL) | `0.436266 s` | `18777.53` | `858.01` | `107.25` | `11.965459823608398` | baseline |
| MORI SDMA base path | `0.400631 s` | `20447.76` | `934.33` | `116.79` | `11.965459823608398` | about `8.9%` |
| MORI SDMA + zero-copy output | `0.375366 s` | `21824.02` | `997.22` | `124.65` | `11.965459823608398` | about `16.2%` |

The result is a roughly `16.2%` end-to-end speedup while preserving the final loss for
the fixed-seed benchmark.

These numbers are from one Qwen 7B BF16 single-node MI300 configuration. The SDMA base
path isolates the zero-CU communication change, while the zero-copy output mode adds the
copy-out-free layout change. The absolute gain is workload-dependent and is expected to
be strongest when FSDP allgather overlaps useful compute on the same node.

## Optimization Progression

The full gain comes from two layers of optimization. The base MORI SDMA path improves
the communication side, while the zero-copy output path modifies the SDMA allgather
operator so it can write directly into the multi-parameter param-contiguous layout,
removing the remaining copy-out/layout transform after allgather.

| Mode | Environment | Main optimization | Avg step time | Gain vs native |
| --- | --- | --- | ---: | ---: |
| Native baseline (RCCL) | unset | RCCL allgather with standard FSDP copy-out | `0.436266 s` | baseline |
| MORI SDMA base path | `MORI_ENABLE_SDMA=1` + `MORI_FSDP_ENABLE_SDMA=1` | SDMA allgather with registered output buffers and reduced staging overhead | `0.400631 s` | about `8.9%` |
| MORI SDMA + zero-copy output | `MORI_ENABLE_SDMA=1` + `MORI_FSDP_ENABLE_SDMA=1` + `MORI_FSDP_ZERO_COPY_OUTPUT=1` | SDMA allgather operator writes directly into multi-parameter param-contiguous output, removing copy-out | `0.375366 s` | about `16.2%` |

In short, the optimization path is:

```text
Native RCCL baseline
  -> MORI_ENABLE_SDMA=1 + MORI_FSDP_ENABLE_SDMA=1
     -> faster SDMA allgather path, about 8.9% gain
  -> MORI_ENABLE_SDMA=1 + MORI_FSDP_ENABLE_SDMA=1 + MORI_FSDP_ZERO_COPY_OUTPUT=1
     -> SDMA allgather plus zero-copy output, about 16.2% gain
```

For upstream discussion, these should be framed as two stages. Stage 1 is the MORI SDMA
allgather backend, which fits the existing FSDP2 `AllGather` communication hook and keeps
the normal rank-major copy-out path. Stage 2 is the backend-agnostic zero-copy output
capability, implemented here with a param-contiguous layout, where a scatter-capable
backend writes the final layout directly and FSDP2 skips `split_with_sizes_copy`.

Stage 2 should not be pitched as a MORI-only hook. The upstreamable shape is an
`AllGather` capability: a backend advertises that it can produce param-contiguous output,
FSDP2 passes split metadata, and FSDP2 constructs parameter views instead of copying out.
MORI SDMA is the concrete implementation used for this benchmark, but the same core
capability could also be used by other scatter-capable backends such as SymmMem or
NVSHMEM. Native NCCL/RCCL allgather is unaffected because it writes contiguous rank-major
blocks.

## Implementation and Operator-Level Changes

The optimization has two parts: FSDP2 integration and a matching SDMA allgather operator
layout. FSDP2 decides when the MORI path can be used, and the SDMA operator writes the
allgather result into the layout that FSDP2 can consume without a copy-out step.

### FSDP2 Integration

In the FSDP2 `fully_shard` path, the integration point is the parameter group's
`AllGather` communication object. During parameter-group initialization, the default
`dist.all_gather_into_tensor` wrapper is replaced by `MoriSdmaAllGather` when the MORI
FSDP SDMA path is enabled:

```python
self._all_gather_comm: AllGather = (
    MoriSdmaAllGather() if is_mori_fsdp_sdma_enabled() else DefaultAllGather()
)
```

After that, the normal FSDP2 unshard flow still calls `foreach_all_gather()`, but the
communication primitive behind the call is MORI SDMA. This keeps the integration local
to the FSDP2 collective path instead of changing the higher-level module hooks:

```python
all_gather_work = all_gather_comm(
    output_tensor=all_gather_output,
    input_tensor=all_gather_input,
    group=group,
    async_op=async_op,
)
```

Inside `foreach_all_gather()`, FSDP2 first flattens the per-parameter allgather inputs
and records their split sizes. Those split sizes are the metadata MORI needs to produce
param-contiguous output:

```python
inp_split_sizes = [t.numel() for t in all_gather_inputs]
param_contiguous_output = _can_use_param_contiguous_all_gather_output(...)
if param_contiguous_output:
    param_contiguous_output = all_gather_comm.prepare_param_contiguous_output(
        inp_split_sizes,
        element_size=torch.empty((), dtype=dtype).element_size(),
        device=device,
    )
```

The integration adds three optimization points. First, MORI allocates and reuses a
registered GPU output buffer so SDMA can write directly into user-visible memory. Second,
for a single contiguous allgather input tensor, FSDP2 can skip the copy-in staging path
and pass the original input tensor directly to MORI. Third, for eligible multi-parameter
groups, FSDP2 passes per-parameter split sizes and offsets to MORI, enabling
param-contiguous output and skipping the normal `split_with_sizes_copy` copy-out. The
multi-parameter param-contiguous path is an output no-copy optimization: it removes the
FSDP copy-out/layout transform, while still packing multiple local input splits into one
contiguous allgather input buffer before communication.

The param-contiguous path is intentionally gated. It is used only for FSDP2 parameter
groups that have multiple parameters, shard on dim 0, do not use DTensor, do not define
custom `fsdp_post_all_gather`, have one allgather input per parameter, and are outside
compiled autograd:

```python
if compiled_autograd_enabled():
    return False
if len(fsdp_params) <= 1:
    return False
for fsdp_param, input_numels, input_dtypes in zip(...):
    if fsdp_param.fsdp_placement.dim != 0:
        return False
    if fsdp_param.is_dtensor:
        return False
    if hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather"):
        return False
    if len(input_numels) != 1 or len(input_dtypes) != 1:
        return False
```

Groups that do not satisfy these conditions can still use MORI SDMA allgather, but they
fall back to the regular rank-major output layout and FSDP2 copy-out path.

The gate is needed because the direct-write path turns the SDMA output buffer into the
backing storage for FSDP2 unsharded parameter views. That is only safe when each
parameter maps to one contiguous input split and the final `[param][rank]` layout matches
FSDP2's view construction. If a parameter uses a different shard dimension, DTensor
placement, custom post-allgather hook, or multiple allgather inputs, the same offset
formula may no longer describe the final parameter layout. In those cases, falling back
to the regular copy-out path preserves correctness.

For the Qwen 7B benchmark, the gate fully matches the observed FSDP2 allgather pattern:
every measured FSDP2 allgather output used the param-contiguous output no-copy path.

For upstream, the initial gate should also explicitly exclude mixed-dtype or fp8
`fsdp_pre_all_gather` cases until byte-offset math is tested, post-forward mesh reshard
modes that move the parameter to a different sharded buffer, and `torch.compile` /
Traceable FSDP2 until the view-based construction is proven trace-safe.

### Buffer Ownership and Lifetime

The output no-copy path changes the lifetime of the allgather output buffer, but it does
not add a second output buffer. FSDP2 already needs an allgather output tensor. MORI uses
that same output-buffer slot, registers the tensor for SDMA, and caches it for reuse by
the same communication object. The optimization removes the later FSDP copy-out into
separate per-parameter output tensors; it does not allocate an additional shadow output
buffer.

The lifetime changes because zero-copy output makes the allgather output tensor
parameter-backing storage. In the default FSDP2 path, the rank-major allgather output is
temporary and can be reused after `split_with_sizes_copy` finishes. In the
param-contiguous path, the unsharded parameters are views of the MORI output buffer, so
that buffer must remain live until the parameter group is resharded.

The current MORI integration preserves this ownership naturally. It creates one
`MoriSdmaAllGather` object per `FSDPParamGroup`, and each object owns its own registered
output buffer. In the current layer-by-layer FSDP2 setup, each parameter group/layer
therefore has an independent registered output buffer. Forward prefetching another layer
writes into that other layer's buffer, not into the buffer backing the current layer's
live parameter views, so the current FSDP2 mechanism does not have a cross-layer
overwrite risk.

The contract to preserve in future refactors or upstream implementations is that
concurrently unsharded FSDP parameter groups must not reuse the same allgather output
buffer. Sharing one `MoriSdmaAllGather` instance across parameter groups through
`set_custom_all_gather()` is only one way to violate that contract; an upstream comm
allocator that returns the same output buffer to multiple live groups would have the
same problem. With forward prefetch, a no-copy comm needs at least `prefetch_depth + 1`
distinct output buffers; with `reshard_after_forward=False`, it can need one per FSDP
parameter group in the module. A production version should document or guard this
contract and test that concurrently unsharded groups do not share the same output buffer.

### SDMA AG Operator-Level Changes

At the operator level, the param-contiguous path changes where each rank's shard is
written. Instead of first producing a rank-major buffer and then asking FSDP to copy it
into per-parameter outputs, MORI uses the split metadata to write each parameter shard
directly into the final param-contiguous layout.

The output layout changes from:

```text
rank-major:
  [rank0 param0 param1 ...][rank1 param0 param1 ...]

param-contiguous:
  [param0 rank0 rank1 ...][param1 rank0 rank1 ...]
```

For each parameter split, the SDMA kernel computes the destination offset as:

```text
output_offset = split_offset * world_size + rank * split_size
```

This means every rank writes its local shard into the correct per-parameter block on
every peer. After the allgather completes, FSDP2 can create each full parameter as a
cheap `narrow()` view of the MORI output buffer, so the normal `split_with_sizes_copy`
copy-out is skipped. Each view keeps its `storage_offset()`, so non-first parameters
still point to the right place inside the shared output buffer.

This turns the original sequence:

```text
allgather -> split_with_sizes_copy -> per-parameter full buffers
```

into:

```text
SDMA allgather directly into param-contiguous layout -> per-parameter views
```

This is the mechanism behind the measured speedup: the optimized path reduces both copy
bandwidth and launch/layout overhead while keeping the parameter views aligned with the
shared allgather output buffer.

## Open Items for Upstreaming

Before proposing the zero-copy output capability upstream, the remaining work is to turn
the current param-contiguous layout gates and lifetime contract into explicit tests:

- Buffer multiplicity and aliasing under forward prefetch and under
  `reshard_after_forward=False`.
- Bit-exact unsharded parameter data and full train-step loss parity against the copy-out
  path.
- Uneven dim-0 sharding with tail padding.
- Gate fall-through for DTensor, non-dim-0 sharding, custom post-allgather, multi-input,
  mixed dtype or fp8 pre-allgather, post-forward mesh reshard, and `torch.compile` /
  Traceable FSDP2.

A profiling trace that separately shows the SDMA base-path gain and the removed
`split_with_sizes_copy` cost would also make the RFC stronger. The main open design
questions are whether the core API should use a capability flag or a distinct
`ParamContiguousAllGather` subtype, whether FSDP or the comm should own the registered
buffer pool and lifetime ceiling, whether the idea should later extend to reduce-scatter,
and whether a minimum size threshold is needed when the extra registered-buffer memory is
not worth the copy-out savings.
