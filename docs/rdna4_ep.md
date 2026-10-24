# gfx1201 EP2 / EP4

MORI provides an opt-in transport for ordinary IntraNode Dispatch/Combine on
2 or 4 gfx1201 GPUs. It supports BF16/FP16 payloads and every integer top-k
from 1 through 64, including top-k 32 on wave32 hardware.

Dispatch sends one hidden payload per destination rank, deduplicating experts
on that rank. Combine returns each rank's contribution and sums in FP32 before
converting to the payload dtype. The caller supplies expert computation between
Dispatch and Combine; these operators do not implement a complete MoE layer.

## Supported configuration

| Setting | Requirement |
| --- | --- |
| GPU architecture | gfx1201 |
| API | `mori.ops.EpDispatchCombineOp` |
| Topology | One node, `world_size=gpu_per_node=2` or `4` |
| Kernel type | `EpDispatchCombineKernelType.IntraNode` |
| Payload | `torch.bfloat16` or `torch.float16`, `max_token_type_size=2` |
| Routing | `1 <= num_experts_per_token <= 64` |
| Hidden dimension | 2048 through 8192, divisible by 8 |
| Quantization | `quant_type="none"`, `scale_dim=0` |
| Input buffer | `use_external_inp_buf=True` |
| Capacity | Positive `max_num_inp_token_per_rank`; leave `max_total_recv_tokens=0` for full receive capacity |
| Communication | SHMEM with a static heap |

Use contiguous inputs, int32 expert indices and optional FP32 weights. Keep
capacity, hidden width, top-k and dtype consistent across ranks. Hidden width,
top-k and dtype must remain fixed for an operator; construct a new operator to
change them.

The ordinary `dispatch()` / `combine()` API supports empty ranks, unequal token
counts, routing handles and replay. Routing handles require positive source
token counts; empty-rank cases use the ordinary API without a handle. External
Combine input may alias Dispatch output. Standard-MoE, quantization,
`use_external_inp_buf=False`, IntraNodeLL, internode and pipelined Combine are
outside this transport's scope.

## Build and enable

Use a ROCm/PyTorch environment supporting gfx1201 and follow the
[installation guide](installation.md) to build this source checkout. Set
`MORI_GPU_ARCHS=gfx1201` during installation. The native library and JIT sources
must come from the same checkout. The transport has been exercised on Radeon
AI PRO R9700 GPUs with PyTorch 2.12.0+rocm10.0.0.

Set these variables before SHMEM initialization and operator construction,
consistently on all ranks:

```bash
export MORI_GPU_ARCHS=gfx1201
export MORI_RDNA4_EP=1
export MORI_EP_COMM=shmem
export MORI_SHMEM_MODE=static_heap
export MORI_SHMEM_HEAP_SIZE=2G
```

Initialize the PyTorch process group and MORI SHMEM, then use the existing
[EP API](MORI-EP-GUIDE.md) with the configuration above. The validation scripts
below include the initialization and teardown steps.

`MORI_RDNA4_EP=1` selects the paired kernels, allocation policy and launch
policy together. Unsupported configurations fail explicitly. Without this
setting, MORI selects its existing generic paths: those paths do not provide
this transport's FP16 or top-k 32 through 64 support. Merely specifying the
GPU architecture does not enable the specialized transport.

The static heap stores control and routing buffers. Its required size grows
with capacity and top-k. Payloads use separate cached IPC allocations and
consume additional VRAM; `MORI_SHMEM_HEAP_SIZE` is not a total VRAM budget.

## Implementation and launch policy

EP2 uses push Dispatch. EP4 pushes hidden payloads and pulls metadata.
Both Combine implementations push remote hidden contributions back and sum
locally. Control buffers remain uncached. Cached payload allocations have a
2 MiB minimum to avoid stale peer visibility observed when smaller cached IPC
allocations are freed and reused on the tested gfx1201 stack.

The policy uses the actual source token count `T`, not allocated capacity:

| Phase | T <= 256 | T > 256 |
| --- | --- | --- |
| EP2 Dispatch | `max(4, ceil(T/16))` blocks, 16 warps/block | 32 blocks, 16 warps/block |
| EP4 Dispatch | `max(8, ceil(T/16))` blocks, 16 warps/block | 32 blocks, 16 warps/block |
| EP2 / EP4 Combine | 32 blocks, 16 warps/block; tiled | 32 blocks, 16 warps/block; whole token |

Tiled Combine uses 512 elements when `T * hidden_dim <= 262144`, otherwise
1024. Explicit per-call launch parameters override these defaults in AUTO and
MANUAL modes. EP2 keeps compatible count exchange when ranks independently
select small and large entries. Completion acknowledgements protect buffers
across consecutive operations.

Top-k 8 retains a separate specialization. Other top-k values use runtime
kernels that scan routing slots in wave32 batches, covering slots 32 through
63 in a second batch. EP4 keeps Dispatch metadata and live Combine weights in
separate buffer regions to protect reuse across calls.

## Correctness validation

Run commands from the repository root after building the native library.
The policy tests do not require PyTorch or a GPU; the host integration tests
require PyTorch and the native library but do not execute GPU kernels:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_rdna4_ep.py \
  tests/python/test_rdna4_ep_integration.py \
  tests/python/test_tuning_config_quant_types.py
```

For GPU checks, select idle gfx1201 devices. The examples below use physical
GPUs 0/1 or 0/1/2/3. Set the enable variables above first. The correctness driver
starts its own processes; invoke it with `python`, not `torchrun`:

```bash
export PYTHONPATH=python:.

# Initial EP2 check; both BF16 and FP16 are exercised.
HIP_VISIBLE_DEVICES=0,1 VERIFY_WORLD_SIZE=2 VERIFY_TOPKS=8,32,64 \
  VERIFY_HIDDEN_SIZES=2056 VERIFY_CAPACITIES=512 \
  VERIFY_HANDLE_INTERLEAVE=1 VERIFY_DOUBLE_COMBINE=1 \
  python tests/python/ops/verify_rdna4_ep.py

# Full top-k range, both hidden sizes and capacity boundaries.
HIP_VISIBLE_DEVICES=0,1 VERIFY_WORLD_SIZE=2 VERIFY_TOPKS=all \
  VERIFY_HIDDEN_SIZES=2056,8192 VERIFY_CAPACITIES=512,256 \
  VERIFY_HANDLE_INTERLEAVE=1 VERIFY_DOUBLE_COMBINE=1 \
  python tests/python/ops/verify_rdna4_ep.py

HIP_VISIBLE_DEVICES=0,1,2,3 VERIFY_WORLD_SIZE=4 VERIFY_TOPKS=all \
  VERIFY_HIDDEN_SIZES=2056,8192 VERIFY_CAPACITIES=512,256 \
  VERIFY_HANDLE_INTERLEAVE=1 VERIFY_DOUBLE_COMBINE=1 \
  python tests/python/ops/verify_rdna4_ep.py
```

The driver compares received payloads, indices, weights, reverse routing maps
and row-dependent Combine results against independent references with zero
tolerance. Cases cover local/remote routes, invalid and dropped routes,
empty/ragged ranks, mixed small/large entries, a remote destination appearing
only in the last expert slot, cross-wave duplicates and all tokens going to
one rank. The extra flags exercise interleaved handle replay and consecutive
Combine without an intervening host synchronization.

`VERIFY_TOPKS` accepts `all` or a comma-separated list. Set
`VERIFY_ROUTING_HANDLES=0` to exercise the ordinary API without handles. Use
`VERIFY_HIDDEN_SIZES=5120 VERIFY_CAPACITIES=1024,192,128,64` to exercise IPC
allocation reuse. A successful run completes every case and exits with status
zero; partial `PASS` output is not sufficient.

## Performance validation

The benchmark validates both implementations and times Graph Dispatch+Combine
against RCCL dense AllGather+ReduceScatter. Tokens are specified per rank.
Unlike the correctness driver, launch the benchmark with `torchrun`:

```bash
HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  tests/python/ops/bench_rdna4_ep.py --topks 1,8,32,64 \
  --hidden-sizes 2048,4096,8192 --tokens 64,1024,4096 \
  --output logs/rdna4_ep2.jsonl

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tests/python/ops/bench_rdna4_ep.py --topks 1,8,32,64 \
  --hidden-sizes 2048,4096,8192 --tokens 64,1024,4096 \
  --output logs/rdna4_ep4.jsonl
```

The default uses both dtypes, FP32 weights, eight warmups and three rounds of
30 samples. Backend timing order alternates between rounds. Each sample takes
the maximum pair latency across ranks before the median is calculated.

JSONL records include correctness status, selected kernels, raw per-rank
samples, software versions and `latency_reduction_percent`, defined as
`100 * (1 - MORI_p50 / RCCL_p50)`. Positive values mean lower MORI latency.
The driver refuses to overwrite an existing output file; choose a fresh name
for each run and record the source revision and GPU model alongside it.

These measurements cover communication only. MORI includes routing and
optional weight communication; the RCCL baseline transports dense hidden
payloads. Neither times expert computation. Correctness support does not imply
that every shape or routing distribution is faster than RCCL.
