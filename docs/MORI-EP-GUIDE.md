# MORI-EP User Guide

MORI-EP provides high-performance MoE (Mixture of Experts) dispatch and combine kernels for Expert Parallelism. It supports both intra-node (XGMI) and inter-node (RDMA) communication, delivering state-of-the-art bandwidth for token routing in models like DeepSeek V3.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. Kernel Types](#1-kernel-types)
- [2. Configuration](#2-configuration)
- [3. Operator API](#3-operator-api)
  - [dispatch()](#dispatch)
  - [combine()](#combine)
  - [Split dispatch/combine (send + recv)](#split-dispatchcombine-send--recv)
  - [reset()](#reset)
  - [get_dispatch_src_token_pos()](#get_dispatch_src_token_pos)
  - [get_registered_combine_input_buffer()](#get_registered_combine_input_buffer)
- [4. Standard MoE Compatibility (DeepEP)](#4-standard-moe-compatibility-deepep)
- [5. Initialization](#5-initialization)
- [6. Launch Configuration](#6-launch-configuration)
- [7. Complete Example](#7-complete-example)
- [8. Environment Variables](#8-environment-variables)
- [9. Benchmarking](#9-benchmarking)
- [10. Profiling with MORI-VIZ](#10-profiling-with-mori-viz)
- [11. Framework Integration](#11-framework-integration)
- [Build Options](#build-options)
- [Source Files](#source-files)

## Quick Reference

```python
import torch
import torch.distributed as dist
import mori

# 1. Initialize torch.distributed (required before MORI-EP shmem initialization)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 2. Initialize shmem (required before any EP operations)
mori.shmem.shmem_torch_process_group_init("default")

# 3. Configure
config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16,
    rank=rank,
    world_size=world_size,
    hidden_dim=7168,
    scale_dim=0,
    scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
    max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,
    num_experts_per_token=8,
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
)

# 3. Create operator and run
op = mori.ops.EpDispatchCombineOp(config)
dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, recv_num_token = \
    op.dispatch(input, weights, scales, indices)
# ... run expert computation on dispatched tokens ...
combine_output, combine_weights = op.combine(expert_output, weights, indices)
op.reset()
```

**Imports:**

| What | Import |
|------|--------|
| Config | `from mori.ops import EpDispatchCombineConfig` |
| Operator | `from mori.ops import EpDispatchCombineOp` |
| Kernel types | `from mori.ops import EpDispatchCombineKernelType` |
| Shmem init | `import mori.shmem` |

---

## 1. Kernel Types

MORI-EP provides five kernel types optimized for different network topologies and latency requirements:

| Kernel Type | Value | Topology | Transport | Use Case |
|-------------|-------|----------|-----------|----------|
| `IntraNode` | 0 | Single node | XGMI (P2P) | EP within a node (e.g., EP8 on 8-GPU node) |
| `InterNode` | 1 | Multi-node | XGMI + RDMA | EP across nodes, baseline inter-node kernel |
| `InterNodeV1` | 2 | Multi-node | XGMI + RDMA | Optimized inter-node with higher bandwidth |
| `InterNodeV1LL` | 3 | Multi-node | XGMI + RDMA | Low-latency variant of InterNodeV1 |
| `AsyncLL` | 4 | Multi-node | XGMI + RDMA | Async low-latency with pipelined transfers |

**How to choose:**

```
Is EP within a single node?
├─ Yes → IntraNode
└─ No (multi-node EP)
   ├─ Throughput priority (large batches) → InterNodeV1
   ├─ Latency priority (small batches)   → InterNodeV1LL or AsyncLL
   └─ Baseline / debugging              → InterNode
```

**Kernel naming in benchmarks:**

| Benchmark Name | Kernel Type | World Size | Notes |
|----------------|-------------|------------|-------|
| EP8 | IntraNode | 8 | Single node, 8 GPUs |
| EP16-V0 | InterNode | 16 | 2 nodes, baseline |
| EP16-V1 | InterNodeV1 | 16 | 2 nodes, optimized |
| EP32-V1-LL | InterNodeV1LL | 32 | 4 nodes, low-latency |

---

## 2. Configuration

### EpDispatchCombineConfig

All configuration is specified through the `EpDispatchCombineConfig` dataclass:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_type` | `torch.dtype` | *(required)* | Token data type (`torch.bfloat16`, `torch.float8_e4m3fnuz`, etc.) |
| `rank` | `int` | *(required)* | Current process rank |
| `world_size` | `int` | *(required)* | Total number of EP ranks |
| `hidden_dim` | `int` | *(required)* | Hidden dimension of token embeddings |
| `scale_dim` | `int` | *(required)* | Scale dimension for quantization (0 if no quantization scales) |
| `scale_type_size` | `int` | *(required)* | Element size of scale data type in bytes |
| `max_token_type_size` | `int` | *(required)* | Element size of the max token data type in bytes (typically `float32`) |
| `max_num_inp_token_per_rank` | `int` | *(required)* | Maximum number of input tokens per rank |
| `num_experts_per_rank` | `int` | *(required)* | Number of experts hosted on each rank |
| `num_experts_per_token` | `int` | *(required)* | Top-K: number of experts selected per token |
| `warp_num_per_block` | `int` | `8` | Warps per GPU thread block |
| `block_num` | `int` | `80` | Number of GPU thread blocks |
| `use_external_inp_buf` | `bool` | `True` | Use external input buffer for combine (vs. zero-copy registered buffer) |
| `kernel_type` | `EpDispatchCombineKernelType` | `IntraNode` | Kernel type selection |
| `gpu_per_node` | `int` | `8` | GPUs per node (for inter-node rank mapping) |
| `rdma_block_num` | `int` | `0` | Thread blocks dedicated to RDMA transfers (inter-node only) |
| `num_qp_per_pe` | `int` | `1` | RDMA queue pairs per PE |

**DeepSeek V3 example** (256 experts, top-8, 8 GPUs):

```python
config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16,
    rank=rank,
    world_size=8,
    hidden_dim=7168,           # DeepSeek V3 hidden dim
    scale_dim=0,               # No quantization scales
    scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
    max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,   # 256 experts / 8 GPUs
    num_experts_per_token=8,   # Top-8 routing
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
)
```

**Inter-node example** (16 GPUs across 2 nodes):

```python
config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16,
    rank=rank,
    world_size=16,
    hidden_dim=7168,
    scale_dim=0,
    scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
    max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=16,   # 256 experts / 16 GPUs
    num_experts_per_token=8,
    kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    gpu_per_node=8,
    rdma_block_num=64,         # Blocks dedicated to RDMA
    block_num=96,              # Total blocks
    warp_num_per_block=8,
)
```

---

## 3. Operator API

### EpDispatchCombineOp

Create the operator with a config:

```python
op = mori.ops.EpDispatchCombineOp(config)
```

### dispatch()

Route input tokens to their assigned expert ranks.

```python
dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, recv_num_token = \
    op.dispatch(input, weights, scales, indices,
                block_num=-1, rdma_block_num=-1, warp_per_block=-1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `torch.Tensor` | Input tokens `[num_tokens, hidden_dim]` |
| `weights` | `torch.Tensor` | Expert weights `[num_tokens, num_experts_per_token]` |
| `scales` | `torch.Tensor` | Quantization scales `[num_tokens, scale_dim]` (pass empty tensor if `scale_dim=0`) |
| `indices` | `torch.Tensor` | Top-K expert indices `[num_tokens, num_experts_per_token]`, dtype `int32` |
| `block_num` | `int` | Override `config.block_num` if > 0 |
| `rdma_block_num` | `int` | Override `config.rdma_block_num` if > 0 |
| `warp_per_block` | `int` | Override `config.warp_num_per_block` if > 0 |

**Returns:** Tuple of 5 tensors:
- `dispatch_output` — Received tokens for this rank's experts `[recv_tokens, hidden_dim]`
- `dispatch_weights` — Corresponding weights `[recv_tokens, num_experts_per_token]`
- `dispatch_scales` — Corresponding scales `[recv_tokens, scale_dim]`
- `dispatch_indices` — Corresponding expert indices `[recv_tokens, num_experts_per_token]`
- `recv_num_token` — Number of tokens received (1-element tensor)

### combine()

Collect expert outputs and route them back to their original ranks.

```python
combine_output, combine_weights = op.combine(
    input, weights, indices,
    block_num=-1, rdma_block_num=-1, warp_per_block=-1,
    use_external_inp_buf=-1, call_reset=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `torch.Tensor` | Expert output tokens |
| `weights` | `torch.Tensor` | Expert weights for weighted combination |
| `indices` | `torch.Tensor` | Top-K expert indices (same as dispatch) |
| `use_external_inp_buf` | `int` | Override `config.use_external_inp_buf`: 0=zero-copy, 1=external buffer |
| `call_reset` | `bool` | Call `reset()` automatically after combine |

**Returns:** Tuple of 2 tensors:
- `combine_output` — Reconstructed tokens at original positions `[num_tokens, hidden_dim]`
- `combine_weights` — Reconstructed weights `[num_tokens, num_experts_per_token]`

### Split dispatch/combine (send + recv)

For overlapping communication with computation, dispatch and combine can be split into separate send and receive phases:

```python
# Dispatch: send phase returns the same 5 values as dispatch()
dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, recv_num_token = \
    op.dispatch_send(input, weights, scales, indices)
# Receive phase (completes the receive-side processing, returns None)
op.dispatch_recv()

# Combine: send phase returns the same 2 values as combine()
combine_output, combine_weights = op.combine_send(expert_output, weights, indices)
# Receive phase (completes the receive-side processing, returns None)
op.combine_recv()
```

> **Note:** `dispatch_send()` delegates to `dispatch()` internally. `dispatch_recv()` and `combine_recv()` perform receive-side processing and return `None`.

### reset()

Reset internal state. Must be called between iterations (unless `call_reset=True` was passed to `combine()`).

```python
op.reset()
```

### get_dispatch_src_token_pos()

Get the source position of each dispatched token (for correctness verification).

```python
torch.cuda.synchronize()
src_token_pos = op.get_dispatch_src_token_pos()
# src_token_pos[i] = src_rank * max_num_inp_token_per_rank + src_token_id
```

### get_registered_combine_input_buffer()

Get the pre-registered combine input buffer for zero-copy mode (`use_external_inp_buf=False`).

```python
combine_buf = op.get_registered_combine_input_buffer(dtype)
combine_buf[:recv_num_token, :].copy_(expert_output[:recv_num_token, :])
```

---

## 4. Standard MoE Compatibility (DeepEP)

MORI-EP provides DeepEP-compatible APIs for frameworks that use standard 3D MoE tensor layouts. These require building with `ENABLE_STANDARD_MOE_ADAPT=ON`:

```bash
ENABLE_STANDARD_MOE_ADAPT=ON pip install -e .
```

### dispatch_standard_moe()

Combined dispatch + format conversion in a single launch:

```python
result = op.dispatch_standard_moe(input, weights, scales, indices)
```

### combine_standard_moe()

Combined combine with standard MoE input format:

```python
output = op.combine_standard_moe(input, weights, indices, call_reset=False)
```

### convert_dispatch_output()

Convert MORI's 2D dispatch output to standard 3D MoE layout:

```python
packed_recv_x, packed_recv_count, packed_recv_src_info, packed_recv_layout_range = \
    op.convert_dispatch_output(dispatch_out_x, dispatch_out_topk_idx)
```

### convert_combine_input()

Prepare standard MoE inputs for MORI combine:

```python
converted = op.convert_combine_input(packed_recv_x, packed_recv_src_info, packed_recv_layout_range)
```

> **Note:** If these methods raise `RuntimeError`, rebuild MORI with `ENABLE_STANDARD_MOE_ADAPT=ON`.

---

## 5. Initialization

MORI-EP requires symmetric memory (shmem) initialization before creating any `EpDispatchCombineOp`. There are two initialization methods:

### Method 1: PyTorch Process Group (Recommended)

```python
import os
import torch
import torch.distributed as dist
import mori

os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

# Initialize PyTorch distributed
torch.cuda.set_device(rank)
dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)

# Register process group and initialize shmem
world_group = dist.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)
mori.shmem.shmem_torch_process_group_init("default")

# ... use MORI-EP ...

# Cleanup
mori.shmem.shmem_finalize()
dist.destroy_process_group()
```

### Method 2: Unique ID (No PyTorch Distributed)

```python
import mori

# Rank 0 generates unique ID and broadcasts to all ranks
if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()
    # broadcast unique_id to all ranks (e.g., via MPI, TCP, file)

# All ranks initialize with the unique ID
mori.shmem.shmem_init_attr(
    mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
    rank, world_size, unique_id
)
```

---

## 6. Launch Configuration

### Manual Mode (Default)

Launch parameters are taken from `EpDispatchCombineConfig` defaults or per-call overrides:

```python
# Use config defaults (block_num=80, warp_num_per_block=8)
op.dispatch(input, weights, scales, indices)

# Override per call
op.dispatch(input, weights, scales, indices, block_num=128, warp_per_block=16)
```

### Auto Mode

Set `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` to use pre-tuned launch parameters:

```bash
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
```

Auto mode selects parameters based on kernel type:

| Kernel Type | block_num | rdma_block_num | warp_per_block |
|-------------|-----------|----------------|----------------|
| InterNodeV1, InterNodeV1LL | 96 | 64 | 8 |
| IntraNode, InterNode, AsyncLL | 128 | 0 | 16 |

When auto mode is active, per-call `block_num`/`rdma_block_num`/`warp_per_block` arguments are ignored.

---

## 7. Complete Example

```python
import os
import time
import torch
import torch.distributed as dist
import mori

os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

def run_ep(rank, world_size):
    # Setup
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank, world_size=world_size,
        device_id=device,
    )
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    mori.shmem.shmem_torch_process_group_init("default")

    # Configuration
    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=7168,
        scale_dim=0,
        scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
        max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
        max_num_inp_token_per_rank=4096,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        use_external_inp_buf=False,
    )

    op = mori.ops.EpDispatchCombineOp(config)

    # Generate test data
    num_tokens = 128
    input_data = torch.randn(num_tokens, config.hidden_dim,
                             dtype=torch.bfloat16, device=device)
    weights = torch.rand(num_tokens, config.num_experts_per_token,
                         dtype=torch.float32, device=device)
    scales = torch.empty(num_tokens, 0, dtype=torch.float8_e4m3fnuz, device=device)

    # Random top-K expert selection
    indices = torch.stack([
        torch.randperm(config.num_experts_per_rank * world_size, device=device)
        [:config.num_experts_per_token]
        for _ in range(num_tokens)
    ]).to(torch.int32)

    # Dispatch tokens to experts
    dispatch_out, dispatch_w, dispatch_s, dispatch_idx, recv_count = \
        op.dispatch(input_data, weights, scales, indices,
                    block_num=80, warp_per_block=16)
    torch.cuda.synchronize()

    total_recv = recv_count[0].item()
    print(f"Rank {rank}: sent {num_tokens} tokens, received {total_recv}")

    # Simulate expert computation (identity for testing)
    expert_output = dispatch_out[:total_recv].clone().to(torch.bfloat16)

    # Combine results back
    combine_out, combine_w = op.combine(
        expert_output, dispatch_w, indices,
        block_num=80, warp_per_block=8,
        call_reset=True,  # reset automatically
    )
    torch.cuda.synchronize()

    # Cleanup
    del op
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = 8
    torch.multiprocessing.spawn(run_ep, args=(world_size,), nprocs=world_size, join=True)
```

---

## 8. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MORI_SHMEM_HEAP_SIZE` | — | Symmetric heap size (e.g., `"6G"`, `"2G"`). Must be set before shmem init. |
| `MORI_RDMA_DEVICES` | all available | RDMA NIC selection. Include: `mlx5_0,mlx5_1`. Exclude: `^mlx5_2,mlx5_3` |
| `MORI_EP_LAUNCH_CONFIG_MODE` | `"MANUAL"` | Launch config mode: `"MANUAL"` or `"AUTO"` |
| `GLOO_SOCKET_IFNAME` | — | TCP interface for torch distributed (e.g., `ens14np0`) |
| `MASTER_ADDR` | — | Torch distributed master address |
| `MASTER_PORT` | — | Torch distributed master port |

---

## 9. Benchmarking

### Intra-node

```bash
cd /path/to/mori
python3 tests/python/ops/bench_dispatch_combine.py
```

### Inter-node

Run on each node (replace `node_rank` and `master_addr`):

```bash
export GLOO_SOCKET_IFNAME=ens14np0
export MORI_RDMA_DEVICES=^mlx5_0,mlx5_1  # Optional: exclude specific NICs

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr="10.194.129.65" --master_port=1234 \
    examples/ops/dispatch_combine/test_dispatch_combine_internode.py --bench
```

The benchmark output reports total tokens received, RDMA token count, and total bandwidth (XGMI + RDMA). To calculate RDMA-only bandwidth:

```
RDMA BW = Total BW × (RDMA tokens / Total tokens)
```

### Reference Performance (DeepSeek V3 config)

4096 tokens, 7168 hidden, top-8, FP8 dispatch + BF16 combine:

| Kernel | CUs | Dispatch XGMI | Dispatch RDMA | Combine XGMI | Combine RDMA |
|--------|-----|---------------|---------------|--------------|--------------|
| EP8 (IntraNode) | 80 | 307 GB/s | — | 330 GB/s | — |
| EP16-V1 (InterNodeV1) | 80 | 208 GB/s | 63 GB/s | 161 GB/s | 49 GB/s |
| EP32-V1-LL (InterNodeV1LL) | 32 | 103 GB/s | 57 GB/s | 91 GB/s | 50 GB/s |

---

## 10. Profiling with MORI-VIZ

Build with profiling enabled:

```bash
ENABLE_PROFILER=ON pip install -e .
```

Capture and export a trace:

```python
from mori.kernel_profiler import export_to_perfetto

# Run dispatch/combine, then:
if hasattr(mori.cpp, "get_debug_time_buf"):
    trace_buffer = mori.cpp.get_debug_time_buf(op._handle)
    export_to_perfetto(trace_buffer, "ep_trace.json")
```

Visualize at [ui.perfetto.dev](https://ui.perfetto.dev/). See [PROFILER.md](PROFILER.md) for full profiler documentation.

---

## 11. Framework Integration

MORI-EP is integrated in several LLM inference frameworks:

| Framework | Integration Point | Notes |
|-----------|------------------|-------|
| [AITER](https://github.com/ROCm/aiter) | `aiter/moe_op/mori_all2all.py` | MoriAll2AllManager wraps dispatch/combine for FusedMoE |
| [ATOM](https://github.com/ROCm/ATOM) | `atom/model_ops/fused_moe/moe.py` | Expert parallelism with FusedMoEParallelConfig |
| [vLLM](https://github.com/vllm-project/vllm) | MoE expert parallelism | Dispatch/combine for distributed MoE |
| [SGLang](https://github.com/sgl-project/sglang) | MoE expert parallelism | Dispatch/combine for distributed MoE |

---

## Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `BUILD_OPS` | `ON` | Build MORI-EP dispatch/combine kernels |
| `BUILD_SHMEM` | `ON` | Build symmetric memory library |
| `BUILD_IO` | `ON` | Build MORI-IO library |
| `BUILD_PYBINDS` | `ON` | Build Python bindings |
| `ENABLE_PROFILER` | `OFF` | Enable MORI-VIZ kernel profiler |
| `ENABLE_STANDARD_MOE_ADAPT` | `OFF` | Enable DeepEP-compatible standard MoE APIs |
| `ENABLE_DEBUG_PRINTF` | `OFF` | Enable debug printf in device kernels |
| `USE_ROCM` | `ON` | Build for ROCm/HIP (vs. CUDA) |
| `USE_BNXT` | `OFF` | Enable Broadcom Thor2 NIC support |
| `USE_IONIC` | `OFF` | Enable AMD Pensando DSC NIC support |

---

## Source Files

| File | Description |
|------|-------------|
| `python/mori/ops/dispatch_combine.py` | Python API: `EpDispatchCombineConfig`, `EpDispatchCombineOp` |
| `python/mori/ops/__init__.py` | Public exports |
| `python/mori/shmem/api.py` | Shmem Python API |
| `include/mori/ops/dispatch_combine/dispatch_combine.hpp` | C++ header: config, handle, kernel args |
| `src/ops/dispatch_combine/dispatch_combine.cpp` | Core dispatch/combine implementation |
| `src/ops/dispatch_combine/internode_v1.cpp` | InterNodeV1/V1LL kernels |
| `src/ops/dispatch_combine/low_latency_async.cpp` | AsyncLL kernels |
| `src/pybind/mori.cpp` | Python bindings for all APIs |
| `examples/ops/dispatch_combine/test_dispatch_combine.py` | Complete Python example |
| `examples/ops/dispatch_combine/test_dispatch_combine_internode.py` | Inter-node example |
| `tests/python/ops/test_dispatch_combine.py` | Correctness tests |
| `tests/python/ops/bench_dispatch_combine.py` | Performance benchmarks |
