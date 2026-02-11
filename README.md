# MORI

<img src="docs/mori_arch_20250819_v0.png">

**MORI** (**Mo**dular **R**DMA **I**nterface) is a **bottom-up, modular, and composable framework** for building high-performance communication applications with a strong focus on **RDMA + GPU integration**. Inspired by the role of MLIR in compiler infrastructure, MORI provides reusable and extensible building blocks that make it **easier for developers to adopt advanced techniques** such as IBGDA (Infiniband GPUDirect Async) and GDS (GPUDirect Storage).

To help developers get started quickly, MORI also includes a suite of optimized libraries — **MORI-EP** (MoE dispatch & combine kernels), **MORI-IO** (P2P communication for KVCache transfer), and **MORI-CCL** (collective communication) — that deliver out-of-the-box performance.

## Features

| Component | Description |
|-----------|-------------|
| **MORI-EP** | Intra and inter-node dispatch/combine kernels for MoE Expert Parallelism with SOTA performance |
| **MORI-IO** | Point-to-point communication library with ultra-low overhead for KVCache transfer |
| **MORI-CCL** | Lightweight collective communication library for latency-sensitive or resource-constrained environments |
| **MORI Shmem** | OpenSHMEM-style symmetric memory APIs for GPU memory management and RDMA |
| **MORI-VIZ** | Warp-level kernel profiler with Perfetto integration |

**Framework building blocks:**
- High-performance building blocks for IBGDA / P2P and more
- Modular & composable components for transport management, topology detection, etc.
- C++ and Python level APIs

**Supported NICs:** AMD Pensando DSC, Broadcom Thor2, NVIDIA Mellanox ConnectX-7

## Quick Start

```python
import os, torch, torch.distributed as dist
import mori

os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

# Obtain rank/world_size from torchrun environment (e.g., torchrun --nproc_per_node=8)
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)

# Initialize distributed + shmem
dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
world_group = dist.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)
mori.shmem.shmem_torch_process_group_init("default")

# Configure MORI-EP (DeepSeek V3: 256 experts, top-8, 8 GPUs)
config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16, rank=rank, world_size=8,
    hidden_dim=7168, scale_dim=0,
    scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
    max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32, num_experts_per_token=8,
)

# Dispatch tokens to experts → run expert computation → combine results back
op = mori.ops.EpDispatchCombineOp(config)
dispatch_out, dispatch_w, dispatch_s, dispatch_idx, recv_count = \
    op.dispatch(input_tokens, weights, scales, expert_indices)
# ... expert computation on dispatch_out ...
combine_out, combine_w = op.combine(expert_output, dispatch_w, expert_indices, call_reset=True)
```

See the [MORI-EP Guide](docs/MORI-EP-GUIDE.md) for the full API reference and more examples.

## Documentation

| **Topic** | **Description** | **Guide** |
|---|---|---|
| MORI-EP | Dispatch/combine API, kernel types, configuration, usage examples | [EP Guide](docs/MORI-EP-GUIDE.md) |
| MORI-EP Benchmark | Intra/inter-node benchmark commands and NIC selection | [EP Benchmark](docs/MORI-EP-BENCHMARK.md) |
| MORI Shmem | Symmetric memory APIs, initialization, memory management | [Shmem Guide](docs/MORI-SHMEM-GUIDE.md) |
| MORI-IO | P2P communication concepts, engine/backend/session design | [IO Introduction](docs/MORI-IO-INTRO.md) |
| MORI-IO Benchmark | IO benchmark commands and performance results | [IO Benchmark](docs/MORI-IO-BENCHMARK.md) |
| MORI-VIZ | Warp-level kernel profiler with Perfetto integration | [Profiler](docs/PROFILER.md) |

## Benchmarks

**Hardware:** 8 x MI300X per node, 8 single-port CX7 400Gb/s RDMA NICs | **Software:** ROCm 6.4.0

### MORI-EP

Benchmark on DeepSeek V3 model configurations:

**Bandwidth** (4096 tokens, 7168 hidden, top-8 experts, FP8 dispatch + BF16 combine)

| **Kernels**| **# CUs**| **Dispatch XGMI** |**Dispatch RDMA** |**Combine XGMI**|**Combine RDMA** |
|------------|----------|-------------------|------------------|----------------|-----------------|
|EP8         | 80       | 307 GB/s          | x                | 330 GB/s       | x               |
|EP16-V0     | 32       | 75 GB/s           | 23 GB/s          | 76 GB/s        | 23 GB/s          |
|EP16-V0     | 80       | 79 GB/s           | 24 GB/s          | 82 GB/s        | 25 GB/s          |
|EP16-V1     | 32       | 185 GB/s          | 57 GB/s          | 172 GB/s       | 52 GB/s          |
|EP16-V1     | 80       | 208 GB/s          | 63 GB/s          | 161 GB/s       | 49 GB/s          |
|EP32-V1-LL  | 32       | 103 GB/s          | 57 GB/s          | 91 GB/s        | 50 GB/s          |

**Latency** (128 tokens, 7168 hidden, top-8 experts, FP8 dispatch + BF16 combine)

| **Kernels**| **# CUs**| **Dispatch Latency** |**Dispatch BW** |**Combine Latency**|**Combine BW** |
|------------|----------|----------------------|----------------|-------------------|---------------|
|EP8         | 64       | 35 us                | 134 GB/s       | 47 us             | 204 GB/s      |
|EP16-V0     | 32       | 226 us               | 33 GB/s        | 296 us            | 49 GB/s       |
|EP16-V1     | 32       | 115 us               | 63 GB/s        | 141 us            | 110 GB/s      |
|EP32-V1-LL  | 32       | 157 us               | 48 GB/s        | 280 us            | 55 GB/s       |

**NOTE:** Best performance values from multiple test rounds to eliminate fluctuations.

### MORI-IO

**NOTE:** This is the preview version of MORI-IO benchmark performance.

GPU Direct RDMA READ, pairwise, 128 consecutive transfers, 1 GPU, MI300X + Thor2:

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|      8      |    128    |      0.00      |      0.03     |      0.03     |    33.38     |    36.33     |
|      16     |    128    |      0.00      |      0.06     |      0.06     |    34.09     |    36.35     |
|      32     |    128    |      0.00      |      0.12     |      0.11     |    34.57     |    36.33     |
|      64     |    128    |      0.01      |      0.24     |      0.23     |    33.62     |    36.33     |
|     128     |    128    |      0.02      |      0.49     |      0.45     |    33.62     |    36.49     |
|     256     |    128    |      0.03      |      0.94     |      0.89     |    34.81     |    36.99     |
|     512     |    128    |      0.07      |      1.86     |      1.77     |    35.29     |    37.01     |
|     1024    |    128    |      0.13      |      3.84     |      3.53     |    34.09     |    37.09     |
|     2048    |    128    |      0.26      |      7.33     |      6.96     |    35.76     |    37.65     |
|     4096    |    128    |      0.52      |     12.94     |     12.46     |    40.53     |    42.09     |
|     8192    |    128    |      1.05      |     20.75     |     20.12     |    50.54     |    52.11     |
|    16384    |    128    |      2.10      |     29.03     |     28.33     |    72.24     |    74.02     |
|    32768    |    128    |      4.19      |     36.50     |     35.91     |    114.92    |    116.81    |
|    65536    |    128    |      8.39      |     41.74     |     41.39     |    200.99    |    202.70    |
|    131072   |    128    |     16.78      |     45.14     |     44.85     |    371.69    |    374.10    |
|    262144   |    128    |     33.55      |     46.93     |     46.76     |    715.02    |    717.56    |
|    524288   |    128    |     67.11      |     47.94     |     47.81     |   1399.99    |   1403.64    |
|   1048576   |    128    |     134.22     |     48.44     |     48.32     |   2770.90    |   2777.76    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

## Framework Integration

MORI-EP is integrated in several LLM inference and training frameworks:

| Framework | Usage |
|-----------|-------|
| [vLLM](https://github.com/vllm-project/vllm) | MoE expert parallelism dispatch/combine |
| [SGLang](https://github.com/sgl-project/sglang) | MoE expert parallelism dispatch/combine |
| [AITER](https://github.com/ROCm/aiter) | MoriAll2AllManager wrapping dispatch/combine for FusedMoE |
| [ATOM](https://github.com/ROCm/ATOM) | Expert parallelism in FusedMoE layer |

## Installation

### Prerequisites

- ROCm >= 6.4.0 with PyTorch
- Linux packages: see `docker/Dockerfile.dev`

Or build the Docker image:

```bash
cd mori && docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
```

### Install with Python

```bash
cd mori
pip install -r requirements-build.txt
git submodule update --init --recursive
pip install .
# NOTE: for venv build, add --no-build-isolation at the end
```

## Testing

### MORI-EP (dispatch / combine)

```bash
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Correctness tests
pytest tests/python/ops/

# Benchmark performance
python3 tests/python/ops/bench_dispatch_combine.py
```

### MORI-IO

```bash
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Correctness tests
pytest tests/python/io/

# Benchmark (run on each of two nodes)
export GLOO_SOCKET_IFNAME=ens14np0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.194.129.65" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.194.129.65" \
    --enable-batch-transfer --enable-sess \
    --buffer-size 32768 --transfer-batch-size 128
```

## Contribution Guide

Welcome to MORI! We appreciate your interest in contributing. Whether you're fixing bugs, adding features, improving documentation, or sharing feedback, your contributions help make MORI better for everyone.

### Code Quality

MORI uses pre-commit hooks to maintain code quality. After cloning the repository:

```bash
pip install pre-commit
cd /path/to/mori
pre-commit install

# Run on all files (first time)
pre-commit run --all-files
```

Pre-commit automatically checks code formatting, linting, license headers, and other quality checks on commit. To skip checks when necessary: `git commit --no-verify`
