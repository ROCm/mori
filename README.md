# MORI

<img src="docs/mori_arch_20250819_v0.png">

**MORI** (**Mo**dular **R**DMA **I**nterface) is a **bottom-up, modular, and composable framework** for building high-performance communication applications with a strong focus on **RDMA + GPU integration**. Inspired by the role of MLIR in compiler infrastructure, MORI provides reusable and extensible building blocks that make it **easier for developers to adopt advanced techniques** such as IBGDA (Infiniband GPUDirect Async) and GDS (GPUDirect Storage).

To help developers get started quickly, MORI also includes a suite of optimized libraries—**MORI-EP** (MoE dispatch & combine kernels), **MORI-IO** (p2p communication for KVCache transfer), and **MORI-CCL** (collective communication)—that deliver out-of-the-box performance.

Feature summary:
- Applications
    - MORI-EP: intra and inter-node dispatch/combine kernels with SOTA performance
    - MORI-IO: point-to-point communication library with ultra-low overhead
    - MORI-CCL: lightweight and flexible collective communication library designed for highly customized use cases such as latency-sensitive or resource-constrained environment
- Framework
    - High-performance building blocks for IBGDA / P2P and more​
    - Modular & composable components for developing communication applications, such as transport management, topology detection and etc.
    - Shmem-style APIs
    - C++ level APIs
    - Python level APIs

## Benchmarks

Configurations:
- Hardware: 8 x MI300X per node, with 8 single-port CX7 400Gb/s RDMA NICs
- Software: ROCm 6.4.0

### MORI-EP

Benchmark result on DeepSeek V3 model configurations:

**Bandwidth Performance**

4096 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatching and BF16 combining

| **Kernels**| **# CUs**| **Dispatch XGMI** |**Dispatch RDMA** |**Combine XGMI**|**Combine RDMA** |
|------------|----------|-------------------|------------------|----------------|-----------------|
|EP8         | 80       | 307 GB/s          | x                | 330 GB/s       | x               |
|EP16-V0     | 32       | 75 GB/s           | 23 GB/s          | 76 GB/s        | 23 GB/s          |
|EP16-V0     | 80       | 79 GB/s           | 24 GB/s          | 82 GB/s        | 25 GB/s          | 
|EP16-V1     | 32       | 185 GB/s          | 57 GB/s          | 172 GB/s       | 52 GB/s          |
|EP16-V1     | 80       | 208 GB/s          | 63 GB/s          | 161 GB/s       | 49 GB/s          |
|EP32-V1-LL  | 32       | 103 GB/s          | 57 GB/s          | 91 GB/s        | 50 GB/s          |

**Latency Performance**

128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatching and BF16 combining

| **Kernels**| **# CUs**| **Dispatch Latency** |**Dispatch BW** |**Combine Latency**|**Combine BW** |
|------------|----------|----------------------|----------------|-------------------|---------------|
|EP8         | 80       | 37 us                | 176 GB/s       | 99 us             | 107 GB/s      |
|EP16-V0     | 32       | 226 us               | 33 GB/s        | 296 us            | 49GB/s        |
|EP16-V1     | 32       | 115 us               | 63 GB/s        | 141 us            | 110GB/s       |
|EP32-V1-LL  | 32       | 157 us               | 48 GB/s        | 280 us            | 55GB/s        |

**NOTE**: We show best performance values measured from multiple test rounds to eliminate fluctuations.

### MORI-IO

**NOTE**: This is the preview version of MORI-IO Benchmark performance, we will soon merge MORI-IO into main branch

Benchmark result on the following configurations:
- Operation: GPU direct RDMA READ
- Mode: pairwise
- Number of consecutive Transfer: 128
- Number of GPUs: 1
- Hardware: MI300X + Thor2

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

- Session is a specific technique used in MORI-IO to reduce overhead

## Installation

### Prerequisites

- pytorch:rocm >= 6.4.0
- Linux packages: see packages in dockerfile

Or build docker image with:
```
cd mori && docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
```

### Install with Python
```
# NOTE: for venv build, add --no-build-isolation at the end
cd mori && pip install -r requirements-build.txt && git submodule update --init --recursive && pip3 install .
```

### Test dispatch / combine
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Test correctness
pytest tests/python/ops/

# Benchmark performance
python3 tests/python/ops/bench_dispatch_combine.py
```

### Test MORI-IO
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Test correctness
pytest tests/python/io/

# Benchmark performance
# Run the following command on two nodes
export GLOO_SOCKET_IFNAME=ens14np0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.129.65" --master_port=1234 tests/python/io/benchmark.py --host="10.194.129.65" --enable-batch-transfer --enable-sess --buffer-size 32768 --transfer-batch-size 128
```

## Contribution Guide

Welcome to MORI! We appreciate your interest in contributing. Whether you're fixing bugs, adding features, improving documentation, or sharing feedback, your contributions help make MORI better for everyone.

### Code Quality

MORI uses pre-commit hooks to maintain code quality. After cloning the repository:

```bash
# Install and setup pre-commit
pip install pre-commit
cd /path/to/mori
pre-commit install

# Run on all files (first time)
pre-commit run --all-files
```

Pre-commit automatically checks code formatting, linting, license headers, and other quality checks on commit. To skip checks when necessary: `git commit --no-verify`
