# Mori Shmem Kernel POC — Triton / MLIR / LLVM IR

Demonstrates using mori's shmem device API directly from custom GPU kernels,
with three compilation paths:

- **Path 1 (Triton):** Vanilla `@triton.jit` + `extern_libs` linking + `@core.extern` device function declarations
- **Path 2 (MLIR):** Programmatic IR construction with MLIR Python bindings (`llvm` + `rocdl` dialects)
- **Path 3 (LLVM IR):** Direct LLVM IR text generation (zero extra dependencies beyond ROCm)

All paths link against `libmori_shmem_device.bc` — the bitcode library
containing mori's `extern "C"` device function wrappers and the `globalGpuStates` symbol.

## Pipelines

```
Path 1 (Triton):
  @triton.jit kernel          Triton compiler         
    @core.extern decls  ──►  compile + link bc  ──►  GPU binary
    extern_libs={bc}              ↑
    shmem_module_init hook        │
                       libmori_shmem_device.bc

Path 2 (MLIR):
  Python (mlir.ir)       mlir-translate       llvm-link + clang
    llvm.LLVMFuncOp  ──►  MLIR text  ──►  LLVM IR  ──►  .hsaco
    llvm.CallOp                                 ↑
    rocdl.kernel                    libmori_shmem_device.bc

Path 3 (LLVM IR):
  Python (string)                              llvm-link + clang
    LLVM IR text  ─────────────────────────►  .hsaco
    declare/call @mori_shmem_*                  ↑
                                    libmori_shmem_device.bc
```

## Prerequisites

### 1. Build and install mori (with device wrapper)

```bash
cd <mori_repo>
BUILD_SHMEM_DEVICE_WRAPPER=ON pip install . --no-build-isolation
```

### 2. Build the shmem device bitcode

```bash
bash tools/build_shmem_bitcode.sh
```

This produces `lib/libmori_shmem_device.bc` in the mori repo root.

### 3. ROCm toolchain

Standard ROCm install provides `llvm-link` and `clang`.

### 4. Triton (Path 1 only)

A working Triton installation (upstream or ROCm fork).

### 5. MLIR Python bindings (Path 2 only)

```bash
pip install nanobind pybind11
cd /tmp
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.2/llvm-project-20.1.2.src.tar.xz -O llvm-src.tar.xz
tar xf llvm-src.tar.xz && mkdir mlir-build && cd mlir-build
cmake -G Ninja /tmp/llvm-project-20.1.2.src/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=AMDGPU \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc) MLIRPythonModules mlir-translate
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo /tmp/mlir-build/tools/mlir/python_packages/mlir_core > $SITE/mlir-python.pth
```

## Usage

### Basic tests (multi-GPU, MLIR + LLVM IR paths)

```bash
cd examples/shmem/mlir_shmem_poc
bash run.sh 2 gfx942
```

### Triton basic tests (put/get)

```bash
torchrun --nproc_per_node=2 test_triton_shmem.py
```

### Triton allreduce (bf16 sum, P2P)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 test_triton_allreduce.py

# 8 GPUs
torchrun --nproc_per_node=8 test_triton_allreduce.py
```

## Tests

| File | Path | Kernels | What it tests |
|------|------|---------|---------------|
| `test_mlir_shmem.py` | MLIR + LLVM IR | `shmem_basic_kernel`, `shmem_put_kernel` | PE query, RDMA put ring |
| `test_triton_shmem.py` | Triton | `shmem_basic_kernel`, `shmem_put_kernel` | PE query, RDMA put ring |
| `test_triton_allreduce.py` | Triton | `allreduce_sum_kernel` | Intra-node allreduce (bf16 sum, 64x7168) via P2P reads |

### Allreduce details

- Each PE reads from all PEs' symmetric memory via P2P pointers and accumulates (fp32) locally
- Host side computes `shmem_ptr_p2p()` for each peer, passes as int64 tensor
- Kernel casts int64 to bf16 pointer via `bitcast`, then `tl.load` for P2P read
- Autotune across BLOCK_SIZE (1024-8192) and num_warps (4-32)

### Benchmark results (MI300X, 64x7168 bf16)

| GPUs | Latency | Effective BW |
|------|---------|-------------|
| 2 | 25 us | 73 GB/s |
| 8 | 70 us | 104 GB/s |

## Files

| File | Description |
|------|-------------|
| `mlir_shmem_kernel.py` | Kernel builder (MLIR API + LLVM IR text) and compile pipelines |
| `test_mlir_shmem.py` | MLIR/LLVM IR path tests |
| `test_triton_shmem.py` | Pure Triton path basic tests (put/get) |
| `test_triton_allreduce.py` | Pure Triton P2P allreduce with autotune |
| `run.sh` | Convenience script for MLIR/LLVM IR tests |
| `../../tools/build_shmem_bitcode.sh` | Builds `lib/libmori_shmem_device.bc` |
