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

For IBGDA/RDMA testing, if you use BNXT NIC need to add `USE_BNXT=ON`, use AINIC need to add `USE_IONIC=ON`

### 2. Build the shmem device bitcode

```bash
bash tools/build_shmem_bitcode.sh
```

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

### Basic tests (MLIR + LLVM IR paths)

```bash
cd examples/shmem/mlir_shmem_poc
bash run.sh 2 gfx942
```

### Triton basic tests (put/get)

```bash
torchrun --nproc_per_node=2 test_triton_shmem.py
```

### Triton allreduce (bf16 sum)

```bash
# P2P mode (intra-node, default)
torchrun --nproc_per_node=8 test_triton_allreduce.py

# IBGDA/RDMA mode (disable P2P, kernel auto-adapts grid strategy)
MORI_DISABLE_P2P=ON torchrun --nproc_per_node=8 test_triton_allreduce.py
```

## Tests

| File | Path | Kernels | What it tests |
|------|------|---------|---------------|
| `test_mlir_shmem.py` | MLIR + LLVM IR | `shmem_basic_kernel`, `shmem_put_kernel` | PE query, RDMA put ring |
| `test_triton_shmem.py` | Triton | `shmem_basic_kernel`, `shmem_put_kernel` | PE query, RDMA put ring |
| `test_triton_allreduce.py` | Triton | `allreduce_p2p_kernel`, `allreduce_put_signal_kernel` | Intra-node allreduce bf16 sum (64x7168) |

### Allreduce kernels

**Kernel A (`allreduce_p2p_kernel`):** Each block calls `mori_shmem_ptr_p2p()` on the device to resolve remote symmetric addresses, then `tl.load` for P2P reads + fp32 accumulate.

**Kernel B (`allreduce_put_signal_kernel`):** All-to-all `putmem_nbi_signal_block` with multi-block parallelism. Data is chunked, multiple blocks put chunks to different PEs with `SIGNAL_ADD`. All blocks wait via `uint64_wait_until_equals`, then accumulate locally. Auto-adapts grid strategy based on transport:

- **P2P mode:** `CHUNKS_PER_PE=8`, `grid=(64,)` — multi-block parallel put via GPU memcpy
- **IBGDA/RDMA mode:** `CHUNKS_PER_PE=1`, `grid=(8,)` — one block per PE to avoid QP contention

Transport detection: host-side `shmem_ptr_p2p()` returns 0 for RDMA peers, non-zero for P2P peers.

| API used in kernel | Kernel A | Kernel B |
|--------------------|----------|----------|
| `mori_shmem_my_pe()` | yes | yes |
| `mori_shmem_ptr_p2p()` | yes | — |
| `mori_shmem_putmem_nbi_signal_block()` | — | yes |
| `mori_shmem_uint64_wait_until_equals()` | — | yes |
| `mori_shmem_quiet_thread()` | — | yes |

### Benchmark (MI300X, 8 GPU, 64x7168 bf16 = 896 KB)

| Kernel | P2P mode | IBGDA mode |
|--------|----------|------------|
| A (shmem_ptr_p2p) | ~76 us | ~76 us |
| B (put+signal) | ~150 us (grid=64) | ~495 us (grid=8) |

## Files

| File | Description |
|------|-------------|
| `mlir_shmem_kernel.py` | Kernel builder (MLIR API + LLVM IR text) and compile pipelines |
| `test_mlir_shmem.py` | MLIR/LLVM IR path tests |
| `test_triton_shmem.py` | Triton path basic tests (put/get) |
| `test_triton_allreduce.py` | Triton allreduce: P2P read + put+signal kernels (auto-adapts P2P/IBGDA) |
| `run.sh` | Convenience script for MLIR/LLVM IR tests |
| `../../tools/build_shmem_bitcode.sh` | Builds `lib/libmori_shmem_device.bc` |
