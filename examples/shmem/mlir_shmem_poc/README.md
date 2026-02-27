# Mori Shmem Kernel POC — MLIR / LLVM IR (No Triton)

Demonstrates using mori's shmem device API directly from custom GPU kernels,
without depending on Triton or triton-dist. Kernels are built via two paths:

- **Path A (MLIR):** Programmatic IR construction with MLIR Python bindings (`llvm` + `rocdl` dialects)
- **Path B (LLVM IR):** Direct LLVM IR text generation (zero extra dependencies beyond ROCm)

Both paths link against `libmori_shmem_device.bc` — the bitcode library
containing mori's `extern "C"` device function wrappers
(`mori_shmem_my_pe`, `mori_shmem_int32_p`, `mori_shmem_quiet_thread`, etc.)
and the `globalGpuStates` symbol.

## Pipelines

```
Path A (MLIR):
  Python (mlir.ir)       mlir-translate       llvm-link           clang
    llvm.LLVMFuncOp  ──►  MLIR text  ──►  LLVM IR  ──►  link bc  ──►  .hsaco
    llvm.CallOp                                 ↑
    rocdl.kernel                                │
                                    libmori_shmem_device.bc

Path B (LLVM IR):
  Python (string)                              llvm-link           clang
    LLVM IR text  ─────────────────────────►  link bc  ──►  .hsaco
    declare @mori_shmem_*                       ↑
    call @mori_shmem_*                          │
                                    libmori_shmem_device.bc

Both paths then:
  hipModuleLoad(.hsaco) → mori shmem_module_init(module) → hipModuleLaunchKernel
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

Standard ROCm install provides `llvm-link` and `clang` (used for linking
and compiling the kernel).

### 4. MLIR Python bindings (Path A only)

Path B works without this. For Path A:

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

### Compile-only (single process)

```bash
python3 examples/shmem/mlir_shmem_poc/mlir_shmem_kernel.py gfx942
```

### Full end-to-end test (multi-GPU, both paths)

```bash
cd examples/shmem/mlir_shmem_poc
bash run.sh 2 gfx942

# Or directly
torchrun --nproc_per_node=2 test_mlir_shmem.py gfx942
```

## Tests

4 tests total (2 kernels x 2 compilation paths):

| Path | Kernel | Mori device functions used |
|------|--------|---------------------------|
| MLIR | `shmem_basic_kernel` | `mori_shmem_my_pe()`, `mori_shmem_n_pes()` |
| MLIR | `shmem_put_kernel` | `mori_shmem_int32_p()`, `mori_shmem_quiet_thread()` |
| LLVM IR | `shmem_basic_kernel` | Same as above |
| LLVM IR | `shmem_put_kernel` | Same as above |

- **basic kernel** — verifies device-side PE query works (globalGpuStates properly initialized)
- **put kernel** — verifies RDMA put in a ring topology (each PE writes to next PE, all verify)

## Files

| File | Description |
|------|-------------|
| `mlir_shmem_kernel.py` | Kernel builder (MLIR API + LLVM IR text) and compile pipelines |
| `test_mlir_shmem.py` | Test driver: both paths, HIP load + `shmem_module_init` + launch + verify |
| `run.sh` | Convenience script |
| `../../tools/build_shmem_bitcode.sh` | Builds `lib/libmori_shmem_device.bc` from mori's compiled BC files |
