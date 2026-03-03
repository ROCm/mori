# MORI JIT Compilation Framework

MORI uses a **host pre-compiled + device JIT** architecture. Host C++ code
(bootstrap, RDMA transport, pybind11) is compiled once during `pip install`.
All GPU kernels — ops dispatch/combine and shmem device bitcode — are
JIT-compiled on first use, targeting the exact GPU architecture and NIC type
of the runtime machine. Compiled artifacts are cached to `~/.mori/jit/`.

## Quick Start

```bash
# 1. Install (compiles host code only, ~30s)
pip install -e .

# 2. (Optional) Pre-compile all device kernels (~22s parallel)
MORI_PRECOMPILE=1 python -c "import mori"

# 3. Run — kernels are JIT-compiled on first use if not pre-compiled
torchrun --nproc_per_node=8 my_app.py
```

## Architecture

```
pip install .  (~30s)
  └── CMake + hipcc → host .so only (no device code)

First run (JIT, one-time)
  ├── detect GPU arch (rocm_agent_enumerator → gfx942)
  ├── detect NIC type (/sys/class/infiniband/ → bnxt/mlx5/ionic)
  ├── hipcc --genco → dispatch/combine kernels (.hsaco)
  └── hipcc --cuda-device-only → shmem bitcode (.bc)

Subsequent runs
  └── cache hit (<1ms) → hipModuleLoad → hipModuleLaunchKernel
```

## What Gets JIT-Compiled

| Component | Compiler | Output | Trigger |
|-----------|----------|--------|---------|
| Ops kernels (dispatch/combine) | `hipcc --genco` | `.hsaco` | `EpDispatchCombineOp.__init__()` |
| Shmem device bitcode | `hipcc --cuda-device-only` + `llvm-link` | `.bc` | `find_bitcode()` / Triton `get_extern_libs()` |

Ops kernels are split by `kernel_type` — only the required group is compiled:

| KernelType | File | Compile Time |
|------------|------|-------------|
| IntraNode | `ep_intranode.hip` | ~9s |
| InterNode | `ep_internode.hip` | ~10s |
| InterNodeV1 | `ep_internode_v1.hip` | ~19s |
| InterNodeV1LL | `ep_internode_v1ll.hip` | ~22s |
| AsyncLL | `ep_async_ll.hip` | ~7s |

## Cache Structure

```
~/.mori/jit/
└── gfx942_bnxt/                          # <gpu_arch>_<nic_type>
    ├── ab065555b30b/                     # content hash of source files
    │   └── ep_intranode.hsaco
    ├── afcfa60c20a2/
    │   └── libmori_shmem_device.bc
    ├── 575fe0455099/
    │   └── cast_kernel.hsaco
    └── ...
```

- Cache key = `<arch>_<nic>/<content_hash>/`
- Source files change → new hash → recompile
- Different GPU/NIC → different directory
- `FileBaton` file lock prevents concurrent compilation conflicts

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MORI_PRECOMPILE=1` | off | Pre-compile all kernels on `import mori` |
| `MORI_DISABLE_JIT=1` | off | Disable bitcode JIT fallback (error if no pre-built .bc) |
| `MORI_JIT_CACHE_DIR` | `~/.mori/jit/` | Custom cache directory |
| `MORI_GPU_ARCHS` | auto-detect | Override GPU architecture (e.g. `gfx942`) |
| `USE_BNXT=ON` | auto-detect | Override NIC type to Broadcom BNXT |
| `USE_IONIC=ON` | auto-detect | Override NIC type to AMD/Pensando IONIC |

## Testing

### Prerequisites

```bash
# Inside Docker container with ROCm + GPUs
docker exec -it <container> bash
cd /path/to/mori

# Install
pip install -e .
export PYTHONPATH=/path/to/mori:$PYTHONPATH
```

### 1. Verify JIT Configuration

```bash
python -c "
from mori.jit.config import detect_build_config, detect_nic_type, get_mori_source_root
cfg = detect_build_config()
print(f'GPU:  {cfg.arch}')
print(f'NIC:  {detect_nic_type()}')
print(f'Root: {get_mori_source_root()}')
print(f'hipcc: {cfg.hipcc}')
"
```

### 2. Pre-compile All Kernels

```bash
rm -rf ~/.mori/jit
MORI_PRECOMPILE=1 python -c "import mori"
# Expected output: ~22s, all kernels cached
```

### 3. Dispatch/Combine Correctness Test

```bash
# Single test case (IntraNode, bf16, 8 GPUs)
pytest 'tests/python/ops/test_dispatch_combine.py::test_dispatch_combine[none-True-8-32-1-1-0-7168-data_type0-8]' -x -v

# Full test suite (256 cases, 80 pass / 176 skip on gfx942)
pytest tests/python/ops/test_dispatch_combine.py -x -v
```

### 4. Dispatch/Combine Benchmark

```bash
python tests/python/ops/bench_dispatch_combine.py
# Expected: Dispatch ~300 GB/s, Combine ~330 GB/s
```

### 5. Shmem API Tests

```bash
pytest tests/python/shmem/test_api.py -x -v
```

### 6. Triton Integration Tests

```bash
# Basic shmem put (2 GPUs)
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

# Allreduce P2P mode (2 or 8 GPUs)
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_allreduce.py
torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py

# Allreduce IBGDA/RDMA mode (8 GPUs, disables P2P)
MORI_DISABLE_P2P=ON torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py
```

### 7. Verify JIT Bitcode NIC Branch

```bash
# Check that bitcode uses the correct NIC provider
python -c "
from mori.ir.bitcode import find_bitcode
import subprocess
bc = find_bitcode()
result = subprocess.run(
    ['/opt/rocm/lib/llvm/bin/opt', '-S', bc, '-o', '-'],
    capture_output=True, text=True
)
bnxt = result.stdout.lower().count('bnxt')
print(f'Bitcode: {bc}')
print(f'BNXT symbols: {bnxt} (should be >0 on BNXT machines)')
"
```

### 8. Clean Rebuild Test

```bash
# Full clean-slate test
rm -rf build ~/.mori/jit
pip install -e .
MORI_PRECOMPILE=1 python -c "import mori"
pytest tests/python/ops/test_dispatch_combine.py -x -q
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py
```

## Kernel Source Files

```
src/ops/kernels/
├── ep_common.hip             # Shared includes, macros, globalGpuStates shim
├── ep_intranode.hip          # IntraNode dispatch + combine + convert
├── ep_internode.hip          # InterNode (legacy) dispatch + combine
├── ep_internode_v1.hip       # InterNodeV1 dispatch + combine + sync
├── ep_internode_v1ll.hip     # InterNodeV1LL low-latency variant
├── ep_async_ll.hip           # AsyncLL send/recv
├── cast_kernel.hip           # Float→FP4 cast (Python-side launcher)
└── dispatch_combine_kernels.hip  # All-in-one (fallback, not used by default)
```

Each kernel is split into `__device__ _body` + `__global__` wrapper in the
original headers, enabling `extern "C"` JIT wrappers without duplicating code.

## Adding a New Kernel

1. Write the kernel with `__device__` body:
   ```cpp
   template <typename T>
   __device__ void MyKernel_body(Args<T> args) { /* impl */ }

   template <typename T>
   __global__ void MyKernel(Args<T> args) { MyKernel_body<T>(args); }
   ```

2. Add `extern "C"` wrappers in a `.hip` file:
   ```cpp
   #include "src/ops/kernels/ep_common.hip"
   MORI_DEFINE_GPU_STATES
   WRAP_ALL_TYPES(MyKernel)
   ```

3. Launch from C++ with `jit_launch`:
   ```cpp
   jit_launch("MyKernel_" + sfx, grid, block, sharedMem, stream, args);
   ```

4. Register in `_KERNEL_TYPE_TO_HIP` if it belongs to a dispatch/combine mode.

## Host/Device NIC Macro Separation

Host and device code use separate macros for NIC selection:

| Macro | Scope | Set by | Controls |
|-------|-------|--------|----------|
| `ENABLE_BNXT` | Host C++ | CMake `find_library` | Link `libbnxt_re.so`, compile `bnxt.cpp` |
| `MORI_DEVICE_NIC_BNXT` | Device JIT | Python `detect_nic_type()` | `DISPATCH_BNXT=1` in IBGDA kernels |

This allows a single host `.so` to be built on a CI machine with all NIC
libraries available, while device kernels are JIT-compiled with the correct
NIC branch for the actual runtime hardware.
