# MORI JIT Compilation Framework

MORI uses a **host pre-compiled + device JIT** architecture. Host C++ code
(bootstrap, RDMA transport, pybind11) is compiled once during `pip install`
using a standard C++ compiler — **hipcc is not required at install time**.
All GPU kernels — ops dispatch/combine and shmem device bitcode — are
JIT-compiled on first use, targeting the exact GPU architecture and NIC type
of the runtime machine. Compiled artifacts are cached to `~/.mori/jit/`.

## Quick Start

```bash
# 1. Install (compiles host code only, ~18s, no hipcc needed)
pip install .

# 2. (Optional) Pre-compile all device kernels (~22s parallel, needs hipcc)
MORI_PRECOMPILE=1 python -c "import mori"

# 3. Run — kernels are JIT-compiled on first use if not pre-compiled
torchrun --nproc_per_node=8 my_app.py
```

Both `pip install .` (wheel) and `pip install -e .` (editable/development)
are supported. The wheel bundles all `.hip` sources and C++ headers needed
for runtime JIT compilation.

## Architecture Overview

```
pip install .  (~18s, CXX compiler only)
  ├── CMake + clang++ → host .so (no device code in ops)
  │   ├── mori_ops        → CXX  (args construction, handle lifecycle)
  │   ├── mori_shmem      → CXX  (host-only: init, runtime, memory)
  │   └── mori_pybinds    → CXX  (pybind11 bindings)
  └── Copy JIT sources → mori/_jit_sources/ (packaged in wheel)

First run (JIT, one-time, needs hipcc at runtime)
  ├── detect GPU arch (rocm_agent_enumerator → gfx942)
  ├── detect NIC type (/sys/class/infiniband/ → bnxt/mlx5/ionic)
  ├── hipcc --genco → dispatch/combine kernels (.hsaco)
  └── hipcc --cuda-device-only → shmem bitcode (.bc)

Kernel launch (Python-driven)
  ├── Python: select kernel name, compute grid/block/sharedMem
  ├── C++ pybind: build typed args struct from handle state
  └── Python ctypes: hipModuleLaunchKernel(func, grid, block, args_ptr)
```

## Packaging: How Wheel Installs Support JIT

During `pip install .`, the build copies JIT-required source files into
`mori/_jit_sources/` inside the Python package:

```
site-packages/mori/
├── _jit_sources/                      # Bundled for runtime JIT
│   ├── include/mori/**/*.hpp          # All C++ headers
│   ├── src/ops/kernels/*.hip          # Kernel source files
│   ├── src/ops/dispatch_combine/      # Kernel implementation headers
│   ├── src/shmem/                     # shmem_device_api_wrapper.cpp only
│   └── 3rdparty/{spdlog,msgpack-c}/include/  # Third-party headers
├── jit/                               # JIT compiler Python code
│   ├── core.py                        # compile_genco(), ensure_bitcode()
│   ├── config.py                      # get_mori_source_root(), detect_build_config()
│   ├── hip_driver.py                  # HipModule / HipFunction (ctypes)
│   └── cache.py                       # Cache directory management
├── ops/dispatch_combine.py            # Python-side kernel launch orchestration
├── ir/bitcode.py                      # Triton bitcode locator
├── libmori_pybinds.so                 # Host pybind11 module
├── libmori_application.so             # Bootstrap + RDMA transport
└── libmori_io.so                      # I/O module
```

`get_mori_source_root()` locates the JIT source tree:

1. **Editable install**: repo root (3 levels up from `mori/jit/config.py`)
2. **Wheel install**: `mori/_jit_sources/` inside the installed package
3. Returns `None` if neither is found

## Compilation Split: Host CXX vs Device JIT

The key design principle is **host code compiles with a standard C++ compiler
(clang/g++), device code is JIT-compiled with hipcc at runtime**.

### What compiles as CXX (at `pip install` time)

| File | Purpose |
|------|---------|
| `dispatch_combine.cpp` | Handle lifecycle, shmem buffer init/finalize, `GetEpDispatchCombineArgsRaw()` |
| `pybind_ops.cpp` | `prepare_inference`, `build_args`/`free_args`, output pointer getters, `get_handle_info` |
| `pybind_shmem.cpp` | `shmem_module_init`, shmem host API bindings |
| All `application/*.cpp` | Bootstrap, RDMA transport, symmetric memory management |
| `shmem/{init,runtime,memory}.cpp` | Shmem host-side initialization and memory management |

These files use `shmem_api.hpp` (host-only) instead of `shmem.hpp` (which
pulls in device kernels). `hip/hip_fp8.h` is guarded with `#ifdef __HIPCC__`
in `data_types.hpp` and `dispatch_combine.hpp` to avoid ROCm 6.x
incompatibilities.

### What compiles as HIP (runtime JIT only)

| File | Purpose | When |
|------|---------|------|
| `ep_intranode.hip` | IntraNode dispatch/combine kernels | Runtime JIT |
| `ep_internode.hip` | InterNode dispatch/combine | Runtime JIT |
| `ep_internode_v1.hip` | InterNodeV1 dispatch/combine + sync | Runtime JIT |
| `ep_internode_v1ll.hip` | InterNodeV1LL low-latency variant | Runtime JIT |
| `ep_async_ll.hip` | AsyncLL send/recv | Runtime JIT |
| `cast_kernel.hip` | Float→FP4 cast | Runtime JIT |
| `shmem_kernels.hip` | shmem barrier + `globalGpuStates` shim | Runtime JIT |
| `shmem_device_api_wrapper.cpp` | shmem device bitcode (put/get/signal) | Runtime JIT (bitcode) |

### Template Args vs Raw Args

Kernel functions take `EpDispatchCombineArgs<T>` by value, where `T` is the
data type (`hip_bfloat16`, `float`, etc.). Since `T` only affects pointer
types (`T*` → 8 bytes regardless of `T`), an `EpDispatchCombineArgsRaw`
struct with `void*` has identical binary layout:

```
C++ pybind (CXX):  build_args(handle) → new EpDispatchCombineArgsRaw{...} → int64 ptr
Python:            hipModuleLaunchKernel(func, grid, block, sharedMem, stream, ptr)
C++ pybind:        free_args(ptr) → delete
```

A `static_assert` (compiled only under hipcc) validates layout equivalence.
The template `EpDispatchCombineArgs<T>` and `EpDispatchCombineArgsVariant`
are likewise hipcc-only; CXX code uses `EpDispatchCombineArgsRaw` exclusively.

## What Gets JIT-Compiled

| Component | Compiler | Output | Trigger |
|-----------|----------|--------|---------|
| Ops kernels | `hipcc --genco` | `.hsaco` | `EpDispatchCombineOp.__init__()` |
| Cast kernel | `hipcc --genco` | `.hsaco` | `cast_jit._get_module()` |
| Shmem bitcode | `hipcc --cuda-device-only` + `llvm-link` | `.bc` | `find_bitcode()` / Triton `get_extern_libs()` |

Ops kernels are split by `kernel_type` — only the required group is compiled:

| KernelType | File | Compile Time |
|------------|------|-------------|
| IntraNode | `ep_intranode.hip` | ~9s |
| InterNode | `ep_internode.hip` | ~10s |
| InterNodeV1 | `ep_internode_v1.hip` | ~19s |
| InterNodeV1LL | `ep_internode_v1ll.hip` | ~22s |
| AsyncLL | `ep_async_ll.hip` | ~7s |

Pre-compile all at once with `MORI_PRECOMPILE=1 python -c "import mori"` (~22s
with 7 parallel hipcc invocations + 1 bitcode build).

## Kernel Launch Flow

```
Python dispatch_combine.py
  │
  ├── mori_cpp.prepare_inference(handle, input_ptr, dtype, num_tokens, ...)
  │     └── C++: handle.inputType = dtype; handle.inpTokenBuf = ptr; ...
  │
  ├── args_ptr = mori_cpp.build_args(handle, rdma_block_num, hidden_dim, ...)
  │     └── C++: new EpDispatchCombineArgsRaw(GetArgsRaw(handle, ...))
  │
  ├── sfx = _DTYPE_SUFFIX[input.dtype]                   # "bf16", "f32", ...
  ├── kernel_name = f"EpDispatchIntraNodeKernel_{sfx}"    # Python selects
  ├── grid  = (block_num,)                                # Python computes
  ├── block = (WARP_SIZE * warp_per_block,)
  ├── shared_mem = dispatch_shared_mem(warp_per_block)
  │
  ├── func = hip_module.get_function(kernel_name)
  ├── func.launch_struct(grid, block, shared_mem, stream, args_ptr)
  │     └── ctypes: hipModuleLaunchKernel(func, gx,gy,gz, bx,by,bz, ...)
  │
  └── mori_cpp.free_args(args_ptr)
        └── C++: delete ptr
```

Dtype suffix mapping:

| PyTorch dtype | Suffix | Kernel example |
|---------------|--------|----------------|
| `torch.bfloat16` | `bf16` | `EpDispatchIntraNodeKernel_bf16` |
| `torch.float32` | `f32` | `EpCombineInterNodeKernel_f32` |
| `torch.float8_e4m3fn` | `fp8_ocp` | `EpDispatchIntraNodeKernel_fp8_ocp` |
| `torch.float8_e4m3fnuz` | `fp8_fnuz` | `EpCombineIntraNodeKernel_fp8_fnuz_nop2p` |

## HIP Driver API (Python ctypes)

`mori.jit.hip_driver` provides a minimal ctypes wrapper around `libamdhip64.so`:

- **`HipModule(hsaco_path)`** — calls `hipModuleLoad`, caches `hipFunction_t`
  handles by name. Destructor calls `hipModuleUnload`.
- **`HipFunction.launch(grid, block, shared_mem, stream, *args)`** — individual
  scalar/pointer arguments, packed as `void**` kernelParams.
- **`HipFunction.launch_struct(grid, block, shared_mem, stream, struct_ptr)`** —
  single struct argument passed by value. `struct_ptr` is a host pointer to the
  struct data; `hipModuleLaunchKernel` copies the struct into kernel argument
  buffer.

After loading a `HipModule`, `shmem_module_init(module._module.value)` must
be called to copy `globalGpuStates` into the JIT module before any kernel
launch.

## Triton Integration

Triton kernels use mori shmem via device bitcode linking:

```python
from mori.ir.triton.runtime import get_extern_libs, install_hook

install_hook()  # One-time: register shmem_module_init as post-compile hook

@triton.jit
def my_kernel(...):
    mori_shmem_put_nbi(...)  # Calls into linked bitcode

my_kernel[(grid,)](..., extern_libs=get_extern_libs())
```

- **`get_extern_libs()`** returns `{"mori_shmem": find_bitcode()}`
- **`install_hook()`** registers `shmem_module_init` as Triton's
  `jit_post_compile_hook` so that `globalGpuStates` is initialized in
  every compiled Triton module

Bitcode search order in `find_bitcode()`:

1. `MORI_SHMEM_BC` environment variable
2. JIT cache (`~/.mori/jit/<arch>_<nic>/<hash>/libmori_shmem_device.bc`)
3. JIT compile via `ensure_bitcode()` (compiles `shmem_device_api_wrapper.cpp`
   + `globalGpuStates.hip` shim → llvm-link → strip lifetime intrinsics)
4. Pre-built `.bc` next to `bitcode.py`, or in `lib/` / `build/lib/`

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
- Source file change → new hash → recompile
- Different GPU/NIC → separate directory
- `FileBaton` file lock prevents concurrent compilation conflicts

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MORI_PRECOMPILE=1` | off | Pre-compile all kernels on `import mori` |
| `MORI_DISABLE_JIT=1` | off | Disable bitcode JIT (error if no pre-built .bc) |
| `MORI_JIT_CACHE_DIR` | `~/.mori/jit/` | Custom cache directory |
| `MORI_GPU_ARCHS` | auto-detect | Override GPU architecture (e.g. `gfx942`) |
| `MORI_SHMEM_BC` | auto | Explicit path to shmem bitcode |
| `USE_BNXT=ON` | auto-detect | Override NIC type to Broadcom BNXT |
| `USE_IONIC=ON` | auto-detect | Override NIC type to AMD/Pensando IONIC |

## Testing

### Docker Environment

```bash
# ROCm 7.1 container (primary)
docker exec -it mori_jit_test bash
cd /workspace/mori

# ROCm 6.4 container (compatibility)
docker exec -it mori_jit_test_rocm64 bash
cd /workspace/mori
```

### 1. Non-Editable Install (Wheel)

```bash
# Build and install as wheel (tests pip install . end-to-end)
rm -rf build
pip install . --no-build-isolation

# Verify _jit_sources bundled in site-packages
python -c "
from mori.jit.config import get_mori_source_root
root = get_mori_source_root()
print(f'Source root: {root}')
assert root is not None, 'JIT source root not found'
assert (root / 'src/ops/kernels/ep_intranode.hip').is_file()
print('OK: JIT sources are packaged')
"
```

### 2. Editable Install (Development)

```bash
rm -rf build
pip install -e . --no-build-isolation
export PYTHONPATH=/workspace/mori:$PYTHONPATH
```

### 3. Pre-compile All Kernels

```bash
rm -rf ~/.mori/jit
MORI_PRECOMPILE=1 python -c "import mori"
# Expected: ~22s, 6 .hsaco files + 1 .bc + 1 cast_kernel.hsaco
```

### 4. Verify JIT Configuration

```bash
python -c "
from mori.jit.config import detect_build_config, detect_nic_type, get_mori_source_root
cfg = detect_build_config()
print(f'GPU:    {cfg.arch}')
print(f'NIC:    {detect_nic_type()}')
print(f'Root:   {get_mori_source_root()}')
print(f'hipcc:  {cfg.hipcc}')
"
```

### 5. Verify Compilation Separation

```bash
# Only works with editable install (build/ directory present)
python -c "
import json
with open('build/compile_commands.json') as f:
    data = json.load(f)
cxx, hip = [], []
for e in data:
    fn = e['file'].split('/')[-1]
    fp = e['file']
    if '/3rdparty/' in fp or '/examples/' in fp or '/benchmarks/' in fp:
        continue
    if any(x in e['command'] for x in ['-x hip', '--offload-arch']):
        hip.append(fn)
    else:
        cxx.append(fn)
print(f'CXX: {len(cxx)} files | HIP: {len(hip)} files')
print(f'HIP files: {sorted(hip)}')
"
# Expected: CXX ~34 | HIP ~1
# HIP: device_link_stub.hip (shmem device code is now JIT-compiled)
```

### 6. Dispatch/Combine Correctness

```bash
# Single test (IntraNode, bf16, 8 GPUs, ~20s)
pytest 'tests/python/ops/test_dispatch_combine.py::test_dispatch_combine[none-True-8-32-1-1-0-7168-data_type0-8]' -x -v

# Full suite (256 cases: 80 pass, 176 skip on gfx942, ~26s)
pytest tests/python/ops/test_dispatch_combine.py -q
```

### 7. Dispatch/Combine Benchmark

```bash
python tests/python/ops/bench_dispatch_combine.py
# Expected: ~300 GB/s dispatch, ~330 GB/s combine
```

### 8. Shmem API

```bash
pytest tests/python/shmem/test_api.py -q
# Expected: 18 passed (~235s)
```

### 9. Triton Integration

```bash
# Basic shmem put (2 GPUs)
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

# Allreduce P2P (8 GPUs)
torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py

# Allreduce IBGDA/RDMA (8 GPUs, P2P disabled)
MORI_DISABLE_P2P=ON torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py
```

### 10. Full Clean-Slate Test

```bash
rm -rf build ~/.mori/jit
pip install . --no-build-isolation
cd /tmp  # leave source tree to verify wheel is self-contained
MORI_PRECOMPILE=1 python -c "import mori"
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH
pytest tests/python/ops/test_dispatch_combine.py -q
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py
```

## Kernel Source Files

```
src/ops/kernels/
├── ep_common.hip                 # Shared includes, macros, globalGpuStates shim
├── ep_intranode.hip              # IntraNode dispatch + combine + convert
├── ep_internode.hip              # InterNode (legacy) dispatch + combine
├── ep_internode_v1.hip           # InterNodeV1 dispatch + combine + sync
├── ep_internode_v1ll.hip         # InterNodeV1LL low-latency variant
├── ep_async_ll.hip               # AsyncLL send/recv
├── cast_kernel.hip               # Float→FP4 cast (Python-side launcher)
└── shmem_kernels.hip             # shmem barrier kernel + globalGpuStates
```

Each kernel is split into `__device__ _body` + `__global__` wrapper in the
headers under `src/ops/dispatch_combine/`, enabling `extern "C"` JIT wrappers
without duplicating code.

## Adding a New Kernel

1. Write the kernel with `__device__` body:

```cpp
template <typename T>
__device__ void MyKernel_body(EpDispatchCombineArgs<T> args) { /* impl */ }

template <typename T>
__global__ void MyKernel(EpDispatchCombineArgs<T> args) { MyKernel_body<T>(args); }
```

2. Add `extern "C"` wrappers in a `.hip` file under `src/ops/kernels/`:

```cpp
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES
WRAP_ALL_TYPES(MyKernel)
```

3. Launch from Python:

```python
# In dispatch_combine.py or your own module:
args_ptr = mori_cpp.build_args(handle, rdma_block_num=rbn, hidden_dim=dim)
func = hip_module.get_function(f"MyKernel_{sfx}")
func.launch_struct(grid, block, shared_mem, stream, args_ptr)
mori_cpp.free_args(args_ptr)
```

4. Register in `_KERNEL_TYPE_TO_HIP` if it belongs to a dispatch/combine mode,
   and in `precompile()` for pre-compilation support.

## Host/Device NIC Macro Separation

| Macro | Scope | Set by | Controls |
|-------|-------|--------|----------|
| `ENABLE_BNXT` | Host C++ | CMake `find_library` | Link `libbnxt_re.so`, compile `bnxt.cpp` |
| `MORI_DEVICE_NIC_BNXT` | Device JIT | Python `detect_nic_type()` | `DISPATCH_BNXT=1` in IBGDA kernels |

A single host `.so` can be built on a CI machine with all NIC libraries
available, while device kernels are JIT-compiled with the correct NIC branch
for the actual runtime hardware.

## ROCm Version Compatibility

| Feature | ROCm 6.4 | ROCm 7.1 |
|---------|----------|----------|
| Host CXX compilation | Yes | Yes |
| `hip/hip_bfloat16.h` in CXX | Yes | Yes |
| `hip/hip_fp8.h` in CXX | No (guarded) | Yes |
| FP8 kernel JIT | Yes | Yes |
| FP4 kernel JIT | No (`hip_ext_ocp.h` absent) | Yes |
| `pip install .` (wheel) | Yes | Yes |

`hip/hip_fp8.h` is guarded with `#ifdef __HIPCC__` in `data_types.hpp` and
`dispatch_combine.hpp` because ROCm 6.x's `amd_warp_functions.h` uses GPU
builtins (`__builtin_amdgcn_*`) unavailable in CXX mode. This guard has no
effect on JIT compilation (which always uses hipcc).
