# MORI-IR Guide

MORI-IR is the **integration layer** that enables external GPU kernel frameworks to use MORI's shmem communication capabilities (P2P, RDMA/IBGDA, SDMA) from device code. It provides two key building blocks:

1. **Device bitcode** (`libmori_shmem_device.bc`) — pre-compiled LLVM bitcode containing all shmem device functions (`extern "C"` wrappers), linkable by any GPU kernel framework
2. **IR Python API** (`mori.ir`) — framework-agnostic utilities for locating the device bitcode and querying device function ABI metadata

These two components are **framework-agnostic by design**. Any system that can link LLVM bitcode — Triton, FlyDSL, MLIR-based compilers, custom HIP/LLVM toolchains, etc. — can integrate MORI's communication primitives into its device kernels. Triton is provided as the first **reference integration**; the same bitcode and ABI metadata apply to all other backends.

**Relationship with `mori.shmem`:** The device functions in the bitcode depend on
the shmem runtime (`globalGpuStates`) being initialized on the host. Before any
device function can be called, the application must:

1. Initialize shmem via `mori.shmem` host APIs (`shmem_torch_process_group_init` or `shmem_init_attr`) — see [Shmem Guide](MORI-SHMEM-GUIDE.md)
2. Allocate symmetric memory via `mori.shmem.shmem_malloc` or `mori_shmem_create_tensor`
3. For dynamically compiled kernels (e.g. Triton), call `mori.shmem.shmem_module_init(hip_module)` to copy GPU states into the kernel module

`mori.ir` handles the **compile/link-time** concern (find bitcode, generate wrappers), while `mori.shmem` handles the **runtime** concern (init, memory, state).

## Table of Contents

- [Architecture](#architecture)
- [1. Host-Side Python API](#1-host-side-python-api)
- [2. Device Bitcode](#2-device-bitcode)
- [3. Device Functions](#3-device-functions)
- [4. Integration Example: Triton](#4-integration-example-triton)
- [5. Integration Example: Raw Bitcode (no framework)](#5-integration-example-raw-bitcode-no-framework)
- [6. Bitcode JIT Compilation](#6-bitcode-jit-compilation)
- [7. Examples and Testing](#7-examples-and-testing)
- [Known Limitations](#known-limitations)

## Architecture

```
 Compile / Link time                    Runtime (host)
 ──────────────────                     ───────────────
┌───────────────────────┐       ┌───────────────────────────┐
│ mori.ir               │       │ mori.shmem (host APIs)    │
│  find_bitcode()       │       │  shmem_*_init()           │
│  MORI_DEVICE_FUNCTIONS│       │  shmem_malloc/free        │
│  mori.ir.triton (ref) │       │  shmem_module_init (JIT)  │
└───────────┬───────────┘       └─────────────┬─────────────┘
            │ locates & links                 │ initializes
┌───────────▼─────────────────────────────────▼─────────────┐
│         libmori_shmem_device.bc (LLVM bitcode)            │
│  Device functions: put, atomics, wait, barrier, ...       │
│  Depends on globalGpuStates (init by mori.shmem)          │
└─────────────────────────┬─────────────────────────────────┘
                          │ called from
┌─────────────────────────▼─────────────────────────────────┐
│    Your GPU Kernel (Triton, FlyDSL, MLIR, HIP, ...)       │
└───────────────────────────────────────────────────────────┘
```

| Component | Stage | Role |
|-----------|-------|------|
| `mori.ir.find_bitcode()` | Compile/link | Locate `libmori_shmem_device.bc` (auto JIT-compiled for current GPU + NIC) |
| `mori.ir.MORI_DEVICE_FUNCTIONS` | Compile/link | ABI metadata dict for generating framework-specific wrappers |
| `mori.ir.triton` | Compile/link | Reference backend: Triton-specific wrappers |
| `mori.shmem.*` | Runtime | Initialize shmem, allocate memory, sync — required before device functions work |

---

## 1. Host-Side Python API

### `mori.ir` (framework-agnostic)

These APIs have **no dependency on any kernel framework** — they work with Triton, FlyDSL, raw HIP, or any custom compiler:

| Export | Description |
|--------|-------------|
| `find_bitcode()` | Return absolute path to `libmori_shmem_device.bc` (auto JIT-compiled for current GPU + NIC) |
| `get_bitcode_path()` | Alias for `find_bitcode()` |
| `MORI_DEVICE_FUNCTIONS` | Dict of all device function ABI metadata — each entry contains `symbol` (C name), `args` (type list), `ret` (return type) |
| `SIGNAL_SET` | Signal operation constant (value 9) |
| `SIGNAL_ADD` | Signal operation constant (value 10) |

**Using ABI metadata to generate your own wrappers:**

```python
from mori.ir import MORI_DEVICE_FUNCTIONS

# Inspect a device function
func = MORI_DEVICE_FUNCTIONS["putmem_nbi_thread"]
print(func["symbol"])  # "mori_shmem_putmem_nbi_thread"
print(func["args"])    # ["uint64", "uint64", "uint64", "int32", "int32"]
print(func["ret"])     # "int32"

# Your framework can iterate MORI_DEVICE_FUNCTIONS to auto-generate
# wrappers for all 50+ device functions
```

### `mori.ir.triton` (Triton-specific reference backend)

| Export | Description |
|--------|-------------|
| `get_extern_libs()` | Returns dict for Triton `extern_libs` parameter |
| `install_hook()` | Install Triton compilation hook for automatic bitcode linking |
| Device wrappers | All functions from `MORI_DEVICE_FUNCTIONS` callable inside `@triton.jit` |

---

## 2. Device Bitcode

`libmori_shmem_device.bc` is an LLVM bitcode library containing `extern "C"` device functions. To use it in your own framework:

```python
from mori.ir import find_bitcode

bc_path = find_bitcode()
# bc_path is e.g. "~/.mori/jit/<arch>/<hash>/libmori_shmem_device.bc"
```

Then link it into your GPU kernel using your framework's mechanism:

```bash
# Example: raw LLVM toolchain
llvm-link my_kernel.bc $(python -c "from mori.ir import find_bitcode; print(find_bitcode())") -o linked.bc
clang -target amdgcn-amd-amdhsa -mcpu=gfx942 linked.bc -o kernel.hsaco
```

Your device code can then call any function from the bitcode directly:

```c
// In your HIP/device code
extern "C" int mori_shmem_my_pe();
extern "C" uint64_t mori_shmem_ptr_p2p(uint64_t dest_ptr, int my_pe, int dest_pe);
extern "C" int mori_shmem_putmem_nbi_thread(void* dest, const void* src, size_t n, int pe, int qp);
```

---

## 3. Device Functions

All functions below are available as `extern "C"` symbols in the bitcode. In Triton, they are callable as `mori_shmem_device.<name>()` inside `@triton.jit` kernels. In other frameworks, call the C symbol directly via bitcode linking.

### Query

| Function | Args | Return |
|----------|------|--------|
| `my_pe()` | — | `int32` |
| `n_pes()` | — | `int32` |

### Point-to-Point

| Function | Args | Return |
|----------|------|--------|
| `ptr_p2p(ptr, my_pe, dest_pe)` | `uint64, int32, int32` | `uint64` |
| `ptr(dest, dest_pe)` | `uint64, int32` | `uint64` |

### PutNbi (Thread / Warp / Block)

| Function | Args |
|----------|------|
| `putmem_nbi_thread(dest, src, nbytes, pe, qp)` | `ptr, ptr, size, int, int` |
| `putmem_nbi_warp(...)` | same |
| `putmem_nbi_block(...)` | same |

Typed variants: `put_uint32_nbi_*`, `put_uint64_nbi_*`, `put_float_nbi_*`, `put_double_nbi_*`

### PutNbi with Signal

| Function | Args |
|----------|------|
| `putmem_nbi_signal_thread(dest, src, n, sig, val, op, pe, qp)` | 8 args |
| `putmem_nbi_signal_warp(...)` | same |
| `putmem_nbi_signal_block(...)` | same |

Signal ops: `SIGNAL_SET` (9), `SIGNAL_ADD` (10)

### Immediate Put

| Function | Args |
|----------|------|
| `int32_p(dest, val, pe, qp)` | `ptr, int32, int, int` |
| `uint64_p(dest, val, pe, qp)` | `ptr, uint64, int, int` |
| `float_p(dest, val, pe, qp)` | `ptr, float32, int, int` |

### Atomics

| Function | Return |
|----------|--------|
| `uint32_atomic_add_thread(dest, val, pe, qp)` | `int32` |
| `uint64_atomic_fetch_add_thread(dest, val, pe, qp)` | `uint64` |
| `atomic_uint32_nonfetch_thread(dest, val, op, pe, qp)` | `int32` |
| `atomic_uint64_fetch_thread(dest, val, cmp, op, pe, qp)` | `uint64` |

### Wait

| Function | Args | Return |
|----------|------|--------|
| `uint64_wait_until_equals(addr, val)` | `ptr, uint64` | `int32` |
| `uint64_wait_until_greater_than(addr, val)` | `ptr, uint64` | `uint64` |
| `uint32_wait_until_equals(addr, val)` | `ptr, uint32` | `int32` |

### Synchronization

| Function | Description |
|----------|-------------|
| `quiet_thread()` | Complete all pending remote operations (thread scope) |
| `quiet_thread_pe(pe)` | Complete pending ops to specific PE |
| `fence_thread()` | Order remote operations (thread scope) |
| `barrier_all_thread()` | Global barrier (thread scope) |
| `barrier_all_block()` | Global barrier (block scope) |

See `python/mori/ir/ops.py` for the complete function table with all C symbols.

---

## 4. Integration Example: Triton

Triton is the first reference backend. `mori.ir.triton` auto-generates Triton wrappers from `MORI_DEVICE_FUNCTIONS`:

```python
import triton
import triton.language as tl
import mori.shmem as ms
from mori.ir import triton as mori_shmem_device
from mori.ir.triton import get_extern_libs, install_hook

# Host: initialize shmem
ms.shmem_torch_process_group_init("default")
buf = ms.mori_shmem_create_tensor((N,), torch.bfloat16)
install_hook()

# Device: Triton kernel
@triton.jit
def my_kernel(buf_ptr, N, BLOCK: tl.constexpr):
    pe = mori_shmem_device.my_pe()
    next_pe = (pe + 1) % mori_shmem_device.n_pes()

    remote = mori_shmem_device.ptr_p2p(buf_ptr.to(tl.uint64, bitcast=True), pe, next_pe)
    remote_ptr = remote.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    data = tl.load(remote_ptr + tl.arange(0, BLOCK))

my_kernel[(grid,)](buf, N, BLOCK=1024, extern_libs=get_extern_libs())
```

## 5. Integration Example: Raw Bitcode (no framework)

For custom compilers or HIP toolchains that don't use Triton:

```python
from mori.ir import find_bitcode, MORI_DEVICE_FUNCTIONS

bc_path = find_bitcode()

# Generate your own wrappers from ABI metadata
for name, meta in MORI_DEVICE_FUNCTIONS.items():
    print(f"{meta['ret']} {meta['symbol']}({', '.join(meta['args'])})")
```

```bash
# Link bitcode into your kernel
llvm-link my_kernel.bc $(python -c "from mori.ir import find_bitcode; print(find_bitcode())") -o linked.bc
clang -target amdgcn-amd-amdhsa -mcpu=gfx942 linked.bc -o kernel.hsaco
```

## 6. Bitcode JIT Compilation

The bitcode is **automatically JIT-compiled** on first use — no manual build step required. `find_bitcode()` compiles `shmem_device_api_wrapper.cpp` with `hipcc --cuda-device-only` and caches the result to `~/.mori/jit/`.

The NIC type (BNXT / AINIC / MLX5) and GPU architecture are auto-detected at runtime.

To precompile ahead of time:

```bash
MORI_PRECOMPILE=1 python -c "import mori"
```

---

## 7. Examples

| File | What it demonstrates |
|------|---------------------|
| `examples/shmem/ir/test_triton_shmem.py` | Basic put/get via `mori.ir.triton` |
| `examples/shmem/ir/test_triton_allreduce.py` | Allreduce: P2P read + put+signal kernels |
| `examples/shmem/ir/test_mlir_shmem.py` | MLIR / LLVM IR paths (no Triton) |

---

## 8. Testing

```bash
# Triton basic tests (2 GPUs)
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py

# Triton allreduce — P2P mode (8 GPUs)
torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py

# Triton allreduce — IBGDA/RDMA mode (8 GPUs)
MORI_DISABLE_P2P=ON torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py

# MLIR + LLVM IR paths (2 GPUs, no Triton)
cd examples/shmem/ir && bash run.sh 2 gfx942
```

---

## Known Limitations

- Triton's `extern_elementwise` forces all device functions to return `int32` even when the C function returns `void`. This is a Triton upstream limitation.
- Pointer arguments are passed as `uint64` (intptr cast) since `extern_elementwise` does not support `pointer_type(void)`.
