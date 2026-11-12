# MORI Shmem Guide

MORI Shmem provides OpenSHMEM-style symmetric memory APIs for GPU memory management and RDMA communication. The original MORI-EP API (`mori.ops.EpDispatchCombineOp`) and MORI-CCL build on it, so shmem must be initialized before using them; on gfx125x the original EP API defaults to CCO instead (`MORI_EP_COMM`). MORI-IO, MORI-CCO and the CCO-based EPv2 API do not use shmem.

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. Concepts](#1-concepts)
- [2. Initialization](#2-initialization)
- [3. Query APIs](#3-query-apis)
- [4. Memory Management](#4-memory-management)
- [5. P2P Address Translation](#5-p2p-address-translation)
- [6. Synchronization](#6-synchronization)
- [7. HIP Module Init (Triton Integration)](#7-hip-module-init-triton-integration)
- [8. Initialization Flags](#8-initialization-flags)
- [Device Bitcode and IR](#device-bitcode-and-ir)
- [Environment Variables](#environment-variables)
- [Source Files](#source-files)

## Quick Reference

```python
import mori

# Initialize from PyTorch process group
mori.shmem.shmem_torch_process_group_init("default")

# Query
my_rank = mori.shmem.shmem_mype()
num_ranks = mori.shmem.shmem_npes()

# Allocate symmetric memory
ptr = mori.shmem.shmem_malloc(size_in_bytes)

# Register existing buffer for RDMA
mori.shmem.shmem_buffer_register(tensor.data_ptr(), tensor.nbytes)

# P2P address translation (intra-node)
remote_ptr = mori.shmem.shmem_ptr_p2p(ptr, my_pe, dest_pe)

# Synchronize
mori.shmem.shmem_barrier_all()

# Cleanup
mori.shmem.shmem_free(ptr)
mori.shmem.shmem_finalize()
```

**Imports:**

| What | Import |
|------|--------|
| All shmem APIs | `import mori.shmem` or `from mori import shmem` |
| Init flags | `mori.shmem.MORI_SHMEM_INIT_WITH_MPI_COMM` |
| | `mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID` |

---

## 1. Concepts

### Symmetric Memory

Symmetric memory is GPU memory that is allocated at the same virtual offset across all participating PEs (Processing Elements / ranks). This enables RDMA hardware to directly access remote GPU memory using simple address arithmetic — no address translation tables needed at runtime.

### Processing Element (PE)

A PE is a participant in the symmetric memory domain. Each GPU rank maps to one PE. PEs are numbered 0 to N-1.

### Symmetric Heap

The symmetric heap is a pre-allocated region of GPU memory from which `shmem_malloc` allocates. Its size is controlled by the `MORI_SHMEM_HEAP_SIZE` environment variable:

```bash
export MORI_SHMEM_HEAP_SIZE=6G  # Must be set before shmem init
```

---

## 2. Initialization

Shmem must be initialized exactly once per process. There are three methods:

### Method 1: PyTorch Process Group (Recommended)

Use this when PyTorch distributed is already initialized. The process group must be registered with a name before calling `shmem_torch_process_group_init`:

```python
import torch
import torch.distributed as dist
import mori

# Standard PyTorch distributed init
torch.cuda.set_device(rank)
dist.init_process_group(
    backend="cpu:gloo,cuda:nccl",
    rank=rank, world_size=world_size,
    device_id=torch.device("cuda", rank),
)

# Register the process group with a name
world_group = dist.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)

# Initialize shmem from the registered process group
mori.shmem.shmem_torch_process_group_init("default")
```

> **Note:** The `_register_process_group("default", ...)` step is required — `shmem_torch_process_group_init` looks up the group by name to extract rank/world_size and bootstrap the shmem connections.

### Method 2: Unique ID (No PyTorch Distributed)

Use this when PyTorch distributed is not available (e.g., standalone applications, custom launchers). Rank 0 generates a unique ID and broadcasts it to all ranks via any transport (file, MPI, TCP, etc.):

```python
import mori

# Rank 0 generates a unique ID
if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()  # Returns 128 bytes
    # Broadcast unique_id to all ranks via your transport

# All ranks initialize with the same unique ID
mori.shmem.shmem_init_attr(
    mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
    rank,        # My rank
    world_size,  # Total ranks
    unique_id,   # Shared unique ID (bytes)
)
```

Example using file-based broadcast (for single-node testing):

```python
import os, time

uid_file = "/tmp/mori_unique_id"
if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()
    with open(uid_file, 'wb') as f:
        f.write(unique_id)
else:
    while not os.path.exists(uid_file):
        time.sleep(0.1)
    with open(uid_file, 'rb') as f:
        unique_id = f.read()

mori.shmem.shmem_init_attr(
    mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
    rank, world_size, unique_id,
)
```

### Method 3: MPI Communicator (C++ only)

Use this in C++ MPI applications, with MORI built with MPI support
(`MORI_WITH_MPI=ON`). Pass the communicator through the attribute struct:

```cpp
#include "mori/shmem/shmem_api.hpp"

mori::shmem::mori_shmem_init_attr_t attr{};
attr.mpi_comm = &comm;  // an MPI_Comm
mori::shmem::ShmemInitAttr(mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM, &attr);
```

> **Note:** There is no Python path for this method: `mori.shmem.shmem_init_attr` takes a
> unique ID and cannot carry an `MPI_Comm`. In Python, use Method 1 or Method 2.

### Initialization Comparison

| Method | When to use | Dependencies |
|--------|-------------|--------------|
| PyTorch Process Group | LLM inference/training frameworks (vLLM, SGLang, etc.) | `torch.distributed` |
| Unique ID | Standalone apps, custom launchers, non-PyTorch environments | None (any broadcast mechanism) |
| MPI Communicator | C++ MPI applications | MPI, and MORI built with `MORI_WITH_MPI=ON` |

### Finalization

Always finalize shmem before process exit:

```python
mori.shmem.shmem_finalize()
# Then destroy PyTorch process group if applicable
dist.destroy_process_group()
```

---

## 3. Query APIs

```python
# Get my PE (rank) ID — 0 to npes-1
my_pe = mori.shmem.shmem_mype()

# Get total number of PEs
total_pes = mori.shmem.shmem_npes()

# Get number of RDMA queue pairs per PE
num_qp = mori.shmem.shmem_num_qp_per_pe()
```

---

## 4. Memory Management

### Allocating Symmetric Memory

```python
# Basic allocation
ptr = mori.shmem.shmem_malloc(size_in_bytes)

# Aligned allocation (alignment must be power of 2)
ptr = mori.shmem.shmem_malloc_align(alignment=256, size=size_in_bytes)

# Allocation with flags
ptr = mori.shmem.shmem_ext_malloc_with_flags(size_in_bytes, flags)
```

All allocation functions return an integer address (`int`). The allocated memory is symmetric — the same offset is reserved on every PE.

### Freeing Symmetric Memory

```python
mori.shmem.shmem_free(ptr)
```

### Registering Existing Buffers

Register a PyTorch tensor or other existing GPU memory for RDMA operations without allocating new symmetric memory:

```python
# Register
tensor = torch.zeros(1024, 7168, dtype=torch.bfloat16, device="cuda")
mori.shmem.shmem_buffer_register(tensor.data_ptr(), tensor.nbytes)

# ... use tensor in MORI operations ...

# Deregister when done
mori.shmem.shmem_buffer_deregister(tensor.data_ptr(), tensor.nbytes)
```

Both functions return a status code (0 for success).

---

## 5. P2P Address Translation

For intra-node GPU-to-GPU access, translate a local symmetric pointer to its P2P-accessible address on a remote PE:

```python
remote_ptr = mori.shmem.shmem_ptr_p2p(local_ptr, my_pe, dest_pe)
```

| Return Value | Meaning |
|-------------|---------|
| Non-zero | P2P address on `dest_pe` (same-node, XGMI-connected GPUs) |
| 0 | Connection uses RDMA transport (different nodes) or pointer is invalid |

---

## 6. Synchronization

Global barrier — blocks until all PEs reach the barrier:

```python
mori.shmem.shmem_barrier_all()
```

---

## 7. HIP Module Init (Triton Integration)

When using Triton-compiled kernels that access shmem device symbols, initialize the HIP module:

```python
mori.shmem.shmem_module_init(hip_module_handle)
```

This copies the current GPU states to the `globalGpuStates` symbol in the dynamically compiled Triton kernel module.

---

## 8. Initialization Flags

| Flag | Value | Description |
|------|-------|-------------|
| `MORI_SHMEM_INIT_WITH_MPI_COMM` | 0 | Initialize using MPI communicator |
| `MORI_SHMEM_INIT_WITH_UNIQUEID` | 1 | Initialize using broadcast unique ID |

---

## Device Bitcode and IR

MORI-SHMEM device functions can be called from GPU kernels written in Triton, FlyDSL, MLIR or HIP. They are pre-compiled to LLVM bitcode (`libmori_shmem_device.bc`) as `extern "C"` wrappers. The `mori.ir` Python package locates the bitcode and gives the ABI metadata of every device function. Any framework that can link LLVM bitcode can use them. Triton is the reference integration.

For the device API of MORI-CCO (LSA, GDA, SDMA), see [CCO IR](MORI-CCO-GUIDE.md#cco-ir-flydsl-and-triton).

**Relationship with `mori.shmem`:** The device functions in the bitcode depend on
the shmem runtime (`globalGpuStates`) being initialized on the host. Before any
device function can be called, the application must:

1. Initialize shmem via `mori.shmem` host APIs (`shmem_torch_process_group_init` or `shmem_init_attr`) — see [Shmem Guide](MORI-SHMEM-GUIDE.md)
2. Allocate symmetric memory via `mori.shmem.shmem_malloc` or `mori_shmem_create_tensor`
3. For dynamically compiled kernels (e.g. Triton), call `mori.shmem.shmem_module_init(hip_module)` to copy GPU states into the kernel module

`mori.ir` handles the **compile/link-time** concern (find bitcode, generate wrappers), while `mori.shmem` handles the **runtime** concern (init, memory, state).

### Architecture

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


### Host-Side Python API

#### `mori.ir` (framework-agnostic)

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

#### `mori.ir.triton` (Triton-specific reference backend)

| Export | Description |
|--------|-------------|
| `get_extern_libs()` | Returns dict for Triton `extern_libs` parameter |
| `install_hook()` | Install Triton compilation hook for automatic bitcode linking |
| Device wrappers | All functions from `MORI_DEVICE_FUNCTIONS` callable inside `@triton.jit` |


### Device Bitcode

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


### Device Functions

All functions below are available as `extern "C"` symbols in the bitcode. In Triton, they are callable as `mori_shmem_device.<name>()` inside `@triton.jit` kernels. In other frameworks, call the C symbol directly via bitcode linking.

#### Query

| Function | Args | Return |
|----------|------|--------|
| `my_pe()` | — | `int32` |
| `n_pes()` | — | `int32` |

#### Point-to-Point

| Function | Args | Return |
|----------|------|--------|
| `ptr_p2p(ptr, my_pe, dest_pe)` | `uint64, int32, int32` | `uint64` |
| `ptr(dest, dest_pe)` | `uint64, int32` | `uint64` |

#### PutNbi (Thread / Warp / Block)

| Function | Args |
|----------|------|
| `putmem_nbi_thread(dest, src, nbytes, pe, qp)` | `ptr, ptr, size, int, int` |
| `putmem_nbi_warp(...)` | same |
| `putmem_nbi_block(...)` | same |

Typed variants: `put_uint32_nbi_*`, `put_uint64_nbi_*`, `put_float_nbi_*`, `put_double_nbi_*`

#### PutNbi with Signal

| Function | Args |
|----------|------|
| `putmem_nbi_signal_thread(dest, src, n, sig, val, op, pe, qp)` | 8 args |
| `putmem_nbi_signal_warp(...)` | same |
| `putmem_nbi_signal_block(...)` | same |

Signal ops: `SIGNAL_SET` (9), `SIGNAL_ADD` (10)

#### Immediate Put

| Function | Args |
|----------|------|
| `int32_p(dest, val, pe, qp)` | `ptr, int32, int, int` |
| `uint64_p(dest, val, pe, qp)` | `ptr, uint64, int, int` |
| `float_p(dest, val, pe, qp)` | `ptr, float32, int, int` |

#### Atomics

| Function | Return |
|----------|--------|
| `uint32_atomic_add_thread(dest, val, pe, qp)` | `int32` |
| `uint64_atomic_fetch_add_thread(dest, val, pe, qp)` | `uint64` |
| `atomic_uint32_nonfetch_thread(dest, val, op, pe, qp)` | `int32` |
| `atomic_uint64_fetch_thread(dest, val, cmp, op, pe, qp)` | `uint64` |

#### Wait

| Function | Args | Return |
|----------|------|--------|
| `uint64_wait_until_equals(addr, val)` | `ptr, uint64` | `int32` |
| `uint64_wait_until_greater_than(addr, val)` | `ptr, uint64` | `uint64` |
| `uint32_wait_until_equals(addr, val)` | `ptr, uint32` | `int32` |

#### Synchronization

| Function | Description |
|----------|-------------|
| `quiet_thread()` | Complete all pending remote operations (thread scope) |
| `quiet_thread_pe(pe)` | Complete pending ops to specific PE |
| `fence_thread()` | Order remote operations (thread scope) |
| `barrier_all_thread()` | Global barrier (thread scope) |
| `barrier_all_block()` | Global barrier (block scope) |

See `python/mori/ir/ops.py` for the complete function table with all C symbols.


### Integration Example: Triton

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

### Integration Example: Raw Bitcode (no framework)

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

### Bitcode JIT Compilation

The bitcode is **automatically JIT-compiled** on first use — no manual build step required. `find_bitcode()` compiles `shmem_device_api_wrapper.cpp` with `hipcc --cuda-device-only` and caches the result to `~/.mori/jit/`.

The NIC type (BNXT / AINIC / MLX5) and GPU architecture are auto-detected at runtime.

To precompile ahead of time:

```bash
MORI_PRECOMPILE=1 python -c "import mori"
```


### Examples

| File | What it demonstrates |
|------|---------------------|
| `examples/shmem/ir/test_triton_shmem.py` | Basic put/get via `mori.ir.triton` |
| `examples/shmem/ir/test_triton_allreduce.py` | Allreduce: P2P read + put+signal kernels |
| `examples/shmem/ir/test_mlir_shmem.py` | MLIR / LLVM IR paths (no Triton) |


### Testing

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


### Known Limitations

- Triton's `extern_elementwise` forces all device functions to return `int32` even when the C function returns `void`. This is a Triton upstream limitation.
- Pointer arguments are passed as `uint64` (intptr cast) since `extern_elementwise` does not support `pointer_type(void)`.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MORI_SHMEM_HEAP_SIZE` | Symmetric heap size (e.g., `"6G"`, `"2G"`, `"512M"`). Must be set before initialization. | 4 GB (`static_heap`), 16 GB (`vmm_heap`) |
| `MORI_SHMEM_MODE` | Heap mode: `"static_heap"`, `"vmm_heap"`, or `"isolation"`. Any other value logs a warning and falls back to `static_heap`. | `"static_heap"` |
| `MORI_SHMEM_HEAP_TYPE` | Heap memory type: `"normal"` (cached) or `"uncached"` | `"uncached"` |
| `MORI_SHMEM_VMM_CHUNK_SIZE` | VMM mode chunk size in bytes | Auto |
| `MORI_SOCKET_IFNAME` | Network interface for shmem bootstrap TCP connections (e.g., `"lo"`, `"eth0"`) | Auto-detect |
| `MORI_RDMA_DEVICES` | RDMA NIC selection. Include: `mlx5_0,mlx5_1`. Exclude: `^mlx5_2,mlx5_3` | All available |
| `LD_LIBRARY_PATH` | mori loads libibverbs dynamically at runtime (`dlopen`) instead of linking it; to use an out-of-tree libibverbs, put its directory here. | `libibverbs.so` / `libibverbs.so.1` |
| `MORI_NUM_QP_PER_PE` | Number of RDMA queue pairs per PE | `4` |
| `MORI_IB_GID_INDEX` | InfiniBand GID index for RDMA connections | Auto-detect |
| `MORI_RDMA_SL` | RDMA service level | Auto |
| `MORI_RDMA_TC` | RDMA traffic class | Auto |
| `MORI_DISABLE_P2P` | Disable P2P (XGMI) transport, force RDMA | Not set |
| `MORI_ENABLE_RAIL_ONLY` | Only create RDMA QPs to same-rail peers (same index within their node). For rail-isolated fabrics where cross-rail QPs cannot be established. See [Rail-only connections](#rail-only-connections). | Not set |
| `MORI_DISABLE_TOPO` | Disable topology detection | Not set |
| `MORI_IGNORE_CPU_AFFINITY` | By default the thread calling `ShmemInit` is pinned to the CPUs local to its GPU (sysfs `local_cpulist`, intersected with the existing cpuset). Within that set the CPUs are grouped into physical cores by `thread_siblings_list` and partitioned into one disjoint slice per GPU on the NUMA node, so two ranks never share the two SMT siblings of one core; the split applies only when the process sees more than one GPU on that NUMA node and there is at least one physical core per GPU, otherwise the whole node is bound. Set to `1` to disable and leave CPU placement to an outer `numactl`/`torchrun`. Works in SPMT (single-process multi-GPU) too: each per-GPU init thread pins to its own GPU's slice. | Not set (binding on) |
| `MORI_CPU_AFFINITY_NO_SPLIT` | Bind to the whole NUMA node instead of this rank's own slice of it. | Not set (split on) |
| `MORI_GLOBAL_LOG_LEVEL` | Global log verbosity: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR` | `INFO` |
| `MORI_PRECOMPILE` | Precompile all JIT kernels on import | Not set |
| `MORI_DISABLE_JIT` | Disable JIT compilation of device bitcode | Not set |

---

## Rail-only connections

By default `Context` builds a full mesh: every cross-node peer gets
`MORI_NUM_QP_PER_PE` queue pairs. On a **rail-isolated** fabric only NICs on
the same rail can reach each other, so the cross-rail QPs never reach RTR and
init hangs — even though no kernel would ever have used them.

`MORI_ENABLE_RAIL_ONLY=1` restricts QP creation to same-rail peers. Two ranks are
on the same rail when they have the same index within their own node, which is
the same relation the EP internode kernels already encode in their proxy:

```cpp
// src/ops/dispatch_combine/internode_v1.cpp
int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);
```

On a 2-node × 8-GPU job this takes each rank from 8 cross-node peers (32 QPs at
4 QP/PE) down to 1 (4 QPs). Skipped peers keep the same empty endpoint stubs
already used for non-RDMA peers, so `rdmaEps` indexing is unchanged; the
predicate is symmetric, so both ends stub each other and the handle AllToAll
stays aligned.

**This is opt-in, and only safe for kernels whose cross-node traffic is
same-rail.**

| Path | Cross-node RDMA target | Rail-only safe |
|------|------------------------|----------------|
| EP `InterNodeV1` / `InterNodeV1LL` | `proxyPe` | yes |
| EP `InterNode` (v0) | arbitrary `destPe` | no |
| EP `AsyncLL` | arbitrary `destPe` | no |
| CCO with `CCO_GDA_CONNECTION_RAIL` | same rail | yes |
| CCO with `FULL` / `CROSSNODE` | arbitrary peer | no |

Requirements checked at init — if any fails, mori logs an error and falls back
to the full mesh rather than mis-pairing ranks:

- node sizes are uniform,
- ranks are node-major contiguous (`rankInNode == rank % gpuPerNode`), which is
  what the EP proxy formula above assumes.

Same-host peers are never dropped: their RDMA loopback (used only under
`MORI_DISABLE_P2P`) does not traverse the fabric.

To confirm it took effect, look for this line from each rank:

```
rail-only: rank 3 rail 3 created QPs to 1 peers (4 QPs): 11
```

---

## Source Files

| File | Description |
|------|-------------|
| `python/mori/shmem/api.py` | Python shmem API |
| `python/mori/shmem/__init__.py` | Public exports |
| `include/mori/shmem/` | C++ shmem headers |
| `src/pybind/mori.cpp` | Python module entry point (`libmori_pybinds`) |
| `src/pybind/pybind_shmem.cpp` | Shmem binding registration (`RegisterMoriShmem`) |
