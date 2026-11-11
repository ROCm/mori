# MORI-IR Guide

MORI-IR is the **integration layer** that enables external GPU kernel frameworks to use MORI's communication capabilities (P2P, RDMA/IBGDA, SDMA) from device code. It covers two device APIs:

- **CCO IR** — the device API of [MORI-CCO](MORI-CCO-GUIDE.md): LSA, GDA and SDMA over an explicit communicator. It is the API that CCO-based kernels use, including the fused GEMM + collective operators. See [CCO IR](#cco-ir).
- **SHMEM IR** — the device API of MORI-SHMEM (OpenSHMEM-style). See [SHMEM IR](#shmem-ir).

Each has two building blocks:

1. **Device bitcode** (`libmori_cco_device.bc` for CCO, `libmori_shmem_device.bc` for SHMEM) — pre-compiled LLVM bitcode containing the device functions (`extern "C"` wrappers), linkable by any GPU kernel framework
2. **IR Python API** — framework-agnostic utilities for locating the device bitcode and querying device function ABI metadata (`mori.cco.device` for CCO, `mori.ir` for SHMEM)

These components are **framework-agnostic by design**. Any system that can link LLVM bitcode — Triton, FlyDSL, MLIR-based compilers, custom HIP/LLVM toolchains, etc. — can integrate MORI's communication primitives into its device kernels. Triton is provided as the first **reference integration**; the same bitcode and ABI metadata apply to all other backends.

## Table of Contents

- [CCO IR](#cco-ir)
- [SHMEM IR](#shmem-ir)
- [Architecture](#architecture)
- [1. Host-Side Python API](#1-host-side-python-api)
- [2. Device Bitcode](#2-device-bitcode)
- [3. Device Functions](#3-device-functions)
- [4. Integration Example: Triton](#4-integration-example-triton)
- [5. Integration Example: Raw Bitcode (no framework)](#5-integration-example-raw-bitcode-no-framework)
- [6. Bitcode JIT Compilation](#6-bitcode-jit-compilation)
- [7. Examples](#7-examples)
- [8. Testing](#8-testing)
- [Known Limitations](#known-limitations)

## CCO IR

CCO IR lets a FlyDSL or Triton kernel call the CCO device API from inside the kernel. The C++ device implementation (`ccoGda<P, Mode>` and `ccoSdma`) is reused unchanged. It is the device side of [MORI-CCO](MORI-CCO-GUIDE.md). The fused GEMM + collective operators use it.

**Difference from SHMEM IR:** CCO has no process-global device state (`globalGpuStates`), so there is no `shmem_module_init` and no `install_hook()`. The device communicator and the windows cross the kernel boundary as explicit `uint64` arguments: `dc.ptr` and `win.handle`.

### Layers

```
CCO_DEVICE_FUNCTIONS (one scalar ABI table, mori.cco.device)
  ├─ FlyDSL: mori.cco.device.flydsl   (DevComm / Window / Gda / Sdma handles)
  │    └─ bitcode, code object v6
  └─ Triton: mori.ir.triton.cco       (DevComm / Window / Gda / Sdma namespaces)
       └─ bitcode, code object v5
            └─ libmori_cco_device.bc  (src/cco/device/cco_device_wrapper.cpp)
                 └─ ccoGda<P, Mode>::put<...>() / ccoSdma::put<...>() / LSA pointers
```

| Component | Role |
|-----------|------|
| `libmori_cco_device.bc` | LLVM bitcode with the `extern "C"` CCO device wrappers. Built per GPU arch, NIC and code object version |
| `mori.cco.device.find_cco_bitcode(cov)` | Locate the bitcode (see below) |
| `mori.cco.device.CCO_DEVICE_FUNCTIONS` | ABI metadata for the wrappers, shared by both frameworks |
| `mori.cco.device.flydsl` | FlyDSL bindings |
| `mori.ir.triton.cco` | Triton bindings. `get_extern_libs()` returns the `extern_libs` dict |

Each template axis (coop scope, thread mode, signal action, doorbell mode) is a separate `extern "C"` symbol. The framework picks the symbol at trace time, so the kernel makes one direct call and pays no dispatch cost.

### Three transports, three models

| Transport | What the kernel gets |
|-----------|----------------------|
| **LSA** (intra-node) | `lsa_ptr`: the load/store address of a peer window. The kernel reads and writes it directly. CCO does not move the data |
| **GDA** (RDMA) | Device ops: `put`, `put_value`, `get`, `signal`, `wait_signal`, `flush`, `flush_async` and `wait`. Peers are world ranks, or GDA-team indices with `team=Team.GDA` |
| **SDMA** (copy engine) | `put` and `get` on windows or resolved pointers, with signals, `wait_signal` and `quiet`. Peers are LSA ranks |

### Locate the bitcode

`find_cco_bitcode()` resolves in this order:

1. `MORI_CCO_BC`: an explicit path
2. JIT compile for the current GPU and NIC (cached in `~/.mori/jit`), unless `MORI_DISABLE_JIT` is set
3. A prebuilt file next to the package, in `<repo>/lib` or in `<repo>/build*/lib`. To build one, run `bash tools/build_cco_bitcode.sh`

FlyDSL uses code object version 6. Triton uses version 5. The bindings pick the right one.

### Host setup

Create a communicator, register a window, create a device communicator, and pass `dc.ptr` and `win.handle` to the kernel:

```python
from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE

with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
    mem = comm.alloc_mem(4096)
    win = comm.register_window(mem.ptr, mem.size)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE   # LSA / SDMA only
    dc = comm.create_dev_comm(reqs)
    # kernel(dc.ptr, win.handle, ...)
```

See the [CCO Guide](MORI-CCO-GUIDE.md) for the full host API.

### Triton

```python
import triton
import triton.language as tl
from mori.ir.triton import cco

@triton.jit
def lsa_copy(dev_comm, win, peer: tl.constexpr, N: tl.constexpr):
    me = cco.DevComm.lsa_rank(dev_comm)
    src_addr = cco.Window.lsa_ptr(win, me, 0)
    dst_addr = cco.Window.lsa_ptr(win, peer, 0)
    src = src_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    dst = dst_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs))

lsa_copy[(1,)](dc.ptr, win.handle, peer=1, N=256, extern_libs=cco.get_extern_libs())
```

`cco.DevComm`, `cco.Window`, `cco.Gda` and `cco.Sdma` are namespaces. Each call takes the raw handle as its first argument, for example `cco.Gda.put(dev_comm, peer, ...)`. `cco.GdaDeferred` and `cco.GdaCallerOrdered` are the same for the other doorbell modes.

### FlyDSL

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco

@flyc.kernel
def put_kernel(dev_comm: fx.Int64, send_win: fx.Int64, recv_win: fx.Int64, nbytes: fx.Int64):
    tid = fx.thread_idx.x
    gda = cco.DevComm(dev_comm).gda(0)      # build the handle once
    if tid == 0:
        gda.put(1, recv_win, 0, send_win, 0, nbytes,
                signal_op=cco.SignalOp.INC, signal_id=0,
                coop=cco.CoopScope.THREAD)
    gda.flush(coop=cco.CoopScope.BLOCK)
```

### Requirements

- **SDMA**: build with `BUILD_CCO_SDMA=ON` (the default) and run with `MORI_ENABLE_SDMA=1`.
- **GDA on one node**: set `MORI_CCO_GDA_CONN=full`.
- **GDA provider**: fixed at build time, one per NIC, as for the C++ kernels.
- **FlyDSL**: `pip install flydsl`. Triton needs a Triton install.

### Examples and tests

| File | What it demonstrates |
|------|---------------------|
| `examples/cco/python/03_flydsl_put` .. `07_flydsl_sdma` | FlyDSL GDA, LSA and SDMA kernels |
| `examples/cco/ir/test_triton_cco.py` | Triton kernels over LSA, SDMA and GDA |

```bash
# Triton, LSA only (2 GPUs)
torchrun --standalone --nproc_per_node=2 examples/cco/ir/test_triton_cco.py --transport lsa

# Triton, SDMA (2 GPUs)
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=2 \
    examples/cco/ir/test_triton_cco.py --transport sdma
```

Design notes, the file map and every GDA doorbell mode: [CCO device API](cco-device-api.md).

---

## SHMEM IR

The rest of this guide describes SHMEM IR: the device bitcode and Python API of MORI-SHMEM.

**Relationship with `mori.shmem`:** The device functions in the bitcode depend on
the shmem runtime (`globalGpuStates`) being initialized on the host. Before any
device function can be called, the application must:

1. Initialize shmem via `mori.shmem` host APIs (`shmem_torch_process_group_init` or `shmem_init_attr`) — see [Shmem Guide](MORI-SHMEM-GUIDE.md)
2. Allocate symmetric memory via `mori.shmem.shmem_malloc` or `mori_shmem_create_tensor`
3. For dynamically compiled kernels (e.g. Triton), call `mori.shmem.shmem_module_init(hip_module)` to copy GPU states into the kernel module

`mori.ir` handles the **compile/link-time** concern (find bitcode, generate wrappers), while `mori.shmem` handles the **runtime** concern (init, memory, state).

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
