# Mori IR — Device Bitcode Integration Guide

Mori IR provides `libmori_shmem_device.bc`, a bitcode library containing all
mori shmem device functions (`extern "C"` wrappers for P2P, RDMA/IBGDA, and
SDMA communication).  Any GPU kernel framework that can link LLVM bitcode can
use it.

The `mori.ir` Python package provides:

- **`mori.ir`** — framework-agnostic bitcode locator and device function ABI metadata
- **`mori.ir.triton`** — shmem Triton integration layer
- **`mori.ir.triton.cco`** — CCO LSA/SDMA/GDA Triton integration backed by
  `libmori_cco_device.bc`

`mori.ir` is designed to be framework-agnostic. Triton is the first supported
backend; the same bitcode and ABI metadata can be used to integrate with
FlyDSL, MLIR-based compilers, or any framework that supports LLVM bitcode linking.

## Quick Start (Triton as example)

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
    npes = mori_shmem_device.n_pes()
    next_pe = (pe + 1) % npes

    # P2P pointer resolution
    remote = mori_shmem_device.ptr_p2p(buf_ptr.to(tl.uint64, bitcast=True), pe, next_pe)
    remote_ptr = remote.to(tl.pointer_type(tl.bfloat16), bitcast=True)
    data = tl.load(remote_ptr + tl.arange(0, BLOCK))

    # Or use put+signal for RDMA
    mori_shmem_device.putmem_nbi_signal_block(dest, src, nbytes, sig, val, SIGNAL_ADD, pe, 0)
    mori_shmem_device.quiet_thread()

my_kernel[(grid,)](buf, N, BLOCK=1024, extern_libs=get_extern_libs())
```

## Quick Start (CCO + Triton)

CCO host state is explicit: create a `Communicator`, register a window, create
a DevComm, then pass `dc.ptr` and `win.handle` as `uint64` kernel arguments.
Unlike shmem, CCO has no `globalGpuStates` symbol and needs no
`install_hook()`.

```python
import triton
import triton.language as tl
from mori.ir.triton import cco

@triton.jit
def lsa_copy(dev_comm, win, peer: tl.constexpr, N: tl.constexpr):
    me = cco.devcomm_lsa_rank(dev_comm)
    src_addr = cco.lsa_ptr(win, me, 0)
    dst_addr = cco.lsa_ptr(win, peer, 0)
    src = src_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    dst = dst_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    offs = tl.arange(0, N)
    tl.store(dst + offs, tl.load(src + offs))

lsa_copy[(1,)](
    dc.ptr,
    win.handle,
    peer=1,
    N=256,
    extern_libs=cco.get_extern_libs(),
)
```

The flat CCO functions mirror the monomorphized C symbols. Examples:

- `devcomm_rank`, `devcomm_lsa_rank`, `lsa_ptr`
- `sdma_put_{thread,warp,block}` and `sdma_quiet_*`
- `gda_put_{it,iw,ib,at}_{none,inc,add}`
- `gda_get_{it,iw,ib,at}`, signal/wait/read/reset, and flush variants

SDMA requires a MORI build with `BUILD_CCO_SDMA=ON` and runtime
`MORI_ENABLE_SDMA=1`. GDA peers are world ranks; LSA/SDMA peers are LSA ranks.

## Quick Start (Bitcode only, no Triton)

```python
from mori.ir import find_bitcode

bc = find_bitcode()
# Use llvm-link to link bc into your GPU kernel:
#   llvm-link my_kernel.bc <bc> -o linked.bc
#   clang -target amdgcn-amd-amdhsa -mcpu=gfx942 linked.bc -o kernel.hsaco
```

## Available Device Functions

The following device functions are defined in `libmori_shmem_device.bc`.
In Triton they are accessible as `mori_shmem_device.<name>()` inside
`@triton.jit` kernels; in other frameworks, call the C symbol directly
via bitcode linking.

### Query

| Function | C symbol | Args | Return |
|----------|----------|------|--------|
| `my_pe()` | `mori_shmem_my_pe` | — | `int32` |
| `n_pes()` | `mori_shmem_n_pes` | — | `int32` |

### Point-to-Point

| Function | C symbol | Args | Return |
|----------|----------|------|--------|
| `ptr_p2p(ptr, my_pe, dest_pe)` | `mori_shmem_ptr_p2p` | `uint64, int32, int32` | `uint64` |
| `ptr(dest, dest_pe)` | `mori_shmem_ptr` | `uint64, int32` | `uint64` |

### PutNbi (Thread / Warp / Block)

| Function | C symbol | Args |
|----------|----------|------|
| `putmem_nbi_thread(d, s, n, pe, qp)` | `mori_shmem_putmem_nbi_thread` | `ptr, ptr, size, int, int` |
| `putmem_nbi_warp(...)` | `mori_shmem_putmem_nbi_warp` | same |
| `putmem_nbi_block(...)` | `mori_shmem_putmem_nbi_block` | same |

### PutNbi with Signal

| Function | C symbol |
|----------|----------|
| `putmem_nbi_signal_thread(d, s, n, sig, val, op, pe, qp)` | `mori_shmem_putmem_nbi_signal_thread` |
| `putmem_nbi_signal_warp(...)` | `mori_shmem_putmem_nbi_signal_warp` |
| `putmem_nbi_signal_block(...)` | `mori_shmem_putmem_nbi_signal_block` |

Signal ops: `from mori.ir import SIGNAL_SET, SIGNAL_ADD`

### Immediate Put

| Function | C symbol |
|----------|----------|
| `int32_p(dest, val, pe, qp)` | `mori_shmem_int32_p` |
| `int64_p(dest, val, pe, qp)` | `mori_shmem_int64_p` |
| `uint64_p(dest, val, pe, qp)` | `mori_shmem_uint64_p` |
| `float_p(dest, val, pe, qp)` | `mori_shmem_float_p` |

### Atomics

| Function | C symbol |
|----------|----------|
| `uint32_atomic_add_thread(d, v, pe, qp)` | `mori_shmem_uint32_atomic_add_thread` |
| `uint64_atomic_fetch_add_thread(d, v, pe, qp)` | `mori_shmem_uint64_atomic_fetch_add_thread` |

### Wait

| Function | C symbol | Return |
|----------|----------|--------|
| `uint64_wait_until_equals(addr, val)` | `mori_shmem_uint64_wait_until_equals` | `int32` |
| `uint64_wait_until_greater_than(addr, val)` | `mori_shmem_uint64_wait_until_greater_than` | `uint64` |

### Synchronization

| Function | C symbol |
|----------|----------|
| `quiet_thread()` | `mori_shmem_quiet_thread` |
| `fence_thread()` | `mori_shmem_fence_thread` |
| `barrier_all_thread()` | `mori_shmem_barrier_all_thread` |
| `barrier_all_block()` | `mori_shmem_barrier_all_block` |

See `python/mori/ir/ops.py` for the complete list.

## Bitcode: JIT Compilation

The bitcode is **automatically JIT-compiled** on first use — no manual build step
required. `find_bitcode()` compiles `shmem_device_api_wrapper.cpp` with
`hipcc --cuda-device-only` and caches the result to `~/.mori/jit/`.

The NIC type (BNXT/IONIC/MLX5) and GPU architecture are auto-detected at runtime,
ensuring the bitcode contains the correct IBGDA provider for the hardware.

To precompile all kernels and bitcode ahead of time:
```bash
MORI_PRECOMPILE=1 python -c "import mori"
```

To manually build bitcode from the command line (e.g. for CI or non-Python workflows):
```bash
bash tools/build_shmem_bitcode.sh [output_dir] [gpu_arch]
```

## Examples

| File | What it demonstrates |
|------|---------------------|
| `examples/shmem/ir/test_triton_shmem.py` | Basic put/get via `mori.ir.triton` |
| `examples/shmem/ir/test_triton_allreduce.py` | Allreduce: P2P read + put+signal kernels |
| `examples/shmem/ir/test_mlir_shmem.py` | MLIR / LLVM IR paths (no Triton) |
| `examples/cco/ir/test_triton_cco.py` | CCO DevComm + LSA/SDMA/GDA via Triton |

## Testing

### Triton basic tests (2 GPUs)

```bash
torchrun --nproc_per_node=2 examples/shmem/ir/test_triton_shmem.py
```

### Triton allreduce — P2P mode (8 GPUs)

```bash
torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py
```

### Triton allreduce — IBGDA/RDMA mode (8 GPUs)

```bash
MORI_DISABLE_P2P=ON torchrun --nproc_per_node=8 examples/shmem/ir/test_triton_allreduce.py
```

### MLIR + LLVM IR paths (2 GPUs, no Triton)

```bash
cd examples/shmem/ir
bash run.sh 2 gfx942
```

### CCO Triton (2 GPUs)

```bash
# LSA
torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport lsa

# SDMA
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport sdma

# GDA-FULL on one AINIC rail
MORI_DEVICE_NIC=ionic MORI_DISABLE_TOPO=1 MORI_RDMA_DEVICES=rocep9s0 \
  torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport gda
```

## Known Limitations

- Triton's `extern_elementwise` forces all device functions to return `int32`
  even when the C function returns `void`.  This is a Triton upstream limitation,
  not a mori issue.
- Pointer arguments are passed as `uint64` (intptr cast) since
  `extern_elementwise` does not support `pointer_type(void)`.
