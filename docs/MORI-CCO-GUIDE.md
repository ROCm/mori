# MORI CCO Guide

**CCO** (Next-Generation Collective Communication Object for GPUs) is MORI's GPU
communication layer. It supports symmetric memory (symm mem) and is natively
designed for compute-communication fusion. It is built around an explicit
communicator handle. Unlike a process-global singleton,
every `ccoComm` is independently allocated, so a single process can hold multiple
independent communicators and drive them concurrently from multiple threads.

The fused GEMM + collective operators built on CCO have their own section,
[MORI-FUSED](MORI-CCO-FUSED-OPS-GUIDE.md).

CCO exposes GPU-initiated one-sided communication over three transports:

- **LSA** (Local Symmetric Access) — intra-node peer-to-peer over a flat
  symmetric virtual address space (XGMI). The kernel gets a peer's
  load/store-addressable pointer and writes it directly.
- **GDA** (GPU-Direct Async) — cross-node one-sided RDMA (put / get / signal /
  counter) issued from the device.
- **SDMA** — copy-engine transfers with a device-visible signal pool.

CCO is built for compute-communication fusion: a GEMM kernel can issue the
transfer itself, on symmetric memory, and overlap it with its own math. The
fused GEMM + collective operators (`mori.ops.gemm_ar`, `gemm_a2a`, `gemm_ag`,
`ag_gemm`, `a2a_gemm`) are described in the
[Fused GEMM + Collective Guide](MORI-CCO-FUSED-OPS-GUIDE.md).

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Concepts](#1-concepts)
3. [Initialization](#2-initialization)
4. [Memory and Windows](#3-memory-and-windows)
5. [Device Communicator](#4-device-communicator)
6. [Host Barrier](#5-host-barrier)
7. [Device-Side Programming](#6-device-side-programming)
8. [Examples](#7-examples)
9. [C++ API](#8-c-api)
10. [Environment Variables](#9-environment-variables)

## Quick Reference

```python
from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE

# Bootstrap: rank 0 mints a unique id, broadcast it out-of-band (MPI / torch.dist)
uid = Communicator.get_unique_id() if rank == 0 else None
uid = comm_mpi.bcast(uid, root=0)

# Create the communicator (reserves per-rank flat VMM)
with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
    # Allocate symmetric memory and register a P2P/RDMA window over it
    mem = comm.alloc_mem(4096)
    win = comm.register_window(mem.ptr, mem.size)

    # Create a device communicator (pass into kernels)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE
    reqs.lsa_barrier_count = 1
    dc = comm.create_dev_comm(reqs)

    comm.barrier()   # collective host barrier
# resources are released on context exit
```

Run with any launcher that can broadcast the unique id:

```bash
mpirun -np 8 python main.py         # mpi4py bootstrap
# or torchrun --standalone --nproc_per_node=8 main.py  (torch.distributed gloo)
```

## 1. Concepts

### Communicator (`ccoComm`)
An explicit handle created collectively by all ranks. Holds the flat symmetric
VA reservation, the per-rank slot allocator, intra-node topology, and transport
resources. Multiple communicators can coexist in one process.

### Symmetric memory and windows
`alloc_mem(size)` reserves GPU memory inside the communicator's flat VA at the
same offset on every rank. `register_window(ptr, size)` makes that region
peer-reachable: it P2P-maps the region to intra-node peers and registers an RDMA
memory region for cross-node peers. A registered window is therefore reachable
by **both** LSA (intra-node) and GDA (cross-node) with no extra setup.

### Device communicator (`ccoDevComm`)
A trivially-copyable host struct filled by `create_dev_comm`. It carries device
pointers plus topology and per-session resources (signal/counter pools,
barriers). Pass it **by value** into kernels — it lands in kernel-argument space,
so kernels read it without a GPU-memory dereference.

### Teams
Logical rank-subset descriptors (`world`, `lsa`, `cross-node`, `rail`) that let
device code address peers without hard-coding topology.

## 2. Initialization

CCO uses a self-contained socket bootstrap that needs only a 128-byte unique id.
Rank 0 generates it; you broadcast it to all ranks with any out-of-band channel
(MPI, `torch.distributed`, a file), then every rank calls `init`.

```python
from mpi4py import MPI
from mori.cco import Communicator

comm_mpi = MPI.COMM_WORLD
rank, nranks = comm_mpi.Get_rank(), comm_mpi.Get_size()

uid = Communicator.get_unique_id() if rank == 0 else None
uid = comm_mpi.bcast(uid, root=0)

comm = Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024)
# ... use comm ...
comm.destroy()   # or use `with Communicator.init(...) as comm:`
```

The rendezvous network interface is selected via `MORI_SOCKET_IFNAME`
(see [Environment Variables](#9-environment-variables)).

## 3. Memory and Windows

```python
mem = comm.alloc_mem(size)          # AllocatedMemory: .ptr, .size
win = comm.register_window(mem.ptr, mem.size)   # RegisteredWindow: .handle, .local_ptr
```

- `alloc_mem` / `register_window` are the two-step form; both track the resource
  on the communicator, so they are freed automatically on `comm.destroy()` (or
  context exit).
- Window registration is **collective**: all ranks must call it in the same order
  with the same size.
- `win.handle` is the device-side window handle to hand to kernels;
  `win.local_ptr` is this rank's local pointer into the window.

## 4. Device Communicator

`create_dev_comm` allocates the per-session device resources described by a
`CCODevCommRequirements`. Common fields:

| Field | Meaning |
|---|---|
| `gda_connection_type` | `GDA_CONNECTION_NONE` / `CROSSNODE` / `FULL` / `RAIL` — which peers get RDMA QPs |
| `gda_signal_count` | # of GDA signal slots (completion signalling) |
| `gda_counter_count` | # of GDA counter slots |
| `lsa_barrier_count` | # of intra-node (LSA) barriers |
| `sdma_queue_count` | # of SDMA queues (0 = default) |

```python
reqs = CCODevCommRequirements()          # safe defaults
reqs.gda_connection_type = GDA_CONNECTION_CROSSNODE
reqs.gda_signal_count = 128
dc = comm.create_dev_comm(reqs)
# dc.rank / dc.world_size / dc.lsa_size / dc.lsa_rank query topology
```

## 5. Host Barrier

```python
comm.barrier()   # collective across all ranks in the communicator
```

## 6. Device-Side Programming

Device kernels include the CCO header directly and take the `ccoDevComm` by value.

- **LSA (intra-node):** get a peer's directly-addressable pointer and
  load/store it in the kernel.

  ```cpp
  #include "mori/cco/cco.hpp"
  // win: ccoWindow_t obtained on the host
  void* peer = ccoGetLsaPeerPtr(win, peerLsaRank, offset);
  // *peer is a normal load/store target
  ```

- **GDA (cross-node):** GPU-initiated RDMA put/get with signalling, via the
  scale-out header. Provider dispatch is compile-time (per NIC). A handle's
  doorbell mode is compile-time too, `ccoGda<P, ccoGdaDoorbellMode>`:
  `Immediate` (default) posts ring for themselves, `Deferred` posts are released
  by the next ring that reaches them, and `CallerOrdered` posts are released
  only by that mode's flush, for callers that order every post before it.
  Handles on one context share its QPs whatever their mode; see
  `ccoGdaDoorbellMode` for the rules.

  ```cpp
  #include "mori/cco/cco_scale_out.hpp"   // pulls in cco.hpp + RDMA core
  ```

- **Barriers:** the LSA barrier session (`ccoLsaBarrierSession`) provides
  `arrive` / `wait` / `sync` at thread / warp / block granularity
  (`ccoCoopThread` / `ccoCoopWarp` / `ccoCoopBlock`); the GDA barrier session
  (`ccoGdaBarrierSession`) provides `sync` at warp or block granularity on an
  Immediate `ccoGda<P>` handle.

FlyDSL and Triton kernels can drive the same device API, through the bitcode described next.

### CCO IR (FlyDSL and Triton)

CCO IR lets a FlyDSL or Triton kernel call the CCO device API from inside the kernel. The C++ device implementation (`ccoGda<P, Mode>` and `ccoSdma`) is reused unchanged. It is the device side of MORI-CCO. The fused GEMM + collective operators use it.

**Difference from MORI-SHMEM:** CCO has no process-global device state (`globalGpuStates`), so there is no `shmem_module_init` and no `install_hook()`. The device communicator and the windows cross the kernel boundary as explicit `uint64` arguments: `dc.ptr` and `win.handle`.

#### Layers

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

#### Three transports, three models

| Transport | What the kernel gets |
|-----------|----------------------|
| **LSA** (intra-node) | `lsa_ptr`: the load/store address of a peer window. The kernel reads and writes it directly. CCO does not move the data |
| **GDA** (RDMA) | Device ops: `put`, `put_value`, `get`, `signal`, `wait_signal`, `flush`, `flush_async` and `wait`. Peers are world ranks, or GDA-team indices with `team=Team.GDA` |
| **SDMA** (copy engine) | `put` and `get` on windows or resolved pointers, with signals, `wait_signal` and `quiet`. Peers are LSA ranks |

#### Locate the bitcode

`find_cco_bitcode()` resolves in this order:

1. `MORI_CCO_BC`: an explicit path
2. JIT compile for the current GPU and NIC (cached in `~/.mori/jit`), unless `MORI_DISABLE_JIT` is set
3. A prebuilt file next to the package, in `<repo>/lib` or in `<repo>/build*/lib`. To build one, run `bash tools/build_cco_bitcode.sh`

FlyDSL uses code object version 6. Triton uses version 5. The bindings pick the right one.

#### Host setup

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

The host API is in sections 2–4 above.

#### Triton

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

#### FlyDSL

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

#### Requirements

- **SDMA**: build with `BUILD_CCO_SDMA=ON` (the default) and run with `MORI_ENABLE_SDMA=1`.
- **GDA on one node**: set `MORI_CCO_GDA_CONN=full`.
- **GDA provider**: fixed at build time, one per NIC, as for the C++ kernels.
- **FlyDSL**: `pip install flydsl`. Triton needs a Triton install.

#### Examples and tests

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

## 7. Examples

Runnable examples live in [`examples/cco/`](https://github.com/ROCm/mori/blob/main/examples/cco/):

| Example | Lang | Shows |
|---|---|---|
| `python/01_barrier` | py | host barrier across ranks (full lifecycle, no kernel) |
| `python/02_lsa_put` | py | intra-node LSA put via a hand-written `.hip` kernel |
| `python/03_flydsl_put` | py | FlyDSL GDA put + signal/wait |
| `python/04_flydsl_lsa_put` | py | FlyDSL LSA direct peer-pointer store |
| `python/05_flydsl_lsa_allreduce` | py | FlyDSL LSA custom all-reduce |
| `python/06_flydsl_gda_modes` | py | FlyDSL GDA (thread_mode, coop) × signal matrix, plus DEFERRED / CALLER_ORDERED handles |
| `cpp/01_lsa_put.cpp` | c++ | intra-node LSA put (includes only `cco.hpp`) |
| `cpp/02_gda_put.cpp` | c++ | GPU-initiated RDMA put + signal/wait (`cco_scale_out.hpp`) |

## 8. C++ API

The host control plane and device layer are a self-contained header pair;
include `cco.hpp` for host + LSA, or `cco_scale_out.hpp` for GDA.

```cpp
#include "mori/cco/cco.hpp"

ccoUniqueId id;
if (rank == 0) ccoGetUniqueId(&id);
/* broadcast id to all ranks */
ccoComm* comm;
ccoCommCreate(id, nRanks, rank, perRankVmmSize, &comm);

void* ptr;
ccoMemAlloc(comm, size, &ptr);
ccoWindow_t win;
ccoWindowRegister(comm, ptr, size, &win);   // or the size-only overload (alloc+register)

ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.gdaConnectionType = CCO_GDA_CONNECTION_CROSSNODE;
ccoDevComm devComm;
ccoDevCommCreate(comm, &reqs, &devComm);     // pass devComm by value into kernels

ccoBarrierAll(comm);

ccoDevCommDestroy(comm, &devComm);
ccoWindowDeregister(comm, win);
ccoMemFree(comm, ptr);
ccoCommDestroy(comm);
```

## 9. Environment Variables

| Variable | Purpose |
|---|---|
| `MORI_SOCKET_IFNAME` | Network interface for the unique-id socket rendezvous |
| `MORI_RDMA_DEVICES` | Comma-separated RDMA devices to use (GDA) |
| `MORI_RDMA_TC` | RDMA traffic class (GDA) |
| `MORI_RDMA_SL` | RDMA service level (GDA) |
| `MORI_IGNORE_CPU_AFFINITY` | `1` skips the CPU bind done at communicator creation, leaving placement to an outer `numactl`/`torchrun` |
| `MORI_CPU_AFFINITY_NO_SPLIT` | `1` binds to the whole NUMA node instead of this rank's own slice of it |

Communicator creation (`ccoCommCreate`) pins the calling thread to the CPUs
local to its current HIP device before the bootstrap and any worker thread, so
new threads inherit the affinity. The GPU-local CPU list (sysfs
`local_cpulist`) is intersected with the existing cpuset, so cgroup limits and
an outer binding are respected. Within that set the CPUs are grouped into
physical cores by `thread_siblings_list` and partitioned into one disjoint
slice per GPU on the NUMA node, so two ranks cannot end up on the two SMT
siblings of one core. The split applies only when the process sees more than
one GPU on that NUMA node and there is at least one physical core per GPU;
otherwise it falls back to binding the whole node.

Supported GPUs: MI308X / MI300X / MI325X / MI355X. Supported NICs: AMD Pollara
(AINIC), Mellanox ConnectX-7, Broadcom Thor2.
