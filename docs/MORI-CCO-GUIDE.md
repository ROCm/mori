# MORI CCO Guide

**CCO** (Next-Generation Collective Communication Object for GPUs) is MORI's GPU
communication layer. It supports symmetric memory (symm mem) and is natively
designed for compute-communication fusion. It is built around an explicit
communicator handle. Unlike a process-global singleton,
every `ccoComm` is independently allocated, so a single process can hold multiple
independent communicators and drive them concurrently from multiple threads.

The fused GEMM + collective operators built on CCO have their own section,
[MORI-FUSED](MORI-CCO-FUSED-OPS-GUIDE.md).

CCO exposes GPU-initiated one-sided communication over four transports:

- **LSA** (Local Symmetric Access) — intra-node peer-to-peer over a flat
  symmetric virtual address space (XGMI). The kernel gets a peer's
  load/store-addressable pointer and writes it directly.
- **GDA** (GPU-Direct Async) — cross-node one-sided RDMA (put / get / signal /
  counter) issued from the device.
- **SDMA** — copy-engine transfers with a device-visible signal pool.
- **ccoProxy** — cross-node RDMA with the GDA window API, where the kernel
  queues each op and a host thread posts it
  (see [ccoProxy](#10-ccoproxy-host-proxy-rdma)).

CCO is built for compute-communication fusion: a GEMM kernel can issue the
transfer itself, on symmetric memory, and overlap it with its own math. The
fused GEMM + collective operators (`mori.ops.gemm_ar`, `gemm_a2a`, `gemm_ag`,
`ag_gemm`, `a2a_gemm`) are described in the
[Fused GEMM + Collective Guide](MORI-CCO-FUSED-OPS-GUIDE.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [1. Concepts](#1-concepts)
- [2. Initialization](#2-initialization)
- [3. Memory and Windows](#3-memory-and-windows)
- [4. Device Communicator](#4-device-communicator)
- [5. Host Barrier](#5-host-barrier)
- [6. Device-Side Programming](#6-device-side-programming)
- [7. Examples](#7-examples)
- [8. C++ API](#8-c-api)
- [9. Environment Variables](#9-environment-variables)
- [10. ccoProxy (host-proxy RDMA)](#10-ccoproxy-host-proxy-rdma)

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
| `sdma_queue_count` | # of SDMA queues per peer. 0 (the default) creates none; SDMA also needs `MORI_ENABLE_SDMA=1`. The first device communicator on a communicator that asks for queues fixes the count |

The NICs are opened only when a peer needs RDMA. That happens at communicator
creation when some peers are outside the LSA team, or at the first device
communicator with a `FULL` GDA or ccoProxy connection (NIC loopback to LSA
peers). A job whose ranks all share one LSA team, such as a single node or a
scale-up fabric spanning hosts, never touches a NIC under `NONE`, `CROSSNODE` or
`RAIL`.

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
    reqs.sdma_queue_count = 2                        # only for SDMA; 0 = no queues
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

`cco.DevComm`, `cco.Window`, `cco.Gda` and `cco.Sdma` are namespaces. Each call takes the raw handle as its first argument, for example `cco.Gda.put(dev_comm, peer, ...)`. `cco.GdaDeferred` is the same for a deferred-doorbell handle. `cco.GdaCallerOrdered`, for a caller-ordered handle, differs: it has no `signal` (put with a `signal_op` instead), and only its own `flush`, `flush_peer` and `flush_async` release its posts.

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

- **SDMA**: build with `BUILD_CCO_SDMA=ON` (the default), run with `MORI_ENABLE_SDMA=1`, and set `sdma_queue_count > 0` on the `CCODevCommRequirements`. The environment variable only permits SDMA: with a count of 0 the communicator has no queues and every SDMA put moves nothing.
- **GDA on one node**: set `gda_connection_type = GDA_CONNECTION_FULL` on the `CCODevCommRequirements` ([4. Device Communicator](#4-device-communicator)). Example 03 exposes this as `MORI_CCO_GDA_CONN=full`; the library does not read that variable.
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
| `cpp/03_proxy_put.cpp` | c++ | ccoProxy put + SignalSet/wait on a proxy-only DevComm, with timeouts |

## 8. C++ API

The host control plane and device layer are a self-contained header pair;
include `cco.hpp` for host + LSA, or `cco_scale_out.hpp` for GDA and ccoProxy.

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

## 10. ccoProxy (host-proxy RDMA)

`ccoProxy` (in `mori/cco/cco_scale_out.hpp`) is a cross-node transport with the
same window and signal model as GDA, but the GPU does not drive the NIC. A
kernel pushes each op as a small descriptor into a device-to-host queue (KIWI);
host proxy threads pop it and post the RDMA work request on host-created RC
QPs. Keys and remote offsets are resolved on the GPU exactly as for GDA, so the
same windows and signal slots work for both.

Use it where GPU-initiated RDMA is unavailable or slow, or where NIC atomics
into GPU memory fail (e.g. ionic VFs in VMs; signal with the atomic-free
`ccoGda_SignalSet`). The cost is host CPU: each DevComm with proxy queues runs
busy-polling proxy threads.

### Enabling it

- **Build:** `BUILD_CCO_PROXY` (CMake option and `setup.py` env var) is `ON` by
  default. The host engine is built when `GPU_TARGETS` contains gfx942 or
  gfx950, and at run time `ccoDevCommCreate` accepts proxy queues only on those
  GPUs. Targets in mori's own CMake build that link `mori_cco` see
  `MORI_CCO_PROXY` when the engine is built in. Code built against the installed
  headers or the wheel finds out at run time: `ccoDevCommCreate` rejects proxy
  queues.
- **Requirements:** request proxy queues per DevComm. Pass the same queue,
  thread and QP counts, connection type and `gdaSignalCount` on every rank. If
  the ranks disagree, a rank cannot run the proxy, or the proxy engine fails to
  set up, `ccoDevCommCreate` fails on every rank; the communicator stays usable
  and you can retry with other requirements.
- **Versions:** compile kernels against the headers of the mori they run with.
  `ccoDevComm` and the proxy descriptors have no stable layout yet.

| Field (C++ / Python) | Meaning |
|---|---|
| `proxySpscQueueCount` / `proxy_spsc_queue_count` | SPSC queues (one wavefront at a time per queue) |
| `proxyMpscQueueCount` / `proxy_mpsc_queue_count` | MPSC queues (any number of wavefronts, and `ccoCoopThread`) |
| `proxyThreadCount` / `proxy_thread_count` | host proxy threads (0 = 4), capped at the queue count |
| `proxyQueueCapacity` / `proxy_queue_capacity` | ring size in 64-byte lines, power of two (0 = 4096). A lane-masked or thread call takes up to 2 lines per calling lane; a call that cannot fit in the ring traps |
| `proxyQpsPerPeer` / `proxy_qps_per_peer` | RC QPs per (proxy thread, peer) (0 = 4). A thread creates at most one per queue it polls, and each queue uses one of them per peer (round robin over the thread's queues) |
| `proxyConnectionType` / `proxy_connection_type` | peers that get proxy QPs: a `CCO_GDA_CONNECTION_*` value, or `CCO_PROXY_CONNECTION_FOLLOW_GDA` / `PROXY_CONNECTION_FOLLOW_GDA` (-1, the default) to follow `gdaConnectionType` |

- `proxyConnectionType` chooses the proxy's peers independently of the GDA QPs
  and resolves like `gdaConnectionType`: CROSSNODE on a single node, no RDMA,
  or no matching peer give NONE, and proxy queues on a NONE connection fail
  creation.
- For a proxy-only DevComm, set `gdaConnectionType = CCO_GDA_CONNECTION_NONE`
  and `proxyConnectionType` to the peers you need. No GDA QPs are created.
- Signals use the GDA signal slots, so set `gdaSignalCount`. The other `gda*`
  fields do not affect ccoProxy: `gdaContextCount` sizes the GDA QPs, and
  `gdaCounterCount` reserves counter slots ccoProxy does not use.
- The GPU's VRAM must be host-mapped (large BAR).
- Create one DevComm with proxy queues per communicator: each starts its own
  busy-polling threads, pinned by the same CPU policy, so several of them poll
  on the same cores.
- For full bandwidth, use large messages (at least a few hundred KB) and set
  `proxyQpsPerPeer` / `proxyThreadCount`. On NICs that serve each QP serially
  (e.g. ionic), use several QPs per peer so a thread's queues run in parallel.

| Variable | Purpose |
|---|---|
| `MORI_CCO_DISABLE_PROXY` | `1` makes `ccoDevCommCreate` reject proxy queues |
| `MORI_CCO_PROXY_CPUS` | cpulist for the proxy threads (thread t on entry t mod size). Default: GPU-local cores in the affinity mask, one physical core per thread, from the top, skipping the core of CPU 0. It applies to the whole process tree: with several ranks per node, set it per rank to disjoint CPUs, or every rank's busy-polling threads share the same cores |

### Device API

Construct `ccoProxy<Queue>` in the kernel from the DevComm and a queue index.
`Queue` is `mori::cco::proxy::queue::SPSCD2HQueue` (the default) or
`mori::cco::proxy::queue::MPSCD2HQueue`; the short names need a `using`
declaration.

| Call | Does |
|---|---|
| `ccoProxy<>(dc, i)` / `ccoProxy<mori::cco::proxy::queue::MPSCD2HQueue>(dc, i)` | SPSC queue `i` (out of range traps); MPSC queue `i` mod the pool size. An empty pool traps |
| `put<CCO_TEAM_GDA>(peer, ...)` etc. | `peer` indexes the team of the proxy's connection type (`dc.proxy.connType`), as `ccoGda` does for `gdaConnectionType`; the default `CCO_TEAM_WORLD` takes world ranks |
| `put(peer, dstWin, dstOff, srcWin, srcOff, bytes, action, coop)` | RDMA write, optionally followed by a signal on the same QP; split above the port's max message size, with the signal on the last piece |
| `putValue(peer, dstWin, dstOff, value, action, coop)` | inline write of a 1–8 byte value, optionally followed by a signal |
| `signal(peer, action, coop)` | signal only |
| action | `ccoGda_NoSignal`, `ccoGda_SignalInc` / `SignalAdd` / `WindowSignalAdd` (NIC atomics), `ccoGda_SignalSet` / `WindowSignalSet` (8-byte write, no atomic) |
| `flush(coop)` | waits until every op pushed to this queue before the flush has completed remotely; then the sources can be reused. It does not notify the peer: send a signal to publish the data |
| `int flush(coop, timeoutCycles)` | never traps: 0 done, 1 timed out (earlier ops may still be in flight; a later flush covers them), 2 DevComm failed. `timeoutCycles` counts `clock64` cycles, whose rate follows the shader clock |
| `int waitSignal(id, least, coop, timeoutCycles)` | never traps: 0 consumed, 1 timed out, 2 DevComm failed; consumes only on 0. There is no `bits` argument, unlike `ccoGda` |
| `readSignal` / `waitSignal` / `resetSignal` | local signal slots, shared with GDA (see [Signal slots](#signal-slots)) |
| `failed()` | whether the DevComm has failed |

**Fences.** Every put, putValue and signal does a system-scope release fence
before it pushes, so stores made earlier are visible to the NIC. `waitSignal`
and `flush` end with a system-scope acquire; after `readSignal`, or after
polling a window yourself, issue one.

**SPSC queues.** Use an SPSC queue from one wavefront at a time. The handle
saves its state in its destructor (or `release()`).
- To hand a queue to another wavefront in the same kernel, call `release()` (or
  let the handle go out of scope) in the old owner, then publish with an
  agent-scope release (e.g. a flag store); the new owner does an agent-scope
  acquire of it before constructing its handle. `__syncthreads()` does both
  within a block.
- Across launches on one stream, stream order is enough. Give kernels that may
  run concurrently (on other streams) different SPSC queues.
- If you only call `readSignal`, `waitSignal`, `resetSignal` or `failed` on a
  handle, you may construct it with any queue index, even one another
  wavefront is pushing to.

### Choosing a Coop

`put`, `putValue` and `signal` take no default Coop (`ccoGda`'s default is
`ccoCoopThread`), so pick one of these:

| Coop | Who calls | Ops per call | Queues | Signals |
|---|---|---|---|---|
| `ccoCoopWarp` | all 64 lanes of the wavefront, converged, same arguments | one | SPSC, MPSC | one |
| `ccoCoopBlock` | the whole block, converged, same arguments; wavefront 0 issues | one | SPSC, MPSC | one |
| `ccoCoopLaneMasked{active}` | all 64 lanes, converged | one per lane with `active`, in one push, in lane order | SPSC, MPSC | one per lane |
| `ccoCoopThread` | any lanes: from divergent code, or a single lane | one per calling lane | MPSC only | `put` / `putValue`: one per peer group; `signal`: one per lane |

- Builds without `NDEBUG` check that `ccoCoopWarp` / `ccoCoopBlock` lanes pass
  the same arguments. A partial wavefront traps with every Coop but
  `ccoCoopThread`. For `ccoCoopBlock`, wavefront 0 (linear thread ids 0–63,
  also in 2-D and 3-D blocks) issues.
- `ccoCoopThread`, as in `ccoGda`: the calling lanes of one call enter the queue
  together, in lane order. A `put` / `putValue` carrying a signal signals once
  per group of lanes with the same peer, after the group's data, with the
  group's highest lane's arguments: count expected signals per group.
- `ccoGdaThreadAggregate` (`put<CCO_TEAM_WORLD, ccoGdaThreadAggregate>(...)`)
  promises that all calling lanes target one peer, so the call skips the
  grouping; builds without `NDEBUG` check it. It needs `ccoCoopThread`.
- SPSC queues reject `ccoCoopThread`, and `ccoGda` rejects `ccoCoopLaneMasked`,
  at compile time.
- Call `flush()` with `ccoCoopWarp` or `ccoCoopBlock`, as in `ccoGda`: after
  lane-masked or thread ops, flush from the converged wavefront.
  `flush(ccoCoopBlock)` also flushes every wavefront of the block that pushed
  through its own handle.

How to choose:

| You want | Use |
|---|---|
| the lowest cost per op, with a queue per wavefront or block | SPSC with `ccoCoopWarp` / `ccoCoopBlock` |
| many wavefronts sharing a few queues | MPSC |
| a different peer, offset or size per thread, from a full wavefront | `ccoCoopLaneMasked`: one push for all lanes |
| `ccoGda` thread code as is (divergent calls, single lanes) | `ccoCoopThread` on MPSC |
| all calling lanes go to one peer | add `ccoGdaThreadAggregate` |

Prefer SPSC queues and one push per wavefront where your code allows: an SPSC
push needs no atomic, while each MPSC push claims its lines with a device-memory
atomic. In new code, avoid calling ccoProxy from divergent code: each
`ccoCoopThread` call pushes on its own, so calls on both sides of a branch claim
twice. Compute the per-lane arguments and call once from converged code, e.g.
`put(c ? p1 : p2, ...)` or `ccoCoopLaneMasked{active}`.

### Signal slots

A slot is a 64-bit word that remote signals add to (`SignalInc` /
`SignalAdd`) or overwrite (`SignalSet`), plus a count of what local waits have
consumed. The semantics are the same as `ccoGda`'s:

- `readSignal(id)` returns the unconsumed amount: the slot's value minus what
  was consumed.
- `waitSignal(id, least, coop)` waits until at least `least` is unconsumed,
  then consumes `least`. It is not an absolute target: a second
  `waitSignal(id, n)` waits for `n` more. (`ccoSdma::waitSignal` does take an
  absolute value.)
- Have one waiter per slot at a time: two concurrent waiters on one slot can
  both take the same arrival. `waitSignal` does not take `ccoCoopLaneMasked`.
- Call `resetSignal(id)` only when no signal to the slot is in flight. It zeroes
  the slot and its consumed count.
- Use slot ids from 0 to `gdaSignalCount - 1`. Unlike `ccoGda`, `ccoProxy`
  rejects a signal to a larger id on a peer: the DevComm fails with
  `BAD_REQUEST` before anything is posted. The local `readSignal` /
  `waitSignal` / `resetSignal` do not check the id, in either transport.

With `SignalSet`, the sender sets 1, 2, 3, ... (one per round) and the receiver
calls `waitSignal(slot, 1)` once per round.

### Ordering and failure

- Ops one wavefront pushes to one queue reach each peer in program order on one
  QP, so a put followed by a signal (or a put carrying a signal) to the same
  peer is delivered data first. Different queues are unordered, as different
  GDA contexts are, and so are `ccoGda` and `ccoProxy` ops, even to the same
  peer. On an MPSC queue, the ops of different wavefronts interleave.
- For `SignalSet` / `WindowSignalSet`, send every set of a slot through one
  queue (sets through two queues can land out of order and move the slot back),
  use increasing values, and don't also use `SignalInc` / `SignalAdd` on that
  slot. A set is a plain RDMA write (as is a `putValue` used as a flag), so
  the data before it is visible first only if writes reach GPU memory in the
  order the NIC issued them:
  - don't use relaxed ordering on the window MRs (`ccoDevCommCreate` warns when
    `MORI_IB_ENABLE_RELAXED_ORDERING=1` applies it);
  - run on a NIC-to-GPU path that keeps PCIe write order, including under
    SR-IOV VFs and device passthrough.
- The first proxy error fails the DevComm for good: later ops are dropped (the
  proxy keeps draining the queues, so pushes still return), `failed()` returns
  true, and `flush(coop, timeoutCycles)` and
  `waitSignal(id, least, coop, timeoutCycles)` return 2. On the host,
  `ccoDevCommGetError(comm, &dc, &err)` returns 1 and the first error (code,
  peer, queue, op, completion status, message). You can call it while kernels
  run.
- A trap ends the process: `flush()` and `waitSignal()` without a timeout trap
  on a failed DevComm, as do usage errors, and HIP then reports the queue error
  and aborts. Builds without `NDEBUG` print the reason first. To keep the process, use the timeout overloads or `failed()`.
- Only the rank that hit the error sees it; peers are not notified. A rank that
  only receives keeps waiting for data that will not come, so give receivers
  their own timeout (`waitSignal(..., timeoutCycles)`).
- To recover, detect the failure (a 2, `failed()`, or `ccoDevCommGetError`),
  agree on it across ranks out of band, then have every rank call
  `ccoDevCommDestroy` and create a new DevComm. Other DevComms are unaffected.
- Teardown is local, as for GDA: `ccoDevCommDestroy` and `ccoCommDestroy` never
  wait for other ranks. Before destroying, make sure no peer still pushes to
  this rank: end the last kernels with `flush()`, synchronize, then call
  `ccoBarrierAll`. An op still in flight to a rank that has destroyed fails on
  the sender after the QP's ACK timeout (about 0.5 s by default) and is
  reported as a teardown error.
- Destroy each DevComm once, on the struct `ccoDevCommCreate` filled. After
  `ccoCommDestroy`, the only valid call on it is `ccoDevCommDestroy(nullptr,
  &dc)`.

### Example

```cpp
#include "mori/cco/cco_scale_out.hpp"

using namespace mori::cco;

// Host: reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;  // no GDA QPs
//       reqs.proxyConnectionType = CCO_GDA_CONNECTION_FULL;
//       reqs.proxySpscQueueCount = nBlocks; reqs.gdaSignalCount >= nBlocks.
// Round r = 1, 2, ...: block b sends chunk b and sets slot b to r (no NIC atomics).
__global__ void send(ccoDevComm dc, ccoWindow_t dst, ccoWindow_t src, size_t chunk, int peer,
                     uint64_t round) {
  ccoProxy<> proxy{dc, static_cast<int>(blockIdx.x)};
  const size_t off = blockIdx.x * chunk;
  proxy.put(peer, dst, off, src, off, chunk, ccoGda_SignalSet{blockIdx.x, round},
            ccoCoopBlock{});
  proxy.flush(ccoCoopBlock{});  // src may be overwritten now
}

// Once per round: consumes one, so chunk blockIdx.x of this round has landed.
__global__ void recv(ccoDevComm dc) {
  ccoProxy<> proxy{dc, static_cast<int>(blockIdx.x)};
  proxy.waitSignal(blockIdx.x, 1, ccoCoopBlock{});
}
```

A complete program (host setup, data check, timeouts, `ccoDevCommGetError`) is
[`examples/cco/cpp/03_proxy_put.cpp`](https://github.com/ROCm/mori/blob/main/examples/cco/cpp/03_proxy_put.cpp).

### Checking it on your platform

Run `tests/cpp/cco/test_proxy_put` (built with `BUILD_TESTS=ON`) before relying
on ccoProxy on a new platform, with one rank on each of two nodes. Both ranks
must use GPUs whose NICs are on the same rail (here GPU 0 on both nodes):

```bash
# both nodes
export MORI_SOCKET_IFNAME=<iface> HIP_VISIBLE_DEVICES=0
# node A: write the bootstrap id, copy it to node B, then start rank 0
./test_proxy_put --gen-uid <iface> 0 uid.bin
MORI_CCO_PROXY_TEST_CASES=atomic_free ./test_proxy_put --rank 0 --world 2 --uid-file uid.bin
# node B
MORI_CCO_PROXY_TEST_CASES=atomic_free ./test_proxy_put --rank 1 --world 2 --uid-file uid.bin
```

1. Run the atomic-free cases first (as above), then the full suite (without
   `MORI_CCO_PROXY_TEST_CASES`). `MORI_CCO_PROXY_TEST_NEGATIVE=all` adds the
   failure-path cases. Each rank prints `PASSED` at the end. A build with MPI
   also runs under `mpirun -np 2`.
2. If the atomic-free cases pass but the `SignalInc` / `SignalAdd` cases fail
   with a `COMPLETION` error, NIC atomics into GPU memory do not work on this
   platform: use `SignalSet` / `WindowSignalSet`.
3. `MORI_SHMEM_LOG_LEVEL=info` prints the engine's setup at `ccoDevCommCreate`:
   queues, threads, QPs per peer and the CPUs the proxy threads run on.
4. `cco_p2p_put_bw -t proxy` and `cco_p2p_put_latency -t proxy`
   ([benchmark/cco](https://github.com/ROCm/mori/blob/main/benchmark/cco/README.md))
   measure bandwidth and latency but do not check the data; they are no
   substitute for the test.
